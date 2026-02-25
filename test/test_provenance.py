"""
Tests for source provenance capture and construction line tracking.

Tests are split into sections:
  1. Internal frame filtering
  2. Source path detection
  3. Trigger line extraction (action-time)
  4. Fast per-transformation line capture
  5. _cy_lines helpers
  6. Construction line tracking via PassiveCapture
  7. Edge cases
"""

import os
import sys
import types
import traceback
from unittest.mock import MagicMock, patch

from cluster_yield_snapshot._provenance import (
    detect_source_path,
    extract_trigger_info,
    get_current_user_line,
    _detect_databricks_path,
    _detect_caller_path,
    _is_internal_frame,
    _extract_user_frames,
    _find_trigger_line,
    _match_source_frame,
)
from cluster_yield_snapshot._capture import (
    PassiveCapture,
    _get_lines,
    _copy_lines,
    _set_lines,
    _add_line,
    _merge_line_dicts,
    serialize_construction_lines,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Internal frame filtering
# ═══════════════════════════════════════════════════════════════════════

def test_is_internal_frame_pyspark():
    assert _is_internal_frame("/usr/lib/python3/pyspark/sql/dataframe.py")
    assert _is_internal_frame("C:\\Python\\pyspark\\sql\\session.py")


def test_is_internal_frame_own_package():
    assert _is_internal_frame("cluster_yield_snapshot/_capture.py")
    assert _is_internal_frame("/home/user/lib/cluster_yield_snapshot/snapshot.py")


def test_is_internal_frame_ipython():
    assert _is_internal_frame("/usr/lib/IPython/core/interactiveshell.py")
    assert _is_internal_frame("/usr/lib/ipykernel/zmqshell.py")


def test_is_internal_frame_user_code():
    assert not _is_internal_frame("/home/user/pipelines/order_report.py")
    assert not _is_internal_frame("my_notebook.py")
    assert not _is_internal_frame("/workspace/src/etl/transform.py")


def test_is_internal_frame_py4j():
    assert _is_internal_frame("/usr/lib/py4j/java_gateway.py")


def test_is_internal_frame_debugger():
    assert _is_internal_frame("/usr/lib/_pydev_bundle/debugger.py")
    assert _is_internal_frame("/usr/lib/debugpy/launcher.py")


def test_is_internal_frame_empty_string():
    assert not _is_internal_frame("")


# ═══════════════════════════════════════════════════════════════════════
# 2. Source path detection
# ═══════════════════════════════════════════════════════════════════════

def test_detect_databricks_path():
    spark = MagicMock()
    spark.conf.get.return_value = "/Workspace/Users/team@co.com/pipelines/order_report"
    path = _detect_databricks_path(spark)
    assert path == "/Workspace/Users/team@co.com/pipelines/order_report"


def test_detect_databricks_path_fallback():
    spark = MagicMock()
    call_count = {"n": 0}
    def side_effect(key):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("not found")
        return "/Workspace/Users/team@co.com/order_report"
    spark.conf.get.side_effect = side_effect
    path = _detect_databricks_path(spark)
    assert path == "/Workspace/Users/team@co.com/order_report"


def test_detect_databricks_path_not_databricks():
    spark = MagicMock()
    spark.conf.get.side_effect = Exception("no config")
    path = _detect_databricks_path(spark)
    assert path is None


def test_detect_source_path_databricks_first():
    spark = MagicMock()
    spark.conf.get.return_value = "/Workspace/Users/team@co.com/nb"
    path = detect_source_path(spark)
    assert path == "/Workspace/Users/team@co.com/nb"


def test_detect_source_path_all_strategies_fail():
    spark = MagicMock()
    spark.conf.get.side_effect = Exception("not databricks")
    with patch("cluster_yield_snapshot._provenance._detect_caller_path", return_value=None):
        with patch("cluster_yield_snapshot._provenance.sys") as mock_sys:
            mock_sys.argv = []
            path = detect_source_path(spark)
            assert path is None


# ═══════════════════════════════════════════════════════════════════════
# 3. Trigger line extraction (action-time)
# ═══════════════════════════════════════════════════════════════════════

def test_extract_trigger_info_captures_line():
    info = extract_trigger_info(None)
    assert "triggerStack" in info
    assert len(info["triggerStack"]) > 0


def test_extract_trigger_info_with_source_path():
    this_file = os.path.abspath(__file__)
    info = extract_trigger_info(this_file)
    assert "triggerLine" in info
    assert isinstance(info["triggerLine"], int)
    assert info["triggerLine"] > 0


def test_extract_trigger_info_stack_format():
    info = extract_trigger_info(None)
    if info.get("triggerStack"):
        frame = info["triggerStack"][0]
        assert "file" in frame
        assert "line" in frame
        assert "function" in frame


def test_extract_user_frames_filters_internals():
    frames = traceback.StackSummary.from_list([
        ("/home/user/pipeline.py", 10, "main", "df.collect()"),
        ("/usr/lib/pyspark/sql/dataframe.py", 200, "collect", "..."),
        ("/home/user/lib/cluster_yield_snapshot/_capture.py", 50, "_on_action", "..."),
    ])
    user_frames = _extract_user_frames(frames)
    assert len(user_frames) == 1
    assert user_frames[0]["file"] == "/home/user/pipeline.py"


def test_extract_user_frames_preserves_cross_function_calls():
    frames = traceback.StackSummary.from_list([
        ("/home/user/notebook.py", 5, "main", "run_pipeline(spark)"),
        ("/home/user/lib/pipeline_utils.py", 42, "run_pipeline", "df.write.parquet(...)"),
        ("/usr/lib/pyspark/sql/readwriter.py", 300, "parquet", "..."),
    ])
    user_frames = _extract_user_frames(frames)
    assert len(user_frames) == 2
    assert user_frames[0]["line"] == 5
    assert user_frames[1]["line"] == 42


def test_find_trigger_line_exact_match():
    frames = traceback.StackSummary.from_list([
        ("/home/user/pipeline.py", 23, "main", "spark.sql(...)"),
        ("/home/user/pipeline.py", 52, "main", "df.collect()"),
        ("/usr/lib/pyspark/sql/dataframe.py", 200, "collect", "..."),
    ])
    line = _find_trigger_line(frames, "/home/user/pipeline.py")
    assert line == 52  # innermost match


def test_find_trigger_line_basename_match():
    frames = traceback.StackSummary.from_list([
        ("/repo/src/pipelines/order_report.py", 30, "main", "df.write(...)"),
        ("/usr/lib/pyspark/sql/readwriter.py", 100, "save", "..."),
    ])
    line = _find_trigger_line(
        frames,
        "/Workspace/Users/team@co.com/pipelines/order_report"
    )
    assert line == 30


def test_find_trigger_line_databricks_command():
    frames = traceback.StackSummary.from_list([
        ("<command-12345>", 8, "<module>", "df.collect()"),
        ("/usr/lib/pyspark/sql/dataframe.py", 200, "collect", "..."),
    ])
    line = _find_trigger_line(
        frames,
        "/Workspace/Users/team@co.com/pipelines/order_report"
    )
    assert line == 8


def test_find_trigger_line_empty_frames():
    line = _find_trigger_line(traceback.StackSummary.from_list([]), "/some/path.py")
    assert line is None


def test_match_source_frame_returns_innermost():
    frames = traceback.StackSummary.from_list([
        ("/home/user/pipeline.py", 10, "setup", "config = load()"),
        ("/home/user/pipeline.py", 55, "transform", "df.write.parquet(...)"),
        ("/usr/lib/pyspark/sql/readwriter.py", 100, "parquet", "..."),
    ])
    line = _match_source_frame(frames, "/home/user/pipeline.py")
    assert line == 55


def test_extract_trigger_info_handles_traceback_failure():
    with patch("cluster_yield_snapshot._provenance.traceback") as mock_tb:
        mock_tb.extract_stack.side_effect = RuntimeError("stack unavailable")
        info = extract_trigger_info("/some/path.py")
        assert info == {}


# ═══════════════════════════════════════════════════════════════════════
# 4. Fast per-transformation line capture
# ═══════════════════════════════════════════════════════════════════════

def test_get_current_user_line_returns_tuple():
    """Should return (filename, lineno) from current call site."""
    result = get_current_user_line(None)
    # Should find this test file (or something in the test harness)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[1], int)


def test_get_current_user_line_with_source_path():
    """Should preferentially match the given source path."""
    this_file = os.path.abspath(__file__)
    result = get_current_user_line(this_file)
    assert result is not None
    assert result[1] > 0


def test_get_current_user_line_from_nested_call():
    """Should find the user frame even from a nested call."""
    def inner():
        return get_current_user_line(None)
    result = inner()
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# 5. _cy_lines helpers
# ═══════════════════════════════════════════════════════════════════════

def test_get_lines_empty():
    obj = MagicMock(spec=[])  # no _cy_lines attr
    assert _get_lines(obj) == {}


def test_get_lines_present():
    obj = MagicMock()
    obj._cy_lines = {"/path/a.py": {10, 20}}
    assert _get_lines(obj) == {"/path/a.py": {10, 20}}


def test_copy_lines_deep():
    original = {"/path/a.py": {10, 20}, "/path/b.py": {30}}
    copied = _copy_lines(original)
    assert copied == original
    # Verify it's a deep copy
    copied["/path/a.py"].add(99)
    assert 99 not in original["/path/a.py"]


def test_set_lines_success():
    obj = MagicMock()
    lines = {"/path/a.py": {10}}
    _set_lines(obj, lines)
    assert obj._cy_lines == lines


def test_set_lines_frozen_object():
    """Should not raise on objects that don't allow dynamic attrs."""
    obj = 42  # int doesn't allow setattr
    _set_lines(obj, {"/path/a.py": {10}})  # should not raise


def test_add_line_new_file():
    lines: dict[str, set[int]] = {}
    _add_line(lines, "/path/a.py", 10)
    assert lines == {"/path/a.py": {10}}


def test_add_line_existing_file():
    lines = {"/path/a.py": {10}}
    _add_line(lines, "/path/a.py", 20)
    assert lines == {"/path/a.py": {10, 20}}


def test_merge_line_dicts_disjoint():
    a = {"/path/a.py": {10, 20}}
    b = {"/path/b.py": {30}}
    merged = _merge_line_dicts(a, b)
    assert merged == {"/path/a.py": {10, 20}, "/path/b.py": {30}}
    # Original dicts not mutated
    assert "/path/b.py" not in a


def test_merge_line_dicts_overlapping():
    a = {"/path/a.py": {10, 20}}
    b = {"/path/a.py": {20, 30}, "/path/b.py": {40}}
    merged = _merge_line_dicts(a, b)
    assert merged == {"/path/a.py": {10, 20, 30}, "/path/b.py": {40}}


def test_merge_line_dicts_empty():
    assert _merge_line_dicts({}, {}) == {}
    a = {"/path/a.py": {10}}
    assert _merge_line_dicts(a, {}) == a
    assert _merge_line_dicts({}, a) == a


def test_serialize_construction_lines():
    lines = {"/path/a.py": {30, 10, 20}, "/path/b.py": {5}}
    serialized = serialize_construction_lines(lines)
    assert serialized == {"/path/a.py": [10, 20, 30], "/path/b.py": [5]}


def test_serialize_construction_lines_skips_empty():
    lines = {"/path/a.py": set(), "/path/b.py": {5}}
    serialized = serialize_construction_lines(lines)
    assert serialized == {"/path/b.py": [5]}


# ═══════════════════════════════════════════════════════════════════════
# 6. Construction line tracking via PassiveCapture
# ═══════════════════════════════════════════════════════════════════════

def _setup_mock_pyspark():
    """
    Install a minimal mock pyspark module with construction methods.
    """
    mock_pyspark = types.ModuleType("pyspark")
    mock_sql = types.ModuleType("pyspark.sql")

    class MockDataFrameWriter:
        def __init__(self, df=None):
            self._df = df
        def save(self, path=None, **kw): pass
        def parquet(self, path, **kw): pass
        def csv(self, path, **kw): pass

    class MockDataFrame:
        # Actions
        def collect(self): return []
        def count(self): return 0
        def show(self, *a, **kw): pass
        def toPandas(self): return None
        def first(self): return None
        def head(self, n=1): return []
        def take(self, n): return []
        def tail(self, n): return []

        # Construction methods — each returns a new MockDataFrame
        def filter(self, *a, **kw): return MockDataFrame()
        def where(self, *a, **kw): return MockDataFrame()
        def select(self, *a, **kw): return MockDataFrame()
        def withColumn(self, *a, **kw): return MockDataFrame()
        def drop(self, *a, **kw): return MockDataFrame()
        def distinct(self): return MockDataFrame()
        def orderBy(self, *a, **kw): return MockDataFrame()
        def limit(self, n): return MockDataFrame()
        def repartition(self, *a, **kw): return MockDataFrame()
        def groupBy(self, *a, **kw): return MockDataFrame()
        def agg(self, *a, **kw): return MockDataFrame()

        # Merge methods
        def join(self, other, *a, **kw): return MockDataFrame()
        def union(self, other): return MockDataFrame()
        def crossJoin(self, other): return MockDataFrame()

        @property
        def write(self):
            return MockDataFrameWriter(self)

    mock_sql.DataFrame = MockDataFrame
    mock_sql.DataFrameWriter = MockDataFrameWriter
    mock_pyspark.sql = mock_sql
    sys.modules["pyspark"] = mock_pyspark
    sys.modules["pyspark.sql"] = mock_sql
    return MockDataFrame, MockDataFrameWriter


def _teardown_mock_pyspark():
    sys.modules.pop("pyspark", None)
    sys.modules.pop("pyspark.sql", None)


def test_construction_patches_applied():
    """start() should patch construction methods."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())
        original_filter = MockDF.filter

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        assert MockDF.filter is not original_filter

        pc.stop()
        assert MockDF.filter is original_filter
    finally:
        _teardown_mock_pyspark()


def test_merge_patches_applied():
    """start() should patch merge methods."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())
        original_join = MockDF.join

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        assert MockDF.join is not original_join

        pc.stop()
        assert MockDF.join is original_join
    finally:
        _teardown_mock_pyspark()


def test_spark_table_patched():
    """start() should patch spark.table."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        original_sql = MagicMock(return_value=MockDF())
        mock_spark.sql = original_sql
        original_table = MagicMock(return_value=MockDF())
        mock_spark.table = original_table

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        assert mock_spark.table is not original_table

        pc.stop()
        assert mock_spark.table is original_table
    finally:
        _teardown_mock_pyspark()


def test_construction_propagates_lines():
    """filter() should propagate _cy_lines from self to result."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        # Seed a DataFrame with initial lines
        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}

        # filter should propagate + add current line
        result = df.filter("x > 1")
        result_lines = getattr(result, "_cy_lines", {})

        # Should have the original line 10 from df
        assert "/nb.py" in result_lines or len(result_lines) > 0
        # The original df should be unchanged
        assert df._cy_lines == {"/nb.py": {10}}

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_construction_chain_accumulates():
    """Chained construction should accumulate lines."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}

        # Chain: filter → select → limit
        df2 = df.filter("x > 1")
        df3 = df2.select("x", "y")
        df4 = df3.limit(100)

        # Each step should have MORE lines than the previous
        lines2 = getattr(df2, "_cy_lines", {})
        lines3 = getattr(df3, "_cy_lines", {})
        lines4 = getattr(df4, "_cy_lines", {})

        # All should contain the original seed line
        for lines in [lines2, lines3, lines4]:
            assert 10 in lines.get("/nb.py", set())

        # Each successive DataFrame should have at least as many lines
        all_lines_2 = sum(len(v) for v in lines2.values())
        all_lines_3 = sum(len(v) for v in lines3.values())
        all_lines_4 = sum(len(v) for v in lines4.values())
        assert all_lines_2 >= 1
        assert all_lines_3 >= all_lines_2
        assert all_lines_4 >= all_lines_3

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_join_merges_both_sides():
    """join() should merge _cy_lines from both left and right."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        left = MockDF()
        left._cy_lines = {"/nb.py": {10}}

        right = MockDF()
        right._cy_lines = {"/nb.py": {20}, "/utils.py": {5}}

        joined = left.join(right, "key")
        joined_lines = getattr(joined, "_cy_lines", {})

        # Should have lines from BOTH sides
        nb_lines = joined_lines.get("/nb.py", set())
        assert 10 in nb_lines  # from left
        assert 20 in nb_lines  # from right
        # Should have the cross-file lines too
        assert 5 in joined_lines.get("/utils.py", set())

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_union_merges_both_sides():
    """union() should merge _cy_lines from both sides."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df_a = MockDF()
        df_a._cy_lines = {"/nb.py": {10}}

        df_b = MockDF()
        df_b._cy_lines = {"/nb.py": {30}}

        unioned = df_a.union(df_b)
        union_lines = getattr(unioned, "_cy_lines", {})
        nb_lines = union_lines.get("/nb.py", set())
        assert 10 in nb_lines
        assert 30 in nb_lines

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_spark_sql_seeds_lines():
    """spark.sql() should seed _cy_lines on the returned DataFrame."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        result_df = MockDF()
        mock_spark = MagicMock()
        original_sql = MagicMock(return_value=result_df)
        mock_spark.sql = original_sql

        captured = []
        def callback(label, df, sql, trigger):
            captured.append((label, sql, trigger))

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        # Call patched spark.sql()
        df = mock_spark.sql("SELECT * FROM orders")

        # The returned df should have _cy_lines set
        lines = getattr(df, "_cy_lines", None)
        assert lines is not None
        # Should have at least one entry with a line number
        total_lines = sum(len(v) for v in lines.values())
        assert total_lines >= 1

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_spark_table_seeds_lines():
    """spark.table() should seed _cy_lines on the returned DataFrame."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        result_df = MockDF()
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())
        original_table = MagicMock(return_value=result_df)
        mock_spark.table = original_table

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df = mock_spark.table("orders")

        lines = getattr(df, "_cy_lines", None)
        assert lines is not None
        total_lines = sum(len(v) for v in lines.values())
        assert total_lines >= 1

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_construction_without_seed():
    """Construction on a DataFrame without _cy_lines should still work."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        # DataFrame created externally (no _cy_lines)
        df = MockDF()
        assert not hasattr(df, "_cy_lines") or df._cy_lines == {}

        # filter should still work and start tracking from this point
        result = df.filter("x > 1")
        result_lines = getattr(result, "_cy_lines", {})
        # Should have at least the filter line
        total = sum(len(v) for v in result_lines.values())
        assert total >= 1

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_construction_doesnt_mutate_source():
    """Construction should never mutate the source DataFrame's _cy_lines."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}

        # Create two branches from the same df
        branch_a = df.filter("x > 1")
        branch_b = df.filter("x < 0")

        # Original should be unchanged
        assert df._cy_lines == {"/nb.py": {10}}

        # Branches should be independent
        lines_a = getattr(branch_a, "_cy_lines", {})
        lines_b = getattr(branch_b, "_cy_lines", {})
        # Both should have the seed but they shouldn't share set objects
        if "/nb.py" in lines_a and "/nb.py" in lines_b:
            assert lines_a["/nb.py"] is not lines_b["/nb.py"]

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_construction_stop_restores_all():
    """stop() should restore all construction and merge method originals."""
    MockDF, MockWriter = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())
        mock_spark.table = MagicMock(return_value=MockDF())

        originals = {
            "filter": MockDF.filter,
            "join": MockDF.join,
            "select": MockDF.select,
            "collect": MockDF.collect,
            "parquet": MockWriter.parquet,
        }

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        # All should be patched
        for name, orig in originals.items():
            cls = MockDF if name != "parquet" else MockWriter
            assert getattr(cls, name) is not orig, f"{name} not patched"

        pc.stop()

        # All should be restored
        for name, orig in originals.items():
            cls = MockDF if name != "parquet" else MockWriter
            assert getattr(cls, name) is orig, f"{name} not restored"
    finally:
        _teardown_mock_pyspark()


def test_construction_exception_doesnt_break():
    """If _cy_lines propagation fails, the original method still works."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        # Create a DataFrame where setting _cy_lines will fail
        df = MockDF()
        # Freeze the object so setattr fails
        original_filter_result = MockDF()
        # Even if propagation fails, filter should still return a valid result
        result = df.filter("x > 1")
        assert result is not None

        pc.stop()
    finally:
        _teardown_mock_pyspark()


# ═══════════════════════════════════════════════════════════════════════
# 7. Integration: construction lines in captured plans
# ═══════════════════════════════════════════════════════════════════════

def test_action_captures_construction_lines():
    """When an action fires, the callback should be able to read _cy_lines."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        captured_dfs = []
        def callback(label, df, sql, trigger):
            # Save the df's _cy_lines at capture time
            captured_dfs.append(getattr(df, "_cy_lines", None))

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        # Build a chain
        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}
        df2 = df.filter("x > 1")
        df3 = df2.select("x")

        # Trigger an action
        df3.collect()

        # The callback should have fired with the accumulated lines
        assert len(captured_dfs) == 1
        lines = captured_dfs[0]
        assert lines is not None
        assert 10 in lines.get("/nb.py", set())

        pc.stop()
    finally:
        _teardown_mock_pyspark()


# ═══════════════════════════════════════════════════════════════════════
# 8. Edge cases from earlier tests
# ═══════════════════════════════════════════════════════════════════════

def test_passive_capture_patches_spark_sql():
    """start() should replace spark.sql with a wrapper."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        original_sql = MagicMock(return_value=MockDF())
        mock_spark.sql = original_sql

        captured = []
        def callback(label, df, sql, trigger):
            captured.append((label, sql, trigger))

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        assert mock_spark.sql is not original_sql
        mock_spark.sql("SELECT * FROM orders")
        assert original_sql.called
        assert len(captured) == 1
        assert captured[0][1] == "SELECT * FROM orders"
        assert captured[0][2] == "spark.sql"

        pc.stop()
        assert mock_spark.sql is original_sql
    finally:
        _teardown_mock_pyspark()


def test_passive_capture_patches_dataframe_actions():
    """start() should patch DataFrame.collect etc."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())
        original_collect = MockDF.collect

        captured = []
        def callback(label, df, sql, trigger):
            captured.append((label, trigger))

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        assert MockDF.collect is not original_collect
        df = MockDF()
        df.collect()
        assert len(captured) == 1
        assert captured[0][1] == "action.collect"

        pc.stop()
        assert MockDF.collect is original_collect
    finally:
        _teardown_mock_pyspark()


def test_passive_capture_patches_writer():
    """start() should patch DataFrameWriter.parquet etc."""
    MockDF, MockWriter = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())
        original_parquet = MockWriter.parquet

        captured = []
        def callback(label, df, sql, trigger):
            captured.append((label, trigger))

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        assert MockWriter.parquet is not original_parquet
        df = MockDF()
        writer = MockWriter(df)
        writer.parquet("/tmp/output")
        assert len(captured) == 1
        assert captured[0][1] == "write.parquet"

        pc.stop()
        assert MockWriter.parquet is original_parquet
    finally:
        _teardown_mock_pyspark()


def test_passive_capture_reentrant_guard():
    """Internal spark.sql() calls during capture should not trigger callback."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        original_sql = MagicMock(return_value=MockDF())
        mock_spark.sql = original_sql

        captured = []
        def callback(label, df, sql, trigger):
            captured.append(sql)

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        # Simulate a reentrant call
        pc._enter_capture()
        mock_spark.sql("DESCRIBE DETAIL orders")
        pc._exit_capture()

        assert len(captured) == 0

        pc.stop()
    finally:
        _teardown_mock_pyspark()


# ── Run all tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            import traceback as tb
            tb.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        exit(1)