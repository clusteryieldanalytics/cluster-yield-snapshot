"""
Tests for source provenance capture and construction line tracking.
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
    compute_cell_fingerprint,
    is_databricks_cell_path,
    _detect_databricks_dbutils_path,
    _detect_databricks_spark_conf_path,
    _detect_caller_path,
    _is_internal_frame,
    _is_databricks_cell_frame,
    _is_user_frame,
    _walk_frames,
    _extract_user_frames,
    _find_trigger_line,
    _ensure_cell_cached,
    _cell_source_cache,
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


def test_is_internal_frame_databricks_infra():
    assert _is_internal_frame("/databricks/python_shell/scripts/db_ipykernel_launcher.py")
    assert _is_internal_frame("/databricks/python/lib/python3.12/site-packages/IPython/core/interactiveshell.py")
    assert _is_internal_frame("/databricks/python_shell/lib/dbruntime/kernel.py")
    assert _is_internal_frame("/databricks/python/lib/python3.12/site-packages/tornado/platform/asyncio.py")


def test_is_internal_frame_empty_string():
    assert not _is_internal_frame("")


# ═══════════════════════════════════════════════════════════════════════
# 2. Databricks cell frame detection
# ═══════════════════════════════════════════════════════════════════════

def test_databricks_cell_frame_real_path():
    """Real Databricks cell frame path from production."""
    assert _is_databricks_cell_frame(
        "/home/spark-88da72a5-41e6-4eb8-b0b7-22/.ipykernel/3322/command-7650185192914251-931726307"
    )


def test_databricks_cell_frame_short():
    assert _is_databricks_cell_frame("/tmp/command-12345-67890")


def test_databricks_cell_frame_not_command():
    assert not _is_databricks_cell_frame("/home/user/pipeline.py")
    assert not _is_databricks_cell_frame("/databricks/python_shell/scripts/db_ipykernel_launcher.py")
    assert not _is_databricks_cell_frame("/usr/lib/ipykernel/zmqshell.py")


def test_databricks_cell_frame_not_synthetic():
    """<command-N> synthetic paths are NOT the real Databricks pattern."""
    # These are synthetic — _is_databricks_cell_frame checks real paths
    assert not _is_databricks_cell_frame("<command-12345>")


def test_is_user_frame_databricks_cell():
    """Databricks cell frames should be recognized as user code
    even though they contain .ipykernel/ in the path."""
    cell = "/home/spark-88da72a5-41e6-4eb8-b0b7-22/.ipykernel/3322/command-7650185192914251-931726307"
    # _is_internal_frame would say True (contains ipykernel/)
    assert _is_internal_frame(cell)
    # But _is_user_frame overrides — cell frames are user code
    assert _is_user_frame(cell)


def test_is_user_frame_regular():
    assert _is_user_frame("/home/user/pipeline.py")
    assert not _is_user_frame("/usr/lib/pyspark/sql/dataframe.py")
    assert not _is_user_frame("<frozen importlib._bootstrap>")


# ═══════════════════════════════════════════════════════════════════════
# 3. Source path detection
# ═══════════════════════════════════════════════════════════════════════

def test_detect_databricks_dbutils_path():
    """dbutils via IPython namespace — primary strategy on Databricks."""
    mock_ip = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.notebookPath.return_value.get.return_value = (
        "/Users/team@co.com/pipelines/order_report"
    )
    mock_dbutils = MagicMock()
    mock_dbutils.notebook.entry_point.getDbutils.return_value \
        .notebook.return_value.getContext.return_value = mock_ctx
    mock_ip.user_ns = {"dbutils": mock_dbutils}

    with patch("cluster_yield_snapshot._provenance._get_ipython", return_value=mock_ip):
        path = _detect_databricks_dbutils_path()
    assert path == "/Users/team@co.com/pipelines/order_report"


def test_detect_databricks_dbutils_path_no_ipython():
    with patch("cluster_yield_snapshot._provenance._get_ipython", return_value=None):
        assert _detect_databricks_dbutils_path() is None


def test_detect_databricks_dbutils_path_no_dbutils():
    mock_ip = MagicMock()
    mock_ip.user_ns = {}
    with patch("cluster_yield_snapshot._provenance._get_ipython", return_value=mock_ip):
        assert _detect_databricks_dbutils_path() is None


def test_detect_databricks_spark_conf_path():
    spark = MagicMock()
    spark.conf.get.return_value = "/Workspace/Users/team@co.com/nb"
    path = _detect_databricks_spark_conf_path(spark)
    assert path == "/Workspace/Users/team@co.com/nb"


def test_detect_databricks_spark_conf_path_fallback():
    spark = MagicMock()
    call_count = {"n": 0}
    def side_effect(key):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("not found")
        return "/Workspace/Users/team@co.com/order_report"
    spark.conf.get.side_effect = side_effect
    path = _detect_databricks_spark_conf_path(spark)
    assert path == "/Workspace/Users/team@co.com/order_report"


def test_detect_databricks_spark_conf_path_not_databricks():
    spark = MagicMock()
    spark.conf.get.side_effect = Exception("no config")
    path = _detect_databricks_spark_conf_path(spark)
    assert path is None


def test_detect_source_path_dbutils_first():
    """dbutils should be tried before spark.conf."""
    spark = MagicMock()
    spark.conf.get.side_effect = Exception("blocked on Spark Connect")
    with patch(
        "cluster_yield_snapshot._provenance._detect_databricks_dbutils_path",
        return_value="/Users/team@co.com/nb",
    ):
        path = detect_source_path(spark)
    assert path == "/Users/team@co.com/nb"


def test_detect_source_path_all_strategies_fail():
    spark = MagicMock()
    spark.conf.get.side_effect = Exception("not databricks")
    with patch("cluster_yield_snapshot._provenance._detect_databricks_dbutils_path", return_value=None):
        with patch("cluster_yield_snapshot._provenance._detect_caller_path", return_value=None):
            with patch("cluster_yield_snapshot._provenance.sys") as mock_sys:
                mock_sys.argv = []
                path = detect_source_path(spark)
                assert path is None


# ═══════════════════════════════════════════════════════════════════════
# 4. Frame walking
# ═══════════════════════════════════════════════════════════════════════

def _make_frame(filename, lineno, parent=None):
    """Build a mock frame object."""
    f = MagicMock()
    f.f_code.co_filename = filename
    f.f_lineno = lineno
    f.f_back = parent
    return f


def test_walk_frames_databricks_cell():
    """Should find Databricks cell frame and return raw path for fingerprinting."""
    cell = _make_frame(
        "/home/spark-UUID/.ipykernel/3322/command-123-456", 10
    )
    infra = _make_frame(
        "/databricks/python_shell/scripts/db_ipykernel_launcher.py", 48,
        parent=cell,
    )
    result = _walk_frames(
        infra,
        "/Workspace/Users/team@co.com/order_report"
    )
    # Returns raw cell path — serializer will fingerprint it later
    assert result == ("/home/spark-UUID/.ipykernel/3322/command-123-456", 10)


def test_walk_frames_databricks_cell_no_source_path():
    """Without source_path, cell frame still found with raw path."""
    cell = _make_frame(
        "/home/spark-UUID/.ipykernel/3322/command-123-456", 10
    )
    result = _walk_frames(cell, None)
    assert result == ("/home/spark-UUID/.ipykernel/3322/command-123-456", 10)


def test_walk_frames_skips_databricks_infra():
    """Should skip all /databricks/ infrastructure frames."""
    cell = _make_frame(
        "/home/spark-UUID/.ipykernel/3322/command-123-456", 5
    )
    ipython = _make_frame(
        "/databricks/python/lib/python3.12/site-packages/IPython/core/interactiveshell.py", 3602,
        parent=cell,
    )
    launcher = _make_frame(
        "/databricks/python_shell/scripts/db_ipykernel_launcher.py", 48,
        parent=ipython,
    )
    result = _walk_frames(
        launcher,
        "/Workspace/Users/team@co.com/nb"
    )
    assert result == ("/home/spark-UUID/.ipykernel/3322/command-123-456", 5)


def test_walk_frames_local_script():
    """On local, should return the actual user file."""
    user = _make_frame("/home/user/pipeline.py", 30)
    pyspark = _make_frame(
        "/usr/lib/pyspark/sql/dataframe.py", 200,
        parent=user,
    )
    result = _walk_frames(pyspark, "/home/user/pipeline.py")
    assert result == ("/home/user/pipeline.py", 30)


def test_walk_frames_cross_file():
    """Should find the innermost user frame (the actual call site)."""
    notebook = _make_frame("/home/user/notebook.py", 5)
    utils = _make_frame("/home/user/utils.py", 42, parent=notebook)
    pyspark = _make_frame(
        "/usr/lib/pyspark/sql/dataframe.py", 200,
        parent=utils,
    )
    # Innermost user frame wins — that's where .filter() was called
    result = _walk_frames(pyspark, None)
    assert result == ("/home/user/utils.py", 42)

    # source_path doesn't override — it's for Databricks normalization,
    # not for preferring one file over another. The actual call site
    # is still utils.py.
    result = _walk_frames(pyspark, "/home/user/notebook.py")
    assert result == ("/home/user/utils.py", 42)


# ═══════════════════════════════════════════════════════════════════════
# 5. Trigger line extraction
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


def test_find_trigger_line_databricks_real_stack():
    """Simulates the actual Databricks stack trace."""
    cell_path = "/home/spark-UUID/.ipykernel/3322/command-123-456"
    _cell_source_cache[cell_path] = "df.collect()\n"
    try:
        frames = traceback.StackSummary.from_list([
            (cell_path, 10, "<module>", "df.collect()"),
            ("/databricks/python/lib/python3.12/site-packages/IPython/core/interactiveshell.py", 3602, "run_code", "..."),
            ("/databricks/python_shell/scripts/db_ipykernel_launcher.py", 48, "main", "..."),
        ])
        result = _find_trigger_line(
            frames,
            "/Workspace/Users/team@co.com/order_report"
        )
        assert result is not None
        assert result["line"] == 10
        assert "cellFingerprint" in result
    finally:
        _cell_source_cache.pop(cell_path, None)


def test_find_trigger_line_exact_match():
    frames = traceback.StackSummary.from_list([
        ("/home/user/pipeline.py", 23, "main", "spark.sql(...)"),
        ("/home/user/pipeline.py", 52, "main", "df.collect()"),
        ("/usr/lib/pyspark/sql/dataframe.py", 200, "collect", "..."),
    ])
    result = _find_trigger_line(frames, "/home/user/pipeline.py")
    assert result is not None
    assert result["line"] == 52  # innermost match
    assert "cellFingerprint" not in result


def test_find_trigger_line_basename_match():
    frames = traceback.StackSummary.from_list([
        ("/repo/src/pipelines/order_report.py", 30, "main", "df.write(...)"),
        ("/usr/lib/pyspark/sql/readwriter.py", 100, "save", "..."),
    ])
    result = _find_trigger_line(
        frames,
        "/Workspace/Users/team@co.com/pipelines/order_report"
    )
    assert result is not None
    assert result["line"] == 30


def test_find_trigger_line_empty_frames():
    result = _find_trigger_line(traceback.StackSummary.from_list([]), "/some/path.py")
    assert result is None


def test_extract_user_frames_databricks_fingerprinted():
    """Databricks cell frames should get cellFingerprint, not file."""
    cell_path = "/home/spark-UUID/.ipykernel/3322/command-123-456"
    _cell_source_cache[cell_path] = "df.collect()\n"
    try:
        frames = traceback.StackSummary.from_list([
            (cell_path, 10, "<module>", "df.collect()"),
            ("/databricks/python/lib/python3.12/site-packages/IPython/core/interactiveshell.py", 3602, "run_code", "..."),
            ("/databricks/python_shell/scripts/db_ipykernel_launcher.py", 48, "main", "..."),
        ])
        user_frames = _extract_user_frames(frames)
        assert len(user_frames) == 1
        assert "cellFingerprint" in user_frames[0]
        assert "file" not in user_frames[0]
        assert user_frames[0]["line"] == 10
    finally:
        _cell_source_cache.pop(cell_path, None)


def test_extract_user_frames_local():
    frames = traceback.StackSummary.from_list([
        ("/home/user/pipeline.py", 10, "main", "df.collect()"),
        ("/usr/lib/pyspark/sql/dataframe.py", 200, "collect", "..."),
        ("/home/user/lib/cluster_yield_snapshot/_capture.py", 50, "_on_action", "..."),
    ])
    user_frames = _extract_user_frames(frames)
    assert len(user_frames) == 1
    assert user_frames[0]["file"] == "/home/user/pipeline.py"
    assert "cellFingerprint" not in user_frames[0]


def test_extract_trigger_info_handles_traceback_failure():
    with patch("cluster_yield_snapshot._provenance.traceback") as mock_tb:
        mock_tb.extract_stack.side_effect = RuntimeError("stack unavailable")
        info = extract_trigger_info("/some/path.py")
        assert info == {}


# ═══════════════════════════════════════════════════════════════════════
# 6. get_current_user_line
# ═══════════════════════════════════════════════════════════════════════

def test_get_current_user_line_returns_tuple():
    result = get_current_user_line(None)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[1], int)


def test_get_current_user_line_with_source_path():
    this_file = os.path.abspath(__file__)
    result = get_current_user_line(this_file)
    assert result is not None
    assert result[1] > 0


def test_get_current_user_line_from_nested_call():
    def inner():
        return get_current_user_line(None)
    result = inner()
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# 7. _cy_lines helpers
# ═══════════════════════════════════════════════════════════════════════

def test_get_lines_empty():
    obj = MagicMock(spec=[])
    assert _get_lines(obj) == {}


def test_get_lines_present():
    obj = MagicMock()
    obj._cy_lines = {"/path/a.py": {10, 20}}
    assert _get_lines(obj) == {"/path/a.py": {10, 20}}


def test_copy_lines_deep():
    original = {"/path/a.py": {10, 20}, "/path/b.py": {30}}
    copied = _copy_lines(original)
    assert copied == original
    copied["/path/a.py"].add(99)
    assert 99 not in original["/path/a.py"]


def test_set_lines_success():
    obj = MagicMock()
    lines = {"/path/a.py": {10}}
    _set_lines(obj, lines)
    assert obj._cy_lines == lines


def test_set_lines_frozen_object():
    obj = 42
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
    assert "/path/b.py" not in a


def test_merge_line_dicts_overlapping():
    a = {"/path/a.py": {10, 20}}
    b = {"/path/a.py": {20, 30}, "/path/b.py": {40}}
    merged = _merge_line_dicts(a, b)
    assert merged == {"/path/a.py": {10, 20, 30}, "/path/b.py": {40}}


def test_merge_line_dicts_empty():
    assert _merge_line_dicts({}, {}) == {}


def test_serialize_construction_lines_file_entries():
    lines = {"/path/a.py": {30, 10, 20}, "/path/b.py": {5}}
    serialized = serialize_construction_lines(lines)
    assert serialized == [
        {"file": "/path/a.py", "lines": [10, 20, 30]},
        {"file": "/path/b.py", "lines": [5]},
    ]


def test_serialize_construction_lines_skips_empty():
    lines = {"/path/a.py": set(), "/path/b.py": {5}}
    serialized = serialize_construction_lines(lines)
    assert serialized == [{"file": "/path/b.py", "lines": [5]}]


def test_serialize_construction_lines_databricks_cells():
    """Databricks cell paths produce cellFingerprint entries via cache."""
    import hashlib

    source = "df = spark.sql('SELECT 1')\ndf.show()\n"
    expected_fp = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    cell_path = "/home/spark-UUID/.ipykernel/999/command-9876543210-123"

    _cell_source_cache[cell_path] = source
    try:
        lines = {cell_path: {1, 2}}
        serialized = serialize_construction_lines(lines)
        assert len(serialized) == 1
        assert serialized[0]["cellFingerprint"] == expected_fp
        assert serialized[0]["lines"] == [1, 2]
        assert "file" not in serialized[0]
    finally:
        _cell_source_cache.pop(cell_path, None)


def test_serialize_construction_lines_mixed():
    """Mix of cell paths and regular file paths."""
    import hashlib

    source = "cell source"
    cell_path = "/home/spark-UUID/.ipykernel/999/command-111222333-444"

    _cell_source_cache[cell_path] = source
    try:
        lines = {
            cell_path: {5, 10},
            "/home/user/utils.py": {42},
        }
        serialized = serialize_construction_lines(lines)
        assert len(serialized) == 2
        cell_entries = [e for e in serialized if "cellFingerprint" in e]
        file_entries = [e for e in serialized if "file" in e]
        assert len(cell_entries) == 1
        assert len(file_entries) == 1
        assert file_entries[0] == {"file": "/home/user/utils.py", "lines": [42]}
    finally:
        _cell_source_cache.pop(cell_path, None)


def test_compute_cell_fingerprint_from_disk():
    """Disk read still works as last resort (for local testing)."""
    import hashlib
    import tempfile

    source = "orders = spark.table('orders')\n"
    expected = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "command-555666777-888")
    with open(path, "w") as f:
        f.write(source)

    try:
        # Clear cache to force disk read
        _cell_source_cache.pop(path, None)
        assert compute_cell_fingerprint(path) == expected
    finally:
        os.unlink(path)
        os.rmdir(tmpdir)


def test_compute_cell_fingerprint_from_cache():
    """Eagerly cached source should be used for fingerprinting."""
    import hashlib

    fake_path = "/home/spark-UUID/.ipykernel/999/command-111222333-444"
    source = "df = spark.sql('SELECT 1')\n"
    expected = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    _cell_source_cache[fake_path] = source
    try:
        assert compute_cell_fingerprint(fake_path) == expected
    finally:
        _cell_source_cache.pop(fake_path, None)


def test_compute_cell_fingerprint_missing_file():
    fake_path = "/nonexistent/command-999888777-666"
    _cell_source_cache.pop(fake_path, None)
    assert compute_cell_fingerprint(fake_path) is None


def test_ensure_cell_cached_from_linecache():
    """_ensure_cell_cached should populate the cache from linecache."""
    import linecache

    fake_path = "/tmp/command-test-ensure-cached-12345"
    source_lines = ["line one\n", "line two\n"]
    # Manually inject into linecache (simulates what Databricks does)
    linecache.cache[fake_path] = (
        len("".join(source_lines)),
        None,
        source_lines,
        fake_path,
    )
    _cell_source_cache.pop(fake_path, None)

    try:
        _ensure_cell_cached(fake_path)
        assert fake_path in _cell_source_cache
        assert _cell_source_cache[fake_path] == "line one\nline two\n"
    finally:
        _cell_source_cache.pop(fake_path, None)
        linecache.cache.pop(fake_path, None)


def test_ensure_cell_cached_idempotent():
    """Second call shouldn't overwrite existing cache entry."""
    fake_path = "/tmp/command-test-idempotent-99999"
    _cell_source_cache[fake_path] = "original"

    try:
        _ensure_cell_cached(fake_path)
        assert _cell_source_cache[fake_path] == "original"
    finally:
        _cell_source_cache.pop(fake_path, None)


# ═══════════════════════════════════════════════════════════════════════
# 8. Construction line tracking via PassiveCapture
# ═══════════════════════════════════════════════════════════════════════

def _setup_mock_pyspark():
    mock_pyspark = types.ModuleType("pyspark")
    mock_sql = types.ModuleType("pyspark.sql")

    class MockDataFrameWriter:
        def __init__(self, df=None):
            self._df = df
        def save(self, path=None, **kw): pass
        def parquet(self, path, **kw): pass
        def csv(self, path, **kw): pass

    class MockDataFrame:
        def collect(self): return []
        def count(self): return 0
        def show(self, *a, **kw): pass
        def toPandas(self): return None
        def first(self): return None
        def head(self, n=1): return []
        def take(self, n): return []
        def tail(self, n): return []
        def filter(self, *a, **kw): return MockDataFrame()
        def where(self, *a, **kw): return MockDataFrame()
        def select(self, *a, **kw): return MockDataFrame()
        def withColumn(self, *a, **kw): return MockDataFrame()
        def drop(self, *a, **kw): return MockDataFrame()
        def distinct(self): return MockDataFrame()
        def orderBy(self, *a, **kw): return MockDataFrame()
        def limit(self, n): return MockDataFrame()
        def repartition(self, *a, **kw): return MockDataFrame()
        def agg(self, *a, **kw): return MockDataFrame()
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


def test_construction_propagates_lines():
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}

        result = df.filter("x > 1")
        result_lines = getattr(result, "_cy_lines", {})
        assert 10 in result_lines.get("/nb.py", set())
        assert df._cy_lines == {"/nb.py": {10}}

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_construction_chain_accumulates():
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}

        df2 = df.filter("x > 1")
        df3 = df2.select("x", "y")
        df4 = df3.limit(100)

        for result in [df2, df3, df4]:
            lines = getattr(result, "_cy_lines", {})
            assert 10 in lines.get("/nb.py", set())

        # Each step should have at least as many lines
        counts = [
            sum(len(v) for v in getattr(d, "_cy_lines", {}).values())
            for d in [df2, df3, df4]
        ]
        assert counts[0] >= 1
        assert counts[1] >= counts[0]
        assert counts[2] >= counts[1]

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_join_merges_both_sides():
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
        nb_lines = joined_lines.get("/nb.py", set())
        assert 10 in nb_lines
        assert 20 in nb_lines
        assert 5 in joined_lines.get("/utils.py", set())

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_union_merges_both_sides():
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
        nb_lines = getattr(unioned, "_cy_lines", {}).get("/nb.py", set())
        assert 10 in nb_lines
        assert 30 in nb_lines

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_spark_sql_seeds_lines():
    MockDF, _ = _setup_mock_pyspark()
    try:
        result_df = MockDF()
        mock_spark = MagicMock()
        original_sql = MagicMock(return_value=result_df)
        mock_spark.sql = original_sql

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df = mock_spark.sql("SELECT * FROM orders")
        lines = getattr(df, "_cy_lines", None)
        assert lines is not None
        total = sum(len(v) for v in lines.values())
        assert total >= 1

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_construction_doesnt_mutate_source():
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}

        branch_a = df.filter("x > 1")
        branch_b = df.filter("x < 0")

        assert df._cy_lines == {"/nb.py": {10}}

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_construction_stop_restores_all():
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
        pc.stop()

        for name, orig in originals.items():
            cls = MockDF if name != "parquet" else MockWriter
            assert getattr(cls, name) is orig, f"{name} not restored"
    finally:
        _teardown_mock_pyspark()


def test_action_captures_construction_lines():
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        captured_dfs = []
        def callback(label, df, sql, trigger):
            captured_dfs.append(getattr(df, "_cy_lines", None))

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        df = MockDF()
        df._cy_lines = {"/nb.py": {10}}
        df2 = df.filter("x > 1")
        df3 = df2.select("x")
        df3.collect()

        assert len(captured_dfs) == 1
        lines = captured_dfs[0]
        assert lines is not None
        assert 10 in lines.get("/nb.py", set())

        pc.stop()
    finally:
        _teardown_mock_pyspark()


# ═══════════════════════════════════════════════════════════════════════
# 9. Existing PassiveCapture tests (preserved)
# ═══════════════════════════════════════════════════════════════════════

def test_passive_capture_patches_spark_sql():
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
        assert len(captured) == 1
        assert captured[0][1] == "SELECT * FROM orders"
        pc.stop()
        assert mock_spark.sql is original_sql
    finally:
        _teardown_mock_pyspark()


def test_passive_capture_patches_writer():
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