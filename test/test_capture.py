"""
Tests for passive capture engine and quick_scan.

Uses mock PySpark classes since these tests run without a real SparkSession.
"""

import sys
import types
from unittest.mock import MagicMock, patch

from cluster_yield_snapshot.quick_scan import quick_scan
from cluster_yield_snapshot.upload import UploadResult
from cluster_yield_snapshot._capture import PassiveCapture


# ── Quick scan tests ─────────────────────────────────────────────────────

def test_quick_scan_detects_cartesian():
    plans = [{
        "label": "bad_query",
        "plan": [{"class": "org.apache.spark.sql.execution.joins.CartesianProductExec"}],
    }]
    teasers = quick_scan(plans, {}, {"optimizerRelevant": {}})
    assert any("Cartesian" in t for t in teasers)


def test_quick_scan_detects_bnlj_in_text():
    plans = [{
        "label": "bad_join",
        "planText": (
            "== Physical Plan ==\n"
            "+- BroadcastNestedLoopJoin BuildRight, Inner\n"
            "   :- FileScan parquet db.big\n"
            "   +- BroadcastExchange\n"
            "      +- FileScan parquet db.small\n"
        ),
    }]
    teasers = quick_scan(plans, {}, {"optimizerRelevant": {}})
    assert any("BroadcastNestedLoopJoin" in t for t in teasers)


def test_quick_scan_detects_broadcast_disabled():
    config = {"optimizerRelevant": {"spark.sql.autoBroadcastJoinThreshold": "-1"}}
    teasers = quick_scan([], {}, config)
    assert any("disabled" in t.lower() for t in teasers)


def test_quick_scan_detects_small_table():
    tables = {"lookup": {"sizeInBytes": 5_000_000}}
    config = {"optimizerRelevant": {"spark.sql.autoBroadcastJoinThreshold": "10485760"}}
    teasers = quick_scan([], tables, config)
    assert any("broadcast" in t.lower() for t in teasers)


def test_quick_scan_detects_default_partitions_large_data():
    tables = {"huge_events": {"sizeInBytes": 100 * 1024**3}}
    config = {"optimizerRelevant": {"spark.sql.shuffle.partitions": "200"}}
    teasers = quick_scan([], tables, config)
    assert any("200 shuffle partitions" in t for t in teasers)


def test_quick_scan_clean():
    assert quick_scan([], {}, {"optimizerRelevant": {}}) == []


# ── UploadResult tests ───────────────────────────────────────────────────

def test_upload_result_properties():
    result = UploadResult({
        "snapshotId": "snap_123",
        "findingsCount": 5,
        "estimatedMonthlyCost": 8000.0,
        "dashboardUrl": "https://app.clusteryield.com/snap/123",
    })
    assert result.snapshot_id == "snap_123"
    assert result.findings_count == 5
    assert result.estimated_monthly_cost == 8000.0
    assert result.dashboard_url is not None


def test_upload_result_missing_fields():
    result = UploadResult({"snapshotId": "snap_456"})
    assert result.findings_count == 0
    assert result.estimated_monthly_cost is None
    assert result.dashboard_url is None


def test_upload_result_repr():
    result = UploadResult({"snapshotId": "snap_789", "findingsCount": 3})
    assert "snap_789" in repr(result)
    assert "findings=3" in repr(result)


# ── PassiveCapture tests (with mock PySpark) ─────────────────────────────

def _setup_mock_pyspark():
    """
    Install a minimal mock pyspark module so PassiveCapture can import
    DataFrame and DataFrameWriter classes to patch.
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
        def collect(self): return []
        def count(self): return 0
        def show(self, *a, **kw): pass
        def toPandas(self): return None
        def first(self): return None
        def head(self, n=1): return []
        def take(self, n): return []
        def tail(self, n): return []

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

        # spark.sql should now be patched
        assert mock_spark.sql is not original_sql

        # Call the patched version
        mock_spark.sql("SELECT * FROM orders")
        assert original_sql.called
        assert len(captured) == 1
        assert "SELECT * FROM orders" in captured[0][0]
        assert captured[0][1] == "SELECT * FROM orders"
        assert captured[0][2] == "spark.sql"

        pc.stop()
        # Should be restored
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

        # DataFrame.collect should now be patched
        assert MockDF.collect is not original_collect

        # Call collect on an instance
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
        assert "/tmp/output" in captured[0][0]

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
            # Simulate an internal call (like catalog stats)
            # This should NOT trigger another capture
            mock_spark.sql("DESCRIBE DETAIL some_table")

        pc = PassiveCapture(mock_spark, callback)
        pc.start()

        mock_spark.sql("SELECT * FROM orders")

        # Only the outer call should be captured, not the DESCRIBE DETAIL
        assert len(captured) == 1
        assert captured[0] == "SELECT * FROM orders"

        pc.stop()
    finally:
        _teardown_mock_pyspark()


def test_passive_capture_stop_restores_all():
    """stop() should restore all original methods."""
    MockDF, MockWriter = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        orig_collect = MockDF.collect
        orig_count = MockDF.count
        orig_parquet = MockWriter.parquet

        pc = PassiveCapture(mock_spark, lambda *a: None)
        pc.start()

        # Everything should be patched
        assert MockDF.collect is not orig_collect
        assert MockDF.count is not orig_count
        assert MockWriter.parquet is not orig_parquet

        pc.stop()

        # Everything should be restored
        assert MockDF.collect is orig_collect
        assert MockDF.count is orig_count
        assert MockWriter.parquet is orig_parquet
    finally:
        _teardown_mock_pyspark()


def test_passive_capture_callback_exception_doesnt_break():
    """If the callback throws, the original method should still work."""
    MockDF, _ = _setup_mock_pyspark()
    try:
        mock_spark = MagicMock()
        mock_spark.sql = MagicMock(return_value=MockDF())

        def bad_callback(label, df, sql, trigger):
            raise RuntimeError("capture failed!")

        pc = PassiveCapture(mock_spark, bad_callback)
        pc.start()

        # This should NOT raise — the exception is swallowed
        result = mock_spark.sql("SELECT * FROM orders")
        assert result is not None  # original return value passes through

        pc.stop()
    finally:
        _teardown_mock_pyspark()


# ── Run all tests ────────────────────────────────────────────────────────

# ── _compat metric extraction tests (mocked JVM) ────────────────────────

def _make_mock_metrics_map(name_value_pairs):
    """
    Build a mock Scala Map[String, SQLMetric] that supports keysIterator + apply.

    This mocks Strategy 2 (keysIterator) from _extract_node_metrics,
    which is the path taken when jvm=None (i.e. in unit tests).
    """
    # Build per-metric mocks
    metric_mocks = {}
    for name, value in name_value_pairs:
        m = MagicMock()
        m.value.return_value = value
        metric_mocks[name] = m

    # keysIterator returns a Java-style Iterator (hasNext/next)
    keys = list(metric_mocks.keys())
    key_iter = MagicMock()
    call_count = {"i": 0}
    def has_next():
        return call_count["i"] < len(keys)
    def next_key():
        idx = call_count["i"]
        call_count["i"] += 1
        return keys[idx]
    key_iter.hasNext = has_next
    key_iter.next = next_key

    mock_map = MagicMock()
    mock_map.keysIterator.return_value = key_iter
    mock_map.apply = lambda k: metric_mocks[str(k)]
    return mock_map


def _make_mock_node(class_name, node_name=None, metrics=None, children=None):
    """Build a mock SparkPlan node for testing."""
    node = MagicMock()
    node.getClass.return_value.getSimpleName.return_value = class_name
    node.nodeName.return_value = node_name or class_name
    if metrics:
        node.metrics.return_value = _make_mock_metrics_map(metrics)
    else:
        # Empty metrics — keysIterator immediately returns False for hasNext
        empty_iter = MagicMock()
        empty_iter.hasNext = lambda: False
        empty_map = MagicMock()
        empty_map.keysIterator.return_value = empty_iter
        node.metrics.return_value = empty_map

    child_seq = MagicMock()
    if children:
        child_seq.size.return_value = len(children)
        child_seq.apply = lambda i: children[i]
    else:
        child_seq.size.return_value = 0
    node.children.return_value = child_seq
    return node


def test_metric_aliases_canonical():
    """Photon metric names should be canonicalized."""
    from cluster_yield_snapshot._compat import _METRIC_ALIASES
    assert _METRIC_ALIASES["photon rows read"] == "number of output rows"
    assert _METRIC_ALIASES["num output rows"] == "number of output rows"
    assert _METRIC_ALIASES["shuffle bytes written"] == "data size"


def test_get_plan_metrics_returns_none_on_connect():
    """get_plan_metrics should return None when _jdf is unavailable."""
    from cluster_yield_snapshot._compat import get_plan_metrics

    mock_df = MagicMock()
    # Simulate Spark Connect: no _jdf attribute
    del mock_df._jdf
    result = get_plan_metrics(mock_df)
    assert result is None


def test_get_plan_metrics_returns_none_on_exception():
    """get_plan_metrics should return None on any JVM error."""
    from cluster_yield_snapshot._compat import get_plan_metrics

    mock_df = MagicMock()
    mock_df._jdf.queryExecution.side_effect = RuntimeError("JVM gone")
    result = get_plan_metrics(mock_df)
    assert result is None


def test_collect_metrics_recursive_basic():
    """_collect_metrics_recursive should walk plan tree and extract metrics."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    mock_node = _make_mock_node(
        "FileSourceScanExec",
        node_name="FileScan parquet",
        metrics=[("number of output rows", 12345)],
    )

    collector: list[dict] = []
    _collect_metrics_recursive(mock_node, collector)

    assert len(collector) == 1
    assert collector[0]["nodeName"] == "FileScan parquet"
    assert collector[0]["simpleClassName"] == "FileSourceScanExec"
    assert collector[0]["metrics"]["number of output rows"] == 12345


def test_collect_metrics_skips_zero_values():
    """Metrics with value 0 should be omitted."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    mock_node = _make_mock_node(
        "ShuffleExchangeExec",
        node_name="Exchange",
        metrics=[("spill size", 0), ("data size", 42)],
    )

    collector: list[dict] = []
    _collect_metrics_recursive(mock_node, collector)

    assert "spill size" not in collector[0].get("metrics", {})
    assert collector[0]["metrics"]["data size"] == 42


def test_collect_metrics_canonicalizes_photon_names():
    """Photon-specific metric names should be mapped to standard names."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    mock_node = _make_mock_node(
        "PhotonFileSourceScanExec",
        node_name="PhotonScan",
        metrics=[("photon rows read", 99999)],
    )

    collector: list[dict] = []
    _collect_metrics_recursive(mock_node, collector)

    # Should be stored under the canonical name
    assert collector[0]["metrics"]["number of output rows"] == 99999
    assert "photon rows read" not in collector[0]["metrics"]


def test_collect_metrics_recursive_with_children():
    """Should walk into child nodes."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    child_node = _make_mock_node("FilterExec", node_name="Filter")
    parent_node = _make_mock_node(
        "ProjectExec", node_name="Project", children=[child_node]
    )

    collector: list[dict] = []
    _collect_metrics_recursive(parent_node, collector)

    assert len(collector) == 2
    assert collector[0]["simpleClassName"] == "ProjectExec"
    assert collector[1]["simpleClassName"] == "FilterExec"


def test_collect_metrics_unwraps_aqe():
    """AdaptiveSparkPlanExec should be unwrapped via .executedPlan()."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    scan_node = _make_mock_node(
        "FileSourceScanExec",
        node_name="FileScan parquet",
        metrics=[("number of output rows", 500000)],
    )

    # Build AQE wrapper — .executedPlan() returns the scan node
    aqe_node = MagicMock()
    aqe_node.getClass.return_value.getSimpleName.return_value = "AdaptiveSparkPlanExec"
    aqe_node.executedPlan.return_value = scan_node

    collector: list[dict] = []
    _collect_metrics_recursive(aqe_node, collector)

    # Should have skipped the AQE wrapper and collected the scan node
    assert len(collector) == 1
    assert collector[0]["simpleClassName"] == "FileSourceScanExec"
    assert collector[0]["metrics"]["number of output rows"] == 500000


def test_collect_metrics_unwraps_query_stage():
    """QueryStageExec variants should be unwrapped via .plan()."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    exchange_node = _make_mock_node(
        "ShuffleExchangeExec",
        node_name="Exchange",
        metrics=[("data size", 1024000)],
    )

    # Build QueryStageExec wrapper — .plan() returns the exchange node
    stage_node = MagicMock()
    stage_node.getClass.return_value.getSimpleName.return_value = "ShuffleQueryStageExec"
    stage_node.plan.return_value = exchange_node

    collector: list[dict] = []
    _collect_metrics_recursive(stage_node, collector)

    # Should have skipped the stage wrapper and collected the exchange
    assert len(collector) == 1
    assert collector[0]["simpleClassName"] == "ShuffleExchangeExec"
    assert collector[0]["metrics"]["data size"] == 1024000


def test_collect_metrics_aqe_fallthrough_on_error():
    """If AQE .executedPlan() throws, fall through to normal traversal."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    aqe_node = _make_mock_node("AdaptiveSparkPlanExec", node_name="AdaptiveSparkPlan")
    # Override executedPlan to throw
    aqe_node.executedPlan.side_effect = RuntimeError("not yet executed")

    collector: list[dict] = []
    _collect_metrics_recursive(aqe_node, collector)

    # Falls through to normal — collects the AQE node itself
    assert len(collector) == 1
    assert collector[0]["simpleClassName"] == "AdaptiveSparkPlanExec"


def test_extract_node_metrics_with_jvm():
    """Strategy 1 (JavaConverters via _jvm) should work when jvm is provided."""
    from cluster_yield_snapshot._compat import _extract_node_metrics

    mock_metric = MagicMock()
    mock_metric.value.return_value = 77777

    mock_entry = MagicMock()
    mock_entry.getKey.return_value = "size of files read"
    mock_entry.getValue.return_value = mock_metric

    mock_iterator = MagicMock()
    call_count = {"i": 0}
    def has_next():
        return call_count["i"] < 1
    def next_entry():
        call_count["i"] += 1
        return mock_entry
    mock_iterator.hasNext = has_next
    mock_iterator.next = next_entry

    mock_entry_set = MagicMock()
    mock_entry_set.iterator.return_value = mock_iterator

    mock_java_map = MagicMock()
    mock_java_map.entrySet.return_value = mock_entry_set

    mock_converter = MagicMock()
    mock_converter.asJava.return_value = mock_java_map

    mock_jvm = MagicMock()
    mock_jvm.scala.collection.JavaConverters.mapAsJavaMapConverter.return_value = mock_converter

    mock_metrics_map = MagicMock()
    mock_node = MagicMock()
    mock_node.metrics.return_value = mock_metrics_map

    result = _extract_node_metrics(mock_node, jvm=mock_jvm)
    assert result["size of files read"] == 77777


def test_extract_node_metrics_tostring_fallback():
    """Strategy 3 (toString parsing) should work when keysIterator fails."""
    from cluster_yield_snapshot._compat import _extract_node_metrics

    mock_metric = MagicMock()
    mock_metric.value.return_value = 42

    mock_metrics_map = MagicMock()
    # keysIterator throws, forcing fallback to toString
    mock_metrics_map.keysIterator.side_effect = Exception("no keysIterator")
    mock_metrics_map.toString.return_value = "Map(data size -> SQLMetric(...))"
    mock_metrics_map.apply = lambda k: mock_metric if k == "data size" else None

    mock_node = MagicMock()
    mock_node.metrics.return_value = mock_metrics_map

    result = _extract_node_metrics(mock_node, jvm=None)
    assert result.get("data size") == 42


# ── Run all tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        exit(1)