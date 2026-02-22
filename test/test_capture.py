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

    # Build a mock JVM plan node with one metric
    mock_metric = MagicMock()
    mock_metric.value.return_value = 12345

    mock_pair = MagicMock()
    mock_pair._1.return_value = "number of output rows"
    mock_pair._2.return_value = mock_metric

    mock_seq = MagicMock()
    mock_seq.size.return_value = 1
    mock_seq.apply.return_value = mock_pair

    mock_metrics_map = MagicMock()
    mock_metrics_map.toSeq.return_value = mock_seq

    mock_node = MagicMock()
    mock_node.nodeName.return_value = "FileScan parquet"
    mock_node.getClass.return_value.getSimpleName.return_value = "FileSourceScanExec"
    mock_node.metrics.return_value = mock_metrics_map
    # No children
    mock_children = MagicMock()
    mock_children.size.return_value = 0
    mock_node.children.return_value = mock_children

    collector: list[dict] = []
    _collect_metrics_recursive(mock_node, collector)

    assert len(collector) == 1
    assert collector[0]["nodeName"] == "FileScan parquet"
    assert collector[0]["simpleClassName"] == "FileSourceScanExec"
    assert collector[0]["metrics"]["number of output rows"] == 12345


def test_collect_metrics_skips_zero_values():
    """Metrics with value 0 should be omitted."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    mock_metric_zero = MagicMock()
    mock_metric_zero.value.return_value = 0
    mock_metric_nonzero = MagicMock()
    mock_metric_nonzero.value.return_value = 42

    mock_pair_zero = MagicMock()
    mock_pair_zero._1.return_value = "spill size"
    mock_pair_zero._2.return_value = mock_metric_zero

    mock_pair_real = MagicMock()
    mock_pair_real._1.return_value = "data size"
    mock_pair_real._2.return_value = mock_metric_nonzero

    mock_seq = MagicMock()
    mock_seq.size.return_value = 2
    mock_seq.apply.side_effect = [mock_pair_zero, mock_pair_real]

    mock_metrics_map = MagicMock()
    mock_metrics_map.toSeq.return_value = mock_seq

    mock_node = MagicMock()
    mock_node.nodeName.return_value = "Exchange"
    mock_node.getClass.return_value.getSimpleName.return_value = "ShuffleExchangeExec"
    mock_node.metrics.return_value = mock_metrics_map
    mock_children = MagicMock()
    mock_children.size.return_value = 0
    mock_node.children.return_value = mock_children

    collector: list[dict] = []
    _collect_metrics_recursive(mock_node, collector)

    assert "spill size" not in collector[0].get("metrics", {})
    assert collector[0]["metrics"]["data size"] == 42


def test_collect_metrics_canonicalizes_photon_names():
    """Photon-specific metric names should be mapped to standard names."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    mock_metric = MagicMock()
    mock_metric.value.return_value = 99999

    mock_pair = MagicMock()
    mock_pair._1.return_value = "photon rows read"  # should become "number of output rows"
    mock_pair._2.return_value = mock_metric

    mock_seq = MagicMock()
    mock_seq.size.return_value = 1
    mock_seq.apply.return_value = mock_pair

    mock_metrics_map = MagicMock()
    mock_metrics_map.toSeq.return_value = mock_seq

    mock_node = MagicMock()
    mock_node.nodeName.return_value = "PhotonScan"
    mock_node.getClass.return_value.getSimpleName.return_value = "PhotonFileSourceScanExec"
    mock_node.metrics.return_value = mock_metrics_map
    mock_children = MagicMock()
    mock_children.size.return_value = 0
    mock_node.children.return_value = mock_children

    collector: list[dict] = []
    _collect_metrics_recursive(mock_node, collector)

    # Should be stored under the canonical name
    assert collector[0]["metrics"]["number of output rows"] == 99999
    assert "photon rows read" not in collector[0]["metrics"]


def test_collect_metrics_recursive_with_children():
    """Should walk into child nodes."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    # Build child node (no metrics, no children)
    child_node = MagicMock()
    child_node.nodeName.return_value = "Filter"
    child_node.getClass.return_value.getSimpleName.return_value = "FilterExec"
    child_metrics_map = MagicMock()
    child_metrics_map.toSeq.return_value.size.return_value = 0
    child_node.metrics.return_value = child_metrics_map
    child_children = MagicMock()
    child_children.size.return_value = 0
    child_node.children.return_value = child_children

    # Build parent node with one child
    parent_node = MagicMock()
    parent_node.nodeName.return_value = "Project"
    parent_node.getClass.return_value.getSimpleName.return_value = "ProjectExec"
    parent_metrics_map = MagicMock()
    parent_metrics_map.toSeq.return_value.size.return_value = 0
    parent_node.metrics.return_value = parent_metrics_map
    parent_children = MagicMock()
    parent_children.size.return_value = 1
    parent_children.apply.return_value = child_node
    parent_node.children.return_value = parent_children

    collector: list[dict] = []
    _collect_metrics_recursive(parent_node, collector)

    assert len(collector) == 2
    assert collector[0]["simpleClassName"] == "ProjectExec"
    assert collector[1]["simpleClassName"] == "FilterExec"


def test_collect_metrics_unwraps_aqe():
    """AdaptiveSparkPlanExec should be unwrapped via .executedPlan()."""
    from cluster_yield_snapshot._compat import _collect_metrics_recursive

    # Build the real scan node inside the final plan (has metrics)
    mock_metric = MagicMock()
    mock_metric.value.return_value = 500000

    mock_pair = MagicMock()
    mock_pair._1.return_value = "number of output rows"
    mock_pair._2.return_value = mock_metric

    mock_seq = MagicMock()
    mock_seq.size.return_value = 1
    mock_seq.apply.return_value = mock_pair

    scan_node = MagicMock()
    scan_node.nodeName.return_value = "FileScan parquet"
    scan_node.getClass.return_value.getSimpleName.return_value = "FileSourceScanExec"
    scan_node.metrics.return_value.toSeq.return_value = mock_seq
    scan_children = MagicMock()
    scan_children.size.return_value = 0
    scan_node.children.return_value = scan_children

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

    # Build the real exchange node inside the query stage
    mock_metric = MagicMock()
    mock_metric.value.return_value = 1024000

    mock_pair = MagicMock()
    mock_pair._1.return_value = "data size"
    mock_pair._2.return_value = mock_metric

    mock_seq = MagicMock()
    mock_seq.size.return_value = 1
    mock_seq.apply.return_value = mock_pair

    exchange_node = MagicMock()
    exchange_node.nodeName.return_value = "Exchange"
    exchange_node.getClass.return_value.getSimpleName.return_value = "ShuffleExchangeExec"
    exchange_node.metrics.return_value.toSeq.return_value = mock_seq
    exchange_children = MagicMock()
    exchange_children.size.return_value = 0
    exchange_node.children.return_value = exchange_children

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

    aqe_node = MagicMock()
    aqe_node.getClass.return_value.getSimpleName.return_value = "AdaptiveSparkPlanExec"
    aqe_node.executedPlan.side_effect = RuntimeError("not yet executed")
    aqe_node.nodeName.return_value = "AdaptiveSparkPlan"
    # No metrics on the wrapper itself
    aqe_node.metrics.return_value.toSeq.return_value.size.return_value = 0
    aqe_children = MagicMock()
    aqe_children.size.return_value = 0
    aqe_node.children.return_value = aqe_children

    collector: list[dict] = []
    _collect_metrics_recursive(aqe_node, collector)

    # Falls through to normal — collects the AQE node itself
    assert len(collector) == 1
    assert collector[0]["simpleClassName"] == "AdaptiveSparkPlanExec"


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