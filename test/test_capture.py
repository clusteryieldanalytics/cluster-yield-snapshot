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