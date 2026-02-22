"""Tests for catalog metadata capture — mocked SparkSession, no Spark."""

from unittest.mock import MagicMock, patch
from cluster_yield_snapshot.catalog import (
    get_table_stats, serialize_catalog,
    _try_describe_detail, _row_count_from_tblproperties,
    _row_count_from_describe_extended, _row_count_from_delta_history,
    _row_count_from_count_star, _get_partition_count,
    _get_column_count, get_partition_columns, _quote_name,
)


def _mock_spark():
    return MagicMock()


def _mock_row(**kwargs):
    """Create a mock Row with asDict() support."""
    m = MagicMock()
    m.asDict.return_value = kwargs
    return m


# ── _quote_name ──

def test_quote_simple():
    assert _quote_name("orders") == "`orders`"

def test_quote_qualified():
    assert _quote_name("db.orders") == "`db`.`orders`"

def test_quote_three_part():
    assert _quote_name("catalog.schema.table") == "`catalog`.`schema`.`table`"


# ── _try_describe_detail ──

def test_describe_detail_full():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        _mock_row(
            sizeInBytes=500_000_000, numFiles=100,
            partitionColumns=["date"], format="delta",
            numRecords=1_000_000, numPartitions=365,
        )
    ]
    stats = _try_describe_detail(spark, "orders")
    assert stats["sizeInBytes"] == 500_000_000
    assert stats["fileCount"] == 100
    assert stats["partitionColumns"] == ["date"]
    assert stats["format"] == "delta"
    assert stats["rowCount"] == 1_000_000
    assert stats["partitionCount"] == 365
    assert stats["avgFileSizeBytes"] == 5_000_000


def test_describe_detail_missing_num_records():
    """numRecords not present — should try SHOW TBLPROPERTIES fallback."""
    spark = _mock_spark()
    # First call: DESCRIBE DETAIL
    detail_row = _mock_row(sizeInBytes=100_000, numFiles=5, partitionColumns=[])
    # Second call: SHOW TBLPROPERTIES
    prop_row = _mock_row(key="spark.sql.statistics.numRows", value="999")

    spark.sql.return_value.collect.side_effect = [
        [detail_row],  # DESCRIBE DETAIL
        [prop_row],    # SHOW TBLPROPERTIES
    ]
    stats = _try_describe_detail(spark, "orders")
    assert stats["sizeInBytes"] == 100_000
    assert stats["rowCount"] == 999


def test_describe_detail_statistics_field():
    """Databricks Spark 4.x includes statistics field with 'NNN bytes, NNN rows'."""
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        _mock_row(
            sizeInBytes=2429, numFiles=1,
            partitionColumns=[], format="delta",
            statistics="2429 bytes, 100 rows",
        )
    ]
    stats = _try_describe_detail(spark, "test_customers")
    assert stats["sizeInBytes"] == 2429
    assert stats["rowCount"] == 100


def test_describe_detail_exception():
    spark = _mock_spark()
    spark.sql.side_effect = Exception("not a Delta table")
    stats = _try_describe_detail(spark, "orders")
    assert stats == {}


# ── Row count strategies ──

def test_row_count_from_tblproperties():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        _mock_row(key="spark.sql.statistics.numRows", value="42000"),
    ]
    assert _row_count_from_tblproperties(spark, "orders") == 42000


def test_row_count_from_tblproperties_missing():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        _mock_row(key="spark.sql.statistics.totalSize", value="100000"),
    ]
    assert _row_count_from_tblproperties(spark, "orders") is None


def test_row_count_from_tblproperties_exception():
    spark = _mock_spark()
    spark.sql.side_effect = Exception("fail")
    assert _row_count_from_tblproperties(spark, "orders") is None


def test_row_count_from_describe_extended():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        MagicMock(__getitem__=lambda s, i: ["col1", "int", ""][i]),
        MagicMock(__getitem__=lambda s, i: ["Statistics", "858993459200 bytes, 2400000000 rows", ""][i]),
    ]
    assert _row_count_from_describe_extended(spark, "orders") == 2400000000


def test_row_count_from_describe_extended_no_rows():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        MagicMock(__getitem__=lambda s, i: ["Statistics", "858993459200 bytes", ""][i]),
    ]
    assert _row_count_from_describe_extended(spark, "orders") is None


def test_row_count_from_delta_history_write():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        _mock_row(
            operation="WRITE",
            operationMetrics={"numOutputRows": "5000000"},
        )
    ]
    assert _row_count_from_delta_history(spark, "orders") == 5000000


def test_row_count_from_delta_history_ctas():
    """CREATE OR REPLACE TABLE AS SELECT should be matched."""
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        _mock_row(
            operation="CREATE OR REPLACE TABLE AS SELECT",
            operationMetrics={"numOutputRows": "100", "numOutputBytes": "2429"},
        )
    ]
    assert _row_count_from_delta_history(spark, "test_customers") == 100


def test_row_count_from_delta_history_merge_skipped():
    """MERGE operations are skipped — they don't report total rows."""
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        _mock_row(
            operation="MERGE",
            operationMetrics={"numTargetRowsInserted": "100"},
        )
    ]
    assert _row_count_from_delta_history(spark, "orders") is None


def test_row_count_from_delta_history_exception():
    spark = _mock_spark()
    spark.sql.side_effect = Exception("not delta")
    assert _row_count_from_delta_history(spark, "orders") is None


def test_row_count_from_count_star():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [{"cnt": 12345}]
    assert _row_count_from_count_star(spark, "orders") == 12345


# ── Partition metadata ──

def test_get_partition_count():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        MagicMock() for _ in range(365)
    ]
    assert _get_partition_count(spark, "orders") == 365


def test_get_partition_count_no_partitions():
    spark = _mock_spark()
    spark.sql.side_effect = Exception("not partitioned")
    assert _get_partition_count(spark, "orders") is None


def test_get_partition_columns():
    spark = _mock_spark()
    spark.sql.return_value.collect.return_value = [
        MagicMock(__getitem__=lambda s, i: ["user_id", "bigint", ""][i]),
        MagicMock(__getitem__=lambda s, i: ["amount", "decimal", ""][i]),
        MagicMock(__getitem__=lambda s, i: ["", "", ""][i]),
        MagicMock(__getitem__=lambda s, i: ["# Partition Information", "", ""][i]),
        MagicMock(__getitem__=lambda s, i: ["# col_name", "data_type", "comment"][i]),
        MagicMock(__getitem__=lambda s, i: ["order_date", "date", ""][i]),
        MagicMock(__getitem__=lambda s, i: ["", "", ""][i]),
    ]
    assert get_partition_columns(spark, "orders") == ["order_date"]


# ── Column count ──

def test_get_column_count():
    spark = _mock_spark()
    field1, field2, field3 = MagicMock(), MagicMock(), MagicMock()
    spark.table.return_value.schema.fields = [field1, field2, field3]
    assert _get_column_count(spark, "orders") == 3


def test_get_column_count_exception():
    spark = _mock_spark()
    spark.table.side_effect = Exception("fail")
    assert _get_column_count(spark, "orders") is None


# ── serialize_catalog ──

def test_serialize_includes_new_fields():
    tables = {
        "orders": {
            "sizeInBytes": 500_000_000,
            "rowCount": 1_000_000,
            "partitionColumns": ["date"],
            "partitionCount": 365,
            "avgPartitionSizeBytes": 1_369_863,
            "fileCount": 100,
            "avgFileSizeBytes": 5_000_000,
            "columnCount": 20,
            "format": "delta",
        }
    }
    result = serialize_catalog(tables)
    entry = result["orders"]
    assert entry["sizeInBytes"] == 500_000_000
    assert entry["rowCount"] == 1_000_000
    assert entry["partitionColumns"] == ["date"]
    assert entry["partitionCount"] == 365
    assert entry["avgPartitionSizeBytes"] == 1_369_863
    assert entry["columnCount"] == 20
    assert entry["format"] == "delta"


def test_serialize_minimal():
    """Table with only sizeInBytes should serialize cleanly."""
    tables = {"orders": {"sizeInBytes": 100}}
    result = serialize_catalog(tables)
    assert result["orders"]["sizeInBytes"] == 100
    assert result["orders"]["partitionColumns"] == []
    assert "partitionCount" not in result["orders"]
    assert "columnCount" not in result["orders"]


# ── get_table_stats integration ──

@patch("cluster_yield_snapshot.catalog._try_describe_detail")
@patch("cluster_yield_snapshot.catalog._get_column_count")
@patch("cluster_yield_snapshot.catalog._get_partition_count")
@patch("cluster_yield_snapshot.catalog.get_partition_columns")
@patch("cluster_yield_snapshot.catalog._row_count_from_describe_extended")
@patch("cluster_yield_snapshot.catalog._row_count_from_delta_history")
@patch("cluster_yield_snapshot._compat.get_catalyst_stats")
def test_get_table_stats_full_flow(
    mock_catalyst, mock_delta_hist, mock_desc_ext,
    mock_pcols, mock_pcount, mock_colcount, mock_detail,
):
    """Full flow: DESCRIBE DETAIL provides most, partition count added."""
    spark = _mock_spark()
    mock_detail.return_value = {
        "sizeInBytes": 800_000_000_000,
        "fileCount": 5000,
        "partitionColumns": ["date"],
        "avgFileSizeBytes": 160_000_000,
        "format": "delta",
    }
    mock_catalyst.return_value = {}
    mock_pcols.return_value = ["date"]
    mock_desc_ext.return_value = 2_400_000_000
    mock_delta_hist.return_value = None
    mock_pcount.return_value = 365
    mock_colcount.return_value = 20

    stats = get_table_stats(spark, "orders")
    assert stats["sizeInBytes"] == 800_000_000_000
    assert stats["rowCount"] == 2_400_000_000
    assert stats["partitionCount"] == 365
    assert stats["avgPartitionSizeBytes"] == 800_000_000_000 // 365
    assert stats["columnCount"] == 20


@patch("cluster_yield_snapshot.catalog._try_describe_detail")
@patch("cluster_yield_snapshot.catalog._get_column_count")
@patch("cluster_yield_snapshot.catalog._get_partition_count")
@patch("cluster_yield_snapshot.catalog.get_partition_columns")
@patch("cluster_yield_snapshot.catalog._row_count_from_describe_extended")
@patch("cluster_yield_snapshot.catalog._row_count_from_delta_history")
@patch("cluster_yield_snapshot.catalog._row_count_from_count_star")
@patch("cluster_yield_snapshot._compat.get_catalyst_stats")
def test_get_table_stats_count_rows_fallback(
    mock_catalyst, mock_count, mock_delta_hist, mock_desc_ext,
    mock_pcols, mock_pcount, mock_colcount, mock_detail,
):
    """When count_rows=True and all else fails, COUNT(*) is used."""
    spark = _mock_spark()
    mock_detail.return_value = {"sizeInBytes": 100}
    mock_catalyst.return_value = {}
    mock_pcols.return_value = []
    mock_desc_ext.return_value = None
    mock_delta_hist.return_value = None
    mock_pcount.return_value = None
    mock_colcount.return_value = 5
    mock_count.return_value = 999

    stats = get_table_stats(spark, "t", count_rows=True)
    assert stats["rowCount"] == 999
    mock_count.assert_called_once()


@patch("cluster_yield_snapshot.catalog._try_describe_detail")
@patch("cluster_yield_snapshot.catalog._get_column_count")
@patch("cluster_yield_snapshot.catalog._get_partition_count")
@patch("cluster_yield_snapshot.catalog.get_partition_columns")
@patch("cluster_yield_snapshot.catalog._row_count_from_describe_extended")
@patch("cluster_yield_snapshot.catalog._row_count_from_delta_history")
@patch("cluster_yield_snapshot.catalog._row_count_from_count_star")
@patch("cluster_yield_snapshot._compat.get_catalyst_stats")
def test_get_table_stats_no_count_rows_default(
    mock_catalyst, mock_count, mock_delta_hist, mock_desc_ext,
    mock_pcols, mock_pcount, mock_colcount, mock_detail,
):
    """When count_rows=False (default), COUNT(*) is NOT called."""
    spark = _mock_spark()
    mock_detail.return_value = {"sizeInBytes": 100}
    mock_catalyst.return_value = {}
    mock_pcols.return_value = []
    mock_desc_ext.return_value = None
    mock_delta_hist.return_value = None
    mock_pcount.return_value = None
    mock_colcount.return_value = 5

    stats = get_table_stats(spark, "t", count_rows=False)
    assert stats.get("rowCount") is None
    mock_count.assert_not_called()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn(); print(f"  ✓ {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed: exit(1)