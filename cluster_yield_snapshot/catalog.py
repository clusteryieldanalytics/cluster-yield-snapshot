"""
Catalog metadata capture.

Extracts table size, row count, file count, partition columns,
partition count, average partition size, and column count using
multiple strategies ordered by cost (cheapest first):

  1. DESCRIBE DETAIL (Delta / Databricks) — most accurate
  2. Catalyst stats from analyzed plan (classic PySpark)
  3. DESCRIBE EXTENDED parsing (universal fallback)
  4. Delta DESCRIBE HISTORY for row count (cheap, 1 row)
  5. SHOW PARTITIONS for partition count (cheap)
  6. Schema inspection for column count (cheap)
  7. COUNT(*) for row count (expensive, opt-in)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from . import _compat

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Columns that look like join keys but almost never are
GENERIC_COLUMNS = frozenset({
    "id", "created_at", "updated_at", "timestamp",
    "_rescued_data", "__index_level_0__",
})


def _quote_name(table_name: str) -> str:
    """
    Quote a possibly-qualified table name for SQL.

    'workspace.test_cy.orders' → '`workspace`.`test_cy`.`orders`'
    'orders'                   → '`orders`'

    Each segment is quoted separately so Unity Catalog resolves
    catalog.schema.table correctly.
    """
    parts = table_name.replace("`", "").split(".")
    return ".".join(f"`{p}`" for p in parts)


def get_table_stats(
    spark: SparkSession,
    table_name: str,
    *,
    count_rows: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Extract table statistics using multiple strategies.

    Priority (cheapest first):
      1. DESCRIBE DETAIL (Delta / Databricks) — most accurate
      2. Catalyst stats from analyzed plan (classic PySpark)
      3. DESCRIBE EXTENDED parsing (fallback)
      4. Delta DESCRIBE HISTORY for row count (if still missing)
      5. SHOW PARTITIONS for partition count
      6. Schema for column count
      7. COUNT(*) for row count (only if count_rows=True)

    Args:
        spark: Active SparkSession
        table_name: Fully qualified or short table name
        count_rows: If True, fall back to COUNT(*) when rowCount
            cannot be obtained cheaply. Expensive on large tables.

    Returns None if no stats could be gathered.
    """
    stats: dict[str, Any] = {}

    # Strategy 1: DESCRIBE DETAIL (Delta tables on Databricks)
    stats = _try_describe_detail(spark, table_name)

    # Strategy 2: Catalyst stats (classic PySpark only)
    if not stats.get("sizeInBytes"):
        catalyst = _compat.get_catalyst_stats(spark, table_name)
        stats.update(catalyst)

    # Strategy 3: Partition columns from DESCRIBE EXTENDED (+ rowCount attempt)
    if "partitionColumns" not in stats:
        stats["partitionColumns"] = get_partition_columns(spark, table_name)

    # Strategy 3b: rowCount from DESCRIBE EXTENDED Statistics line
    if not stats.get("rowCount"):
        rc = _row_count_from_describe_extended(spark, table_name)
        if rc is not None:
            stats["rowCount"] = rc

    # Strategy 4: rowCount from Delta DESCRIBE HISTORY
    if not stats.get("rowCount"):
        rc = _row_count_from_delta_history(spark, table_name)
        if rc is not None:
            stats["rowCount"] = rc

    # Strategy 5: Partition count from SHOW PARTITIONS
    pcols = stats.get("partitionColumns", [])
    if pcols and "partitionCount" not in stats:
        pcount = _get_partition_count(spark, table_name)
        if pcount is not None:
            stats["partitionCount"] = pcount
            total_size = stats.get("sizeInBytes")
            if total_size and pcount > 0:
                stats["avgPartitionSizeBytes"] = total_size // pcount

    # Strategy 6: Column count from schema
    if "columnCount" not in stats:
        cc = _get_column_count(spark, table_name)
        if cc is not None:
            stats["columnCount"] = cc

    # Strategy 7 (opt-in): COUNT(*) for row count — last resort
    if count_rows and not stats.get("rowCount"):
        rc = _row_count_from_count_star(spark, table_name)
        if rc is not None:
            stats["rowCount"] = rc

    return stats if stats else None


def _try_describe_detail(spark: SparkSession, table_name: str) -> dict[str, Any]:
    """Try DESCRIBE DETAIL (Delta / Databricks) for table stats."""
    stats: dict[str, Any] = {}
    try:
        detail = spark.sql(f"DESCRIBE DETAIL {_quote_name(table_name)}").collect()
        if not detail:
            return stats
        rd = detail[0].asDict()

        if "sizeInBytes" in rd and rd["sizeInBytes"] is not None:
            stats["sizeInBytes"] = int(rd["sizeInBytes"])
        elif "size" in rd and rd["size"] is not None:
            stats["sizeInBytes"] = int(rd["size"])

        if "numFiles" in rd and rd["numFiles"] is not None:
            stats["fileCount"] = int(rd["numFiles"])

        if "partitionColumns" in rd and rd["partitionColumns"]:
            pcols = rd["partitionColumns"]
            if isinstance(pcols, (list, tuple)):
                stats["partitionColumns"] = [str(c) for c in pcols if c]

        if stats.get("sizeInBytes") and stats.get("fileCount"):
            stats["avgFileSizeBytes"] = (
                stats["sizeInBytes"] // max(stats["fileCount"], 1)
            )

        # Some Databricks builds include numRecords in DESCRIBE DETAIL
        if "numRecords" in rd and rd["numRecords"] is not None:
            try:
                stats["rowCount"] = int(rd["numRecords"])
            except (ValueError, TypeError):
                pass

        # Databricks Spark 4.x+ includes a `statistics` field in DESCRIBE DETAIL
        # containing "NNN bytes, NNN rows" — same format as DESCRIBE EXTENDED
        if not stats.get("rowCount") and "statistics" in rd and rd["statistics"]:
            stat_str = str(rd["statistics"])
            m = re.search(r"(\d[\d,]*)\s+rows", stat_str)
            if m:
                try:
                    stats["rowCount"] = int(m.group(1).replace(",", ""))
                except (ValueError, TypeError):
                    pass

        # Fallback: try SHOW TBLPROPERTIES for row count
        if not stats.get("rowCount") and stats.get("sizeInBytes"):
            row_info = _row_count_from_tblproperties(spark, table_name)
            if row_info is not None:
                stats["rowCount"] = row_info

        # numPartitions from DESCRIBE DETAIL (some builds)
        if "numPartitions" in rd and rd["numPartitions"] is not None:
            try:
                stats["partitionCount"] = int(rd["numPartitions"])
            except (ValueError, TypeError):
                pass

        # format field (delta, parquet, etc.)
        if "format" in rd and rd["format"]:
            stats["format"] = str(rd["format"]).lower()

    except Exception:
        pass
    return stats


# ── Row count strategies ────────────────────────────────────────────────

def _row_count_from_tblproperties(
    spark: SparkSession, table_name: str
) -> Optional[int]:
    """Try SHOW TBLPROPERTIES for spark.sql.statistics.numRows."""
    try:
        props = spark.sql(f"SHOW TBLPROPERTIES {_quote_name(table_name)}").collect()
        for row in props:
            rd = row.asDict()
            key = rd.get("key", "")
            value = rd.get("value", "")
            if key == "spark.sql.statistics.numRows" and value:
                try:
                    v = int(value)
                    if v > 0:
                        return v
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return None


def _row_count_from_describe_extended(
    spark: SparkSession, table_name: str
) -> Optional[int]:
    """
    Parse rowCount from DESCRIBE EXTENDED output.

    Spark stores ANALYZE TABLE results in the table's Statistics metadata,
    visible in DESCRIBE EXTENDED as a line like:
        Statistics    | 858993459200 bytes, 2400000000 rows
    or in newer Spark:
        Statistics    | 800.0 GiB, 2.4 billion rows

    We parse the raw bytes + row count from this.
    """
    try:
        rows = spark.sql(f"DESCRIBE EXTENDED {_quote_name(table_name)}").collect()
        for row in rows:
            col_name = str(row[0]).strip() if row[0] else ""
            data_type = str(row[1]).strip() if row[1] else ""
            if col_name == "Statistics" and data_type:
                # Look for "NNN rows" pattern
                m = re.search(r"(\d[\d,]*)\s+rows", data_type)
                if m:
                    return int(m.group(1).replace(",", ""))
    except Exception:
        pass
    return None


def _row_count_from_delta_history(
    spark: SparkSession, table_name: str
) -> Optional[int]:
    """
    Extract row count from the most recent Delta DESCRIBE HISTORY.

    The operationMetrics map on write-like operations includes
    numOutputRows. This covers WRITE, CREATE TABLE AS SELECT,
    CREATE OR REPLACE TABLE AS SELECT, REPLACE TABLE AS SELECT,
    and similar operations where the metric represents total rows
    written to the table.

    Returns None for non-Delta tables or if no useful metric is found.
    """
    # Operations where numOutputRows represents total rows in the table
    _WRITE_OPS = frozenset({
        "WRITE",
        "CREATE TABLE AS SELECT",
        "CREATE OR REPLACE TABLE AS SELECT",
        "REPLACE TABLE AS SELECT",
        "SHALLOW CLONE",
        "CLONE",
        "RESTORE",
        "CONVERT",
    })

    try:
        history = spark.sql(
            f"DESCRIBE HISTORY {_quote_name(table_name)} LIMIT 5"
        ).collect()
        for entry in history:
            rd = entry.asDict()
            metrics = rd.get("operationMetrics")
            if not metrics or not isinstance(metrics, dict):
                continue
            op = str(rd.get("operation", "")).upper()

            if op in _WRITE_OPS:
                for key in ("numOutputRows", "numOutputRecords"):
                    val = metrics.get(key)
                    if val is not None:
                        try:
                            v = int(val)
                            if v > 0:
                                return v
                        except (ValueError, TypeError):
                            pass

            # MERGE/UPDATE/DELETE — harder, we'd need to sum up.
            # For now, skip these — the number isn't total rows.
    except Exception:
        pass
    return None


def _row_count_from_count_star(
    spark: SparkSession, table_name: str
) -> Optional[int]:
    """
    Get exact row count via COUNT(*).

    Expensive on large tables — only call when explicitly opted in.
    Uses LIMIT 1 on the aggregate to avoid full result materialization.
    """
    try:
        result = spark.sql(
            f"SELECT COUNT(*) AS cnt FROM {_quote_name(table_name)}"
        ).collect()
        if result:
            return int(result[0]["cnt"])
    except Exception:
        pass
    return None


# ── Partition metadata ──────────────────────────────────────────────────

def _get_partition_count(
    spark: SparkSession, table_name: str
) -> Optional[int]:
    """
    Count partitions via SHOW PARTITIONS.

    Works for Hive-style partitioned tables and Delta tables on Databricks.
    Returns None for non-partitioned tables or unsupported formats.
    """
    try:
        parts = spark.sql(f"SHOW PARTITIONS {_quote_name(table_name)}").collect()
        if parts:
            return len(parts)
    except Exception:
        # Expected for non-partitioned tables, Iceberg, etc.
        pass
    return None


def get_partition_columns(spark: SparkSession, table_name: str) -> list[str]:
    """Extract partition columns via DESCRIBE EXTENDED."""
    try:
        rows = spark.sql(f"DESCRIBE EXTENDED {_quote_name(table_name)}").collect()
        in_partition_section = False
        columns: list[str] = []
        for row in rows:
            col_name = str(row[0]).strip() if row[0] else ""
            if col_name == "# Partition Information":
                in_partition_section = True
                continue
            if in_partition_section:
                if col_name.startswith("# ") or col_name == "":
                    if columns:
                        break
                    continue
                if col_name and not col_name.startswith("#"):
                    columns.append(col_name)
        return columns
    except Exception:
        return []


# ── Column count ────────────────────────────────────────────────────────

def _get_column_count(spark: SparkSession, table_name: str) -> Optional[int]:
    """
    Get total column count from the table schema.

    Uses spark.table().schema which works on both classic PySpark
    and Spark Connect. Excludes hidden partition columns from the
    count (they're metadata, not data columns).
    """
    try:
        schema = spark.table(_quote_name(table_name)).schema
        return len(schema.fields)
    except Exception:
        pass
    return None


# ── Table listing ───────────────────────────────────────────────────────

def list_tables(spark: SparkSession, database: str) -> list[str]:
    """List non-temporary table names in a database."""
    try:
        tables = spark.catalog.listTables(database)
        return [t.name for t in tables if not t.isTemporary]
    except Exception:
        # Fallback for Unity Catalog or non-standard catalogs
        try:
            rows = spark.sql(f"SHOW TABLES IN `{database}`").collect()
            return [row["tableName"] for row in rows]
        except Exception:
            return []


def current_database(spark: SparkSession) -> str:
    """Get the current database name."""
    try:
        return spark.catalog.currentDatabase()
    except Exception:
        return "default"


def get_table_columns(spark: SparkSession, fq_name: str) -> Optional[set[str]]:
    """Get column names for a table, or None on failure."""
    try:
        return {f.name for f in spark.table(fq_name).schema.fields}
    except Exception:
        return None


# ── Serialization ───────────────────────────────────────────────────────

def serialize_catalog(tables: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Serialize catalog stats for the snapshot JSON envelope.

    Normalizes and filters fields to the canonical set.
    """
    result: dict[str, dict[str, Any]] = {}
    for table_name, stats in tables.items():
        entry: dict[str, Any] = {}
        if "sizeInBytes" in stats:
            entry["sizeInBytes"] = stats["sizeInBytes"]
        if "rowCount" in stats:
            entry["rowCount"] = stats["rowCount"]
        entry["partitionColumns"] = stats.get("partitionColumns", [])
        if "partitionCount" in stats:
            entry["partitionCount"] = stats["partitionCount"]
        if "avgPartitionSizeBytes" in stats:
            entry["avgPartitionSizeBytes"] = stats["avgPartitionSizeBytes"]
        if "fileCount" in stats:
            entry["fileCount"] = stats["fileCount"]
        if "avgFileSizeBytes" in stats:
            entry["avgFileSizeBytes"] = stats["avgFileSizeBytes"]
        if "columnCount" in stats:
            entry["columnCount"] = stats["columnCount"]
        if "format" in stats:
            entry["format"] = stats["format"]
        result[table_name] = entry
    return result