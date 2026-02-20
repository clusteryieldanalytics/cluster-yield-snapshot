"""
Catalog metadata capture.

Extracts table size, row count, file count, partition columns, and
average file size using multiple strategies:
  1. DESCRIBE DETAIL (Delta / Databricks) — most accurate
  2. Catalyst stats from analyzed plan (classic PySpark)
  3. DESCRIBE EXTENDED parsing (universal fallback)
"""

from __future__ import annotations

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


def get_table_stats(spark: SparkSession, table_name: str) -> Optional[dict[str, Any]]:
    """
    Extract table statistics using multiple strategies.

    Priority:
      1. DESCRIBE DETAIL (Delta / Databricks) — most accurate
      2. Catalyst stats from analyzed plan (classic PySpark)
      3. DESCRIBE EXTENDED parsing (fallback)

    Returns None if no stats could be gathered.
    """
    stats: dict[str, Any] = {}

    # Strategy 1: DESCRIBE DETAIL (Delta tables on Databricks)
    stats = _try_describe_detail(spark, table_name)

    # Strategy 2: Catalyst stats (classic PySpark only)
    if not stats.get("sizeInBytes"):
        catalyst = _compat.get_catalyst_stats(spark, table_name)
        stats.update(catalyst)

    # Strategy 3: Partition columns from DESCRIBE EXTENDED
    if "partitionColumns" not in stats:
        stats["partitionColumns"] = get_partition_columns(spark, table_name)

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

        if stats.get("sizeInBytes"):
            row_info = _get_row_count(spark, table_name)
            stats.update(row_info)
    except Exception:
        pass
    return stats


def _get_row_count(spark: SparkSession, table_name: str) -> dict[str, Any]:
    """Try to get row count cheaply (without a full COUNT(*))."""
    try:
        props = spark.sql(f"SHOW TBLPROPERTIES {_quote_name(table_name)}").collect()
        for row in props:
            rd = row.asDict()
            key = rd.get("key", "")
            value = rd.get("value", "")
            if key == "spark.sql.statistics.numRows" and value:
                try:
                    return {"rowCount": int(value)}
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return {}


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
        if "fileCount" in stats:
            entry["fileCount"] = stats["fileCount"]
        if "avgFileSizeBytes" in stats:
            entry["avgFileSizeBytes"] = stats["avgFileSizeBytes"]
        result[table_name] = entry
    return result