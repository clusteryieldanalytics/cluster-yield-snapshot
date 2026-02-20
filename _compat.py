"""
Spark version detection and compatibility helpers.

Abstracts differences between classic PySpark (JVM-backed) and
Spark Connect (serverless, Databricks Connect) so the rest of the
package doesn't have to care.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


class SparkMode(Enum):
    """Which PySpark API surface is available."""
    CLASSIC = "classic"     # Full JVM access via _jdf, _jvm, sparkContext
    CONNECT = "connect"     # Spark Connect — no JVM, limited APIs


def detect_mode(spark: SparkSession) -> SparkMode:
    """Detect whether we're running classic PySpark or Spark Connect."""
    try:
        # Spark Connect sessions don't expose sparkContext
        _ = spark.sparkContext
        return SparkMode.CLASSIC
    except Exception:
        return SparkMode.CONNECT


def get_plan_json(df: DataFrame) -> Optional[str]:
    """
    Extract the physical plan as JSON from a DataFrame.

    Returns the JSON string on classic PySpark, None on Spark Connect
    (where _jdf is not available).
    """
    try:
        executed = df._jdf.queryExecution().executedPlan()
        return executed.toJSON()
    except (AttributeError, TypeError):
        return None


def get_plan_text(df: DataFrame) -> Optional[str]:
    """
    Extract the explain text from a DataFrame.

    Works on both classic PySpark and Spark Connect.
    """
    # Strategy 1: _explain_string (PySpark 3.4+, Spark Connect)
    try:
        return df._explain_string(extended=True)
    except AttributeError:
        pass

    # Strategy 2: Capture explain() stdout (older PySpark)
    try:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df.explain(extended=True)
        text = buf.getvalue()
        return text if text.strip() else None
    except Exception:
        return None


def get_conf_dict(spark: SparkSession) -> dict[str, str]:
    """
    Get all Spark SQL config as a dict.

    Tries sparkContext.getConf() first (classic), falls back to SET -v (Connect).
    """
    try:
        return dict(spark.sparkContext.getConf().getAll())
    except Exception:
        pass

    # Serverless / Connect fallback
    conf: dict[str, str] = {}
    try:
        for row in spark.sql("SET -v").collect():
            rd = row.asDict()
            key = rd.get("key", "")
            value = rd.get("value", "")
            if key and isinstance(key, str):
                conf[key] = str(value)
    except Exception:
        pass
    return conf


def get_conf_value(spark: SparkSession, key: str) -> Optional[str]:
    """Get a single Spark config value, returning None if unavailable."""
    try:
        return spark.conf.get(key)
    except Exception:
        return None


def get_catalyst_stats(spark: SparkSession, table_name: str) -> dict[str, Any]:
    """
    Extract table stats from Catalyst's analyzed plan.

    Only works on classic PySpark (requires _jdf). Returns an empty dict
    on Spark Connect.
    """
    stats: dict[str, Any] = {}
    try:
        plan = spark.table(table_name)._jdf.queryExecution().analyzed()
        jstats = plan.stats()

        size_bytes = jstats.sizeInBytes()
        if size_bytes is not None:
            stats["sizeInBytes"] = int(str(size_bytes))

        row_count = jstats.rowCount()
        if row_count is not None and row_count.isDefined():
            stats["rowCount"] = int(str(row_count.get()))
    except Exception:
        pass
    return stats


def get_executor_info(spark: SparkSession) -> dict[str, Any]:
    """Get cluster shape info (executor count, parallelism). Classic only."""
    info: dict[str, Any] = {}
    try:
        sc = spark.sparkContext
        info["executorCount"] = sc._jsc.sc().getExecutorMemoryStatus().size()
        info["defaultParallelism"] = sc.defaultParallelism
    except Exception:
        pass
    return info
