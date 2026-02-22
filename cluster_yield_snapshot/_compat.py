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


# ── Runtime metric extraction ─────────────────────────────────────────────

# Photon and Databricks use different metric names for the same concept.
# Canonicalize to a standard set so downstream code doesn't branch.
_METRIC_ALIASES: dict[str, str] = {
    "photon rows read": "number of output rows",
    "num output rows": "number of output rows",
    "photon scan time": "scan time",
    "photon data size": "data size",
    "shuffle bytes written": "data size",
}


def get_plan_metrics(df: DataFrame) -> Optional[list[dict[str, Any]]]:
    """
    Extract runtime SQLMetric values from every node in the executed plan.

    After a DataFrame action completes, each SparkPlan node in the
    executedPlan() tree has its ``metrics`` map populated with actual
    runtime values (bytes scanned, rows read, shuffle size, etc.).
    These are the numbers visible in the Spark UI SQL tab.

    Only works on classic PySpark post-execution. Returns None on
    Spark Connect or if the JVM plan is inaccessible.
    """
    try:
        plan = df._jdf.queryExecution().executedPlan()
        nodes: list[dict[str, Any]] = []
        _collect_metrics_recursive(plan, nodes)
        # Only return if we got at least one node with actual data
        if any(n.get("metrics") for n in nodes):
            return nodes
        return None
    except Exception:
        return None


def _collect_metrics_recursive(
    plan_node: Any,
    collector: list[dict[str, Any]],
) -> None:
    """
    Walk the SparkPlan tree depth-first, extracting metrics from each node.

    Handles AQE (Adaptive Query Execution) transparently:

    - **AdaptiveSparkPlanExec**: The top-level wrapper when AQE is enabled
      (default on Spark 3.2+ and all Databricks runtimes). Its ``.children()``
      returns the *initial* plan where all metrics are zero.  We instead call
      ``.executedPlan()`` on it to get the *final* plan with populated metrics.

    - **QueryStageExec** variants (``ShuffleQueryStageExec``,
      ``BroadcastQueryStageExec``): These wrap individual stages that AQE
      materialized at runtime.  Their ``.plan()`` returns the underlying
      executed plan for that stage, which carries the real metrics.

    Each node produces::

        {
            "nodeName": "FileScan parquet ...",
            "simpleClassName": "FileSourceScanExec",
            "metrics": {"number of output rows": 184729, ...}
        }

    Metrics with value 0 are omitted to keep the snapshot compact —
    most nodes have many registered metrics that never fire.
    """
    try:
        class_name = str(plan_node.getClass().getSimpleName())
    except Exception:
        class_name = "unknown"

    # ── AQE unwrapping ────────────────────────────────────────────────
    # AdaptiveSparkPlanExec.children() → initial plan (metrics = 0).
    # AdaptiveSparkPlanExec.executedPlan → final plan (metrics populated).
    if class_name == "AdaptiveSparkPlanExec":
        try:
            final_plan = plan_node.executedPlan()
            _collect_metrics_recursive(final_plan, collector)
            return
        except Exception:
            pass  # fall through to normal traversal

    # QueryStageExec wraps a materialized stage.  .plan() has the real
    # operators with metrics; the QueryStageExec itself is just a shell.
    if "QueryStageExec" in class_name:
        try:
            stage_plan = plan_node.plan()
            _collect_metrics_recursive(stage_plan, collector)
            return
        except Exception:
            pass  # fall through to normal traversal

    # ── Normal node processing ────────────────────────────────────────
    try:
        node_name = str(plan_node.nodeName())
    except Exception:
        node_name = class_name

    node_info: dict[str, Any] = {
        "nodeName": node_name,
        "simpleClassName": class_name,
    }

    metrics = _extract_node_metrics(plan_node)
    if metrics:
        node_info["metrics"] = metrics

    collector.append(node_info)

    # Recurse into children
    try:
        children = plan_node.children()
        size = children.size()
        for i in range(size):
            _collect_metrics_recursive(children.apply(i), collector)
    except Exception:
        pass


def _extract_node_metrics(plan_node: Any) -> dict[str, int]:
    """
    Extract the metrics map from a single SparkPlan node.

    Tries two JVM iteration strategies:
      1. Scala .toSeq() — works on most Spark/Databricks versions
      2. JavaConverters — fallback for environments where .toSeq() fails

    Returns only non-zero metrics. Canonicalizes Photon/Databricks
    metric names via _METRIC_ALIASES.
    """
    metrics: dict[str, int] = {}

    try:
        metrics_map = plan_node.metrics()
    except Exception:
        return metrics

    # Strategy 1: iterate via .toSeq()
    try:
        seq = metrics_map.toSeq()
        size = seq.size()
        for i in range(size):
            pair = seq.apply(i)
            name = str(pair._1())
            try:
                value = int(pair._2().value())
                if value != 0:
                    canonical = _METRIC_ALIASES.get(name, name)
                    metrics[canonical] = value
            except Exception:
                pass
        return metrics
    except Exception:
        pass

    # Strategy 2: JavaConverters fallback (Scala 2.12 / Databricks)
    try:
        # Access the JVM gateway through the py4j bridge
        gateway = plan_node._gateway  # type: ignore[attr-defined]
        java_map = (
            gateway.jvm.scala.collection.JavaConverters
            .mapAsJavaMapConverter(metrics_map)
            .asJava()
        )
        for name_obj, metric_obj in java_map.items():
            name = str(name_obj)
            try:
                value = int(metric_obj.value())
                if value != 0:
                    canonical = _METRIC_ALIASES.get(name, name)
                    metrics[canonical] = value
            except Exception:
                pass
    except Exception:
        pass

    return metrics