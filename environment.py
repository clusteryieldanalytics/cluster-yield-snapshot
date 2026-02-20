"""
Spark platform and runtime environment detection.

Detects Databricks (serverless, classic), YARN (EMR, Dataproc),
and other Spark environments. Captures cluster shape when accessible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _compat

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def capture_environment(spark: SparkSession, captured_at: str) -> dict[str, Any]:
    """
    Detect the Spark platform, runtime version, and cluster shape.

    Returns:
        {
            "sparkVersion": "3.5.1",
            "capturedAt": "2026-02-19T...",
            "platform": "databricks",
            "databricksRuntime": "15.4",
            "computeType": "serverless",
            "executorCount": 8,
            "defaultParallelism": 32
        }
    """
    env: dict[str, Any] = {
        "sparkVersion": spark.version,
        "capturedAt": captured_at,
    }

    # Detect Databricks runtime
    dbr = _compat.get_conf_value(
        spark, "spark.databricks.clusterUsageTags.sparkVersion"
    )
    if dbr:
        env["platform"] = "databricks"
        env["databricksRuntime"] = dbr

    # Detect serverless vs classic
    cluster_type = _compat.get_conf_value(
        spark, "spark.databricks.clusterUsageTags.clusterType"
    )
    if cluster_type:
        if "serverless" in cluster_type.lower():
            env["computeType"] = "serverless"
        else:
            env["computeType"] = cluster_type

    # Cluster shape (classic PySpark only)
    env.update(_compat.get_executor_info(spark))

    # Platform fallback detection
    if "platform" not in env:
        env["platform"] = _detect_platform_fallback(spark)

    return env


def _detect_platform_fallback(spark: SparkSession) -> str:
    """Detect platform via config probing when primary detection fails."""
    probes = [
        ("spark.databricks.cloudProvider", "databricks"),
        ("spark.yarn.app.id", "yarn"),       # EMR / Dataproc
        ("spark.kubernetes.driver.pod.name", "kubernetes"),
    ]
    for config_key, platform in probes:
        if _compat.get_conf_value(spark, config_key) is not None:
            return platform
    return "unknown"
