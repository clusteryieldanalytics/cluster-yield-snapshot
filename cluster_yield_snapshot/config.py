"""
Spark configuration capture and drift detection.

Captures optimizer-relevant Spark SQL settings and identifies
non-default values that may affect physical plan strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _compat

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Spark configs that affect physical plan strategy
OPTIMIZER_CONFIGS = [
    "spark.sql.adaptive.enabled",
    "spark.sql.adaptive.coalescePartitions.enabled",
    "spark.sql.adaptive.skewJoin.enabled",
    "spark.sql.autoBroadcastJoinThreshold",
    "spark.sql.shuffle.partitions",
    "spark.sql.adaptive.advisoryPartitionSizeInBytes",
    "spark.sql.join.preferSortMergeJoin",
    "spark.sql.optimizer.dynamicPartitionPruning.enabled",
    "spark.sql.adaptive.localShuffleReader.enabled",
    "spark.sql.adaptive.autoBroadcastJoinThreshold",
    "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold",
    "spark.sql.broadcastTimeout",
    "spark.sql.optimizer.nestedSchemaPruning.enabled",
    "spark.sql.files.maxPartitionBytes",
    "spark.sql.files.openCostInBytes",
]

# Known Spark defaults — used for drift detection
SPARK_DEFAULTS: dict[str, str] = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.autoBroadcastJoinThreshold": "10485760",
    "spark.sql.shuffle.partitions": "200",
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "67108864",
    "spark.sql.join.preferSortMergeJoin": "true",
    "spark.sql.optimizer.dynamicPartitionPruning.enabled": "true",
}


def capture_config(spark: SparkSession) -> dict[str, Any]:
    """
    Capture effective Spark config with non-default identification.

    Returns:
        {
            "all": { <all spark.sql.* settings> },
            "optimizerRelevant": { <settings that affect plans> },
            "nonDefault": {
                "spark.sql.shuffle.partitions": {
                    "value": "400",
                    "sparkDefault": "200"
                }
            }
        }
    """
    all_conf = _compat.get_conf_dict(spark)

    # Extract optimizer-relevant settings
    optimizer: dict[str, str] = {}
    for key in OPTIMIZER_CONFIGS:
        val = _compat.get_conf_value(spark, key)
        if val is not None:
            optimizer[key] = val

    # Identify non-default settings
    non_default: dict[str, dict[str, str]] = {}
    for key, val in optimizer.items():
        default = SPARK_DEFAULTS.get(key)
        if default is not None and str(val) != default:
            non_default[key] = {
                "value": str(val),
                "sparkDefault": default,
            }

    return {
        "all": {
            k: str(v) for k, v in sorted(all_conf.items())
            if k.startswith("spark.sql.")
        },
        "optimizerRelevant": {k: str(v) for k, v in sorted(optimizer.items())},
        "nonDefault": non_default,
    }
