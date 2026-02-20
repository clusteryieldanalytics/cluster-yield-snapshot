# cluster-yield-snapshot

Passive Spark plan capture for [Cluster Yield](https://clusteryield.com) analysis. Drop two lines into any notebook — no refactoring, no query registration, no code changes.

Works on Databricks (serverless + classic), EMR, Dataproc, and open-source Spark.

## Install

```bash
pip install cluster-yield-snapshot

# In a Databricks notebook
%pip install cluster-yield-snapshot
```

## How it works

Two lines at the top. Two lines at the bottom. Everything in between is untouched:

```python
# Cell 1 — start capture
from cluster_yield_snapshot import CYSnapshot
cy = CYSnapshot(spark).start()
```

```python
# ═══════════════════════════════════════════
# Rest of the notebook — completely unchanged
# ═══════════════════════════════════════════
df = spark.sql("SELECT * FROM orders WHERE date > '2024-01-01'")
users = spark.table("analytics.users")
enriched = df.join(users, "user_id").groupBy("region").agg(sum("amount"))
enriched.write.parquet("s3://output/regional_revenue")
```

```python
# Last cell — harvest
cy.stop().save()
```

That's it. Every `spark.sql()` call, every `.collect()`, every `.write.parquet()` in between is silently captured with its full physical plan. On `stop()`, catalog stats (table sizes, partitions, file counts) are automatically gathered for every table that appeared in the plans.

## What it captures

`start()` hooks into three places:

| Hook | What it catches | Plan timing |
|------|----------------|-------------|
| `spark.sql()` | Every SQL query | At creation (pre-AQE) |
| DataFrame actions (`.collect()`, `.show()`, `.count()`, `.toPandas()`, etc.) | Execution results | Post-AQE (final plan) |
| Write methods (`.write.parquet()`, `.save()`, `.saveAsTable()`, etc.) | Data output | Post-AQE (final plan) |

When the same query is captured at both `spark.sql()` time and action time, the action-time plan (post-AQE) replaces the earlier one. You get the plan Spark *actually executed*, not just the plan it *intended* to execute.

`stop()` then collects catalog metadata:

| Data | Source |
|------|--------|
| Table size (bytes) | `DESCRIBE DETAIL` / Catalyst stats |
| Row count | Table properties / Catalyst stats |
| File count, avg file size | `DESCRIBE DETAIL` |
| Partition columns | `DESCRIBE EXTENDED` |
| Spark config + drift | `sparkContext.getConf()` / `SET -v` |
| Environment | Platform detection (Databricks / YARN / K8s) |

## Upload to Cluster Yield

The server analyzes on ingest — runs detectors, estimates costs, diffs against your last snapshot:

```python
cy = CYSnapshot(spark, api_key="cy_...", environment="prod-analytics").start()
# ... notebook ...
cy.stop().upload()
```

Install with upload support: `pip install cluster-yield-snapshot[upload]`

## Context manager

```python
with CYSnapshot(spark) as cy:
    df = spark.sql("SELECT ...")
    df.show()

cy.save()
```

## Manual capture (edge cases)

For queries you can't run through `start()`/`stop()` (e.g. building a snapshot from known queries without executing them):

```python
cy = CYSnapshot(spark)
cy.query("daily_revenue", "SELECT region, SUM(amount) FROM orders GROUP BY region")
cy.df("enriched", some_existing_dataframe)
cy.save()
```

## Safety

The capture hooks are read-only and wrapped in `try/except`:

- They only read `queryExecution.executedPlan` — no writes, no modifications
- If our code fails for any reason, the user's code continues normally
- `stop()` cleanly restores all original methods
- A re-entrancy guard prevents our internal Spark calls (catalog stats) from being captured
- The notebook behaves identically with or without capture running

## Snapshot JSON envelope

```json
{
  "snapshot": { "version": "0.3.0", "capturedAt": "...", "snapshotType": "environment" },
  "environment": { "sparkVersion": "3.5.1", "platform": "databricks", ... },
  "config": { "all": {}, "optimizerRelevant": {}, "nonDefault": {} },
  "catalog": { "tables": { "default.orders": { "sizeInBytes": 85899345920, ... } } },
  "plans": [
    {
      "label": "sql-1-SELECT * FROM orders WHERE ...",
      "fingerprint": "a1b2c3d4...",
      "plan": [...],
      "sql": "SELECT * FROM orders WHERE date > '2024-01-01'",
      "trigger": "action.collect"
    }
  ],
  "errors": null
}
```

Compatible with the Cluster Yield Scala analysis engine, the JVM `PlanCaptureListener`, and the `PlanExtractor` — the analyzer is agnostic to capture method.

## Module structure

```
cluster_yield_snapshot/
├── __init__.py        # Public API: CYSnapshot, snapshot_capture
├── snapshot.py        # Orchestrator: start/stop/save/upload
├── _capture.py        # Passive capture engine (monkey-patching)
├── plans.py           # Plan extraction, operator parsing, fingerprinting
├── catalog.py         # Table stats (DESCRIBE DETAIL/EXTENDED/Catalyst)
├── config.py          # Spark config capture + drift detection
├── environment.py     # Platform detection (Databricks, YARN, K8s)
├── upload.py          # HTTP upload to SaaS backend
├── quick_scan.py      # Lightweight teaser findings
├── formatting.py      # Terminal summary + Databricks HTML
├── _compat.py         # Classic PySpark vs Spark Connect abstraction
└── _util.py           # Shared utilities
```

## Spark Connect / Serverless

On Spark Connect, the JVM is not accessible. Plan capture falls back to text explain. Catalog stats fall back to `DESCRIBE DETAIL` and `DESCRIBE EXTENDED` (no Catalyst stats). The text plan parser runs server-side for full analysis.
