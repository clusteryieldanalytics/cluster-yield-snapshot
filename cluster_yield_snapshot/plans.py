"""
Plan capture, operator extraction, and fingerprinting.

Captures Spark physical plans from DataFrames via two strategies:
  1. JSON via JVM (classic PySpark) — full fidelity for the Scala analyzer
  2. Text plan via explain (Spark Connect / serverless) — parsed into operators

Both produce enough data for the quick scan. Full analyzer requires JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Optional

from . import _compat

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

# ── Operator name patterns ────────────────────────────────────────────────

# Tree-drawing prefix: any combo of spaces, |, :, +, -, !, *
_TREE_PREFIX = re.compile(r"^[\s|:+\-!*]+")
# Codegen stage prefix: *(2), (3), etc.
_CODEGEN_PREFIX = re.compile(r"^\*?\(\d+\)\s*")
# Operator name: starts with uppercase, word chars
_OPERATOR_NAME = re.compile(r"([A-Z]\w*)")

# Scan node class names that reference tables
SCAN_CLASSES = frozenset({
    "FileSourceScanExec", "FileScan", "BatchScanExec",
    "HiveTableScanExec", "InMemoryTableScanExec",
    # Databricks Photon
    "PhotonFileSourceScanExec", "PhotonBatchScanExec",
    "PhotonScan",
})

# Regex for table names in text plans
_SCAN_PATTERN = re.compile(
    r"(?:FileScan|BatchScan|Scan|HiveTableScan|PhotonScan)\s+"
    r"(?:\w+\s+)?"           # optional format (parquet, orc, delta)
    r"([\w.`]+?)(?:\[|\s|$)"  # table name (possibly qualified)
)
_INMEM_PATTERN = re.compile(r"InMemoryTableScan\s+\[.*?\],\s+([\w.]+)")


def capture_plan(df: DataFrame, label: str, sql: Optional[str] = None) -> dict[str, Any]:
    """
    Capture the physical plan from a DataFrame.

    Returns a plan entry dict with:
      - label, fingerprint, nodeCount
      - plan (JSON array) or planText (text explain)
      - planFormat ("json" or "text")
      - metrics (list of per-node runtime metrics, if post-execution)
      - sql (if provided)

    Raises nothing — returns a partial entry on failure.
    """
    plan_json = None
    plan_text = None

    # Strategy 1: JSON via JVM (classic PySpark)
    raw = _compat.get_plan_json(df)
    if raw is not None:
        try:
            plan_json = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 2: Text plan (works everywhere)
    plan_text = _compat.get_plan_text(df)

    # Extract operators from whichever source we got
    if plan_json is not None:
        operators = operators_from_json(plan_json)
        node_count = len(plan_json) if isinstance(plan_json, list) else 0
    elif plan_text:
        operators = operators_from_text(plan_text)
        node_count = len(operators)
    else:
        operators = []
        node_count = 0

    entry: dict[str, Any] = {
        "label": label,
        "fingerprint": fingerprint_operators(operators),
        "nodeCount": node_count,
    }

    if plan_json is not None:
        entry["plan"] = plan_json
        entry["planFormat"] = "json"
    elif plan_text:
        entry["planFormat"] = "text"

    if plan_text:
        entry["planText"] = plan_text

    if sql:
        entry["sql"] = sql

    # Runtime metrics — only populated post-execution on classic PySpark.
    # Pre-execution captures (trigger="spark.sql") will get None here,
    # which is correct — the job hasn't run yet. The dedup logic in
    # CYSnapshot._on_plan_captured replaces pre-execution entries with
    # post-execution ones, so metrics arrive with the final capture.
    metrics = _compat.get_plan_metrics(df)
    if metrics:
        entry["metrics"] = metrics

    return entry


# ── Operator extraction ──────────────────────────────────────────────────

def operators_from_json(plan_data: list | Any) -> list[str]:
    """Extract operator simple class names from a JSON plan array."""
    if not isinstance(plan_data, list):
        return []
    operators = []
    for node in plan_data:
        if isinstance(node, dict):
            cls = node.get("class", "")
            simple = cls.rsplit(".", 1)[-1] if "." in cls else cls
            if simple:
                operators.append(simple)
    return operators


def operators_from_text(plan_text: str) -> list[str]:
    """
    Extract operator names from a text explain plan.

    Parses lines like:
        +- SortMergeJoin [key#1L], [key#2L], Inner
        :  +- Exchange hashpartitioning(key#1L, 200)
           +- FileScan parquet default.orders[...]
    """
    operators = []
    physical = extract_physical_plan(plan_text)
    for line in physical.split("\n"):
        op = parse_operator_line(line)
        if op:
            operators.append(op)
    return operators


def operators_from_entry(plan_entry: dict[str, Any]) -> list[str]:
    """Extract operators from a plan entry (handles both JSON and text)."""
    plan_json = plan_entry.get("plan")
    if plan_json and isinstance(plan_json, list):
        return operators_from_json(plan_json)
    plan_text = plan_entry.get("planText", "")
    if plan_text:
        return operators_from_text(plan_text)
    return []


def extract_physical_plan(explain_text: str) -> str:
    """
    Extract the Physical Plan tree section from EXPLAIN EXTENDED output.

    EXPLAIN EXTENDED has multiple sections (Parsed, Analyzed, Optimized,
    Physical). We want only the tree portion of Physical. If section
    headers are absent, the entire text is treated as the physical plan.

    Stops at the numbered detail section (lines like "(1) FileScan ...")
    which Spark appends after the tree. These detail lines start with
    a parenthesized number and would otherwise be misidentified as
    operators by parse_operator_line.
    """
    marker = "== Physical Plan =="
    idx = explain_text.find(marker)
    if idx == -1:
        return explain_text

    after = explain_text[idx + len(marker):]
    next_section = after.find("\n== ")
    if next_section != -1:
        after = after[:next_section]

    # Stop at the numbered detail section boundary.
    # Detail nodes start with "(N) OperatorName" at the start of a line.
    lines = after.split("\n")
    tree_lines: list[str] = []
    for line in lines:
        if re.match(r"^\(\d+\)\s+\w+", line):
            break
        tree_lines.append(line)

    return "\n".join(tree_lines)


def parse_operator_line(line: str) -> Optional[str]:
    """
    Extract the operator name from a single plan tree line.

    Handles tree chars (+-, :-, *, etc.) and codegen prefixes (*(2)).
    Examples:
        "+- SortMergeJoin [k1], Inner"    → "SortMergeJoin"
        ":  +- FileScan parquet db.t[…]"  → "FileScan"
        "*(2) HashAggregate(keys=[…])"    → "HashAggregate"
    """
    stripped = _TREE_PREFIX.sub("", line)
    stripped = _CODEGEN_PREFIX.sub("", stripped)
    if not stripped:
        return None
    match = _OPERATOR_NAME.match(stripped)
    return match.group(1) if match else None


# ── Fingerprinting ───────────────────────────────────────────────────────

def fingerprint_operators(operators: list[str]) -> str:
    """SHA-256 fingerprint (first 16 hex chars) from operator names."""
    if not operators:
        return "empty"
    return hashlib.sha256(",".join(operators).encode()).hexdigest()[:16]


# ── Table extraction from plans ──────────────────────────────────────────

def tables_from_json(plan_data: list | Any) -> set[str]:
    """Pull table names from scan nodes in a JSON plan array."""
    tables: set[str] = set()
    if not isinstance(plan_data, list):
        return tables
    for node in plan_data:
        if not isinstance(node, dict):
            continue
        cls = node.get("class", "")
        simple = cls.rsplit(".", 1)[-1] if "." in cls else cls
        if simple in SCAN_CLASSES:
            for key in ("tableName", "relation"):
                val = node.get(key)
                if val and isinstance(val, str) and val not in ("", "None"):
                    tables.add(val)
            loc = node.get("location")
            if loc and isinstance(loc, str):
                name = loc.rstrip("/").rsplit("/", 1)[-1]
                if name:
                    tables.add(name)
    return tables


def tables_from_text(plan_text: str) -> set[str]:
    """Pull table names from a text explain plan."""
    tables: set[str] = set()
    physical = extract_physical_plan(plan_text)

    for match in _SCAN_PATTERN.finditer(physical):
        table = match.group(1).strip("`")
        if table and not table.startswith("("):
            parts = table.split(".")
            if len(parts) >= 3 and parts[0] in ("spark_catalog", "hive_metastore"):
                table = ".".join(parts[1:])
            tables.add(table)

    for match in _INMEM_PATTERN.finditer(physical):
        tables.add(match.group(1))

    return tables


def tables_from_entry(plan_entry: dict[str, Any]) -> set[str]:
    """Extract table names from a plan entry (handles both formats)."""
    plan_json = plan_entry.get("plan")
    if plan_json and isinstance(plan_json, list):
        return tables_from_json(plan_json)
    plan_text = plan_entry.get("planText", "")
    if plan_text:
        return tables_from_text(plan_text)
    return set()


# ── Runtime metric accessors ─────────────────────────────────────────────
#
# These pull aggregated cost-relevant numbers from the per-node metrics
# captured post-execution. Used by quick_scan for teasers and by the
# server-side cost estimator (roadmap #15).


def _sum_metric(plan_entry: dict[str, Any], class_filter: str,
                metric_name: str) -> int:
    """Sum a named metric across all nodes whose simpleClassName contains class_filter."""
    total = 0
    for node in plan_entry.get("metrics", []):
        if class_filter in node.get("simpleClassName", ""):
            total += node.get("metrics", {}).get(metric_name, 0)
    return total


def scan_bytes_from_entry(plan_entry: dict[str, Any]) -> int:
    """Total bytes scanned across all scan nodes (from runtime metrics)."""
    return _sum_metric(plan_entry, "Scan", "size of files read")


def scan_rows_from_entry(plan_entry: dict[str, Any]) -> int:
    """Total rows output from all scan nodes (from runtime metrics)."""
    return _sum_metric(plan_entry, "Scan", "number of output rows")


def scan_files_from_entry(plan_entry: dict[str, Any]) -> int:
    """Total files read across all scan nodes (from runtime metrics)."""
    return _sum_metric(plan_entry, "Scan", "number of files read")


def shuffle_bytes_from_entry(plan_entry: dict[str, Any]) -> int:
    """Total bytes shuffled across all exchange nodes (from runtime metrics)."""
    # Match ShuffleExchange, PhotonShuffleExchangeSink, etc. but not Broadcast
    total = 0
    for node in plan_entry.get("metrics", []):
        cn = node.get("simpleClassName", "")
        if ("Shuffle" in cn or "Exchange" in cn) and "Broadcast" not in cn:
            total += node.get("metrics", {}).get("data size", 0)
    return total


def broadcast_bytes_from_entry(plan_entry: dict[str, Any]) -> int:
    """Total bytes broadcast across all broadcast exchange nodes."""
    total = 0
    for node in plan_entry.get("metrics", []):
        cn = node.get("simpleClassName", "")
        if "Broadcast" in cn and "Exchange" in cn:
            total += node.get("metrics", {}).get("data size", 0)
    return total


def has_metrics(plan_entry: dict[str, Any]) -> bool:
    """Return True if this plan entry has runtime metrics."""
    metrics = plan_entry.get("metrics")
    return bool(metrics and any(n.get("metrics") for n in metrics))


def metrics_summary(plan_entry: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a flat summary of runtime metrics for a plan entry.

    Returns a dict with top-level cost-relevant aggregates that the
    server-side cost estimator can consume directly.  All byte values
    are raw ints; formatting is the caller's responsibility.
    """
    if not has_metrics(plan_entry):
        return {}
    return {
        "scanBytes": scan_bytes_from_entry(plan_entry),
        "scanRows": scan_rows_from_entry(plan_entry),
        "scanFiles": scan_files_from_entry(plan_entry),
        "shuffleBytes": shuffle_bytes_from_entry(plan_entry),
        "broadcastBytes": broadcast_bytes_from_entry(plan_entry),
    }