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
      - planDetails (structured metadata parsed from text plan detail sections)
      - metrics (list of per-node runtime metrics, if post-execution on classic PySpark)
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
        # Parse structured metadata from the numbered detail sections.
        # This is the primary cost-estimation data source on Spark Connect
        # (where JVM metrics are unavailable).
        details = parse_plan_details(plan_text)
        if details.get("nodes"):
            entry["planDetails"] = details

    if sql:
        entry["sql"] = sql

    # Runtime metrics — only populated post-execution on classic PySpark.
    # Returns None on Spark Connect (no JVM access).
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


# ── Detail section parsing ──────────────────────────────────────────────
#
# The EXPLAIN text has two parts:
#   A) The tree section (operators with +-, :- connectors)
#   B) The numbered detail sections: "(N) OperatorName\n  Key: value\n..."
#
# Section B is where cost-relevant metadata lives — filters, schemas,
# join keys, exchange types.  On Spark Connect (no JVM), this is the
# ONLY source of structured plan data.

_NODE_HEADER = re.compile(r"^\((\d+)\)\s+(.+)")
_OUTPUT_COLS = re.compile(r"^Output\s+\[(\d+)\]:\s+\[(.+)\]")
_READ_SCHEMA = re.compile(r"^ReadSchema:\s+struct<(.+)>")
_FILTER_LINE = re.compile(
    r"^(PartitionFilters|DataFilters|RequiredDataFilters|"
    r"DictionaryFilters|OptionalDataFilters|PushedFilters):\s+\[(.+)\]"
)
_JOIN_TYPE = re.compile(r"^Join type:\s+(\w+)")
_JOIN_KEYS = re.compile(r"^(Left|Right) keys\s+\[\d+\]:\s+\[(.+)\]")
_JOIN_COND = re.compile(r"^Join condition:\s+(.+)")
_ARGUMENTS = re.compile(r"^Arguments:\s+(.+)")
_LOCATION = re.compile(r"^Location:\s+(\w+)\s+\[(.+)\]")


def parse_plan_details(explain_text: str) -> dict[str, Any]:
    """
    Parse the numbered detail sections from EXPLAIN output into structured data.

    Returns a dict with:
      - ``nodes``: dict mapping node ID → per-node metadata
      - ``scans``: list of scan summaries (table, format, column count, filters)
      - ``joins``: list of join summaries (strategy, type, keys)
      - ``exchanges``: list of exchange summaries (type, partitioning)
      - ``photonEnabled``: bool — all operators use Photon
      - ``aqeFinalPlan``: bool — whether this is the final AQE plan

    Works on both OSS Spark and Databricks Photon plan text.
    Returns ``{"nodes": {}}`` if no detail sections are found.
    """
    # Find where the detail sections start
    detail_text = _extract_detail_section(explain_text)
    if not detail_text:
        return {"nodes": {}}

    # Parse into per-node blocks
    nodes: dict[str, dict[str, Any]] = {}
    current_id: Optional[str] = None
    current_lines: list[str] = []

    for line in detail_text.split("\n"):
        header = _NODE_HEADER.match(line)
        if header:
            # Flush previous node
            if current_id is not None:
                nodes[current_id] = _parse_node_block(
                    current_id, current_lines
                )
            current_id = header.group(1)
            current_lines = [line]
        elif current_id is not None:
            current_lines.append(line)

    # Flush last node
    if current_id is not None:
        nodes[current_id] = _parse_node_block(current_id, current_lines)

    # Build typed summary lists
    scans: list[dict[str, Any]] = []
    joins: list[dict[str, Any]] = []
    exchanges: list[dict[str, Any]] = []
    photon_count = 0

    for nid, node in nodes.items():
        op = node.get("operator", "")
        if "Photon" in op:
            photon_count += 1

        if "Scan" in op:
            scan: dict[str, Any] = {"nodeId": nid, "operator": op}
            for key in ("table", "format", "readColumnCount", "outputCount",
                        "hasPartitionFilter", "hasDataFilter",
                        "partitionFilters", "dataFilters",
                        "dictionaryFilters", "readSchema"):
                if key in node:
                    scan[key] = node[key]
            scans.append(scan)
        elif "Join" in op:
            join: dict[str, Any] = {"nodeId": nid, "operator": op}
            # Infer strategy from operator name
            if "BroadcastHash" in op:
                join["strategy"] = "BroadcastHashJoin"
            elif "SortMerge" in op:
                join["strategy"] = "SortMergeJoin"
            elif "ShuffledHash" in op:
                join["strategy"] = "ShuffledHashJoin"
            elif "NestedLoop" in op or "Cartesian" in op:
                join["strategy"] = "NestedLoopJoin"
            for key in ("joinType", "leftKeys", "rightKeys", "joinCondition"):
                if key in node:
                    join[key] = node[key]
            joins.append(join)
        elif "Exchange" in op or "Shuffle" in op:
            exch: dict[str, Any] = {"nodeId": nid, "operator": op}
            for key in ("exchangeType", "partitioning", "partitionKeys",
                        "numPartitions"):
                if key in node:
                    exch[key] = node[key]
            exchanges.append(exch)

    # Detect AQE status from tree section
    aqe_final = False
    if "isFinalPlan=true" in explain_text:
        aqe_final = True

    # Detect Photon
    photon_enabled = photon_count > 0 and all(
        "Photon" in n.get("operator", "")
        or n.get("operator", "") in ("AdaptiveSparkPlan", "ColumnarToRow")
        for n in nodes.values()
    )

    return {
        "nodes": nodes,
        "scans": scans,
        "joins": joins,
        "exchanges": exchanges,
        "photonEnabled": photon_enabled,
        "aqeFinalPlan": aqe_final,
    }


def _extract_detail_section(explain_text: str) -> str:
    """
    Extract the numbered detail section from EXPLAIN text.

    This is everything after the tree section, starting from the first
    line matching ``(N) OperatorName``.  Stops at section headers like
    ``== Photon Explanation ==``.
    """
    lines = explain_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if _NODE_HEADER.match(line):
            start = i
            break
    if start is None:
        return ""

    detail_lines: list[str] = []
    for line in lines[start:]:
        if line.startswith("== ") and line.endswith(" =="):
            break
        detail_lines.append(line)
    return "\n".join(detail_lines)


def _parse_node_block(
    node_id: str, lines: list[str]
) -> dict[str, Any]:
    """Parse a single numbered node block into structured metadata."""
    node: dict[str, Any] = {}

    # First line is the header: "(N) OperatorName rest..."
    header = _NODE_HEADER.match(lines[0])
    if header:
        full_op = header.group(2).strip()
        # Parse operator name and optional format/table from scans
        # e.g. "PhotonScan parquet workspace.test_cy.orders"
        # e.g. "FileScan parquet default.orders[col1, col2]"
        parts = full_op.split()
        node["operator"] = parts[0] if parts else full_op
        if len(parts) >= 3 and "Scan" in parts[0]:
            node["format"] = parts[1]
            # Table name — strip trailing [...] if present
            table = parts[2].split("[")[0].strip("`")
            node["table"] = table
        elif len(parts) >= 2 and "Scan" in parts[0]:
            # Could be format or table
            second = parts[1]
            if second in ("parquet", "orc", "delta", "csv", "json", "avro"):
                node["format"] = second
            else:
                node["table"] = second.split("[")[0].strip("`")
        # Join type from header: "PhotonBroadcastHashJoin Inner"
        if "Join" in (parts[0] if parts else ""):
            for p in parts[1:]:
                if p in ("Inner", "LeftOuter", "RightOuter", "FullOuter",
                         "LeftSemi", "LeftAnti", "Cross"):
                    node["joinType"] = p

    # Parse attribute lines
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Output columns
        m = _OUTPUT_COLS.match(stripped)
        if m:
            node["outputCount"] = int(m.group(1))
            cols = _parse_column_refs(m.group(2))
            if cols:
                node["output"] = cols
            continue

        # ReadSchema
        m = _READ_SCHEMA.match(stripped)
        if m:
            schema = _parse_struct_schema(m.group(1))
            if schema:
                node["readSchema"] = schema
                node["readColumnCount"] = len(schema)
            continue

        # Filter lines (partition, data, dictionary, optional, pushed)
        m = _FILTER_LINE.match(stripped)
        if m:
            filter_type = m.group(1)
            filters = _parse_filter_list(m.group(2))
            key_map = {
                "PartitionFilters": "partitionFilters",
                "DataFilters": "dataFilters",
                "RequiredDataFilters": "dataFilters",
                "DictionaryFilters": "dictionaryFilters",
                "OptionalDataFilters": "optionalDataFilters",
                "PushedFilters": "pushedFilters",
            }
            key = key_map.get(filter_type, filter_type)
            existing = node.get(key, [])
            # Merge RequiredDataFilters into dataFilters
            node[key] = existing + filters if existing else filters
            if filter_type == "PartitionFilters" and filters:
                node["hasPartitionFilter"] = True
            if filter_type in ("DataFilters", "RequiredDataFilters",
                               "DictionaryFilters") and filters:
                node["hasDataFilter"] = True
            continue

        # Join type (from detail line)
        m = _JOIN_TYPE.match(stripped)
        if m:
            node["joinType"] = m.group(1)
            continue

        # Join keys
        m = _JOIN_KEYS.match(stripped)
        if m:
            side = m.group(1).lower() + "Keys"
            node[side] = _parse_column_refs(m.group(2))
            continue

        # Join condition
        m = _JOIN_COND.match(stripped)
        if m:
            cond = m.group(1).strip()
            node["joinCondition"] = None if cond == "None" else cond
            continue

        # Arguments (exchanges and aggregations)
        m = _ARGUMENTS.match(stripped)
        if m:
            args = m.group(1).strip()
            _parse_arguments(args, node)
            continue

        # Location
        m = _LOCATION.match(stripped)
        if m:
            node["locationIndex"] = m.group(1)
            node["locationPath"] = m.group(2)
            continue

    # Ensure boolean flags default to False for scans
    if "Scan" in node.get("operator", ""):
        node.setdefault("hasPartitionFilter", False)
        node.setdefault("hasDataFilter", False)

    return node


def _parse_column_refs(col_str: str) -> list[str]:
    """Parse column references like 'user_id#123L, amount#456' → ['user_id', 'amount']."""
    cols: list[str] = []
    for part in col_str.split(","):
        part = part.strip()
        if not part:
            continue
        # Strip Spark's internal ref suffix: #123, #123L, etc.
        name = re.split(r"#\d+", part)[0].strip()
        if name:
            cols.append(name)
    return cols


def _parse_struct_schema(schema_str: str) -> dict[str, str]:
    """Parse 'user_id:bigint,amount:decimal(10,2)' → {'user_id': 'bigint', ...}."""
    schema: dict[str, str] = {}
    # Split carefully — types like decimal(10,2) contain commas
    depth = 0
    current = ""
    for ch in schema_str:
        if ch in ("(", "<"):
            depth += 1
            current += ch
        elif ch in (")", ">"):
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            _add_schema_field(current.strip(), schema)
            current = ""
        else:
            current += ch
    if current.strip():
        _add_schema_field(current.strip(), schema)
    return schema


def _add_schema_field(field: str, schema: dict[str, str]) -> None:
    """Parse 'name:type' and add to schema dict."""
    parts = field.split(":", 1)
    if len(parts) == 2:
        schema[parts[0].strip()] = parts[1].strip()


def _parse_filter_list(filter_str: str) -> list[str]:
    """Parse '[expr1, expr2, ...]' contents into individual expressions."""
    # Filters are comma-separated but expressions can contain commas
    # inside parentheses.  Split at top-level commas only.
    filters: list[str] = []
    depth = 0
    current = ""
    for ch in filter_str:
        if ch in ("(", "["):
            depth += 1
            current += ch
        elif ch in (")", "]"):
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            expr = current.strip()
            if expr:
                filters.append(expr)
            current = ""
        else:
            current += ch
    expr = current.strip()
    if expr:
        filters.append(expr)
    return filters


def _parse_arguments(args: str, node: dict[str, Any]) -> None:
    """
    Parse an Arguments line into node metadata.

    Handles patterns like:
      - "hashpartitioning(category#123, country#456, 16)"
      - "SinglePartition"
      - "EXECUTOR_BROADCAST, [id=#1234]"
      - "rangepartitioning(amount#789 ASC, 200)"
    """
    if args.startswith("hashpartitioning("):
        node["exchangeType"] = "shuffle"
        node["partitioning"] = "hashpartitioning"
        inner = args[len("hashpartitioning("):-1] if args.endswith(")") else args
        parts = [p.strip() for p in inner.split(",")]
        # Last element is numPartitions (an integer)
        keys: list[str] = []
        for p in parts:
            # Is it a number (partition count)?
            stripped = p.strip()
            if stripped.isdigit():
                node["numPartitions"] = int(stripped)
            else:
                name = re.split(r"#\d+", stripped)[0].strip()
                if name:
                    keys.append(name)
        if keys:
            node["partitionKeys"] = keys
    elif args.startswith("rangepartitioning("):
        node["exchangeType"] = "shuffle"
        node["partitioning"] = "rangepartitioning"
        inner = args[len("rangepartitioning("):-1] if args.endswith(")") else args
        parts = [p.strip() for p in inner.split(",")]
        for p in parts:
            if p.strip().isdigit():
                node["numPartitions"] = int(p.strip())
    elif "SinglePartition" in args:
        node["exchangeType"] = "broadcast"
        node["partitioning"] = "SinglePartition"
    elif "EXECUTOR_BROADCAST" in args or "BROADCAST" in args:
        node["exchangeType"] = "broadcast"
    elif "roundrobinpartitioning" in args.lower():
        node["exchangeType"] = "shuffle"
        node["partitioning"] = "roundrobinpartitioning"


# ── Plan detail accessors ───────────────────────────────────────────────
#
# Convenience functions for the server-side cost estimator and quick_scan.


def scans_from_entry(plan_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return scan details from planDetails."""
    return plan_entry.get("planDetails", {}).get("scans", [])


def joins_from_entry(plan_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return join details from planDetails."""
    return plan_entry.get("planDetails", {}).get("joins", [])


def exchanges_from_entry(plan_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exchange details from planDetails."""
    return plan_entry.get("planDetails", {}).get("exchanges", [])


def has_plan_details(plan_entry: dict[str, Any]) -> bool:
    """Return True if this plan entry has parsed detail metadata."""
    details = plan_entry.get("planDetails", {})
    return bool(details.get("nodes"))


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