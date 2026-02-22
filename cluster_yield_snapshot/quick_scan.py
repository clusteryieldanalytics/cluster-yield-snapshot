"""
Quick scan — lightweight plan analysis for immediate teaser findings.

Not a replacement for the full Scala analyzer. Just enough to show
value in the snapshot summary before the user uploads for full analysis.
"""

from __future__ import annotations

from typing import Any

from .plans import (
    operators_from_entry, has_metrics, has_plan_details,
    scan_bytes_from_entry, scan_rows_from_entry, scan_files_from_entry,
    shuffle_bytes_from_entry,
    scans_from_entry, joins_from_entry, exchanges_from_entry,
)
from ._util import fmt_bytes, parse_int


def quick_scan(
    plans: list[dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    """
    Scan captured plans, tables, and config for obvious issues.

    Returns a list of human-readable teaser strings.
    """
    teasers: list[str] = []
    opt = config.get("optimizerRelevant", {})
    threshold = parse_int(
        opt.get("spark.sql.autoBroadcastJoinThreshold", "10485760")
    )

    # Plan-level checks
    for p in plans:
        label = p.get("label", "?")
        operators = operators_from_entry(p)
        for op in operators:
            if op in ("CartesianProductExec", "CartesianProduct"):
                teasers.append(
                    f"Cartesian product in `{label}` — O(n×m) row explosion"
                )
            if op in ("BroadcastNestedLoopJoinExec", "BroadcastNestedLoopJoin"):
                teasers.append(
                    f"BroadcastNestedLoopJoin in `{label}` — "
                    f"quadratic join, check for missing equi-condition"
                )

        # Runtime metric teasers (classic PySpark post-execution only)
        if has_metrics(p):
            scan_b = scan_bytes_from_entry(p)
            scan_r = scan_rows_from_entry(p)
            scan_f = scan_files_from_entry(p)
            shuf_b = shuffle_bytes_from_entry(p)

            short_label = label[:60]
            if scan_b > 0:
                parts = [fmt_bytes(scan_b)]
                if scan_r > 0:
                    parts.append(f"{scan_r:,} rows")
                if scan_f > 0:
                    parts.append(f"{scan_f} files")
                teasers.append(
                    f"`{short_label}` scanned {', '.join(parts)}"
                )
            if shuf_b > 0:
                teasers.append(
                    f"`{short_label}` shuffled {fmt_bytes(shuf_b)}"
                )

        # Plan detail teasers (Spark Connect and classic — from text plan)
        if has_plan_details(p):
            _teasers_from_details(p, tables, teasers)

    # Broadcast disabled check
    if threshold is not None and threshold <= 0:
        teasers.append(
            "Broadcast joins disabled (autoBroadcastJoinThreshold = -1)"
        )

    # Small tables that could benefit from broadcast
    if threshold and threshold > 0:
        for tname, tstats in tables.items():
            size = tstats.get("sizeInBytes")
            if size and 0 < size < threshold * 5:
                teasers.append(
                    f"`{tname}` ({fmt_bytes(size)}) is small enough to "
                    f"broadcast — verify joins use BroadcastHashJoin"
                )

    # Default shuffle partitions on large data
    partitions = parse_int(opt.get("spark.sql.shuffle.partitions", "200"))
    if partitions == 200:
        large_tables = [
            (name, stats["sizeInBytes"])
            for name, stats in tables.items()
            if stats.get("sizeInBytes", 0) > 50 * 1024**3
        ]
        if large_tables:
            biggest = max(large_tables, key=lambda x: x[1])
            teasers.append(
                f"Default 200 shuffle partitions with `{biggest[0]}` "
                f"at {fmt_bytes(biggest[1])} — likely under-partitioned"
            )

    return teasers


def _teasers_from_details(
    plan_entry: dict[str, Any],
    tables: dict[str, dict[str, Any]],
    teasers: list[str],
) -> None:
    """Generate teasers from parsed plan detail metadata."""
    scans = scans_from_entry(plan_entry)
    joins = joins_from_entry(plan_entry)
    exchanges = exchanges_from_entry(plan_entry)

    for scan in scans:
        table = scan.get("table", "")
        short_table = table.rsplit(".", 1)[-1] if "." in table else table
        cols = scan.get("readColumnCount", 0)

        # Column pruning: compare read columns to catalog total
        cat = tables.get(table, {})
        total_cols = cat.get("columnCount")
        if cols and total_cols and cols < total_cols:
            teasers.append(
                f"`{short_table}` reads {cols} of {total_cols} columns"
            )

        # Missing partition filter on partitioned table
        if not scan.get("hasPartitionFilter"):
            part_cols = cat.get("partitionColumns", [])
            if part_cols:
                teasers.append(
                    f"`{short_table}` scanned without partition filter "
                    f"(partitioned by {', '.join(part_cols)})"
                )

    # Join summaries
    smj_count = sum(1 for j in joins if j.get("strategy") == "SortMergeJoin")
    if smj_count:
        teasers.append(
            f"{smj_count} SortMergeJoin{'s' if smj_count > 1 else ''} "
            f"— each requires 2 shuffles + 2 sorts"
        )

    # Shuffle partition count check
    for exch in exchanges:
        n = exch.get("numPartitions")
        if n is not None and n <= 1 and exch.get("exchangeType") == "shuffle":
            teasers.append(
                f"Shuffle to {n} partition — single-partition bottleneck"
            )

    # Photon status
    details = plan_entry.get("planDetails", {})
    if details.get("photonEnabled"):
        teasers.append("Photon enabled — all operators accelerated")