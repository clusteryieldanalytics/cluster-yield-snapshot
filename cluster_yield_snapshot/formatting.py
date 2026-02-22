"""
Output formatting — terminal summary and inline HTML for Databricks.
"""

from __future__ import annotations

from typing import Any

from ._util import fmt_bytes, esc_html, parse_int
from .plans import (
    operators_from_entry, has_metrics, has_plan_details,
    scan_bytes_from_entry, shuffle_bytes_from_entry,
    scans_from_entry, joins_from_entry, exchanges_from_entry,
)


def print_summary(snapshot: dict[str, Any], teasers: list[str] | None = None) -> None:
    """Print a human-readable terminal summary of the snapshot."""
    plans = snapshot.get("plans", [])
    tables = snapshot.get("catalog", {}).get("tables", {})
    config = snapshot.get("config", {})
    env = snapshot.get("environment", {})
    errors = snapshot.get("errors") or []
    non_default = config.get("nonDefault", {})

    total_nodes = sum(p.get("nodeCount", 0) for p in plans)
    total_size = sum(t.get("sizeInBytes", 0) for t in tables.values())

    # Aggregate runtime I/O metrics across all plans
    total_scan = sum(scan_bytes_from_entry(p) for p in plans)
    total_shuffle = sum(shuffle_bytes_from_entry(p) for p in plans)
    plans_with_metrics = sum(1 for p in plans if has_metrics(p))

    # Partition enrichment stats
    tables_with_partitions = sum(
        1 for t in tables.values() if t.get("partitionCount")
    )
    tables_with_rows = sum(1 for t in tables.values() if t.get("rowCount"))

    print("\n" + "═" * 60)
    print("  CLUSTER YIELD SNAPSHOT SUMMARY")
    print("═" * 60)

    compute = env.get("computeType", "")
    platform_str = env.get("platform", "?")
    if compute:
        platform_str += f" ({compute})"
    print(f"  Spark {env.get('sparkVersion', '?')} on {platform_str}")

    print(f"  Plans captured:     {len(plans)}")
    print(f"  Total plan nodes:   {total_nodes}")
    print(f"  Tables cataloged:   {len(tables)}")
    if total_size > 0:
        print(f"  Total data volume:  {fmt_bytes(total_size)}")
    if tables_with_rows:
        print(f"  Row counts:         {tables_with_rows}/{len(tables)} tables")
    if tables_with_partitions:
        print(f"  Partition metadata:  {tables_with_partitions}/{len(tables)} tables")

    if plans_with_metrics > 0:
        print(f"\n  ── Runtime I/O ({plans_with_metrics} plan"
              f"{'s' if plans_with_metrics != 1 else ''} with metrics) ──")
        if total_scan > 0:
            print(f"  Total scanned:      {fmt_bytes(total_scan)}")
        if total_shuffle > 0:
            print(f"  Total shuffled:     {fmt_bytes(total_shuffle)}")

    if non_default:
        print(f"\n  Non-default configs: {len(non_default)}")
        for key, info in non_default.items():
            short_key = key.replace("spark.sql.", "")
            print(f"    {short_key}: {info['value']} "
                  f"(default: {info['sparkDefault']})")
    if errors:
        print(f"  Warnings:           {len(errors)}")

    if teasers:
        print(f"\n  ⚡ Quick scan ({len(teasers)} potential issue"
              f"{'s' if len(teasers) != 1 else ''}):")
        for t in teasers[:5]:
            print(f"    • {t}")
        if len(teasers) > 5:
            print(f"    ... and {len(teasers) - 5} more")

    print("\n  Run full analysis for detailed findings and cost estimates.")
    print("═" * 60)


def render_html(snapshot: dict[str, Any], teasers: list[str] | None = None) -> str:
    """Render an HTML summary card for inline Databricks display."""
    env = snapshot.get("environment", {})
    config = snapshot.get("config", {})
    plans = snapshot.get("plans", [])
    tables = snapshot.get("catalog", {}).get("tables", {})
    errors = snapshot.get("errors") or []
    non_default = config.get("nonDefault", {})

    total_nodes = sum(p.get("nodeCount", 0) for p in plans)
    total_size = sum(t.get("sizeInBytes", 0) for t in tables.values())

    # Aggregate runtime I/O metrics
    total_scan = sum(scan_bytes_from_entry(p) for p in plans)
    total_shuffle = sum(shuffle_bytes_from_entry(p) for p in plans)
    plans_with_metrics = sum(1 for p in plans if has_metrics(p))
    plans_with_details = sum(1 for p in plans if has_plan_details(p))

    # Table rows
    table_rows = ""
    for tname, tstats in sorted(
        tables.items(), key=lambda x: x[1].get("sizeInBytes", 0), reverse=True,
    ):
        size = tstats.get("sizeInBytes", 0)
        rows = tstats.get("rowCount")
        pcols = tstats.get("partitionColumns", [])
        pcount = tstats.get("partitionCount")
        files = tstats.get("fileCount")
        cols = tstats.get("columnCount")

        row_str = f"{rows:,}" if rows else "—"
        pcol_str = ", ".join(pcols) if pcols else "—"
        if pcount is not None:
            pcol_str += f" ({pcount:,})"
        file_str = str(files) if files else "—"
        col_str = str(cols) if cols else "—"

        table_rows += (
            f'<tr><td style="font-family:monospace;font-size:13px">{esc_html(tname)}</td>'
            f'<td style="text-align:right">{fmt_bytes(size)}</td>'
            f'<td style="text-align:right">{row_str}</td>'
            f'<td style="text-align:right">{col_str}</td>'
            f'<td>{pcol_str}</td>'
            f'<td style="text-align:right">{file_str}</td></tr>'
        )

    # Plan rows
    plan_rows = ""
    for p in plans:
        label = p.get("label", "?")
        nodes = p.get("nodeCount", 0)
        fp = p.get("fingerprint", "?")[:12]
        sql = p.get("sql", "")
        sql_preview = (sql[:80] + "...") if len(sql) > 80 else sql

        # Per-plan summary: scans/joins/shuffles from planDetails
        detail_parts: list[str] = []
        if has_plan_details(p):
            scans = scans_from_entry(p)
            joins_list = joins_from_entry(p)
            exchs = exchanges_from_entry(p)
            if scans:
                detail_parts.append(f"{len(scans)} scan{'s' if len(scans) != 1 else ''}")
            if joins_list:
                strategies = set(j.get("strategy", "?") for j in joins_list)
                detail_parts.append(
                    f"{len(joins_list)} join{'s' if len(joins_list) != 1 else ''} "
                    f"({', '.join(s.replace('Join', '') for s in strategies)})"
                )
            shuffle_count = sum(1 for e in exchs if e.get("exchangeType") == "shuffle")
            if shuffle_count:
                detail_parts.append(f"{shuffle_count} shuffle{'s' if shuffle_count != 1 else ''}")

        detail_str = esc_html(", ".join(detail_parts)) if detail_parts else "—"

        plan_rows += (
            f'<tr><td style="font-family:monospace;font-size:13px">{esc_html(label)}</td>'
            f'<td style="text-align:right">{nodes}</td>'
            f'<td style="font-size:12px">{detail_str}</td>'
            f'<td style="font-family:monospace;font-size:11px;color:#666">{fp}</td>'
            f'<td style="font-size:12px;color:#555">{esc_html(sql_preview)}</td></tr>'
        )

    # Scan details section (from planDetails across all plans)
    scan_rows = ""
    if plans_with_details > 0:
        for p in plans:
            for scan in scans_from_entry(p):
                tbl = scan.get("table", "?")
                short_tbl = tbl.rsplit(".", 1)[-1] if "." in tbl else tbl
                fmt = scan.get("format", "?")
                cols = scan.get("readColumnCount", "?")
                pf = "✓" if scan.get("hasPartitionFilter") else "✗"
                df_ = "✓" if scan.get("hasDataFilter") else "✗"
                scan_rows += (
                    f'<tr><td style="font-family:monospace;font-size:13px">{esc_html(short_tbl)}</td>'
                    f'<td>{esc_html(fmt)}</td>'
                    f'<td style="text-align:right">{cols}</td>'
                    f'<td style="text-align:center">{pf}</td>'
                    f'<td style="text-align:center">{df_}</td></tr>'
                )

    # Config drift
    config_rows = ""
    for key, info in non_default.items():
        short_key = key.replace("spark.sql.", "")
        config_rows += (
            f'<tr><td style="font-family:monospace;font-size:13px">{esc_html(short_key)}</td>'
            f'<td style="font-family:monospace;color:#c0392b">{esc_html(str(info["value"]))}</td>'
            f'<td style="font-family:monospace;color:#999">{esc_html(info["sparkDefault"])}</td></tr>'
        )

    # Teasers
    teaser_html = ""
    if teasers:
        items = "".join(f"<li>{t}</li>" for t in teasers[:8])
        teaser_html = (
            '<div style="background:#fff8e1;border-left:4px solid #f9a825;'
            'padding:12px 16px;margin:16px 0;border-radius:4px">'
            f'<strong>⚡ Quick Scan</strong>'
            f'<ul style="margin:8px 0 0 0;padding-left:20px">{items}</ul></div>'
        )

    config_section = ""
    if config_rows:
        config_section = (
            '<h3 style="margin:20px 0 8px">Config Drift</h3>'
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f5f5f5">'
            '<th style="text-align:left;padding:6px 12px">Setting</th>'
            '<th style="text-align:left;padding:6px 12px">Current</th>'
            '<th style="text-align:left;padding:6px 12px">Spark Default</th>'
            f'</tr>{config_rows}</table>'
        )

    compute = env.get("computeType", "")
    compute_str = f" · {esc_html(compute)}" if compute else ""

    # Detail stats cards (scans/joins from planDetails)
    detail_cards = ""
    if plans_with_details > 0:
        total_scans = sum(len(scans_from_entry(p)) for p in plans)
        total_joins = sum(len(joins_from_entry(p)) for p in plans)
        total_shuffles = sum(
            1 for p in plans for e in exchanges_from_entry(p)
            if e.get("exchangeType") == "shuffle"
        )
        detail_cards = (
            f'<div><span style="font-size:28px;font-weight:700">{total_scans}</span>'
            f'<br><span style="font-size:12px;color:#666">Scans</span></div>'
            f'<div><span style="font-size:28px;font-weight:700">{total_joins}</span>'
            f'<br><span style="font-size:12px;color:#666">Joins</span></div>'
            f'<div><span style="font-size:28px;font-weight:700">{total_shuffles}</span>'
            f'<br><span style="font-size:12px;color:#666">Shuffles</span></div>'
        )

    # Runtime I/O stats cards (only when JVM metrics are present — classic PySpark)
    io_cards = ""
    if plans_with_metrics > 0:
        io_cards = (
            f'<div><span style="font-size:28px;font-weight:700">{fmt_bytes(total_scan)}</span>'
            f'<br><span style="font-size:12px;color:#666">Scanned</span></div>'
            f'<div><span style="font-size:28px;font-weight:700">{fmt_bytes(total_shuffle)}</span>'
            f'<br><span style="font-size:12px;color:#666">Shuffled</span></div>'
        )

    # Scan details section
    scan_section = ""
    if scan_rows:
        scan_section = (
            '<h3 style="margin:20px 0 8px">Scan Details</h3>'
            '<table style="border-collapse:collapse;width:100%">'
            '<tr style="background:#f5f5f5">'
            '<th style="text-align:left;padding:6px 12px">Table</th>'
            '<th style="text-align:left;padding:6px 12px">Format</th>'
            '<th style="text-align:right;padding:6px 12px">Columns Read</th>'
            '<th style="text-align:center;padding:6px 12px">Part. Filter</th>'
            '<th style="text-align:center;padding:6px 12px">Data Filter</th>'
            f'</tr>{scan_rows}</table>'
        )

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;
                max-width:900px;margin:20px auto;color:#333">
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                    color:white;padding:20px 24px;border-radius:8px 8px 0 0">
            <h2 style="margin:0 0 4px;font-size:20px">⚡ Cluster Yield Snapshot</h2>
            <div style="font-size:14px;opacity:0.8">
                Spark {esc_html(env.get('sparkVersion', '?'))} ·
                {esc_html(env.get('platform', '?'))}{compute_str}
            </div>
        </div>
        <div style="background:#f8f9fa;padding:16px 24px;border:1px solid #dee2e6;border-top:0">
            <div style="display:flex;gap:32px;flex-wrap:wrap">
                <div><span style="font-size:28px;font-weight:700">{len(plans)}</span>
                     <br><span style="font-size:12px;color:#666">Plans</span></div>
                <div><span style="font-size:28px;font-weight:700">{total_nodes}</span>
                     <br><span style="font-size:12px;color:#666">Operators</span></div>
                <div><span style="font-size:28px;font-weight:700">{len(tables)}</span>
                     <br><span style="font-size:12px;color:#666">Tables</span></div>
                <div><span style="font-size:28px;font-weight:700">{fmt_bytes(total_size)}</span>
                     <br><span style="font-size:12px;color:#666">Data Volume</span></div>
                {detail_cards}
                {io_cards}
            </div>
        </div>
        <div style="background:white;padding:16px 24px;border:1px solid #dee2e6;border-top:0;
                    border-radius:0 0 8px 8px">
            {teaser_html}
            {config_section}
            <h3 style="margin:20px 0 8px">Catalog</h3>
            <table style="border-collapse:collapse;width:100%">
                <tr style="background:#f5f5f5">
                    <th style="text-align:left;padding:6px 12px">Table</th>
                    <th style="text-align:right;padding:6px 12px">Size</th>
                    <th style="text-align:right;padding:6px 12px">Rows</th>
                    <th style="text-align:right;padding:6px 12px">Cols</th>
                    <th style="text-align:left;padding:6px 12px">Partitions</th>
                    <th style="text-align:right;padding:6px 12px">Files</th>
                </tr>
                {table_rows or '<tr><td colspan="6" style="padding:12px;color:#999">No tables cataloged</td></tr>'}
            </table>
            {scan_section}
            <h3 style="margin:20px 0 8px">Captured Plans</h3>
            <table style="border-collapse:collapse;width:100%">
                <tr style="background:#f5f5f5">
                    <th style="text-align:left;padding:6px 12px">Label</th>
                    <th style="text-align:right;padding:6px 12px">Nodes</th>
                    <th style="text-align:left;padding:6px 12px">Details</th>
                    <th style="text-align:left;padding:6px 12px">Fingerprint</th>
                    <th style="text-align:left;padding:6px 12px">SQL</th>
                </tr>
                {plan_rows or '<tr><td colspan="5" style="padding:12px;color:#999">No plans captured</td></tr>'}
            </table>
            {"<p style='margin-top:16px;color:#999;font-size:12px'>⚠ " + str(len(errors)) + " warnings — see snapshot JSON for details.</p>" if errors else ""}
            <p style="margin-top:20px;font-size:13px;color:#666">
                Run full analysis for detailed findings and cost estimates.
            </p>
        </div>
    </div>"""