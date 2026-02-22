"""
Output formatting — terminal summary and inline HTML for Databricks.
"""

from __future__ import annotations

from typing import Any

from ._util import fmt_bytes, esc_html, parse_int
from .plans import (
    operators_from_entry, has_metrics,
    scan_bytes_from_entry, shuffle_bytes_from_entry,
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

    # Table rows
    table_rows = ""
    for tname, tstats in sorted(
        tables.items(), key=lambda x: x[1].get("sizeInBytes", 0), reverse=True,
    ):
        size = tstats.get("sizeInBytes", 0)
        rows = tstats.get("rowCount")
        pcols = tstats.get("partitionColumns", [])
        files = tstats.get("fileCount")

        row_str = f"{rows:,}" if rows else "—"
        pcol_str = ", ".join(pcols) if pcols else "—"
        file_str = str(files) if files else "—"

        table_rows += (
            f'<tr><td style="font-family:monospace;font-size:13px">{esc_html(tname)}</td>'
            f'<td style="text-align:right">{fmt_bytes(size)}</td>'
            f'<td style="text-align:right">{row_str}</td>'
            f'<td>{pcol_str}</td>'
            f'<td style="text-align:right">{file_str}</td></tr>'
        )

    # Plan rows — include per-plan scan/shuffle when metrics are available
    plan_rows = ""
    for p in plans:
        label = p.get("label", "?")
        nodes = p.get("nodeCount", 0)
        fp = p.get("fingerprint", "?")[:12]
        sql = p.get("sql", "")
        sql_preview = (sql[:80] + "...") if len(sql) > 80 else sql

        if has_metrics(p):
            scan_b = scan_bytes_from_entry(p)
            shuf_b = shuffle_bytes_from_entry(p)
            io_str = fmt_bytes(scan_b) if scan_b else "—"
            shuf_str = fmt_bytes(shuf_b) if shuf_b else "—"
        else:
            io_str = "—"
            shuf_str = "—"

        plan_rows += (
            f'<tr><td style="font-family:monospace;font-size:13px">{esc_html(label)}</td>'
            f'<td style="text-align:right">{nodes}</td>'
            f'<td style="text-align:right">{io_str}</td>'
            f'<td style="text-align:right">{shuf_str}</td>'
            f'<td style="font-family:monospace;font-size:11px;color:#666">{fp}</td>'
            f'<td style="font-size:12px;color:#555">{esc_html(sql_preview)}</td></tr>'
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

    # Runtime I/O stats cards (only when metrics are present)
    io_cards = ""
    if plans_with_metrics > 0:
        io_cards = (
            f'<div><span style="font-size:28px;font-weight:700">{fmt_bytes(total_scan)}</span>'
            f'<br><span style="font-size:12px;color:#666">Scanned</span></div>'
            f'<div><span style="font-size:28px;font-weight:700">{fmt_bytes(total_shuffle)}</span>'
            f'<br><span style="font-size:12px;color:#666">Shuffled</span></div>'
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
                    <th style="text-align:left;padding:6px 12px">Partitions</th>
                    <th style="text-align:right;padding:6px 12px">Files</th>
                </tr>
                {table_rows or '<tr><td colspan="5" style="padding:12px;color:#999">No tables cataloged</td></tr>'}
            </table>
            <h3 style="margin:20px 0 8px">Captured Plans</h3>
            <table style="border-collapse:collapse;width:100%">
                <tr style="background:#f5f5f5">
                    <th style="text-align:left;padding:6px 12px">Label</th>
                    <th style="text-align:right;padding:6px 12px">Nodes</th>
                    <th style="text-align:right;padding:6px 12px">Scanned</th>
                    <th style="text-align:right;padding:6px 12px">Shuffled</th>
                    <th style="text-align:left;padding:6px 12px">Fingerprint</th>
                    <th style="text-align:left;padding:6px 12px">SQL</th>
                </tr>
                {plan_rows or '<tr><td colspan="6" style="padding:12px;color:#999">No plans captured</td></tr>'}
            </table>
            {"<p style='margin-top:16px;color:#999;font-size:12px'>⚠ " + str(len(errors)) + " warnings — see snapshot JSON for details.</p>" if errors else ""}
            <p style="margin-top:20px;font-size:13px;color:#666">
                Run full analysis for detailed findings and cost estimates.
            </p>
        </div>
    </div>"""