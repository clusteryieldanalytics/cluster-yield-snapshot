"""
CYSnapshot — passive Spark plan capture.

Drop two lines at the top of any notebook, two at the bottom.
Everything in between runs unmodified.

    from cluster_yield_snapshot import CYSnapshot
    cy = CYSnapshot(spark).start()

    # ═══ their notebook, completely untouched ═══
    df = spark.sql("SELECT * FROM orders WHERE date > '2024-01-01'")
    enriched = df.join(users, "user_id").groupBy("region").agg(sum("amount"))
    enriched.write.parquet("s3://output/daily")

    # last cell
    cy.stop().save()

Or with upload:

    cy = CYSnapshot(spark, api_key="cy_...", environment="prod").start()
    # ... notebook ...
    cy.stop().upload()
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

from . import plans as _plans
from . import catalog as _catalog
from . import config as _config
from . import environment as _env
from ._capture import PassiveCapture
from .quick_scan import quick_scan
from .formatting import print_summary, render_html
from .upload import upload_snapshot, upload_snapshot_urllib, UploadResult, UploadError

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

VERSION = "0.3.7"


class CYSnapshot:
    """
    Passively captures Spark plans, config, and catalog stats.

    Primary usage (passive — captures everything automatically):

        cy = CYSnapshot(spark).start()
        # ... entire notebook runs unchanged ...
        cy.stop().save()

    Manual capture (for edge cases):

        cy = CYSnapshot(spark)
        cy.query("label", "SELECT ...")
        cy.df("label", some_dataframe)
        cy.save()
    """

    def __init__(
        self,
        spark: SparkSession,
        *,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        base_url: Optional[str] = None,
        quiet: bool = False,
    ):
        """
        Args:
            spark: Active SparkSession
            api_key: Cluster Yield API key for upload (cy_...)
            environment: Environment label (e.g. "prod-analytics")
            base_url: Override API base URL (for self-hosted)
            quiet: Suppress progress output
        """
        self._spark = spark
        self._api_key = api_key
        self._environment = environment
        self._base_url = base_url
        self._quiet = quiet

        self._plans: list[dict[str, Any]] = []
        self._catalog_tables: dict[str, dict[str, Any]] = {}
        self._errors: list[str] = []
        self._captured_at = datetime.now(timezone.utc).isoformat()
        self._cached_snapshot: Optional[dict[str, Any]] = None

        # Passive capture engine
        self._capture: Optional[PassiveCapture] = None

        # Track fingerprints to avoid storing duplicate plans.
        # Same query captured at spark.sql() time and again at action
        # time will have the same fingerprint if AQE didn't change it.
        # We keep the later capture (post-AQE) and drop the earlier one.
        self._fingerprints: dict[str, int] = {}  # fingerprint -> index in _plans

        # Secondary dedup by normalized SQL text. Catches duplicates where
        # the same query produces slightly different plan fingerprints
        # (e.g. different column reference IDs on Connect).
        self._sql_hashes: dict[str, int] = {}  # sql_hash -> index in _plans

        if not quiet:
            print(f"[CY] Snapshot initialized — Spark {spark.version}")

    # ── Passive capture ──────────────────────────────────────────────────

    def start(self) -> CYSnapshot:
        """
        Start passive plan capture.

        Hooks into spark.sql(), DataFrame actions (collect, show, count,
        toPandas, etc.), and DataFrameWriter methods (parquet, save, etc.)
        to silently record every physical plan Spark produces.

        The user's code runs completely unmodified. Our hooks never raise —
        if capture fails for any plan, it's silently skipped.
        """
        if self._capture is not None and self._capture.active:
            self._log("Already capturing — ignoring duplicate start()")
            return self

        self._capture = PassiveCapture(self._spark, self._on_plan_captured)
        self._capture.start()
        self._log("Passive capture started — all queries will be recorded")
        return self

    def stop(self) -> CYSnapshot:
        """
        Stop passive capture and collect catalog metadata.

        Restores all patched methods, then gathers catalog stats
        for every table that appeared in the captured plans.
        """
        if self._capture is None or not self._capture.active:
            self._log("Not capturing — ignoring stop()")
            return self

        self._capture.stop()
        self._capture = None

        # Gather catalog stats for all tables we saw in plans
        discovered_tables: set[str] = set()
        for p in self._plans:
            discovered_tables |= _plans.tables_from_entry(p)

        if discovered_tables:
            self._log(f"Collecting catalog stats for {len(discovered_tables)} "
                      f"discovered tables...")
            for table_name in discovered_tables:
                self._capture_catalog_stats(table_name)

        self._log(f"Capture complete — {len(self._plans)} plans, "
                  f"{len(self._catalog_tables)} tables cataloged")
        return self

    # Minimum plan nodes to keep. Plans with fewer nodes are trivial
    # (e.g. empty DDL results, single-node metadata queries).
    _MIN_PLAN_NODES = 2

    def _on_plan_captured(
        self,
        label: str,
        df: Any,  # DataFrame — avoid import for safety
        sql: Optional[str],
        trigger: str,
    ) -> None:
        """
        Callback from PassiveCapture when a plan is observed.

        Called from monkey-patched methods, so this MUST NOT raise.
        """
        try:
            self._invalidate_cache()
            entry = _plans.capture_plan(df, label, sql=sql)
            entry["trigger"] = trigger
            nodes = entry.get("nodeCount", 0)
            fp = entry.get("fingerprint", "")

            # ── Filter: skip trivial plans ────────────────────────────
            if nodes < self._MIN_PLAN_NODES:
                return

            # ── Filter: skip empty fingerprints (failed extraction) ───
            if not fp or fp == "empty":
                return

            # ── Dedup: only suppress true duplicates ──────────────────
            # Two cases where we dedup:
            #   1. Same SQL captured at spark.sql() time, then again at
            #      action time → replace pre-AQE with post-AQE plan
            #   2. Exact same SQL text fired twice (cell re-run, etc.)
            #
            # We do NOT dedup by fingerprint alone because different
            # queries can produce the same operator sequence (e.g. two
            # different 3-way joins both produce SMJ→Exchange→FileScan).

            sql_hash = self._sql_hash(sql)

            # Case 1: SQL text match (most reliable)
            if sql_hash and sql_hash in self._sql_hashes:
                idx = self._sql_hashes[sql_hash]
                old_trigger = self._plans[idx].get("trigger", "")
                if old_trigger == "spark.sql" and trigger != "spark.sql":
                    # Upgrade pre-AQE → post-AQE
                    entry["sql"] = sql or self._plans[idx].get("sql")
                    entry["label"] = self._plans[idx].get("label", label)
                    self._plans[idx] = entry
                    self._fingerprints[fp] = idx
                    self._log(f"  ↻ Updated '{entry['label'][:60]}' "
                              f"with post-execution plan [{trigger}]")
                return  # either upgraded or exact dup — don't add

            # Case 2: Same fingerprint
            if fp in self._fingerprints:
                idx = self._fingerprints[fp]
                old_trigger = self._plans[idx].get("trigger", "")
                old_sql = self._plans[idx].get("sql")

                if old_trigger == "spark.sql" and trigger != "spark.sql":
                    # Pre-AQE → post-AQE upgrade for the same query
                    if old_sql and not entry.get("sql"):
                        entry["sql"] = old_sql
                        entry["label"] = self._plans[idx]["label"]
                    self._plans[idx] = entry
                    if sql_hash:
                        self._sql_hashes[sql_hash] = idx
                    self._log(f"  ↻ Updated '{entry['label'][:60]}' "
                              f"with post-execution plan [{trigger}]")
                    return

                # Same fingerprint, both from actions/writes.
                # If they have different SQL text, they're different
                # queries with the same plan shape → keep both.
                # If neither has SQL (DF API chains) or same SQL → dedup.
                old_sql_hash = self._sql_hash(old_sql)
                if sql_hash and old_sql_hash and sql_hash != old_sql_hash:
                    pass  # Different queries, same shape → fall through to store
                else:
                    # Same DF API chain executed twice, or cell re-run → skip
                    return

            # ── Store the plan ────────────────────────────────────────
            idx = len(self._plans)
            self._plans.append(entry)
            self._fingerprints[fp] = idx
            if sql_hash:
                self._sql_hashes[sql_hash] = idx

            mode = entry.get("planFormat", "?")
            self._log(f"  ✓ Captured '{label[:60]}' — "
                      f"{nodes} nodes [{mode}] ({trigger})")
        except Exception:
            pass  # Absolutely never break user code

    # ── Manual capture (still available) ─────────────────────────────────

    def query(self, label: str, sql: str) -> CYSnapshot:
        """Manually capture the physical plan for a SQL query."""
        self._invalidate_cache()
        try:
            df = self._spark.sql(sql)
            entry = _plans.capture_plan(df, label, sql=sql)
            entry["trigger"] = "manual.query"
            self._plans.append(entry)
            fp = entry.get("fingerprint", "")
            if fp and fp != "empty":
                self._fingerprints[fp] = len(self._plans) - 1
            self._log(f"  Plan '{label}': {entry.get('nodeCount', 0)} nodes")
        except Exception as e:
            self._record_error("query", label, e)
        return self

    def df(self, label: str, dataframe: DataFrame) -> CYSnapshot:
        """Manually capture the physical plan for an existing DataFrame."""
        self._invalidate_cache()
        try:
            entry = _plans.capture_plan(dataframe, label)
            entry["trigger"] = "manual.df"
            self._plans.append(entry)
            fp = entry.get("fingerprint", "")
            if fp and fp != "empty":
                self._fingerprints[fp] = len(self._plans) - 1
            self._log(f"  Plan '{label}': {entry.get('nodeCount', 0)} nodes")
        except Exception as e:
            self._record_error("df", label, e)
        return self

    def tables(self, *table_names: str) -> CYSnapshot:
        """Manually capture catalog stats for specific tables."""
        self._invalidate_cache()
        for t in table_names:
            self._capture_catalog_stats(t)
        return self

    # ── Output ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Write the snapshot to a JSON file and print a summary.

        Auto-calls stop() if capture is still running.
        """
        if self._capture is not None and self._capture.active:
            self._log("Auto-stopping capture before save...")
            self.stop()

        snapshot = self.to_dict()
        json_str = json.dumps(snapshot, indent=2, default=str)

        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"/tmp/cluster_yield_snapshot_{ts}.json"

        saved_path = self._write_file(path, json_str)

        if saved_path:
            print(f"\n[CY] Snapshot saved → {saved_path}")
            if saved_path.startswith("/dbfs/"):
                print(f"     DBFS: {saved_path.replace('/dbfs/', 'dbfs:/')}")
        else:
            print(f"\n[CY] ⚠ Could not write to {path}")
            print("     Use cy.to_dict() to access the data in memory.")

        self._print_summary(snapshot)
        return saved_path or path

    def upload(
        self,
        *,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> UploadResult:
        """
        Upload the snapshot to the Cluster Yield backend.

        Auto-calls stop() if capture is still running.
        Server analyzes on ingest: detectors, cost estimation, diff.
        """
        if self._capture is not None and self._capture.active:
            self._log("Auto-stopping capture before upload...")
            self.stop()

        key = api_key or self._api_key
        env = environment or self._environment
        url = base_url or self._base_url

        if not key:
            raise ValueError(
                "api_key required. Pass to CYSnapshot() or upload()."
            )
        if not env:
            raise ValueError(
                "environment required. Pass to CYSnapshot() or upload()."
            )

        snapshot = self.to_dict()
        kwargs: dict[str, Any] = {"api_key": key, "environment": env}
        if url:
            kwargs["base_url"] = url

        try:
            result = upload_snapshot(snapshot, **kwargs)
        except ImportError:
            result = upload_snapshot_urllib(snapshot, **kwargs)

        print(f"\n[CY] Uploaded → {result}")
        if result.dashboard_url:
            print(f"     Dashboard: {result.dashboard_url}")

        self._print_summary(snapshot)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as a Python dict."""
        if self._cached_snapshot is not None:
            return self._cached_snapshot
        self._cached_snapshot = self._build_snapshot()
        return self._cached_snapshot

    def to_json(self, indent: int = 2) -> str:
        """Return the snapshot as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def display_summary(self) -> CYSnapshot:
        """Print a summary without saving."""
        self._print_summary(self.to_dict())
        return self

    def display_html(self) -> CYSnapshot:
        """Render an HTML summary inline in a Databricks notebook."""
        snapshot = self.to_dict()
        teasers = quick_scan(
            snapshot.get("plans", []),
            snapshot.get("catalog", {}).get("tables", {}),
            snapshot.get("config", {}),
        )
        html = render_html(snapshot, teasers)
        try:
            from IPython.display import display, HTML
            display(HTML(html))
        except ImportError:
            print("[CY] HTML display not available — use save() instead.")
        return self

    # ── Internals ────────────────────────────────────────────────────────

    def _capture_catalog_stats(self, table_name: str) -> None:
        if table_name in self._catalog_tables:
            return
        try:
            stats = _catalog.get_table_stats(self._spark, table_name)
            if stats:
                self._catalog_tables[table_name] = stats
        except Exception as e:
            self._record_error("catalog", table_name, e)

    def _build_snapshot(self) -> dict[str, Any]:
        return {
            "snapshot": {
                "version": VERSION,
                "capturedAt": self._captured_at,
                "snapshotType": "environment",
            },
            "environment": _env.capture_environment(
                self._spark, self._captured_at
            ),
            "config": _config.capture_config(self._spark),
            "catalog": {
                "tables": _catalog.serialize_catalog(self._catalog_tables),
            },
            "plans": self._plans,
            "errors": self._errors if self._errors else None,
        }

    def _write_file(self, path: str, content: str) -> Optional[str]:
        try:
            os.makedirs(os.path.dirname(path) or "/tmp", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return path
        except (OSError, IOError) as e:
            self._record_error("save", path, e)

        if path.startswith("/dbfs/"):
            fallback = path.replace("/dbfs/", "/tmp/cy_")
            try:
                with open(fallback, "w") as f:
                    f.write(content)
                self._log(f"  Fell back to local: {fallback}")
                return fallback
            except (OSError, IOError):
                pass

        if path != "/tmp/cy_snapshot.json":
            try:
                with open("/tmp/cy_snapshot.json", "w") as f:
                    f.write(content)
                return "/tmp/cy_snapshot.json"
            except (OSError, IOError):
                pass

        return None

    def _invalidate_cache(self) -> None:
        self._cached_snapshot = None

    @staticmethod
    def _sql_hash(sql: Optional[str]) -> Optional[str]:
        """Normalize SQL and return a short hash for dedup."""
        if not sql:
            return None
        import hashlib
        # Normalize: collapse whitespace, lowercase
        normalized = " ".join(sql.split()).lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _record_error(self, phase: str, target: str, exception: Exception) -> None:
        msg = f"[{phase}] {target}: {type(exception).__name__}: {exception}"
        self._errors.append(msg)
        self._log(f"  ⚠ {msg}")

    def _log(self, message: str) -> None:
        if not self._quiet:
            print(f"[CY] {message}")

    def _print_summary(self, snapshot: dict[str, Any]) -> None:
        teasers = quick_scan(
            snapshot.get("plans", []),
            snapshot.get("catalog", {}).get("tables", {}),
            snapshot.get("config", {}),
        )
        print_summary(snapshot, teasers)

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> CYSnapshot:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# ── Convenience ──────────────────────────────────────────────────────────

def snapshot_capture(
    spark: SparkSession,
    run: Optional[Callable[[], Any]] = None,
    path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    One-liner with a callable:

        path = snapshot_capture(spark, run=lambda: run_notebook())

    Or wrap a block with the context manager:

        with CYSnapshot(spark) as cy:
            # ... notebook code ...
        cy.save()
    """
    cy = CYSnapshot(spark, **kwargs)
    if run is not None:
        cy.start()
        try:
            run()
        finally:
            cy.stop()
    return cy.save(path=path)