"""
Passive plan capture via monkey-patching.

Hooks into SparkSession.sql(), DataFrame action methods, and
DataFrameWriter methods to silently capture every physical plan
that Spark produces during a session. The user's code runs
completely unmodified.

Additionally hooks DataFrame construction methods (.filter(), .join(),
.select(), etc.) to track _cy_lines: a per-DataFrame dict mapping
file paths to sets of line numbers that shaped the DataFrame. When an
action fires, the accumulated _cy_lines become the plan's
constructionLines — the set of source lines that, if changed in a PR,
should trigger re-evaluation of this plan's cost.

Safety contract:
  - Every hook is wrapped in try/except — our code NEVER breaks theirs
  - Original methods are stored and restored cleanly on stop()
  - Re-entrancy guard prevents our own internal Spark calls from
    being captured (e.g. DESCRIBE DETAIL for catalog stats)
  - All return values pass through untouched
  - Construction line tracking is purely additive — if it fails for
    any reason, the plan is still captured without lines
"""

from __future__ import annotations

import functools
import threading
import weakref
from typing import TYPE_CHECKING, Any, Callable, Optional

from ._provenance import get_current_user_line

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


# SQL statement prefixes that produce trivial or no-op plans.
# These are DDL, metadata, and session commands — never interesting.
_SKIP_SQL_PREFIXES = (
    "DROP ", "CREATE ", "ALTER ", "TRUNCATE ",
    "DESCRIBE ", "SHOW ", "SET ", "RESET ", "USE ",
    "GRANT ", "REVOKE ", "DENY ",
    "MSCK ", "REFRESH ", "CACHE ", "UNCACHE ", "CLEAR ",
    "ADD ", "LIST ",
    "EXPLAIN ",
)

# Exact SQL statements to skip (probes, health checks)
_SKIP_SQL_EXACT = frozenset({
    "SELECT 1",
    "SELECT 1 AS _",
    "SELECT CURRENT_VERSION()",
})

# DataFrame action methods that trigger execution.
# We capture the plan AFTER the action returns so we get the
# post-AQE executed plan on classic PySpark.
_ACTION_METHODS = (
    "collect", "count", "show", "toPandas",
    "first", "head", "take", "tail",
)

# DataFrameWriter methods that trigger a write action.
_WRITER_METHODS = (
    "save", "saveAsTable", "insertInto",
    "parquet", "csv", "json", "orc",
    "text", "jdbc",
)

# ── Construction tracking ────────────────────────────────────────────────
#
# DataFrame methods that return a NEW DataFrame derived from self.
# Each patched method propagates _cy_lines from self → result and
# adds the current user code line.

_CONSTRUCTION_METHODS = (
    "filter", "where",
    "select", "selectExpr",
    "withColumn", "withColumnRenamed",
    "drop",
    "distinct", "dropDuplicates",
    "orderBy", "sort",
    "limit",
    "sample",
    "repartition", "coalesce",
    "toDF",
    "agg",
)

# DataFrame methods that merge TWO DataFrames — result gets the union
# of both sides' _cy_lines plus the current line.
_MERGE_METHODS = (
    "join", "crossJoin",
    "union", "unionAll", "unionByName",
    "subtract", "intersect", "intersectAll", "exceptAll",
)


# ── _cy_lines helpers ────────────────────────────────────────────────────

def _get_lines(df: Any) -> dict[str, set[int]]:
    """Get _cy_lines from a DataFrame, or empty dict."""
    return getattr(df, "_cy_lines", {})


def _copy_lines(lines: dict[str, set[int]]) -> dict[str, set[int]]:
    """Deep copy a construction lines dict."""
    return {k: set(v) for k, v in lines.items()}


def _set_lines(df: Any, lines: dict[str, set[int]]) -> None:
    """Set _cy_lines on a DataFrame. Silently fails if object is frozen."""
    try:
        df._cy_lines = lines
    except (AttributeError, TypeError):
        pass  # Some DataFrame implementations may not allow dynamic attrs


def _add_line(lines: dict[str, set[int]], filename: str, lineno: int) -> None:
    """Add a (filename, lineno) pair to a construction lines dict in place."""
    if filename in lines:
        lines[filename].add(lineno)
    else:
        lines[filename] = {lineno}


def _merge_line_dicts(
    a: dict[str, set[int]],
    b: dict[str, set[int]],
) -> dict[str, set[int]]:
    """Merge two construction line dicts. Returns a new dict."""
    result = _copy_lines(a)
    for k, v in b.items():
        if k in result:
            result[k] |= v
        else:
            result[k] = set(v)
    return result


def serialize_construction_lines(
    lines: dict[str, set[int]],
) -> list[dict[str, Any]]:
    """
    Convert _cy_lines to JSON-serializable construction line entries.

    Returns a list of entries, each with either:
      - {"file": "...", "lines": [sorted ints]}
            For absolute file line numbers (local scripts, utility modules)
      - {"cellFingerprint": "...", "lines": [sorted ints]}
            For Databricks cells (cell-relative line numbers).
            The server matches the fingerprint against the notebook
            source from git to compute the cell's offset.

    Multiple entries from the same Databricks cell are merged by
    fingerprint. If fingerprinting fails (file unreadable), the
    entry is silently dropped — partial data is better than bad data.
    """
    from ._provenance import is_databricks_cell_path, compute_cell_fingerprint

    # Separate cell paths from regular file paths.
    # Cell paths get fingerprinted; file paths pass through directly.
    cell_groups: dict[str, set[int]] = {}   # fingerprint → lines
    file_groups: dict[str, set[int]] = {}   # filepath → lines

    for path, line_set in lines.items():
        if not line_set:
            continue

        if is_databricks_cell_path(path):
            fp = compute_cell_fingerprint(path)
            if fp is not None:
                if fp in cell_groups:
                    cell_groups[fp] |= line_set
                else:
                    cell_groups[fp] = set(line_set)
            # If fingerprint fails, entry is dropped (file was deleted)
        else:
            if path in file_groups:
                file_groups[path] |= line_set
            else:
                file_groups[path] = set(line_set)

    result: list[dict[str, Any]] = []
    for fp in sorted(cell_groups):
        result.append({
            "cellFingerprint": fp,
            "lines": sorted(cell_groups[fp]),
        })
    for path in sorted(file_groups):
        result.append({
            "file": path,
            "lines": sorted(file_groups[path]),
        })
    return result


class CapturedPlan:
    """A plan captured during passive observation."""

    __slots__ = ("label", "df_ref", "sql", "trigger")

    def __init__(
        self,
        label: str,
        df: DataFrame,
        sql: Optional[str] = None,
        trigger: str = "unknown",
    ):
        self.label = label
        # Weak ref so we don't prevent GC of large DataFrames.
        # If the df is gone by the time stop() runs, we already
        # captured its plan at action time.
        self.df_ref: Callable[[], Optional[DataFrame]] = weakref.ref(df)
        self.sql = sql
        self.trigger = trigger


PlanCallback = Callable[[str, "DataFrame", Optional[str], str], None]


class PassiveCapture:
    """
    Monkey-patches Spark to passively capture plans.

    Usage:
        capture = PassiveCapture(spark, on_plan_captured=my_callback,
                                 source_path="/path/to/notebook.py")
        capture.start()
        # ... user's code runs normally ...
        capture.stop()

    The callback fires for every captured plan with:
        (label, dataframe, sql_text_or_none, trigger)

    Construction line tracking is automatic: each DataFrame accumulates
    a _cy_lines dict recording which source lines shaped it. The
    callback caller can read df._cy_lines to get the full line set.
    """

    def __init__(
        self,
        spark: SparkSession,
        on_plan_captured: PlanCallback,
        source_path: Optional[str] = None,
    ):
        self._spark = spark
        self._on_plan = on_plan_captured
        self._source_path = source_path
        self._originals: dict[str, Any] = {}
        self._active = False
        self._counter = 0

        # Re-entrancy guard: prevents our own internal spark.sql()
        # calls (catalog stats, DESCRIBE DETAIL, etc.) from being captured.
        self._inside_capture = threading.local()

        # Track SQL text for DataFrames created via spark.sql()
        # so we can label action-time captures with the original SQL.
        # Keyed on id(df) — not perfect (id reuse) but good enough
        # for a single notebook session.
        self._sql_texts: dict[int, str] = {}

        # Actual runtime classes (set during patching, used for restore)
        self._df_class: Optional[type] = None
        self._writer_class: Optional[type] = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> PassiveCapture:
        """Start capturing. Patches SparkSession and DataFrame classes."""
        if self._active:
            return self
        # Detect runtime classes BEFORE any patching, so the probe
        # query doesn't get captured by our own hooks.
        self._df_class = self._detect_dataframe_class()
        self._writer_class = self._detect_writer_class()
        self._patch_spark_sql()
        self._patch_spark_table()
        self._patch_construction_methods()
        self._patch_merge_methods()
        self._patch_dataframe_actions()
        self._patch_writer_actions()
        self._active = True
        return self

    def stop(self) -> PassiveCapture:
        """Stop capturing. Restores all original methods."""
        if not self._active:
            return self
        self._active = False
        self._restore_all()
        self._sql_texts.clear()
        return self

    # ── Patching: spark.sql() ────────────────────────────────────────────

    def _patch_spark_sql(self) -> None:
        """Patch SparkSession.sql to capture SQL queries and seed _cy_lines."""
        session = self._spark
        original = session.sql
        self._originals["spark.sql"] = original
        capture = self

        @functools.wraps(original)
        def patched_sql(sql_text: str, *args: Any, **kwargs: Any) -> DataFrame:
            df = original(sql_text, *args, **kwargs)
            if not capture._is_reentrant():
                capture._on_sql_called(sql_text, df)
                # Seed construction lines on the returned DataFrame
                capture._seed_lines(df)
            return df

        session.sql = patched_sql

    def _on_sql_called(self, sql_text: str, df: DataFrame) -> None:
        """Called when spark.sql() is invoked. Captures the plan."""
        try:
            # Skip DDL and metadata queries — they produce trivial plans
            if self._should_skip_sql(sql_text):
                return

            self._enter_capture()
            self._counter += 1
            # Store SQL text so action-time capture can use it as label
            self._sql_texts[id(df)] = sql_text
            # Capture immediately — this gives us the Catalyst-optimized
            # plan. If an action is later called on this df, we'll capture
            # again with the post-AQE plan.
            label = self._sql_label(sql_text, self._counter)
            self._on_plan(label, df, sql_text, "spark.sql")
        except Exception:
            pass  # Never break user code
        finally:
            self._exit_capture()

    # ── Patching: spark.table() ──────────────────────────────────────────

    def _patch_spark_table(self) -> None:
        """Patch SparkSession.table to seed _cy_lines on reader DataFrames."""
        session = self._spark
        original = getattr(session, "table", None)
        if original is None:
            return
        self._originals["spark.table"] = original
        capture = self

        @functools.wraps(original)
        def patched_table(table_name: str, *args: Any, **kwargs: Any) -> DataFrame:
            df = original(table_name, *args, **kwargs)
            if not capture._is_reentrant():
                capture._seed_lines(df)
            return df

        session.table = patched_table

    # ── Patching: construction methods ───────────────────────────────────
    #
    # These methods return a new DataFrame derived from self. Each
    # patched version propagates self._cy_lines to the result and adds
    # the current user code line.

    def _patch_construction_methods(self) -> None:
        """Patch DataFrame transformation methods to track construction lines."""
        DFClass = self._df_class
        if DFClass is None:
            return

        for method_name in _CONSTRUCTION_METHODS:
            original = getattr(DFClass, method_name, None)
            if original is None:
                continue
            # Don't overwrite if we already patched this as an action
            key = f"DataFrame.{method_name}"
            if key in self._originals:
                continue
            self._originals[key] = original
            self._install_construction_patch(DFClass, method_name, original)

    def _install_construction_patch(
        self, df_class: type, method_name: str, original: Any
    ) -> None:
        """Install a wrapper on a DataFrame construction method."""
        capture = self

        @functools.wraps(original)
        def patched(df_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(df_self, *args, **kwargs)
            try:
                if capture._active:
                    # Propagate _cy_lines from self → result, adding current line
                    lines = _copy_lines(_get_lines(df_self))
                    line_info = get_current_user_line(capture._source_path)
                    if line_info:
                        _add_line(lines, line_info[0], line_info[1])
                    _set_lines(result, lines)
            except Exception:
                pass  # Never break user code
            return result

        setattr(df_class, method_name, patched)

    # ── Patching: merge methods ──────────────────────────────────────────
    #
    # These methods merge two DataFrames. The result gets the union
    # of both sides' _cy_lines.

    def _patch_merge_methods(self) -> None:
        """Patch DataFrame merge methods (.join, .union, etc.)."""
        DFClass = self._df_class
        if DFClass is None:
            return

        for method_name in _MERGE_METHODS:
            original = getattr(DFClass, method_name, None)
            if original is None:
                continue
            key = f"DataFrame.{method_name}"
            if key in self._originals:
                continue
            self._originals[key] = original
            self._install_merge_patch(DFClass, method_name, original)

    def _install_merge_patch(
        self, df_class: type, method_name: str, original: Any
    ) -> None:
        """Install a wrapper on a DataFrame merge method."""
        capture = self

        @functools.wraps(original)
        def patched(df_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(df_self, *args, **kwargs)
            try:
                if capture._active:
                    # Merge _cy_lines from both sides
                    left_lines = _get_lines(df_self)
                    right_lines = _get_lines(args[0]) if args else {}
                    merged = _merge_line_dicts(left_lines, right_lines)
                    line_info = get_current_user_line(capture._source_path)
                    if line_info:
                        _add_line(merged, line_info[0], line_info[1])
                    _set_lines(result, merged)
            except Exception:
                pass  # Never break user code
            return result

        setattr(df_class, method_name, patched)

    # ── Seed _cy_lines on new DataFrames ─────────────────────────────────

    def _seed_lines(self, df: Any) -> None:
        """Seed _cy_lines on a DataFrame created by spark.sql/table/read."""
        try:
            lines: dict[str, set[int]] = {}
            line_info = get_current_user_line(self._source_path)
            if line_info:
                _add_line(lines, line_info[0], line_info[1])
            _set_lines(df, lines)
        except Exception:
            pass

    # ── Patching: DataFrame actions ──────────────────────────────────────

    def _patch_dataframe_actions(self) -> None:
        """Patch DataFrame action methods to capture post-execution plans."""
        DFClass = self._df_class
        if DFClass is None:
            return

        for method_name in _ACTION_METHODS:
            original = getattr(DFClass, method_name, None)
            if original is None:
                continue
            self._originals[f"DataFrame.{method_name}"] = original
            self._install_action_patch(DFClass, method_name, original)

    def _install_action_patch(
        self, df_class: type, method_name: str, original: Any
    ) -> None:
        """Install a wrapper on a DataFrame action method."""
        capture = self

        @functools.wraps(original)
        def patched(df_self: DataFrame, *args: Any, **kwargs: Any) -> Any:
            result = original(df_self, *args, **kwargs)
            if capture._active and not capture._is_reentrant():
                capture._on_action_called(df_self, method_name)
            return result

        setattr(df_class, method_name, patched)

    def _on_action_called(self, df: DataFrame, method_name: str) -> None:
        """Called after a DataFrame action completes."""
        try:
            self._enter_capture()
            self._counter += 1

            # Try to get the original SQL text for a better label
            sql_text = self._sql_texts.get(id(df))
            if sql_text:
                label = self._sql_label(sql_text, self._counter)
            else:
                label = f"{method_name}-{self._counter}"

            self._on_plan(label, df, sql_text, f"action.{method_name}")
        except Exception:
            pass
        finally:
            self._exit_capture()

    # ── Patching: DataFrameWriter ────────────────────────────────────────

    def _patch_writer_actions(self) -> None:
        """Patch DataFrameWriter write methods to capture write plans."""
        WriterClass = self._writer_class
        if WriterClass is None:
            return

        for method_name in _WRITER_METHODS:
            original = getattr(WriterClass, method_name, None)
            if original is None:
                continue
            self._originals[f"DataFrameWriter.{method_name}"] = original
            self._install_writer_patch(WriterClass, method_name, original)

    def _install_writer_patch(
        self, writer_class: type, method_name: str, original: Any
    ) -> None:
        """Install a wrapper on a DataFrameWriter method."""
        capture = self

        @functools.wraps(original)
        def patched(writer_self: Any, *args: Any, **kwargs: Any) -> Any:
            # Grab the DataFrame before the write executes.
            # PySpark's DataFrameWriter stores it as ._df
            source_df = getattr(writer_self, "_df", None)

            result = original(writer_self, *args, **kwargs)

            if capture._active and not capture._is_reentrant() and source_df is not None:
                capture._on_write_called(source_df, method_name, args)
            return result

        setattr(writer_class, method_name, patched)

    def _on_write_called(
        self, df: DataFrame, method_name: str, args: tuple
    ) -> None:
        """Called after a DataFrameWriter method completes."""
        try:
            self._enter_capture()
            self._counter += 1

            # Build a descriptive label
            sql_text = self._sql_texts.get(id(df))
            if sql_text:
                label = f"write-{self._sql_label(sql_text, self._counter)}"
            else:
                # Try to include the output path/table for context
                target = ""
                if args:
                    target = str(args[0])
                    if len(target) > 60:
                        target = "..." + target[-57:]
                if target:
                    label = f"write-{method_name}-{target}"
                else:
                    label = f"write-{method_name}-{self._counter}"

            self._on_plan(label, df, sql_text, f"write.{method_name}")
        except Exception:
            pass
        finally:
            self._exit_capture()

    # ── Restore ──────────────────────────────────────────────────────────

    def _restore_all(self) -> None:
        """Restore all patched methods to their originals."""
        for key, original in self._originals.items():
            try:
                if key == "spark.sql":
                    self._spark.sql = original
                elif key == "spark.table":
                    self._spark.table = original
                elif key.startswith("DataFrame."):
                    # Use the stored runtime class, not the import
                    cls = getattr(self, "_df_class", None)
                    if cls is not None:
                        method_name = key.split(".", 1)[1]
                        setattr(cls, method_name, original)
                elif key.startswith("DataFrameWriter."):
                    cls = getattr(self, "_writer_class", None)
                    if cls is not None:
                        method_name = key.split(".", 1)[1]
                        setattr(cls, method_name, original)
            except Exception:
                pass
        self._originals.clear()

    # ── Runtime class detection ──────────────────────────────────────────

    def _detect_dataframe_class(self) -> Optional[type]:
        """
        Detect the actual DataFrame class used at runtime.

        On classic PySpark: pyspark.sql.dataframe.DataFrame
        On Spark Connect:   pyspark.sql.connect.dataframe.DataFrame
        On Databricks:      may be either, depending on cluster type

        We probe with a real query because the import path
        `from pyspark.sql import DataFrame` always returns the classic
        class, even when Connect is active.
        """
        try:
            probe = self._spark.sql("SELECT 1")
            return type(probe)
        except Exception:
            pass
        # Fallback to import
        try:
            from pyspark.sql import DataFrame
            return DataFrame
        except ImportError:
            return None

    def _detect_writer_class(self) -> Optional[type]:
        """Detect the actual DataFrameWriter class used at runtime."""
        try:
            probe = self._spark.sql("SELECT 1")
            return type(probe.write)
        except Exception:
            pass
        try:
            from pyspark.sql import DataFrameWriter
            return DataFrameWriter
        except ImportError:
            return None

    # ── Re-entrancy guard ────────────────────────────────────────────────

    def _is_reentrant(self) -> bool:
        return getattr(self._inside_capture, "flag", False)

    def _enter_capture(self) -> None:
        self._inside_capture.flag = True

    def _exit_capture(self) -> None:
        self._inside_capture.flag = False

    # ── Filtering ─────────────────────────────────────────────────────────

    @staticmethod
    def _should_skip_sql(sql_text: str) -> bool:
        """
        Return True for SQL that produces trivial or no-op plans.

        DDL (DROP, CREATE, ALTER), metadata (DESCRIBE, SHOW),
        session commands (SET, USE), and known probes (SELECT 1)
        are never interesting for plan analysis.
        """
        stripped = sql_text.strip()
        upper = stripped.upper()

        # Exact matches (probes, health checks)
        if upper in _SKIP_SQL_EXACT:
            return True

        # Handle multi-line SQL — check the first non-empty keyword
        # Also handle comments: skip leading -- or /* ... */
        for line in upper.split("\n"):
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            # Strip block comments at start
            while line.startswith("/*"):
                end = line.find("*/")
                if end == -1:
                    break
                line = line[end + 2:].strip()
            if not line:
                continue
            return line.startswith(_SKIP_SQL_PREFIXES)
        return False

    # ── Label helpers ────────────────────────────────────────────────────

    @staticmethod
    def _sql_label(sql_text: str, counter: int) -> str:
        """Build a label from SQL text."""
        # Clean up whitespace
        clean = " ".join(sql_text.split())
        if len(clean) > 80:
            clean = clean[:77] + "..."
        return f"sql-{counter}-{clean}"