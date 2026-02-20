"""
Passive plan capture via monkey-patching.

Hooks into SparkSession.sql(), DataFrame action methods, and
DataFrameWriter methods to silently capture every physical plan
that Spark produces during a session. The user's code runs
completely unmodified.

Safety contract:
  - Every hook is wrapped in try/except — our code NEVER breaks theirs
  - Original methods are stored and restored cleanly on stop()
  - Re-entrancy guard prevents our own internal Spark calls from
    being captured (e.g. DESCRIBE DETAIL for catalog stats)
  - All return values pass through untouched
"""

from __future__ import annotations

import functools
import threading
import weakref
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


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
        capture = PassiveCapture(spark, on_plan_captured=my_callback)
        capture.start()
        # ... user's code runs normally ...
        capture.stop()

    The callback fires for every captured plan with:
        (label, dataframe, sql_text_or_none, trigger)
    """

    def __init__(self, spark: SparkSession, on_plan_captured: PlanCallback):
        self._spark = spark
        self._on_plan = on_plan_captured
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

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> PassiveCapture:
        """Start capturing. Patches SparkSession and DataFrame classes."""
        if self._active:
            return self
        self._patch_spark_sql()
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
        """Patch SparkSession.sql to capture SQL queries."""
        session = self._spark
        original = session.sql
        self._originals["spark.sql"] = original
        capture = self

        @functools.wraps(original)
        def patched_sql(sql_text: str, *args: Any, **kwargs: Any) -> DataFrame:
            df = original(sql_text, *args, **kwargs)
            if not capture._is_reentrant():
                capture._on_sql_called(sql_text, df)
            return df

        session.sql = patched_sql

    def _on_sql_called(self, sql_text: str, df: DataFrame) -> None:
        """Called when spark.sql() is invoked. Captures the plan."""
        try:
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

    # ── Patching: DataFrame actions ──────────────────────────────────────

    def _patch_dataframe_actions(self) -> None:
        """Patch DataFrame action methods to capture post-execution plans."""
        try:
            from pyspark.sql import DataFrame as DFClass
        except ImportError:
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
        try:
            from pyspark.sql import DataFrameWriter as WriterClass
        except ImportError:
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
                elif key.startswith("DataFrame."):
                    from pyspark.sql import DataFrame as DFClass
                    method_name = key.split(".", 1)[1]
                    setattr(DFClass, method_name, original)
                elif key.startswith("DataFrameWriter."):
                    from pyspark.sql import DataFrameWriter as WriterClass
                    method_name = key.split(".", 1)[1]
                    setattr(WriterClass, method_name, original)
            except Exception:
                pass
        self._originals.clear()

    # ── Re-entrancy guard ────────────────────────────────────────────────

    def _is_reentrant(self) -> bool:
        return getattr(self._inside_capture, "flag", False)

    def _enter_capture(self) -> None:
        self._inside_capture.flag = True

    def _exit_capture(self) -> None:
        self._inside_capture.flag = False

    # ── Label helpers ────────────────────────────────────────────────────

    @staticmethod
    def _sql_label(sql_text: str, counter: int) -> str:
        """Build a label from SQL text."""
        # Clean up whitespace
        clean = " ".join(sql_text.split())
        if len(clean) > 80:
            clean = clean[:77] + "..."
        return f"sql-{counter}-{clean}"
