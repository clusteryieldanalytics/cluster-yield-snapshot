"""
Source provenance capture for plan-to-code matching.

Records three facts at capture time that enable deterministic matching
at CI time:

  1. sourcePath — the notebook or script that produced the snapshot
  2. constructionLines — the full set of source lines that shaped each
     plan's DataFrame chain (filter, join, select, etc.)
  3. triggerLine + triggerStack — the line (and full call chain) that
     triggered each plan's execution

These are captured automatically from the execution context. Zero
user-facing changes.

Databricks cell frame detection
────────────────────────────────
On Databricks, notebook cell code executes from a temp file like:
  /home/spark-88da72a5-.../.ipykernel/3322/command-7650185192914251-931726307

This path contains ".ipykernel/" which would match our internal frame
markers. We detect these cell frames FIRST (via the "command-" basename)
before checking internal markers, and normalize the filename to the
known sourcePath so constructionLines is keyed by the notebook path.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import sys
import traceback
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


# ── Source path detection ────────────────────────────────────────────────

# Directories and path fragments that indicate internal frames
# (not user code). Order doesn't matter — we check with `in`.
_INTERNAL_MARKERS = (
    "cluster_yield_snapshot",
    "pyspark/",
    "pyspark\\",
    "py4j/",
    "py4j\\",
    "IPython/",
    "ipykernel/",
    "ipykernel\\",
    "_pydev_",
    "pydevd",
    "debugpy/",
    "debugpy\\",
    "runpy.py",
    "threading.py",
    "<frozen ",
    # Databricks infrastructure (NOT cell frames — those are handled
    # separately by _is_databricks_cell_frame)
    "/databricks/",
    "python_shell/",
    "db_ipykernel_launcher",
    "db_driver_proxy",
    "PythonShell",
    # asyncio internals
    "asyncio/",
    # Tornado (Databricks kernel event loop)
    "tornado/",
)

# Pattern for Databricks cell frame basenames.
# Matches: command-7650185192914251-931726307
_DATABRICKS_CELL_RE = re.compile(r"^command-\d+")


def _is_databricks_cell_frame(filename: str) -> bool:
    """
    Return True if this frame is a Databricks notebook cell.

    On Databricks, cell code runs from temp files like:
      /home/spark-UUID/.ipykernel/PID/command-BIGNUM-BIGNUM

    These must be detected BEFORE _is_internal_frame since the path
    contains ".ipykernel/" which would match internal markers.
    """
    basename = os.path.basename(filename)
    return bool(_DATABRICKS_CELL_RE.match(basename))


def _is_internal_frame(filename: str) -> bool:
    """Return True if the filename looks like an internal/framework frame."""
    for marker in _INTERNAL_MARKERS:
        if marker in filename:
            return True
    return False


def _is_user_frame(filename: str) -> bool:
    """
    Return True if this frame is user code.

    Checks in priority order:
      1. Databricks cell frame → yes (even though path contains ipykernel)
      2. Synthetic <...> frames → no (unless it's a cell frame)
      3. Internal markers → no
      4. Everything else → yes
    """
    if _is_databricks_cell_frame(filename):
        return True
    if filename.startswith("<"):
        return False
    return not _is_internal_frame(filename)


# Public alias for use by _capture.py
is_databricks_cell_path = _is_databricks_cell_frame

# Cache of cell source text, keyed by cell temp path.
# Populated eagerly during frame walking (when linecache has the source).
# Read at serialization time by compute_cell_fingerprint.
_cell_source_cache: dict[str, str] = {}


def _ensure_cell_cached(cell_path: str) -> None:
    """
    Eagerly cache a Databricks cell's source from linecache.

    Called during frame walking (hot path) when we detect a cell frame.
    linecache is fast — it's a dict lookup in CPython. The cell source
    must be captured NOW because:
      - The temp file doesn't exist on disk (compiled in memory)
      - linecache entries may be evicted later
      - By serialization time, the source may be gone
    """
    if cell_path in _cell_source_cache:
        return
    try:
        import linecache
        lines = linecache.getlines(cell_path)
        if lines:
            _cell_source_cache[cell_path] = "".join(lines)
    except Exception:
        pass


def compute_cell_fingerprint(cell_path: str) -> Optional[str]:
    """
    Compute a SHA256 fingerprint of a Databricks cell's source.

    Uses the eagerly cached source from _ensure_cell_cached (populated
    during frame walking). Falls back to linecache and then disk read.

    Returns the first 16 hex chars of the SHA256, or None if the
    source can't be retrieved.
    """
    source = _cell_source_cache.get(cell_path)

    if source is None:
        # Fallback: try linecache directly
        try:
            import linecache
            lines = linecache.getlines(cell_path)
            if lines:
                source = "".join(lines)
                _cell_source_cache[cell_path] = source
        except Exception:
            pass

    if source is None:
        # Last resort: try disk (works for local testing)
        try:
            with open(cell_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception:
            return None

    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def detect_source_path(spark: SparkSession) -> Optional[str]:
    """
    Detect the source notebook or script path.

    Detection order:
      1. Databricks dbutils (via IPython namespace — works on Spark Connect)
      2. Databricks Spark config (classic clusters only)
      3. Caller's filename via inspect.stack()
      4. sys.argv[0] as last resort

    Returns the path as a string, or None if detection fails.
    """
    # Strategy 1: Databricks dbutils (most reliable on Spark Connect)
    path = _detect_databricks_dbutils_path()
    if path:
        return path

    # Strategy 2: Databricks Spark config (classic clusters)
    path = _detect_databricks_spark_conf_path(spark)
    if path:
        return path

    # Strategy 3: Walk the call stack to find the outermost user frame.
    path = _detect_caller_path()
    if path:
        return path

    # Strategy 4: sys.argv[0] (works for scripts invoked from CLI)
    if sys.argv and sys.argv[0] and sys.argv[0] != "-c":
        argv_path = os.path.abspath(sys.argv[0])
        if os.path.isfile(argv_path):
            return argv_path

    return None


def _detect_databricks_dbutils_path() -> Optional[str]:
    """
    Get notebook path from dbutils via IPython namespace.

    On Databricks, dbutils is injected into the notebook's IPython
    namespace. This works on both classic and Spark Connect runtimes.
    """
    try:
        ip = _get_ipython()
        if ip is None:
            return None
        dbutils = ip.user_ns.get("dbutils")
        if dbutils is None:
            return None
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        path = ctx.notebookPath().get()
        if path:
            return path
    except Exception:
        pass
    return None


def _get_ipython():
    """Get the IPython shell instance, or None if not in IPython."""
    try:
        from IPython import get_ipython as _ipy_get
        return _ipy_get()
    except Exception:
        return None


def _detect_databricks_spark_conf_path(spark: SparkSession) -> Optional[str]:
    """Try to get the notebook path from Databricks Spark config (classic only)."""
    for key in (
        "spark.databricks.notebook.path",
        "spark.databricks.clusterUsageTags.notebookPath",
    ):
        try:
            path = spark.conf.get(key)
            if path:
                return path
        except Exception:
            pass
    return None


def _detect_caller_path() -> Optional[str]:
    """
    Walk the call stack to find the outermost user script.

    Skips internal frames (our package, PySpark, IPython, etc.)
    and returns the filename of the outermost remaining frame.
    """
    try:
        stack = inspect.stack()
    except Exception:
        return None

    best: Optional[str] = None
    for frame_info in reversed(stack):
        filename = frame_info.filename
        if not filename:
            continue
        if not _is_user_frame(filename):
            continue
        # Don't use Databricks cell temp paths as source path —
        # those are meaningless outside the runtime. The Databricks
        # path comes from spark config instead.
        if _is_databricks_cell_frame(filename):
            continue
        if best is None:
            best = os.path.abspath(filename)

    return best


# ── Fast per-transformation line capture ─────────────────────────────────
#
# Called on EVERY DataFrame construction method (filter, join, select, etc.)
# so it must be fast. Uses sys._getframe() instead of traceback.extract_stack()
# to avoid allocating FrameSummary objects.


def get_current_user_line(
    source_path: Optional[str],
) -> Optional[Tuple[str, int]]:
    """
    Get the (filename, lineno) of the current user code site.

    Fast path using sys._getframe() — no allocations beyond the frame
    walk itself. Called from every patched construction method.

    Args:
        source_path: The known source path (cached from start()).
            Used as the normalized key when the actual frame filename
            is a Databricks cell temp path.

    Returns:
        (filename, lineno) tuple, or None if no user frame found.
        On Databricks, filename is normalized to source_path.
    """
    try:
        # Start 2 frames up: caller → patched method → this function
        frame = sys._getframe(2)
    except (ValueError, AttributeError):
        return None

    return _walk_frames(frame, source_path)


def _walk_frames(
    frame: Any,
    source_path: Optional[str],
) -> Optional[Tuple[str, int]]:
    """
    Walk frames to find the first user code frame.

    On Databricks, the cell frame is recognized by its "command-*"
    basename and the returned filename is normalized to source_path.
    For local scripts, the actual filename is returned.
    """
    source_stem: Optional[str] = None
    if source_path:
        source_stem = os.path.splitext(os.path.basename(source_path))[0]

    f = frame
    while f is not None:
        try:
            fn = f.f_code.co_filename
        except AttributeError:
            break

        # Priority 1: Databricks cell frame — keep raw path so
        # serialize_construction_lines can group by cell and fingerprint.
        # Eagerly cache the source from linecache NOW while the frame
        # is alive — by serialization time, the source may be gone.
        if _is_databricks_cell_frame(fn):
            _ensure_cell_cached(fn)
            return (fn, f.f_lineno)

        # Priority 2: Exact match to source_path
        if source_path and fn == source_path:
            return (fn, f.f_lineno)

        # Priority 3: Basename/stem match (workspace vs repo paths)
        if source_stem:
            frame_stem = os.path.splitext(os.path.basename(fn))[0]
            if frame_stem == source_stem and not _is_internal_frame(fn):
                return (fn, f.f_lineno)

        # Priority 4: Any non-internal, non-synthetic frame
        if fn and not fn.startswith("<") and not _is_internal_frame(fn):
            return (fn, f.f_lineno)

        f = f.f_back

    return None


# ── Trigger line extraction (action-time, full fidelity) ─────────────────

def extract_trigger_info(
    source_path: Optional[str],
) -> dict[str, Any]:
    """
    Extract the trigger line and call stack for the current plan capture.

    Called from within _on_plan_captured, which is called from within
    the monkey-patched action method. The stack at this point includes
    the full user call chain.

    Returns a dict with:
      - triggerLine: int or None — line number (cell-relative on Databricks)
      - triggerCellFingerprint: str or None — only for Databricks cells
      - triggerStack: list[dict] — compact stack trace (user frames only)
        Each entry has {line, function} plus either {file} or {cellFingerprint}
    """
    result: dict[str, Any] = {}

    try:
        frames = traceback.extract_stack()
    except Exception:
        return result

    user_frames = _extract_user_frames(frames)
    if user_frames:
        result["triggerStack"] = user_frames

    trigger = _find_trigger_line(frames, source_path)
    if trigger is not None:
        result["triggerLine"] = trigger["line"]
        if "cellFingerprint" in trigger:
            result["triggerCellFingerprint"] = trigger["cellFingerprint"]

    return result


def _extract_user_frames(
    frames: traceback.StackSummary,
) -> list[dict[str, Any]]:
    """
    Filter stack frames to just user code, in compact format.

    Databricks cell frames get a cellFingerprint (with source eagerly
    cached). Regular user frames get a file path.
    """
    user_frames: list[dict[str, Any]] = []
    for frame in frames:
        if _is_databricks_cell_frame(frame.filename):
            _ensure_cell_cached(frame.filename)
            fp = compute_cell_fingerprint(frame.filename)
            entry: dict[str, Any] = {
                "line": frame.lineno,
                "function": frame.name,
            }
            if fp:
                entry["cellFingerprint"] = fp
            user_frames.append(entry)
        elif _is_user_frame(frame.filename):
            user_frames.append({
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
            })
    return user_frames


def _find_trigger_line(
    frames: traceback.StackSummary,
    source_path: Optional[str],
) -> Optional[dict[str, Any]]:
    """
    Find the trigger frame info.

    Returns a dict with "line" and optionally "cellFingerprint",
    or None if no match found.
    """
    if not frames:
        return None

    source_stem: Optional[str] = None
    if source_path:
        source_stem = os.path.splitext(os.path.basename(source_path))[0]

    # Scan innermost-first for the best match
    for frame in reversed(frames):
        # Priority 1: Databricks cell frame
        if _is_databricks_cell_frame(frame.filename):
            _ensure_cell_cached(frame.filename)
            fp = compute_cell_fingerprint(frame.filename)
            result: dict[str, Any] = {"line": frame.lineno}
            if fp:
                result["cellFingerprint"] = fp
            return result

        # Priority 2: Exact match to source_path
        if source_path and frame.filename == source_path:
            return {"line": frame.lineno}

        # Priority 3: Absolute path match
        if source_path:
            try:
                if os.path.abspath(frame.filename) == os.path.abspath(source_path):
                    return {"line": frame.lineno}
            except (OSError, ValueError):
                pass

        # Priority 4: Basename match
        if source_stem:
            frame_stem = os.path.splitext(os.path.basename(frame.filename))[0]
            if frame_stem == source_stem and not _is_internal_frame(frame.filename):
                return {"line": frame.lineno}

    # Fallback: outermost user frame
    for frame in frames:
        if _is_user_frame(frame.filename):
            return {"line": frame.lineno}

    return None