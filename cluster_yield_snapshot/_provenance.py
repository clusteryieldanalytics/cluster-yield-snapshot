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
"""

from __future__ import annotations

import inspect
import os
import sys
import traceback
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


# ── Source path detection ────────────────────────────────────────────────

# Our own package directory — used to filter internal frames.
_OWN_PACKAGE_DIR = str(PurePosixPath(__file__).parent)

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
)


def detect_source_path(spark: SparkSession) -> Optional[str]:
    """
    Detect the source notebook or script path.

    Detection order:
      1. Databricks notebook path (from Spark config)
      2. Caller's filename via inspect.stack()
      3. sys.argv[0] as last resort

    Returns the path as a string, or None if detection fails.
    """
    # Strategy 1: Databricks notebook path
    path = _detect_databricks_path(spark)
    if path:
        return path

    # Strategy 2: Walk the call stack to find the outermost user frame.
    path = _detect_caller_path()
    if path:
        return path

    # Strategy 3: sys.argv[0] (works for scripts invoked from CLI)
    if sys.argv and sys.argv[0] and sys.argv[0] != "-c":
        argv_path = os.path.abspath(sys.argv[0])
        if os.path.isfile(argv_path):
            return argv_path

    return None


def _detect_databricks_path(spark: SparkSession) -> Optional[str]:
    """Try to get the notebook path from Databricks Spark config."""
    try:
        path = spark.conf.get("spark.databricks.notebook.path")
        if path:
            return path
    except Exception:
        pass

    try:
        path = spark.conf.get(
            "spark.databricks.clusterUsageTags.notebookPath"
        )
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
        if not filename or filename.startswith("<"):
            continue
        if _is_internal_frame(filename):
            continue
        if best is None:
            best = os.path.abspath(filename)

    return best


def _is_internal_frame(filename: str) -> bool:
    """Return True if the filename looks like an internal/framework frame."""
    for marker in _INTERNAL_MARKERS:
        if marker in filename:
            return True
    return False


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
            Used for priority matching. If None, returns the first
            non-internal frame.

    Returns:
        (filename, lineno) tuple, or None if no user frame found.
    """
    try:
        # Start 2 frames up: caller → patched method → this function
        frame = sys._getframe(2)
    except (ValueError, AttributeError):
        return None

    # If we have a source_path, look for it specifically first
    if source_path:
        result = _walk_for_source(frame, source_path)
        if result is not None:
            return result

    # Fallback: first non-internal frame
    return _walk_for_any_user(frame)


def _walk_for_source(
    frame: Any,
    source_path: str,
) -> Optional[Tuple[str, int]]:
    """Walk frames looking for one matching source_path."""
    source_basename = os.path.basename(source_path)
    source_stem = os.path.splitext(source_basename)[0]

    f = frame
    while f is not None:
        try:
            fn = f.f_code.co_filename
        except AttributeError:
            break

        # Exact match
        if fn == source_path:
            return (fn, f.f_lineno)

        # Basename/stem match (workspace path vs repo path)
        if source_stem:
            frame_stem = os.path.splitext(os.path.basename(fn))[0]
            if frame_stem == source_stem and not _is_internal_frame(fn):
                return (fn, f.f_lineno)

        # Databricks <command-N> frames
        if (fn.startswith("<command")
                and source_path.startswith(("/Workspace/", "/Repos/"))):
            return (source_path, f.f_lineno)

        f = f.f_back

    return None


def _walk_for_any_user(frame: Any) -> Optional[Tuple[str, int]]:
    """Walk frames looking for the first non-internal frame."""
    f = frame
    while f is not None:
        try:
            fn = f.f_code.co_filename
        except AttributeError:
            break

        if fn and not fn.startswith("<") and not _is_internal_frame(fn):
            return (fn, f.f_lineno)

        # Also accept Databricks <command-N> frames
        if fn and fn.startswith("<command"):
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
      - triggerLine: int or None — line number in the source file
      - triggerStack: list[dict] — compact stack trace (user frames only)
    """
    result: dict[str, Any] = {}

    try:
        frames = traceback.extract_stack()
    except Exception:
        return result

    user_frames = _extract_user_frames(frames)
    if user_frames:
        result["triggerStack"] = user_frames

    trigger_line = _find_trigger_line(frames, source_path)
    if trigger_line is not None:
        result["triggerLine"] = trigger_line

    return result


def _extract_user_frames(
    frames: traceback.StackSummary,
) -> list[dict[str, Any]]:
    """Filter stack frames to just user code, in compact format."""
    user_frames: list[dict[str, Any]] = []
    for frame in frames:
        if _is_internal_frame(frame.filename):
            continue
        if frame.filename.startswith("<") and "command" not in frame.filename:
            continue
        user_frames.append({
            "file": frame.filename,
            "line": frame.lineno,
            "function": frame.name,
        })
    return user_frames


def _find_trigger_line(
    frames: traceback.StackSummary,
    source_path: Optional[str],
) -> Optional[int]:
    """Find the line number in the source file that triggered plan execution."""
    if not frames:
        return None

    if source_path:
        match = _match_source_frame(frames, source_path)
        if match is not None:
            return match

    # Fallback: outermost user frame
    outermost_line: Optional[int] = None
    for frame in frames:
        if _is_internal_frame(frame.filename):
            continue
        if frame.filename.startswith("<") and "command" not in frame.filename:
            continue
        if outermost_line is None:
            outermost_line = frame.lineno

    return outermost_line


def _match_source_frame(
    frames: traceback.StackSummary,
    source_path: str,
) -> Optional[int]:
    """Find the frame matching source_path, scanning innermost-first."""
    source_basename = os.path.basename(source_path)
    source_stem = os.path.splitext(source_basename)[0]

    for frame in reversed(frames):
        if _is_internal_frame(frame.filename):
            continue

        if frame.filename == source_path:
            return frame.lineno

        try:
            if os.path.abspath(frame.filename) == os.path.abspath(source_path):
                return frame.lineno
        except (OSError, ValueError):
            pass

        frame_basename = os.path.basename(frame.filename)
        frame_stem = os.path.splitext(frame_basename)[0]
        if frame_stem and frame_stem == source_stem:
            return frame.lineno

        if (frame.filename.startswith("<command")
                and source_path.startswith(("/Workspace/", "/Repos/"))):
            return frame.lineno

    return None