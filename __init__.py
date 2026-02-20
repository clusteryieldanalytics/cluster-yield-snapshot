"""
cluster-yield-snapshot: Passive Spark plan capture.

Drop two lines at the top of any notebook, two at the bottom:

    from cluster_yield_snapshot import CYSnapshot
    cy = CYSnapshot(spark).start()

    # ═══ rest of notebook runs unchanged ═══

    cy.stop().save()

Or upload directly to Cluster Yield:

    cy = CYSnapshot(spark, api_key="cy_...", environment="prod").start()
    # ... notebook ...
    cy.stop().upload()
"""

from .snapshot import CYSnapshot, snapshot_capture, VERSION
from .upload import UploadError, UploadResult

__version__ = VERSION

__all__ = [
    "CYSnapshot",
    "snapshot_capture",
    "UploadError",
    "UploadResult",
    "__version__",
]
