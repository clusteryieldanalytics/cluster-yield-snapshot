"""
Upload snapshots to the Cluster Yield SaaS backend.

Handles authentication, retries, and response parsing.
Requires the `upload` extra: pip install cluster-yield-snapshot[upload]
"""

from __future__ import annotations

import json
from typing import Any, Optional

DEFAULT_BASE_URL = "https://api.clusteryield.com"


class UploadError(Exception):
    """Raised when a snapshot upload fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class UploadResult:
    """Result of a successful snapshot upload."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def snapshot_id(self) -> str:
        return self._data.get("snapshotId", "")

    @property
    def findings_count(self) -> int:
        return self._data.get("findingsCount", 0)

    @property
    def estimated_monthly_cost(self) -> Optional[float]:
        return self._data.get("estimatedMonthlyCost")

    @property
    def dashboard_url(self) -> Optional[str]:
        return self._data.get("dashboardUrl")

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        parts = [f"snapshot_id={self.snapshot_id!r}"]
        if self.findings_count:
            parts.append(f"findings={self.findings_count}")
        if self.estimated_monthly_cost is not None:
            parts.append(f"est_cost=${self.estimated_monthly_cost:,.0f}/mo")
        return f"UploadResult({', '.join(parts)})"


def upload_snapshot(
    snapshot: dict[str, Any],
    *,
    api_key: str,
    environment: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
) -> UploadResult:
    """
    Upload a snapshot to the Cluster Yield backend.

    POST /api/v1/snapshots
    Authorization: Bearer <api_key>
    Content-Type: application/json

    The server analyzes on ingest: parses plans, runs detectors,
    computes cost estimates, diffs against previous snapshot for the
    same environment, and stores everything.

    Args:
        snapshot: The snapshot dict (from CYSnapshot.to_dict())
        api_key: Cluster Yield API key (cy_...)
        environment: Environment label (e.g. "prod-analytics")
        base_url: API base URL (override for self-hosted)
        timeout: Request timeout in seconds

    Returns:
        UploadResult with snapshot_id, findings summary, and dashboard URL

    Raises:
        UploadError: On HTTP errors or network failures
        ImportError: If httpx is not installed
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for upload. "
            "Install with: pip install cluster-yield-snapshot[upload]"
        )

    # Inject environment into the snapshot envelope
    payload = dict(snapshot)
    payload.setdefault("snapshot", {})["environment"] = environment

    url = f"{base_url.rstrip('/')}/api/v1/snapshots"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"cluster-yield-snapshot/{snapshot.get('snapshot', {}).get('version', '?')}",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                content=json.dumps(payload, default=str),
                headers=headers,
            )
    except httpx.TimeoutException:
        raise UploadError(f"Upload timed out after {timeout}s")
    except httpx.ConnectError as e:
        raise UploadError(f"Could not connect to {base_url}: {e}")
    except httpx.HTTPError as e:
        raise UploadError(f"HTTP error: {e}")

    if response.status_code == 401:
        raise UploadError("Invalid API key", status_code=401)
    if response.status_code == 413:
        raise UploadError("Snapshot too large", status_code=413)
    if response.status_code >= 400:
        detail = ""
        try:
            detail = response.json().get("error", response.text[:200])
        except Exception:
            detail = response.text[:200]
        raise UploadError(
            f"Server error {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    try:
        data = response.json()
    except Exception:
        raise UploadError("Invalid response from server")

    return UploadResult(data)


def upload_snapshot_urllib(
    snapshot: dict[str, Any],
    *,
    api_key: str,
    environment: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
) -> UploadResult:
    """
    Upload using only stdlib (urllib). Fallback when httpx is not available.

    Same interface as upload_snapshot but without connection pooling or
    retry niceties.
    """
    import urllib.request
    import urllib.error

    payload = dict(snapshot)
    payload.setdefault("snapshot", {})["environment"] = environment

    url = f"{base_url.rstrip('/')}/api/v1/snapshots"
    body = json.dumps(payload, default=str).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"cluster-yield-snapshot/{snapshot.get('snapshot', {}).get('version', '?')}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return UploadResult(data)
    except urllib.error.HTTPError as e:
        raise UploadError(
            f"Server error {e.code}: {e.read().decode()[:200]}",
            status_code=e.code,
        )
    except urllib.error.URLError as e:
        raise UploadError(f"Could not connect to {base_url}: {e.reason}")
