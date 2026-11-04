"""RunPod access with an ownership guard.

The account holds a pod this project did not create (``beowulf-bakeoff``). Every
mutating call therefore goes through :meth:`RunPodClient.require_owned`, which
refuses any pod id absent from this project's state file. The guard is the
enforcement point; callers must not talk to the API directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

API_BASE = "https://rest.runpod.io/v1"
STATE_PATH = Path("runs/pods.json")


class NotOwnedError(RuntimeError):
    """Raised when a mutating call targets a pod this project did not create."""


class RunPodClient:
    """Minimal RunPod REST client, scoped to pods this project owns."""

    def __init__(
        self,
        api_key: str,
        *,
        state_path: str | Path = STATE_PATH,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._state_path = Path(state_path)
        self._client = client or httpx.Client(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self._owns_client = client is None

    def __enter__(self) -> "RunPodClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- ownership state ---------------------------------------------------

    def owned_ids(self) -> set[str]:
        if not self._state_path.exists():
            return set()
        data = json.loads(self._state_path.read_text(encoding="utf-8"))
        return {entry["id"] for entry in data.get("pods", [])}

    def record(self, pod_id: str, name: str, cost_per_hr: float) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"pods": []}
        if self._state_path.exists():
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        data.setdefault("pods", []).append(
            {"id": pod_id, "name": name, "cost_per_hr": cost_per_hr}
        )
        self._state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def require_owned(self, pod_id: str) -> None:
        if pod_id not in self.owned_ids():
            raise NotOwnedError(
                f"refusing to modify pod {pod_id!r}: not created by this project. "
                f"Owned pods: {sorted(self.owned_ids()) or 'none'}"
            )

    # -- reads -------------------------------------------------------------

    def list_pods(self) -> list[dict]:
        response = self._client.get("/pods")
        response.raise_for_status()
        return response.json()

    def get_pod(self, pod_id: str) -> dict:
        response = self._client.get(f"/pods/{pod_id}")
        response.raise_for_status()
        return response.json()

    # -- guarded mutations -------------------------------------------------

    def stop_pod(self, pod_id: str) -> dict:
        self.require_owned(pod_id)
        response = self._client.post(f"/pods/{pod_id}/stop")
        response.raise_for_status()
        return response.json()

    def terminate_pod(self, pod_id: str) -> None:
        self.require_owned(pod_id)
        response = self._client.delete(f"/pods/{pod_id}")
        response.raise_for_status()
