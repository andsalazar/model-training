"""Tests for the RunPod ownership guard.

The account holds a pod this project did not create. These tests pin the
guarantee that no mutating call can reach it.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from wsp.runpod import NotOwnedError, RunPodClient

FOREIGN_POD = "lisq494fny3g98"


def _client(tmp_path: Path, handler) -> RunPodClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(
        base_url="https://rest.runpod.io/v1",
        headers={"Authorization": "Bearer test"},
        transport=transport,
    )
    return RunPodClient("test", state_path=tmp_path / "pods.json", client=http)


def test_owned_ids_empty_when_no_state(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, json=[]))
    assert client.owned_ids() == set()


def test_require_owned_rejects_foreign_pod(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, json=[]))
    with pytest.raises(NotOwnedError, match="not created by this project"):
        client.require_owned(FOREIGN_POD)


def test_stop_pod_refuses_foreign_pod_without_network_call(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    client = _client(tmp_path, handler)
    with pytest.raises(NotOwnedError):
        client.stop_pod(FOREIGN_POD)

    assert calls == [], "guard must reject before any request is sent"


def test_terminate_pod_refuses_foreign_pod_without_network_call(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    client = _client(tmp_path, handler)
    with pytest.raises(NotOwnedError):
        client.terminate_pod(FOREIGN_POD)

    assert calls == []


def test_record_then_stop_allows_owned_pod(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"id": "pod-abc"})

    client = _client(tmp_path, handler)
    client.record("pod-abc", "wsp-stage0", 1.99)

    assert client.owned_ids() == {"pod-abc"}
    client.stop_pod("pod-abc")
    assert calls == ["/v1/pods/pod-abc/stop"]


def test_record_appends_without_dropping_existing(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, json={}))
    client.record("pod-1", "first", 1.0)
    client.record("pod-2", "second", 2.0)

    data = json.loads((tmp_path / "pods.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in data["pods"]] == ["pod-1", "pod-2"]
