"""Tests for the pod watchdog's decision rule and its ownership-guarded termination."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_watchdog.py"


def _load():
    spec = importlib.util.spec_from_file_location("pod_watchdog", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_watchdog"] = module
    spec.loader.exec_module(module)
    return module


dog = _load()


class TestDecide:
    def test_done_finishes_even_past_the_deadline(self):
        assert dog.decide(done=True, aborted_for=None, now=200, deadline=100, grace=60) == "finish"

    def test_deadline_terminates_a_run_that_is_still_going(self):
        assert dog.decide(done=False, aborted_for=None, now=100, deadline=100, grace=60) == "deadline"

    def test_an_abort_gets_a_grace_period(self):
        assert dog.decide(done=False, aborted_for=10, now=0, deadline=100, grace=60) == "wait"
        assert dog.decide(done=False, aborted_for=60, now=0, deadline=100, grace=60) == "abort"

    def test_the_deadline_applies_during_the_grace_period(self):
        assert dog.decide(done=False, aborted_for=10, now=100, deadline=100, grace=60) == "deadline"

    def test_otherwise_wait(self):
        assert dog.decide(done=False, aborted_for=None, now=0, deadline=100, grace=60) == "wait"


class TestTerminate:
    def _client(self, monkeypatch, tmp_path, handler, owned):
        from wsp import runpod

        state = tmp_path / "pods.json"
        state.write_text(json.dumps({"pods": [{"id": i} for i in owned]}))
        real = runpod.RunPodClient

        def build(key):
            http = httpx.Client(base_url=runpod.API_BASE, transport=httpx.MockTransport(handler))
            return real(key, state_path=state, client=http)

        monkeypatch.setattr(runpod, "RunPodClient", build)
        monkeypatch.setenv("RUNPOD_MODEL_TRAINING_API_KEY", "test-key")
        monkeypatch.setattr(dog.time, "sleep", lambda _: None)

    def test_deletes_then_confirms_by_reading_back(self, monkeypatch, tmp_path):
        calls = []

        def handler(request):
            calls.append((request.method, request.url.path))
            return httpx.Response(204 if request.method == "DELETE" else 404)

        self._client(monkeypatch, tmp_path, handler, owned=["mine"])
        assert dog.terminate("mine", tmp_path / "log.txt") is True
        assert calls == [("DELETE", "/v1/pods/mine"), ("GET", "/v1/pods/mine")]
        assert "read back 404" in (tmp_path / "log.txt").read_text()

    def test_a_pod_that_survives_is_reported_not_assumed_gone(self, monkeypatch, tmp_path):
        handler = lambda request: httpx.Response(200, json={"id": "mine"})  # noqa: E731
        self._client(monkeypatch, tmp_path, handler, owned=["mine"])
        assert dog.terminate("mine", tmp_path / "log.txt") is False
        assert "FAILED TO CONFIRM" in (tmp_path / "log.txt").read_text()

    def test_refuses_a_pod_this_project_did_not_create(self, monkeypatch, tmp_path):
        from wsp.runpod import NotOwnedError

        calls = []

        def handler(request):
            calls.append(request.method)
            return httpx.Response(204)

        self._client(monkeypatch, tmp_path, handler, owned=["mine"])
        with pytest.raises(NotOwnedError):
            dog.terminate("someone-elses", tmp_path / "log.txt")
        assert calls == []

    def test_without_a_key_nothing_is_called(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RUNPOD_MODEL_TRAINING_API_KEY", raising=False)
        assert dog.terminate("mine", tmp_path / "log.txt") is False
        assert "CANNOT TERMINATE" in (tmp_path / "log.txt").read_text()
