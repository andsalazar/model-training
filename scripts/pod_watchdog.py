"""Watch a pod's run from this machine; copy the results back; terminate the pod.

A long run on a rented GPU must not depend on someone remembering to stop it. This runs
locally, polls the pod over SSH, and ends in one of three ways:

* the run's ``DONE`` marker appears: copy the results, terminate the pod;
* the run's log shows ``ABORT``: copy what exists, terminate the pod;
* the deadline passes: copy what exists, terminate the pod, whatever state it is in.

So the pod's cost is bounded by the deadline even if nothing else is watching. Small
result files are also copied on every poll, so a partial run is not lost if the pod dies.

Termination goes through ``wsp.runpod.RunPodClient``, which refuses any pod this project
did not create (``runs/pods.json``), and is confirmed by reading the pod back. The key
comes from ``RUNPOD_MODEL_TRAINING_API_KEY`` (use ``doppler run``) and is never printed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: Copied on every poll (small). Paths are relative to the remote work directory.
SMALL = ["eval/eval.json", "adherence.json", "audio_mp4.json", "noise.json", "gpu.txt",
         "ovi_commit.txt", "seeds_done.txt", "teacher_log.json", "run.log", "DONE"]


def decide(*, done: bool, aborted_for: float | None, now: float, deadline: float, grace: float) -> str:
    """What to do after a poll: ``finish``, ``abort``, ``deadline`` or ``wait``.

    ``aborted_for`` is how long the log has shown an abort, or ``None``. An aborted run is
    given ``grace`` seconds to be fixed and relaunched (which starts a new log) before the
    pod is terminated; the deadline applies throughout.
    """
    if done:
        return "finish"
    if now >= deadline:
        return "deadline"
    if aborted_for is not None and aborted_for >= grace:
        return "abort"
    return "wait"


class Pod:
    def __init__(self, host: str, port: int, key: Path) -> None:
        self.host, self.port = host, port
        self.options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
                        "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-i", str(key)]

    def run(self, command: str, timeout: int = 120) -> tuple[int, str]:
        try:
            out = subprocess.run(["ssh", *self.options, "-p", str(self.port), f"root@{self.host}", command],
                                 capture_output=True, text=True, timeout=timeout)
            return out.returncode, out.stdout
        except (subprocess.TimeoutExpired, OSError) as error:
            return 255, f"{type(error).__name__}"

    def fetch(self, remote: str, local: Path, timeout: int = 600) -> bool:
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            out = subprocess.run(["scp", *self.options, "-P", str(self.port), "-q",
                                  f"root@{self.host}:{remote}", str(local)],
                                 capture_output=True, text=True, timeout=timeout)
            return out.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


def log(path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def copy_small(pod: Pod, remote_dir: str, local_dir: Path) -> list[str]:
    copied = []
    for name in SMALL:
        target = local_dir / ("eval.json" if name == "eval/eval.json" else name)
        if pod.fetch(f"{remote_dir}/{name}", target, timeout=120):
            copied.append(name)
    return copied


def copy_clips(pod: Pod, remote_dir: str, local_dir: Path, pattern: str) -> bool:
    """Tar the clips matching ``pattern`` on the pod and bring the archive back."""
    code, _ = pod.run(f"cd {remote_dir} && tar cf /tmp/clips.tar $(ls {pattern} 2>/dev/null) 2>/dev/null; "
                      "ls -la /tmp/clips.tar", timeout=300)
    return code == 0 and pod.fetch("/tmp/clips.tar", local_dir / "clips.tar", timeout=1500)


def terminate(pod_id: str, log_path: Path) -> bool:
    import httpx

    from wsp.runpod import RunPodClient

    key = os.environ.get("RUNPOD_MODEL_TRAINING_API_KEY")
    if not key:
        log(log_path, "CANNOT TERMINATE: RUNPOD_MODEL_TRAINING_API_KEY is not set")
        return False
    with RunPodClient(key) as client:
        for attempt in range(1, 6):
            try:
                client.terminate_pod(pod_id)
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 404:
                    log(log_path, f"terminate attempt {attempt}: HTTP {error.response.status_code}")
            except httpx.HTTPError as error:
                log(log_path, f"terminate attempt {attempt}: {type(error).__name__}")
            time.sleep(10)
            try:
                client.get_pod(pod_id)
                log(log_path, f"terminate attempt {attempt}: pod still exists")
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404:
                    log(log_path, f"TERMINATED {pod_id}: read back 404")
                    return True
            except httpx.HTTPError as error:
                log(log_path, f"read-back attempt {attempt}: {type(error).__name__}")
    log(log_path, f"FAILED TO CONFIRM TERMINATION of {pod_id}: terminate it by hand")
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--remote-dir", default="/local/v4_decide")
    p.add_argument("--local-dir", type=Path, default=ROOT / "artifacts" / "v4_decide")
    p.add_argument("--deadline", required=True, help="local time, ISO format: terminate at this time regardless")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--abort-grace", type=float, default=30.0,
                   help="minutes an aborted run may sit before the pod is terminated")
    p.add_argument("--clips", default="teacher/*_s100.mp4 eval/*_s100.mp4",
                   help="clips to bring back before terminating (shell patterns under the work dir)")
    p.add_argument("--key-path", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    p.add_argument("--log", type=Path, default=ROOT / "runs" / "watchdog.log")
    args = p.parse_args(argv)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    pod = Pod(args.host, args.port, args.key_path)
    log(args.log, f"watching {args.pod_id} at {args.host}:{args.port}; deadline {args.deadline}")
    abort_seen: float | None = None
    while True:
        code, out = pod.run(f"test -f {args.remote_dir}/DONE && echo __DONE__; "
                            f"grep -c ABORT {args.remote_dir}/run.log 2>/dev/null; "
                            f"tail -n 1 {args.remote_dir}/run.log 2>/dev/null | cut -c1-160")
        reachable = code != 255
        done = "__DONE__" in out
        lines = [x for x in out.replace("__DONE__", "").splitlines() if x.strip()]
        aborted = bool(lines) and lines[0].strip().isdigit() and int(lines[0]) > 0
        if reachable:
            abort_seen = (abort_seen or time.time()) if aborted else None
        action = decide(done=done, aborted_for=None if abort_seen is None else time.time() - abort_seen,
                        now=time.time(), deadline=deadline, grace=args.abort_grace * 60)
        copied = copy_small(pod, args.remote_dir, args.local_dir) if reachable else []
        log(args.log, f"{action}: reachable={reachable} copied={len(copied)} last={lines[-1][:120] if lines else ''!r}")
        if action == "wait":
            time.sleep(min(args.interval, max(5.0, deadline - time.time())))
            continue
        if reachable and action == "finish" and args.clips:
            log(args.log, f"clips copied: {copy_clips(pod, args.remote_dir, args.local_dir, args.clips)}")
        ok = terminate(args.pod_id, args.log)
        log(args.log, f"exit: {action}, terminated={ok}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
