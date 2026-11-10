"""Stage 8: can a language model tell what a network does from its raw weights?

A replication of Experiment 2 of *Sparks of Weight-Space Programming*: a reader gets the
23,824 weights of a small diffusion model as text, with no tools, and is asked what images
the model generates. See ``wsp.reading`` and ``docs/stage8-reading-weights.md``.

Three stages, run locally (no GPU):

``build``  copy the paper's three networks, train fresh ones with the same recipe
           (new seeds, and one shape the paper did not use), make a shuffled control,
           check that every trained network really generates its shape, write a manifest.
``ask``    send each network's prompt to each reader through OpenRouter and save the reply.
           Needs ``OPENROUTER_API_KEY``; stops before the spend limit is crossed.
``score``  apply the fixed keyword rule to every reply and write ``scores.json``.
``quality`` re-check what each network generates, without retraining.

The answer key lives in ``manifest.json``. ``ask`` reads only network ids from it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wsp import reading  # noqa: E402

#: Fresh networks: (shape, training seed). The paper's own networks are seed 0.
FRESH = [("circle", 1), ("square", 1), ("triangle", 1), ("cross", 1), ("cross", 2)]
#: The control is the first fresh network with its weights shuffled inside each tensor.
CONTROL_OF = ("circle", 1)
DATASET_SIZE = 20_000
QUALITY_SAMPLES = 1000
#: Readers, cheapest first, so a fault shows up before the expensive calls.
DEFAULT_EDITORS = ["deepseek", "glm", "qwen", "kimi", "astra"]
MAX_TOKENS = 16_000
#: A reply cut off by the limit before it reaches its ``Images:`` section is asked for
#: once more with this limit. The cut-off reply is kept as ``<id>.truncated.json``.
RETRY_MAX_TOKENS = 64_000
REASONING = {"effort": "high"}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_net(directory: Path, network_id: str) -> dict:
    with np.load(directory / "networks" / f"{network_id}.npz") as z:
        return {name: z[name] for name in reading.TENSORS}


# --------------------------------------------------------------------------- build


def train_classifier(seed: int = 0):
    """A small classifier over the four shapes, to check what a network generates."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    xs, ys = [], []
    for label, shape in enumerate(reading.SHAPES):
        images = reading.make_dataset(shape, 12_000, seed=20_000 + label)
        xs.append(images)
        ys.append(np.full(len(images), label))
    x, y = torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))
    order = torch.randperm(len(x))
    x, y = x[order], y[order]
    held = len(x) // 8
    model = nn.Sequential(nn.Linear(reading.PIXELS, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, len(reading.SHAPES)))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(60):
        for i in range(held, len(x), 256):
            loss = nn.functional.cross_entropy(model(x[i:i + 256]), y[i:i + 256])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        accuracy = float((model(x[:held]).argmax(1) == y[:held]).float().mean())

    def classify(images: np.ndarray) -> dict:
        with torch.no_grad():
            predicted = model(torch.from_numpy(images.astype(np.float32))).argmax(1).numpy()
        return {s: float((predicted == i).mean()) for i, s in enumerate(reading.SHAPES)}

    return classify, accuracy


def quality(net: dict, classify, seed: int = 0) -> dict:
    samples = reading.sample(net, QUALITY_SAMPLES, seed=seed)
    return {
        # A forced choice among the four shapes: meaningless for a network that draws none.
        "classified_as": classify(samples),
        # How much the generated images differ from each other; near 0 means one fixed image.
        "pixel_std_across_samples": float(samples.std(axis=0).mean()),
        "mean_pixel": float(samples.mean()),
        "frac_outside_range": float((np.abs(samples) > 1.0001).mean()),
    }


def stage_quality(args) -> int:
    """Re-check what every network in the manifest generates, without retraining any."""
    path = args.dir / "manifest.json"
    manifest = json.loads(path.read_text())
    classify, accuracy = train_classifier()
    manifest["classifier_accuracy_real"] = accuracy
    for trial in manifest["trials"]:
        trial["quality"] = quality(load_net(args.dir, trial["id"]), classify)
        print(f"{trial['id']} {trial['source']:8} {trial['truth']!s:9} {trial['quality']['classified_as']}",
              flush=True)
    write_json(path, manifest)
    print(f"classifier held-out accuracy on real images: {accuracy:.4f}")
    return 0


def stage_build(args) -> int:
    out: Path = args.dir
    (out / "networks").mkdir(parents=True, exist_ok=True)
    classify, accuracy = train_classifier()
    print(f"shape classifier, held-out accuracy on real images: {accuracy:.4f}", flush=True)
    trials = []

    paper = args.paper_logs / "exp2_diffusion"
    key = json.loads((paper / "eval" / "answer_key.json").read_text())
    for network_id, info in sorted(key.items()):
        with np.load(paper / "networks" / f"{network_id}.npz") as z:
            net = {name: z[name] for name in reading.TENSORS}
        prompt = reading.build_prompt(net)
        theirs = (paper / "runs" / info["shape"] / "prompt.txt").read_text(encoding="utf-8")
        if prompt != theirs:
            raise RuntimeError(f"{network_id}: rebuilt prompt differs from the paper's prompt.txt")
        np.savez(out / "networks" / f"{network_id}.npz", **net)
        q = quality(net, classify)
        trials.append({"id": network_id, "source": "paper", "truth": info["shape"], "seed": info["seed"],
                       "prompt_sha256": sha256(prompt), "prompt_chars": len(prompt), "quality": q,
                       "paper_reported_intended": info["quality"]["intended"]})
        print(f"paper {network_id} {info['shape']}: {q['classified_as']}", flush=True)

    control_source = None
    for shape, seed in FRESH:
        started = time.time()
        images = reading.make_dataset(shape, DATASET_SIZE, seed=1000 + seed)
        net, curve = reading.train_ddpm(images, steps=args.steps, seed=seed)
        network_id = reading.network_id(net)
        np.savez(out / "networks" / f"{network_id}.npz", **net)
        prompt = reading.build_prompt(net)
        q = quality(net, classify)
        final_loss = curve[-1]["loss"] if curve else None
        trials.append({"id": network_id, "source": "fresh", "truth": shape, "seed": seed,
                       "prompt_sha256": sha256(prompt), "prompt_chars": len(prompt), "quality": q,
                       "final_loss": final_loss, "train_seconds": round(time.time() - started, 1)})
        print(f"fresh {network_id} {shape} s{seed}: loss {final_loss}, "
              f"{q['classified_as']}, {time.time() - started:.0f}s", flush=True)
        if (shape, seed) == CONTROL_OF:
            control_source = (network_id, net)

    source_id, source = control_source
    shuffled = reading.shuffle_within_tensors(source, seed=0)
    network_id = reading.network_id(shuffled)
    np.savez(out / "networks" / f"{network_id}.npz", **shuffled)
    prompt = reading.build_prompt(shuffled)
    trials.append({"id": network_id, "source": "control", "truth": None, "shuffled_from": source_id,
                   "prompt_sha256": sha256(prompt), "prompt_chars": len(prompt),
                   "quality": quality(shuffled, classify)})
    print(f"control {network_id}: shuffled from {source_id}", flush=True)

    write_json(out / "manifest.json", {
        "classifier_accuracy_real": accuracy,
        "recipe": {"steps": args.steps, "batch": 512, "lr": 1e-3, "dataset_size": DATASET_SIZE,
                   "hidden": reading.HIDDEN},
        # Sorted by id, so the order says nothing about the shapes.
        "trials": sorted(trials, key=lambda t: t["id"]),
    })
    print(f"wrote {out / 'manifest.json'} ({len(trials)} networks)")
    return 0


# --------------------------------------------------------------------------- ask


def live_prices(model_ids: list[str]) -> dict[str, tuple[float, float]]:
    """USD per token (input, output) from OpenRouter's public model list."""
    import httpx

    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    by_id = {m["id"]: m.get("pricing") or {} for m in data}
    return {m: (float(by_id[m]["prompt"]), float(by_id[m]["completion"])) for m in model_ids if m in by_id}


class Budget:
    """Refuses a call whose worst case would cross the limit."""

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.spent = 0.0
        self.reserved = 0.0

    def reserve(self, worst: float) -> bool:
        if self.spent + self.reserved + worst > self.limit:
            return False
        self.reserved += worst
        return True

    def settle(self, worst: float, actual: float) -> None:
        self.reserved -= worst
        self.spent += actual


def cut_off(record: dict) -> bool:
    """Whether a reply hit the output limit before giving an ``Images:`` section."""
    return (record.get("finish_reason") == "length"
            and reading.answer_section(record.get("answer") or "", "Images") is None)


async def ask_one(editor, alias: str, trial_id: str, prompt: str, price: tuple[float, float],
                  budget: Budget, out_path: Path) -> str:
    truncated_path = out_path.with_suffix(".truncated.json")
    attempts = [RETRY_MAX_TOKENS] if truncated_path.exists() else [MAX_TOKENS, RETRY_MAX_TOKENS]
    total = 0.0
    for max_tokens in attempts:
        # About 1.6 characters per token for these prompts (the paper: 99,800 tokens for
        # 158,000 characters). 1.2 leaves room for a tokenizer that splits numbers further.
        worst = len(prompt) / 1.2 * price[0] + max_tokens * price[1]
        if not budget.reserve(worst):
            return "skipped (spend limit)"
        started = time.time()
        try:
            response = await editor.complete(
                reading.SYSTEM, prompt, temperature=None, max_tokens=max_tokens,
                extra={"reasoning": REASONING, "usage": {"include": True}},
            )
        except Exception as error:  # noqa: BLE001 - record the failure, do not lose the run
            budget.settle(worst, 0.0)
            write_json(out_path.with_suffix(".error.json"),
                       {"editor": alias, "id": trial_id, "error": f"{type(error).__name__}: {error}"[:2000]})
            return f"error: {type(error).__name__}"
        body = response.raw or {}
        usage = body.get("usage") or {}
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        computed = response.usage.prompt_tokens * price[0] + response.usage.completion_tokens * price[1]
        cost = usage.get("cost")
        actual = float(cost) if isinstance(cost, (int, float)) else computed
        budget.settle(worst, actual)
        total += actual
        record = {
            "editor": alias, "id": trial_id, "model_requested": editor.model_id, "model": response.model,
            "answer": message.get("content") or "", "reasoning": message.get("reasoning"),
            "finish_reason": choice.get("finish_reason"),
            "tool_calls": message.get("tool_calls"),
            "usage": usage, "cost_usd": actual,
            "cost_source": "api" if isinstance(cost, (int, float)) else "computed",
            "seconds": round(time.time() - started, 1), "prompt_sha256": sha256(prompt),
            "request": {"max_tokens": max_tokens, "reasoning": REASONING, "temperature": None,
                        "system": reading.SYSTEM},
            "asked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if cut_off(record) and max_tokens != RETRY_MAX_TOKENS:
            write_json(truncated_path, record)
            continue
        record["retried_after_cut_off"] = truncated_path.exists()
        write_json(out_path, record)
        # A transport error from an earlier run is superseded once a reply exists.
        out_path.with_suffix(".error.json").unlink(missing_ok=True)
        return f"ok, ${total:.3f}, {record['seconds']:.0f}s" + (" (retried)" if record["retried_after_cut_off"] else "")
    return "unreachable"


async def run_ask(args) -> int:
    from wsp.config import Settings
    from wsp.editors.openrouter import OpenRouterEditor
    from wsp.editors.registry import get_spec

    settings = Settings.from_env()
    manifest = json.loads((args.dir / "manifest.json").read_text())
    ids = [t["id"] for t in manifest["trials"]]
    if args.only:
        ids = [i for i in ids if i in args.only]
    specs = {alias: get_spec(alias) for alias in args.editors}
    prices = live_prices([s.model_id for s in specs.values()])
    budget = Budget(args.max_usd)
    gate = asyncio.Semaphore(args.concurrency)

    async def worker(alias: str) -> None:
        spec = specs[alias]
        price = prices.get(spec.model_id, (spec.usd_per_mtok_in / 1e6, spec.usd_per_mtok_out / 1e6))
        editor = OpenRouterEditor(spec.model_id, settings.openrouter_api_key, timeout=args.timeout,
                                  max_retries=2)
        try:
            for trial_id in ids:
                out_path = args.dir / "answers" / alias / f"{trial_id}.json"
                if out_path.exists():
                    continue
                prompt = reading.build_prompt(load_net(args.dir, trial_id))
                async with gate:
                    status = await ask_one(editor, alias, trial_id, prompt, price, budget, out_path)
                print(f"{alias:9} {trial_id}  {status}  (spent ${budget.spent:.2f})", flush=True)
        finally:
            await editor.aclose()

    await asyncio.gather(*(worker(alias) for alias in args.editors))
    print(f"spent this run: ${budget.spent:.2f} of a ${args.max_usd:.2f} limit")
    return 0


# --------------------------------------------------------------------------- score


def stage_score(args) -> int:
    manifest = json.loads((args.dir / "manifest.json").read_text())
    trials = {t["id"]: t for t in manifest["trials"]}
    rows, summary = [], {}
    for editor_dir in sorted(p for p in (args.dir / "answers").iterdir() if p.is_dir()):
        alias = editor_dir.name
        counts = {"paper": [0, 0], "fresh_same": [0, 0], "fresh_novel": [0, 0]}
        control, cost, missing = [], 0.0, []
        for trial_id, trial in trials.items():
            path = editor_dir / f"{trial_id}.json"
            if not path.exists():
                missing.append(trial_id)
                continue
            reply = json.loads(path.read_text(encoding="utf-8"))
            score = reading.score_answer(reply["answer"], trial["truth"])
            runs = reading.quoted_runs(reply["answer"])
            net = load_net(args.dir, trial_id) if runs else None
            score["quoted_runs"] = len(runs)
            score["quoted_runs_found"] = sum(reading.locate_run(net, r) is not None for r in runs)
            cost += reply.get("cost_usd") or 0.0
            first = path.with_suffix(".truncated.json")
            if first.exists():
                cost += json.loads(first.read_text(encoding="utf-8")).get("cost_usd") or 0.0
            group = ("control" if trial["source"] == "control" else "paper" if trial["source"] == "paper"
                     else "fresh_same" if trial["truth"] in reading.PAPER_SHAPES else "fresh_novel")
            rows.append({"editor": alias, "id": trial_id, "group": group, "truth": trial["truth"], **score,
                         "finish_reason": reply.get("finish_reason"), "tool_free": not reply.get("tool_calls"),
                         "retried_after_cut_off": bool(reply.get("retried_after_cut_off")),
                         "prompt_tokens": (reply.get("usage") or {}).get("prompt_tokens"),
                         "completion_tokens": (reply.get("usage") or {}).get("completion_tokens"),
                         "cost_usd": reply.get("cost_usd")})
            if group == "control":
                control.append({"guess": score["guess"], "confidence": score["confidence"]})
            else:
                counts[group][0] += bool(score["correct"])
                counts[group][1] += 1
        mine = [r for r in rows if r["editor"] == alias]
        summary[alias] = {**{k: {"correct": v[0], "n": v[1]} for k, v in counts.items()},
                          "control": control, "cost_usd": round(cost, 4), "missing": missing,
                          "no_images_section": sum(r["guess"] == "missing" for r in mine),
                          "quoted_runs": sum(r["quoted_runs"] for r in mine),
                          "quoted_runs_found": sum(r["quoted_runs_found"] for r in mine)}
    write_json(args.dir / "scores.json", {"summary": summary, "rows": rows})
    print(f"{'reader':10}{'paper nets':>12}{'fresh, same':>13}{'fresh, cross':>14}  control (guess, confidence)   cost")
    for alias, s in summary.items():
        cells = [f"{s[k]['correct']}/{s[k]['n']}" for k in ("paper", "fresh_same", "fresh_novel")]
        ctl = ", ".join(f"{c['guess']} {c['confidence']}" for c in s["control"]) or "-"
        print(f"{alias:10}{cells[0]:>12}{cells[1]:>13}{cells[2]:>14}  {ctl:28} ${s['cost_usd']:.2f}"
              f"  no answer {s['no_images_section']}/9  quotes found {s['quoted_runs_found']}/{s['quoted_runs']}"
              + (f"  not asked {len(s['missing'])}" if s["missing"] else ""))
    print("\nper answer:")
    for r in rows:
        mark = "-" if r["correct"] is None else "yes" if r["correct"] else "NO"
        print(f"  {r['editor']:9} {r['id']} {r['group']:12} truth {r['truth']!s:9} guess {r['guess']:10} "
              f"{mark:4} conf {r['confidence']}  {r['images_line'][:90]!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dir", type=Path, default=ROOT / "artifacts" / "stage8_reading")
    b.add_argument("--paper-logs", type=Path, default=ROOT / "runs" / "paper-logs")
    b.add_argument("--steps", type=int, default=40_000)
    a = sub.add_parser("ask")
    a.add_argument("--dir", type=Path, default=ROOT / "artifacts" / "stage8_reading")
    a.add_argument("--editors", nargs="+", default=DEFAULT_EDITORS)
    a.add_argument("--only", nargs="+", default=None, help="network ids to ask about (default: all)")
    a.add_argument("--max-usd", type=float, required=True, help="stop before spending more than this")
    a.add_argument("--concurrency", type=int, default=4)
    a.add_argument("--timeout", type=float, default=900.0)
    s = sub.add_parser("score")
    s.add_argument("--dir", type=Path, default=ROOT / "artifacts" / "stage8_reading")
    q = sub.add_parser("quality")
    q.add_argument("--dir", type=Path, default=ROOT / "artifacts" / "stage8_reading")
    args = p.parse_args(argv)
    if args.stage == "quality":
        return stage_quality(args)
    if args.stage == "build":
        return stage_build(args)
    if args.stage == "ask":
        return asyncio.run(run_ask(args))
    return stage_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
