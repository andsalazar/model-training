"""Repair characters that a PowerShell stdin pipe turned into '?'.

Context rules (genuine question marks follow a letter or closing quote and are kept):
  ' ? '          -> ' — '   (em dash)
  digit?digit    -> digit–digit (range)
  digit? <non-digit> -> digit×  (multiplier)
  '(?'+digit     -> '(≈'
  ' ? ' between 'river'/'x' and a digit in tables -> ' × '
"""
import re, sys, pathlib

FILES = ["docs/stage6d-attention-contrast.md", "docs/stage6e-review-corrections.md",
         "docs/stage7-long30.md", "docs/stage7-wsp-adapter.md", "README.md"]
apply = "--apply" in sys.argv
RULES = [
    (re.compile(r"(river|seeds?|blocks?|heads?) \? (\d)"), r"\1 × \2"),
    (re.compile(r"\(\?(0\.05)"), r"(≥\1"),
    (re.compile(r"blocks \?(\d)"), r"blocks ≥\1"),
    (re.compile(r"\(\?(\d)"), r"(≈\1"),
    (re.compile(r"(\d)\?(\d)"), r"\1–\2"),
    (re.compile(r"(\d)\?(?=[\s,;:.)|*/]|$)"), r"\1×"),
    (re.compile(r"\|\?"), "|Δ"),
    (re.compile(r", \?, "), ", …, "),
    (re.compile(r" \? "), " — "),
    (re.compile(r" \?\n"), " —\n"),
    (re.compile(r"^\? ", re.M), "— "),
]
for name in FILES:
    path = pathlib.Path(name)
    text = path.read_text(encoding="utf-8")
    new = text
    for rx, rep in RULES:
        new = rx.sub(rep, new)
    left = [m.start() for m in re.finditer(r"\?", new)]
    changed = sum(1 for a, b in zip(text, new) if a != b)
    print(f"{name}: {changed} chars changed; remaining '?':")
    for pos in left:
        print("    ", ascii(new[max(0, pos - 40):pos + 25]))
    if apply:
        path.write_bytes(new.encode("utf-8"))