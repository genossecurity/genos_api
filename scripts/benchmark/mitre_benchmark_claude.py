"""
MITRE technique accuracy benchmark — Claude Opus 4.7 only.
Uses the same corpus & sampling as mitre_benchmark.py (one sample per class).
Appends Claude results to logs/mitre_benchmark.json.
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[2]
TEST_JSONL = BASE / "data/training/genos_residual_expanded/specialist_test_variant_a.jsonl"
LABEL_MAP  = BASE / "config/specialist_map.json"
OUTPUT     = BASE / "logs/mitre_benchmark.json"

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")

if not ANTHROPIC_KEY:
    sys.exit("Set ANTHROPIC_API_KEY env var")

# ── Load data ─────────────────────────────────────────────────────────────────
print("[+] Loading test data ...")
with open(TEST_JSONL) as f:
    all_rows = [json.loads(l) for l in f]
print(f"    {len(all_rows)} rows loaded")

print("[+] Loading label map ...")
with open(LABEL_MAP) as f:
    label_map = json.load(f)
covered_classes = set(label_map.keys())

# One sample per class
seen = set()
rows = []
for r in all_rows:
    lbl = r["label"]
    if lbl in covered_classes and lbl not in seen:
        rows.append(r)
        seen.add(lbl)
n = len(rows)
print(f"    Selected {n} rows (one per class)")

# ── Client ────────────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

SYSTEM_PROMPT = (
    "You are a cybersecurity expert specializing in MITRE ATT&CK. "
    "Given a command-line string, respond with the top 3 most likely "
    "MITRE ATT&CK technique IDs, comma-separated, most likely first. "
    "Example: T1059,T1105,T1071\n"
    "No explanation. No punctuation. Just the three T-codes."
)

def claude_predict(command: str):
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=30,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": command}],
        )
        ms = (time.perf_counter() - t0) * 1000
        raw = resp.content[0].text.strip().upper()
        codes = [c.split(".")[0].strip() for c in raw.split(",") if c.strip().startswith("T")]
        return codes[:3], round(ms, 1)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        print(f"    [ERR] {e}")
        return [], round(ms, 1)

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\n[+] Benchmarking {n} commands with {ANTHROPIC_MODEL} ...\n")
print(f"{'#':>4}  {'GT':<8} {'Claude':^8}  Command")
print("-" * 70)

results = []
c_top1 = c_top3 = 0
c_ms = 0.0

for i, row in enumerate(rows, 1):
    gt  = row["label"]
    cmd = row["raw_command"]

    preds, ms = claude_predict(cmd)

    t1 = preds[0] == gt if preds else False
    t3 = gt in preds     if preds else False

    if t1: c_top1 += 1
    if t3: c_top3 += 1
    c_ms += ms

    sym = "✓" if t1 else "✗"
    print(f"[{i:>3}/{n}]  {gt:<8} {sym}{preds[0] if preds else '-':<7}  {cmd[:55]}")

    results.append({
        "id": i, "gt": gt, "command": cmd,
        "claude_top3": preds,
        "claude_top1_correct": t1,
        "claude_top3_correct": t3,
        "claude_ms": ms,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  MITRE BENCHMARK — {ANTHROPIC_MODEL}")
print(f"  Corpus: {n} commands (one per class)")
print(f"  Date  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print(f"\n  Top-1 : {c_top1}/{n} = {c_top1/n:.1%}")
print(f"  Top-3 : {c_top3}/{n} = {c_top3/n:.1%}")
print(f"  Avg ms: {c_ms/n:.1f}")
print()

# ── Append to existing mitre_benchmark.json ───────────────────────────────────
if OUTPUT.exists():
    with open(OUTPUT) as f:
        data = json.load(f)
else:
    data = {}

data["claude"] = {
    "top1": c_top1 / n,
    "top3": c_top3 / n,
    "avg_ms": c_ms / n,
    "model": ANTHROPIC_MODEL,
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
}
data["claude_results"] = results

with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)
print(f"[+] Results appended to {OUTPUT}")
