"""
MITRE technique accuracy benchmark:
  - Genos /scan/free  (top-1, top-3)
  - GPT-5.4           (top-1)
  - TF-IDF char RF    (top-1, top-3)
Corpus: specialist_test_variant_a.jsonl  (829 rows, ground-truth T-codes)
Output: logs/mitre_benchmark.json + summary printed to stdout
"""

import json
import os
import sys
import time
import joblib
from collections import defaultdict
from pathlib import Path

import httpx
import openai

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[2]
TEST_JSONL = BASE / "data/training/genos_residual_expanded/specialist_test_variant_a.jsonl"
TFIDF_MODEL= BASE / "models/specialist_tfidf_char_rf.pkl"
LABEL_MAP  = BASE / "config/specialist_map.json"
OUTPUT     = BASE / "logs/mitre_benchmark.json"
GENOS_URL  = "http://127.0.0.1:6001/scan/free"
MODEL_NAME = "gpt-5.4"
API_KEY    = os.environ.get("OPENAI_API_KEY", "")

# ── Load data ─────────────────────────────────────────────────────────────────
print("[+] Loading test data ...")
with open(TEST_JSONL) as f:
    all_rows = [json.loads(l) for l in f]
print(f"    {len(all_rows)} rows loaded")

# ── Load label map first so we can filter ─────────────────────────────────────
print("[+] Loading label map ...")
with open(LABEL_MAP) as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
covered_classes = set(label_map.keys())

# One sample per class — guarantees all 147 classes are tested
seen = set()
rows = []
for r in all_rows:
    lbl = r["label"]
    if lbl in covered_classes and lbl not in seen:
        rows.append(r)
        seen.add(lbl)
n = len(rows)
missing = covered_classes - seen
print(f"    Selected {n} rows (one per class). Missing from test set: {sorted(missing) or 'none'}")

# ── Load TF-IDF ───────────────────────────────────────────────────────────────
print("[+] Loading TF-IDF model ...")
pipe = joblib.load(TFIDF_MODEL)

# ── Clients ───────────────────────────────────────────────────────────────────
oai_client = openai.OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are a cybersecurity expert specializing in MITRE ATT&CK. "
    "Given a command-line string, respond with the top 3 most likely "
    "MITRE ATT&CK technique IDs, comma-separated, most likely first. "
    "Example: T1059,T1105,T1071\n"
    "No explanation. No punctuation. Just the three T-codes."
)

# ── Predict functions ─────────────────────────────────────────────────────────
def tfidf_predict(input_text: str):
    t0 = time.perf_counter()
    proba = pipe.predict_proba([input_text])[0]
    ms = (time.perf_counter() - t0) * 1000
    top3_idx = proba.argsort()[-3:][::-1]
    top3 = [idx_to_label.get(int(pipe.classes_[i]), "?") for i in top3_idx]
    return top3, round(ms, 1)

def genos_predict(command: str):
    t0 = time.perf_counter()
    try:
        r = httpx.post(GENOS_URL, json={"command": command}, timeout=15)
        ms = (time.perf_counter() - t0) * 1000
        codes = [m["code"] for m in r.json().get("MITRE_codes", [])[:3]]
        return codes, round(ms, 1)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return [], round(ms, 1)

def gpt_predict(command: str):
    t0 = time.perf_counter()
    try:
        resp = oai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": command},
            ],
            temperature=0,
            max_completion_tokens=30,
        )
        ms = (time.perf_counter() - t0) * 1000
        raw = resp.choices[0].message.content.strip().upper()
        codes = [c.split(".")[0].strip() for c in raw.split(",") if c.strip().startswith("T")]
        return codes[:3], round(ms, 1)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return [], round(ms, 1)

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"[+] Benchmarking {n} commands ...\n")
print(f"{'#':>4}  {'GT':<8} {'TF-IDF':^8} {'Genos':^8} {'GPT':^8}  Command")
print("-" * 90)

results = []
tf_top1 = tf_top3 = g_top1 = g_top3 = gpt_top1 = gpt_top3 = 0
tf_ms = g_ms = gpt_ms = 0.0

for i, row in enumerate(rows, 1):
    gt    = row["label"]
    cmd   = row["raw_command"]
    itxt  = row["input_text"]

    tf_preds, tf_t  = tfidf_predict(itxt)
    g_preds,  g_t   = genos_predict(cmd)
    gpt_preds, gpt_t = gpt_predict(cmd)

    tf_t1  = tf_preds[0] == gt  if tf_preds  else False
    tf_t3  = gt in tf_preds      if tf_preds  else False
    g_t1   = g_preds[0] == gt   if g_preds   else False
    g_t3   = gt in g_preds       if g_preds   else False
    gpt_t1 = gpt_preds[0] == gt if gpt_preds else False
    gpt_t3 = gt in gpt_preds     if gpt_preds else False

    if tf_t1:  tf_top1  += 1
    if tf_t3:  tf_top3  += 1
    if g_t1:   g_top1   += 1
    if g_t3:   g_top3   += 1
    if gpt_t1: gpt_top1 += 1
    if gpt_t3: gpt_top3 += 1
    tf_ms  += tf_t
    g_ms   += g_t
    gpt_ms += gpt_t

    tf_sym  = "✓" if tf_t1 else "✗"
    g_sym   = "✓" if g_t1  else "✗"
    gpt_sym = "✓" if gpt_t1 else "✗"
    print(f"[{i:>3}/{n}]  {gt:<8} {tf_sym}{tf_preds[0] if tf_preds else '-':<7} "
          f"{g_sym}{g_preds[0] if g_preds else '-':<7} "
          f"{gpt_sym}{gpt_preds[0] if gpt_preds else '-':<7}  {cmd[:50]}")

    results.append({
        "id": i, "gt": gt, "command": cmd,
        "tfidf_top3": tf_preds, "tfidf_top1_correct": tf_t1, "tfidf_top3_correct": tf_t3,
        "genos_top3": g_preds,  "genos_top1_correct": g_t1,  "genos_top3_correct": g_t3,
        "gpt_top3":   gpt_preds, "gpt_top1_correct": gpt_t1, "gpt_top3_correct": gpt_t3,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  MITRE TECHNIQUE BENCHMARK")
print(f"  Corpus: {TEST_JSONL.name}  ({n} commands)")
print(f"  Date  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print(f"\n{'System':<20} {'Top-1':>8} {'Top-3':>8} {'Avg Lat':>10}")
print("-" * 50)
print(f"{'TF-IDF':<20} {tf_top1/n:>7.1%} {tf_top3/n:>7.1%} {tf_ms/n:>8.1f}ms")
print(f"{'Genos (/scan/free)':<20} {g_top1/n:>7.1%} {g_top3/n:>7.1%} {g_ms/n:>8.1f}ms")
print(f"{'GPT-5.4':<20} {gpt_top1/n:>7.1%} {gpt_top3/n:>7.1%} {gpt_ms/n:>8.1f}ms")
print()

# ── Per-technique breakdown ───────────────────────────────────────────────────
by_tech = defaultdict(lambda: {"n": 0, "tf": 0, "g": 0, "gpt": 0})
for r in results:
    t = r["gt"]
    by_tech[t]["n"] += 1
    if r["tfidf_top1_correct"]: by_tech[t]["tf"] += 1
    if r["genos_top1_correct"]: by_tech[t]["g"]  += 1
    if r["gpt_top1_correct"]:   by_tech[t]["gpt"] += 1

print(f"\nPer-technique top-1 accuracy (techniques with ≥5 samples):")
print(f"  {'Technique':<10} {'n':>4} {'TF-IDF':>8} {'Genos':>8} {'GPT':>8}")
print("  " + "-" * 44)
for tech in sorted(by_tech, key=lambda t: -by_tech[t]["n"]):
    d = by_tech[tech]
    if d["n"] < 5:
        continue
    print(f"  {tech:<10} {d['n']:>4} {d['tf']/d['n']:>7.1%} {d['g']/d['n']:>7.1%} {d['gpt']/d['n']:>7.1%}")

print("=" * 70)

# ── Save ───────────────────────────────────────────────────────────────────────
OUTPUT.parent.mkdir(exist_ok=True)
summary = {
    "n": n, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tfidf":  {"top1": tf_top1/n,  "top3": tf_top3/n,  "avg_ms": tf_ms/n},
    "genos":  {"top1": g_top1/n,   "top3": g_top3/n,   "avg_ms": g_ms/n},
    "gpt54":  {"top1": gpt_top1/n, "avg_ms": gpt_ms/n},
    "per_technique": {t: d for t, d in by_tech.items()},
    "results": results,
}
with open(OUTPUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[+] Results saved to {OUTPUT}")
