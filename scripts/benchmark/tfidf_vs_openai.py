"""
Quick head-to-head: Char TF-IDF + RF  vs  OpenAI GPT-4o
Samples 15 random rows from the expanded test set.
"""

import json
import os
import random
import time
import joblib
from pathlib import Path

import openai

# ── paths ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[2]
TEST_JSONL  = BASE / "data/training/genos_residual_expanded/specialist_test_variant_a.jsonl"
TFIDF_MODEL = BASE / "models/specialist_tfidf_char_rf.pkl"
LABEL_MAP   = BASE / "config/specialist_map.json"

N_SAMPLES   = 15
SEED        = 42
MODEL_NAME  = "gpt-5.4"

# ── load ────────────────────────────────────────────────────────────────────
print("[+] Loading test data ...")
with open(TEST_JSONL) as f:
    rows = [json.loads(l) for l in f]

random.seed(SEED)
samples = random.sample(rows, N_SAMPLES)

print("[+] Loading TF-IDF model ...")
pipe = joblib.load(TFIDF_MODEL)
# pipe.classes_ are integer indices — need reverse mapping
with open(LABEL_MAP) as f:
    label_map = json.load(f)          # {"T1059": 0, ...}
idx_to_label = {v: k for k, v in label_map.items()}

# ── OpenAI client ────────────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY", "")
client  = openai.OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "You are a cybersecurity expert specializing in MITRE ATT&CK. "
    "Given a Windows/Linux command-line string, respond with ONLY the single "
    "most likely MITRE ATT&CK technique ID (e.g. T1059). "
    "No explanation. No punctuation. Just the T-code."
)

def openai_predict(raw_command: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": raw_command},
        ],
        temperature=0,
        max_completion_tokens=10,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    prediction = resp.choices[0].message.content.strip().upper()
    # normalise subtechnique (T1562.001 → T1562) to match ground truth format
    prediction = prediction.split(".")[0]
    return prediction, elapsed_ms

def tfidf_predict(input_text: str) -> tuple[str, str, float]:
    t0 = time.perf_counter()
    proba  = pipe.predict_proba([input_text])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    top3_idx   = proba.argsort()[-3:][::-1]
    top1_label = idx_to_label.get(int(pipe.classes_[top3_idx[0]]), "?")
    top3_labels= [idx_to_label.get(int(pipe.classes_[i]), "?") for i in top3_idx]
    conf       = proba[top3_idx[0]]
    return top1_label, top3_labels, conf, elapsed_ms

# ── run ──────────────────────────────────────────────────────────────────────
results = []
tfidf_correct = 0
oai_correct   = 0
oai_top3_correct = 0

print(f"\n{'#':>2}  {'Ground Truth':<10}  {'TF-IDF Top-1':<12}  {'Conf':>5}  {'TF-IDF Top-3':<32}  {'OpenAI':<10}  {'Raw Command (truncated)'}")
print("-" * 130)

for i, s in enumerate(samples, 1):
    gt          = s["label"].strip().upper()
    raw_cmd     = s["raw_command"]
    input_text  = s["input_text"]

    tfidf_top1, tfidf_top3, tfidf_conf, tfidf_ms = tfidf_predict(input_text)
    oai_pred, oai_ms = openai_predict(raw_cmd)

    t_ok  = "✓" if tfidf_top1 == gt else "✗"
    t3_ok = "✓" if gt in tfidf_top3 else "✗"
    o_ok  = "✓" if oai_pred == gt else "✗"
    o3_ok = "✓" if gt in tfidf_top3 else " "  # reuse tfidf top3 marker

    if tfidf_top1 == gt: tfidf_correct += 1
    if oai_pred   == gt: oai_correct   += 1
    if gt in tfidf_top3: oai_top3_correct += 1  # reusing var for top3 tfidf

    top3_str = ", ".join(tfidf_top3)
    print(f"{i:>2}  {gt:<10}  {t_ok}{tfidf_top1:<11}  {tfidf_conf:>4.0%}  {t3_ok}{top3_str:<31}  {o_ok}{oai_pred:<9}  {raw_cmd[:55]}")

    results.append({
        "ground_truth":  gt,
        "raw_command":   raw_cmd,
        "tfidf_top1":    tfidf_top1,
        "tfidf_top3":    tfidf_top3,
        "tfidf_conf":    round(float(tfidf_conf), 4),
        "tfidf_ms":      round(tfidf_ms, 2),
        "openai_pred":   oai_pred,
        "openai_ms":     round(oai_ms, 2),
        "tfidf_correct": tfidf_top1 == gt,
        "openai_correct": oai_pred  == gt,
    })

print("-" * 130)
print(f"\nSummary ({N_SAMPLES} samples, seed={SEED})")
print(f"  TF-IDF  Top-1 : {tfidf_correct}/{N_SAMPLES}  = {tfidf_correct/N_SAMPLES:.0%}")
print(f"  TF-IDF  Top-3 : {oai_top3_correct}/{N_SAMPLES}  = {oai_top3_correct/N_SAMPLES:.0%}")
print(f"  OpenAI  Top-1 : {oai_correct}/{N_SAMPLES}  = {oai_correct/N_SAMPLES:.0%}")

avg_tfidf_ms = sum(r["tfidf_ms"] for r in results) / N_SAMPLES
avg_oai_ms   = sum(r["openai_ms"] for r in results) / N_SAMPLES
print(f"\n  Avg latency TF-IDF : {avg_tfidf_ms:.1f}ms")
print(f"  Avg latency OpenAI : {avg_oai_ms:.1f}ms")

# save
out_path = BASE / "logs/tfidf_vs_openai.json"
with open(out_path, "w") as f:
    json.dump({"samples": results, "summary": {
        "n": N_SAMPLES,
        "tfidf_top1": tfidf_correct,
        "tfidf_top3": oai_top3_correct,
        "openai_top1": oai_correct,
        "avg_tfidf_ms": round(avg_tfidf_ms, 2),
        "avg_oai_ms":   round(avg_oai_ms, 2),
    }}, f, indent=2)
print(f"\n[+] Results saved to {out_path}")
