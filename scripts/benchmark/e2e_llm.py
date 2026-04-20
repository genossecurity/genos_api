"""
End-to-end benchmark: Genos (/scan/free) vs GPT-5.4
Evaluates: label accuracy (Benign/Suspicious/Malicious) + MITRE technique predictions
Corpus: test_commands_graded.txt  (difficulty|label|command, ~711 rows)
Output: logs/e2e-llm.txt
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import httpx
import openai

# ── Config ───────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parents[2]
CORPUS       = BASE / "scripts/benchmark/3class/test_commands_graded.txt"
OUTPUT       = BASE / "logs/e2e-llm.txt"
GENOS_URL    = "http://127.0.0.1:6001/scan/free"
MODEL_NAME   = "gpt-5.4"
API_KEY      = os.environ.get("OPENAI_API_KEY", "")

LABEL_MAP = {"benign": "Benign", "suspicious": "Suspicious", "malicious": "Malicious"}

SYSTEM_PROMPT = """\
You are a security analysis engine.

Analyze the command below and return ONLY minified JSON with this exact schema:

{{"label":"Benign|Suspicious|Malicious","label_confidence":0,"MITRE_codes":[{{"code":"Txxxx","confidence":0}}]}}

Rules:
- Return JSON only — no markdown, no extra keys
- label must be exactly "Benign", "Suspicious", or "Malicious"
- label_confidence is 0-100
- MITRE_codes: up to 3 ranked technique predictions, each with a MITRE code and confidence (0-100)
- MITRE_codes must be an empty array for Benign classifications
- Focus on command behavior only

Command:
{command}"""

# ── Load corpus ───────────────────────────────────────────────────────────────
def load_commands(path):
    cases = []
    known_diff   = {"easy", "medium", "hard", "unspecified"}
    known_labels = {"benign", "suspicious", "malicious"}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) == 3 and parts[0].lower() in known_diff and parts[1].lower() in known_labels:
                cases.append({"difficulty": parts[0].lower(),
                               "label": parts[1].lower(),
                               "command": parts[2].strip()})
    return cases

# ── Genos call ────────────────────────────────────────────────────────────────
def genos_predict(command: str):
    t0 = time.perf_counter()
    try:
        r = httpx.post(GENOS_URL, json={"command": command}, timeout=15)
        ms = (time.perf_counter() - t0) * 1000
        data = r.json()
        label = data.get("canonical_label") or data.get("label", "error")
        # normalise Context_Dependent → Suspicious
        if label == "Context_Dependent":
            label = "Suspicious"
        mitre = [m["code"] for m in data.get("MITRE_codes", [])[:3]]
        return label, mitre, round(ms, 1)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return f"error:{e}", [], round(ms, 1)

# ── OpenAI call ───────────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=API_KEY)

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            return json.loads(text[s:e+1])
    raise ValueError("no JSON found")

def openai_predict(command: str):
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": SYSTEM_PROMPT.format(command=command)}],
            temperature=0,
            max_completion_tokens=200,
        )
        ms = (time.perf_counter() - t0) * 1000
        data = extract_json(resp.choices[0].message.content)
        label = data.get("label", "error")
        if label not in ("Benign", "Suspicious", "Malicious"):
            label = "error:" + label
        # normalise subtechniques T1562.001 → T1562
        mitre = [m["code"].split(".")[0].upper() for m in data.get("MITRE_codes", [])[:3]]
        return label, mitre, round(ms, 1)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return f"error:{e}", [], round(ms, 1)

# ── Run ───────────────────────────────────────────────────────────────────────
cases = load_commands(CORPUS)
n = len(cases)
print(f"[+] Loaded {n} commands from corpus")
print(f"[+] Running Genos + GPT-{MODEL_NAME.replace('gpt-','')} on {n} commands ...\n")
print(f"{'#':>4}  {'GT':<12} {'Genos Label':<14} {'GPT Label':<14} {'Genos MITRE':<22} {'GPT MITRE':<22} Command")
print("-" * 130)

results = []
genos_ms_total = oai_ms_total = 0
genos_correct = oai_correct = 0
mitre_agree = 0  # cases where top-1 MITRE matches between systems

genos_by_diff  = defaultdict(lambda: [0, 0])
oai_by_diff    = defaultdict(lambda: [0, 0])
genos_by_label = defaultdict(lambda: [0, 0])
oai_by_label   = defaultdict(lambda: [0, 0])

for i, case in enumerate(cases, 1):
    cmd  = case["command"]
    gt   = LABEL_MAP[case["label"]]
    diff = case["difficulty"]

    g_label, g_mitre, g_ms = genos_predict(cmd)
    o_label, o_mitre, o_ms = openai_predict(cmd)

    g_ok = g_label == gt
    o_ok = o_label == gt
    top1_agree = bool(g_mitre and o_mitre and g_mitre[0] == o_mitre[0])

    if g_ok: genos_correct += 1
    if o_ok: oai_correct   += 1
    if top1_agree: mitre_agree += 1
    genos_ms_total += g_ms
    oai_ms_total   += o_ms

    genos_by_diff[diff][1]       += 1;  oai_by_diff[diff][1]       += 1
    genos_by_label[case["label"]][1] += 1;  oai_by_label[case["label"]][1] += 1
    if g_ok: genos_by_diff[diff][0] += 1;  genos_by_label[case["label"]][0] += 1
    if o_ok: oai_by_diff[diff][0]   += 1;  oai_by_label[case["label"]][0]   += 1

    gm = "✓" if g_ok else "✗"
    om = "✓" if o_ok else "✗"
    g_mitre_str = ",".join(g_mitre) if g_mitre else "-"
    o_mitre_str = ",".join(o_mitre) if o_mitre else "-"
    print(f"[{i:>3}/{n}]  {gt:<12} {gm}{g_label:<13} {om}{o_label:<13} {g_mitre_str:<22} {o_mitre_str:<22} {cmd[:45]}")

    results.append({
        "id": i, "difficulty": diff, "ground_truth": gt, "command": cmd,
        "genos_label": g_label, "genos_mitre": g_mitre, "genos_ms": g_ms, "genos_correct": g_ok,
        "openai_label": o_label, "openai_mitre": o_mitre, "openai_ms": o_ms, "openai_correct": o_ok,
        "mitre_top1_agree": top1_agree,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
non_benign = sum(1 for r in results if r["ground_truth"] != "Benign")
lines = []
def ln(s=""): lines.append(s)

ln("=" * 80)
ln("  END-TO-END BENCHMARK: Genos vs GPT-5.4")
ln(f"  Corpus : {CORPUS.name}  ({n} commands)")
ln(f"  Date   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
ln("=" * 80)
ln()
ln(f"{'System':<20} {'Label Acc':>10} {'Avg Latency':>13}")
ln("-" * 45)
ln(f"{'Genos':<20} {genos_correct/n:>9.1%} {genos_ms_total/n:>11.1f}ms")
ln(f"{'GPT-5.4':<20} {oai_correct/n:>9.1%} {oai_ms_total/n:>11.1f}ms")
ln()
ln(f"MITRE Top-1 Agreement (non-benign): {mitre_agree}/{non_benign} = {mitre_agree/max(non_benign,1):.1%}")
ln()
ln("Label Accuracy by Difficulty:")
ln(f"  {'Difficulty':<12} {'Genos':>8} {'GPT-5.4':>10}")
ln("  " + "-" * 32)
for d in sorted(genos_by_diff):
    gc, gt_ = genos_by_diff[d]; oc, ot_ = oai_by_diff[d]
    ln(f"  {d:<12} {gc/gt_:>7.1%}  {oc/ot_:>8.1%}  (n={gt_})")
ln()
ln("Label Accuracy by Ground Truth Label:")
ln(f"  {'Label':<12} {'Genos':>8} {'GPT-5.4':>10}")
ln("  " + "-" * 32)
for lab in ["benign", "suspicious", "malicious"]:
    if lab in genos_by_label:
        gc, gt_ = genos_by_label[lab]; oc, ot_ = oai_by_label[lab]
        ln(f"  {lab:<12} {gc/gt_:>7.1%}  {oc/ot_:>8.1%}  (n={gt_})")
ln()
ln("=" * 80)

summary_text = "\n".join(lines)
print("\n" + summary_text)

OUTPUT.parent.mkdir(exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write(summary_text + "\n\n")
    f.write("Per-command results:\n")
    f.write(f"{'#':>4}  {'Diff':<10} {'GT':<12} {'Genos Label':<16} {'GPT Label':<16} {'Genos MITRE':<20} {'GPT MITRE':<20} Command\n")
    f.write("-" * 130 + "\n")
    for r in results:
        gm = "✓" if r["genos_correct"] else "✗"
        om = "✓" if r["openai_correct"] else "✗"
        gms = ",".join(r["genos_mitre"]) if r["genos_mitre"] else "-"
        oms = ",".join(r["openai_mitre"]) if r["openai_mitre"] else "-"
        f.write(f"{r['id']:>4}  {r['difficulty']:<10} {r['ground_truth']:<12} "
                f"{gm}{r['genos_label']:<15} {om}{r['openai_label']:<15} "
                f"{gms:<20} {oms:<20} {r['command'][:50]}\n")

print(f"\n[+] Results saved to {OUTPUT}")
