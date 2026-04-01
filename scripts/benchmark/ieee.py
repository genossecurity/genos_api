import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)

# ============================================================
# CONFIG
# ============================================================

SEED = 42
MAX_LENGTH = 256

BENCHMARK_DIR = Path(__file__).resolve().parent
BASE_DIR = BENCHMARK_DIR.parent.parent
DATA_DIR = BASE_DIR / "data" / "training"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COMMAND_COL = "command"
MITRE_COL = "mitre_id"

sys.path.insert(0, str(BASE_DIR))

try:
    from engine import GenosEngine
except ImportError:
    print("[!] ERROR: Could not import 'GenosEngine'. Run from scripts/benchmark/ directory.")
    sys.exit(1)

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# DATA LOADING
# ============================================================

def _read_csv(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None

    df = pd.read_csv(path)
    if COMMAND_COL not in df.columns:
        raise ValueError(f"{path} missing required column: {COMMAND_COL}")

    df[COMMAND_COL] = df[COMMAND_COL].astype(str).fillna("")
    if MITRE_COL in df.columns:
        df[MITRE_COL] = df[MITRE_COL].astype(str).fillna("")

    return df


def load_data() -> Dict[str, Optional[pd.DataFrame]]:
    return {
        "gk_test": _read_csv(DATA_DIR / "gatekeeper_test.csv", required=True),   # benign
        "sp_train": _read_csv(DATA_DIR / "specialist_train.csv", required=True), # malicious train
        "sp_test": _read_csv(DATA_DIR / "specialist_test.csv", required=True),   # malicious test
    }

# ============================================================
# BASELINE
# ============================================================

def topk_hit_rate(y_true: List, y_topk: List[np.ndarray]) -> float:
    if not y_true:
        return 0.0
    hits = 0
    for i in range(len(y_true)):
        if y_true[i] in y_topk[i]:
            hits += 1
    return hits / len(y_true)


def macro_f1(y_true: List, y_pred: List) -> float:
    return precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]


def train_baseline(train_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1,
    )

    X_train = vectorizer.fit_transform(train_df[COMMAND_COL].astype(str))
    y_train = train_df[MITRE_COL].astype(str)
    clf.fit(X_train, y_train)
    return vectorizer, clf


def evaluate_baseline(vectorizer, clf, test_df: pd.DataFrame) -> Dict[str, float]:
    y_true = test_df[MITRE_COL].astype(str).tolist()
    y_pred_top1 = []
    y_pred_top3 = []
    latencies_ms = []

    classes = clf.classes_

    for cmd in tqdm(test_df[COMMAND_COL].astype(str).tolist(), desc="Baseline inference"):
        start = time.perf_counter()

        x = vectorizer.transform([cmd])
        y_prob = clf.predict_proba(x)[0]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        top1_idx = int(np.argmax(y_prob))
        top3_idx = np.argsort(y_prob)[-3:]

        y_pred_top1.append(classes[top1_idx])
        y_pred_top3.append(classes[top3_idx])

    return {
        "top1_acc": accuracy_score(y_true, y_pred_top1),
        "top3_acc": topk_hit_rate(y_true, y_pred_top3),
        "macro_f1": macro_f1(y_true, y_pred_top1),
        "lat_ms": float(np.mean(latencies_ms)),
    }

# ============================================================
# PIPELINE HELPERS (DEPLOYMENT-ALIGNED)
# ============================================================

def preprocess_command(engine: GenosEngine, raw_cmd: str) -> Dict[str, object]:
    """
    Mirrors the engine's preprocessing logic:
    - is_obfuscated
    - recursive deobfuscation up to engine.max_deobfuscation_layers
    - lower/strip normalization before tokenization
    """
    original_cmd = str(raw_cmd).strip()
    current_cmd = original_cmd

    was_obfuscated = engine.is_obfuscated(current_cmd)
    prev_entropy = engine.calculate_entropy(current_cmd)

    deobf_start = time.perf_counter()

    for _ in range(engine.max_deobfuscation_layers):
        if engine.is_obfuscated(current_cmd):
            new_cmd = engine.deobfuscate_layer(current_cmd)
            if new_cmd == current_cmd:
                break
            current_cmd = new_cmd

            new_entropy = engine.calculate_entropy(current_cmd)
            if abs(prev_entropy - new_entropy) < 0.01:
                break
            prev_entropy = new_entropy
        else:
            break

    deobf_time_ms = (time.perf_counter() - deobf_start) * 1000.0
    processed_cmd = current_cmd.lower().strip()
    changed = processed_cmd != original_cmd.lower().strip()

    return {
        "original_cmd": original_cmd,
        "processed_cmd": processed_cmd,
        "was_obfuscated": bool(was_obfuscated),
        "was_changed": bool(changed),
        "deobf_time_ms": deobf_time_ms,
    }


def tokenize_command(engine: GenosEngine, processed_cmd: str):
    return engine.tokenizer(
        processed_cmd,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=getattr(engine, "max_length", MAX_LENGTH),
    ).to(engine.device)


def infer_tier1(engine: GenosEngine, processed_cmd: str) -> Dict[str, float]:
    inputs = tokenize_command(engine, processed_cmd)
    with torch.no_grad():
        logits = engine.t1(inputs["input_ids"], inputs["attention_mask"])
        probs = F.softmax(logits, dim=1).squeeze(0)

    benign_prob = float(probs[0].item())
    malicious_prob = float(probs[1].item())
    pred_idx = int(torch.argmax(probs).item())

    return {
        "benign_prob": benign_prob,
        "malicious_prob": malicious_prob,
        "pred_idx": pred_idx,
        "pred_label": 1 if pred_idx == 1 else 0,  # deployed argmax behavior
    }


def infer_tier2(engine: GenosEngine, processed_cmd: str, topk: int = 3) -> Dict[str, object]:
    inputs = tokenize_command(engine, processed_cmd)
    with torch.no_grad():
        logits = engine.t2(inputs["input_ids"], inputs["attention_mask"])
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_vals, top_idxs = torch.topk(probs, k=min(topk, len(probs)), largest=True, sorted=True)
    top_idxs_np = top_idxs.cpu().numpy()
    top_codes = [engine.s_map[int(i)] for i in top_idxs_np]

    return {
        "top1_idx": int(top_idxs_np[0]),
        "topk_idxs": top_idxs_np,
        "topk_codes": top_codes,
        "topk_probs": [float(v.item()) for v in top_vals],
    }

# ============================================================
# TIER 1 EVALUATION
# ============================================================

def evaluate_tier1_pipeline(
    engine: GenosEngine,
    benign_df: pd.DataFrame,
    malicious_df: pd.DataFrame,
) -> Dict[str, object]:
    y_true = []
    y_scores = []
    y_pred = []

    preproc_times = []
    obf_flags = []
    changed_flags = []

    all_rows = (
        [(0, cmd) for cmd in benign_df[COMMAND_COL].tolist()] +
        [(1, cmd) for cmd in malicious_df[COMMAND_COL].tolist()]
    )

    for label, cmd in tqdm(all_rows, desc="Tier 1 pipeline eval"):
        prep = preprocess_command(engine, cmd)
        t1 = infer_tier1(engine, prep["processed_cmd"])

        y_true.append(label)
        y_scores.append(t1["malicious_prob"])
        y_pred.append(t1["pred_label"])

        preproc_times.append(prep["deobf_time_ms"])
        obf_flags.append(prep["was_obfuscated"])
        changed_flags.append(prep["was_changed"])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "auc": float(auc_score),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "mean_preproc_ms": float(np.mean(preproc_times)) if preproc_times else 0.0,
        "obfuscation_flag_rate": float(np.mean(obf_flags)) if obf_flags else 0.0,
        "deobfuscation_change_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
    }

# ============================================================
# TIER 2 EVALUATION (FULL PREPROCESSING PATH)
# ============================================================

def evaluate_tier2_pipeline(
    engine: GenosEngine,
    sp_test_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Evaluates Tier 2 over malicious samples, but through the same preprocessing
    path used by deployment.
    """
    inv_map = {v: k for k, v in engine.s_map.items()}

    y_true = []
    y_pred_top1 = []
    y_pred_top3 = []
    latencies_ms = []

    preproc_times = []
    changed_flags = []
    skipped_missing_label = 0

    for _, row in tqdm(sp_test_df.iterrows(), total=len(sp_test_df), desc="Tier 2 pipeline eval"):
        true_mitre = str(row[MITRE_COL])
        if true_mitre not in inv_map:
            skipped_missing_label += 1
            continue

        start = time.perf_counter()

        prep = preprocess_command(engine, row[COMMAND_COL])
        t2 = infer_tier2(engine, prep["processed_cmd"], topk=3)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        y_true.append(inv_map[true_mitre])
        y_pred_top1.append(t2["top1_idx"])
        y_pred_top3.append(t2["topk_idxs"])

        latencies_ms.append(elapsed_ms)
        preproc_times.append(prep["deobf_time_ms"])
        changed_flags.append(prep["was_changed"])

    return {
        "top1_acc": accuracy_score(y_true, y_pred_top1),
        "top3_acc": topk_hit_rate(y_true, y_pred_top3),
        "macro_f1": macro_f1(y_true, y_pred_top1),
        "lat_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "mean_preproc_ms": float(np.mean(preproc_times)) if preproc_times else 0.0,
        "deobfuscation_change_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
        "n_eval": len(y_true),
        "n_skipped_missing_label": skipped_missing_label,
    }

# ============================================================
# END-TO-END PIPELINE EVALUATION
# ============================================================

def build_mixed_stream(
    benign_df: pd.DataFrame,
    malicious_df: pd.DataFrame,
    benign_ratio: float = 0.99,
    total_samples: int = 10000,
) -> pd.DataFrame:
    n_benign = int(total_samples * benign_ratio)
    n_mal = total_samples - n_benign

    benign_sample = benign_df.sample(
        n=n_benign,
        replace=n_benign > len(benign_df),
        random_state=SEED,
    )
    malicious_sample = malicious_df.sample(
        n=n_mal,
        replace=n_mal > len(malicious_df),
        random_state=SEED,
    )

    mixed_df = pd.concat(
        [
            benign_sample[[COMMAND_COL]].assign(binary_label=0),
            malicious_sample[[COMMAND_COL]].assign(binary_label=1),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return mixed_df

def evaluate_full_pipeline_end_to_end(
    engine: GenosEngine,
    benign_df: pd.DataFrame,
    malicious_df: pd.DataFrame,
    benign_ratio: float = 0.99,
    total_samples: int = 10000,
) -> Dict[str, object]:
    """
    Measures the exact deployed cascade logic:
    preprocess -> Tier 1 argmax route -> conditional Tier 2
    """
    mixed_df = build_mixed_stream(
        benign_df=benign_df,
        malicious_df=malicious_df,
        benign_ratio=benign_ratio,
        total_samples=total_samples,
    )

    total_latencies_ms = []
    preproc_latencies_ms = []
    tier2_trigger_count = 0
    obf_flags = []
    changed_flags = []

    for _, row in tqdm(mixed_df.iterrows(), total=len(mixed_df), desc="Full pipeline end-to-end"):
        start_total = time.perf_counter()

        prep = preprocess_command(engine, row[COMMAND_COL])
        t1 = infer_tier1(engine, prep["processed_cmd"])

        if t1["pred_label"] == 1:
            tier2_trigger_count += 1
            _ = infer_tier2(engine, prep["processed_cmd"], topk=3)

        total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0

        total_latencies_ms.append(total_elapsed_ms)
        preproc_latencies_ms.append(prep["deobf_time_ms"])
        obf_flags.append(prep["was_obfuscated"])
        changed_flags.append(prep["was_changed"])

    return {
        "end_to_end_lat_ms": float(np.mean(total_latencies_ms)) if total_latencies_ms else 0.0,
        "mean_preproc_ms": float(np.mean(preproc_latencies_ms)) if preproc_latencies_ms else 0.0,
        "tier2_trigger_rate": tier2_trigger_count / len(mixed_df) if len(mixed_df) else 0.0,
        "obfuscation_flag_rate": float(np.mean(obf_flags)) if obf_flags else 0.0,
        "deobfuscation_change_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
        "n_stream": len(mixed_df),
    }

# ============================================================
# ROC PLOT
# ============================================================

def save_roc_plot(fpr, tpr, auc_score, output_path: Path) -> None:
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, lw=2, label=f"Tier 1 Gatekeeper (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Tier 1 ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(SEED)
    print("GENOS IEEE PIPELINE BENCHMARK v3.0")

    data = load_data()
    engine = GenosEngine()

    gk_test = data["gk_test"]
    sp_train = data["sp_train"]
    sp_test = data["sp_test"]
    traffic_ratios = [0.50, 0.90, 0.99]

    print("\n[*] TRAINING BASELINE...")
    baseline_vectorizer, baseline_clf = train_baseline(sp_train)
    baseline_res = evaluate_baseline(baseline_vectorizer, baseline_clf, sp_test)

    print("\n[*] EVALUATING TIER 1 (DEPLOYMENT-ALIGNED)...")
    tier1_res = evaluate_tier1_pipeline(engine, gk_test, sp_test)

    print("\n[*] EVALUATING TIER 2 (WITH PREPROCESSING)...")
    tier2_res = evaluate_tier2_pipeline(engine, sp_test)

    print("\n[*] EVALUATING FULL END-TO-END PIPELINE...")
    full_results = {
        ratio: evaluate_full_pipeline_end_to_end(
            engine,
            gk_test,
            sp_test,
            benign_ratio=ratio,
        )
        for ratio in traffic_ratios
    }

    print("\n" + "=" * 108)
    print(f"{'IEEE PAPER RESULTS SUMMARY (PIPELINE-ALIGNED)':^108}")
    print("=" * 108)
    print(
        f"{'Architecture':<32} | "
        f"{'Top-1 Acc':<10} | "
        f"{'Top-3 Acc':<10} | "
        f"{'Macro F1':<10} | "
        f"{'Latency (ms)':<12}"
    )
    print("-" * 108)
    print(
        f"{'TF-IDF + Random Forest':<32} | "
        f"{baseline_res['top1_acc']*100:>9.2f}% | "
        f"{baseline_res['top3_acc']*100:>9.2f}% | "
        f"{baseline_res['macro_f1']*100:>9.2f}% | "
        f"{baseline_res['lat_ms']:>11.2f}"
    )
    print(
        f"{'CodeBERT Specialist + Preprocessing':<32} | "
        f"{tier2_res['top1_acc']*100:>9.2f}% | "
        f"{tier2_res['top3_acc']*100:>9.2f}% | "
        f"{tier2_res['macro_f1']*100:>9.2f}% | "
        f"{tier2_res['lat_ms']:>11.2f}"
    )
    print("-" * 108)
    print(f"Tier 1 Gatekeeper AUC          : {tier1_res['auc']:.4f}")
    print(f"Tier 1 Precision               : {tier1_res['precision']*100:.2f}%")
    print(f"Tier 1 Recall                  : {tier1_res['recall']*100:.2f}%")
    print(f"Tier 1 F1                      : {tier1_res['f1']*100:.2f}%")
    print(f"Mean De-obfuscation Time       : {full_results[0.99]['mean_preproc_ms']:.2f} ms")
    print("\nEnd-to-End Sweep")
    print("-" * 108)
    print(f"{'Benign Ratio':<14} | {'Latency (ms)':<12} | {'Tier 2 Trigger':<14} | {'Obf Flag Rate':<13} | {'De-obf Change':<13}")
    print("-" * 108)
    for ratio in traffic_ratios:
        full_res = full_results[ratio]
        print(
            f"{ratio:<14.2f} | "
            f"{full_res['end_to_end_lat_ms']:>11.2f} | "
            f"{full_res['tier2_trigger_rate']*100:>13.2f}% | "
            f"{full_res['obfuscation_flag_rate']*100:>12.2f}% | "
            f"{full_res['deobfuscation_change_rate']*100:>12.2f}%"
        )
    print("=" * 108)

    roc_path = LOG_DIR / "ieee_roc_curve_pipeline.png"
    save_roc_plot(tier1_res["fpr"], tier1_res["tpr"], tier1_res["auc"], roc_path)
    print(f"\n[+] ROC curve saved to: {roc_path}")

    print("\nPAPER-READY METRICS")
    print("-" * 108)
    print(f"Baseline Top-1 Accuracy              : {baseline_res['top1_acc']*100:.2f}%")
    print(f"Baseline Top-3 Accuracy              : {baseline_res['top3_acc']*100:.2f}%")
    print(f"Baseline Macro F1                    : {baseline_res['macro_f1']*100:.2f}%")
    print(f"Baseline Latency                     : {baseline_res['lat_ms']:.2f} ms")
    print()
    print(f"Tier 1 AUC                           : {tier1_res['auc']:.4f}")
    print(f"Tier 1 Precision                     : {tier1_res['precision']*100:.2f}%")
    print(f"Tier 1 Recall                        : {tier1_res['recall']*100:.2f}%")
    print(f"Tier 1 F1                            : {tier1_res['f1']*100:.2f}%")
    print()
    print(f"Tier 2 Top-1 Accuracy                : {tier2_res['top1_acc']*100:.2f}%")
    print(f"Tier 2 Top-3 Accuracy                : {tier2_res['top3_acc']*100:.2f}%")
    print(f"Tier 2 Macro F1                      : {tier2_res['macro_f1']*100:.2f}%")
    print(f"Tier 2 Latency (with preprocessing)  : {tier2_res['lat_ms']:.2f} ms")
    print()
    for ratio in traffic_ratios:
        full_res = full_results[ratio]
        print(f"End-to-End Pipeline Latency @ {ratio:.2f} benign : {full_res['end_to_end_lat_ms']:.2f} ms")
        print(f"Tier 2 Trigger Rate @ {ratio:.2f} benign         : {full_res['tier2_trigger_rate']*100:.2f}%")
        print(f"Obfuscation Flag Rate @ {ratio:.2f} benign       : {full_res['obfuscation_flag_rate']*100:.2f}%")
        print(f"De-obfuscation Change Rate @ {ratio:.2f} benign  : {full_res['deobfuscation_change_rate']*100:.2f}%")
        print()
    print("-" * 108)


if __name__ == "__main__":
    main()