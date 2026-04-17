import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

SEED = 42
MAX_LENGTH = 256

BENCHMARK_DIR = Path(__file__).resolve().parent
BASE_DIR = BENCHMARK_DIR.parent.parent
DATA_DIR = BASE_DIR / "data" / "training"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COMMAND_COL = "command"
MITRE_COL = "mitre_id"

sys.path.insert(0, str(BASE_DIR))

try:
    from engine import GenosEngine
except ImportError:
    print("[!] ERROR: Could not import 'GenosEngine'. Run from scripts/benchmark/ directory.")
    sys.exit(1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_csv(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    candidates = [path]
    if path.parent == DATA_DIR:
        candidates.append(DATA_DIR / "genos_dataset" / path.name)

    actual_path = next((c for c in candidates if c.exists()), None)

    if actual_path is None:
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None

    df = pd.read_csv(actual_path)
    if COMMAND_COL not in df.columns:
        raise ValueError(f"{actual_path} missing required column: {COMMAND_COL}")

    df[COMMAND_COL] = df[COMMAND_COL].astype(str).fillna("").str.strip()
    if MITRE_COL in df.columns:
        df[MITRE_COL] = df[MITRE_COL].astype(str).fillna("").str.strip()

    return df


def _parse_benign_fp_txt(path: Path) -> pd.DataFrame:
    commands = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if "| " in line and ("FP" in line or "OK" in line):
            cmd = line.split("|")[-1].strip()
            commands.append(cmd)
    if not commands:
        raise ValueError(f"Could not parse benign command lines from {path}")
    return pd.DataFrame({"command": commands, "label": [0] * len(commands)})


def load_real_world_benign() -> Optional[pd.DataFrame]:
    candidates = [
        BENCHMARK_DIR / "benign_fp.txt",
        BENCHMARK_DIR / "real_world_benign.csv",
        DATA_DIR / "real_world_benign.csv",
        DATA_DIR / "benign_fp.csv",
        BASE_DIR / "benign_fp.txt",
        DATA_DIR / "benign_fp.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            if "command" not in df.columns:
                continue
            if "label" not in df.columns:
                df["label"] = 0
            df["command"] = df["command"].astype(str).fillna("").str.strip()
            df["label"] = df["label"].astype(int)
            return df
        if path.suffix.lower() == ".txt":
            return _parse_benign_fp_txt(path)
    return None


def load_data() -> Dict[str, Optional[pd.DataFrame]]:
    return {
        "gk_test": _read_csv(DATA_DIR / "gatekeeper_test.csv", required=True),
        "sp_train": _read_csv(DATA_DIR / "specialist_train.csv", required=True),
        "sp_test": _read_csv(DATA_DIR / "specialist_test.csv", required=True),
        "real_benign": load_real_world_benign(),
    }


def load_gatekeeper_threshold() -> Optional[float]:
    meta_path = CONFIG_DIR / "gatekeeper_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if "threshold" in meta:
        return float(meta["threshold"])
    for key in ("test_metrics", "val_metrics"):
        if isinstance(meta.get(key), dict) and "threshold" in meta[key]:
            return float(meta[key]["threshold"])
    return None


def topk_hit_rate(y_true: List, y_topk: List[np.ndarray]) -> float:
    if not y_true:
        return 0.0
    hits = sum(1 for i in range(len(y_true)) if y_true[i] in y_topk[i])
    return hits / len(y_true)


def macro_f1(y_true: List, y_pred: List) -> float:
    return precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]


def train_baseline(train_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
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
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

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


def preprocess_command(engine: GenosEngine, raw_cmd: str) -> Dict[str, object]:
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


def infer_tier1(engine: GenosEngine, processed_cmd: str, threshold: Optional[float]) -> Dict[str, float]:
    inputs = tokenize_command(engine, processed_cmd)
    with torch.no_grad():
        logits = engine.t1(inputs["input_ids"], inputs["attention_mask"])
        probs = F.softmax(logits, dim=1).squeeze(0)

    benign_prob = float(probs[0].item())
    malicious_prob = float(probs[1].item())

    if threshold is None:
        pred_label = 1 if malicious_prob >= benign_prob else 0
        threshold_used = None
        decision_mode = "argmax"
    else:
        pred_label = 1 if malicious_prob >= threshold else 0
        threshold_used = float(threshold)
        decision_mode = "threshold"

    return {
        "benign_prob": benign_prob,
        "malicious_prob": malicious_prob,
        "pred_label": pred_label,
        "threshold_used": threshold_used,
        "decision_mode": decision_mode,
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


def evaluate_tier1_pipeline(
    engine: GenosEngine,
    benign_df: pd.DataFrame,
    malicious_df: pd.DataFrame,
    threshold: Optional[float],
) -> Dict[str, object]:
    y_true = []
    y_scores = []
    y_pred = []

    preproc_times = []
    obf_flags = []
    changed_flags = []

    all_rows = [(0, cmd) for cmd in benign_df[COMMAND_COL].tolist()] + [(1, cmd) for cmd in malicious_df[COMMAND_COL].tolist()]

    decision_mode = "argmax" if threshold is None else "threshold"

    for label, cmd in tqdm(all_rows, desc="Tier 1 pipeline eval"):
        prep = preprocess_command(engine, cmd)
        t1 = infer_tier1(engine, prep["processed_cmd"], threshold=threshold)

        y_true.append(label)
        y_scores.append(t1["malicious_prob"])
        y_pred.append(t1["pred_label"])

        preproc_times.append(prep["deobf_time_ms"])
        obf_flags.append(prep["was_obfuscated"])
        changed_flags.append(prep["was_changed"])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)
    balanced_acc = (specificity + recall) / 2.0

    return {
        "auc": float(auc_score),
        "ap": float(ap_score),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": float(balanced_acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
        "threshold_curve": thresholds.tolist(),
        "threshold_used": threshold,
        "decision_mode": decision_mode,
        "mean_preproc_ms": float(np.mean(preproc_times)) if preproc_times else 0.0,
        "obfuscation_flag_rate": float(np.mean(obf_flags)) if obf_flags else 0.0,
        "deobfuscation_change_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
    }


def evaluate_tier2_pipeline(engine: GenosEngine, sp_test_df: pd.DataFrame) -> Dict[str, object]:
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


def build_mixed_stream(
    benign_df: pd.DataFrame,
    malicious_df: pd.DataFrame,
    benign_ratio: float = 0.99,
    total_samples: int = 10000,
) -> pd.DataFrame:
    n_benign = int(total_samples * benign_ratio)
    n_mal = total_samples - n_benign

    benign_sample = benign_df.sample(n=n_benign, replace=n_benign > len(benign_df), random_state=SEED)
    malicious_sample = malicious_df.sample(n=n_mal, replace=n_mal > len(malicious_df), random_state=SEED)

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
    threshold: Optional[float],
    benign_ratio: float = 0.99,
    total_samples: int = 10000,
) -> Dict[str, object]:
    mixed_df = build_mixed_stream(benign_df, malicious_df, benign_ratio=benign_ratio, total_samples=total_samples)

    total_latencies_ms = []
    preproc_latencies_ms = []
    tier2_trigger_count = 0
    obf_flags = []
    changed_flags = []
    y_true = []
    y_pred = []

    for _, row in tqdm(mixed_df.iterrows(), total=len(mixed_df), desc="Full pipeline end-to-end"):
        start_total = time.perf_counter()
        prep = preprocess_command(engine, row[COMMAND_COL])
        t1 = infer_tier1(engine, prep["processed_cmd"], threshold=threshold)

        if t1["pred_label"] == 1:
            tier2_trigger_count += 1
            _ = infer_tier2(engine, prep["processed_cmd"], topk=3)

        total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0

        y_true.append(int(row["binary_label"]))
        y_pred.append(int(t1["pred_label"]))
        total_latencies_ms.append(total_elapsed_ms)
        preproc_latencies_ms.append(prep["deobf_time_ms"])
        obf_flags.append(prep["was_obfuscated"])
        changed_flags.append(prep["was_changed"])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    bal_acc = (recall + specificity) / 2.0
    bin_acc = accuracy_score(y_true, y_pred)

    return {
        "end_to_end_lat_ms": float(np.mean(total_latencies_ms)) if total_latencies_ms else 0.0,
        "mean_preproc_ms": float(np.mean(preproc_latencies_ms)) if preproc_latencies_ms else 0.0,
        "tier2_trigger_rate": tier2_trigger_count / len(mixed_df) if len(mixed_df) else 0.0,
        "obfuscation_flag_rate": float(np.mean(obf_flags)) if obf_flags else 0.0,
        "deobfuscation_change_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
        "binary_acc": float(bin_acc),
        "binary_bal_acc": float(bal_acc),
        "n_stream": len(mixed_df),
    }


def evaluate_real_world_benign(
    engine: GenosEngine,
    df: pd.DataFrame,
    threshold: Optional[float],
) -> Dict[str, object]:
    rows = []
    fp_count = 0
    tn_count = 0

    for cmd in tqdm(df["command"].astype(str).tolist(), desc="Real-world benign eval"):
        prep = preprocess_command(engine, cmd)
        t1 = infer_tier1(engine, prep["processed_cmd"], threshold=threshold)
        is_fp = int(t1["pred_label"] == 1)
        fp_count += is_fp
        tn_count += int(not is_fp)

        rows.append(
            {
                "command": cmd,
                "pred_label": t1["pred_label"],
                "malicious_prob": t1["malicious_prob"],
                "benign_prob": t1["benign_prob"],
                "is_false_positive": is_fp,
            }
        )

    results_df = pd.DataFrame(rows)
    out_csv = LOG_DIR / "real_world_benign_results.csv"
    results_df.to_csv(out_csv, index=False)

    return {
        "n_total": int(len(results_df)),
        "fp_count": int(fp_count),
        "true_negative_count": int(tn_count),
        "fp_rate": float(fp_count / max(1, len(results_df))),
        "mean_malicious_prob": float(results_df["malicious_prob"].mean()) if len(results_df) else 0.0,
        "csv_path": str(out_csv),
    }


def save_roc_plot(fpr, tpr, auc_score, output_path: Path) -> None:
    plt.figure(figsize=(8, 8), dpi=200)
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


def main() -> None:
    set_seed(SEED)
    print("GENOS IEEE PIPELINE BENCHMARK (THRESHOLD-AWARE)")

    data = load_data()
    threshold = load_gatekeeper_threshold()
    engine = GenosEngine()

    gk_test = data["gk_test"]
    sp_train = data["sp_train"]
    sp_test = data["sp_test"]
    real_benign = data["real_benign"]

    traffic_ratios = [0.50, 0.90, 0.99]

    print("\n[*] TRAINING BASELINE...")
    baseline_vectorizer, baseline_clf = train_baseline(sp_train)
    baseline_res = evaluate_baseline(baseline_vectorizer, baseline_clf, sp_test)

    print("\n[*] EVALUATING TIER 1 (THRESHOLD-AWARE)...")
    tier1_res = evaluate_tier1_pipeline(engine, gk_test, sp_test, threshold=threshold)

    print("\n[*] EVALUATING TIER 2...")
    tier2_res = evaluate_tier2_pipeline(engine, sp_test)

    print("\n[*] EVALUATING FULL END-TO-END PIPELINE...")
    full_results = {
        ratio: evaluate_full_pipeline_end_to_end(engine, gk_test, sp_test, threshold=threshold, benign_ratio=ratio)
        for ratio in traffic_ratios
    }

    real_world_res = None
    if real_benign is not None:
        print("\n[*] EVALUATING REAL-WORLD BENIGN BENCHMARK...")
        real_world_res = evaluate_real_world_benign(engine, real_benign, threshold=threshold)
        print(f"[+] Saved real-world benign results to: {real_world_res['csv_path']}")

    print("\n" + "=" * 112)
    print(f"{'IEEE PAPER RESULTS SUMMARY (THRESHOLD-AWARE)':^112}")
    print("=" * 112)
    print(f"{'Architecture':<32} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'Macro F1':<10} | {'Latency (ms)':<12}")
    print("-" * 112)
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
    print("-" * 112)
    print(f"Tier 1 Gatekeeper AUC                : {tier1_res['auc']:.4f}")
    print(f"Tier 1 Gatekeeper AP                 : {tier1_res['ap']:.4f}")
    print(f"Tier 1 Precision                     : {tier1_res['precision']*100:.2f}%")
    print(f"Tier 1 Recall                        : {tier1_res['recall']*100:.2f}%")
    print(f"Tier 1 F1                            : {tier1_res['f1']*100:.2f}%")
    print(f"Tier 1 Balanced Accuracy             : {tier1_res['balanced_acc']*100:.2f}%")
    print(
        f"Tier 1 Confusion Matrix              : "
        f"TN={tier1_res['tn']} FP={tier1_res['fp']} FN={tier1_res['fn']} TP={tier1_res['tp']}"
    )
    print(
        f"Tier 1 Threshold Used                : "
        f"{'argmax' if tier1_res['threshold_used'] is None else tier1_res['threshold_used']}"
    )

    if real_world_res is not None:
        print("\nReal-World Benign Benchmark")
        print("-" * 112)
        print(f"Commands Tested                      : {real_world_res['n_total']}")
        print(f"False Positives                      : {real_world_res['fp_count']}")
        print(f"True Negatives                       : {real_world_res['true_negative_count']}")
        print(f"False Positive Rate                  : {real_world_res['fp_rate']*100:.2f}%")
        print(f"Mean Malicious Probability           : {real_world_res['mean_malicious_prob']*100:.2f}%")

    print("\nEnd-to-End Sweep")
    print("-" * 112)
    print(f"{'Benign Ratio':<14} | {'Latency (ms)':<12} | {'Tier 2 Trigger':<14} | {'Binary Acc':<11} | {'Balanced Acc':<13}")
    print("-" * 112)
    for ratio in traffic_ratios:
        full_res = full_results[ratio]
        print(
            f"{ratio:<14.2f} | "
            f"{full_res['end_to_end_lat_ms']:>11.2f} | "
            f"{full_res['tier2_trigger_rate']*100:>13.2f}% | "
            f"{full_res['binary_acc']*100:>10.2f}% | "
            f"{full_res['binary_bal_acc']*100:>12.2f}%"
        )
    print("=" * 112)

    roc_path = LOG_DIR / "ieee_roc_curve_threshold_aware.png"
    save_roc_plot(tier1_res["fpr_curve"], tier1_res["tpr_curve"], tier1_res["auc"], roc_path)
    print(f"\n[+] ROC curve saved to: {roc_path}")

    results = {
        "baseline": baseline_res,
        "tier1": {
            "auc": tier1_res["auc"],
            "ap": tier1_res["ap"],
            "precision": tier1_res["precision"],
            "recall": tier1_res["recall"],
            "f1": tier1_res["f1"],
            "balanced_acc": tier1_res["balanced_acc"],
            "tn": tier1_res["tn"],
            "fp": tier1_res["fp"],
            "fn": tier1_res["fn"],
            "tp": tier1_res["tp"],
            "threshold_used": tier1_res["threshold_used"],
            "decision_mode": tier1_res["decision_mode"],
            "mean_preproc_ms": tier1_res["mean_preproc_ms"],
            "obfuscation_flag_rate": tier1_res["obfuscation_flag_rate"],
            "deobfuscation_change_rate": tier1_res["deobfuscation_change_rate"],
        },
        "tier2": tier2_res,
        "full_pipeline": full_results,
        "real_world_benign": real_world_res,
    }

    json_path = LOG_DIR / "ieee_results_threshold_aware.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[+] JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
