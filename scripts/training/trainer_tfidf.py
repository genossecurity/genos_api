"""
trainer_tfidf.py — TF-IDF + Gradient Boosted Trees specialist (Tier 2).

Trains on the residual-expanded JSONL dataset using input_text (RAW + RESIDUAL + FEATURES)
which encodes tool names, flags, residual tokens, and semantic tags — ideal for TF-IDF.

Uses HistGradientBoostingClassifier (sklearn) which handles multi-class well and supports
class_weight balancing natively.  Optionally benchmarks a Random Forest alongside.

Usage:
    python3 trainer_tfidf.py
    python3 trainer_tfidf.py --data-dir data/training/genos_residual_expanded
    python3 trainer_tfidf.py --model rf          # Random Forest only
    python3 trainer_tfidf.py --model hgb         # HistGradientBoosting only
    python3 trainer_tfidf.py --model both        # train & compare both (default)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "training"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_DATA_DIR = DATA_DIR / "genos_residual_expanded"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path, label_map: Dict[str, int]) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lbl = row["label"]
            if lbl not in label_map:
                skipped += 1
                continue
            texts.append(row["input_text"])
            labels.append(label_map[lbl])
    if skipped:
        print(f"    Skipped {skipped} rows with unknown labels")
    return texts, labels


def topk_acc(y_true: List[int], proba: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    hits = sum(1 for i, y in enumerate(y_true) if y in topk[i])
    return hits / max(1, len(y_true))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

def build_rf_pipeline(n_estimators: int = 400, seed: int = 42) -> Pipeline:
    """
    TF-IDF with character + word n-grams → Random Forest.
    analyzer='char_wb' on top of word n-grams captures flag patterns (-sS, --exec).
    """
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=30000,
        sublinear_tf=True,
        min_df=1,
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def build_char_rf_pipeline(n_estimators: int = 400, seed: int = 42) -> Pipeline:
    """Character n-gram TF-IDF → Random Forest (captures --flags, /paths, etc.)."""
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=30000,
        sublinear_tf=True,
        min_df=1,
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(
    name: str,
    pipeline: Pipeline,
    X_train: List[str],
    y_train: List[int],
    X_val: List[str],
    y_val: List[int],
    X_test: List[str],
    y_test: List[int],
    num_classes: int,
) -> Dict:
    print(f"\n[*] Training {name}...")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_secs = time.perf_counter() - t0
    print(f"    Train time: {train_secs:.1f}s")

    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        t0 = time.perf_counter()
        proba = pipeline.predict_proba(X)
        lat_ms = (time.perf_counter() - t0) * 1000 / max(1, len(X))
        pred = np.argmax(proba, axis=1)

        top1 = accuracy_score(y, pred)
        top3 = topk_acc(y, proba, k=3)
        mf1 = f1_score(y, pred, average="macro", zero_division=0)

        results[split_name] = {
            "top1_acc": float(top1),
            "top3_acc": float(top3),
            "macro_f1": float(mf1),
            "lat_ms": float(lat_ms),
            "n_eval": len(y),
        }
        print(f"    {split_name}: top1={top1*100:.2f}%  top3={top3*100:.2f}%  macro_f1={mf1*100:.2f}%  lat={lat_ms:.2f}ms/sample")

    results["train_secs"] = train_secs
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--model", default="both", choices=["rf", "char_rf", "both"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    map_path = CONFIG_DIR / "specialist_map.json"
    with open(map_path) as f:
        raw_map = json.load(f)
    label_map: Dict[str, int] = {k: int(v) for k, v in raw_map.items()}
    num_classes = len(label_map)
    print(f"[*] {num_classes} classes from {map_path}")

    print("\n[*] Loading datasets...")
    X_train, y_train = load_jsonl(data_dir / "specialist_train_variant_a.jsonl", label_map)
    X_val,   y_val   = load_jsonl(data_dir / "specialist_val_variant_a.jsonl",   label_map)
    X_test,  y_test  = load_jsonl(data_dir / "specialist_test_variant_a.jsonl",  label_map)
    print(f"    train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    all_results = {}

    if args.model in ("rf", "both"):
        pipe = build_rf_pipeline(n_estimators=args.n_estimators, seed=args.seed)
        res = train_and_eval(
            "Word TF-IDF + RF", pipe,
            X_train, y_train, X_val, y_val, X_test, y_test, num_classes,
        )
        all_results["word_rf"] = res
        save_path = MODELS_DIR / "specialist_tfidf_rf.pkl"
        joblib.dump(pipe, save_path)
        print(f"    Saved → {save_path}")

    if args.model in ("char_rf", "both"):
        pipe = build_char_rf_pipeline(n_estimators=args.n_estimators, seed=args.seed)
        res = train_and_eval(
            "Char TF-IDF + RF", pipe,
            X_train, y_train, X_val, y_val, X_test, y_test, num_classes,
        )
        all_results["char_rf"] = res
        save_path = MODELS_DIR / "specialist_tfidf_char_rf.pkl"
        joblib.dump(pipe, save_path)
        print(f"    Saved → {save_path}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'TF-IDF SPECIALIST RESULTS':^90}")
    print("=" * 90)
    print(f"{'Model':<30} | {'Val Top-1':<10} | {'Val Top-3':<10} | {'Test Top-1':<11} | {'Test Top-3':<11} | {'Macro F1 (test)'}")
    print("-" * 90)
    for name, res in all_results.items():
        print(
            f"{name:<30} | "
            f"{res['val']['top1_acc']*100:>9.2f}% | "
            f"{res['val']['top3_acc']*100:>9.2f}% | "
            f"{res['test']['top1_acc']*100:>10.2f}% | "
            f"{res['test']['top3_acc']*100:>10.2f}% | "
            f"{res['test']['macro_f1']*100:>9.2f}%"
        )
    print("=" * 90)

    out_path = BASE_DIR / "logs" / "tfidf_specialist_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[+] Results saved → {out_path}")


if __name__ == "__main__":
    main()
