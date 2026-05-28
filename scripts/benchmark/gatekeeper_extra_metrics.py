#!/usr/bin/env python3
"""Extra metrics for the current gatekeeper checkpoint.

Computes:
  - malicious one-vs-rest AUC/AP
  - non-benign AUC/AP
  - balanced accuracy
  - local benign false-positive audit
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import RobertaTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from engine import GenosEngine
from scripts.training.trainer1 import GatekeeperModel, LABEL_NAMES, LABEL_TO_IDX, resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON report.",
    )
    return parser.parse_args()


def load_benign_commands() -> list[str]:
    source = (BASE_DIR / "scripts" / "benchmark" / "benign_fp_test.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BENIGN_COMMANDS":
                    return ast.literal_eval(node.value)
    raise RuntimeError("Could not extract BENIGN_COMMANDS from benign_fp_test.py")


def load_test_rows() -> list[dict[str, str]]:
    csv_path = Path(resolve_data_path("gatekeeper_3class_test.csv"))
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def load_model(device: torch.device) -> tuple[GatekeeperModel, RobertaTokenizer]:
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = GatekeeperModel(num_classes=len(LABEL_NAMES))
    state = torch.load(BASE_DIR / "models" / "gatekeeper.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_test_set(model: GatekeeperModel, tokenizer: RobertaTokenizer, device: torch.device, rows: list[dict[str, str]]) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[list[float]] = []

    with torch.no_grad():
        for row in rows:
            encoded = tokenizer(
                row["command"],
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            outputs = model(ids, mask)
            logits = outputs["verdict_logits"]
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            y_scores.append(probs)
            y_pred.append(max(range(len(probs)), key=lambda idx: probs[idx]))
            y_true.append(LABEL_TO_IDX[row["label"]])

    y_true_names = [LABEL_NAMES[index] for index in y_true]
    y_pred_names = [LABEL_NAMES[index] for index in y_pred]
    score_tensor = torch.tensor(y_scores)
    malicious_index = LABEL_TO_IDX["Malicious"]
    non_benign_true = [1 if label != LABEL_TO_IDX["Benign"] else 0 for label in y_true]
    malicious_true = [1 if label == malicious_index else 0 for label in y_true]
    malicious_scores = score_tensor[:, malicious_index].tolist()
    non_benign_scores = (1.0 - score_tensor[:, LABEL_TO_IDX["Benign"]]).tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_names,
        y_pred_names,
        labels=LABEL_NAMES,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(LABEL_NAMES, precision, recall, f1, support)
    }

    return {
        "auc_malicious_one_vs_rest": float(roc_auc_score(malicious_true, malicious_scores)),
        "ap_malicious_one_vs_rest": float(average_precision_score(malicious_true, malicious_scores)),
        "auc_non_benign": float(roc_auc_score(non_benign_true, non_benign_scores)),
        "ap_non_benign": float(average_precision_score(non_benign_true, non_benign_scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_names, y_pred_names)),
        "per_class": per_class,
    }


def evaluate_local_benign_fp(engine: GenosEngine, benign_commands: list[str]) -> dict:
    fp_examples = []
    fp_count = 0
    for command in benign_commands:
        result = engine.scan(command)
        label = result.get("label", "Unknown")
        confidence = float(result.get("label_confidence", 0.0))
        if label != "Benign":
            fp_count += 1
            if len(fp_examples) < 20:
                fp_examples.append(
                    {
                        "command": command,
                        "label": label,
                        "confidence": round(confidence, 2),
                    }
                )
    return {
        "n": len(benign_commands),
        "fp_count": fp_count,
        "fp_rate": fp_count / max(1, len(benign_commands)),
        "examples": fp_examples,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(device)
    rows = load_test_rows()
    benign_commands = load_benign_commands()
    engine = GenosEngine()

    report = {
        "test": evaluate_test_set(model, tokenizer, device, rows),
        "benign_fp_local": evaluate_local_benign_fp(engine, benign_commands),
    }

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()