import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_data_path(filename: str) -> str:
    candidates = [
        BASE_DIR / "data" / "training" / "genos_dataset" / filename,
        BASE_DIR / "data" / "training" / filename,
        BASE_DIR / "data" / "archive" / "training" / filename,
        Path(filename),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Could not find {filename}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


NUM_CLASSES = 3
LABEL_NAMES = ["Benign", "Malicious", "Context_Dependent"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}
NON_BENIGN_TARGET = {
    "Benign": 0.0,
    "Malicious": 1.0,
    "Context_Dependent": 1.0,
}
MALICIOUS_GIVEN_NON_BENIGN_TARGET = {
    "Malicious": 1.0,
    "Context_Dependent": 0.0,
}
ORDINAL_RISK_TARGET = {
    "Benign": 0.0,
    "Context_Dependent": 0.5,
    "Malicious": 1.0,
}


class GatekeeperModel(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
        self.non_benign_head = nn.Linear(768, 1)
        self.malicious_given_non_benign_head = nn.Linear(768, 1)
        self.ordinal_risk_head = nn.Linear(768, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(ids, attention_mask=mask)
        pooled = out.last_hidden_state[:, 0, :]
        return {
            "verdict_logits": self.classifier(pooled),
            "non_benign_logit": self.non_benign_head(pooled).squeeze(-1),
            "malicious_given_non_benign_logit": self.malicious_given_non_benign_head(pooled).squeeze(-1),
            "ordinal_risk_logit": self.ordinal_risk_head(pooled).squeeze(-1),
        }


class ThreeClassDataset(Dataset):
    """Load a unified 3-class CSV (command, label, original_label, mitre_id)."""

    def __init__(self, csv_path: str, tokenizer: RobertaTokenizer, max_len: int = 256, phase: str = "Train"):
        print(f"[*] Loading and pre-tokenizing Gatekeeper 3-class dataset ({phase})...")
        df = pd.read_csv(csv_path).dropna(subset=["command"])

        self.texts = [str(c).lower().strip() for c in df["command"].tolist()]
        raw_labels = [str(label) for label in df["label"].tolist()]
        self.labels = [LABEL_TO_IDX[lbl] for lbl in raw_labels]
        self.non_benign_labels = [NON_BENIGN_TARGET[lbl] for lbl in raw_labels]
        self.malicious_given_non_benign_labels = [MALICIOUS_GIVEN_NON_BENIGN_TARGET.get(lbl, 0.0) for lbl in raw_labels]
        self.ordinal_risk_targets = [ORDINAL_RISK_TARGET[lbl] for lbl in raw_labels]
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        counts = Counter(self.labels)
        parts = " | ".join(f"{counts.get(i, 0)} {LABEL_NAMES[i].lower()}" for i in range(NUM_CLASSES))
        print(f"[+] {phase} load complete: {parts}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "ids": self.encodings["input_ids"][idx],
            "mask": self.encodings["attention_mask"][idx],
            "lbl": torch.tensor(self.labels[idx], dtype=torch.long),
            "non_benign_lbl": torch.tensor(self.non_benign_labels[idx], dtype=torch.float32),
            "malicious_given_non_benign_lbl": torch.tensor(self.malicious_given_non_benign_labels[idx], dtype=torch.float32),
            "ordinal_risk_target": torch.tensor(self.ordinal_risk_targets[idx], dtype=torch.float32),
        }


@torch.no_grad()
def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[List[int], List[List[float]], float, Dict[str, List[float]]]:
    """Collect ground-truth labels, verdict probabilities, and auxiliary-head outputs."""
    model.eval()
    targets: List[int] = []
    all_probs: List[List[float]] = []
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    aux_outputs: Dict[str, List[float]] = {
        "non_benign_targets": [],
        "non_benign_scores": [],
        "malicious_context_targets": [],
        "malicious_context_scores": [],
        "ordinal_risk_targets": [],
        "ordinal_risk_scores": [],
    }

    for batch in loader:
        ids = batch["ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["lbl"].to(device, non_blocking=True)
        non_benign_labels = batch["non_benign_lbl"].to(device, non_blocking=True)
        malicious_context_labels = batch["malicious_given_non_benign_lbl"].to(device, non_blocking=True)
        ordinal_risk_targets = batch["ordinal_risk_target"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = model(ids, mask)
            logits = outputs["verdict_logits"]
            loss = criterion(logits, labels)

        batch_probs = torch.softmax(logits, dim=1)
        non_benign_scores = torch.sigmoid(outputs["non_benign_logit"])
        malicious_context_scores = torch.sigmoid(outputs["malicious_given_non_benign_logit"])
        ordinal_risk_scores = torch.sigmoid(outputs["ordinal_risk_logit"])
        non_benign_mask = non_benign_labels > 0.5

        targets.extend(labels.cpu().tolist())
        all_probs.extend(batch_probs.cpu().tolist())
        aux_outputs["non_benign_targets"].extend(non_benign_labels.cpu().tolist())
        aux_outputs["non_benign_scores"].extend(non_benign_scores.cpu().tolist())
        aux_outputs["ordinal_risk_targets"].extend(ordinal_risk_targets.cpu().tolist())
        aux_outputs["ordinal_risk_scores"].extend(ordinal_risk_scores.cpu().tolist())
        aux_outputs["malicious_context_targets"].extend(malicious_context_labels[non_benign_mask].cpu().tolist())
        aux_outputs["malicious_context_scores"].extend(malicious_context_scores[non_benign_mask].cpu().tolist())
        total_loss += loss.item()
        total_items += labels.size(0)

    avg_loss = total_loss / max(1, total_items)
    return targets, all_probs, avg_loss, aux_outputs


def multiclass_metrics(targets: List[int], all_probs: List[List[float]]) -> Dict[str, float]:
    """Compute per-class and overall metrics from argmax predictions."""
    preds = [max(range(NUM_CLASSES), key=lambda c: p[c]) for p in all_probs]
    total = max(1, len(targets))

    # Overall accuracy
    accuracy = sum(1 for y, p in zip(targets, preds) if y == p) / total

    # Per-class precision, recall, F1
    per_class = {}
    for c in range(NUM_CLASSES):
        tp = sum(1 for y, p in zip(targets, preds) if y == c and p == c)
        fp = sum(1 for y, p in zip(targets, preds) if y != c and p == c)
        fn = sum(1 for y, p in zip(targets, preds) if y == c and p != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        per_class[LABEL_NAMES[c]] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / NUM_CLASSES
    macro_prec = sum(v["precision"] for v in per_class.values()) / NUM_CLASSES
    macro_rec = sum(v["recall"] for v in per_class.values()) / NUM_CLASSES

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class,
    }


def _safe_binary_auc_ap(targets: List[float], scores: List[float]) -> Dict[str, float | None]:
    if not targets or average_precision_score is None or roc_auc_score is None:
        return {"auc": None, "ap": None}
    unique_targets = {int(round(target)) for target in targets}
    if len(unique_targets) < 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(targets, scores)),
        "ap": float(average_precision_score(targets, scores)),
    }


def auxiliary_head_metrics(aux_outputs: Dict[str, List[float]]) -> Dict[str, float | None]:
    non_benign_metrics = _safe_binary_auc_ap(
        aux_outputs["non_benign_targets"],
        aux_outputs["non_benign_scores"],
    )
    malicious_context_metrics = _safe_binary_auc_ap(
        aux_outputs["malicious_context_targets"],
        aux_outputs["malicious_context_scores"],
    )

    ordinal_targets = aux_outputs["ordinal_risk_targets"]
    ordinal_scores = aux_outputs["ordinal_risk_scores"]
    ordinal_mae = None
    if ordinal_targets and ordinal_scores:
        ordinal_mae = float(
            sum(abs(target - score) for target, score in zip(ordinal_targets, ordinal_scores))
            / max(1, len(ordinal_targets))
        )

    return {
        "non_benign_auc": non_benign_metrics["auc"],
        "non_benign_ap": non_benign_metrics["ap"],
        "malicious_given_non_benign_auc": malicious_context_metrics["auc"],
        "malicious_given_non_benign_ap": malicious_context_metrics["ap"],
        "ordinal_mae": ordinal_mae,
    }


def compute_class_weights(
    labels: List[int],
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    power: float = 1.0,
) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp_min(1.0))
    if power != 1.0:
        weights = weights.pow(power)
    return weights.to(device)


def build_balanced_sampler(labels: List[int], num_classes: int = NUM_CLASSES, power: float = 1.0) -> WeightedRandomSampler:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float().clamp_min(1.0)
    class_sample_weights = (1.0 / counts).pow(power)
    sample_weights = class_sample_weights[torch.tensor(labels, dtype=torch.long)]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(labels),
        replacement=True,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    progress_is_tty = sys.stderr.isatty()
    log_interval = max(1, args.log_interval)
    print(f"[*] Training on device: {device}", flush=True)

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = ThreeClassDataset(
        resolve_data_path("gatekeeper_3class_train.csv"),
        tokenizer,
        max_len=args.max_len,
        phase="Train",
    )
    val_dataset = ThreeClassDataset(
        resolve_data_path("gatekeeper_3class_val.csv"),
        tokenizer,
        max_len=args.max_len,
        phase="Validation",
    )
    test_dataset = ThreeClassDataset(
        resolve_data_path("gatekeeper_3class_test.csv"),
        tokenizer,
        max_len=args.max_len,
        phase="Test",
    )

    pin_memory = device.type == "cuda"
    train_sampler = None
    if args.balanced_sampling:
        train_sampler = build_balanced_sampler(
            train_dataset.labels,
            power=args.sampling_power,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )

    grad_acc_steps = max(1, args.effective_batch_size // args.micro_batch_size)
    effective_batch_size = args.micro_batch_size * grad_acc_steps
    print(f"[*] Batch config: micro={args.micro_batch_size}, grad_acc={grad_acc_steps}, effective={effective_batch_size}", flush=True)

    raw_model = GatekeeperModel().to(device)
    use_compile = args.use_compile and hasattr(torch, "compile")
    if use_compile:
        try:
            model = torch.compile(raw_model)
            print("[+] torch.compile() enabled.", flush=True)
        except Exception as exc:  # pragma: no cover
            model = raw_model
            print(f"[-] torch.compile() failed: {exc}. Proceeding without it.", flush=True)
    else:
        model = raw_model
        print("[*] torch.compile() disabled.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(
        train_dataset.labels,
        device,
        power=args.class_weight_power,
    )
    wt_str = ", ".join(f"{LABEL_NAMES[i]}={class_weights[i].item():.4f}" for i in range(NUM_CLASSES))
    print(f"[*] Dynamic class weights: {wt_str}", flush=True)
    if args.balanced_sampling:
        print(f"[*] Balanced sampling enabled: power={args.sampling_power:.2f} (replacement sampling across classes)", flush=True)
    else:
        print("[*] Balanced sampling disabled.", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    non_benign_criterion = nn.BCEWithLogitsLoss()
    malicious_context_criterion = nn.BCEWithLogitsLoss()
    ordinal_risk_criterion = nn.MSELoss()
    scaler = GradScaler(enabled=amp_enabled)

    if args.stage_loss_weight > 0 or args.action_loss_weight > 0 or args.contrastive_loss_weight > 0:
        raise ValueError("Stage/action/contrastive losses are disabled for Tier 1. Use decision-decomposition losses instead.")

    models_dir = BASE_DIR / "models"
    config_dir = BASE_DIR / "config"
    models_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / "gatekeeper.pt"
    meta_save_path = config_dir / "gatekeeper_meta.json"

    best_bundle = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", disable=not progress_is_tty)
        for step, batch in enumerate(loop, start=1):
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["lbl"].to(device, non_blocking=True)
            non_benign_labels = batch["non_benign_lbl"].to(device, non_blocking=True)
            malicious_context_labels = batch["malicious_given_non_benign_lbl"].to(device, non_blocking=True)
            ordinal_risk_targets = batch["ordinal_risk_target"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(ids, mask)
                verdict_logits = outputs["verdict_logits"]
                verdict_loss = criterion(verdict_logits, labels)
                non_benign_loss = non_benign_criterion(outputs["non_benign_logit"], non_benign_labels)
                non_benign_mask = non_benign_labels > 0.5
                malicious_context_loss = verdict_logits.new_tensor(0.0)
                if non_benign_mask.any():
                    malicious_context_loss = malicious_context_criterion(
                        outputs["malicious_given_non_benign_logit"][non_benign_mask],
                        malicious_context_labels[non_benign_mask],
                    )
                ordinal_risk_loss = ordinal_risk_criterion(
                    torch.sigmoid(outputs["ordinal_risk_logit"]),
                    ordinal_risk_targets,
                )
                loss = (
                    verdict_loss
                    + args.non_benign_loss_weight * non_benign_loss
                    + args.malicious_context_loss_weight * malicious_context_loss
                    + args.ordinal_risk_loss_weight * ordinal_risk_loss
                ) / grad_acc_steps

            scaler.scale(loss).backward()

            if step % grad_acc_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            preds = torch.argmax(verdict_logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item() * grad_acc_steps
            current_acc = (train_correct / max(1, train_total)) * 100
            loop.set_postfix(acc=f"{current_acc:.2f}%")
            if not progress_is_tty and (step % log_interval == 0 or step == len(train_loader)):
                print(
                    f"[*] Epoch {epoch + 1}/{args.epochs} step {step}/{len(train_loader)} "
                    f"train_acc={current_acc:.2f}%",
                    flush=True,
                )

        train_acc = train_correct / max(1, train_total)
        train_loss = train_loss_sum / max(1, len(train_loader))

        val_targets, val_probs, val_loss, val_aux_outputs = collect_outputs(model, val_loader, device, amp_enabled)
        val_metrics = multiclass_metrics(val_targets, val_probs)
        val_metrics["loss"] = val_loss
        val_metrics["auxiliary"] = auxiliary_head_metrics(val_aux_outputs)

        print(
            f"\n[Epoch {epoch + 1}] train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
            f"| val_acc={val_metrics['accuracy'] * 100:.2f}% val_macro_f1={val_metrics['macro_f1']:.4f}"
        , flush=True)
        for cls_name in LABEL_NAMES:
            cm = val_metrics["per_class"][cls_name]
            print(f"  {cls_name:20s}  P={cm['precision']:.4f}  R={cm['recall']:.4f}  F1={cm['f1']:.4f}", flush=True)
        aux = val_metrics["auxiliary"]
        print(
            "  Aux heads             "
            f"NB-AUC={aux['non_benign_auc'] if aux['non_benign_auc'] is not None else 'n/a'}  "
            f"NB-AP={aux['non_benign_ap'] if aux['non_benign_ap'] is not None else 'n/a'}  "
            f"MC-AUC={aux['malicious_given_non_benign_auc'] if aux['malicious_given_non_benign_auc'] is not None else 'n/a'}  "
            f"MC-AP={aux['malicious_given_non_benign_ap'] if aux['malicious_given_non_benign_ap'] is not None else 'n/a'}  "
            f"Risk-MAE={aux['ordinal_mae'] if aux['ordinal_mae'] is not None else 'n/a'}"
        , flush=True)

        ranking = (val_metrics["macro_f1"], val_metrics["accuracy"])
        if best_bundle is None or ranking > best_bundle["ranking"]:
            best_bundle = {
                "ranking": ranking,
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()],
                "train_loss": train_loss,
                "train_accuracy": train_acc,
            }
            torch.save(raw_model.state_dict(), model_save_path)
            print(f"[+] Gatekeeper improved. Saved weights to {model_save_path}", flush=True)
        print("-" * 72, flush=True)

    assert best_bundle is not None
    raw_model.load_state_dict(torch.load(model_save_path, map_location=device))
    raw_model.to(device)

    test_targets, test_probs, test_loss, test_aux_outputs = collect_outputs(raw_model, test_loader, device, amp_enabled)
    test_metrics = multiclass_metrics(test_targets, test_probs)
    test_metrics["loss"] = test_loss
    test_metrics["auxiliary"] = auxiliary_head_metrics(test_aux_outputs)

    meta = {
        "seed": args.seed,
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": effective_batch_size,
        "max_len": args.max_len,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "balanced_sampling": args.balanced_sampling,
        "sampling_power": args.sampling_power,
        "class_weight_power": args.class_weight_power,
        "num_classes": NUM_CLASSES,
        "label_names": LABEL_NAMES,
        "label_map": LABEL_TO_IDX,
        "id_to_label": {str(i): label for i, label in enumerate(LABEL_NAMES)},
        "model_architecture": "shared_codebert_with_decision_decomposition",
        "decision_heads": [
            "verdict_logits",
            "non_benign_logit",
            "malicious_given_non_benign_logit",
            "ordinal_risk_logit",
        ],
        "non_benign_supervision_source": "verdict_label_projection",
        "malicious_given_non_benign_supervision_source": "verdict_label_projection",
        "ordinal_risk_supervision_source": "verdict_label_projection",
        "non_benign_loss_weight": args.non_benign_loss_weight,
        "malicious_context_loss_weight": args.malicious_context_loss_weight,
        "ordinal_risk_loss_weight": args.ordinal_risk_loss_weight,
        "stage_loss_weight": args.stage_loss_weight,
        "action_loss_weight": args.action_loss_weight,
        "contrastive_loss_weight": args.contrastive_loss_weight,
        "best_epoch": best_bundle["epoch"],
        "class_weights": best_bundle["class_weights"],
        "val_metrics": best_bundle["val_metrics"],
        "test_metrics": test_metrics,
        "model_path": str(model_save_path),
    }
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[*] Final Gatekeeper results", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[+] Metadata saved to {meta_save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Gatekeeper (Tier 1) model on fixed CSV splits.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("GENOS_T1_EPOCHS", "5")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("GENOS_T1_LR", "1e-5")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("GENOS_T1_WEIGHT_DECAY", "0.01")))
    parser.add_argument("--max-len", type=int, default=int(os.getenv("GENOS_T1_MAX_LEN", "256")))
    parser.add_argument("--micro-batch-size", type=int, default=int(os.getenv("GENOS_T1_MICRO_BATCH", "32")))
    parser.add_argument("--effective-batch-size", type=int, default=int(os.getenv("GENOS_T1_EFFECTIVE_BATCH", "256")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("GENOS_T1_NUM_WORKERS", "0")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("GENOS_T1_SEED", "42")))
    parser.add_argument("--label-smoothing", type=float, default=float(os.getenv("GENOS_T1_LABEL_SMOOTHING", "0.05")))
    parser.add_argument("--balanced-sampling", action="store_true", default=os.getenv("GENOS_T1_BALANCED_SAMPLING", "1") == "1")
    parser.add_argument("--sampling-power", type=float, default=float(os.getenv("GENOS_T1_SAMPLING_POWER", "1.0")))
    parser.add_argument("--class-weight-power", type=float, default=float(os.getenv("GENOS_T1_CLASS_WEIGHT_POWER", "0.5")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("GENOS_T1_LOG_INTERVAL", "100")))
    parser.add_argument("--use-compile", action="store_true", default=os.getenv("GENOS_T1_USE_COMPILE", "0") == "1")
    parser.add_argument("--deterministic", action="store_true", default=os.getenv("GENOS_T1_DETERMINISTIC", "0") == "1")
    parser.add_argument("--non-benign-loss-weight", type=float, default=float(os.getenv("GENOS_T1_NON_BENIGN_LOSS_WEIGHT", "0.0")))
    parser.add_argument("--malicious-context-loss-weight", type=float, default=float(os.getenv("GENOS_T1_MALICIOUS_CONTEXT_LOSS_WEIGHT", "0.0")))
    parser.add_argument("--ordinal-risk-loss-weight", type=float, default=float(os.getenv("GENOS_T1_ORDINAL_RISK_LOSS_WEIGHT", "0.0")))
    parser.add_argument("--stage-loss-weight", type=float, default=float(os.getenv("GENOS_T1_STAGE_LOSS_WEIGHT", "0.3")))
    parser.add_argument("--action-loss-weight", type=float, default=float(os.getenv("GENOS_T1_ACTION_LOSS_WEIGHT", "0.0")))
    parser.add_argument("--contrastive-loss-weight", type=float, default=float(os.getenv("GENOS_T1_CONTRASTIVE_LOSS_WEIGHT", "0.0")))
    train(parser.parse_args())
