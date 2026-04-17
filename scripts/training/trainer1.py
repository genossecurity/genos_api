import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
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


class GatekeeperModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(ids, attention_mask=mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class BinaryDataset(Dataset):
    def __init__(self, benign_csv: str, malicious_csv: str, tokenizer: RobertaTokenizer, max_len: int = 256, phase: str = "Train"):
        print(f"[*] Loading and pre-tokenizing Gatekeeper dataset ({phase})...")
        benign_df = pd.read_csv(benign_csv).dropna(subset=["command"])
        malicious_df = pd.read_csv(malicious_csv).dropna(subset=["command"])

        benign_cmds = [str(c).lower().strip() for c in benign_df["command"].tolist()]
        malicious_cmds = [str(c).lower().strip() for c in malicious_df["command"].tolist()]

        self.texts = benign_cmds + malicious_cmds
        self.labels = [0] * len(benign_cmds) + [1] * len(malicious_cmds)
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        print(f"[+] {phase} load complete: {len(benign_cmds)} benign | {len(malicious_cmds)} malicious")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "ids": self.encodings["input_ids"][idx],
            "mask": self.encodings["attention_mask"][idx],
            "lbl": torch.tensor(self.labels[idx], dtype=torch.long),
        }


@torch.no_grad()
def collect_binary_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[List[int], List[float], float]:
    model.eval()
    targets: List[int] = []
    probs: List[float] = []
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        ids = batch["ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["lbl"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(ids, mask)
            loss = criterion(logits, labels)

        batch_probs = torch.softmax(logits, dim=1)[:, 1]
        targets.extend(labels.cpu().tolist())
        probs.extend(batch_probs.cpu().tolist())
        total_loss += loss.item()
        total_items += labels.size(0)

    avg_loss = total_loss / max(1, total_items)
    return targets, probs, avg_loss


def binary_metrics(targets: List[int], probs: List[float], threshold: float) -> Dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for y, p in zip(targets, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(targets, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(targets, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(targets, preds) if y == 1 and p == 0)
    total = max(1, len(targets))

    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }
    if roc_auc_score is not None and len(set(targets)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(targets, probs))
        except Exception:
            pass
    if average_precision_score is not None and len(set(targets)) > 1:
        try:
            metrics["pr_auc"] = float(average_precision_score(targets, probs))
        except Exception:
            pass
    return metrics


def choose_threshold(targets: List[int], probs: List[float]) -> Dict[str, float]:
    best = None
    for i in range(5, 96):
        threshold = i / 100.0
        metrics = binary_metrics(targets, probs, threshold)
        score = (metrics["f1"], metrics["balanced_accuracy"], metrics["precision"], metrics["accuracy"])
        if best is None or score > best[0]:
            best = (score, metrics)
    assert best is not None
    return best[1]


def compute_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2).float()
    total = counts.sum()
    weights = total / (len(counts) * counts.clamp_min(1.0))
    return weights.to(device)


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
) -> Dict[str, float]:
    targets, probs, avg_loss = collect_binary_outputs(model, loader, device, amp_enabled)
    metrics = binary_metrics(targets, probs, threshold)
    metrics["loss"] = avg_loss
    return metrics


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"[*] Training on device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = BinaryDataset(
        resolve_data_path("gatekeeper_train.csv"),
        resolve_data_path("specialist_train.csv"),
        tokenizer,
        max_len=args.max_len,
        phase="Train",
    )
    val_dataset = BinaryDataset(
        resolve_data_path("gatekeeper_val.csv"),
        resolve_data_path("specialist_val.csv"),
        tokenizer,
        max_len=args.max_len,
        phase="Validation",
    )
    test_dataset = BinaryDataset(
        resolve_data_path("gatekeeper_test.csv"),
        resolve_data_path("specialist_test.csv"),
        tokenizer,
        max_len=args.max_len,
        phase="Test",
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
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
    print(f"[*] Batch config: micro={args.micro_batch_size}, grad_acc={grad_acc_steps}, effective={effective_batch_size}")

    raw_model = GatekeeperModel().to(device)
    use_compile = args.use_compile and hasattr(torch, "compile")
    if use_compile:
        try:
            model = torch.compile(raw_model)
            print("[+] torch.compile() enabled.")
        except Exception as exc:  # pragma: no cover
            model = raw_model
            print(f"[-] torch.compile() failed: {exc}. Proceeding without it.")
    else:
        model = raw_model
        print("[*] torch.compile() disabled.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(train_dataset.labels, device)
    print(f"[*] Dynamic class weights: benign={class_weights[0].item():.4f}, malicious={class_weights[1].item():.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(enabled=amp_enabled)

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

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for step, batch in enumerate(loop, start=1):
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["lbl"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(ids, mask)
                loss = criterion(logits, labels) / grad_acc_steps

            scaler.scale(loss).backward()

            if step % grad_acc_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item() * grad_acc_steps
            loop.set_postfix(acc=f"{(train_correct / max(1, train_total)) * 100:.2f}%")

        train_acc = train_correct / max(1, train_total)
        train_loss = train_loss_sum / max(1, len(train_loader))

        val_targets, val_probs, val_loss = collect_binary_outputs(model, val_loader, device, amp_enabled)
        tuned_val = choose_threshold(val_targets, val_probs)
        tuned_val["loss"] = val_loss

        print(
            f"\n[Epoch {epoch + 1}] train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
            f"| val_f1={tuned_val['f1']:.4f} val_bal_acc={tuned_val['balanced_accuracy']:.4f} "
            f"| val_precision={tuned_val['precision']:.4f} val_recall={tuned_val['recall']:.4f} "
            f"| val_threshold={tuned_val['threshold']:.2f}"
        )

        ranking = (tuned_val["f1"], tuned_val["balanced_accuracy"], tuned_val["precision"], tuned_val["accuracy"])
        if best_bundle is None or ranking > best_bundle["ranking"]:
            best_bundle = {
                "ranking": ranking,
                "epoch": epoch + 1,
                "val_metrics": tuned_val,
                "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()],
                "train_loss": train_loss,
                "train_accuracy": train_acc,
            }
            torch.save(raw_model.state_dict(), model_save_path)
            print(f"[+] Gatekeeper improved. Saved weights to {model_save_path}")
        print("-" * 72)

    assert best_bundle is not None
    raw_model.load_state_dict(torch.load(model_save_path, map_location=device))
    raw_model.to(device)

    test_metrics = evaluate_binary(
        raw_model,
        test_loader,
        device,
        amp_enabled,
        threshold=best_bundle["val_metrics"]["threshold"],
    )

    meta = {
        "seed": args.seed,
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": effective_batch_size,
        "max_len": args.max_len,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "best_epoch": best_bundle["epoch"],
        "class_weights": best_bundle["class_weights"],
        "val_metrics": best_bundle["val_metrics"],
        "test_metrics": test_metrics,
        "model_path": str(model_save_path),
    }
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[*] Final Gatekeeper results")
    print(json.dumps(meta, indent=2))
    print(f"[+] Metadata saved to {meta_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Gatekeeper (Tier 1) model on fixed CSV splits.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("GENOS_T1_EPOCHS", "5")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("GENOS_T1_LR", "1e-5")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("GENOS_T1_WEIGHT_DECAY", "0.01")))
    parser.add_argument("--max-len", type=int, default=int(os.getenv("GENOS_T1_MAX_LEN", "256")))
    parser.add_argument("--micro-batch-size", type=int, default=int(os.getenv("GENOS_T1_MICRO_BATCH", "32")))
    parser.add_argument("--effective-batch-size", type=int, default=int(os.getenv("GENOS_T1_EFFECTIVE_BATCH", "256")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("GENOS_T1_NUM_WORKERS", "4")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("GENOS_T1_SEED", "42")))
    parser.add_argument("--use-compile", action="store_true", default=os.getenv("GENOS_T1_USE_COMPILE", "0") == "1")
    parser.add_argument("--deterministic", action="store_true", default=os.getenv("GENOS_T1_DETERMINISTIC", "0") == "1")
    train(parser.parse_args())
