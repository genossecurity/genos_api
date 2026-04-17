import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel, get_cosine_schedule_with_warmup

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "microsoft/codebert-base"


def resolve_data_path(filename: str) -> Path:
    candidates = [
        BASE_DIR / "data" / "training" / "genos_dataset" / filename,
        BASE_DIR / "data" / "training" / filename,
        BASE_DIR / "data" / "archive" / "training" / filename,
        Path(filename),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MeanPool(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class SpecialistModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        self.pool = MeanPool()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(ids, attention_mask=mask)
        pooled = self.pool(out.last_hidden_state, mask)
        return self.classifier(pooled)


class SpecialistDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_map: Dict[str, int],
        tokenizer,
        max_len: int = 256,
        phase: str = "Train",
    ) -> None:
        print(f"[*] Pre-tokenizing specialist data ({phase})...")
        df = df.dropna(subset=["command", "mitre_id"]).copy()
        df["command"] = df["command"].astype(str).str.lower().str.strip()
        df["mitre_id"] = df["mitre_id"].astype(str).str.strip()

        self.labels = [label_map[x] for x in df["mitre_id"].tolist()]
        self.texts = df["command"].tolist()
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        print(f"[+] {phase} load complete: {len(self.labels)} samples across {len(set(self.labels))} classes")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "ids": self.encodings["input_ids"][idx],
            "mask": self.encodings["attention_mask"][idx],
            "lbl": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_label_map(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    train_classes = sorted({str(x) for x in train_df["mitre_id"].dropna().tolist()})
    val_classes = {str(x) for x in val_df["mitre_id"].dropna().tolist()}
    test_classes = {str(x) for x in test_df["mitre_id"].dropna().tolist()}

    missing_from_train = (val_classes | test_classes) - set(train_classes)
    if missing_from_train:
        raise ValueError(f"Validation/test contain classes not present in training: {sorted(missing_from_train)}")

    return {mitre_id: idx for idx, mitre_id in enumerate(train_classes)}


def build_weighted_sampler(labels: Sequence[int], num_classes: int) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    label_tensor = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes).float().clamp_min(1.0)

    # Mild balancing: inverse sqrt to avoid extreme oversampling noise.
    class_weights = torch.sqrt(counts.sum() / (num_classes * counts))
    sample_weights = class_weights[label_tensor]
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(label_tensor),
        replacement=True,
    )
    return sampler, class_weights


def macro_metrics_from_predictions(
    targets: List[int],
    preds: List[int],
    topk_preds: List[List[int]],
    num_classes: int,
) -> Dict[str, float]:
    total = max(1, len(targets))
    correct = sum(1 for y, p in zip(targets, preds) if y == p)
    top3_correct = sum(1 for y, p in zip(targets, topk_preds) if y in p)

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0

    for cls in range(num_classes):
        tp = sum(1 for y, p in zip(targets, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(targets, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(targets, preds) if y == cls and p != cls)
        support = sum(1 for y in targets if y == cls)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_f1 += f1 * support

    return {
        "accuracy": correct / total,
        "top3_accuracy": top3_correct / total,
        "macro_precision": macro_precision / max(1, num_classes),
        "macro_recall": macro_recall / max(1, num_classes),
        "macro_f1": macro_f1 / max(1, num_classes),
        "weighted_f1": weighted_f1 / total,
    }


@torch.no_grad()
def evaluate_specialist(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    num_classes: int,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    targets: List[int] = []
    preds: List[int] = []
    topk_preds: List[List[int]] = []

    for batch in loader:
        ids = batch["ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["lbl"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(ids, mask)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

        batch_preds = torch.argmax(logits, dim=1)
        k = min(3, num_classes)
        batch_topk = torch.topk(logits, k=k, dim=1).indices

        targets.extend(labels.cpu().tolist())
        preds.extend(batch_preds.cpu().tolist())
        topk_preds.extend(batch_topk.cpu().tolist())

    metrics = macro_metrics_from_predictions(targets, preds, topk_preds, num_classes)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics


def load_warm_start(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        print("[*] No existing specialist weights found. Starting fresh.")
        return

    print(f"[*] Found existing weights at {checkpoint_path}. Attempting warm start...")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"[+] Warm start loaded {len(compatible)} tensors; skipped {len(skipped)} incompatible tensors.")
    if skipped:
        print(f"[*] Example skipped keys: {skipped[:5]}")


def set_encoder_trainable(model: SpecialistModel, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def train_specialist(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    pin_memory = device.type == "cuda"

    models_dir = BASE_DIR / "models"
    config_dir = BASE_DIR / "config"
    models_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = models_dir / "specialist.pt"
    map_path = config_dir / "specialist_map.json"
    meta_path = config_dir / "specialist_meta.json"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    df_train = pd.read_csv(resolve_data_path("specialist_train.csv"))
    df_val = pd.read_csv(resolve_data_path("specialist_val.csv"))
    df_test = pd.read_csv(resolve_data_path("specialist_test.csv"))
    print(f"[*] Loaded splits. Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    label_map = build_label_map(df_train, df_val, df_test)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)
    print(f"[*] Wrote dataset-aligned class map with {len(label_map)} classes to {map_path}")

    train_dataset = SpecialistDataset(df_train, label_map, tokenizer, max_len=args.max_len, phase="Train")
    val_dataset = SpecialistDataset(df_val, label_map, tokenizer, max_len=args.max_len, phase="Validation")
    test_dataset = SpecialistDataset(df_test, label_map, tokenizer, max_len=args.max_len, phase="Test")

    num_classes = len(label_map)
    sampler, sampler_class_weights = build_weighted_sampler(train_dataset.labels, num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model = SpecialistModel(num_classes=num_classes, dropout=args.dropout).to(device)

    if args.warm_start:
        load_warm_start(model, model_save_path, device)
    else:
        print("[*] Warm start disabled. Training specialist from scratch.")

    if args.freeze_encoder_epochs > 0:
        set_encoder_trainable(model, trainable=False)
        print(f"[*] Freezing encoder for first {args.freeze_encoder_epochs} epoch(s).")
    else:
        set_encoder_trainable(model, trainable=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Mild weighting only; the sampler already handles most imbalance.
    ce_weights = (sampler_class_weights / sampler_class_weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=args.label_smoothing)

    grad_acc_steps = max(1, args.effective_batch_size // args.micro_batch_size)
    effective_batch_size = grad_acc_steps * args.micro_batch_size
    steps_per_epoch = math.ceil(len(train_loader) / grad_acc_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler(enabled=amp_enabled)

    print(
        f"[*] Batch config: micro={args.micro_batch_size}, grad_acc={grad_acc_steps}, "
        f"effective={effective_batch_size}"
    )
    print(
        f"[*] Training config: epochs={args.epochs}, encoder_lr={args.encoder_lr}, head_lr={args.head_lr}, "
        f"warmup_steps={warmup_steps}, total_steps={total_steps}"
    )
    print(
        f"[*] Sampler/class weights ready. sampler_min={sampler_class_weights.min().item():.4f} "
        f"sampler_max={sampler_class_weights.max().item():.4f} "
        f"ce_min={ce_weights.min().item():.4f} ce_max={ce_weights.max().item():.4f}"
    )

    best = None
    patience_left = args.patience

    for epoch in range(args.epochs):
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs:
            set_encoder_trainable(model, trainable=True)
            print("[*] Encoder unfrozen.")

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_items = 0
        train_correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for step, batch in enumerate(loop, start=1):
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = batch["lbl"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(ids, mask)
                loss = criterion(logits, targets) / grad_acc_steps

            scaler.scale(loss).backward()

            if step % grad_acc_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == targets).sum().item()
            total_loss += loss.item() * grad_acc_steps * targets.size(0)
            total_items += targets.size(0)

            loop.set_postfix(
                acc=f"{(train_correct / max(1, total_items)) * 100:.2f}%",
                lr=f"{scheduler.get_last_lr()[-1]:.2e}",
            )

        train_metrics = {
            "loss": total_loss / max(1, total_items),
            "accuracy": train_correct / max(1, total_items),
        }
        val_metrics = evaluate_specialist(model, val_loader, device, amp_enabled, num_classes, criterion)

        print(
            f"\n[Epoch {epoch + 1}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"| val_macro_f1={val_metrics['macro_f1']:.4f} val_top3={val_metrics['top3_accuracy']:.4f} "
            f"| val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )

        ranking = (
            val_metrics["macro_f1"],
            val_metrics["top3_accuracy"],
            val_metrics["weighted_f1"],
            val_metrics["accuracy"],
        )

        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "sampler_class_weights": [float(x) for x in sampler_class_weights.cpu().tolist()],
                "ce_class_weights": [float(x) for x in ce_weights.detach().cpu().tolist()],
            }
            torch.save(model.state_dict(), model_save_path)
            print(f"[+] Specialist improved. Saved weights to {model_save_path}")
            patience_left = args.patience
        else:
            patience_left -= 1
            print(f"[*] No improvement. Early-stop patience left: {patience_left}")
            if patience_left <= 0:
                print("[*] Early stopping triggered.")
                break

        print("-" * 80)

    assert best is not None
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    test_metrics = evaluate_specialist(model, test_loader, device, amp_enabled, num_classes, criterion)

    meta = {
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": effective_batch_size,
        "max_len": args.max_len,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "warmup_ratio": args.warmup_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "warm_start": args.warm_start,
        "best_epoch": best["epoch"],
        "num_classes": num_classes,
        "label_map_path": str(map_path),
        "model_path": str(model_save_path),
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[*] Final Specialist results")
    print(json.dumps(meta, indent=2))
    print(f"[+] Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stronger Specialist (Tier 2) trainer for fixed CSV splits.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("GENOS_T2_EPOCHS", "25")))
    parser.add_argument("--encoder-lr", type=float, default=float(os.getenv("GENOS_T2_ENCODER_LR", "1e-5")))
    parser.add_argument("--head-lr", type=float, default=float(os.getenv("GENOS_T2_HEAD_LR", "5e-5")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("GENOS_T2_WEIGHT_DECAY", "0.01")))
    parser.add_argument("--label-smoothing", type=float, default=float(os.getenv("GENOS_T2_LABEL_SMOOTHING", "0.0")))
    parser.add_argument("--max-len", type=int, default=int(os.getenv("GENOS_T2_MAX_LEN", "256")))
    parser.add_argument("--micro-batch-size", type=int, default=int(os.getenv("GENOS_T2_MICRO_BATCH", "16")))
    parser.add_argument("--effective-batch-size", type=int, default=int(os.getenv("GENOS_T2_EFFECTIVE_BATCH", "64")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("GENOS_T2_NUM_WORKERS", "4")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("GENOS_T2_SEED", "42")))
    parser.add_argument("--dropout", type=float, default=float(os.getenv("GENOS_T2_DROPOUT", "0.2")))
    parser.add_argument("--warmup-ratio", type=float, default=float(os.getenv("GENOS_T2_WARMUP_RATIO", "0.08")))
    parser.add_argument("--grad-clip-norm", type=float, default=float(os.getenv("GENOS_T2_GRAD_CLIP", "1.0")))
    parser.add_argument("--patience", type=int, default=int(os.getenv("GENOS_T2_PATIENCE", "5")))
    parser.add_argument("--freeze-encoder-epochs", type=int, default=int(os.getenv("GENOS_T2_FREEZE_EPOCHS", "1")))
    parser.add_argument("--warm-start", action="store_true", default=os.getenv("GENOS_T2_WARM_START", "0") == "1")
    parser.add_argument("--deterministic", action="store_true", default=os.getenv("GENOS_T2_DETERMINISTIC", "0") == "1")
    train_specialist(parser.parse_args())

