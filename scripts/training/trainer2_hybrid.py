"""
trainer2_hybrid.py — Residual-aligned Specialist (Tier 2) trainer.

Trains the specialist model on pipeline-aligned residual datasets where
the input text already encodes structured context from the parser, semantic
features, and rule engine.

NO candidate masking during training.
NO prior fusion during training.
Priors are applied at runtime inference only (via candidate_mask.py).

Reads JSONL produced by build_residual_dataset.py.

Usage:
    python3 trainer2_hybrid.py --variant a              # Variant A: RAW + RESIDUAL + FEATURES
    python3 trainer2_hybrid.py --variant b              # Variant B: RESIDUAL + FEATURES
    python3 trainer2_hybrid.py --variant c              # Variant C: RESIDUAL only
    python3 trainer2_hybrid.py --variant a --warm-start # warm-start from existing specialist.pt
    python3 trainer2_hybrid.py --eval-only --variant a  # evaluate existing checkpoint
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel, get_cosine_schedule_with_warmup

try:
    import numpy as np
except ImportError:
    np = None

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "microsoft/codebert-base"


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

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


class ResidualSpecialistModel(nn.Module):
    """Same architecture as Tier2_Specialist in engine.py."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Residual JSONL dataset (clean — no masks, no priors)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualDataset(Dataset):
    """
    Reads JSONL produced by build_residual_dataset.py.

    Each sample provides:
        ids    (seq_len,)  tokenised input_text
        mask   (seq_len,)  attention mask
        lbl    scalar      label index
    """

    def __init__(
        self,
        jsonl_path: Path,
        label_map: Dict[str, int],
        tokenizer,
        max_len: int = 256,
        phase: str = "Train",
    ) -> None:
        print(f"[*] Loading residual dataset ({phase}): {jsonl_path}")
        self.texts: List[str] = []
        self.labels: List[int] = []
        self.rule_strengths: List[str] = []

        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                label_str = row["label"]
                if label_str not in label_map:
                    skipped += 1
                    continue
                self.texts.append(row["input_text"])
                self.labels.append(label_map[label_str])
                self.rule_strengths.append(row.get("rule_strength", "none"))

        if skipped:
            print(f"    Skipped {skipped} rows with unknown labels")

        print(f"[*] Pre-tokenizing {len(self.texts)} samples...")
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        print(f"[+] {phase}: {len(self.labels)} samples, "
              f"{len(set(self.labels))} classes")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "ids":  self.encodings["input_ids"][idx],
            "mask": self.encodings["attention_mask"][idx],
            "lbl":  torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sampler / metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(
    labels: Sequence[int], num_classes: int,
) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    lt = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(lt, minlength=num_classes).float().clamp_min(1.0)
    cw = torch.sqrt(counts.sum() / (num_classes * counts))
    sw = cw[lt]
    sampler = WeightedRandomSampler(sw.double(), len(lt), replacement=True)
    return sampler, cw


def macro_metrics(targets, preds, topk_preds, num_classes) -> dict:
    total = max(1, len(targets))
    correct = sum(1 for y, p in zip(targets, preds) if y == p)
    top3_ok = sum(1 for y, tk in zip(targets, topk_preds) if y in tk)

    mp = mr = mf = wf = 0.0
    for cls in range(num_classes):
        tp = sum(1 for y, p in zip(targets, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(targets, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(targets, preds) if y == cls and p != cls)
        sup = sum(1 for y in targets if y == cls)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        mp += prec; mr += rec; mf += f1; wf += f1 * sup

    nc = max(1, num_classes)
    return {
        "accuracy":         correct / total,
        "top3_accuracy":    top3_ok / total,
        "macro_precision":  mp / nc,
        "macro_recall":     mr / nc,
        "macro_f1":         mf / nc,
        "weighted_f1":      wf / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clean evaluation (no masking, no fusion — pure model output)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    num_classes: int,
    criterion: nn.Module,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    targets_all, preds_all, topk_all = [], [], []

    for batch in loader:
        ids = batch["ids"].to(device, non_blocking=True)
        attn = batch["mask"].to(device, non_blocking=True)
        labels = batch["lbl"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(ids, attn)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

        preds_batch = torch.argmax(logits, dim=1)
        k = min(3, num_classes)
        topk_batch = torch.topk(logits, k=k, dim=1).indices

        targets_all.extend(labels.cpu().tolist())
        preds_all.extend(preds_batch.cpu().tolist())
        topk_all.extend(topk_batch.cpu().tolist())

    m = macro_metrics(targets_all, preds_all, topk_all, num_classes)
    m["loss"] = total_loss / max(1, total_items)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Resolve paths
# ─────────────────────────────────────────────────────────────────────────────

def resolve_jsonl(filename: str) -> Path:
    candidates = [
        BASE_DIR / "data" / "training" / "genos_residual" / filename,
        BASE_DIR / "data" / "training" / "genos_dataset" / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find {filename}")


def variant_filenames(variant: str) -> Tuple[str, str, str]:
    """Return (train, val, test) JSONL filenames for a given variant."""
    v = variant.lower()
    return (
        f"specialist_train_variant_{v}.jsonl",
        f"specialist_val_variant_{v}.jsonl",
        f"specialist_test_variant_{v}.jsonl",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop — clean, no masking, no fusion
# ─────────────────────────────────────────────────────────────────────────────

def train_residual(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    pin = device.type == "cuda"

    models_dir = BASE_DIR / "models"
    config_dir = BASE_DIR / "config"
    models_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    v = args.variant.lower()
    save_path = models_dir / f"specialist_residual_{v}.pt"
    meta_path = config_dir / f"specialist_residual_{v}_meta.json"

    # Label map
    map_path = config_dir / "specialist_map.json"
    with open(map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"[*] {num_classes} classes from {map_path}")
    print(f"[*] Training variant: {v.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Datasets
    train_f, val_f, test_f = variant_filenames(v)
    train_ds = ResidualDataset(
        resolve_jsonl(train_f), label_map, tokenizer,
        max_len=args.max_len, phase="Train",
    )
    val_ds = ResidualDataset(
        resolve_jsonl(val_f), label_map, tokenizer,
        max_len=args.max_len, phase="Val",
    )
    test_ds = ResidualDataset(
        resolve_jsonl(test_f), label_map, tokenizer,
        max_len=args.max_len, phase="Test",
    )

    sampler, sampler_cw = build_weighted_sampler(train_ds.labels, num_classes)

    mkloader = lambda ds, samp=None, shuf=False: DataLoader(
        ds, batch_size=args.micro_batch_size, sampler=samp,
        shuffle=shuf, pin_memory=pin, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = mkloader(train_ds, samp=sampler)
    val_loader   = mkloader(val_ds)
    test_loader  = mkloader(test_ds)

    model = ResidualSpecialistModel(
        num_classes=num_classes, dropout=args.dropout,
    ).to(device)

    # Optional warm-start from existing specialist
    if args.warm_start:
        ckpt = models_dir / "specialist.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            compat = {k: v for k, v in state.items()
                      if k in model.state_dict()
                      and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)
            print(f"[+] Warm-started from {ckpt} ({len(compat)} tensors)")
        else:
            print("[*] No existing specialist checkpoint for warm start.")

    # Freeze encoder initially
    if args.freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"[*] Encoder frozen for {args.freeze_encoder_epochs} epoch(s)")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    ce_w = (sampler_cw / sampler_cw.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=args.label_smoothing)

    grad_acc = max(1, args.effective_batch_size // args.micro_batch_size)
    steps_per_ep = math.ceil(len(train_loader) / grad_acc)
    total_steps = max(1, steps_per_ep * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=amp_enabled)

    print(f"[*] Config: epochs={args.epochs}, variant={v}, "
          f"grad_acc={grad_acc}, seed={args.seed}")

    best = None
    patience_left = args.patience

    for epoch in range(args.epochs):
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print("[*] Encoder unfrozen.")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_items = train_correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(loop, 1):
            ids = batch["ids"].to(device, non_blocking=True)
            attn = batch["mask"].to(device, non_blocking=True)
            tgt = batch["lbl"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(ids, attn)
                loss = criterion(logits, tgt) / grad_acc

            scaler.scale(loss).backward()

            if step % grad_acc == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == tgt).sum().item()
            total_loss += loss.item() * grad_acc * tgt.size(0)
            total_items += tgt.size(0)
            loop.set_postfix(acc=f"{train_correct/max(1,total_items)*100:.1f}%")

        # Validation — clean (no mask, no fusion)
        val_m = evaluate_model(
            model, val_loader, device, amp_enabled,
            num_classes, criterion,
        )

        print(
            f"\n[Epoch {epoch+1}] "
            f"train_acc={train_correct/max(1,total_items):.4f} "
            f"| val: acc={val_m['accuracy']:.4f} "
            f"f1={val_m['macro_f1']:.4f} "
            f"top3={val_m['top3_accuracy']:.4f} "
            f"wf1={val_m['weighted_f1']:.4f} "
            f"loss={val_m['loss']:.4f}"
        )

        ranking = (
            val_m["macro_f1"],
            val_m["top3_accuracy"],
            val_m["weighted_f1"],
            val_m["accuracy"],
        )
        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "epoch": epoch + 1,
                "val": val_m,
            }
            torch.save(model.state_dict(), save_path)
            print(f"[+] Improved. Saved to {save_path}")
            patience_left = args.patience
        else:
            patience_left -= 1
            print(f"[*] No improvement. Patience left: {patience_left}")
            if patience_left <= 0:
                print("[*] Early stopping.")
                break
        print("-" * 80)

    # Final eval on test split
    assert best is not None
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    model.to(device)

    test_m = evaluate_model(
        model, test_loader, device, amp_enabled,
        num_classes, criterion,
    )

    meta = {
        "variant": v,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best["epoch"],
        "num_classes": num_classes,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": grad_acc * args.micro_batch_size,
        "max_len": args.max_len,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "dropout": args.dropout,
        "warm_start": args.warm_start,
        "model_path": str(save_path),
        "val": best["val"],
        "test": test_m,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESIDUAL SPECIALIST — VARIANT {v.upper()}")
    print("=" * 70)
    for label, m in [("val", best["val"]), ("test", test_m)]:
        print(
            f"\n  {label}: acc={m['accuracy']:.4f}  "
            f"top3={m['top3_accuracy']:.4f}  "
            f"macro_f1={m['macro_f1']:.4f}  "
            f"wf1={m['weighted_f1']:.4f}  "
            f"loss={m['loss']:.4f}"
        )
    print(f"\n  Best epoch: {best['epoch']}")
    print(f"  Model:      {save_path}")
    print(f"  Metadata:   {meta_path}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Eval-only mode
# ─────────────────────────────────────────────────────────────────────────────

def eval_only(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    config_dir = BASE_DIR / "config"
    models_dir = BASE_DIR / "models"

    with open(config_dir / "specialist_map.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    v = args.variant.lower()
    _, val_f, test_f = variant_filenames(v)

    val_ds = ResidualDataset(
        resolve_jsonl(val_f), label_map, tokenizer,
        max_len=args.max_len, phase="Val",
    )
    test_ds = ResidualDataset(
        resolve_jsonl(test_f), label_map, tokenizer,
        max_len=args.max_len, phase="Test",
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.micro_batch_size, shuffle=False,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.micro_batch_size, shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = ResidualSpecialistModel(
        num_classes=num_classes, dropout=args.dropout,
    ).to(device)

    ckpt = models_dir / f"specialist_residual_{v}.pt"
    if not ckpt.exists():
        ckpt = models_dir / "specialist.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )
    print(f"[*] Loaded weights from {ckpt}")

    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 70)
    print(f"EVALUATION — VARIANT {v.upper()}")
    print("=" * 70)

    for label, loader in [("val", val_loader), ("test", test_loader)]:
        m = evaluate_model(
            model, loader, device, amp_enabled,
            num_classes, criterion,
        )
        print(
            f"\n  {label}: acc={m['accuracy']:.4f}  "
            f"top3={m['top3_accuracy']:.4f}  "
            f"macro_f1={m['macro_f1']:.4f}  "
            f"wf1={m['weighted_f1']:.4f}  "
            f"loss={m['loss']:.4f}"
        )
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Residual-aligned Specialist trainer",
    )
    p.add_argument("--variant", type=str, default="a",
                   choices=["a", "b", "c"],
                   help="Dataset variant: a=RAW+RESIDUAL+FEATURES, "
                        "b=RESIDUAL+FEATURES, c=RESIDUAL only")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--effective-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--warmup-ratio", type=float, default=0.08)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--freeze-encoder-epochs", type=int, default=1)
    p.add_argument("--warm-start", action="store_true",
                   help="Warm-start from existing specialist.pt")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; evaluate existing checkpoint")

    args = p.parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train_residual(args)
