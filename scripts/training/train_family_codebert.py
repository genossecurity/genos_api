#!/usr/bin/env python3
"""Fine-tune CodeBERT as a multi-label 11-family classifier on the template-grouped family splits."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import read_rows, require_disjoint, sha256_file
from scripts.training.train_family_specialist import metric_summary, targets_for

BACKBONE = os.getenv("GENOS_CODEBERT_BACKBONE", "microsoft/codebert-base")


class FamilyModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(BACKBONE, use_safetensors=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.head(self.dropout(pooled))


class Rows(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.encoded = tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]
        self.targets = targets
        self.pad = tokenizer.pad_token_id

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index], self.targets[index]


def collate(pad_id):
    def inner(batch):
        width = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (values, _) in enumerate(batch):
            ids[row, :len(values)] = torch.tensor(values)
            mask[row, :len(values)] = 1
        return ids, mask, torch.tensor(np.stack([t for _, t in batch]), dtype=torch.float32)
    return inner


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    outputs = []
    for ids, mask, _ in loader:
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(ids.to(device), mask.to(device))
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs)


def export(path, rows, targets, probabilities, labels, threshold):
    with path.open("w", encoding="utf-8") as handle:
        for row, target, scores in zip(rows, targets, probabilities):
            handle.write(json.dumps({
                "command": row["command"],
                "family_targets": [labels[i] for i in np.flatnonzero(target)],
                "family_probabilities": {label: float(score) for label, score in zip(labels, scores)},
                "predicted_families": [labels[i] for i in np.flatnonzero(scores >= threshold)],
                "template_id": row["template_id"],
            }, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/family_specialist_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = list(manifest["family_labels"])
    paths = {s: args.data_dir / f"family_{s}.jsonl" for s in ("train", "val", "test")}
    rows = {s: read_rows(p) for s, p in paths.items()}
    audit = require_disjoint({s: [dict(r, holdout_group=r["template_id"]) for r in v] for s, v in rows.items()})
    targets = {s: targets_for(v, labels) for s, v in rows.items()}
    texts = {s: [r["command"].lower().strip() for r in v] for s, v in rows.items()}

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    pad = collate(tokenizer.pad_token_id)
    sets = {s: Rows(texts[s], targets[s].astype(np.float32), tokenizer, args.max_len) for s in rows}
    train_loader = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=pad)
    eval_loaders = {s: DataLoader(sets[s], batch_size=128, collate_fn=pad) for s in ("val", "test")}

    positives = targets["train"].sum(axis=0)
    pos_weight = torch.tensor(
        np.minimum(np.sqrt((len(targets["train"]) - positives) / np.maximum(positives, 1)), args.max_pos_weight),
        dtype=torch.float32, device=device,
    )
    model = FamilyModel(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(enabled=amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "family_codebert.pt"
    best_f1, best_epoch, history = -1.0, 0, []
    total_steps = len(train_loader)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (ids, mask, y) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(ids.to(device), mask.to(device))
            loss = loss_fn(logits.float(), y.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % args.log_every == 0 or step == total_steps:
                print(f"epoch {epoch}/{args.epochs} step {step}/{total_steps} loss={running / step:.4f} "
                      f"elapsed={(time.perf_counter() - started) / 60:.1f}m", flush=True)
        val_probs = predict(model, eval_loaders["val"], device, amp)
        val = metric_summary(targets["val"], val_probs, labels, 0.5)
        history.append({"epoch": epoch, "val_macro_f1_at_0.5": val["macro_f1"], "val_top1": val["top1_accuracy_any_true_family"]})
        print(f"[epoch {epoch}] val macro_f1={val['macro_f1']:.4f} top1={val['top1_accuracy_any_true_family']:.4f} "
              f"top2={val['top2_accuracy_any_true_family']:.4f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save(model.state_dict(), best_path)
            print(f"[+] saved best checkpoint (epoch {epoch})", flush=True)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    probabilities = {s: predict(model, eval_loaders[s], device, amp) for s in ("val", "test")}
    results = {s: metric_summary(targets[s], probabilities[s], labels, 0.5) for s in probabilities}
    for s in probabilities:
        export(args.output_dir / f"{s}_predictions.jsonl", rows[s], targets[s], probabilities[s], labels, 0.5)

    metadata = {
        "model_type": "multi_label_family_codebert", "family_labels": labels, "backbone": BACKBONE,
        "training_arguments": json.loads(json.dumps(vars(args), default=str)),
        "pos_weight": dict(zip(labels, [float(v) for v in pos_weight.cpu()])),
        "best_epoch": best_epoch, "history": history, "train_seconds": time.perf_counter() - started,
        "split_independence_audit": audit, "dataset_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "checkpoint_sha256": sha256_file(best_path), "metrics_at_threshold_0.5": results,
        "label_basis": "weak source-derived tactic families and gatekeeper benign labels; not human-verified",
        "technique_ids_in_model_or_rows": False,
    }
    (args.output_dir / "family_codebert.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "val_macro_f1": results["val"]["macro_f1"],
                      "test_macro_f1": results["test"]["macro_f1"],
                      "test_top1": results["test"]["top1_accuracy_any_true_family"],
                      "test_top2": results["test"]["top2_accuracy_any_true_family"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
