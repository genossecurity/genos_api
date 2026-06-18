import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "training" / "genos_behavior"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"


@dataclass
class Row:
    input_text: str
    stage_label: str
    action_tags: list[str]


class BehaviorDataset(Dataset):
    def __init__(self, rows, tokenizer, stage_map, action_map, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.stage_map = stage_map
        self.action_map = action_map
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row.input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        action_target = torch.zeros(len(self.action_map), dtype=torch.float32)
        for tag in row.action_tags:
            if tag in self.action_map:
                action_target[self.action_map[tag]] = 1.0
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "stage_target": torch.tensor(self.stage_map[row.stage_label], dtype=torch.long),
            "action_target": action_target,
        }


class BehaviorEncoderModel(nn.Module):
    def __init__(self, num_stages, num_actions):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
        self.dropout = nn.Dropout(0.2)
        self.stage_head = nn.Linear(768, num_stages)
        self.action_head = nn.Linear(768, num_actions)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.stage_head(pooled), self.action_head(pooled)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "behavior_encoder.pt")
    parser.add_argument(
        "--weighted-loss", action="store_true",
        help="Weight stage CrossEntropyLoss by inverse class frequency to address imbalance",
    )
    parser.add_argument(
        "--max-weight", type=float, default=8.0,
        help="Cap on any single class weight (prevents extreme gradients for very rare classes)",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            rows.append(Row(
                input_text=raw["input_text"],
                stage_label=raw["stage_label"],
                action_tags=list(raw.get("action_tags") or []),
            ))
    return rows


def load_maps():
    stage_map = json.loads((CONFIG_DIR / "behavior_stage_map.json").read_text(encoding="utf-8"))
    action_map = json.loads((CONFIG_DIR / "behavior_action_map.json").read_text(encoding="utf-8"))
    return stage_map, action_map


def evaluate(model, loader, device):
    model.eval()
    stage_preds = []
    stage_true = []
    action_true = []
    action_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stage_target = batch["stage_target"].to(device)
            action_target = batch["action_target"].to(device)

            stage_logits, action_logits = model(input_ids, attention_mask)
            stage_preds.extend(torch.argmax(stage_logits, dim=1).cpu().tolist())
            stage_true.extend(stage_target.cpu().tolist())

            action_true.append(action_target.cpu().numpy())
            action_pred.append((torch.sigmoid(action_logits) >= 0.5).cpu().numpy())

    action_true = np.concatenate(action_true, axis=0)
    action_pred = np.concatenate(action_pred, axis=0)
    return {
        "stage_acc": float(accuracy_score(stage_true, stage_preds)),
        "stage_macro_f1": float(f1_score(stage_true, stage_preds, average="macro", zero_division=0)),
        "action_micro_f1": float(f1_score(action_true, action_pred, average="micro", zero_division=0)),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    stage_map, action_map = load_maps()
    print(f"[+] Loaded label maps: stages={len(stage_map)} actions={len(action_map)}", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    print("[+] Loaded tokenizer: microsoft/codebert-base", flush=True)

    train_rows = load_rows(args.data_dir / "behavior_train.jsonl")
    val_rows = load_rows(args.data_dir / "behavior_val.jsonl")
    test_rows = load_rows(args.data_dir / "behavior_test.jsonl")
    print(
        f"[+] Loaded dataset rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}",
        flush=True,
    )

    train_ds = BehaviorDataset(train_rows, tokenizer, stage_map, action_map, args.max_length)
    val_ds = BehaviorDataset(val_rows, tokenizer, stage_map, action_map, args.max_length)
    test_ds = BehaviorDataset(test_rows, tokenizer, stage_map, action_map, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorEncoderModel(len(stage_map), len(action_map)).to(device)
    print(f"[+] Loaded model on device={device}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.weighted_loss:
        # Compute inverse-frequency weights from training set
        from collections import Counter
        stage_counts = Counter(r.stage_label for r in train_rows)
        total = sum(stage_counts.values())
        num_classes = len(stage_map)
        weights = torch.ones(num_classes, dtype=torch.float32)
        for stage_name, idx in stage_map.items():
            count = stage_counts.get(stage_name, 1)
            w = min(total / (num_classes * count), args.max_weight)
            weights[idx] = w
        weights = weights.to(device)
        print("[+] Stage loss weights (class: weight):")
        for stage_name, idx in sorted(stage_map.items(), key=lambda x: x[1]):
            print(f"    {stage_name:<30} {weights[idx].item():.3f}")
        stage_loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        stage_loss_fn = nn.CrossEntropyLoss()

    action_loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None
    total_batches = max(1, len(train_loader))
    progress_is_tty = sys.stderr.isatty()

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        running_loss = 0.0
        print(f"[+] Starting epoch {epoch}/{args.epochs} with {total_batches} batches", flush=True)

        loop = tqdm(
            train_loader,
            total=total_batches,
            desc=f"Epoch {epoch}/{args.epochs}",
            disable=not progress_is_tty,
        )
        for step, batch in enumerate(loop, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stage_target = batch["stage_target"].to(device)
            action_target = batch["action_target"].to(device)

            optimizer.zero_grad()
            stage_logits, action_logits = model(input_ids, attention_mask)
            stage_loss = stage_loss_fn(stage_logits, stage_target)
            action_loss = action_loss_fn(action_logits, action_target)
            loss = stage_loss + action_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(avg_loss=f"{running_loss / step:.4f}")

            if step == 1 or step % max(1, args.log_every) == 0 or step == total_batches:
                elapsed = time.perf_counter() - start
                avg_loss = running_loss / step
                print(
                    f"epoch={epoch} step={step}/{total_batches} avg_loss={avg_loss:.4f} secs={elapsed:.1f}",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device)
        score = val_metrics["stage_macro_f1"] + val_metrics["action_micro_f1"]
        if score > best_val:
            best_val = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        elapsed = time.perf_counter() - start
        print(
            f"epoch={epoch} loss={running_loss / max(1, len(train_loader)):.4f} "
            f"val_stage_acc={val_metrics['stage_acc']:.4f} "
            f"val_stage_macro_f1={val_metrics['stage_macro_f1']:.4f} "
            f"val_action_micro_f1={val_metrics['action_micro_f1']:.4f} "
            f"secs={elapsed:.1f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(json.dumps({"test": test_metrics}, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    meta = {
        "backbone": "microsoft/codebert-base",
        "stage_map": stage_map,
        "action_map": action_map,
        "test_metrics": test_metrics,
        "max_length": args.max_length,
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] Saved model to {args.output}", flush=True)
    print(f"[+] Saved metadata to {args.output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
