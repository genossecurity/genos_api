import base64
import csv
import json
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import RobertaModel, RobertaTokenizer

try:
    import pyminusone
except ImportError:
    pyminusone = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_asset_path(path_value: str, fallback_relpaths: list[str] | None = None) -> str:
    """Resolve asset path with support for multiple fallbacks."""
    if os.path.isabs(path_value):
        if os.path.exists(path_value):
            return path_value
    else:
        cwd_candidate = os.path.join(os.getcwd(), path_value)
        if os.path.exists(cwd_candidate):
            return cwd_candidate

        base_candidate = os.path.join(BASE_DIR, path_value)
        if os.path.exists(base_candidate):
            return base_candidate

    if fallback_relpaths:
        for fallback in fallback_relpaths if isinstance(fallback_relpaths, list) else [fallback_relpaths]:
            fallback_candidate = os.path.join(BASE_DIR, fallback)
            if os.path.exists(fallback_candidate):
                return fallback_candidate

    return path_value


class Tier1_Gatekeeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits


class _MeanPool(nn.Module):
    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class Tier2_Specialist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
        self.pool = _MeanPool()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        return logits


class GenosEngine:
    def __init__(
        self,
        t1_path="models/gatekeeper.pt",
        t2_path="models/specialist.pt",
        map_path=None,
        raw_mitre_path="data/training/mitre_atlas_raw.csv",
        gatekeeper_meta_path=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_length = int(os.getenv("GENOS_MAX_TOKENS", "256"))

        t1_path = _resolve_asset_path(t1_path, ["models/gatekeeper.pt"])
        t2_path = _resolve_asset_path(t2_path, ["models/specialist.pt"])

        # Prefer explicit specialist map JSON when provided (backward compatibility).
        map_candidates = ["config/specialist_map.json", "models/specialist_map.json"]
        if map_path:
            map_candidates = [map_path] + map_candidates

        resolved_map_path = None
        for candidate in map_candidates:
            resolved = _resolve_asset_path(candidate)
            if os.path.exists(resolved):
                resolved_map_path = resolved
                break

        if resolved_map_path:
            self.s_map = self._load_map_from_json(resolved_map_path)
        else:
            raw_csv_path = _resolve_asset_path(
                raw_mitre_path,
                [
                    "data/art/mitre_atlas_raw.csv"
                ],
            )
            self.s_map = self._build_map_from_csv(raw_csv_path)

        meta_candidates = ["config/gatekeeper_meta.json"]
        if gatekeeper_meta_path:
            meta_candidates = [gatekeeper_meta_path] + meta_candidates

        self.gatekeeper_threshold = None
        self.gatekeeper_threshold_source = None
        for candidate in meta_candidates:
            resolved = _resolve_asset_path(candidate, ["config/gatekeeper_meta.json"])
            if os.path.exists(resolved):
                threshold = self._load_gatekeeper_threshold(resolved)
                if threshold is not None:
                    self.gatekeeper_threshold = float(threshold)
                    self.gatekeeper_threshold_source = resolved
                    break

        self.t1 = Tier1_Gatekeeper().to(self.device)
        self.t1.load_state_dict(torch.load(t1_path, map_location=self.device, weights_only=True), strict=False)
        self.t1.eval()

        self.t2 = Tier2_Specialist(num_classes=len(self.s_map)).to(self.device)
        self.t2.load_state_dict(torch.load(t2_path, map_location=self.device, weights_only=True), strict=False)
        self.t2.eval()

        self.max_deobfuscation_layers = 5

    def _load_map_from_json(self, json_path: str) -> dict:
        """Load specialist label map from JSON file as {int_index: mitre_id}."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        return {int(v): k for k, v in raw_map.items()}

    def _load_gatekeeper_threshold(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None

        if isinstance(meta, dict):
            if "threshold" in meta:
                return meta["threshold"]
            for key in ("test_metrics", "val_metrics"):
                if isinstance(meta.get(key), dict) and "threshold" in meta[key]:
                    return meta[key]["threshold"]
        return None

    def _build_map_from_csv(self, csv_path: str) -> dict:
        """Reads the raw MITRE CSV, extracts unique IDs, sorts them, and maps them to ints."""
        unique_ids = set()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot build specialist map. Missing: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "mitre_id" in row and row["mitre_id"].strip():
                    unique_ids.add(row["mitre_id"].strip())

        sorted_ids = sorted(list(unique_ids))
        return {i: mitre_id for i, mitre_id in enumerate(sorted_ids)}

    def calculate_entropy(self, text):
        if not text:
            return 0
        entropy = 0
        for x in range(256):
            p_x = float(text.count(chr(x))) / len(text)
            if p_x > 0:
                entropy += -p_x * math.log(p_x, 2)
        return entropy

    def is_obfuscated(self, text: str) -> bool:
        patterns = [
            r"\[char\]",
            r"base64",
            r"frombase64",
            r"reverse\(",
            r"\+[ ]*'",
            r"\$[a-z0-9_]{10,}",
            r"\\x[0-9a-f]{2}",
        ]
        if any(re.search(p, text, re.I) for p in patterns):
            return True
        if self.calculate_entropy(text) > 5.2:
            return True
        return False

    def deobfuscate_layer(self, text: str) -> str:
        text = self.universal_decoder(text)
        text = self.decode_embedded_base64(text)

        payload_only = self.extract_powershell_payload(text)
        if payload_only:
            text = payload_only

        text = self.deobfuscate_char_constructions(text)
        text = self.clean_concatenation(text)

        if pyminusone:
            try:
                text = pyminusone.deobfuscate(text, lang="powershell")
            except Exception:
                pass

        text = self.deobfuscate_char_constructions(text)
        text = self.clean_concatenation(text)

        return text

    def deobfuscate_char_constructions(self, text: str) -> str:
        range_loop_pattern = re.compile(
            r"\(\s*(\d{1,3})\s*\.\.\s*(\d{1,3})\s*\)\s*\|\s*%\s*\{\s*\[char\]\s*\$_\s*\}",
            re.I,
        )

        def _range_to_chars(match):
            start = int(match.group(1))
            end = int(match.group(2))
            if start > end:
                start, end = end, start
            start = max(0, min(start, 255))
            end = max(0, min(end, 255))
            return "".join(chr(i) for i in range(start, end + 1))

        text = range_loop_pattern.sub(lambda m: json.dumps(_range_to_chars(m)), text)

        single_char_pattern = re.compile(r"\[char\]\s*\(?\s*(\d{1,3})\s*\)?", re.I)

        def _single_char(match):
            value = int(match.group(1))
            value = max(0, min(value, 255))
            return json.dumps(chr(value))

        text = single_char_pattern.sub(_single_char, text)

        mixed_concat_pattern = re.compile(
            r"\(\s*(\d{1,3})\s*\.\.\s*(\d{1,3})\s*\)\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|\s*%\s*\{\s*\[char\]\s*\$_\s*\}",
            re.I,
        )

        def _mixed_concat(match):
            start = int(match.group(1))
            end = int(match.group(2))
            suffix = match.group(3)
            lead = chr(max(0, min(start, 255)))
            if abs(start - end) <= 32:
                return json.dumps(f"{lead}{suffix}")
            step = 1 if end >= start else -1
            decoded = "".join(chr(max(0, min(i, 255))) for i in range(start, end + step, step))
            return json.dumps(f"{decoded}{suffix}")

        return mixed_concat_pattern.sub(_mixed_concat, text)

    def clean_concatenation(self, text: str) -> str:
        quoted_join_pattern = re.compile(r"\"((?:\\.|[^\"\\])*)\"\s*\+\s*\"((?:\\.|[^\"\\])*)\"")

        while True:
            new_text = quoted_join_pattern.sub(lambda m: json.dumps(m.group(1) + m.group(2)), text)
            if new_text == text:
                break
            text = new_text

        q_plus_word = re.compile(r"\"((?:\\.|[^\"\\])*)\"\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)")
        text = q_plus_word.sub(lambda m: json.dumps(m.group(1) + m.group(2)), text)

        return text

    def extract_powershell_payload(self, text: str):
        payload = self._extract_invocation_payload(text)
        if payload is None:
            payload = text.strip()

        utf8_match = re.match(
            r"^\s*\[System\.Text\.Encoding\]::UTF8\.GetString\(\s*\[System\.Convert\]::(?P<quoted>(?:\"(?:\\.|[^\"\\])*\")|(?:'(?:\\.|[^'\\])*'))\s*\)\s*$",
            payload,
            re.I,
        )
        if utf8_match:
            quoted = utf8_match.group("quoted")
            if quoted.startswith('"'):
                try:
                    return json.loads(quoted).strip()
                except Exception:
                    return quoted.strip('"').strip()
            return quoted.strip("'").strip()

        return payload if payload != text.strip() else None

    def _extract_invocation_payload(self, text: str):
        s = text.strip()
        if not s.startswith("&("):
            return None

        builder_start = s.find("(")
        builder_end = self._find_matching_paren(s, builder_start)
        if builder_end == -1:
            return None

        idx = builder_end + 1
        while idx < len(s) and s[idx].isspace():
            idx += 1

        if idx >= len(s) or s[idx] != "(":
            return None

        payload_end = self._find_matching_paren(s, idx)
        if payload_end == -1:
            return None

        if s[payload_end + 1 :].strip():
            return None

        payload = s[idx + 1 : payload_end].strip()
        return payload or None

    def _find_matching_paren(self, text: str, start_index: int) -> int:
        if start_index < 0 or start_index >= len(text) or text[start_index] != "(":
            return -1

        depth = 0
        in_single = False
        in_double = False

        i = start_index
        while i < len(text):
            ch = text[i]

            if ch == "`":
                i += 2
                continue

            if in_single:
                if ch == "'":
                    in_single = False
                i += 1
                continue

            if in_double:
                if ch == '"':
                    in_double = False
                i += 1
                continue

            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i

            i += 1

        return -1

    def decode_embedded_base64(self, text: str) -> str:
        pattern = re.compile(r"FromBase64String\(\s*['\"]([A-Za-z0-9+/=]{8,})['\"]\s*\)", re.I)

        def _decode(match):
            b64_payload = match.group(1)
            try:
                decoded = base64.b64decode(b64_payload).decode("utf-8", errors="ignore")
                return json.dumps(decoded)
            except Exception:
                return match.group(0)

        return pattern.sub(_decode, text)

    def universal_decoder(self, text: str) -> str:
        try:
            if re.match(
                r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$",
                text,
            ):
                decoded = base64.b64decode(text).decode("utf-8", errors="ignore")
                if len(decoded) > 3:
                    return decoded
        except Exception:
            pass
        return text

    def _tier1_decision(self, probs: torch.Tensor):
        probs = probs.squeeze(0)
        benign_prob = float(probs[0].item())
        malicious_prob = float(probs[1].item())

        if self.gatekeeper_threshold is not None:
            is_malicious = malicious_prob >= self.gatekeeper_threshold
            decision_mode = "threshold"
            threshold_used = self.gatekeeper_threshold
        else:
            is_malicious = malicious_prob >= benign_prob
            decision_mode = "argmax"
            threshold_used = None

        label_conf = malicious_prob if is_malicious else benign_prob
        return {
            "is_malicious": is_malicious,
            "benign_prob": benign_prob,
            "malicious_prob": malicious_prob,
            "label_conf": label_conf,
            "decision_mode": decision_mode,
            "threshold_used": threshold_used,
        }

    def scan(self, raw_cmd):
        current_cmd = raw_cmd.strip()
        was_obfuscated = self.is_obfuscated(current_cmd)

        prev_entropy = self.calculate_entropy(current_cmd)

        for _ in range(self.max_deobfuscation_layers):
            if self.is_obfuscated(current_cmd):
                new_cmd = self.deobfuscate_layer(current_cmd)
                if new_cmd == current_cmd:
                    break
                current_cmd = new_cmd

                new_entropy = self.calculate_entropy(current_cmd)
                if abs(prev_entropy - new_entropy) < 0.01:
                    break
                prev_entropy = new_entropy
            else:
                break

        processed_cmd = current_cmd.lower().strip()
        inputs = self.tokenizer(
            processed_cmd,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)

        device_type = "cuda" if "cuda" in self.device.type else "cpu"
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with torch.no_grad():
            with autocast(device_type=device_type, dtype=autocast_dtype):
                g_logits = self.t1(inputs["input_ids"], inputs["attention_mask"])
                g_probs = F.softmax(g_logits, dim=1)
                gate = self._tier1_decision(g_probs)

                response = {
                    "label": "Malicious" if gate["is_malicious"] else "Benign",
                    "label_confidence": round(gate["label_conf"] * 100, 2),
                    "label_probabilities": {
                        "benign": round(gate["benign_prob"] * 100, 2),
                        "malicious": round(gate["malicious_prob"] * 100, 2),
                    },
                    "gatekeeper": {
                        "decision_mode": gate["decision_mode"],
                        "threshold_used": gate["threshold_used"],
                        "threshold_source": self.gatekeeper_threshold_source,
                    },
                    "deobfuscated_cmd": current_cmd if was_obfuscated else None,
                }

                if gate["is_malicious"]:
                    s_logits = self.t2(inputs["input_ids"], inputs["attention_mask"])
                    s_probs = F.softmax(s_logits / 0.5, dim=1).squeeze(0)

                    top_vals, top_idxs = torch.topk(
                        s_probs, k=min(3, len(s_probs)), largest=True, sorted=True
                    )

                    response["MITRE_codes"] = [
                        {"code": self.s_map[idx.item()], "confidence": round(val.item() * 100, 2)}
                        for idx, val in zip(top_idxs, top_vals)
                    ]

        return response
