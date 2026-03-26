import base64
import json
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer

try:
    import pyminusone
except ImportError:
    pyminusone = None


class Tier1_Gatekeeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
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


class Tier2_Specialist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        logits = self.classifier(outputs)
        return logits


class GenosEngine:
    def __init__(
        self,
        t1_path="models/gatekeeper.pt",
        t2_path="models/specialist.pt",
        map_path="models/specialist_map.json",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_length = int(os.getenv("GENOS_MAX_TOKENS", "256"))

        self.t1 = Tier1_Gatekeeper().to(self.device)
        self.t1.load_state_dict(torch.load(t1_path, map_location=self.device), strict=False)
        self.t1.eval()

        with open(map_path, "r") as f:
            self.s_map = {int(v): k for k, v in json.load(f).items()}

        self.t2 = Tier2_Specialist(num_classes=len(self.s_map)).to(self.device)
        self.t2.load_state_dict(torch.load(t2_path, map_location=self.device), strict=False)
        self.t2.eval()

        # Keep bounded to avoid deobfuscation bombs while allowing deeper peeling.
        self.max_deobfuscation_layers = 5

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

        # Decode common PowerShell patterns like FromBase64String('...')
        text = self.decode_embedded_base64(text)

        # If this is an execution wrapper around a decoded payload, return payload only.
        payload_only = self.extract_powershell_payload(text)
        if payload_only:
            text = payload_only

        # Resolve common PowerShell [char] constructions into plain text.
        text = self.deobfuscate_char_constructions(text)

        if pyminusone:
            try:
                text = pyminusone.deobfuscate(text, lang="powershell")
            except Exception:
                pass

        # Run once more after AST simplification in case new [char] blocks appear.
        text = self.deobfuscate_char_constructions(text)

        return text

    def deobfuscate_char_constructions(self, text: str) -> str:
        # Pattern: (65..67) | % { [char]$_ } -> ABC
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

        # Pattern: [char]65 or [char](65) -> 'A'
        single_char_pattern = re.compile(r"\[char\]\s*\(?\s*(\d{1,3})\s*\)?", re.I)

        def _single_char(match):
            value = int(match.group(1))
            value = max(0, min(value, 255))
            return json.dumps(chr(value))

        return single_char_pattern.sub(_single_char, text)

    def extract_powershell_payload(self, text: str):
        payload = self._extract_invocation_payload(text)
        if payload is None:
            payload = text.strip()

        # Common decode wrapper after base64 replacement:
        # [System.Text.Encoding]::UTF8.GetString([System.Convert]::"...decoded...")
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
        pattern = re.compile(
            r"FromBase64String\(\s*['\"]([A-Za-z0-9+/=]{8,})['\"]\s*\)",
            re.I,
        )

        def _decode(match):
            b64_payload = match.group(1)
            try:
                decoded = base64.b64decode(b64_payload).decode("utf-8", errors="ignore")
                # Keep it as a quoted, escaped string so output remains parseable.
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

                # Stop if transformations no longer simplify the text.
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

        with torch.no_grad():
            g_logits = self.t1(inputs["input_ids"], inputs["attention_mask"])
            g_probs = F.softmax(g_logits, dim=1)
            g_conf, g_idx = torch.max(g_probs, dim=1)
            is_malicious = g_idx.item() == 1

            response = {
                "label": "Malicious" if is_malicious else "Benign",
                "label_confidence": round(g_conf.item(), 2),
                "deobfuscated_cmd": current_cmd if was_obfuscated else None,
            }

            if is_malicious:
                s_logits = self.t2(inputs["input_ids"], inputs["attention_mask"])
                s_probs = F.softmax(s_logits, dim=1).squeeze(0)
                top_vals, top_idxs = torch.topk(
                    s_probs, k=min(5, len(s_probs)), largest=True, sorted=True
                )

                response["MITRE_codes"] = [
                    {"code": self.s_map[idx.item()], "confidence": round(val.item(), 2)}
                    for idx, val in zip(top_idxs, top_vals)
                ]

        return response
