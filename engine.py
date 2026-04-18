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

import sys as _sys
_PARSER_DIR = os.path.join(BASE_DIR, "parser")
if _PARSER_DIR not in _sys.path:
    _sys.path.insert(0, _PARSER_DIR)

try:
    from parser import parse_command as _parse_command
    from semantic_features import build_semantic_features as _build_semantic_features
    from rule_engine import build_rule_result as _build_rule_result
    from candidate_mask import build_prior_vector as _build_prior_vector
    from build_residual_dataset import build_residual as _build_residual, build_feature_tags as _build_feature_tags
    _RESIDUAL_PIPELINE_AVAILABLE = True
except ImportError:
    _RESIDUAL_PIPELINE_AVAILABLE = False


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
    _TECHNIQUE_REASON_HINTS = {
        "T1490": [
            "Deletes shadow copies or backup artifacts",
            "Targets recovery paths or system restore mechanisms",
        ],
        "T1486": [
            "Shows impact-oriented destructive or recovery-inhibiting behavior",
        ],
        "T1140": [
            "Decodes or unwraps embedded payload content",
        ],
        "T1027": [
            "Uses encoded or obfuscated command content",
        ],
        "T1053": [
            "Creates or updates scheduled task execution",
        ],
        "T1547": [
            "Modifies autorun locations for persistence",
        ],
        "T1083": [
            "Enumerates files or directories on the local system",
        ],
        "T1018": [
            "Performs local network discovery activity",
        ],
        "T1087": [
            "Enumerates local or domain account information",
        ],
    }

    def __init__(
        self,
        t1_path="models/gatekeeper.pt",
        t2_path="models/specialist_residual_a.pt",
        map_path=None,
        raw_mitre_path="data/training/mitre_atlas_raw.csv",
        gatekeeper_meta_path=None,
        use_residual_format=True,
        prior_alphas=None,
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
        self.use_residual_format = use_residual_format and _RESIDUAL_PIPELINE_AVAILABLE
        self.prior_alphas = prior_alphas or {"strong": 2.0, "weak": 1.5, "none": 0.0}
        # Forward map {mitre_id: int_index} used by build_prior_vector
        self._specialist_map_fwd = {mitre: idx for idx, mitre in self.s_map.items()}

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

    # ── Evidence helpers ──────────────────────────────────────────────────

    # Flags worth surfacing to analysts (ignore single-letter noise unless meaningful)
    _HIGH_SIGNAL_FLAGS = frozenset({
        "-enc", "-encodedcommand", "-e", "-c", "/c", "-nop", "-noni",
        "-noninteractive", "-windowstyle", "-w", "-exec", "-executionpolicy",
        "-ep", "-bypass", "-command", "/create", "/sc", "/tr", "/tn",
        "/f", "/d", "/t", "/v", "/s", "/add", "/delete",
        "-split", "-f", "-o", "--output", "-urlcache",
        "-decode", "-encode", "-decodefile", "-p", "--post-file",
        "-b64", "--allow-overwrite", "--on-download-complete",
        "-perm", "+4000", "+2000", "-rf", "--flush",
        "/set", "/get", "/query", "/export", "/import",
    })

    # Curated semantic feature labels to surface (boolean True ones)
    _SURFACE_SEM_FEATURES = frozenset({
        "downloads_remote_resource",
        "writes_executable_like_file",
        "modifies_registry_autorun",
        "creates_scheduled_task",
        "creates_or_modifies_service",
        "archive_create",
        "archive_extract",
        "deletes_shadow_copies",
        "remote_execution_or_session",
        "transfers_file_to_remote",
        "runs_interpreter",
        "executes_inline_code",
        "enumerates_identity",
        "enumerates_network_config",
        "reads_credential_store",
        "uses_encoded_payload",
        "uses_obfuscation",
        "uses_signed_proxy_binary",
    })

    # Interpreter detection
    _INTERPRETER_NAMES = {
        "bash": "bash", "sh": "sh", "zsh": "zsh", "fish": "fish",
        "python": "python", "python3": "python", "py": "python",
        "perl": "perl", "ruby": "ruby", "php": "php",
        "node": "node", "node.exe": "node",
        "powershell": "powershell", "powershell.exe": "powershell",
        "pwsh": "powershell", "cmd": "cmd", "cmd.exe": "cmd",
        "wscript": "wscript", "cscript": "cscript",
        "mshta": "mshta", "mshta.exe": "mshta",
    }

    def _build_evidence(self, parsed: dict, sem: dict, rule_result: dict,
                        was_obfuscated: bool = False,
                        deobfuscated_cmd: str | None = None) -> dict:
        """Build curated evidence dict from pipeline outputs."""
        exe = (parsed.get("executable") or "").lower()
        flags = parsed.get("flags") or []

        # ── Execution identity ────────────────────────────────────────
        platform = parsed.get("platform") or "unknown"
        interpreter = (
            self._INTERPRETER_NAMES.get(exe)
            or (parsed.get("interpreter_markers") or [None])[0]
            or None
        )

        # ── High-signal flags ─────────────────────────────────────────
        high_signal_flags = sorted({
            f.lower() for f in flags
            if f.lower() in self._HIGH_SIGNAL_FLAGS
        })

        # ── Structural behavior ───────────────────────────────────────
        has_pipe     = bool(parsed.get("has_pipe"))
        has_redirect = bool(parsed.get("has_redirect"))
        has_chain    = bool(parsed.get("has_chain"))
        inline_code  = bool(parsed.get("inline_code")) or bool(sem.get("executes_inline_code"))

        # ── Obfuscation ───────────────────────────────────────────────
        uses_encoded_payload  = bool(sem.get("uses_encoded_payload"))
        uses_obfuscation_flag = bool(sem.get("uses_obfuscation")) or was_obfuscated
        obfuscation_markers   = list(parsed.get("encoded_markers") or []) + list(parsed.get("obfuscation_markers") or [])
        deob_cmd = deobfuscated_cmd or parsed.get("deobfuscated_command") or None

        # ── LOLBin ────────────────────────────────────────────────────
        lolbin_matches = list(parsed.get("lolbin_matches") or [])
        if exe and exe in {
            "certutil", "mshta", "rundll32", "regsvr32", "wmic", "bitsadmin",
            "powershell", "powershell.exe", "cmd", "cmd.exe", "wscript", "cscript",
            "bash", "sh", "curl", "wget",
        }:
            if exe not in lolbin_matches:
                lolbin_matches.insert(0, exe)
        uses_signed_proxy = bool(sem.get("uses_signed_proxy_binary")) or bool(lolbin_matches)

        # ── Semantic features (curated) ───────────────────────────────
        semantic_features = [
            k for k in self._SURFACE_SEM_FEATURES
            if sem.get(k)
        ]

        # ── Rule metadata ─────────────────────────────────────────────
        rule_strength = rule_result.get("rule_strength", "none")
        raw_rules = rule_result.get("fired_rules") or []
        fired_rules = [r.replace("_rule_", "").replace("_", " ") for r in raw_rules]

        # ── Evidence summary sentence ─────────────────────────────────
        evidence_summary = self._generate_evidence_summary(
            exe, platform, sem, rule_strength, fired_rules
        )

        # ── Derived: primary_artifact_type ────────────────────────────
        primary_artifact_type = None
        if parsed.get("registry_paths"):
            primary_artifact_type = "registry"
        elif sem.get("creates_scheduled_task"):
            primary_artifact_type = "task"
        elif sem.get("creates_or_modifies_service"):
            primary_artifact_type = "service"
        elif sem.get("archive_create") or sem.get("archive_extract"):
            primary_artifact_type = "archive"
        elif (parsed.get("urls") or parsed.get("remote_targets") or
              sem.get("downloads_remote_resource")):
            primary_artifact_type = "network"
        elif sem.get("runs_interpreter") or sem.get("executes_inline_code"):
            primary_artifact_type = "script"
        elif parsed.get("file_paths"):
            primary_artifact_type = "file"

        # ── Derived: execution_style ──────────────────────────────────
        execution_style = None
        if sem.get("downloads_remote_resource") and (
                sem.get("executes_inline_code") or has_pipe):
            execution_style = "download-and-execute"
        elif sem.get("creates_scheduled_task"):
            execution_style = "scheduled"
        elif sem.get("remote_execution_or_session"):
            execution_style = "remote-session"
        elif sem.get("executes_inline_code") or inline_code:
            execution_style = "inline"
        elif sem.get("downloads_remote_resource"):
            execution_style = "download-and-execute"

        return {
            # Execution identity
            "platform":           platform,
            "executable":         parsed.get("executable") or None,
            "subcommand":         parsed.get("subcommand") or None,
            "interpreter":        interpreter,
            # High-signal flags
            "high_signal_flags":  high_signal_flags,
            # Targets / artifacts
            "file_paths":         list(parsed.get("file_paths") or []),
            "registry_paths":     list(parsed.get("registry_paths") or []),
            "local_targets":      list(parsed.get("local_targets") or []),
            "remote_targets":     list(parsed.get("remote_targets") or []),
            # Network indicators
            "urls":               list(parsed.get("urls") or []),
            "domains":            list(parsed.get("domains") or []),
            "ips":                list(parsed.get("ips") or []),
            "ports":              list(parsed.get("ports") or []),
            # Structural behavior
            "has_pipe":           has_pipe,
            "has_redirect":       has_redirect,
            "has_chain":          has_chain,
            "inline_code":        inline_code,
            # Obfuscation / encoding
            "uses_encoded_payload":  uses_encoded_payload,
            "uses_obfuscation":      uses_obfuscation_flag,
            "obfuscation_markers":   obfuscation_markers,
            "deobfuscated_command":  deob_cmd,
            # LOLBin
            "lolbin_matches":            lolbin_matches,
            "uses_signed_proxy_binary":  uses_signed_proxy,
            # Semantic features
            "semantic_features":  semantic_features,
            # Rule / reasoning metadata
            "rule_strength":      rule_strength,
            "fired_rules":        fired_rules,
            "evidence_summary":   evidence_summary,
            # Derived
            "primary_artifact_type": primary_artifact_type,
            "execution_style":       execution_style,
        }

    def _generate_evidence_summary(self, exe: str, platform: str,
                                   sem: dict, rule_strength: str,
                                   fired_rules: list) -> str:
        """Generate a compact analyst-facing evidence sentence."""
        parts = []
        if exe:
            parts.append(exe)
        if sem.get("uses_encoded_payload") or sem.get("uses_obfuscation"):
            parts.append("encoded/obfuscated execution")
        if sem.get("downloads_remote_resource"):
            parts.append("remote resource download")
        if sem.get("executes_inline_code"):
            parts.append("inline code execution")
        if sem.get("modifies_registry_autorun"):
            parts.append("registry autorun persistence")
        if sem.get("creates_scheduled_task"):
            parts.append("scheduled task creation")
        if sem.get("deletes_shadow_copies"):
            parts.append("shadow copy deletion")
        if sem.get("enumerates_identity"):
            parts.append("account enumeration")
        if sem.get("enumerates_network_config"):
            parts.append("network discovery")
        if sem.get("reads_credential_store"):
            parts.append("credential access")
        if sem.get("remote_execution_or_session"):
            parts.append("remote execution")
        if not parts:
            if fired_rules:
                parts.append(fired_rules[0] + " behavior")
            else:
                return "No distinctive behaviors detected."
        summary = (exe.capitalize() + " " if exe else "") + ", ".join(parts[1:] or ["execution"]) + "."
        return summary.strip()

    def _build_mapping_reasons(self, top_code: str | None, evidence: dict) -> list[str]:
        reasons = []
        exe = evidence.get("executable")
        semantic = set(evidence.get("semantic_features") or [])
        fired_rules = list(evidence.get("fired_rules") or [])

        if top_code:
            reasons.extend(self._TECHNIQUE_REASON_HINTS.get(top_code, []))

        if exe and evidence.get("uses_signed_proxy_binary"):
            reasons.append(f"Uses {exe} as a signed proxy binary")
        if "deletes_shadow_copies" in semantic:
            reasons.append("Deletes shadow copies or backup restore points")
        if "modifies_registry_autorun" in semantic:
            reasons.append("Modifies registry autorun paths for persistence")
        if "creates_scheduled_task" in semantic:
            reasons.append("Creates scheduled execution for follow-on activity")
        if "downloads_remote_resource" in semantic:
            reasons.append("Retrieves content from a remote location")
        if "remote_execution_or_session" in semantic:
            reasons.append("Establishes or uses remote execution paths")
        if evidence.get("uses_encoded_payload"):
            reasons.append("Carries encoded command content")
        if evidence.get("uses_obfuscation"):
            reasons.append("Includes obfuscation markers consistent with evasion")
        if evidence.get("high_signal_flags"):
            flag_sample = ", ".join(evidence["high_signal_flags"][:2])
            reasons.append(f"Invokes high-signal flags such as {flag_sample}")
        if fired_rules:
            reasons.append(f"Triggers rule logic for {fired_rules[0]}")

        deduped = []
        seen = set()
        for reason in reasons:
            if reason not in seen:
                deduped.append(reason)
                seen.add(reason)
            if len(deduped) == 3:
                break
        return deduped

    def _build_why_mapped(self, top_code: str | None, mapping_reasons: list[str]) -> str | None:
        if not mapping_reasons:
            return None
        prefix = f"Mapped to {top_code} because " if top_code else "Mapped based on "
        if len(mapping_reasons) == 1:
            return prefix + mapping_reasons[0].lower() + "."
        return prefix + mapping_reasons[0].lower() + " and " + mapping_reasons[1].lower() + "."

    def _build_ioc_summary(self, evidence: dict) -> dict:
        file_paths = list(evidence.get("file_paths") or [])
        notable_files = [
            path for path in file_paths
            if re.search(r"\.(?:exe|dll|ps1|bat|cmd|sh|so|bin|zip|7z|tar|gz|jar)$", path, re.I)
        ]
        if not notable_files:
            notable_files = file_paths[:3]

        return {
            "domains": list(evidence.get("domains") or [])[:5],
            "ips": list(evidence.get("ips") or [])[:5],
            "urls": list(evidence.get("urls") or [])[:5],
            "notable_files": notable_files[:5],
            "registry_paths": list(evidence.get("registry_paths") or [])[:5],
        }

    def _derive_confidence_driver(self, rule_result: dict | None) -> str:
        if not rule_result:
            return "Model-led"
        strength = rule_result.get("rule_strength", "none")
        if strength == "strong":
            return "Rule-reinforced"
        if strength == "weak":
            return "Rule-supported"
        return "Model-led"

    def _build_analyst_hint(self, top_code: str | None, evidence: dict) -> str | None:
        semantic = set(evidence.get("semantic_features") or [])
        if "deletes_shadow_copies" in semantic or top_code in {"T1490", "T1486"}:
            return "This behavior is commonly associated with recovery inhibition and destructive impact activity."
        if "reads_credential_store" in semantic:
            return "This pattern is often seen in credential access workflows."
        if "modifies_registry_autorun" in semantic or "creates_scheduled_task" in semantic:
            return "This pattern is often seen in persistence setup."
        if "enumerates_identity" in semantic or top_code == "T1087":
            return "This command appears consistent with account discovery activity."
        if "enumerates_network_config" in semantic or top_code == "T1018":
            return "This command appears consistent with host or network discovery activity."
        if "downloads_remote_resource" in semantic and "executes_inline_code" in semantic:
            return "This pattern is commonly used to fetch and immediately execute a payload."
        if "remote_execution_or_session" in semantic:
            return "This pattern is often seen in remote execution or lateral movement chains."
        if evidence.get("uses_encoded_payload") or evidence.get("uses_obfuscation"):
            return "This command uses concealment patterns that are commonly associated with evasive execution."
        return None

    def _build_response_enrichment(self, top_code: str | None, evidence: dict, rule_result: dict | None) -> dict:
        mapping_reasons = self._build_mapping_reasons(top_code, evidence)
        return {
            "mapping_reasons": mapping_reasons,
            "why_mapped": self._build_why_mapped(top_code, mapping_reasons),
            "ioc_summary": self._build_ioc_summary(evidence),
            "confidence_driver": self._derive_confidence_driver(rule_result),
            "analyst_hint": self._build_analyst_hint(top_code, evidence),
        }

    def _build_variant_a_text(self, cmd: str):
        """
        Build Variant A specialist input text and return (text, rule_result).
        Format matches training exactly:
          RAW: {cmd}
          RESIDUAL: {residual}
          FEATURES: {tags}   (line omitted when no tags fire)
        """
        parsed = _parse_command(cmd)
        sem = _build_semantic_features(parsed)
        rules = _build_rule_result(parsed, sem)
        residual = _build_residual(parsed, sem, rules)
        feature_tags = _build_feature_tags(sem, rules)
        parts = [f"RAW: {cmd}", f"RESIDUAL: {residual}"]
        if feature_tags:
            parts.append(f"FEATURES: {' '.join(feature_tags)}")
        return "\n".join(parts), rules

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
                    # Build T2 input: Variant A format when pipeline is available
                    if self.use_residual_format:
                        t2_text, rule_result = self._build_variant_a_text(raw_cmd.strip())
                        t2_inputs = self.tokenizer(
                            t2_text,
                            return_tensors="pt",
                            truncation=True,
                            padding="max_length",
                            max_length=self.max_length,
                        ).to(self.device)
                    else:
                        t2_inputs = inputs
                        rule_result = None

                    s_logits = self.t2(t2_inputs["input_ids"], t2_inputs["attention_mask"])

                    # Soft prior fusion — additive only, no hard masking
                    if rule_result is not None:
                        pv = _build_prior_vector(
                            rule_result,
                            self._specialist_map_fwd,
                            alpha_overrides=self.prior_alphas,
                        )
                        pv_tensor = torch.tensor(
                            pv["prior_vector"], dtype=s_logits.dtype, device=self.device
                        ).unsqueeze(0)
                        s_logits = s_logits + pv_tensor

                    s_probs = F.softmax(s_logits / 0.5, dim=1).squeeze(0)

                    top_vals, top_idxs = torch.topk(
                        s_probs, k=min(3, len(s_probs)), largest=True, sorted=True
                    )

                    response["MITRE_codes"] = [
                        {"code": self.s_map[idx.item()], "confidence": round(val.item() * 100, 2)}
                        for idx, val in zip(top_idxs, top_vals)
                    ]

                    # Build evidence from parser pipeline
                    if self.use_residual_format and rule_result is not None:
                        try:
                            _parsed_ev = _parse_command(raw_cmd.strip())
                            _sem_ev    = _build_semantic_features(_parsed_ev)
                            response["evidence"] = self._build_evidence(
                                _parsed_ev, _sem_ev, rule_result,
                                was_obfuscated=was_obfuscated,
                                deobfuscated_cmd=current_cmd if was_obfuscated else None,
                            )
                            top_code = response["MITRE_codes"][0]["code"] if response["MITRE_codes"] else None
                            response.update(
                                self._build_response_enrichment(top_code, response["evidence"], rule_result)
                            )
                        except Exception:
                            pass

        return response
