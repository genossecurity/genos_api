"""
specialist.py — Tier 2 Specialist engine for Genos.

Covers the 11 tactic heads/families in the MITRE ATT&CK chain:
1.  Execution
2.  Persistence
3.  Privilege Escalation
4.  Defense Evasion
5.  Credential Access
6.  Discovery
7.  Lateral Movement
8.  Command-and-Control / Payload Retrieval
9.  Exfiltration
10. Impact
11. Benign Admin

Also performs behavior stage & action inference (via BehaviorEncoderModel or heuristic fallback),
as well as indicator / evidence extraction.
"""

import csv
import json
import math
import os
import re
import sys
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from .scientific_validation import sha256_file
from .huggingface import pretrained_kwargs, resolve_backbone
from .evidence import (
    HIGH_SIGNAL_FLAGS as _HIGH_SIGNAL_FLAGS,
    INTERPRETER_NAMES as _INTERPRETER_NAMES,
    SURFACE_SEM_FEATURES as _SURFACE_SEM_FEATURES,
    build_evidence as _build_evidence_fn,
    collect_indicator_evidence as _collect_indicator_evidence_fn,
    generate_evidence_summary as _generate_evidence_summary_fn,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PARSER_DIR = os.path.join(BASE_DIR, "parser")
if _PARSER_DIR not in sys.path:
    sys.path.insert(0, _PARSER_DIR)

try:
    from parser import parse_command as _parse_command
    from semantic_features import build_semantic_features as _build_semantic_features
    from rule_engine import build_rule_result as _build_rule_result
    from build_residual_dataset import build_residual as _build_residual, build_feature_tags as _build_feature_tags
    _RESIDUAL_PIPELINE_AVAILABLE = True
except ImportError:
    _RESIDUAL_PIPELINE_AVAILABLE = False


# ── The 11 MITRE ATT&CK Chain Families ───────────────────────────────────────

FAMILY_LABELS = [
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Command-and-Control / Payload Retrieval",
    "Exfiltration",
    "Impact",
    "Benign Admin",
]


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


def _resolve_behavior_model_path(primary_path: str | None = None) -> str:
    """Resolve exactly the requested checkpoint; never silently substitute one."""
    return _resolve_asset_path(primary_path or "models/behavior_encoder.pt")


# ── Specialist Model Architectures ───────────────────────────────────────────

class _MeanPool(nn.Module):
    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class Tier2_Specialist(nn.Module):
    def __init__(self, num_classes, backbone_path="microsoft/codebert-base", local_files_only=False):
        super().__init__()
        config = RobertaConfig.from_pretrained(
            backbone_path, **pretrained_kwargs(backbone_path, local_files_only)
        )
        self.encoder = RobertaModel(config)
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


class BehaviorEncoderModel(nn.Module):
    def __init__(self, num_stages: int, num_actions: int, backbone_path="microsoft/codebert-base", local_files_only=False):
        super().__init__()
        config = RobertaConfig.from_pretrained(
            backbone_path, **pretrained_kwargs(backbone_path, local_files_only)
        )
        self.encoder = RobertaModel(config)
        self.dropout = nn.Dropout(0.2)
        self.stage_head = nn.Linear(768, num_stages)
        self.action_head = nn.Linear(768, num_actions)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.stage_head(pooled), self.action_head(pooled)


# ── Action & Stage Mapping Tables ─────────────────────────────────────────────

_SEMANTIC_TO_ACTION_TAG = {
    "downloads_remote_resource": "download_remote_resource",
    "transfers_file_to_remote": "transfer_file_remote",
    "creates_scheduled_task": "establish_persistence",
    "modifies_registry_autorun": "establish_persistence",
    "creates_or_modifies_service": "establish_persistence",
    "remote_execution_or_session": "remote_session",
    "enumerates_identity": "enumerate_identity",
    "enumerates_network_config": "enumerate_network",
    "reads_credential_store": "access_credential_store",
    "deletes_shadow_copies": "delete_shadow_copies",
    "uses_encoded_payload": "encode_payload",
    "uses_obfuscation": "obfuscate_command",
    "runs_interpreter": "run_script_interpreter",
    "executes_inline_code": "execute_inline_code",
    "uses_signed_proxy_binary": "use_signed_proxy",
    "archive_create": "archive_or_stage_data",
    "archive_extract": "extract_archive",
    "writes_executable_like_file": "drop_executable",
}

_RULE_TAG_TO_ACTION_TAG = {
    "curl_pipe_shell": "pipe_to_shell",
    "reverse_shell": "reverse_shell",
    "shadow_copy_delete": "delete_shadow_copies",
    "certutil_download": "download_remote_resource",
    "bitsadmin_download": "download_remote_resource",
    "encoded_ps_inline": "encode_payload",
    "powershell_enc": "encode_payload",
    "mshta_inline": "use_signed_proxy",
    "schtasks_create": "establish_persistence",
    "registry_autorun": "establish_persistence",
    "recon_whoami": "enumerate_identity",
    "recon_net_user": "enumerate_identity",
    "recon_ipconfig": "enumerate_network",
}

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


# ── Specialist Engine ────────────────────────────────────────────────────────

class Specialist:
    """
    Tier 2 Specialist Engine.

    Evaluates commands against 11 MITRE ATT&CK tactic heads/families,
    infers attack stage and fine-grained action tags, and gathers indicators.
    """

    def __init__(
        self,
        specialist_mode: str = "family",
        family_path: Optional[str] = None,
        behavior_path: Optional[str] = None,
        mitre_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[RobertaTokenizer] = None,
        max_length: int = 256,
        view_policy: str = "mean",
        use_residual_format: bool = True,
        allow_behavior_fallback: bool = False,
        backbone_path: Optional[str] = None,
        local_files_only: bool = False,
    ):
        self.specialist_mode = specialist_mode.strip().lower()
        if self.specialist_mode not in {"family", "mitre"}:
            raise ValueError("Specialist mode must be family or mitre")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_path, resolved_local_only = resolve_backbone(backbone_path)
        self.local_files_only = local_files_only or resolved_local_only
        self.tokenizer = tokenizer or RobertaTokenizer.from_pretrained(
            self.backbone_path, **pretrained_kwargs(self.backbone_path, self.local_files_only)
        )
        self.max_length = max_length
        self.view_policy = view_policy
        self.use_residual_format = use_residual_format
        self.allow_behavior_fallback = allow_behavior_fallback

        self.family_labels = list(FAMILY_LABELS)
        self.family_bundle = None
        self.family_metadata = None
        self.family_path = None
        self.family_threshold = 0.5

        self.t2 = None
        self.s_map = {}
        self._tfidf_idx_to_label = {}
        self.mitre_metadata = None

        if self.specialist_mode == "family":
            req_path = family_path or os.getenv("GENOS_FAMILY_SPECIALIST_PATH", "models/family_specialist_tfidf.joblib")
            self.family_path = _resolve_asset_path(req_path)
            if os.path.exists(self.family_path):
                self.family_bundle = joblib.load(self.family_path)
                meta_path = os.path.splitext(self.family_path)[0] + ".json"
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as h:
                        self.family_metadata = json.load(h)
                    declared_labels = self.family_metadata.get("family_labels") or []
                    if len(declared_labels) == 11:
                        self.family_labels = list(declared_labels)
                self.family_threshold = float(self.family_bundle.get("decision_threshold", 0.5))
        else:
            mitre_model_path = _resolve_asset_path(mitre_path or os.getenv("GENOS_MITRE_MODEL_PATH", "models/specialist_tfidf_char_rf.pkl"))
            if os.path.exists(mitre_model_path):
                self.t2 = joblib.load(mitre_model_path)
                meta_path = os.path.splitext(mitre_model_path)[0] + ".json"
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as h:
                        self.mitre_metadata = json.load(h)
                map_candidates = ["config/specialist_map.json", "models/specialist_map.json"]
                for c in map_candidates:
                    res = _resolve_asset_path(c)
                    if os.path.exists(res):
                        with open(res, encoding="utf-8") as h:
                            raw_map = json.load(h)
                        self.s_map = {int(v): k for k, v in raw_map.items()} if any(type(k) is str and type(v) is int for k,v in raw_map.items()) else {int(k): v for k, v in raw_map.items()}
                        break
                self._tfidf_idx_to_label = dict(self.s_map)

        self.behavior_model_path = _resolve_behavior_model_path(behavior_path)
        self.behavior_model = None
        self.behavior_stage_labels = {}
        self.behavior_action_labels = {}
        self.behavior_action_threshold = float(os.getenv("GENOS_BEHAVIOR_ACTION_THRESHOLD", "0.5"))
        self.behavior_action_thresholds = {}
        self.behavior_load_error = None
        self.behavior_input_format = "structured"
        self._load_behavior_model()

    def _normalize_index_map(self, raw_map: dict) -> dict:
        if not isinstance(raw_map, dict) or any(type(v) is not int for v in raw_map.values()):
            raise ValueError("Model label maps must map names to integer indices")
        return {str(k): v for k, v in raw_map.items()}

    def _load_behavior_model(self) -> None:
        try:
            if not os.path.exists(self.behavior_model_path):
                raise FileNotFoundError(self.behavior_model_path)
            meta_candidate = os.path.splitext(self.behavior_model_path)[0] + ".json"
            with open(meta_candidate, encoding="utf-8") as handle:
                meta = json.load(handle)
            stage_map = self._normalize_index_map(meta.get("stage_map") or {})
            action_map = self._normalize_index_map(meta.get("action_map") or {})
            for name, mapping in (("stage", stage_map), ("action", action_map)):
                if not mapping or sorted(mapping.values()) != list(range(len(mapping))):
                    raise ValueError(f"Behavior {name} map must have unique contiguous indices")
            self.behavior_input_format = meta.get("input_format", "structured")
            if self.behavior_input_format not in {"raw", "structured"}:
                raise ValueError("Unsupported behavior input format")
            if int(meta.get("max_length", self.max_length)) != self.max_length:
                raise ValueError("Behavior training/runtime token limits differ")
            if meta.get("checkpoint_sha256") and meta["checkpoint_sha256"] != sha256_file(self.behavior_model_path):
                raise ValueError("Behavior checkpoint hash differs from metadata")
            model = BehaviorEncoderModel(
                len(stage_map), len(action_map), backbone_path=self.backbone_path,
                local_files_only=self.local_files_only,
            ).to(self.device)
            state_dict = torch.load(self.behavior_model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
        except Exception as exc:
            self.behavior_load_error = f"{type(exc).__name__}: {exc}"
            if not self.allow_behavior_fallback:
                raise RuntimeError("Behavior model failed to load; explicitly set GENOS_ALLOW_BEHAVIOR_FALLBACK=1 to allow the heuristic baseline") from exc
            warnings.warn(f"Using heuristic behavior baseline: {self.behavior_load_error}", RuntimeWarning)
            return
        self.behavior_model = model
        self.behavior_stage_labels = {index: label for label, index in stage_map.items()}
        self.behavior_action_labels = {index: label for label, index in action_map.items()}

    def predict_family_specialist(self, command: str) -> Dict[str, any]:
        """Predict multi-label membership across the 11 MITRE ATT&CK tactic families."""
        if not self.family_bundle:
            return {
                "model_type": "tfidf_family_specialist",
                "decision_threshold": self.family_threshold,
                "predicted_families": [],
                "all_family_scores": [],
            }

        normalized = (command or "").lower().strip()
        vector = self.family_bundle["vectorizer"].transform([normalized])
        probabilities = []
        for estimator in self.family_bundle["estimators"]:
            classes = np.asarray(estimator.classes_, dtype=int)
            positive_index = np.flatnonzero(classes == 1)
            if len(positive_index) != 1:
                raise RuntimeError("Family estimator has no positive class")
            probabilities.append(float(estimator.predict_proba(vector)[0, positive_index[0]]))

        score_rows = [
            {"family": family, "probability": round(probability * 100, 2), "selected": probability >= self.family_threshold}
            for family, probability in zip(self.family_labels, probabilities)
        ]
        selected = [row for row in score_rows if row["selected"]]
        selected.sort(key=lambda row: (-row["probability"], row["family"]))

        return {
            "model_type": "tfidf_family_specialist",
            "score_type": "cross_validated_calibrated_model_estimate",
            "decision_threshold": self.family_threshold,
            "predicted_families": selected,
            "all_family_scores": score_rows,
        }

    def _build_behavior_input(self, cmd: str, view_cache: dict | None = None):
        entry = view_cache.setdefault(cmd, {}) if view_cache is not None else {}
        parsed = entry.get("parsed")
        if parsed is None:
            parsed = _parse_command(cmd, deobfuscate_input=False)
            entry["parsed"] = parsed
        sem = entry.get("sem")
        if sem is None:
            sem = _build_semantic_features(parsed)
            entry["sem"] = sem
        rules = entry.get("rules")
        if rules is None:
            rules = _build_rule_result(parsed, sem)
            entry["rules"] = rules
        residual = _build_residual(parsed, sem, rules)
        feature_tags = _build_feature_tags(sem, rules)
        parts = [f"RAW: {cmd}", f"RESIDUAL: {residual}"]
        if feature_tags:
            parts.append(f"FEATURES: {' '.join(feature_tags)}")
        return "\n".join(parts), rules

    def _extract_behavior_action_tags(self, sem: dict, rules: dict, features: dict) -> list[str]:
        tags = {
            mapped for key, mapped in _SEMANTIC_TO_ACTION_TAG.items() if sem.get(key)
        }

        for raw_rule in rules.get("fired_rules") or []:
            normalized = raw_rule.replace("_rule_", "")
            mapped = _RULE_TAG_TO_ACTION_TAG.get(normalized)
            if mapped:
                tags.add(mapped)

        feature_tag_map = {
            "has_pipe_to_shell": "pipe_to_shell",
            "has_reverse_shell_pattern": "reverse_shell",
            "has_download": "download_remote_resource",
            "has_remote_transfer": "transfer_file_remote",
            "has_sensitive_file_read": "access_sensitive_file",
            "has_persistence_change": "establish_persistence",
            "has_privilege_escalation": "attempt_privilege_escalation",
            "has_defense_impairment": "disable_defenses",
            "has_destructive_write": "destructive_write",
            "has_archive_or_bulk_copy": "archive_or_stage_data",
            "has_tunneling": "establish_tunnel",
            "has_exfil_data_movement": "exfiltrate_data",
            "has_enumeration_recon": "enumerate_environment",
            "has_exploit_or_attack_tooling": "use_attack_tooling",
        }
        for feature_name, mapped in feature_tag_map.items():
            if features.get(feature_name):
                tags.add(mapped)

        return sorted(tags)

    def _infer_behavior_stage(self, label: str, sem: dict, rules: dict, features: dict) -> str:
        if label == "Benign":
            return "Benign Administration"
        if label in {"Context_Dependent", "Suspicious"}:
            pass

        priority_checks = [
            (features.get("has_destructive_write") or sem.get("deletes_shadow_copies"), "Impact"),
            (features.get("has_credential_dumping_pattern") or sem.get("reads_credential_store"), "Credential Access"),
            (features.get("has_privilege_escalation"), "Privilege Escalation"),
            (features.get("has_persistence_change") or sem.get("creates_scheduled_task") or sem.get("modifies_registry_autorun"), "Persistence"),
            (features.get("has_exfil_data_movement"), "Exfiltration"),
            (features.get("has_remote_transfer") or sem.get("downloads_remote_resource"), "Payload Retrieval"),
            (features.get("has_tunneling") or features.get("has_reverse_shell_pattern") or sem.get("remote_execution_or_session"), "C2 / Remote Access"),
            (features.get("has_enumeration_recon") or sem.get("enumerates_identity") or sem.get("enumerates_network_config"), "Discovery / Recon"),
            (features.get("has_base64_or_encoded_exec") or sem.get("uses_encoded_payload") or sem.get("uses_obfuscation"), "Defense Evasion"),
            (sem.get("runs_interpreter") or sem.get("executes_inline_code") or features.get("has_shell_spawn"), "Execution"),
        ]
        for matched, stage in priority_checks:
            if matched:
                return stage

        if (rules or {}).get("rule_strength") in {"weak", "strong"}:
            return "Execution"
        return "Context Required"

    def predict_behavior(
        self,
        cmd: str,
        routed_label: str,
        features: dict,
        raw_cmd: str | None = None,
        view_cache: dict | None = None,
    ) -> Tuple[dict, dict]:
        """Predict attack stage and fine-grained action tags."""
        cache = view_cache if view_cache is not None else {}
        behavior_text, rule_result = self._build_behavior_input(cmd, cache)
        sem = cache[cmd]["sem"]

        commands = list(dict.fromkeys([raw_cmd or cmd, cmd]))
        policy = getattr(self, "view_policy", "mean")
        if policy == "raw":
            commands = commands[:1]
        elif policy == "decoded":
            commands = commands[-1:]
        texts = [view if getattr(self, "behavior_input_format", "structured") == "raw" else
                 (behavior_text if view == cmd else self._build_behavior_input(view, cache)[0])
                 for view in commands]
        learned_behavior = self._predict_behavior_with_model(texts)
        if learned_behavior is not None:
            learned_behavior["input_text"] = texts[0]
            learned_behavior["input_views"] = texts
            learned_behavior["view_policy"] = policy
            return learned_behavior, rule_result

        stage = self._infer_behavior_stage(routed_label, sem, rule_result, features)
        action_tags = self._extract_behavior_action_tags(sem, rule_result, features)

        return {
            "stage": stage,
            "stage_confidence": None,
            "action_tags": action_tags,
            "model_type": "heuristic_bootstrap",
            "input_text": behavior_text,
        }, rule_result

    def _predict_behavior_with_model(self, behavior_text: str | list[str]) -> dict | None:
        if self.behavior_model is None:
            return None

        encoded = self.tokenizer(
            behavior_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        device_type = "cuda" if "cuda" in self.device.type else "cpu"
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with torch.no_grad():
            with autocast(device_type=device_type, dtype=autocast_dtype):
                stage_logits, action_logits = self.behavior_model(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                )

        stage_probs = F.softmax(stage_logits.float(), dim=1).mean(dim=0)
        action_probs = torch.sigmoid(action_logits.float()).mean(dim=0)
        stage_index = int(torch.argmax(stage_probs).item())
        stage_label = self.behavior_stage_labels.get(stage_index)
        if stage_label is None:
            raise RuntimeError("Behavior prediction has no matching stage label")

        action_tags = []
        for action_index, probability in enumerate(action_probs.tolist()):
            action_label = self.behavior_action_labels.get(action_index)
            threshold = getattr(self, "behavior_action_thresholds", {}).get(action_label, self.behavior_action_threshold)
            if action_label and probability >= threshold:
                action_tags.append(action_label)

        return {
            "stage": stage_label,
            "stage_confidence": round(float(stage_probs[stage_index].item()) * 100, 2),
            "action_tags": sorted(action_tags),
            "model_type": "behavior_encoder",
            "score_type": "uncalibrated_model_estimate",
            "_stage_probabilities": stage_probs.tolist(),
            "_action_probabilities": action_probs.cpu().tolist(),
            "action_score_type": "uncalibrated_model_estimate",
            "action_threshold": self.behavior_action_threshold,
            "action_thresholds": getattr(self, "behavior_action_thresholds", {}),
            "action_threshold_source": "validation_fitted" if getattr(self, "behavior_action_thresholds", {}) else "default",
            "input_truncated": any(len(self.tokenizer.encode(text, truncation=False)) > self.max_length for text in ([behavior_text] if isinstance(behavior_text, str) else behavior_text)),
        }

    # ── Legacy MITRE Mode ─────────────────────────────────────────────────────

    def _mitre_distribution(self, raw_cmd: str, decoded_cmd: str | None = None):
        if self.specialist_mode != "mitre":
            raise RuntimeError("Technique scoring is disabled in family specialist mode")
        commands = list(dict.fromkeys(cmd for cmd in (raw_cmd, decoded_cmd) if cmd is not None))
        texts = [self._build_behavior_input(cmd)[0] if self.use_residual_format else cmd for cmd in commands]
        probabilities = self.t2.predict_proba(texts)
        policy = getattr(self, "view_policy", "mean")
        if policy == "raw":
            probabilities = probabilities[:1]
        elif policy == "decoded":
            probabilities = probabilities[-1:]
        pooled = [[sum(float(row[column]) for row in probabilities) / len(probabilities)
                   for column in range(len(self.t2.classes_))]]
        return pooled[0]

    def predict_mitre_codes(self, raw_cmd: str, decoded_cmd: str | None = None) -> list[dict]:
        """Rank normalized scores; candidate techniques."""
        pooled = self._mitre_distribution(raw_cmd, decoded_cmd)
        ranked = [(self._tfidf_idx_to_label[int(index)], float(pooled[column]))
                  for column, index in enumerate(self.t2.classes_)]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [{"code": code, "confidence": round(probability * 100, 2), "score_type": "uncalibrated_model_estimate"} for code, probability in ranked[:5]]

    # ── Indicators / Evidence (Delegated to evidence.py) ──────────────────────

    def build_evidence(self, parsed: dict, sem: dict, rule_result: dict,
                       was_obfuscated: bool = False,
                       deobfuscated_cmd: str | None = None) -> dict:
        return _build_evidence_fn(parsed, sem, rule_result, was_obfuscated=was_obfuscated,
                                 deobfuscated_cmd=deobfuscated_cmd)

    def generate_evidence_summary(self, exe: str, sem: dict, fired_rules: list) -> str:
        return _generate_evidence_summary_fn(exe, sem, fired_rules)

    def collect_indicator_evidence(self, raw_cmd: str, decoded_cmd: str, was_obfuscated: bool) -> dict:
        return _collect_indicator_evidence_fn(raw_cmd, decoded_cmd, was_obfuscated=was_obfuscated)
