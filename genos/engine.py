"""
engine.py — Main Router for Genos.

Coordinates the multi-stage security analysis pipeline:
1. Baseline Pre-filter & Evaluation (with repeat sighting write-back)
2. Deobfuscation (multi-layer unrolling via deobfuscator.py)
3. Tier 1 Gatekeeper Triage (Benign vs Suspicious via gatekeeper.py)
4. Tier 2 Specialist Analysis (11 MITRE ATT&CK chain families + behavior via specialist.py)
5. Evidence Synthesis & Baseline Write-Back
"""

import base64
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
import platform
import re
import warnings
from contextlib import nullcontext
from importlib.metadata import version as package_version
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from .scientific_validation import PREPROCESSING_VERSION, sha256_file, temperature_scale
from .huggingface import pretrained_kwargs, resolve_backbone
from .baseline import (
    BASELINE_VERSION,
    SIGNATURE_SCHEMA_VERSION,
    BaselineEvaluation,
    BaselineMode,
    BaselineStatus,
    BaselineStore,
    ExecutionContext,
    SignatureExtractor,
)
from .deobfuscator import (
    Deobfuscator,
    calculate_entropy,
    clean_concatenation,
    decode_bare_base64,
    decode_embedded_base64,
    decode_powershell_encoded_command,
    decode_shell_base64_pipe,
    deobfuscate,
    deobfuscate_char_constructions,
    deobfuscate_layer,
    deobfuscate_with_metadata,
    extract_powershell_payload,
    is_obfuscated,
    universal_decoder,
)
from .gatekeeper import (
    GATE_LABELS_BINARY,
    GATE_LABELS_3CLASS,
    SUSPICIOUS_ROUTING_FEATURES,
    Gatekeeper,
    Tier1_Gatekeeper,
    _MeanPool,
    extract_routing_features,
)
from .specialist import (
    FAMILY_LABELS,
    BehaviorEncoderModel,
    Specialist,
    Tier2_Specialist,
    _resolve_asset_path,
    _resolve_behavior_model_path,
)
from .evidence import (
    HIGH_SIGNAL_FLAGS,
    INTERPRETER_NAMES,
    SURFACE_SEM_FEATURES,
    EvidenceExtractor,
    build_evidence,
    collect_indicator_evidence,
    generate_evidence_summary,
)

try:
    import pyminusone
except ImportError:
    pyminusone = None


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys as _sys
_PARSER_DIR = os.path.join(BASE_DIR, "parser")
if _PARSER_DIR not in _sys.path:
    _sys.path.insert(0, _PARSER_DIR)

try:
    from parser import parse_command as _parse_command
    from semantic_features import build_semantic_features as _build_semantic_features
    from rule_engine import build_rule_result as _build_rule_result
    from build_residual_dataset import build_residual as _build_residual, build_feature_tags as _build_feature_tags
    _RESIDUAL_PIPELINE_AVAILABLE = True
except ImportError:
    _RESIDUAL_PIPELINE_AVAILABLE = False


class GenosEngine:
    """
    Genos Security Analysis Engine & Router.

    Delegates processing across three dedicated components:
    - deobfuscator: multi-layer deobfuscation and payload unrolling
    - gatekeeper: binary triage into Benign vs Suspicious
    - specialist: 11-family MITRE ATT&CK chain analysis & behavior modeling
    """

    _PUBLIC_LABEL_MAP = {
        "Benign": "Benign",
        "Suspicious": "Suspicious",
        "Malicious": "Suspicious",
        "Context_Dependent": "Context_Dependent",
    }
    _INTERNAL_LABEL_MAP = {value: key for key, value in _PUBLIC_LABEL_MAP.items()}

    _GATE_LABELS = ["Benign", "Malicious", "Context_Dependent"]

    @classmethod
    def _to_public_label(cls, label: str) -> str:
        return cls._PUBLIC_LABEL_MAP.get(label, label)

    def __init__(
        self,
        t1_path="models/gatekeeper.pt",
        t2_path="models/behavior_encoder.pt",
        map_path=None,
        raw_mitre_path="data/training/mitre_atlas_raw.csv",
        gatekeeper_meta_path=None,
        use_residual_format=True,
        prior_alphas=None,
        view_policy=None,
        allow_behavior_fallback=None,
        gatekeeper_backend=None,
        specialist_mode=None,
        family_specialist_path=None,
        two_class_gatekeeper=True,
    ):
        self.view_policy = view_policy or os.getenv("GENOS_VIEW_POLICY", "mean")
        if self.view_policy not in {"raw", "decoded", "mean"}:
            raise ValueError("GENOS_VIEW_POLICY must be raw, decoded, or mean")

        self.allow_behavior_fallback = (
            _env_flag("GENOS_ALLOW_BEHAVIOR_FALLBACK", False)
            if allow_behavior_fallback is None
            else allow_behavior_fallback
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codebert_path, self.codebert_local_only = resolve_backbone()
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.codebert_path, **pretrained_kwargs(self.codebert_path, self.codebert_local_only)
        )
        self.max_length = int(os.getenv("GENOS_MAX_TOKENS", "256"))
        self.gatekeeper_backend = (gatekeeper_backend or os.getenv("GENOS_GATEKEEPER_BACKEND", "codebert")).strip().lower()
        if self.gatekeeper_backend not in {"codebert", "tfidf"}:
            raise ValueError("GENOS_GATEKEEPER_BACKEND must be codebert or tfidf")
        self.specialist_mode = (specialist_mode or os.getenv("GENOS_SPECIALIST_MODE", "family")).strip().lower()
        if self.specialist_mode not in {"mitre", "family"}:
            raise ValueError("GENOS_SPECIALIST_MODE must be mitre or family")

        self.two_class_gatekeeper = two_class_gatekeeper
        self.max_deobfuscation_layers = 5

        # Initialize Sub-Components
        self.deobfuscator = Deobfuscator(
            max_layers=self.max_deobfuscation_layers,
            entropy_threshold=5.2,
            entropy_delta_stop=0.01,
        )

        self.gatekeeper = Gatekeeper(
            backend=self.gatekeeper_backend,
            model_path=t1_path,
            meta_path=gatekeeper_meta_path,
            device=self.device,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            view_policy=self.view_policy,
            two_class=self.two_class_gatekeeper,
            backbone_path=self.codebert_path,
            local_files_only=self.codebert_local_only,
        )

        self.specialist = Specialist(
            specialist_mode=self.specialist_mode,
            family_path=family_specialist_path,
            behavior_path=t2_path,
            device=self.device,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            view_policy=self.view_policy,
            use_residual_format=use_residual_format,
            allow_behavior_fallback=self.allow_behavior_fallback,
            backbone_path=self.codebert_path,
            local_files_only=self.codebert_local_only,
        )

        # Baseline engine
        self.baseline_mode = os.getenv("GENOS_BASELINE_MODE", "audit").strip().lower()
        try:
            self.baseline_mode = BaselineMode(self.baseline_mode)
        except ValueError as exc:
            raise ValueError("GENOS_BASELINE_MODE must be learn, audit, or enforce") from exc
        self.baseline_store = BaselineStore(mode=self.baseline_mode)

        # Legacy backward-compatible attributes
        self._gate_labels = list(self.gatekeeper._labels)
        self.gatekeeper_model = getattr(self.gatekeeper, "model", None)
        self.gatekeeper_model_path = self.gatekeeper.model_path
        self.gatekeeper_meta_path = self.gatekeeper.meta_path
        self.gatekeeper_meta = self.gatekeeper.meta
        self.t1 = getattr(self.gatekeeper, "t1", None)

        self.family_labels = list(self.specialist.family_labels)
        self.family_specialist_bundle = self.specialist.family_bundle
        self.family_specialist_metadata = self.specialist.family_metadata
        self.family_specialist_path = self.specialist.family_path
        self.family_specialist_threshold = self.specialist.family_threshold
        self.t2 = getattr(self.specialist, "t2", None)
        self.s_map = self.specialist.s_map
        self._tfidf_idx_to_label = self.specialist._tfidf_idx_to_label
        self.mitre_metadata = self.specialist.mitre_metadata
        self.behavior_model_path = self.specialist.behavior_model_path
        self.behavior_model = self.specialist.behavior_model
        self.behavior_stage_labels = self.specialist.behavior_stage_labels
        self.behavior_action_labels = self.specialist.behavior_action_labels
        self.behavior_action_threshold = self.specialist.behavior_action_threshold
        self.behavior_action_thresholds = self.specialist.behavior_action_thresholds
        self.behavior_load_error = self.specialist.behavior_load_error
        self.behavior_input_format = self.specialist.behavior_input_format

        self.use_residual_format = use_residual_format
        self.prior_alphas = prior_alphas or {"strong": 2.0, "weak": 1.5, "none": 0.0}

        # Provenance metadata
        self.provenance = {
            "preprocessing_version": PREPROCESSING_VERSION,
            "implementation_sha256": {
                name: sha256_file(os.path.join(BASE_DIR, name))
                for name in (
                    "genos/engine.py",
                    "genos/deobfuscator.py",
                    "genos/gatekeeper.py",
                    "genos/specialist.py",
                    "genos/evidence.py",
                    "genos/scientific_validation.py",
                    "parser/parser.py",
                    "parser/semantic_features.py",
                    "parser/rule_engine.py",
                    "parser/build_residual_dataset.py",
                )
                if os.path.exists(os.path.join(BASE_DIR, name))
            },
            "view_policy": self.view_policy,
            "max_length": self.max_length,
            "gatekeeper_backend": self.gatekeeper_backend,
            "gatekeeper_device": "cpu" if self.gatekeeper_backend == "tfidf" else self.device.type,
            "gatekeeper_sha256": sha256_file(self.gatekeeper_model_path) if self.gatekeeper_model_path else None,
            "gatekeeper_metadata_sha256": sha256_file(self.gatekeeper_meta_path) if self.gatekeeper_meta_path else None,
            "specialist_mode": self.specialist_mode,
            "family_specialist_sha256": sha256_file(self.family_specialist_path) if self.family_specialist_path else None,
            "family_specialist_metadata_sha256": sha256_file(os.path.splitext(self.family_specialist_path)[0] + ".json") if self.family_specialist_path else None,
            "family_labels": self.family_labels,
            "training_validity": "independent_annotation_not_established",
            "mitre_label_map": self.s_map,
            "mitre_input_format": ("structured" if self.use_residual_format else "raw") if self.specialist_mode == "mitre" else None,
            "behavior_sha256": sha256_file(self.behavior_model_path) if self.behavior_model is not None else None,
            "behavior_input_format": getattr(self, "behavior_input_format", "structured"),
            "behavior_metadata_sha256": sha256_file(os.path.splitext(self.behavior_model_path)[0] + ".json") if self.behavior_model is not None else None,
            "behavior_action_threshold": self.behavior_action_threshold,
            "baseline_mode": self.baseline_mode.value,
            "baseline_version": BASELINE_VERSION,
            "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
            "torch_version": torch.__version__,
            "package_versions": {name: package_version(name) for name in ("transformers", "scikit-learn", "numpy", "joblib")},
            "encoder_config_sha256": hashlib.sha256(self.t1.encoder.config.to_json_string().encode()).hexdigest() if self.gatekeeper_backend == "codebert" and self.t1 else None,
            "tokenizer_sha256": hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "deobfuscation_policy": {"entropy_threshold": 5.2, "entropy_delta_stop": 0.01, "max_layers": self.max_deobfuscation_layers},
            "device_type": self.device.type,
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "cuda_runtime": torch.version.cuda,
            "inference_autocast_dtype": "float16" if self.device.type == "cuda" else "bfloat16",
            "behavior_model_type": "behavior_encoder" if self.behavior_model is not None else "heuristic_bootstrap",
            "behavior_fallback_reason": self.behavior_load_error,
            "optional_deobfuscator": "pyminusone" if pyminusone is not None else None,
            "base_score_semantics": (
                "cross_validated_calibrated_family_specialist_and_uncalibrated_gatekeeper_behavior"
                if self.specialist_mode == "family" and self.gatekeeper_backend == "codebert"
                else "cross_validated_calibrated_family_specialist_gatekeeper_and_uncalibrated_behavior"
                if self.specialist_mode == "family"
                else "cross_validated_calibrated_gatekeeper_and_uncalibrated_specialist_behavior"
                if self.gatekeeper_backend == "tfidf"
                else "uncalibrated_model_estimates"
            ),
        }

        self.calibration = None
        calibration_path = os.getenv("GENOS_CALIBRATION_PATH")
        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path, encoding="utf-8") as handle:
                self.calibration = json.load(handle)

    def _calibrate(self, probabilities, component):
        if getattr(self, "calibration", None) and component in self.calibration.get("temperatures", {}):
            return temperature_scale(probabilities, self.calibration["temperatures"][component])
        return probabilities

    def _score_status(self, component):
        if getattr(self, "calibration", None) and component in self.calibration.get("temperatures", {}):
            return "validation_temperature_scaled"
        if component == "gatekeeper" and getattr(self, "gatekeeper_backend", "codebert") == "tfidf":
            return "cross_validated_calibrated_model_estimate"
        if component == "family_specialist" and getattr(self, "specialist_mode", "mitre") == "family":
            return "cross_validated_calibrated_model_estimate"
        if component == "mitre" and getattr(self, "specialist_mode", "mitre") == "family":
            return "disabled_family_mode"
        return "uncalibrated_model_estimate"

    # ── Deobfuscation Delegates ───────────────────────────────────────────────

    def calculate_entropy(self, text: str) -> float:
        return calculate_entropy(text)

    def is_obfuscated(self, text: str) -> bool:
        return is_obfuscated(text)

    def deobfuscate_layer(self, text: str) -> str:
        return deobfuscate_layer(text)

    @staticmethod
    def _decode_bare_base64(text: str) -> str:
        return decode_bare_base64(text)

    # ── Gatekeeper Delegates ──────────────────────────────────────────────────

    def _gate_probs(self, text: str) -> torch.Tensor:
        if hasattr(self, "gatekeeper"):
            return self.gatekeeper.predict_probs(text)
        normalized = (text or "").lower().strip()
        if getattr(self, "gatekeeper_backend", "codebert") == "tfidf":
            probabilities = self.gatekeeper_model.predict_proba([normalized])[0]
            class_probabilities = {int(index): float(value) for index, value in zip(self.gatekeeper_model.classes_, probabilities)}
            ordered = [class_probabilities[index] for index in range(len(self._gate_labels))]
            return torch.as_tensor([ordered], dtype=torch.float32)
        encoded = self.tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)
        outputs = self.t1(encoded["input_ids"], encoded["attention_mask"])
        return F.softmax(outputs["verdict_logits"].float(), dim=1)

    def _summarize_gate_probs(self, probs: torch.Tensor) -> dict:
        if hasattr(self, "gatekeeper"):
            return self.gatekeeper.summarize_probs(probs)
        probs = probs.squeeze(0)
        benign_prob = float(probs[0].item())
        malicious_prob = float(probs[1].item())
        ctx_prob = float(probs[2].item()) if probs.size(0) > 2 else 0.0
        top_vals, top_idxs = torch.topk(probs, k=min(2, probs.size(0)), largest=True, sorted=True)
        predicted_idx = int(top_idxs[0].item())
        second_idx = int(top_idxs[1].item()) if len(top_idxs) > 1 else predicted_idx
        label = self._gate_labels[predicted_idx]
        second_label = self._gate_labels[second_idx]
        label_conf = float(top_vals[0].item())
        second_conf = float(top_vals[1].item()) if len(top_vals) > 1 else 0.0
        return {
            "label": label,
            "public_label": self._PUBLIC_LABEL_MAP.get(label, label),
            "second_label": second_label,
            "second_public_label": self._PUBLIC_LABEL_MAP.get(second_label, second_label),
            "benign_prob": benign_prob,
            "malicious_prob": malicious_prob,
            "ctx_prob": ctx_prob,
            "label_conf": label_conf,
            "second_conf": second_conf,
            "decision_margin": max(0.0, label_conf - second_conf),
            "class_probabilities": {
                "Benign": benign_prob,
                "Context_Dependent": ctx_prob,
                "Malicious": malicious_prob,
            },
            "decision_mode": "model_probs",
        }

    def _select_gate_summary(self, primary_probs: torch.Tensor, raw_probs: torch.Tensor | None = None) -> dict:
        policy = getattr(self, "view_policy", "mean")
        if raw_probs is None:
            pooled, view = primary_probs, "raw"
        elif policy == "raw":
            pooled, view = raw_probs, "raw"
        elif policy == "decoded":
            pooled, view = primary_probs, "decoded"
        else:
            pooled, view = (primary_probs.float() + raw_probs.float()) / 2, "mean_raw_decoded"
        calibrated = self._calibrate(pooled.float().cpu().numpy(), "gatekeeper")
        summary = self._summarize_gate_probs(torch.as_tensor(calibrated))
        summary["model_view"] = view
        summary["view_policy"] = policy
        return summary

    def _extract_routing_features(self, raw_cmd: str, deobfuscated_cmd: str | None = None) -> dict:
        return extract_routing_features(raw_cmd, deobfuscated_cmd)

    def _route_gatekeeper(self, gate: dict, features: dict) -> dict:
        if hasattr(self, "gatekeeper"):
            return self.gatekeeper.route(gate, features)
        # Fallback for uninitialized instances (e.g. __new__ in test harnesses)
        class_probs = gate["class_probabilities"]
        top_label = gate["public_label"]
        final_label = top_label
        reason = f"Model top class {top_label}; view policy {getattr(self, 'view_policy', 'mean')}."
        policy = "model_top_class"
        return {
            "label": self._INTERNAL_LABEL_MAP.get(final_label, final_label),
            "label_confidence": class_probs.get(final_label, gate.get("label_conf", 0.5)),
            "model_confidence": class_probs.get(final_label, gate.get("label_conf", 0.5)),
            "confidence_driver": "model_aligned",
            "reason": reason,
            "routing_policy": policy,
            "triggered_features": sorted(k for k, v in features.items() if v),
            "should_run_specialist": True,
        }

    # ── Specialist Delegates ──────────────────────────────────────────────────

    def _predict_family_specialist(self, command: str) -> dict:
        return self.specialist.predict_family_specialist(command)

    def _build_behavior_input(self, cmd: str):
        if hasattr(self, "specialist"):
            return self.specialist._build_behavior_input(cmd)
        return Specialist._build_behavior_input(self, cmd)

    def _build_variant_a_text(self, cmd: str):
        return self._build_behavior_input(cmd)

    def _infer_behavior_stage(self, label: str, sem: dict, rules: dict, features: dict) -> str:
        if hasattr(self, "specialist"):
            return self.specialist._infer_behavior_stage(label, sem, rules, features)
        return Specialist._infer_behavior_stage(self, label, sem, rules, features)

    def _extract_behavior_action_tags(self, sem: dict, rules: dict, features: dict) -> list[str]:
        if hasattr(self, "specialist"):
            return self.specialist._extract_behavior_action_tags(sem, rules, features)
        return Specialist._extract_behavior_action_tags(self, sem, rules, features)

    def _predict_behavior(self, cmd: str, routed_label: str, features: dict, raw_cmd: str | None = None) -> tuple[dict, dict]:
        if "_predict_behavior_with_model" in self.__dict__ or not hasattr(self, "specialist"):
            return Specialist.predict_behavior(self, cmd, routed_label, features, raw_cmd=raw_cmd)
        return self.specialist.predict_behavior(cmd, routed_label, features, raw_cmd=raw_cmd)

    def _predict_behavior_with_model(self, behavior_text: str | list[str]) -> dict | None:
        if hasattr(self, "specialist"):
            return self.specialist._predict_behavior_with_model(behavior_text)
        return Specialist._predict_behavior_with_model(self, behavior_text)

    def _mitre_distribution(self, raw_cmd: str, decoded_cmd: str | None = None):
        if hasattr(self, "specialist") and getattr(self, "specialist_mode", "mitre") == "mitre" and self.specialist.t2 is not None:
            pooled = self.specialist._mitre_distribution(raw_cmd, decoded_cmd)
        else:
            commands = list(dict.fromkeys(cmd for cmd in (raw_cmd, decoded_cmd) if cmd is not None))
            texts = [self._build_variant_a_text(cmd)[0] if getattr(self, "use_residual_format", False) else cmd for cmd in commands]
            probabilities = self.t2.predict_proba(texts)
            policy = getattr(self, "view_policy", "mean")
            if policy == "raw":
                probabilities = probabilities[:1]
            elif policy == "decoded":
                probabilities = probabilities[-1:]
            pooled = [sum(float(row[column]) for row in probabilities) / len(probabilities)
                      for column in range(len(self.t2.classes_))]
        pooled = self._calibrate([pooled], "mitre")
        return pooled[0]

    def _predict_mitre_codes(self, raw_cmd: str, decoded_cmd: str | None = None) -> list[dict]:
        pooled = self._mitre_distribution(raw_cmd, decoded_cmd)
        ranked = [(self._tfidf_idx_to_label[int(index)], float(pooled[column]))
                  for column, index in enumerate(self.t2.classes_)]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [{"code": code, "confidence": round(probability * 100, 2), "score_type": self._score_status("mitre")}
                for code, probability in ranked[:5]]

    def _collect_indicator_evidence(self, raw_cmd: str, decoded_cmd: str, was_obfuscated: bool) -> dict:
        return collect_indicator_evidence(raw_cmd, decoded_cmd, was_obfuscated=was_obfuscated)

    def _load_behavior_model(self) -> None:
        if not hasattr(self, "specialist"):
            Specialist._load_behavior_model(self)
            return
        self.specialist.behavior_model_path = getattr(self, "behavior_model_path", self.specialist.behavior_model_path)
        self.specialist.allow_behavior_fallback = getattr(self, "allow_behavior_fallback", self.specialist.allow_behavior_fallback)
        self.specialist._load_behavior_model()
        self.behavior_model = self.specialist.behavior_model
        self.behavior_stage_labels = self.specialist.behavior_stage_labels
        self.behavior_action_labels = self.specialist.behavior_action_labels
        self.behavior_load_error = self.specialist.behavior_load_error

    # ── Pipeline Execution ────────────────────────────────────────────────────

    def _create_default_context(self) -> ExecutionContext:
        """Construct fallback ExecutionContext for callers that do not supply one."""
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        return ExecutionContext(
            host_id=platform.node() or "default-host",
            parent_process=os.getenv("GENOS_DEFAULT_PARENT_PROCESS", "shell"),
            user=user,
            timestamp=datetime.now(timezone.utc),
            role=os.getenv("GENOS_DEFAULT_ROLE", "user"),
            fleet=os.getenv("GENOS_DEFAULT_FLEET", "default-fleet"),
        )

    def scan(
        self, raw_cmd: str, include_evaluation: bool = False, context: Any = None,
        baseline_store: Any = None, *, run_specialist: bool = True,
        collect_iocs: bool = True, use_baseline: bool = True,
    ) -> dict:
        """
        Execute full Genos inspection pipeline:
        1. Baseline pre-filter & evaluation (with write-back on hit)
        2. Multi-layer deobfuscation
        3. Gatekeeper triage (Benign vs Suspicious, with routing feature overrides)
        4. Specialist execution for Suspicious commands (11 families & behavior)
        5. Evidence gathering & baseline store write-back

        run_specialist and collect_iocs independently control optional work.
        use_baseline=False always analyzes the command without baseline shortcuts
        or sighting writes, as required by the stateless GET API.
        """
        baseline_eval = None
        baseline_signature = None
        baseline_ctx = None

        if context is not None:
            if isinstance(context, ExecutionContext):
                baseline_ctx = context
            elif isinstance(context, dict):
                baseline_ctx = ExecutionContext.from_dict(context)
        elif use_baseline and _env_flag("GENOS_ENABLE_DEFAULT_CONTEXT", True):
            baseline_ctx = self._create_default_context()

        store = (baseline_store if baseline_store is not None else getattr(self, "baseline_store", None)) if use_baseline else None
        if store is not None and baseline_ctx is not None:
            try:
                parsed = _parse_command(raw_cmd)
                baseline_signature = SignatureExtractor.extract(parsed)
                baseline_eval = store.evaluate(baseline_signature, baseline_ctx)
            except Exception:
                baseline_eval = BaselineEvaluation(
                    status=BaselineStatus.UNSCOPED,
                    scope="none",
                    seen_count=0,
                    first_seen=None,
                    novelty_score=0.0,
                    lineage_novel=False,
                    surprise_exe=0.0,
                    surprise_full=0.0,
                    should_run_genos=True,
                    rejection_reason="parse_error",
                )
        else:
            baseline_eval = BaselineEvaluation(
                status=BaselineStatus.UNSCOPED,
                scope="none",
                seen_count=0,
                first_seen=None,
                novelty_score=0.0,
                lineage_novel=False,
                surprise_exe=0.0,
                surprise_full=0.0,
                should_run_genos=True,
                rejection_reason="missing_context" if baseline_ctx is None else "missing_store",
            )

        # Baseline Short-Circuit with Repeat-Sighting Write-Back
        if baseline_eval.status == BaselineStatus.KNOWN_STABLE and not baseline_eval.should_run_genos:
            response = {
                "label": "Benign",
                "internal_label": "Benign",
                "public_label": "Benign",
                "label_confidence": 100.0,
                "model_confidence": 100.0,
                "confidence_driver": "baseline_known_stable",
                "class_probabilities": {"Benign": 100.0, "Suspicious": 0.0, "Context_Dependent": 0.0, "Malicious": 0.0},
                "label_probabilities": {"benign": 100.0, "suspicious": 0.0, "malicious": 0.0, "context_dependent": 0.0},
                "decision_margin": 100.0,
                "reason": "known_stable_baseline",
                "triggered_features": [],
                "routing_policy": "baseline",
                "should_run_specialist": False,
                "baseline_status": baseline_eval.status.value,
                "scope": baseline_eval.scope,
                "seen_count": baseline_eval.seen_count + 1,
                "first_seen": baseline_eval.first_seen.isoformat() if baseline_eval.first_seen else None,
                "lineage_novel": baseline_eval.lineage_novel,
                "novelty_score": baseline_eval.novelty_score,
                "action": "pass",
                "provenance": {
                    **self.provenance,
                    "baseline_version": BASELINE_VERSION,
                    "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
                },
            }
            # Record sighting for drift and frequency tracking
            if store is not None and baseline_ctx is not None and baseline_signature is not None:
                store.observe_and_admit(baseline_signature, baseline_ctx, response)
            return response

        # Obfuscation detection and deobfuscation happen before Stage 1.
        was_obfuscated = self.deobfuscator.is_obfuscated(raw_cmd)
        if was_obfuscated:
            current_cmd, _, _ = self.deobfuscator.deobfuscate_with_metadata(raw_cmd)
        else:
            current_cmd = raw_cmd.strip()
        processed_cmd = current_cmd.lower().strip()
        raw_processed = raw_cmd.strip().lower()

        device_type = "cuda" if "cuda" in self.device.type else "cpu"
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with torch.no_grad():
            gate_autocast = autocast(device_type=device_type, dtype=autocast_dtype) if self.gatekeeper_backend == "codebert" else nullcontext()
            with gate_autocast:
                g_probs = self._gate_probs(processed_cmd)
                raw_g_probs = self._gate_probs(raw_processed) if was_obfuscated and raw_processed != processed_cmd else None

                gate = self._select_gate_summary(g_probs, raw_g_probs)
                routing_features = self._extract_routing_features(
                    raw_cmd.strip(),
                    current_cmd if was_obfuscated else None,
                )
                routed = self._route_gatekeeper(gate, routing_features)
                # Stage 2 is optional and only applies to non-benign verdicts.
                run_stage2 = run_specialist and routed["label"] != "Benign" and (
                    routed["should_run_specialist"] or include_evaluation
                )

                # Class probabilities formatting
                benign_p = round(gate.get("benign_prob", 0.0) * 100, 2)
                suspicious_p = round(gate.get("suspicious_prob", 1.0 - gate.get("benign_prob", 0.0)) * 100, 2)
                malicious_p = round(gate.get("malicious_prob", 0.0) * 100, 2)
                ctx_p = round(gate.get("ctx_prob", 0.0) * 100, 2)

                raw_probabilities = {
                    "Benign": benign_p,
                    "Suspicious": suspicious_p,
                    "Context_Dependent": ctx_p,
                    "Malicious": malicious_p,
                }

                public_label = self._to_public_label(routed["label"])

                response = {
                    "label": public_label,
                    "internal_label": routed["label"],
                    "public_label": public_label,
                    "label_confidence": round(float(routed["label_confidence"]) * 100, 2),
                    "model_confidence": round(float(routed["model_confidence"]) * 100, 2),
                    "confidence_driver": routed["confidence_driver"],
                    "class_probabilities": raw_probabilities,
                    "label_probabilities": {
                        "benign": benign_p,
                        "suspicious": suspicious_p,
                        "malicious": malicious_p,
                        "context_dependent": ctx_p,
                    },
                    "decision_margin": round(float(gate["decision_margin"]) * 100, 2),
                    "reason": routed["reason"],
                    "triggered_features": routed["triggered_features"],
                    "routing_policy": routed["routing_policy"],
                    "should_run_specialist": run_stage2,
                    "gatekeeper": {
                        "decision_mode": routed["routing_policy"],
                        "label_names": list(self._gate_labels),
                        "model_top_internal_label": gate["label"],
                        "model_top_label": gate["public_label"],
                        "model_top_public_label": gate["public_label"],
                        "model_top_confidence": round(float(gate["label_conf"]) * 100, 2),
                        "model_second_internal_label": gate["second_label"],
                        "model_second_label": gate["second_public_label"],
                        "model_second_public_label": gate["second_public_label"],
                        "model_second_confidence": round(float(gate["second_conf"]) * 100, 2),
                        "model_view": gate.get("model_view"),
                        "metadata_path": self.gatekeeper_meta_path,
                        "view_policy": self.view_policy,
                        "score_type": self._score_status("gatekeeper"),
                    },
                    "evidence": {
                        "triggered_features": routed["triggered_features"],
                        "routing_reason": routed["reason"],
                        "routing_policy": routed["routing_policy"],
                    },
                    "deobfuscated_cmd": current_cmd if was_obfuscated else None,
                }

                if routed["label"] in {"Suspicious", "Context_Dependent"}:
                    response["action"] = "requires_context"
                elif routed["label"] == "Benign":
                    response["action"] = "pass"

                response["input_truncated"] = {
                    "raw": len(self.tokenizer.encode(raw_cmd.lower().strip(), truncation=False)) > self.max_length,
                    "decoded": len(self.tokenizer.encode(processed_cmd, truncation=False)) > self.max_length,
                }
                response["provenance"] = {
                    **self.provenance,
                    "baseline_version": BASELINE_VERSION,
                    "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
                }
                response["score_type"] = self._score_status("gatekeeper")
                response["baseline_status"] = baseline_eval.status.value if baseline_eval else BaselineStatus.UNSCOPED.value
                response["scope"] = baseline_eval.scope if baseline_eval else "none"
                response["seen_count"] = baseline_eval.seen_count if baseline_eval else 0
                response["first_seen"] = baseline_eval.first_seen.isoformat() if baseline_eval and baseline_eval.first_seen else None
                response["lineage_novel"] = baseline_eval.lineage_novel if baseline_eval else False
                response["novelty_score"] = baseline_eval.novelty_score if baseline_eval else 0.0

                if self.specialist_mode == "mitre":
                    response["mitre_scope"] = "closed_set_candidate_ranking"
                    response["calibration"] = {
                        "gatekeeper": self._score_status("gatekeeper"),
                        "mitre": self._score_status("mitre"),
                        "behavior": self._score_status("behavior") if self.behavior_model is not None else "heuristic_baseline",
                    }
                    if run_stage2:
                        response["MITRE_codes"] = self._predict_mitre_codes(
                            raw_cmd.strip(),
                            current_cmd if was_obfuscated and current_cmd != raw_cmd.strip() else None,
                        )
                else:
                    response["specialist_mode"] = "family"
                    response["mitre_scope"] = "disabled_family_specialist_mode"
                    response["calibration"] = {
                        "gatekeeper": self._score_status("gatekeeper"),
                        "family_specialist": self._score_status("family_specialist"),
                        "mitre": "disabled",
                        "behavior": self._score_status("behavior") if self.behavior_model is not None else "heuristic_baseline",
                    }

                # Optional Stage 2 specialist analysis.
                behavior_probabilities = None
                action_probabilities = None

                if run_stage2:
                    specialist_cmd = current_cmd if was_obfuscated and current_cmd != raw_cmd.strip() else raw_cmd.strip()
                    if self.specialist_mode == "family":
                        response["attack_families"] = self._predict_family_specialist(specialist_cmd)

                    behavior, _ = self._predict_behavior(
                        specialist_cmd,
                        routed["label"],
                        routing_features,
                        raw_cmd=raw_cmd.strip(),
                    )
                    behavior_probabilities = behavior.pop("_stage_probabilities", None)
                    action_probabilities = behavior.pop("_action_probabilities", None)
                    response["behavior"] = behavior
                    response["input_truncated"]["behavior"] = behavior.get("input_truncated", False)
                    response["attack_stage"] = behavior["stage"]

                    if was_obfuscated and specialist_cmd != raw_cmd.strip():
                        response["decoded_payload"] = specialist_cmd
                elif routed["label"] == "Benign":
                    # Benign commands bypass heavy specialist models
                    response["attack_stage"] = "Benign Administration"
                    response["behavior"] = {
                        "stage": "Benign Administration",
                        "stage_confidence": 100.0,
                        "action_tags": [],
                        "model_type": "benign_bypass",
                    }

                if collect_iocs and _RESIDUAL_PIPELINE_AVAILABLE:
                    evidence = self._collect_indicator_evidence(raw_cmd.strip(), current_cmd, was_obfuscated)
                    evidence.update({
                        "triggered_features": routed["triggered_features"],
                        "routing_reason": routed["reason"],
                        "routing_policy": routed["routing_policy"],
                    })
                    response["evidence"] = evidence
                    response["analyst_hint"] = evidence.get("evidence_summary")
                    response["ioc_summary"] = {
                        "urls": evidence["urls"],
                        "domains": evidence["domains"],
                        "ips": evidence["ips"],
                        "notable_files": evidence["file_paths"],
                        "registry_paths": evidence["registry_paths"],
                    }

                if include_evaluation:
                    response["_evaluation"] = {
                        "gatekeeper": [gate["class_probabilities"][name] for name in self._gate_labels if name in gate["class_probabilities"]],
                        "behavior": behavior_probabilities,
                        "behavior_actions": action_probabilities,
                    }
                    if self.specialist_mode == "mitre":
                        response["_evaluation"]["mitre"] = (
                            list(map(float, self._mitre_distribution(raw_cmd.strip(), current_cmd if was_obfuscated else None)))
                            if run_stage2 else None
                        )
                    else:
                        response["_evaluation"]["family_specialist"] = (
                            [
                                row["probability"] / 100
                                for row in response.get("attack_families", {}).get("all_family_scores", [])
                            ]
                            if run_stage2 else None
                        )

        # Write-back sighting to BaselineStore
        if store is not None and baseline_ctx is not None and baseline_signature is not None:
            store.observe_and_admit(baseline_signature, baseline_ctx, response)

        return response
