"""
gatekeeper.py — Tier 1 Gatekeeper triage engine for Genos.

Classifies incoming commands into Benign vs Suspicious:
- Evaluates CodeBERT or TF-IDF gatekeeper models
- Extracts ~24 semantic/structural regex routing features
- Enforces routing overrides when high-risk patterns are detected
- Directs execution: Benign commands skip specialists; Suspicious commands route to Tier 2
"""

import json
import os
import re
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from .scientific_validation import sha256_file
from .huggingface import pretrained_kwargs, resolve_backbone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Gatekeeper Labels ─────────────────────────────────────────────────────────

GATE_LABELS_BINARY = ["Benign", "Suspicious"]
GATE_LABELS_3CLASS = ["Benign", "Malicious", "Context_Dependent"]


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


# ── Model Architectures ───────────────────────────────────────────────────────

class Tier1_Gatekeeper(nn.Module):
    """Shared CodeBERT model with classification and auxiliary heads."""

    def __init__(self, num_classes=3, backbone_path="microsoft/codebert-base", local_files_only=False):
        super().__init__()
        config = RobertaConfig.from_pretrained(
            backbone_path, **pretrained_kwargs(backbone_path, local_files_only)
        )
        self.encoder = RobertaModel(config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
        self.non_benign_head = nn.Linear(768, 1)
        self.malicious_given_non_benign_head = nn.Linear(768, 1)
        self.ordinal_risk_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return {
            "verdict_logits": self.classifier(pooled),
            "non_benign_logit": self.non_benign_head(pooled).squeeze(-1),
            "malicious_given_non_benign_logit": self.malicious_given_non_benign_head(pooled).squeeze(-1),
            "ordinal_risk_logit": self.ordinal_risk_head(pooled).squeeze(-1),
        }


class _MeanPool(nn.Module):
    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


# ── Regex Detection Patterns ──────────────────────────────────────────────────

_DOWNLOAD_RE = re.compile(
    r"\b(?:curl|wget|invoke-webrequest|iwr|bitsadmin|certutil(?:\.exe)?|aria2c|fetch)\b",
    re.I,
)
_PIPE_TO_SHELL_RE = re.compile(
    r"(?:curl|wget|invoke-webrequest|iwr|echo|printf).{0,200}\|\s*(?:(?:/bin/)?(?:ba|z)?sh|pwsh?|powershell)\b",
    re.I | re.S,
)
_REVERSE_SHELL_RE = re.compile(
    r"(?:/dev/tcp/|\bnc(?:at)?\b.*(?:-e|-c)\s*/bin/(?:ba)?sh\b|mkfifo\b.*\bnc(?:at)?\b|"
    r"socket\.socket\(\).*connect\(|tcpsocket\.open\(|fsockopen\(|s_client\b.*\|\s*/bin/(?:ba)?sh\b)",
    re.I | re.S,
)
_BASE64_EXEC_RE = re.compile(
    r"(?:-enc(?:odedcommand)?\s+[A-Za-z0-9+/=]{20,}|frombase64string|base64\s+-d\b|certutil\s+-decode\b)",
    re.I,
)
_EVAL_EXEC_RE = re.compile(
    r"\b(?:eval|iex\b|invoke-expression\b|exec\s*\(|python\d?\s+-c\b|perl\s+-e\b|ruby\s+-e\b|php\s+-r\b|node\s+-e\b)",
    re.I,
)
_SHELL_SPAWN_RE = re.compile(
    r"(?:/bin/(?:ba)?sh\b|cmd(?:\.exe)?\s+/c\b|powershell(?:\.exe)?\b|pwsh\b)",
    re.I,
)
_SENSITIVE_FILE_READ_RE = re.compile(
    r"\b(?:cat|less|more|head|tail|grep|awk|sed|cut|strings|xxd|od|nl|wc|stat|file)\b.*"
    r"(?:/etc/(?:shadow|sudoers)|/root/\.ssh/|authorized_keys|id_rsa|\.kube/config|"
    r"/proc/\d+/environ)",
    re.I,
)
_NETWORK_ENUM_RE = re.compile(
    r"^\s*(?:nmap|masscan|zmap|netstat|ss|ifconfig|arp|route|traceroute|tracepath|mtr|dig|nslookup|host)\b"
    r"|^\s*ip\s+(?:addr|route|neigh|link|rule)\b",
    re.I,
)
_PROCESS_ENUM_RE = re.compile(
    r"^\s*(?:ps|top|pstree|pgrep|pidof|lsof)\b|^\s*docker\s+(?:ps|top|inspect)\b",
    re.I,
)
_TUNNELING_RE = re.compile(
    r"\bssh\b.*\s-[DLR]\s|\bchisel\b|\bsocat\b|\bsshuttle\b|\bfrp[cps]\b|\bnc(?:at)?\b.*\s-l\b",
    re.I,
)
_PACKET_CAPTURE_RE = re.compile(r"^\s*(?:tcpdump|tshark|dumpcap|wireshark)\b", re.I)
_DEBUG_TRACE_RE = re.compile(r"^\s*(?:strace|ltrace|gdb|perf\s+trace)\b", re.I)

_ENUMERATION_RECON_RE = re.compile(
    r"(?:"
    r"^\s*(?:ss|netstat)\s+-[a-zA-Z]*(?:a|p)[a-zA-Z]*\b"
    r"|"
    r"^\s*(?:last(?:\s+-\d+)?|w|who)\s*$"
    r"|"
    r"^\s*(?:ls|cat|find|stat)\b.*/etc/cron"
    r"|"
    r"^\s*ps\s+aux(?:ww)?\s*$"
    r"|"
    r"^\s*ls\b.*(?:/dev/shm|/root)\b"
    r"|"
    r"^\s*env\s*\|\s*grep\s+-i\s*(?:secret|token|key|password|cred)"
    r"|"
    r"^\s*(?:iptables\s+-L|aa-status|sestatus|apparmor_status)\b"
    r"|"
    r"^\s*sudo\s+-l\b"
    r"|"
    r"^\s*(?:systemd-detect-virt|dmidecode)\b"
    r"|"
    r"\bdmesg\b.*\bgrep\b.*\b(?:virtual|vbox|vmware|hyperv|qemu|kvm)\b"
    r"|"
    r"^\s*ip\s+route\s*$"
    r"|"
    r"^\s*python3?\s+-m\s+http\.server\s+(?!127\.0\.0\.1)\d"
    r"|"
    r"^\s*cat\s+/proc/version\b"
    r"|"
    r"^\s*getfacl\b.*(?:/etc/shadow|/etc/sudoers)"
    r"|"
    r"^\s*lsmod\b.*\|\s*grep\b.*\b(?:vbox|vmware|hyperv)\b"
    r")",
    re.I | re.M,
)

_EXFIL_DATA_MOVEMENT_RE = re.compile(
    r"(?:"
    r"^\s*curl\b.*(?:-X\s+POST|--request\s+POST)\b.*-d\s+@"
    r"|"
    r"^\s*tar\b.*(?:/home/[^\s]+/\.ssh|/root/\.ssh|/etc/shadow)"
    r"|"
    r"^\s*(?:scp|rsync)\b.*(?:/etc/passwd|/etc/shadow).*@"
    r"|"
    r"^\s*(?:curl|wget)\b.*https?://(?:10\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.168\.)\S+"
    r")",
    re.I | re.M,
)

_OFFENSIVE_TOOLING_RE = re.compile(
    r"^\s*(?:nmap|nikto|hydra|sqlmap|masscan|zmap|enum4linux|crackmapexec|responder|"
    r"impacket-|msfconsole|mimikatz(?:\.exe)?|john\b|hashcat\b|linpeas(?:\.sh)?|pspy\d*|bloodhound-python)\b",
    re.I,
)

_PERSISTENCE_RE = re.compile(
    r"(?:crontab\s+-(?!l\b)|echo\s+.*\|\s*crontab\b|schtasks\s+/create\b|currentversion\\run\b|"
    r"authorized_keys\b.*>>|systemctl\s+enable\b|rc\.local|"
    r"(?:cp|mv|install|tee)\b.*\b/etc/cron\.(?:d|daily|hourly|monthly|weekly)\b|"
    r"(?:echo|printf)\b.*>\s*/etc/cron\.(?:d|daily|hourly|monthly|weekly)\b|"
    r"at\s+\d)",
    re.I,
)

_PRIVESC_RE = re.compile(
    r"(?:chmod\s+(?:u\+s|4[0-7]{3})\s+/(?:bin|usr/bin|sbin)|/etc/sudoers\b.*>|useradd\s+.*-u\s+0\b|setcap\s+cap_setuid"
    r"|usermod\b.*-aG\s+(?:sudo|wheel|root|admin|docker)\b"
    r"|passwd\s+-d\s+root\b"
    r"|PermitRootLogin\s+yes.*>>\s*/etc/ssh)",
    re.I,
)

_DEFENSE_IMPAIR_RE = re.compile(
    r"(?:iptables\s+-F\b|ufw\s+disable\b|setenforce\s+0\b|systemctl\s+(?:stop|disable)\s+"
    r"(?:firewalld|ufw|auditd|sysmon)|auditctl\b.*-e\s+0|sc\s+stop\s+windefend|powershell.*set-mppreference"
    r"|truncate\s+-s\s+0\s+/var/log"
    r"|shred\b.*(?:/var/log|/etc/|auth\.log|syslog|kern\.log))",
    re.I,
)

_DESTRUCTIVE_RE = re.compile(
    r"(?:\bdd\b.*(?:if=/dev/(?:zero|urandom)).*(?:of=/dev/(?:sd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|vd[a-z]\d*))"
    r"|\bmkfs(?:\.[a-z0-9_+-]+)?\b\s+/dev/"
    r"|\brm\s+-rf\s+/+(?:\s|$)"
    r"|shred\b.*\s+/dev/"
    r"|\bwipefs\s+-a\b"
    r"|\buserdel\s+-r\b"
    r"|\bpkill\s+-9\s+(?:sshd|init|systemd)\b"
    r"|\bkill\s+-9\s+1\b)",
    re.I,
)

_ARCHIVE_BULK_RE = re.compile(
    r"^\s*(?:tar|zip|7z|rar|rsync)\b.*(?:/etc|/var/log|/home|/opt|/srv)"
    r"|^\s*cp\b.*(?:/etc(?:/|\s)|/var/log).*(?:/etc|/var/log|/home|/opt|/srv)"
    r"|^\s*find\b(?!.*-maxdepth\s+[12]\b).*(?:/etc|/var/log|/home).*(?:-name|-type)",
    re.I,
)

_REMOTE_TRANSFER_RE = re.compile(
    r"^\s*(?:scp|sftp|ftp|rsync)\b.*[@:][^ ]+|^\s*(?:curl|wget)\b.*(?:--upload-file|-T|--data-binary\s+@|--form\s+@|-d\s+@)",
    re.I,
)

_EXPLOIT_OR_ATTACK_TOOLING_RE = re.compile(
    r"^\s*(?:hydra|sqlmap|nikto|msfconsole|mimikatz(?:\.exe)?|john\b|hashcat\b|responder\b|ettercap\b|arpspoof\b|"
    r"crackmapexec\b|impacket-|metasploit\b|secretsdump\b|mshta\b)",
    re.I,
)

_CREDENTIAL_DUMP_RE = re.compile(
    r"(?:/etc/shadow\b|mimikatz|sekurlsa|hashdump|lsass|sam hive|unshadow\b|john\b.*rockyou|secretsdump)",
    re.I,
)

# High-risk features that mandate suspicious classification
SUSPICIOUS_ROUTING_FEATURES = frozenset({
    "has_pipe_to_shell",
    "has_reverse_shell_pattern",
    "has_base64_or_encoded_exec",
    "has_eval_exec",
    "has_shell_spawn",
    "has_sensitive_file_read",
    "has_tunneling",
    "has_offensive_tooling",
    "has_persistence_change",
    "has_privilege_escalation",
    "has_defense_impairment",
    "has_destructive_write",
    "has_credential_dumping_pattern",
    "has_exploit_or_attack_tooling",
    "has_exfil_data_movement",
    "has_remote_transfer",
})


def _matches_any(text_views: list[str], pattern: re.Pattern) -> bool:
    return any(pattern.search(view) for view in text_views if view)


def extract_routing_features(raw_cmd: str, deobfuscated_cmd: str | None = None) -> Dict[str, bool]:
    """Extract structural and behavioral regex features from command views."""
    text_views = []
    for candidate in (raw_cmd, deobfuscated_cmd):
        if candidate:
            normalized = candidate.lower().strip()
            if normalized and normalized not in text_views:
                text_views.append(normalized)

    return {
        "has_download": _matches_any(text_views, _DOWNLOAD_RE),
        "has_pipe_to_shell": _matches_any(text_views, _PIPE_TO_SHELL_RE),
        "has_reverse_shell_pattern": _matches_any(text_views, _REVERSE_SHELL_RE),
        "has_base64_or_encoded_exec": _matches_any(text_views, _BASE64_EXEC_RE),
        "has_eval_exec": _matches_any(text_views, _EVAL_EXEC_RE),
        "has_shell_spawn": _matches_any(text_views, _SHELL_SPAWN_RE),
        "has_sensitive_file_read": _matches_any(text_views, _SENSITIVE_FILE_READ_RE),
        "has_network_enum": _matches_any(text_views, _NETWORK_ENUM_RE),
        "has_process_enum": _matches_any(text_views, _PROCESS_ENUM_RE),
        "has_tunneling": _matches_any(text_views, _TUNNELING_RE),
        "has_packet_capture": _matches_any(text_views, _PACKET_CAPTURE_RE),
        "has_debug_trace": _matches_any(text_views, _DEBUG_TRACE_RE),
        "has_offensive_tooling": _matches_any(text_views, _OFFENSIVE_TOOLING_RE),
        "has_persistence_change": _matches_any(text_views, _PERSISTENCE_RE),
        "has_privilege_escalation": _matches_any(text_views, _PRIVESC_RE),
        "has_defense_impairment": _matches_any(text_views, _DEFENSE_IMPAIR_RE),
        "has_destructive_write": _matches_any(text_views, _DESTRUCTIVE_RE),
        "has_archive_or_bulk_copy": _matches_any(text_views, _ARCHIVE_BULK_RE),
        "has_remote_transfer": _matches_any(text_views, _REMOTE_TRANSFER_RE),
        "has_credential_dumping_pattern": _matches_any(text_views, _CREDENTIAL_DUMP_RE),
        "has_exploit_or_attack_tooling": _matches_any(text_views, _EXPLOIT_OR_ATTACK_TOOLING_RE),
        "has_enumeration_recon": _matches_any(text_views, _ENUMERATION_RECON_RE),
        "has_exfil_data_movement": _matches_any(text_views, _EXFIL_DATA_MOVEMENT_RE),
    }


# ── Gatekeeper Triage Engine ──────────────────────────────────────────────────

class Gatekeeper:
    """
    Tier 1 Gatekeeper Engine.

    Evaluates commands to determine if they are Benign or Suspicious.
    When suspicious patterns or model probabilities warrant inspection, routes
    to the specialist tier. Otherwise marks as Benign without specialist overhead.
    """

    _PUBLIC_LABEL_MAP = {
        "Benign": "Benign",
        "Suspicious": "Suspicious",
        "Malicious": "Suspicious",
        "Context_Dependent": "Context_Dependent",
    }

    def __init__(
        self,
        backend: str = "codebert",
        model_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[RobertaTokenizer] = None,
        max_length: int = 256,
        view_policy: str = "mean",
        two_class: bool = True,
        backbone_path: Optional[str] = None,
        local_files_only: bool = False,
    ):
        self.backend = backend.strip().lower()
        if self.backend not in {"codebert", "tfidf"}:
            raise ValueError("Gatekeeper backend must be codebert or tfidf")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_path, resolved_local_only = resolve_backbone(backbone_path)
        self.local_files_only = local_files_only or resolved_local_only
        self.tokenizer = tokenizer or RobertaTokenizer.from_pretrained(
            self.backbone_path, **pretrained_kwargs(self.backbone_path, self.local_files_only)
        )
        self.max_length = max_length
        self.view_policy = view_policy
        self.two_class = two_class

        self.model = None
        self.model_path = None
        self.meta = {}
        self.meta_path = _resolve_asset_path(meta_path or "config/gatekeeper_meta.json")

        if os.path.exists(self.meta_path):
            with open(self.meta_path, encoding="utf-8") as h:
                self.meta = json.load(h)

        self._labels = self._load_gatekeeper_labels(self.meta) or list(GATE_LABELS_3CLASS)

        if self.backend == "codebert":
            t1_path = _resolve_asset_path(model_path or "models/gatekeeper.pt")
            self.model_path = t1_path
            self.t1 = Tier1_Gatekeeper(
                num_classes=len(self._labels), backbone_path=self.backbone_path,
                local_files_only=self.local_files_only,
            ).to(self.device)
            self.t1.load_state_dict(torch.load(t1_path, map_location=self.device, weights_only=True), strict=True)
            self.t1.eval()
            expected_hash = self.meta.get("checkpoint_sha256")
            if expected_hash and expected_hash != sha256_file(t1_path):
                raise ValueError("Gatekeeper checkpoint hash differs from metadata")
        else:
            tfidf_path = model_path or os.getenv("GENOS_GATEKEEPER_TFIDF_PATH")
            if not tfidf_path:
                raise ValueError("GENOS_GATEKEEPER_TFIDF_PATH is required for tfidf backend")
            self.model_path = _resolve_asset_path(tfidf_path)
            self.model = joblib.load(self.model_path)
            meta_json = os.path.splitext(self.model_path)[0] + ".json"
            if os.path.exists(meta_json):
                with open(meta_json, encoding="utf-8") as h:
                    self.meta = json.load(h)

    def _load_gatekeeper_labels(self, meta: dict) -> list[str] | None:
        label_names = meta.get("label_names")
        if isinstance(label_names, list) and label_names:
            return [str(l) for l in label_names]
        return None

    def predict_probs(self, text: str) -> torch.Tensor:
        """Calculate gate probabilities for input text."""
        normalized = (text or "").lower().strip()
        if self.backend == "tfidf":
            probabilities = self.model.predict_proba([normalized])[0]
            class_probabilities = {int(idx): float(val) for idx, val in zip(self.model.classes_, probabilities)}
            ordered = [class_probabilities[idx] for idx in range(len(self._labels))]
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

    def summarize_probs(self, probs: torch.Tensor) -> Dict[str, any]:
        """Summarize output probabilities into class scores and predictions."""
        probs = probs.squeeze(0)
        benign_prob = float(probs[0].item())
        malicious_prob = float(probs[1].item()) if probs.size(0) > 1 else 0.0
        ctx_prob = float(probs[2].item()) if probs.size(0) > 2 else 0.0

        if self.two_class:
            suspicious_prob = malicious_prob + ctx_prob
            if len(self._labels) == 2:
                suspicious_prob = float(probs[1].item())
            is_benign = benign_prob >= 0.5
            label = "Benign" if is_benign else "Suspicious"
            label_conf = benign_prob if is_benign else suspicious_prob
            second_label = "Suspicious" if is_benign else "Benign"
            second_conf = suspicious_prob if is_benign else benign_prob
            return {
                "label": label,
                "public_label": label,
                "second_label": second_label,
                "second_public_label": second_label,
                "benign_prob": benign_prob,
                "suspicious_prob": suspicious_prob,
                "malicious_prob": malicious_prob,
                "ctx_prob": ctx_prob,
                "label_conf": label_conf,
                "second_conf": second_conf,
                "decision_margin": max(0.0, label_conf - second_conf),
                "class_probabilities": {
                    "Benign": benign_prob,
                    "Suspicious": suspicious_prob,
                },
                "decision_mode": "model_probs",
            }
        else:
            top_vals, top_idxs = torch.topk(probs, k=min(2, probs.size(0)), largest=True, sorted=True)
            predicted_idx = int(top_idxs[0].item())
            second_idx = int(top_idxs[1].item()) if len(top_idxs) > 1 else predicted_idx
            label = self._labels[predicted_idx]
            second_label = self._labels[second_idx]
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

    def select_gate_summary(
        self,
        primary_probs: torch.Tensor,
        raw_probs: Optional[torch.Tensor] = None,
        view_policy: Optional[str] = None,
    ) -> Dict[str, any]:
        """Aggregate primary and raw views based on configured view policy."""
        policy = view_policy or self.view_policy
        if raw_probs is None:
            pooled, view = primary_probs, "raw"
        elif policy == "raw":
            pooled, view = raw_probs, "raw"
        elif policy == "decoded":
            pooled, view = primary_probs, "decoded"
        else:
            pooled, view = (primary_probs.float() + raw_probs.float()) / 2, "mean_raw_decoded"

        summary = self.summarize_probs(pooled)
        summary["model_view"] = view
        summary["view_policy"] = policy
        return summary

    def route(self, gate: Dict[str, any], features: Dict[str, bool]) -> Dict[str, any]:
        """
        Route command based on model probabilities and high-signal regex features.

        - Routing flags directly affect the label: any triggered high-risk feature
          forces a 'Suspicious' verdict.
        - Benign commands with no high-risk features do NOT run specialists (should_run_specialist=False).
        - Suspicious commands route to Tier 2 specialist (should_run_specialist=True).
        """
        triggered_suspicious = sorted(
            f for f in SUSPICIOUS_ROUTING_FEATURES if features.get(f)
        )
        all_triggered = sorted(k for k, v in features.items() if v)

        if self.two_class:
            class_probs = gate["class_probabilities"]
            benign_prob = gate.get("benign_prob", 0.0)
            suspicious_prob = gate.get("suspicious_prob", 1.0 - benign_prob)

            if triggered_suspicious:
                # High-risk features override model to Suspicious
                final_label = "Suspicious"
                label_conf = max(suspicious_prob, 0.90)
                reason = f"Routing features flagged suspicious: {', '.join(triggered_suspicious)}"
                policy = "feature_override"
                confidence_driver = "feature_override"
                should_run_specialist = True
            elif gate["label"] == "Suspicious":
                final_label = "Suspicious"
                label_conf = suspicious_prob
                reason = f"Model top class Suspicious (p={suspicious_prob:.2f})."
                policy = "model_top_class"
                confidence_driver = "model_aligned"
                should_run_specialist = True
            else:
                final_label = "Benign"
                label_conf = benign_prob
                reason = f"Model top class Benign (p={benign_prob:.2f}); no suspicious features detected."
                policy = "model_top_class"
                confidence_driver = "model_aligned"
                should_run_specialist = False

            return {
                "label": final_label,
                "label_confidence": label_conf,
                "model_confidence": class_probs.get(final_label, label_conf),
                "confidence_driver": confidence_driver,
                "reason": reason,
                "routing_policy": policy,
                "triggered_features": all_triggered,
                "triggered_suspicious_features": triggered_suspicious,
                "should_run_specialist": should_run_specialist,
            }
        else:
            # Legacy 3-class routing mode
            class_probs = gate["class_probabilities"]
            top_label = gate["public_label"]
            final_label = top_label
            reason = f"Model top class {top_label}; view policy {self.view_policy}."
            policy = "model_top_class"
            should_run = True

            if triggered_suspicious and final_label == "Benign":
                final_label = "Context_Dependent"
                reason = f"Routing features flagged context required: {', '.join(triggered_suspicious)}"
                policy = "feature_override"

            return {
                "label": final_label,
                "label_confidence": class_probs.get(final_label, gate.get("label_conf", 0.5)),
                "model_confidence": class_probs.get(final_label, gate.get("label_conf", 0.5)),
                "confidence_driver": "model_aligned" if policy == "model_top_class" else "feature_override",
                "reason": reason,
                "routing_policy": policy,
                "triggered_features": all_triggered,
                "triggered_suspicious_features": triggered_suspicious,
                "should_run_specialist": should_run,
            }
