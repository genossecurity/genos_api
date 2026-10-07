import base64
import csv
import json
import math
import os
import re
import warnings
import hashlib
from contextlib import nullcontext
from importlib.metadata import version as package_version

from scientific_validation import PREPROCESSING_VERSION, sha256_file, temperature_scale
from baseline import (
    BASELINE_VERSION,
    SIGNATURE_SCHEMA_VERSION,
    BaselineEvaluation,
    BaselineMode,
    BaselineStatus,
    BaselineStore,
    ExecutionContext,
    SignatureExtractor,
)

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

try:
    import pyminusone
except ImportError:
    pyminusone = None


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


class Tier1_Gatekeeper(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = RobertaModel(RobertaConfig.from_pretrained("microsoft/codebert-base"))
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


class Tier2_Specialist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel(RobertaConfig.from_pretrained("microsoft/codebert-base"))
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
    def __init__(self, num_stages: int, num_actions: int):
        super().__init__()
        self.encoder = RobertaModel(RobertaConfig.from_pretrained("microsoft/codebert-base"))
        self.dropout = nn.Dropout(0.2)
        self.stage_head = nn.Linear(768, num_stages)
        self.action_head = nn.Linear(768, num_actions)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.stage_head(pooled), self.action_head(pooled)


class GenosEngine:
    _PUBLIC_LABEL_MAP = {
        "Benign": "Benign",
        "Malicious": "Malicious",
        "Context_Dependent": "Context_Dependent",
    }
    _INTERNAL_LABEL_MAP = {value: key for key, value in _PUBLIC_LABEL_MAP.items()}

    @classmethod
    def _to_public_label(cls, label: str) -> str:
        return cls._PUBLIC_LABEL_MAP.get(label, label)

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
        # Network socket enumeration with all-connections or process-display flags
        r"^\s*(?:ss|netstat)\s+-[a-zA-Z]*(?:a|p)[a-zA-Z]*\b"
        r"|"
        # Login/session enumeration
        r"^\s*(?:last(?:\s+-\d+)?|w|who)\s*$"
        r"|"
        # Cron/scheduled job inspection
        r"^\s*(?:ls|cat|find|stat)\b.*/etc/cron"
        r"|"
        # Process enumeration with wide output
        r"^\s*ps\s+aux(?:ww)?\s*$"
        r"|"
        # Sensitive directory listing (/dev/shm, /root)
        r"^\s*ls\b.*(?:/dev/shm|/root)\b"
        r"|"
        # Secret/token hunting in environment
        r"^\s*env\s*\|\s*grep\s+-i\s*(?:secret|token|key|password|cred)"
        r"|"
        # Firewall / security policy inspection
        r"^\s*(?:iptables\s+-L|aa-status|sestatus|apparmor_status)\b"
        r"|"
        # Privilege check
        r"^\s*sudo\s+-l\b"
        r"|"
        # VM detection / fingerprinting
        r"^\s*(?:systemd-detect-virt|dmidecode)\b"
        r"|"
        r"\bdmesg\b.*\bgrep\b.*\b(?:virtual|vbox|vmware|hyperv|qemu|kvm)\b"
        r"|"
        # Routing table enumeration (ip route without addr/link)
        r"^\s*ip\s+route\s*$"
        r"|"
        # HTTP server (can be used for exfil staging)
        r"^\s*python3?\s+-m\s+http\.server\s+(?!127\.0\.0\.1)\d"
        r"|"
        # /proc/version for kernel fingerprinting
        r"^\s*cat\s+/proc/version\b"
        r"|"
        # getfacl on sensitive files
        r"^\s*getfacl\b.*(?:/etc/shadow|/etc/sudoers)"
        r"|"
        # Kernel module listing for VM detection
        r"^\s*lsmod\b.*\|\s*grep\b.*\b(?:vbox|vmware|hyperv)\b"
        r")",
        re.I | re.M,
    )
    _EXFIL_DATA_MOVEMENT_RE = re.compile(
        r"(?:"
        # curl POST with file data to non-standard targets
        r"^\s*curl\b.*(?:-X\s+POST|--request\s+POST)\b.*-d\s+@"
        r"|"
        # tar/archive of sensitive user directories (.ssh, etc)
        r"^\s*tar\b.*(?:/home/[^\s]+/\.ssh|/root/\.ssh|/etc/shadow)"
        r"|"
        # scp/rsync of highly sensitive system files to remote
        r"^\s*(?:scp|rsync)\b.*(?:/etc/passwd|/etc/shadow).*@"
        r"|"
        # Download from internal/private IPs (lateral tool transfer)
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
    ):
        self.view_policy = view_policy or os.getenv("GENOS_VIEW_POLICY", "mean")
        if self.view_policy not in {"raw", "decoded", "mean"}:
            raise ValueError("GENOS_VIEW_POLICY must be raw, decoded, or mean")
        self.allow_behavior_fallback = (_env_flag("GENOS_ALLOW_BEHAVIOR_FALLBACK", False)
                                        if allow_behavior_fallback is None else allow_behavior_fallback)
        self.behavior_load_error = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_length = int(os.getenv("GENOS_MAX_TOKENS", "256"))
        self.gatekeeper_backend = (gatekeeper_backend or os.getenv("GENOS_GATEKEEPER_BACKEND", "codebert")).strip().lower()
        if self.gatekeeper_backend not in {"codebert", "tfidf"}:
            raise ValueError("GENOS_GATEKEEPER_BACKEND must be codebert or tfidf")
        self.specialist_mode = (specialist_mode or os.getenv("GENOS_SPECIALIST_MODE", "family")).strip().lower()
        if self.specialist_mode not in {"mitre", "family"}:
            raise ValueError("GENOS_SPECIALIST_MODE must be mitre or family")
        self.gatekeeper_model = None
        self.gatekeeper_model_path = None
        self.gatekeeper_model_metadata = {}
        self.gatekeeper_meta_path = None
        self.gatekeeper_meta = {}
        self._gate_labels = list(self._GATE_LABELS)

        t1_path = _resolve_asset_path(t1_path) if self.gatekeeper_backend == "codebert" else None
        t2_path = _resolve_asset_path(t2_path)

        if self.specialist_mode == "mitre":
            map_candidates = ["config/specialist_map.json", "models/specialist_map.json"]
            if map_path:
                map_candidates = [map_path]
                if not os.path.exists(_resolve_asset_path(map_path)):
                    raise FileNotFoundError(map_path)

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
                    ["data/art/mitre_atlas_raw.csv"],
                )
                self.s_map = self._build_map_from_csv(raw_csv_path)
        else:
            self.s_map = {}

        self.gatekeeper_meta_path = _resolve_asset_path(gatekeeper_meta_path or "config/gatekeeper_meta.json")
        with open(self.gatekeeper_meta_path, encoding="utf-8") as handle:
            self.gatekeeper_meta = json.load(handle)
        self._gate_labels = self._load_gatekeeper_labels(self.gatekeeper_meta)
        if self._gate_labels != self._GATE_LABELS:
            raise ValueError("Gatekeeper metadata must declare Benign/Malicious/Context_Dependent in training order")
        if int(self.gatekeeper_meta.get("max_len", self.max_length)) != self.max_length:
            raise ValueError("Gatekeeper training/runtime token limits differ")

        if self.gatekeeper_backend == "codebert":
            self.gatekeeper_model_path = t1_path
            self.t1 = Tier1_Gatekeeper().to(self.device)
            self.t1.load_state_dict(torch.load(t1_path, map_location=self.device, weights_only=True), strict=True)
            self.t1.eval()
            expected_hash = self.gatekeeper_meta.get("checkpoint_sha256")
            if expected_hash and expected_hash != sha256_file(t1_path):
                raise ValueError("Gatekeeper checkpoint hash differs from metadata")
        else:
            tfidf_path = os.getenv("GENOS_GATEKEEPER_TFIDF_PATH")
            if not tfidf_path:
                raise ValueError("GENOS_GATEKEEPER_TFIDF_PATH is required for the tfidf gatekeeper backend")
            self.gatekeeper_model_path = _resolve_asset_path(tfidf_path)
            self.gatekeeper_model = joblib.load(self.gatekeeper_model_path)
            metadata_path = os.path.splitext(self.gatekeeper_model_path)[0] + ".json"
            with open(metadata_path, encoding="utf-8") as handle:
                self.gatekeeper_model_metadata = json.load(handle)
            if self.gatekeeper_model_metadata.get("checkpoint_sha256") != sha256_file(self.gatekeeper_model_path):
                raise ValueError("TF-IDF gatekeeper checkpoint differs from metadata")
            if self.gatekeeper_model_metadata.get("class_names") != self._gate_labels:
                raise ValueError("TF-IDF gatekeeper class order differs from runtime")
            if sorted(int(index) for index in self.gatekeeper_model.classes_) != list(range(len(self._gate_labels))):
                raise ValueError("TF-IDF gatekeeper classes must be contiguous in training order")
            self.gatekeeper_meta_path = metadata_path
        # Behavior inference is required; heuristic fallback requires an explicit flag.
        self.behavior_model_path = _resolve_behavior_model_path(t2_path)
        self.behavior_model = None
        self.behavior_stage_labels = {}
        self.behavior_action_labels = {}
        self.behavior_action_threshold = float(os.getenv("GENOS_BEHAVIOR_ACTION_THRESHOLD", "0.5"))
        if not 0 <= self.behavior_action_threshold <= 1:
            raise ValueError("Behavior action threshold must be in [0, 1]")
        self._load_behavior_model()

        self.max_deobfuscation_layers = 5
        if not _RESIDUAL_PIPELINE_AVAILABLE:
            raise RuntimeError("Required parser/residual pipeline is unavailable")
        self.use_residual_format = use_residual_format
        self.baseline_mode = os.getenv("GENOS_BASELINE_MODE", "audit").strip().lower()
        try:
            self.baseline_mode = BaselineMode(self.baseline_mode)
        except ValueError as exc:
            raise ValueError("GENOS_BASELINE_MODE must be learn, audit, or enforce") from exc
        self.baseline_store = BaselineStore(mode=self.baseline_mode)
        self.prior_alphas = prior_alphas or {"strong": 2.0, "weak": 1.5, "none": 0.0}
        self._specialist_map_fwd = {mitre: idx for idx, mitre in self.s_map.items()}
        self.family_specialist_bundle = None
        self.family_specialist_metadata = None
        self.family_specialist_path = None
        self.family_labels = []
        self.mitre_metadata = None
        mitre_model_path = None
        if self.specialist_mode == "mitre":
            mitre_model_path = _resolve_asset_path(os.getenv(
                "GENOS_MITRE_MODEL_PATH", "models/specialist_tfidf_char_rf.pkl"
            ))
            self.t2 = joblib.load(mitre_model_path)
            mitre_meta_path = os.path.splitext(mitre_model_path)[0] + ".json"
            if os.path.exists(mitre_meta_path):
                with open(mitre_meta_path, encoding="utf-8") as handle:
                    self.mitre_metadata = json.load(handle)
                expected_format = "structured" if self.use_residual_format else "raw"
                if self.mitre_metadata.get("input_format") != expected_format:
                    raise ValueError("MITRE training/runtime input formats differ")
                if self.mitre_metadata.get("checkpoint_sha256") != sha256_file(mitre_model_path):
                    raise ValueError("MITRE checkpoint hash differs from metadata")
                if {str(k): int(v) for k, v in self.mitre_metadata.get("label_map", {}).items()} != {v: k for k, v in self.s_map.items()}:
                    raise ValueError("MITRE checkpoint label map differs from runtime")
            self._tfidf_idx_to_label = dict(self.s_map)
            unknown_classes = {int(index) for index in self.t2.classes_} - self.s_map.keys()
            if unknown_classes:
                raise ValueError(f"MITRE classifier has unmapped class indices: {sorted(unknown_classes)}")
        else:
            requested_path = family_specialist_path or os.getenv(
                "GENOS_FAMILY_SPECIALIST_PATH", "models/family_specialist_tfidf.joblib"
            )
            self.family_specialist_path = _resolve_asset_path(requested_path)
            self.family_specialist_bundle = joblib.load(self.family_specialist_path)
            family_meta_path = os.path.splitext(self.family_specialist_path)[0] + ".json"
            with open(family_meta_path, encoding="utf-8") as handle:
                self.family_specialist_metadata = json.load(handle)
            if self.family_specialist_metadata.get("checkpoint_sha256") != sha256_file(self.family_specialist_path):
                raise ValueError("Family specialist checkpoint differs from metadata")
            self.family_labels = list(self.family_specialist_metadata.get("family_labels") or [])
            if len(self.family_labels) != 11 or self.family_labels != self.family_specialist_bundle.get("family_labels"):
                raise ValueError("Family specialist metadata and model must declare the same 11 labels")
            if self.family_specialist_metadata.get("technique_ids_in_model_or_rows") is not False:
                raise ValueError("Family specialist artifact must not use technique IDs as model labels or row fields")
            self.family_specialist_threshold = float(self.family_specialist_bundle.get("decision_threshold", 0.5))
            if not 0 < self.family_specialist_threshold < 1:
                raise ValueError("Family specialist threshold must be in (0, 1)")
            self.t2 = None
            self._tfidf_idx_to_label = {}

        self.provenance = {
            "preprocessing_version": PREPROCESSING_VERSION,
            "implementation_sha256": {name: sha256_file(os.path.join(BASE_DIR, name)) for name in (
                "engine.py", "scientific_validation.py", "parser/parser.py", "parser/semantic_features.py",
                "parser/rule_engine.py", "parser/build_residual_dataset.py")},
            "view_policy": self.view_policy,
            "max_length": self.max_length,
            "gatekeeper_backend": self.gatekeeper_backend,
            "gatekeeper_device": "cpu" if self.gatekeeper_backend == "tfidf" else self.device.type,
            "gatekeeper_sha256": sha256_file(self.gatekeeper_model_path),
            "gatekeeper_metadata_sha256": sha256_file(self.gatekeeper_meta_path) if self.gatekeeper_meta_path else None,
            "specialist_mode": self.specialist_mode,
            "mitre_sha256": sha256_file(mitre_model_path) if mitre_model_path else None,
            "mitre_metadata_status": ("verified" if self.mitre_metadata else "legacy_missing_training_manifest") if self.specialist_mode == "mitre" else "disabled_family_mode",
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
            "encoder_config_sha256": hashlib.sha256(self.t1.encoder.config.to_json_string().encode()).hexdigest() if self.gatekeeper_backend == "codebert" else None,
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
        self.behavior_action_thresholds = {}
        policy_path = os.getenv("GENOS_BEHAVIOR_POLICY_PATH")
        if policy_path:
            with open(policy_path, encoding="utf-8") as handle:
                policy = json.load(handle)
            if policy.get("fit_split") != "validation" or policy.get("runtime_signature") != json.loads(json.dumps(self.provenance)):
                raise ValueError("Behavior policy does not match runtime or validation provenance")
            thresholds = policy.get("action_thresholds", {})
            if set(thresholds) != set(self.behavior_action_labels.values()):
                raise ValueError("Behavior policy action labels differ from checkpoint")
            if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in thresholds.values()):
                raise ValueError("Invalid behavior action threshold")
            self.behavior_action_thresholds = {str(k): float(v) for k,v in thresholds.items()}
            self.provenance["behavior_policy_sha256"] = sha256_file(policy_path)
        self.calibration = None
        calibration_path = os.getenv("GENOS_CALIBRATION_PATH")
        if calibration_path:
            with open(calibration_path, encoding="utf-8") as handle:
                self.calibration = json.load(handle)
            # Bind the artifact to the exact deployed models and preprocessing.
            if self.calibration.get("runtime_signature") != json.loads(json.dumps(self.provenance)):
                raise ValueError("Calibration artifact does not match runtime models/preprocessing")
            if self.calibration.get("fit_split") != "validation":
                raise ValueError("Calibration must be fitted on validation data")
            if not isinstance(self.calibration.get("temperatures"), dict) or not self.calibration["temperatures"]:
                raise ValueError("Calibration artifact must contain fitted temperatures")
            for component, value in self.calibration["temperatures"].items():
                if component not in {"gatekeeper", "mitre", "behavior"} or not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError("Invalid calibration temperature")

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

    def _gate_probs(self, text: str) -> torch.Tensor:
        normalized = (text or "").lower().strip()
        if self.gatekeeper_backend == "tfidf":
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

    def _load_gatekeeper_labels(self, meta: dict) -> list[str] | None:
        label_names = meta.get("label_names")
        if isinstance(label_names, list) and label_names:
            return [str(label) for label in label_names]

        id_to_label = meta.get("id_to_label")
        if isinstance(id_to_label, dict) and id_to_label:
            labels = []
            try:
                for idx in sorted(id_to_label.keys(), key=lambda value: int(value)):
                    labels.append(str(id_to_label[idx]))
            except (TypeError, ValueError):
                return None
            return labels if labels else None

        label_map = meta.get("label_map")
        if isinstance(label_map, dict) and label_map:
            labels = [None] * len(label_map)
            for label, idx in label_map.items():
                try:
                    labels[int(idx)] = str(label)
                except (TypeError, ValueError, IndexError):
                    return None
            return labels if all(label is not None for label in labels) else None

        return None

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
            model = BehaviorEncoderModel(len(stage_map), len(action_map)).to(self.device)
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

    def _load_map_from_json(self, json_path: str) -> dict:
        """Load specialist label map from JSON file as {int_index: mitre_id}."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        mapping = self._normalize_index_map(raw_map)
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("MITRE label map indices must be unique and contiguous")
        return {v: k for k, v in mapping.items()}

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
            exe, sem, fired_rules
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

    def _generate_evidence_summary(self, exe: str, sem: dict, fired_rules: list) -> str:
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


    _SEMANTIC_TO_ACTION_TAG = {
        "downloads_remote_resource": "download_remote_resource",
        "writes_executable_like_file": "write_executable_like_file",
        "modifies_registry_autorun": "modify_autorun",
        "creates_scheduled_task": "create_scheduled_task",
        "creates_or_modifies_service": "modify_service",
        "archive_create": "archive_data",
        "archive_extract": "extract_archive",
        "deletes_shadow_copies": "delete_shadow_copies",
        "remote_execution_or_session": "remote_execution",
        "transfers_file_to_remote": "transfer_file_remote",
        "runs_interpreter": "execute_interpreter",
        "executes_inline_code": "execute_inline_code",
        "enumerates_identity": "enumerate_identity",
        "enumerates_network_config": "enumerate_network_config",
        "reads_credential_store": "read_credential_store",
        "uses_encoded_payload": "use_encoded_payload",
        "uses_obfuscation": "use_obfuscation",
        "uses_signed_proxy_binary": "use_signed_proxy_binary",
    }

    _RULE_TAG_TO_ACTION_TAG = {
        "encoded_execution": "use_encoded_payload",
        "scripting_builtins": "execute_interpreter",
        "interpreter_general": "execute_interpreter",
        "signed_binary_proxy_execution": "use_signed_proxy_binary",
        "obfuscated_files_or_information": "use_obfuscation",
    }


    def _build_behavior_input(self, cmd: str):
        """Build the canonical behavior-model input text and return (text, rule_result)."""
        parsed = _parse_command(cmd)
        sem = _build_semantic_features(parsed)
        rules = _build_rule_result(parsed, sem)
        residual = _build_residual(parsed, sem, rules)
        feature_tags = _build_feature_tags(sem, rules)
        parts = [f"RAW: {cmd}", f"RESIDUAL: {residual}"]
        if feature_tags:
            parts.append(f"FEATURES: {' '.join(feature_tags)}")
        return "\n".join(parts), rules

    def _build_variant_a_text(self, cmd: str):
        """Backward-compatible alias for the previous specialist input builder."""
        return self._build_behavior_input(cmd)

    def _collect_indicator_evidence(self, raw_cmd: str, decoded_cmd: str, was_obfuscated: bool) -> dict:
        """Extract observable artifacts independently of model routing."""
        parsed = _parse_command(raw_cmd)
        sem = _build_semantic_features(parsed)
        if decoded_cmd != raw_cmd:
            decoded = _parse_command(decoded_cmd)
            decoded_sem = _build_semantic_features(decoded)
            for key, value in decoded_sem.items():
                if value and not sem.get(key):
                    sem[key] = value
            for key in ("file_paths", "registry_paths", "urls", "domains", "ips", "ports",
                        "lolbin_matches", "local_targets", "remote_targets"):
                parsed[key] = list(dict.fromkeys([*(parsed.get(key) or []), *(decoded.get(key) or [])]))
        rules = _build_rule_result(parsed, sem)
        return self._build_evidence(parsed, sem, rules, was_obfuscated=was_obfuscated,
                                    deobfuscated_cmd=decoded_cmd if was_obfuscated else None)

    def _mitre_distribution(self, raw_cmd: str, decoded_cmd: str | None = None):
        if self.specialist_mode != "mitre":
            raise RuntimeError("Technique scoring is disabled in family specialist mode")
        commands = list(dict.fromkeys(cmd for cmd in (raw_cmd, decoded_cmd) if cmd is not None))
        texts = [self._build_variant_a_text(cmd)[0] if self.use_residual_format else cmd for cmd in commands]
        probabilities = self.t2.predict_proba(texts)
        policy = getattr(self, "view_policy", "mean")
        if policy == "raw":
            probabilities = probabilities[:1]
        elif policy == "decoded":
            probabilities = probabilities[-1:]
        pooled = [[sum(float(row[column]) for row in probabilities) / len(probabilities)
                   for column in range(len(self.t2.classes_))]]
        pooled = self._calibrate(pooled, "mitre")
        return pooled[0]

    def _predict_mitre_codes(self, raw_cmd: str, decoded_cmd: str | None = None) -> list[dict]:
        """Rank normalized scores; these are candidate techniques, not proof of attack."""
        pooled = self._mitre_distribution(raw_cmd, decoded_cmd)
        ranked = [(self._tfidf_idx_to_label[int(index)], float(pooled[column]))
                  for column, index in enumerate(self.t2.classes_)]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [{"code": code, "confidence": round(probability * 100, 2), "score_type": self._score_status("mitre")} for code, probability in ranked[:5]]

    def _predict_family_specialist(self, command: str) -> dict:
        normalized = (command or "").lower().strip()
        vector = self.family_specialist_bundle["vectorizer"].transform([normalized])
        probabilities = []
        for estimator in self.family_specialist_bundle["estimators"]:
            classes = np.asarray(estimator.classes_, dtype=int)
            positive_index = np.flatnonzero(classes == 1)
            if len(positive_index) != 1:
                raise RuntimeError("Family estimator has no positive class")
            probabilities.append(float(estimator.predict_proba(vector)[0, positive_index[0]]))
        score_rows = [
            {"family": family, "probability": round(probability * 100, 2), "selected": probability >= self.family_specialist_threshold}
            for family, probability in zip(self.family_labels, probabilities)
        ]
        selected = [row for row in score_rows if row["selected"]]
        selected.sort(key=lambda row: (-row["probability"], row["family"]))
        return {
            "model_type": "tfidf_family_specialist",
            "score_type": self._score_status("family_specialist"),
            "decision_threshold": self.family_specialist_threshold,
            "predicted_families": selected,
            "all_family_scores": score_rows,
        }

    def _extract_behavior_action_tags(self, sem: dict, rules: dict, features: dict) -> list[str]:
        tags = {
            mapped for key, mapped in self._SEMANTIC_TO_ACTION_TAG.items() if sem.get(key)
        }

        for raw_rule in rules.get("fired_rules") or []:
            normalized = raw_rule.replace("_rule_", "")
            mapped = self._RULE_TAG_TO_ACTION_TAG.get(normalized)
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
        if label == "Context_Dependent":
            return "Context Required"

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

    def _predict_behavior(self, cmd: str, routed_label: str, features: dict, raw_cmd: str | None = None) -> tuple[dict, dict]:
        behavior_text, rule_result = self._build_behavior_input(cmd)
        parsed = _parse_command(cmd)
        sem = _build_semantic_features(parsed)

        commands = list(dict.fromkeys([raw_cmd or cmd, cmd]))
        policy = getattr(self, "view_policy", "mean")
        if policy == "raw":
            commands = commands[:1]
        elif policy == "decoded":
            commands = commands[-1:]
        texts = [view if getattr(self, "behavior_input_format", "structured") == "raw" else self._build_behavior_input(view)[0] for view in commands]
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
            padding="max_length",
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
        stage_probs = torch.as_tensor(self._calibrate(stage_probs.cpu().numpy()[None, :], "behavior")[0])
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
            "score_type": self._score_status("behavior"),
            "_stage_probabilities": stage_probs.tolist(),
            "_action_probabilities": action_probs.cpu().tolist(),
            "action_score_type": "uncalibrated_model_estimate",
            "action_threshold": self.behavior_action_threshold,
            "action_thresholds": getattr(self, "behavior_action_thresholds", {}),
            "action_threshold_source": "validation_fitted" if getattr(self, "behavior_action_thresholds", {}) else "default",
            "input_truncated": any(len(self.tokenizer.encode(text, truncation=False)) > self.max_length for text in ([behavior_text] if isinstance(behavior_text, str) else behavior_text)),
        }

    def calculate_entropy(self, text):
        if not text:
            return 0
        entropy = 0
        for x in range(256):
            p_x = float(text.count(chr(x))) / len(text)
            if p_x > 0:
                entropy += -p_x * math.log(p_x, 2)
        return entropy

    @staticmethod
    def _decode_bare_base64(text: str) -> str:
        if len(text) < 8 or len(text) % 4 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", text):
            return text
        try:
            raw = base64.b64decode(text, validate=True)
            for encoding in ("utf-16-le", "utf-8"):
                try:
                    decoded = raw.decode(encoding)
                except UnicodeDecodeError:
                    continue
                printable = sum(c in "\r\n\t" or " " <= c <= "~" for c in decoded)
                if len(decoded) > 3 and printable / len(decoded) > 0.9:
                    return decoded
        except ValueError:
            pass
        return text

    def is_obfuscated(self, text: str) -> bool:
        if self._decode_bare_base64(text) != text:
            return True
        patterns = [
            r"\[char\]",
            r"base64",
            r"frombase64",
            r"reverse\(",
            r"\+[ ]*'",
            r"\$[a-z0-9_]{10,}",
            r"\\x[0-9a-f]{2}",
            r"(?i)-enc(?:odedcommand)?\s+[A-Za-z0-9+/=]{20,}",
        ]
        if any(re.search(p, text, re.I) for p in patterns):
            return True
        if self.calculate_entropy(text) > 5.2:
            return True
        return False

    _ENCODED_CMD_RE = re.compile(
        r"(?i)-(?:enc(?:odedcommand)?)\s+([A-Za-z0-9+/=]{20,})"
    )

    _SHELL_B64_PIPE_RE = re.compile(
        r"""(?:echo|printf|echo\s+-[neE]+)\s+
            ['"]?
            ([A-Za-z0-9+/]{20,}={0,2})
            ['"]?
            \s*\|\s*base64\s+-d""",
        re.X | re.I,
    )

    def deobfuscate_layer(self, text: str) -> str:
        text = self._decode_bare_base64(text)
        text = self._decode_powershell_encoded_command(text)
        text = self._decode_shell_base64_pipe(text)
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

    def _decode_powershell_encoded_command(self, text: str) -> str:
        match = self._ENCODED_CMD_RE.search(text)
        if not match:
            return text
        blob = match.group(1)
        try:
            raw = base64.b64decode(blob)
            try:
                utf16 = raw.decode("utf-16-le")
                ascii_printable = sum(1 for c in utf16 if '\x20' <= c <= '\x7e' or c in '\r\n\t')
                if ascii_printable > len(utf16) * 0.6 and len(utf16) > 3:
                    return utf16
            except (UnicodeDecodeError, ValueError):
                pass
            decoded = raw.decode("utf-8", errors="ignore")
            if len(decoded) > 3:
                return decoded
        except Exception:
            pass
        return text

    def _decode_shell_base64_pipe(self, text: str) -> str:
        def _repl(m):
            blob = m.group(1)
            try:
                decoded = base64.b64decode(blob).decode("utf-8", errors="ignore")
                printable = sum(1 for c in decoded if c.isprintable() or c in '\r\n\t')
                if printable > len(decoded) * 0.7 and len(decoded) > 3:
                    return m.group(0).replace(blob, decoded)
            except Exception:
                pass
            return m.group(0)
        return self._SHELL_B64_PIPE_RE.sub(_repl, text)

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

    # Class indices: 0=Benign, 1=Malicious, 2=Context_Dependent
    _GATE_LABELS = ["Benign", "Malicious", "Context_Dependent"]

    def _summarize_gate_probs(self, probs: torch.Tensor) -> dict:
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
            "public_label": self._PUBLIC_LABEL_MAP[label],
            "second_label": second_label,
            "second_public_label": self._PUBLIC_LABEL_MAP[second_label],
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

    def _matches_any(self, text_views: list[str], pattern: re.Pattern) -> bool:
        return any(pattern.search(view) for view in text_views if view)

    def _extract_routing_features(self, raw_cmd: str, deobfuscated_cmd: str | None = None) -> dict:
        text_views = []
        for candidate in (raw_cmd, deobfuscated_cmd):
            if candidate:
                normalized = candidate.lower().strip()
                if normalized and normalized not in text_views:
                    text_views.append(normalized)

        features = {
            "has_download": self._matches_any(text_views, self._DOWNLOAD_RE),
            "has_pipe_to_shell": self._matches_any(text_views, self._PIPE_TO_SHELL_RE),
            "has_reverse_shell_pattern": self._matches_any(text_views, self._REVERSE_SHELL_RE),
            "has_base64_or_encoded_exec": self._matches_any(text_views, self._BASE64_EXEC_RE),
            "has_eval_exec": self._matches_any(text_views, self._EVAL_EXEC_RE),
            "has_shell_spawn": self._matches_any(text_views, self._SHELL_SPAWN_RE),
            "has_sensitive_file_read": self._matches_any(text_views, self._SENSITIVE_FILE_READ_RE),
            "has_network_enum": self._matches_any(text_views, self._NETWORK_ENUM_RE),
            "has_process_enum": self._matches_any(text_views, self._PROCESS_ENUM_RE),
            "has_tunneling": self._matches_any(text_views, self._TUNNELING_RE),
            "has_packet_capture": self._matches_any(text_views, self._PACKET_CAPTURE_RE),
            "has_debug_trace": self._matches_any(text_views, self._DEBUG_TRACE_RE),
            "has_offensive_tooling": self._matches_any(text_views, self._OFFENSIVE_TOOLING_RE),
            "has_persistence_change": self._matches_any(text_views, self._PERSISTENCE_RE),
            "has_privilege_escalation": self._matches_any(text_views, self._PRIVESC_RE),
            "has_defense_impairment": self._matches_any(text_views, self._DEFENSE_IMPAIR_RE),
            "has_destructive_write": self._matches_any(text_views, self._DESTRUCTIVE_RE),
            "has_archive_or_bulk_copy": self._matches_any(text_views, self._ARCHIVE_BULK_RE),
            "has_remote_transfer": self._matches_any(text_views, self._REMOTE_TRANSFER_RE),
            "has_credential_dumping_pattern": self._matches_any(text_views, self._CREDENTIAL_DUMP_RE),
            "has_exploit_or_attack_tooling": self._matches_any(text_views, self._EXPLOIT_OR_ATTACK_TOOLING_RE),
            "has_enumeration_recon": self._matches_any(text_views, self._ENUMERATION_RECON_RE),
            "has_exfil_data_movement": self._matches_any(text_views, self._EXFIL_DATA_MOVEMENT_RE),
        }
        return features


    def _triggered_features(self, features: dict) -> list[str]:
        return sorted(name for name, value in features.items() if value)

    def _build_route_result(
        self,
        label: str,
        label_confidence: float,
        reason: str,
        policy: str,
        features: dict,
        should_run_specialist: bool,
        class_probs: dict | None = None,
    ) -> dict:
        model_confidence = class_probs[label] if class_probs is not None else label_confidence
        return {
            "label": self._INTERNAL_LABEL_MAP[label],
            "label_confidence": label_confidence,
            "model_confidence": model_confidence,
            "confidence_driver": "model_aligned",
            "reason": reason,
            "routing_policy": policy,
            "triggered_features": self._triggered_features(features),
            "should_run_specialist": should_run_specialist,
        }

    def _route_gatekeeper(self, gate: dict, features: dict) -> dict:
        class_probs = gate["class_probabilities"]
        top_label = gate["public_label"]
        final_label = top_label
        reason = f"Model top class {top_label}; view policy {getattr(self, 'view_policy', 'mean')}."
        policy = "model_top_class"

        return self._build_route_result(
            label=final_label,
            label_confidence=class_probs[final_label],
            reason=reason,
            policy=policy,
            features=features,
            should_run_specialist=True,
            class_probs=class_probs,
        )

    def scan(self, raw_cmd, include_evaluation=False, context=None, baseline_store=None):
        baseline_eval = None
        baseline_signature = None
        baseline_ctx = None
        if context is not None:
            if isinstance(context, ExecutionContext):
                baseline_ctx = context
            elif isinstance(context, dict):
                baseline_ctx = ExecutionContext.from_dict(context)

        store = baseline_store if baseline_store is not None else getattr(self, "baseline_store", None)
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
                rejection_reason="missing_context",
            )

        if baseline_eval.status == BaselineStatus.KNOWN_STABLE and not baseline_eval.should_run_genos:
            return {
                "label": "Benign",
                "internal_label": "Benign",
                "public_label": "Benign",
                "label_confidence": 100.0,
                "model_confidence": 100.0,
                "confidence_driver": "baseline_known_stable",
                "class_probabilities": {"Benign": 100.0, "Context_Dependent": 0.0, "Malicious": 0.0},
                "label_probabilities": {"benign": 100.0, "malicious": 0.0, "context_dependent": 0.0},
                "decision_margin": 100.0,
                "reason": "known_stable_baseline",
                "triggered_features": [],
                "routing_policy": "baseline",
                "should_run_specialist": False,
                "baseline_status": baseline_eval.status.value,
                "scope": baseline_eval.scope,
                "seen_count": baseline_eval.seen_count,
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

                raw_probabilities = {
                    "Benign": round(gate["class_probabilities"]["Benign"] * 100, 2),
                    "Context_Dependent": round(gate["class_probabilities"]["Context_Dependent"] * 100, 2),
                    "Malicious": round(gate["class_probabilities"]["Malicious"] * 100, 2),
                }

                public_label = self._to_public_label(routed["label"])

                response = {
                    "label": public_label,
                    "internal_label": routed["label"],
                    "public_label": public_label,
                    "label_confidence": round(routed["label_confidence"] * 100, 2),
                    "model_confidence": round(routed["model_confidence"] * 100, 2),
                    "confidence_driver": routed["confidence_driver"],
                    "class_probabilities": raw_probabilities,
                    "label_probabilities": {
                        "benign": raw_probabilities["Benign"],
                        "malicious": raw_probabilities["Malicious"],
                        "context_dependent": raw_probabilities["Context_Dependent"],
                    },
                    "decision_margin": round(gate["decision_margin"] * 100, 2),
                    "reason": routed["reason"],
                    "triggered_features": routed["triggered_features"],
                    "routing_policy": routed["routing_policy"],
                    "should_run_specialist": routed["should_run_specialist"],
                    "gatekeeper": {
                        "decision_mode": routed["routing_policy"],
                        "label_names": list(self._gate_labels),
                        "model_top_internal_label": gate["label"],
                        "model_top_label": gate["public_label"],
                        "model_top_public_label": gate["public_label"],
                        "model_top_confidence": round(gate["label_conf"] * 100, 2),
                        "model_second_internal_label": gate["second_label"],
                        "model_second_label": gate["second_public_label"],
                        "model_second_public_label": gate["second_public_label"],
                        "model_second_confidence": round(gate["second_conf"] * 100, 2),
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

                if public_label == "Context_Dependent":
                    response["action"] = "requires_context"

                response["input_truncated"] = {
                    "raw": len(self.tokenizer.encode(raw_cmd.lower().strip(), truncation=False)) > self.max_length,
                    "decoded": len(self.tokenizer.encode(processed_cmd, truncation=False)) > self.max_length,
                }
                response["provenance"] = {**self.provenance, "baseline_version": BASELINE_VERSION, "signature_schema_version": SIGNATURE_SCHEMA_VERSION}
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
                    response["MITRE_codes"] = self._predict_mitre_codes(
                        raw_cmd.strip(), current_cmd if was_obfuscated and current_cmd != raw_cmd.strip() else None,
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
                if routed["should_run_specialist"]:
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

                if _RESIDUAL_PIPELINE_AVAILABLE:
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
                        "gatekeeper": [gate["class_probabilities"][name] for name in self._GATE_LABELS],
                        "behavior": behavior_probabilities,
                        "behavior_actions": action_probabilities,
                    }
                    if self.specialist_mode == "mitre":
                        response["_evaluation"]["mitre"] = list(map(float, self._mitre_distribution(raw_cmd.strip(), current_cmd if was_obfuscated else None)))
                    else:
                        response["_evaluation"]["family_specialist"] = [
                            row["probability"] / 100 for row in response["attack_families"]["all_family_scores"]
                        ]

        if baseline_store is None and hasattr(self, "baseline_store") and baseline_ctx is not None and baseline_signature is not None:
            self.baseline_store.observe_and_admit(baseline_signature, baseline_ctx, response)
        elif baseline_store is not None and baseline_ctx is not None and baseline_signature is not None:
            baseline_store.observe_and_admit(baseline_signature, baseline_ctx, response)

        return response
