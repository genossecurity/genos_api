#!/usr/bin/env python3
"""Build 3-class gatekeeper dataset: Benign | Malicious | Context_Dependent

Approach:
  Phase 1 — Relabel existing benign/malicious data with expanded CTX rules
  Phase 2 — Generate synthetic CTX commands for under-represented categories
  Phase 3 — Merge, split, and write train/val/test CSVs

Usage:
    python parser/build_3class_dataset.py [--dry-run]

Outputs (into data/training/genos_dataset/):
    gatekeeper_3class_train.csv
    gatekeeper_3class_val.csv
    gatekeeper_3class_test.csv

Labeling rubric:
    A command is Context_Dependent if the command string alone does not let
    a skilled analyst confidently decide benign vs malicious.
"""

import argparse
import csv
import os
import random
import re
import sys
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "training", "genos_dataset")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — RELABELING RULES
# ═══════════════════════════════════════════════════════════════════════════════

# ── MITRE techniques that are inherently context-dependent ──────────────────
DISCOVERY_TECHNIQUES = {
    "T1007", "T1010", "T1012", "T1016", "T1018", "T1033", "T1040",
    "T1046", "T1049", "T1057", "T1069", "T1082", "T1083", "T1087",
    "T1120", "T1135", "T1201", "T1217", "T1518", "T1526", "T1580",
    "T1613", "T1614", "T1615", "T1619", "T1622", "T1654",
}
COLLECTION_TECHNIQUES = {"T1005"}
CTX_TECHNIQUES = DISCOVERY_TECHNIQUES | COLLECTION_TECHNIQUES

# ── Malicious-indicator patterns ────────────────────────────────────────────
_MALICIOUS_INDICATORS = re.compile(
    r"|".join([
        r"/dev/tcp/",
        r"mkfifo\s.*/tmp/",
        r"\bexec\s+\d+<>/dev/",
        r"curl\s.*\|\s*(?:ba)?sh",
        r"wget\s.*\|\s*(?:ba)?sh",
        r"curl\s.*-o\s.*/tmp/.*&&.*chmod",
        r"wget\s.*-O\s.*/tmp/.*&&.*chmod",
        r"base64\s+-d.*\|\s*(?:ba)?sh",
        r"eval\s.*\$\(",
        r"-enc\s+[A-Za-z0-9+/=]{20,}",
        r"evil\.com",
        r"malware\.",
        r"attacker\.",
        r"c2\.",
        r"/etc/shadow",
        r"mimikatz",
        r"sekurlsa",
        r"hashdump",
        r"crontab.*\|.*(?:ba)?sh",
        r"/etc/sudoers",
        r"chmod\s+(?:u\+s|4[0-7]{3})\s+/(?:bin|usr/bin|sbin)/",
    ]),
    re.IGNORECASE,
)


def _has_malicious_indicators(cmd: str) -> bool:
    return bool(_MALICIOUS_INDICATORS.search(cmd))


# ── Benign → Context_Dependent rules (EXPANDED) ────────────────────────────

_IDENTITY_ENUM = re.compile(
    r"^\s*(?:whoami|who|w|users|groups|last|lastlog|lastb|finger)\b"
    r"|^\s*id(?:\s|$)"
    r"|^\s*getent\s+(?:passwd|group|shadow)\b",
    re.IGNORECASE,
)

_SYSINFO = re.compile(
    r"^\s*(?:hostname|uname|arch|nproc|lscpu|lshw|dmidecode)\b"
    r"|^\s*lsb_release\b"
    r"|^\s*free\s+(?:-[mhgtl])",
    re.IGNORECASE,
)

_SENSITIVE_FILE_READ = re.compile(
    r"(?:cat|head|tail|less|more|strings|xxd|od)\s+(?:[-\w]*\s+)*/(?:etc|proc)/",
    re.IGNORECASE,
)

_SECURITY_LOGS = re.compile(
    r"/var/log/(?:auth|secure|audit|btmp|wtmp|faillog|lastlog|messages|syslog|kern)",
    re.IGNORECASE,
)

_FILE_INSPECT = re.compile(
    r"(?:stat|file|md5sum|sha\d*sum)\s+/(?:etc|proc|bin|usr/bin|sbin)/",
    re.IGNORECASE,
)

_WC_SENSITIVE = re.compile(r"\bwc\s+.*\s+/etc/", re.IGNORECASE)

_LS_SENSITIVE = re.compile(
    r"\bls\s+(?:-\w+\s+)*/(?:etc|root|home|var/log)(?:/|\s|$)",
    re.IGNORECASE,
)

_NET_ENUM = re.compile(
    r"^\s*(?:netstat|ifconfig|ss|arp|route|traceroute|tracepath|mtr)\b"
    r"|^\s*ip\s+(?:addr|route|neigh|link|rule)\b",
    re.IGNORECASE,
)

_IP_DISCLOSURE = re.compile(
    r"(?:curl|wget)\s+.*(?:ifconfig\.me|icanhazip|ipinfo\.io|checkip|wtfismyip|ip\.me)",
    re.IGNORECASE,
)

_PROC_ENUM = re.compile(
    r"^\s*(?:ps)\b"
    r"|^\s*top\s+-b"
    r"|^\s*(?:pgrep|pstree|pidof)\b",
    re.IGNORECASE,
)

_LSOF = re.compile(r"^\s*lsof\b", re.IGNORECASE)

_RECON_TOOLS = re.compile(
    r"^\s*(?:nmap|masscan|zmap|nbtscan|enum4linux)\b",
    re.IGNORECASE,
)

_PACKET_CAPTURE = re.compile(
    r"^\s*(?:tcpdump|tshark|wireshark)\b",
    re.IGNORECASE,
)

_DEBUG_TRACE = re.compile(r"^\s*(?:strace|ltrace)\b", re.IGNORECASE)

_TUNNELING = re.compile(
    r"ssh\s+.*-[DLR]\s"
    r"|^\s*(?:nc|ncat)\s+.*-l"
    r"|^\s*socat\b",
    re.IGNORECASE,
)

_ENV_INSPECT = re.compile(
    r"^\s*(?:env|printenv)\s*(?:$|\|)"
    r"|^\s*set\s*$",
    re.IGNORECASE,
)

_CRONTAB_LIST = re.compile(r"^\s*crontab\s+-l", re.IGNORECASE)

_SUID_DISCOVERY = re.compile(
    r"find\s+/\s.*-perm\s+-[24]000"
    r"|find\s+/\s.*-writable"
    r"|getcap\s+-r\s+/",
    re.IGNORECASE,
)

_FIND_ROOT = re.compile(r"^\s*find\s+/\s", re.IGNORECASE)
_FIND_CLEANUP = re.compile(r"-delete|-exec\s+rm\b", re.IGNORECASE)

_GREP_SENSITIVE = re.compile(
    r"grep\s+.*(?:password|passwd|shadow|secret|token|credential|private.key|authorized_keys)",
    re.IGNORECASE,
)

_DD_DEVICE = re.compile(
    r"dd\s+.*(?:if=/dev/sd|of=/dev/sd|if=/dev/nvme|of=/dev/nvme|if=/dev/vd|of=/dev/vd)",
    re.IGNORECASE,
)

_CHMOD_SUS = re.compile(
    r"chmod\s+(?:u\+s|g\+s|4[0-7]{3}|2[0-7]{3}|777|a\+rwx)\b",
    re.IGNORECASE,
)

_MOUNT_ENUM = re.compile(
    r"^\s*(?:mount|lsblk|blkid|findmnt)\b"
    r"|^\s*(?:fdisk|parted)\s+-l",
    re.IGNORECASE,
)

_SERVICE_ENUM = re.compile(
    r"systemctl\s+(?:status|list-units|list-unit-files|is-active|is-enabled)"
    r"|^\s*service\s+--status-all",
    re.IGNORECASE,
)

_NC_GENERAL = re.compile(r"^\s*(?:nc|ncat)\b", re.IGNORECASE)

_PASSWD_PARSE = re.compile(
    r"(?:awk|cut|sed)\s+.*(?:/etc/passwd|/etc/group|/etc/shadow)",
    re.IGNORECASE,
)

_SYSCTL = re.compile(r"^\s*sysctl\s+", re.IGNORECASE)

_PROC_PID = re.compile(
    r"(?:cat|head|tail|strings|ls)\s+/proc/\d+/",
    re.IGNORECASE,
)

_HISTORY_CMD = re.compile(r"^\s*history\b", re.IGNORECASE)

_FIND_HOME = re.compile(r"^\s*find\s+/home\b", re.IGNORECASE)

_WHICH_CMD = re.compile(
    r"^\s*(?:which|whereis)\s+"
    r"|^\s*command\s+-v\b"
    r"|^\s*type\s+\w",
    re.IGNORECASE,
)

_DOCKER_ENUM = re.compile(
    r"^\s*docker\s+(?:ps|images|inspect|network\s+ls|volume\s+ls|info|version)\b"
    r"|^\s*docker\s+(?:logs|port|top|stats|diff)\b",
    re.IGNORECASE,
)

_KUBECTL_ENUM = re.compile(
    r"^\s*kubectl\s+(?:get|describe|auth|cluster-info|top|api-resources)\b",
    re.IGNORECASE,
)

_FILE_SYSPATH = re.compile(
    r"^\s*file\s+/(?:bin|usr/bin|sbin|usr/sbin|lib|opt)/",
    re.IGNORECASE,
)


def _benign_is_context_dependent(cmd: str) -> bool:
    """Return True if a currently-benign command should be context_dependent."""
    c = cmd.strip()
    return bool(
        _IDENTITY_ENUM.search(c)
        or _SYSINFO.search(c)
        or _SENSITIVE_FILE_READ.search(c)
        or _SECURITY_LOGS.search(c)
        or _FILE_INSPECT.search(c)
        or _WC_SENSITIVE.search(c)
        or _LS_SENSITIVE.search(c)
        or _NET_ENUM.search(c)
        or _IP_DISCLOSURE.search(c)
        or _PROC_ENUM.search(c)
        or _LSOF.search(c)
        or _RECON_TOOLS.search(c)
        or _PACKET_CAPTURE.search(c)
        or _DEBUG_TRACE.search(c)
        or _TUNNELING.search(c)
        or _ENV_INSPECT.search(c)
        or _CRONTAB_LIST.search(c)
        or _SUID_DISCOVERY.search(c)
        or (_FIND_ROOT.search(c) and not _FIND_CLEANUP.search(c))
        or _GREP_SENSITIVE.search(c)
        or _DD_DEVICE.search(c)
        or _CHMOD_SUS.search(c)
        or _MOUNT_ENUM.search(c)
        or _SERVICE_ENUM.search(c)
        or _NC_GENERAL.search(c)
        or _PASSWD_PARSE.search(c)
        or _SYSCTL.search(c)
        or _PROC_PID.search(c)
        or _HISTORY_CMD.search(c)
        or _FIND_HOME.search(c)
        or _WHICH_CMD.search(c)
        or _DOCKER_ENUM.search(c)
        or _KUBECTL_ENUM.search(c)
        or _FILE_SYSPATH.search(c)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SYNTHETIC CTX GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_ctx():
    """Generate diverse synthetic context-dependent commands."""
    random.seed(42)
    commands = set()

    # Parameter pools
    targets = [
        "10.0.0.1", "10.0.0.5", "10.0.0.10", "10.0.0.100",
        "10.10.10.10", "10.10.10.50",
        "192.168.1.1", "192.168.1.10", "192.168.1.100", "192.168.1.254",
        "192.168.0.1", "192.168.0.50",
        "172.16.0.1", "172.16.0.10", "172.16.0.100",
        "172.16.5.1", "172.16.5.20",
    ]
    subnets = [
        "10.0.0.0/24", "10.0.0.0/16", "10.10.10.0/24",
        "192.168.1.0/24", "192.168.0.0/24", "192.168.0.0/16",
        "172.16.0.0/24", "172.16.0.0/16",
    ]
    users = [
        "root", "admin", "user", "www-data", "nobody", "daemon",
        "postgres", "mysql", "oracle", "tomcat", "apache", "nginx",
        "sshd", "operator", "backup", "deploy", "jenkins",
    ]
    ports = [
        "22", "80", "443", "8080", "8443", "3306", "5432",
        "6379", "27017", "1080", "3389", "445", "139", "53",
        "25", "110", "143", "993", "995", "21", "23", "161",
        "1433", "9200", "5601", "8888", "4444", "9090", "2222",
    ]
    ifaces = ["eth0", "ens33", "ens160", "wlan0", "enp0s3", "enp0s8", "br0"]
    sensitive_files = [
        "/etc/passwd", "/etc/group", "/etc/hosts", "/etc/hostname",
        "/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/os-release",
        "/etc/issue", "/etc/motd", "/etc/profile", "/etc/bashrc",
        "/etc/environment", "/etc/shells", "/etc/fstab",
        "/etc/ssh/sshd_config", "/etc/ssh/ssh_config",
        "/etc/crontab", "/etc/sudoers", "/etc/login.defs",
        "/etc/security/limits.conf", "/etc/pam.d/common-auth",
        "/etc/sysctl.conf", "/etc/default/grub",
        "/etc/apt/sources.list", "/etc/yum.conf",
        "/etc/network/interfaces", "/etc/netplan/01-netcfg.yaml",
        "/etc/exports", "/etc/xinetd.conf",
    ]
    log_files = [
        "/var/log/auth.log", "/var/log/syslog", "/var/log/messages",
        "/var/log/secure", "/var/log/kern.log", "/var/log/dmesg",
        "/var/log/boot.log", "/var/log/cron", "/var/log/maillog",
        "/var/log/apache2/access.log", "/var/log/apache2/error.log",
        "/var/log/nginx/access.log", "/var/log/nginx/error.log",
        "/var/log/mysql/error.log", "/var/log/postgresql/postgresql.log",
        "/var/log/audit/audit.log", "/var/log/faillog",
        "/var/log/wtmp", "/var/log/btmp", "/var/log/lastlog",
    ]
    proc_files = [
        "/proc/version", "/proc/cpuinfo", "/proc/meminfo",
        "/proc/cmdline", "/proc/mounts", "/proc/partitions",
        "/proc/net/tcp", "/proc/net/udp", "/proc/net/arp",
        "/proc/net/route", "/proc/net/dev", "/proc/net/fib_trie",
        "/proc/sys/kernel/hostname", "/proc/sys/net/ipv4/ip_forward",
        "/proc/self/status", "/proc/self/maps", "/proc/self/environ",
    ]
    services = [
        "sshd", "nginx", "apache2", "httpd", "mysql", "postgresql",
        "redis", "docker", "kubelet", "cron", "rsyslog",
        "firewalld", "ufw", "fail2ban", "auditd",
        "vsftpd", "smbd", "nfs-server", "postfix",
    ]
    find_patterns = [
        "*.conf", "*.cfg", "*.ini", "*.xml", "*.yaml", "*.yml",
        "*.log", "*.bak", "*.old", "*.swp", "*.key", "*.pem",
        "*.crt", "*.p12", "*.db", "*.sqlite", "*.sql",
        "id_rsa", "id_ed25519", "authorized_keys", ".bash_history",
        ".ssh", ".gnupg", ".git-credentials", "shadow", "passwd",
        "*.sh", "*.py", "*.pl",
    ]
    pids = ["1", "2", "1234", "4567", "$(pgrep sshd)", "$(pgrep nginx)"]

    # ── 1. Identity / User Enumeration ──────────────────────────────────
    for cmd in [
        "whoami", "id", "id -u", "id -g", "id -G", "id -Gn", "id -un",
        "id -gn", "id -p", "who", "who -a", "who -b", "who -q",
        "who -r", "who am i", "w", "w -h", "w -s", "w -f",
        "users", "groups",
        "last", "last -5", "last -10", "last -20", "last -50",
        "last -a", "last -d", "last -i", "last -w",
        "lastlog", "lastb",
        "getent passwd", "getent group", "getent shadow",
        "getent hosts", "getent services",
        "compgen -u", "compgen -g",
        "cat /etc/passwd | wc -l",
        "awk -F: '{print $1}' /etc/passwd",
        "cut -d: -f1 /etc/passwd",
        "awk -F: '{print $1}' /etc/group",
        "cut -d: -f1 /etc/group",
        "grep -c '' /etc/passwd",
        "cat /etc/passwd | awk -F: '{print $1, $3, $6}'",
        "getent passwd | awk -F: '$3 >= 1000 {print $1}'",
        "getent passwd | awk -F: '$3 == 0 {print $1}'",
    ]:
        commands.add(cmd)
    for u in users[:12]:
        commands.add(f"id {u}")
        commands.add(f"last {u}")
        commands.add(f"finger {u}")
        commands.add(f"getent passwd {u}")
        commands.add(f"groups {u}")
        commands.add(f"chage -l {u}")

    # ── 2. System Information Discovery ─────────────────────────────────
    for cmd in [
        "hostname", "hostname -f", "hostname -i", "hostname -I",
        "hostname -d", "hostname -s",
        "uname -a", "uname -r", "uname -m", "uname -s", "uname -n",
        "uname -p", "uname -o", "uname -v",
        "arch", "nproc", "nproc --all",
        "lscpu", "lscpu -e", "lscpu --json",
        "lshw -short", "lshw -class network", "lshw -class disk",
        "dmidecode -t system", "dmidecode -t bios", "dmidecode -t memory",
        "dmidecode --type 1",
        "free -m", "free -h", "free -g", "free -t", "free --si",
        "uptime", "uptime -p", "uptime -s",
        "lsb_release -a", "lsb_release -d", "lsb_release -r",
        "cat /etc/os-release", "cat /etc/lsb-release",
        "cat /etc/redhat-release", "cat /etc/debian_version",
        "cat /etc/issue", "cat /etc/issue.net",
        "timedatectl", "timedatectl status",
        "date", "date -u", "hwclock",
        "dmesg | head -20", "dmesg | tail -20",
        "dmesg | grep -i error", "dmesg | grep -i warning",
        "sysctl -a 2>/dev/null | head -50",
        "sysctl kernel.hostname", "sysctl kernel.osrelease",
        "sysctl net.ipv4.ip_forward",
        "ulimit -a", "locale", "locale -a",
        "cat /proc/version",
    ]:
        commands.add(cmd)
    for pf in proc_files:
        commands.add(f"cat {pf}")
        if "sys" not in pf:
            commands.add(f"head -20 {pf}")

    # ── 3. Network Enumeration ──────────────────────────────────────────
    for cmd in [
        "ifconfig", "ifconfig -a",
        "ip addr", "ip addr show", "ip -4 addr", "ip -6 addr",
        "ip route", "ip route show", "ip route get 8.8.8.8",
        "ip neigh", "ip neigh show",
        "ip link", "ip link show", "ip rule show", "ip tunnel show",
        "netstat -tulpn", "netstat -an", "netstat -rn", "netstat -i",
        "netstat -s", "netstat -lntp", "netstat -aunp",
        "netstat -plant", "netstat --listening",
        "ss -tulpn", "ss -an", "ss -lntp", "ss -s",
        "ss -anp", "ss -o state established",
        "ss -tunlp", "ss -4 state listening",
        "arp -a", "arp -n", "arp -e",
        "route -n", "route",
        "traceroute 8.8.8.8", "traceroute -n 8.8.8.8",
        "tracepath 8.8.8.8",
        "mtr --report 8.8.8.8", "mtr -n 8.8.8.8",
        "curl -s ifconfig.me", "curl -s icanhazip.com",
        "curl -s ipinfo.io", "curl -s checkip.amazonaws.com",
        "wget -qO- ifconfig.me", "wget -qO- icanhazip.com",
        "dig +short myip.opendns.com @resolver1.opendns.com",
    ]:
        commands.add(cmd)
    for iface in ifaces[:5]:
        commands.add(f"ifconfig {iface}")
        commands.add(f"ip addr show {iface}")
        commands.add(f"ethtool {iface}")
        commands.add(f"ip -s link show {iface}")
    for t in targets[:10]:
        commands.add(f"traceroute {t}")
        commands.add(f"traceroute -n {t}")
        commands.add(f"ping -c 4 {t}")
        commands.add(f"arping -c 3 {t}")

    # ── 4. Sensitive File Inspection ────────────────────────────────────
    for sf in sensitive_files:
        commands.add(f"cat {sf}")
        commands.add(f"head -20 {sf}")
        commands.add(f"tail -20 {sf}")
        commands.add(f"stat {sf}")
        if "/etc/" in sf:
            commands.add(f"wc -l {sf}")
    for lf in log_files[:12]:
        commands.add(f"cat {lf}")
        commands.add(f"tail -50 {lf}")
        commands.add(f"tail -100 {lf}")
        commands.add(f"head -50 {lf}")
        commands.add(f"grep -i error {lf}")
        commands.add(f"grep -i fail {lf}")
    for cmd in [
        "ls -la /etc/", "ls -la /etc/ssh/", "ls -la /etc/cron.d/",
        "ls -la /etc/pam.d/", "ls -la /etc/security/",
        "ls -la /root/", "ls -la /root/.ssh/",
        "ls -la /home/", "ls -laR /home/",
        "ls -la /var/log/", "ls -la /var/spool/cron/",
        "ls -la /tmp/", "ls -la /var/tmp/",
        "ls -la /opt/", "ls -la /srv/",
        "ls -la /var/www/", "ls -la /var/www/html/",
        "find /etc -name '*.conf' -type f",
        "find /etc -name '*.cfg' -type f",
        "find /var/log -name '*.log' -type f",
        "find /var/log -mtime -1 -type f",
        "find /tmp -type f -ls", "find /var/tmp -type f -ls",
        "find /home -name '.bash_history' -type f",
        "find /home -name 'id_rsa' -type f 2>/dev/null",
        "find /home -name '.ssh' -type d 2>/dev/null",
        "find /home -name 'authorized_keys' 2>/dev/null",
        "find /root -type f -ls 2>/dev/null",
        "grep -r 'password' /etc/ 2>/dev/null | head -20",
        "grep -r 'secret' /etc/ 2>/dev/null | head -20",
        "grep -r 'PASS' /etc/ 2>/dev/null | head -20",
        "grep -ri 'api.key' /etc/ 2>/dev/null",
        "grep -ri 'token' /etc/ 2>/dev/null | head -20",
        "grep -ri 'credential' /etc/ 2>/dev/null",
        "grep -r 'password' /var/www/ 2>/dev/null",
        "grep -r 'DB_PASS' /var/www/ 2>/dev/null",
        "cat /etc/passwd | grep -v nologin",
        "cat /etc/passwd | grep '/bin/bash'",
        "cat /etc/group | grep sudo",
        "cat /etc/group | grep wheel",
        "cat /etc/group | grep admin",
        "strings /proc/1/environ 2>/dev/null",
    ]:
        commands.add(cmd)

    # ── 5. Process Enumeration ──────────────────────────────────────────
    for cmd in [
        "ps aux", "ps -ef", "ps auxww", "ps -eo pid,user,args",
        "ps -eo pid,ppid,user,stat,args", "ps -eo pid,user,%cpu,%mem,args",
        "ps -eo pid,user,args --sort=-%mem | head -20",
        "ps -eo pid,user,args --sort=-%cpu | head -20",
        "ps aux --sort=-%mem | head -20",
        "ps aux --sort=-%cpu | head -20",
        "ps aux | grep root", "ps aux | grep -v root",
        "ps -ef | grep ssh", "ps -ef | grep python",
        "ps -ef | grep java", "ps -ef | grep -v grep",
        "ps auxf", "ps auxwww",
        "top -bn1", "top -bn1 | head -20",
        "top -bn1 -o %MEM | head -20",
        "top -bn1 -o %CPU | head -20",
        "pstree", "pstree -p", "pstree -u", "pstree -a",
        "pgrep -la ssh", "pgrep -la python", "pgrep -la nginx",
        "pgrep -la apache", "pgrep -la java",
        "pgrep -u root", "pgrep -u root --list-full",
        "lsof -i", "lsof -i -P -n",
        "lsof -i :22", "lsof -i :80", "lsof -i :443", "lsof -i :8080",
        "lsof -i :3306", "lsof -i :5432",
        "lsof -u root", "lsof +D /tmp", "lsof +D /var/tmp",
        "lsof -c sshd", "lsof -c nginx", "lsof -c apache",
        "fuser 80/tcp", "fuser 443/tcp", "fuser 22/tcp",
        "fuser -v /dev/sda1",
    ]:
        commands.add(cmd)
    for pid in pids[:5]:
        commands.add(f"cat /proc/{pid}/cmdline")
        commands.add(f"cat /proc/{pid}/status")
        commands.add(f"cat /proc/{pid}/maps")
        commands.add(f"cat /proc/{pid}/environ")
        commands.add(f"ls -la /proc/{pid}/fd/")
        commands.add(f"readlink /proc/{pid}/exe")

    # ── 6. Recon / Offensive Dual-Use Tools ─────────────────────────────
    for t in targets[:12]:
        commands.add(f"nmap {t}")
        commands.add(f"nmap -sV {t}")
        commands.add(f"nmap -sS {t}")
        commands.add(f"nmap -A {t}")
        commands.add(f"nmap -O {t}")
        commands.add(f"nmap --script=default {t}")
        commands.add(f"nmap -sU {t}")
    for sn in subnets[:5]:
        commands.add(f"nmap {sn}")
        commands.add(f"nmap -sn {sn}")
        commands.add(f"nmap -sP {sn}")
        commands.add(f"masscan {sn} -p 22,80,443 --rate 1000")
        commands.add(f"masscan {sn} --top-ports 100 --rate 10000")
    for t in targets[:8]:
        for p in ["22", "80", "443", "8080", "3306"]:
            commands.add(f"nmap -p {p} {t}")
    for t in targets[:6]:
        commands.add(f"nmap -sV --script=vuln {t}")
        commands.add(f"nmap -sC -sV -p- {t}")
        commands.add(f"nmap --script=smb-enum-shares {t}")
        commands.add(f"nmap --script=http-enum {t}")
        commands.add(f"nikto -h http://{t}")
        commands.add(f"nikto -h https://{t}")
        commands.add(f"nikto -h http://{t}:8080")
        commands.add(f"gobuster dir -u http://{t} -w /usr/share/wordlists/dirb/common.txt")
        commands.add(f"dirb http://{t}")
        commands.add(f"enum4linux {t}")
        commands.add(f"enum4linux -a {t}")
        commands.add(f"smbclient -L {t} -N")
        commands.add(f"rpcclient -U '' -N {t}")
        commands.add(f"nbtscan {t}")
        commands.add(f"onesixtyone {t} public")
        commands.add(f"snmpwalk -v2c -c public {t}")
    for t in targets[:5]:
        for p in ports[:6]:
            commands.add(f"nmap -sV -p {p} {t}")
    for cmd in [
        "sqlmap -u 'http://target.com/page?id=1' --batch",
        "sqlmap -u 'http://target.com/page?id=1' --dbs",
        "sqlmap -u 'http://target.com/page?id=1' --tables",
        "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.0.0.1",
        "hydra -l admin -P /usr/share/wordlists/rockyou.txt ftp://10.0.0.1",
        "hydra -l root -P wordlist.txt ssh://192.168.1.1",
        "hydra -L users.txt -P passwords.txt 10.0.0.1 ssh",
        "medusa -h 10.0.0.1 -u admin -P wordlist.txt -M ssh",
        "wpscan --url http://target.com",
        "wpscan --url http://target.com --enumerate u",
        "wpscan --url http://target.com --enumerate vp",
        "whatweb http://target.com",
        "wafw00f http://target.com",
        "fierce --domain target.com",
        "dnsrecon -d target.com",
        "dnsenum target.com",
        "theHarvester -d target.com -b google",
        "amass enum -d target.com",
        "subfinder -d target.com",
        "dig axfr target.com @ns1.target.com",
        "host -t axfr target.com ns1.target.com",
    ]:
        commands.add(cmd)

    # ── 7. Tunneling / Forwarding ───────────────────────────────────────
    for t in targets[:8]:
        commands.add(f"ssh -D 1080 user@{t}")
        commands.add(f"ssh -D 9050 user@{t}")
        commands.add(f"ssh -D 1080 -fNq user@{t}")
        commands.add(f"ssh -L 8080:localhost:80 user@{t}")
        commands.add(f"ssh -L 3306:localhost:3306 user@{t}")
        commands.add(f"ssh -L 5432:localhost:5432 user@{t}")
        commands.add(f"ssh -R 8080:localhost:80 user@{t}")
        commands.add(f"ssh -R 4444:localhost:22 user@{t}")
        commands.add(f"ssh -J jumphost user@{t}")
        commands.add(f"ssh -fN -L 8080:{t}:80 user@{t}")
    for p in ["4444", "8080", "1234", "9999", "1337"]:
        commands.add(f"nc -lvp {p}")
        commands.add(f"nc -lnvp {p}")
        commands.add(f"ncat -lvp {p}")
        commands.add(f"ncat -l {p}")
    for cmd in [
        "socat TCP-LISTEN:8080,fork TCP:10.0.0.1:80",
        "socat TCP-LISTEN:4444,reuseaddr,fork EXEC:/bin/bash",
        "socat TCP-LISTEN:1234,fork TCP:192.168.1.1:22",
        "chisel server --reverse --port 8080",
        "chisel client 10.0.0.1:8080 R:1080:socks",
        "chisel client 10.0.0.1:8080 R:8443:127.0.0.1:443",
        "ssh -w 0:0 user@10.0.0.1",
        "sshuttle -r user@10.0.0.1 10.0.0.0/24",
        "sshuttle -r user@10.0.0.1 0/0",
        "proxychains nmap -sT 10.0.0.0/24",
        "proxychains curl http://10.0.0.1",
    ]:
        commands.add(cmd)

    # ── 8. Packet Capture / Monitoring ──────────────────────────────────
    for iface in ifaces[:5]:
        commands.add(f"tcpdump -i {iface}")
        commands.add(f"tcpdump -i {iface} -n")
        commands.add(f"tcpdump -i {iface} -w /tmp/capture.pcap")
        commands.add(f"tcpdump -i {iface} -c 100")
        commands.add(f"tcpdump -i {iface} port 80")
        commands.add(f"tcpdump -i {iface} port 443")
        commands.add(f"tcpdump -i {iface} port 22")
        commands.add(f"tcpdump -i {iface} -nn")
        commands.add(f"tshark -i {iface}")
        commands.add(f"tshark -i {iface} -w /tmp/capture.pcap")
        commands.add(f"tshark -i {iface} -f 'port 80'")
    for t in targets[:6]:
        commands.add(f"tcpdump -i eth0 host {t}")
        commands.add(f"tcpdump -i eth0 dst {t}")
        commands.add(f"tcpdump -i eth0 src {t}")
        commands.add(f"tshark -i eth0 -f 'host {t}'")
    for cmd in [
        "tcpdump -i any", "tcpdump -i any -n -c 50",
        "tcpdump -D", "tcpdump --list-interfaces",
        "tshark -D", "tshark -r /tmp/capture.pcap",
        "tcpdump -i eth0 -A | grep -i password",
        "tcpdump -i eth0 -A | grep -i cookie",
        "tcpdump -i eth0 icmp",
        "tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn) != 0'",
    ]:
        commands.add(cmd)

    # ── 9. Shell Spawning (ambiguous) ───────────────────────────────────
    for cmd in [
        "python3 -c 'import pty; pty.spawn(\"/bin/bash\")'",
        "python -c 'import pty; pty.spawn(\"/bin/bash\")'",
        "python3 -c 'import pty; pty.spawn(\"/bin/sh\")'",
        "python -c 'import pty; pty.spawn(\"/bin/sh\")'",
        "script -qc /bin/bash /dev/null",
        "script -qc /bin/sh /dev/null",
        "expect -c 'spawn /bin/bash; interact'",
        "rlwrap nc -lvp 4444", "rlwrap ncat -lvp 4444",
        "socat file:`tty`,raw,echo=0 tcp-listen:4444",
        "python3 -c 'import os; os.system(\"/bin/bash\")'",
        "perl -e 'exec(\"/bin/bash\")'",
        "ruby -e 'exec \"/bin/bash\"'",
        "lua -e 'os.execute(\"/bin/bash\")'",
        "/bin/bash -i", "/bin/sh -i",
        "bash --norc --noprofile", "zsh -f",
    ]:
        commands.add(cmd)

    # ── 10. SUID / Permission Finding ───────────────────────────────────
    for cmd in [
        "find / -perm -4000 -type f 2>/dev/null",
        "find / -perm -4000 -type f -ls 2>/dev/null",
        "find / -perm -u=s -type f 2>/dev/null",
        "find / -perm -2000 -type f 2>/dev/null",
        "find / -perm -g=s -type f 2>/dev/null",
        "find / -perm /6000 -type f 2>/dev/null",
        "find / -perm -4000 -user root -type f 2>/dev/null",
        "find / -writable -type d 2>/dev/null",
        "find / -writable -type f 2>/dev/null",
        "find / -writable ! -user $(whoami) 2>/dev/null",
        "find / -perm -o+w -type d 2>/dev/null",
        "find / -perm -o+w -type f 2>/dev/null",
        "find / -user root -writable -type f 2>/dev/null",
        "find / -nouser -type f 2>/dev/null",
        "find / -nogroup -type f 2>/dev/null",
        "getcap -r / 2>/dev/null",
        "getcap -r /usr 2>/dev/null",
        "/sbin/getcap -r / 2>/dev/null",
        "find / -name '*.sh' -perm -4000 2>/dev/null",
    ]:
        commands.add(cmd)

    # ── 11. Root Filesystem Exploration ─────────────────────────────────
    for pat in find_patterns:
        commands.add(f"find / -name '{pat}' -type f 2>/dev/null")
        if not pat.startswith("."):
            commands.add(f"find / -name '{pat}' 2>/dev/null | head -20")
    for cmd in [
        "find / -type f -size +100M 2>/dev/null",
        "find / -type f -size +500M 2>/dev/null",
        "find / -type f -mtime -1 2>/dev/null | head -50",
        "find / -type f -mtime -7 2>/dev/null | head -50",
        "find / -type f -newer /tmp/timestamp 2>/dev/null",
        "find / -name '*.log' -mtime -1 2>/dev/null",
        "find / -type f -name '*.php' 2>/dev/null",
        "find / -type f -name '*.jsp' 2>/dev/null",
        "find / -type f -name 'wp-config*' 2>/dev/null",
        "find / -type f -name '*.properties' 2>/dev/null",
        "find / -type f -name '.env' 2>/dev/null",
        "find / -type f -name 'config.php' 2>/dev/null",
        "find / -type f -name 'settings.py' 2>/dev/null",
        "find / -maxdepth 3 -type f -name '*.key' 2>/dev/null",
        "find / -maxdepth 3 -type f -name '*.pem' 2>/dev/null",
        "find / -maxdepth 4 -name '.git' -type d 2>/dev/null",
        "find / -name 'id_rsa' -o -name 'id_dsa' -o -name 'id_ed25519' 2>/dev/null",
        "ls -laR /", "ls -la /",
        "tree / -L 2 2>/dev/null | head -50",
        "du -sh /* 2>/dev/null",
    ]:
        commands.add(cmd)

    # ── 12. Crontab / Scheduled Tasks ───────────────────────────────────
    for cmd in [
        "crontab -l", "crontab -l -u root",
        "cat /etc/crontab",
        "ls -la /etc/cron.d/", "ls -la /etc/cron.daily/",
        "ls -la /etc/cron.hourly/", "ls -la /etc/cron.weekly/",
        "ls -la /etc/cron.monthly/",
        "cat /etc/cron.d/*", "cat /etc/cron.daily/*",
        "find /etc/cron* -type f -ls",
        "systemctl list-timers", "systemctl list-timers --all",
        "atq", "at -l",
        "ls -la /var/spool/cron/",
        "ls -la /var/spool/cron/crontabs/",
    ]:
        commands.add(cmd)
    for u in users[:6]:
        commands.add(f"crontab -l -u {u}")

    # ── 13. Environment / Configuration ─────────────────────────────────
    for cmd in [
        "env", "printenv", "set",
        "env | sort", "printenv | sort",
        "echo $PATH", "echo $HOME", "echo $USER", "echo $SHELL",
        "echo $HOSTNAME", "echo $SSH_CLIENT", "echo $SSH_CONNECTION",
        "echo $DISPLAY", "echo $TERM", "echo $LANG",
        "echo $LD_PRELOAD", "echo $LD_LIBRARY_PATH",
        "env | grep -i proxy", "env | grep -i pass",
        "env | grep -i key", "env | grep -i secret",
        "env | grep -i token", "env | grep -i aws",
        "env | grep PATH", "env | grep SSH",
        "printenv HOME", "printenv PATH", "printenv SHELL",
        "sysctl -a 2>/dev/null | wc -l",
        "sysctl net.ipv4.ip_forward",
        "sysctl net.ipv4.conf.all.forwarding",
        "sysctl kernel.randomize_va_space",
        "sysctl fs.protected_hardlinks",
        "cat /etc/environment",
        "cat /etc/profile", "cat /etc/bashrc",
        "cat ~/.bashrc", "cat ~/.bash_profile", "cat ~/.profile",
        "cat ~/.bash_history",
        "history", "history | tail -50",
        "cat ~/.zsh_history 2>/dev/null",
        "cat ~/.mysql_history 2>/dev/null",
    ]:
        commands.add(cmd)

    # ── 14. Service / Package Enumeration ───────────────────────────────
    for cmd in [
        "systemctl list-units", "systemctl list-units --type=service",
        "systemctl list-unit-files",
        "systemctl list-unit-files --state=enabled",
        "systemctl list-unit-files --state=disabled",
        "service --status-all",
        "chkconfig --list 2>/dev/null",
        "dpkg -l", "dpkg -l | wc -l",
        "dpkg -l | grep -i ssh", "dpkg -l | grep -i ssl",
        "dpkg -l | grep -i python", "dpkg -l | grep -i java",
        "dpkg --get-selections",
        "rpm -qa", "rpm -qa | wc -l",
        "rpm -qa | grep -i ssh", "rpm -qa | sort",
        "apt list --installed 2>/dev/null",
        "apt list --installed 2>/dev/null | wc -l",
        "pip list", "pip3 list",
        "pip list 2>/dev/null | grep -i request",
        "gem list", "npm list -g --depth=0",
        "snap list 2>/dev/null",
    ]:
        commands.add(cmd)
    for svc in services[:12]:
        commands.add(f"systemctl status {svc}")
        commands.add(f"systemctl is-active {svc}")
        commands.add(f"systemctl is-enabled {svc}")

    # ── 15. Debugging / Tracing ─────────────────────────────────────────
    for cmd in [
        "strace -p 1 2>&1 | head -20",
        "strace -e trace=network -p 1234",
        "strace -e trace=file -p 1234",
        "strace -f -e trace=open,read,write -p 1234",
        "strace -c ls /", "ltrace ls /", "ltrace -p 1234",
    ]:
        commands.add(cmd)
    for pid in pids[:3]:
        commands.add(f"strace -p {pid}")
        commands.add(f"ltrace -p {pid}")
        commands.add(f"gdb -batch -p {pid} -ex 'info proc mappings'")

    # ── 16. Disk / Mount / Filesystem ───────────────────────────────────
    for cmd in [
        "mount", "mount -l", "mount -l -t nfs",
        "mount | grep -v tmpfs",
        "df -h", "df -i", "df -hT",
        "lsblk", "lsblk -f", "lsblk -a",
        "blkid", "blkid -o list",
        "fdisk -l 2>/dev/null", "parted -l 2>/dev/null",
        "findmnt", "findmnt -t nfs", "findmnt -t ext4",
        "cat /etc/fstab", "cat /proc/mounts", "cat /proc/partitions",
        "du -sh /home/*", "du -sh /var/*",
        "du -sh /tmp/*", "du -sh /opt/*",
        "pvs 2>/dev/null", "vgs 2>/dev/null", "lvs 2>/dev/null",
    ]:
        commands.add(cmd)

    # ── 17. Container / Cloud Enumeration ──────────────────────────────
    for cmd in [
        "docker ps", "docker ps -a", "docker ps -q", "docker images",
        "docker images -a", "docker inspect $(docker ps -q)",
        "docker network ls", "docker network inspect bridge",
        "docker volume ls", "docker info", "docker version",
        "docker logs $(docker ps -q | head -1)",
        "docker top $(docker ps -q | head -1)",
        "docker stats --no-stream",
        "docker exec -it $(docker ps -q | head -1) /bin/bash",
        "kubectl get pods", "kubectl get pods -A",
        "kubectl get pods -o wide", "kubectl get pods --all-namespaces",
        "kubectl get secrets", "kubectl get secrets -A",
        "kubectl get configmaps", "kubectl get configmaps -A",
        "kubectl get nodes", "kubectl get nodes -o wide",
        "kubectl get svc", "kubectl get svc -A",
        "kubectl get namespaces", "kubectl get ingress -A",
        "kubectl get deployments -A", "kubectl get daemonsets -A",
        "kubectl get pv", "kubectl get pvc -A",
        "kubectl auth can-i --list",
        "kubectl auth can-i create pods",
        "kubectl cluster-info",
        "kubectl top nodes", "kubectl top pods -A",
        "kubectl api-resources",
        "kubectl describe nodes",
        "kubectl get events -A",
        "kubectl get rolebindings -A",
        "kubectl get clusterrolebindings",
    ]:
        commands.add(cmd)

    # ── 18. Cloud Metadata ──────────────────────────────────────────────
    for cmd in [
        "curl -s http://169.254.169.254/latest/meta-data/",
        "curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "curl -s http://169.254.169.254/latest/meta-data/hostname",
        "curl -s http://169.254.169.254/latest/meta-data/local-ipv4",
        "curl -s http://169.254.169.254/latest/meta-data/public-ipv4",
        "curl -s http://169.254.169.254/latest/meta-data/security-groups",
        "curl -s http://169.254.169.254/latest/user-data",
        "curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/",
        "curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/hostname",
        "curl -s -H 'Metadata: true' http://169.254.169.254/metadata/instance?api-version=2021-02-01",
        "wget -qO- http://169.254.169.254/latest/meta-data/",
        "wget -qO- http://169.254.169.254/latest/user-data",
    ]:
        commands.add(cmd)

    # ── 19. Tool / Binary Discovery ─────────────────────────────────────
    tools = [
        "python", "python3", "perl", "ruby", "gcc", "cc", "make",
        "wget", "curl", "nc", "ncat", "socat", "nmap", "tcpdump",
        "gdb", "strace", "ltrace", "gawk", "lua", "php",
        "ssh", "scp", "rsync", "tar", "zip", "base64",
        "openssl", "gpg", "xxd", "strings",
    ]
    for tool in tools:
        commands.add(f"which {tool}")
        commands.add(f"whereis {tool}")
        commands.add(f"command -v {tool}")
        commands.add(f"type {tool}")
        commands.add(f"file $(which {tool} 2>/dev/null)")

    # ── 20. Sudo / Privilege Enumeration ────────────────────────────────
    for cmd in [
        "sudo -l", "sudo -l -U root", "sudo -l -U admin",
        "sudo -V", "sudo -n id",
        "cat /etc/sudoers 2>/dev/null",
        "cat /etc/sudoers.d/* 2>/dev/null",
        "sudo cat /etc/shadow",
        "pkexec --help",
        "doas -C /etc/doas.conf whoami 2>/dev/null",
    ]:
        commands.add(cmd)

    # ── 21. Wireless / Network Tools ────────────────────────────────────
    for cmd in [
        "iwconfig", "iwlist wlan0 scan",
        "airmon-ng", "airodump-ng wlan0",
        "nmcli device wifi list", "nmcli device show",
        "nmcli connection show", "iw dev wlan0 scan",
    ]:
        commands.add(cmd)

    # ── 22. Command History Access ──────────────────────────────────────
    for cmd in [
        "history", "history | tail -50", "history | tail -100",
        "history | grep ssh", "history | grep pass",
        "history | grep sudo", "history | grep curl",
        "history | grep wget", "history | grep connect",
        "cat ~/.bash_history", "cat ~/.bash_history | tail -100",
        "cat ~/.bash_history | grep -i password",
        "cat ~/.bash_history | grep -i ssh",
        "cat ~/.zsh_history 2>/dev/null",
        "cat ~/.mysql_history 2>/dev/null",
        "cat ~/.python_history 2>/dev/null",
        "cat ~/.psql_history 2>/dev/null",
        "cat ~/.node_repl_history 2>/dev/null",
        "cat ~/.rediscli_history 2>/dev/null",
    ]:
        commands.add(cmd)

    # ── 23. DNS Recon ───────────────────────────────────────────────────
    domains = ["target.com", "internal.corp", "example.com", "10.0.0.1"]
    for d in domains:
        commands.add(f"dig {d}")
        commands.add(f"dig {d} ANY")
        commands.add(f"dig {d} MX")
        commands.add(f"dig {d} NS")
        commands.add(f"dig {d} TXT")
        commands.add(f"nslookup {d}")
        commands.add(f"host {d}")
        commands.add(f"host -t mx {d}")

    # ── 24. Extended find variations ────────────────────────────────────
    for d in ["/home", "/opt", "/srv", "/var/www", "/var/lib", "/tmp"]:
        commands.add(f"find {d} -name '*.conf' -type f 2>/dev/null")
        commands.add(f"find {d} -name '*.key' -type f 2>/dev/null")
        commands.add(f"find {d} -name '*.pem' -type f 2>/dev/null")
        commands.add(f"find {d} -name '.env' -type f 2>/dev/null")
        commands.add(f"find {d} -name 'id_rsa' -type f 2>/dev/null")
        commands.add(f"find {d} -writable -type f 2>/dev/null")
        commands.add(f"ls -laR {d}/ 2>/dev/null | head -100")

    # ── 25. Extended port-specific enumeration ──────────────────────────
    for p in ports[:15]:
        commands.add(f"netstat -an | grep :{p}")
        commands.add(f"ss -lntp | grep :{p}")
        commands.add(f"lsof -i :{p}")

    result = sorted(commands)
    random.shuffle(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DATASET ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def relabel_split(split: str):
    """Relabel one split. Returns (rows, stats)."""
    gatekeeper_path = os.path.join(BASE, f"gatekeeper_{split}.csv")
    specialist_path = os.path.join(BASE, f"specialist_{split}.csv")
    stats = Counter()
    rows = []

    with open(gatekeeper_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            cmd = row[0]
            if _benign_is_context_dependent(cmd):
                new_label = "Context_Dependent"
                stats["benign→ctx"] += 1
            else:
                new_label = "Benign"
                stats["benign→benign"] += 1
            rows.append((cmd, new_label, "Benign", "Benign"))

    with open(specialist_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            cmd, mitre_id = row[0], row[1]
            if mitre_id in CTX_TECHNIQUES and not _has_malicious_indicators(cmd):
                new_label = "Context_Dependent"
                stats["malicious→ctx"] += 1
            else:
                new_label = "Malicious"
                stats["malicious→malicious"] += 1
            rows.append((cmd, new_label, "Malicious", mitre_id))

    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    print("=" * 72)
    print("  GATEKEEPER 3-CLASS DATASET BUILDER (v2 — expanded CTX)")
    print("=" * 72)

    # Phase 2: Generate synthetic CTX
    synthetic_cmds = _generate_synthetic_ctx()
    print(f"\n  Synthetic CTX generated: {len(synthetic_cmds)} unique commands")

    # Split synthetic 80/10/10
    random.seed(42)
    n = len(synthetic_cmds)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    syn_splits = {
        "train": synthetic_cmds[:n_train],
        "val":   synthetic_cmds[n_train:n_train + n_val],
        "test":  synthetic_cmds[n_train + n_val:],
    }
    for s, cmds in syn_splits.items():
        print(f"    {s}: {len(cmds)} synthetic CTX")

    # Phase 1 + 3: Relabel and merge
    grand_stats = Counter()
    for split in ("train", "val", "test"):
        rows, stats = relabel_split(split)
        grand_stats += stats

        for cmd in syn_splits[split]:
            rows.append((cmd, "Context_Dependent", "Synthetic", "Synthetic"))
        stats["synthetic_ctx"] = len(syn_splits[split])

        random.seed(hash(split))
        random.shuffle(rows)

        total = len(rows)
        benign = sum(1 for r in rows if r[1] == "Benign")
        malicious = sum(1 for r in rows if r[1] == "Malicious")
        ctx = sum(1 for r in rows if r[1] == "Context_Dependent")

        print(f"\n── {split.upper()} ({'dry run' if args.dry_run else 'written'}) ──")
        print(f"  Total:             {total:>6}")
        print(f"  Benign:            {benign:>6}  ({100*benign/total:.1f}%)")
        print(f"  Malicious:         {malicious:>6}  ({100*malicious/total:.1f}%)")
        print(f"  Context_Dependent: {ctx:>6}  ({100*ctx/total:.1f}%)")
        print(f"  ── Sources ──")
        print(f"    benign→benign:       {stats['benign→benign']:>5}")
        print(f"    benign→ctx:          {stats['benign→ctx']:>5}")
        print(f"    malicious→malicious: {stats['malicious→malicious']:>5}")
        print(f"    malicious→ctx:       {stats['malicious→ctx']:>5}")
        print(f"    synthetic ctx:       {stats['synthetic_ctx']:>5}")

        if not args.dry_run:
            out_path = os.path.join(BASE, f"gatekeeper_3class_{split}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["command", "label", "original_label", "mitre_id"])
                writer.writerows(rows)

    # Grand totals
    print(f"\n{'=' * 72}")
    print(f"  GRAND TOTALS")
    print(f"{'=' * 72}")
    for k in ("benign→benign", "benign→ctx", "malicious→malicious", "malicious→ctx"):
        print(f"  {k:30s} {grand_stats[k]:>6}")
    total_relabeled_ctx = grand_stats["benign→ctx"] + grand_stats["malicious→ctx"]
    print(f"  {'relabeled ctx':30s} {total_relabeled_ctx:>6}")
    print(f"  {'synthetic ctx':30s} {len(synthetic_cmds):>6}")
    print(f"  {'total ctx':30s} {total_relabeled_ctx + len(synthetic_cmds):>6}")

    if not args.dry_run:
        print(f"\n  Sample synthetic (first 15):")
        for cmd in synthetic_cmds[:15]:
            print(f"    {cmd[:90]}")

    print(f"\n{'=' * 72}")
    if args.dry_run:
        print("  DRY RUN — no files written")
    else:
        print("  DONE — datasets written to data/training/genos_dataset/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()