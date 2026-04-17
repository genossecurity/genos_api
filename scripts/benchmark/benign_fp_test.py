"""
Benign False Positive Test
Sends 100 truly benign commands to the Genos API and reports any false positives.
"""
import json
import requests

API_URL = "http://127.0.0.1:6001/scan/free"

BENIGN_COMMANDS = [
    # File & directory operations
    "ls -la /home/user/documents",
    "cd /var/log",
    "pwd",
    "mkdir -p /home/user/projects/new_app",
    "cp config.yaml config.yaml.bak",
    "mv report.pdf ~/Documents/",
    "rm old_logfile.log",
    "touch notes.txt",
    "find . -name '*.py' -type f",
    "tree -L 2",
    # File viewing / editing
    "cat /etc/hostname",
    "head -n 20 access.log",
    "tail -f /var/log/syslog",
    "less README.md",
    "nano server.conf",
    "vim ~/.bashrc",
    "wc -l main.py",
    "diff file1.txt file2.txt",
    "sort names.txt | uniq",
    "grep -r 'TODO' src/",
    # Package management
    "apt list --installed",
    "pip install flask",
    "pip freeze > requirements.txt",
    "npm install express",
    "npm run build",
    "yarn add react",
    "pip install --upgrade pip",
    "apt-get update",
    "conda activate myenv",
    "brew install wget",
    # Git operations
    "git status",
    "git log --oneline -10",
    "git diff HEAD~1",
    "git add .",
    "git commit -m 'fix: update readme'",
    "git pull origin main",
    "git push origin feature-branch",
    "git branch -a",
    "git checkout -b new-feature",
    "git stash pop",
    # System monitoring
    "top -bn1 | head -20",
    "df -h",
    "du -sh /var/log",
    "free -m",
    "uptime",
    "whoami",
    "hostname",
    "uname -a",
    "ps aux | grep python",
    "lsblk",
    # Networking (benign)
    "ping -c 4 google.com",
    "curl -I https://example.com",
    "wget https://example.com/file.tar.gz",
    "ifconfig",
    "ip addr show",
    "netstat -tlnp",
    "ss -tuln",
    "dig example.com",
    "nslookup google.com",
    "traceroute 8.8.8.8",
    # Docker
    "docker ps",
    "docker images",
    "docker-compose up -d",
    "docker logs my-container",
    "docker build -t myapp:latest .",
    "docker stop web-server",
    "docker exec -it db bash",
    "docker volume ls",
    "docker network ls",
    "docker pull nginx:latest",
    # Python / dev
    "python3 manage.py runserver",
    "python3 -m pytest tests/",
    "python3 setup.py install",
    "python3 -c 'print(\"hello world\")'",
    "flask run --host=0.0.0.0",
    "gunicorn app:app --bind 0.0.0.0:8000",
    "node server.js",
    "gcc -o main main.c",
    "make clean && make",
    "java -jar application.jar",
    # File permissions & users
    "chmod 644 index.html",
    "chown www-data:www-data /var/www",
    "id",
    "groups",
    "passwd",
    # Misc admin
    "crontab -l",
    "systemctl status nginx",
    "journalctl -u ssh --since today",
    "tar -czf backup.tar.gz /home/user/data",
    "unzip archive.zip -d /tmp/output",
    "rsync -avz src/ dest/",
    "scp file.txt user@remote:/home/user/",
    "ssh user@192.168.1.10",
    "date",
    "cal",
]

def main():
    print(f"[*] Testing {len(BENIGN_COMMANDS)} benign commands against {API_URL}")
    print("=" * 70)

    false_positives = []
    errors = []

    for i, cmd in enumerate(BENIGN_COMMANDS, 1):
        try:
            resp = requests.post(API_URL, json={"command": cmd}, timeout=30)
            if resp.status_code != 200:
                errors.append((cmd, f"HTTP {resp.status_code}: {resp.text}"))
                print(f"  [{i:3d}] ERROR  | {cmd[:60]}")
                continue

            result = resp.json()
            label = result.get("label", "unknown")
            conf = result.get("label_confidence", 0)
            mitre = result.get("MITRE_codes", [])

            if label.lower() != "benign":
                false_positives.append({
                    "command": cmd,
                    "label": label,
                    "confidence": conf,
                    "mitre_codes": mitre,
                })
                top_code = mitre[0]["code"] if mitre else "N/A"
                print(f"  [{i:3d}] FP     | {label} ({conf}%) {top_code} | {cmd[:50]}")
            else:
                print(f"  [{i:3d}] OK     | Benign ({conf}%) | {cmd[:50]}")

        except Exception as e:
            errors.append((cmd, str(e)))
            print(f"  [{i:3d}] ERROR  | {cmd[:60]} -> {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(BENIGN_COMMANDS)
    fp_count = len(false_positives)
    err_count = len(errors)
    tp_count = total - fp_count - err_count

    print(f"  Total commands tested : {total}")
    print(f"  Correctly benign (TN) : {tp_count}")
    print(f"  False positives  (FP) : {fp_count}")
    print(f"  Errors                : {err_count}")
    print(f"  FP Rate               : {fp_count / total * 100:.1f}%")

    if false_positives:
        print(f"\n{'─' * 70}")
        print("FALSE POSITIVE DETAILS:")
        print(f"{'─' * 70}")
        for fp in false_positives:
            print(f"\n  Command    : {fp['command']}")
            print(f"  Label      : {fp['label']} ({fp['confidence']}%)")
            if fp['mitre_codes']:
                for m in fp['mitre_codes']:
                    print(f"  MITRE      : {m['code']} ({m['confidence']}%)")

    if errors:
        print(f"\n{'─' * 70}")
        print("ERRORS:")
        print(f"{'─' * 70}")
        for cmd, err in errors:
            print(f"  {cmd[:50]} -> {err}")


if __name__ == "__main__":
    main()
