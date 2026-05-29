#!/usr/bin/env python3
"""Build a benign-first Tier 1 provenance patch dataset.

Outputs a deterministic JSONL file of routine operational commands that should
anchor the model's benign baseline before any further ambiguity modeling work.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.benchmark.tier1_case_sets import build_expanded_benign_commands, build_tier1_sanity_buckets


PRIORITY_COMMANDS = [
    "pwd",
    "cal",
    "lsmem",
    "hostname",
    "date",
    "date -u",
    "uptime",
    "uptime -p",
    "whoami",
    "id",
    "groups",
    "docker ps",
    "kubectl get pods",
    "kubectl get pods -n default",
    "cat /etc/os-release",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "data" / "training" / "genos_dataset"))
    parser.add_argument("--output-stem", default="gatekeeper_benign_core_patch")
    return parser.parse_args()


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


PRIORITY_COMMAND_SET = {_normalize_command(command) for command in PRIORITY_COMMANDS}
PRIORITY_COMMAND_RANK = {_normalize_command(command): idx for idx, command in enumerate(PRIORITY_COMMANDS)}


def _priority_sort(commands: list[str]) -> list[str]:
    return sorted(
        commands,
        key=lambda command: (
            PRIORITY_COMMAND_RANK.get(_normalize_command(command), math.inf),
            _normalize_command(command),
        ),
    )


def _ordered_commands(commands: list[str], rng: random.Random) -> list[str]:
    ordered_commands = _priority_sort(commands)
    priority_commands = [command for command in ordered_commands if _normalize_command(command) in PRIORITY_COMMAND_SET]
    deferred_commands = [command for command in ordered_commands if _normalize_command(command) not in PRIORITY_COMMAND_SET]
    rng.shuffle(deferred_commands)
    return priority_commands + deferred_commands


def _scaled_targets(family_specs: list[dict[str, object]], target_rows: int) -> dict[str, int]:
    base_total = sum(int(spec["target"]) for spec in family_specs)
    raw_targets: list[tuple[float, str]] = []
    scaled_targets: dict[str, int] = {}
    total = 0

    for spec in family_specs:
        name = str(spec["name"])
        scaled = (int(spec["target"]) / base_total) * target_rows
        floored = int(math.floor(scaled))
        scaled_targets[name] = floored
        raw_targets.append((scaled - floored, name))
        total += floored

    for _, name in sorted(raw_targets, reverse=True):
        if total >= target_rows:
            break
        scaled_targets[name] += 1
        total += 1

    return scaled_targets


def build_rows(seed: int, target_rows: int) -> tuple[list[dict[str, str]], dict[str, object]]:
    rng = random.Random(seed)
    expanded_benign = build_expanded_benign_commands()
    sanity_buckets = build_tier1_sanity_buckets()

    expanded_by_bucket: dict[str, list[str]] = defaultdict(list)
    for row in expanded_benign:
        expanded_by_bucket[row["bucket"]].append(row["command"])

    selected: list[dict[str, str]] = []
    seen_commands: set[str] = set()
    family_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()

    def add_row(command: str, source_family: str, label_basis: str, provenance_source: str) -> bool:
        normalized = _normalize_command(command)
        if not normalized or normalized in seen_commands:
            return False
        seen_commands.add(normalized)
        selected.append(
            {
                "command": normalized,
                "label": "Benign",
                "label_basis": label_basis,
                "source_type": "benign_core_patch",
                "source_family": source_family,
                "provenance_source": provenance_source,
            }
        )
        family_counts[source_family] += 1
        provenance_counts[provenance_source] += 1
        return True

    family_specs = [
        {
            "name": "trivial_shell_basics",
            "target": 160,
            "label_basis": "routine_operational:trivial_shell_basics",
            "sources": [
                ("tier1_sanity", [row["command"] for row in sanity_buckets["trivial_benign"]]),
                (
                    "curated_routine_operational",
                    [
                        "date -u", "date +%s", "uptime -p", "hostnamectl", "who", "w", "users", "logname", "tty -s",
                        "echo $HOME", "echo $SHELL", "printenv USER", "printenv LANG", "printenv TERM", "which sh", "which ls",
                        "which grep", "command -v python", "command -v pip", "type grep", "type cat", "type pwd", "type history",
                        "printf '%s\n' hello", "printf '%s\n' $PWD", "readlink /bin/sh", "realpath /etc/hostname", "basename /var/log/syslog",
                        "dirname /var/log/syslog", "ls -1", "ls -1a", "ls -1 /etc | head -20", "ls -1 /var/log | head -20",
                        "head -n 5 /proc/version", "head -n 5 /proc/uptime", "head -n 5 /proc/loadavg", "head -n 5 /etc/shells",
                        "wc -l /etc/shells", "wc -c /etc/hosts", "cut -d: -f1 /etc/group | head -10", "sed -n '1,5p' /etc/group",
                        "awk 'NR<=5 {print}' /etc/group", "sort /etc/hosts", "uniq /etc/hosts", "sha1sum /etc/hostname", "cksum /etc/hostname",
                        "stat /etc/os-release", "stat /proc/version", "file /etc/os-release", "file /etc/hosts", "du -sh /etc",
                        "du -sh /var", "du -sh /home", "find /etc -maxdepth 1 -type d | head -10", "find /usr/bin -maxdepth 1 -type f | head -30",
                        "find /bin -maxdepth 1 -type f | head -20", "find /sbin -maxdepth 1 -type f | head -20",
                        "ls -ld /tmp", "ls -ld /var/log", "ls -ld /etc", "wc -w /etc/hostname", "cat /etc/issue",
                    ],
                ),
            ],
        },
        {
            "name": "routine_file_admin_inspection",
            "target": 240,
            "label_basis": "routine_operational:routine_file_admin_inspection",
            "sources": [
                ("tier1_sanity", [row["command"] for row in sanity_buckets["routine_admin"] if "docker" not in row["command"] and "kubectl" not in row["command"]]),
                ("expanded_benign:linux_admin", expanded_by_bucket["linux_admin"]),
                ("expanded_benign:logs", expanded_by_bucket["logs"]),
                ("expanded_benign:filesystem_matrix", expanded_by_bucket["filesystem_matrix"]),
                ("expanded_benign:config_review", expanded_by_bucket["config_review"]),
                ("expanded_benign:linux_service_matrix", expanded_by_bucket["linux_service_matrix"]),
            ],
        },
        {
            "name": "docker_routine_inspection",
            "target": 160,
            "label_basis": "routine_operational:docker_routine_inspection",
            "sources": [
                ("expanded_benign:docker", expanded_by_bucket["docker"]),
                (
                    "curated_routine_operational",
                    [
                        f"docker inspect --format '{{{{.State.Status}}}}' {name}" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        f"docker inspect --format '{{{{json .Config.Env}}}}' {name}" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        f"docker logs {name} --tail=50" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        f"docker exec {name} sh -lc 'pwd && ls -la /app || ls -la /srv || ls -la /var/log'" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        f"docker stats {name} --no-stream" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        f"docker top {name} -eo pid,comm" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        f"docker port {name}" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        f"docker diff {name}" for name in ["api", "db", "redis", "nginx"]
                    ] + [
                        f"docker cp {name}:/etc/hosts /tmp/{name}.hosts" for name in ["api", "db", "redis", "nginx", "worker", "web"]
                    ] + [
                        "docker compose ps", "docker compose config", "docker compose logs --tail=100", "docker system df", "docker context ls",
                        "docker plugin ls", "docker builder ls", "docker manifest inspect nginx:latest",
                    ],
                ),
            ],
        },
        {
            "name": "kubernetes_routine_inspection",
            "target": 160,
            "label_basis": "routine_operational:kubernetes_routine_inspection",
            "sources": [
                ("expanded_benign:kubernetes", expanded_by_bucket["kubernetes"]),
                (
                    "curated_routine_operational",
                    [
                        f"kubectl get pods -n {ns} -o wide" for ns in ["default", "prod", "staging", "ops", "kube-system", "monitoring"]
                    ] + [
                        f"kubectl get svc -n {ns} -o wide" for ns in ["default", "prod", "staging", "ops", "monitoring"]
                    ] + [
                        f"kubectl get deploy -n {ns}" for ns in ["default", "prod", "staging", "ops", "monitoring"]
                    ] + [
                        f"kubectl describe deploy api -n {ns}" for ns in ["default", "prod", "staging", "ops"]
                    ] + [
                        f"kubectl logs deploy/api -n {ns} --tail=50" for ns in ["default", "prod", "staging", "ops"]
                    ] + [
                        f"kubectl top pods -n {ns}" for ns in ["default", "prod", "staging", "ops"]
                    ] + [
                        f"kubectl get configmap -n {ns}" for ns in ["default", "prod", "staging", "ops"]
                    ] + [
                        f"kubectl get ingress -n {ns}" for ns in ["default", "prod", "staging", "ops"]
                    ] + [
                        f"kubectl rollout status deploy/api -n {ns}" for ns in ["default", "prod", "staging", "ops"]
                    ] + [
                        f"kubectl get events -n {ns} --sort-by=.metadata.creationTimestamp | tail -20" for ns in ["default", "prod", "staging", "ops"]
                    ] + [
                        "kubectl version --client", "kubectl cluster-info", "kubectl get nodes -o wide", "kubectl top nodes", "kubectl get namespaces",
                        "kubectl api-resources | head -30", "kubectl config current-context", "kubectl auth can-i get pods -n prod",
                    ],
                ),
            ],
        },
        {
            "name": "network_system_inspection",
            "target": 140,
            "label_basis": "routine_operational:network_system_inspection",
            "sources": [
                ("expanded_benign:networking_benign", expanded_by_bucket["networking_benign"]),
                (
                    "curated_routine_operational",
                    [
                        "ip link show", "ip -br addr", "ip -br route", "ss -s", "ss -tan | head -20", "ss -uan | head -20",
                        "netstat -tlnp | head -20", "netstat -rn", "arp -an", "route -n", "ifconfig -a", "hostname -I",
                        "resolvectl status", "nmcli device status", "networkctl list", "ping -c 2 1.1.1.1", "ping -c 2 8.8.8.8",
                        "dig +short example.com", "dig +short github.com", "nslookup github.com", "host github.com", "curl -I https://github.com",
                        "curl -I https://pypi.org", "wget -S --spider https://example.com 2>&1 | head -20", "traceroute 8.8.8.8 | head -20",
                        "openssl s_client -connect example.com:443 -brief </dev/null", "openssl s_client -connect github.com:443 -brief </dev/null",
                        "tcpdump -D", "ethtool eth0", "ethtool -i eth0", "ip neigh show", "ip rule show", "ip route get 8.8.8.8",
                    ],
                ),
            ],
        },
        {
            "name": "services_packages_and_workflows",
            "target": 140,
            "label_basis": "routine_operational:services_packages_and_workflows",
            "sources": [
                ("expanded_benign:package_managers", expanded_by_bucket["package_managers"]),
                ("expanded_benign:developer_workflows", expanded_by_bucket["developer_workflows"]),
                ("expanded_benign:cloud_cli", expanded_by_bucket["cloud_cli"]),
                ("expanded_benign:database_admin", expanded_by_bucket["database_admin"]),
            ],
        },
    ]
    family_targets = _scaled_targets(family_specs, target_rows)

    actual_family_counts: dict[str, int] = {}
    for spec in family_specs:
        target = family_targets[str(spec["name"])]
        if target <= 0:
            actual_family_counts[str(spec["name"])] = 0
            continue

        commands_by_source = [(source_name, list(commands)) for source_name, commands in spec["sources"]]
        for _, commands in commands_by_source:
            commands[:] = _ordered_commands(commands, rng)
        added = 0
        for source_name, commands in commands_by_source:
            for command in commands:
                if add_row(command, spec["name"], spec["label_basis"], source_name):
                    added += 1
                if added >= target:
                    break
            if added >= target:
                break
        actual_family_counts[spec["name"]] = added

    if len(selected) < target_rows:
        for spec in family_specs:
            if len(selected) >= target_rows:
                break
            commands_by_source = [(source_name, _ordered_commands(list(commands), rng)) for source_name, commands in spec["sources"]]
            for source_name, commands in commands_by_source:
                for command in commands:
                    if len(selected) >= target_rows:
                        break
                    if add_row(command, spec["name"], spec["label_basis"], source_name):
                        actual_family_counts[spec["name"]] += 1
                if len(selected) >= target_rows:
                    break

    if len(selected) < target_rows:
        generated_overflow = []
        admin_paths = [
            "/var/log",
            "/etc/nginx",
            "/srv/app/current",
            "/home/dev/project",
            "/opt/services",
            "/etc/ssh",
            "/var/tmp",
            "/srv/releases/current",
            "/etc/systemd/system",
            "/var/log/nginx",
            "/srv/backups",
            "/opt/config",
            "/usr/local/bin",
            "/var/lib/docker",
            "/etc/kubernetes",
        ]
        service_names = ["nginx", "docker", "sshd", "postgresql", "redis", "kubelet", "cron", "rsyslog", "containerd", "networking", "systemd-timesyncd"]
        docker_names = ["api", "db", "redis", "nginx", "worker", "web", "scheduler", "proxy", "cache", "metrics", "jobs", "frontend"]
        kube_namespaces = ["default", "prod", "staging", "ops", "monitoring", "kube-system", "ingress", "logging"]
        network_targets = ["example.com", "github.com", "pypi.org", "registry.npmjs.org", "python.org", "docker.io", "1.1.1.1", "8.8.8.8"]

        generated_overflow.extend(f"ls -lah {path}" for path in admin_paths)
        generated_overflow.extend(f"du -sh {path}" for path in admin_paths)
        generated_overflow.extend(f"stat {path}" for path in admin_paths)
        generated_overflow.extend(f"file {path}" for path in admin_paths)
        generated_overflow.extend(f"find {path} -maxdepth 2 -type f | head -20" for path in admin_paths)
        generated_overflow.extend(f"find {path} -maxdepth 2 -type d | head -20" for path in admin_paths)
        generated_overflow.extend(f"grep -R 'error' {path} 2>/dev/null | head -20" for path in admin_paths)
        generated_overflow.extend(f"grep -R 'listen' {path} 2>/dev/null | head -20" for path in admin_paths)
        generated_overflow.extend(f"sha256sum {path} 2>/dev/null" for path in admin_paths)
        generated_overflow.extend(f"find {path} -maxdepth 3 -type f | wc -l" for path in admin_paths)
        generated_overflow.extend(f"find {path} -maxdepth 3 -type d | wc -l" for path in admin_paths)
        generated_overflow.extend(f"tree -L 2 {path} 2>/dev/null | head -40" for path in admin_paths)
        generated_overflow.extend(f"lsattr -R {path} 2>/dev/null | head -20" for path in admin_paths)
        generated_overflow.extend(f"systemctl status {name}" for name in service_names)
        generated_overflow.extend(f"systemctl show {name} --property=SubState,ActiveState" for name in service_names)
        generated_overflow.extend(f"journalctl -u {name} --no-pager -n 50" for name in service_names)
        generated_overflow.extend(f"systemctl cat {name} | head -40" for name in service_names)
        generated_overflow.extend(f"journalctl -u {name} --since today | tail -100" for name in service_names)
        generated_overflow.extend(f"systemctl is-enabled {name}" for name in service_names)
        generated_overflow.extend(f"systemctl is-active {name}" for name in service_names)
        generated_overflow.extend(f"service {name} status" for name in service_names)
        generated_overflow.extend(f"docker logs {name} --tail=100" for name in docker_names)
        generated_overflow.extend(f"docker inspect {name}" for name in docker_names)
        generated_overflow.extend(f"docker inspect --format '{{{{.State.Status}}}}' {name}" for name in docker_names)
        generated_overflow.extend(f"docker inspect --format '{{{{.Config.Image}}}}' {name}" for name in docker_names)
        generated_overflow.extend(f"docker inspect --format '{{{{json .NetworkSettings.Ports}}}}' {name}" for name in docker_names)
        generated_overflow.extend(f"docker top {name}" for name in docker_names)
        generated_overflow.extend(f"docker port {name}" for name in docker_names)
        generated_overflow.extend(f"docker diff {name}" for name in docker_names)
        generated_overflow.extend(f"docker exec {name} sh -lc 'pwd && ls -la /app || ls -la /srv || ls -la /var/log'" for name in docker_names)
        generated_overflow.extend(f"docker cp {name}:/etc/hosts /tmp/{name}.hosts" for name in docker_names)
        generated_overflow.extend(f"docker stats {name} --no-stream" for name in docker_names)
        generated_overflow.extend(f"docker exec {name} sh -lc 'env | sort | head -40'" for name in docker_names)
        generated_overflow.extend(f"docker exec {name} sh -lc 'df -h && free -m'" for name in docker_names)
        generated_overflow.extend(f"kubectl get pods -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl get svc -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl get deploy -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl describe deploy api -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl logs deploy/api -n {ns} --tail=100" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl top pods -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl get events -n {ns} --sort-by=.metadata.creationTimestamp | tail -20" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl get configmap -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl get ingress -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl get hpa -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl rollout status deploy/api -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl describe svc api -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"kubectl get endpoints -n {ns}" for ns in kube_namespaces)
        generated_overflow.extend(f"ip route get {target}" for target in ["1.1.1.1", "8.8.8.8"])
        generated_overflow.extend(f"ping -c 2 {target}" for target in network_targets)
        generated_overflow.extend(f"dig +short {target}" for target in network_targets)
        generated_overflow.extend(f"nslookup {target}" for target in network_targets)
        generated_overflow.extend(f"host {target}" for target in network_targets)
        generated_overflow.extend(f"curl -I https://{target}" for target in network_targets if "." in target and not target[0].isdigit())
        generated_overflow.extend([
            "ip -br addr",
            "ip -br route",
            "ip link show",
            "ip neigh show",
            "ip rule show",
            "ss -s",
            "ss -tan | head -20",
            "ss -uan | head -20",
            "netstat -tlnp | head -20",
            "arp -an",
            "ifconfig -a",
            "hostname -I",
            "resolvectl status",
            "nmcli device status",
            "networkctl list",
            "ethtool eth0",
            "ethtool -i eth0",
            "tcpdump -D",
        ])
        generated_overflow.extend(
            [
                "apt list --installed | head -50",
                "apt-cache policy nginx",
                "apt-cache search postgresql | head -20",
                "dpkg -l | grep python",
                "pip list | head -50",
                "pip show flask",
                "pip index versions requests | head -20",
                "npm list --depth=0",
                "npm view react version",
                "yarn list --depth=0",
                "yarn info react version",
                "brew list | head -50",
                "brew info openssl",
                "brew outdated",
                "aws sts get-caller-identity",
                "aws s3 ls",
                "aws eks list-clusters",
                "aws ec2 describe-instances --max-items 5",
                "gcloud projects list",
                "gcloud compute instances list --limit=10",
                "gcloud container clusters list",
                "az account show",
                "az vm list -o table",
                "az aks list -o table",
                "psql -h db.internal -U readonly -c '\\l'",
                "psql -h db.internal -U readonly -c '\\du'",
                "mysql -h db.internal -e 'show databases'",
                "mysql -h db.internal -e 'show processlist'",
                "sqlite3 app.db '.tables'",
                "sqlite3 app.db 'select count(*) from users'",
                "redis-cli info | head -40",
                "git status",
                "git log --oneline -10",
                "git diff --stat HEAD~1",
                "git branch -a",
                "git remote -v",
                "python3 -m pytest tests/unit",
                "python3 -m http.server 8000",
                "npm run build",
                "npm run lint",
                "make test",
                "terraform plan -out=tfplan",
                "helm lint chart/api",
                "go test ./... -run TestHealth",
                "cargo test -- --nocapture",
                "dnf list installed | head -50",
                "yum list installed | head -50",
                "conda env list",
                "conda list",
                "gem list | head -50",
                "cargo install --list | head -50",
                "gh run list --limit 10",
                "gh repo view --json name,defaultBranchRef",
                "poetry install --no-root",
                "poetry run pytest tests/unit -q",
                "ruff check .",
                "mypy src",
                "eslint src --ext .ts,.tsx",
                "prettier --check .",
                "tox -e py312",
                "gradle test --info",
                "mvn test -q",
                "cmake -S . -B build && cmake --build build",
                "go vet ./...",
                "cargo fmt --check",
                "helm template api chart/api | head -80",
                "ansible-playbook site.yml --check",
                "docker compose build",
                "docker build -t app:test .",
                "kubectl version --client",
                "kubectl api-resources | head -40",
                "kubectl config current-context",
                "kubectl auth can-i get pods -n prod",
                "kubectl get nodes -o wide",
                "kubectl get namespaces -o wide",
                "docker system df",
                "docker context ls",
                "docker plugin ls",
                "docker builder ls",
                "docker manifest inspect nginx:latest",
                "docker image history nginx:latest",
                "docker image ls --digests",
                "docker container ls --format '{{.Names}} {{.Status}}'",
                "docker network inspect bridge",
                "docker volume inspect pgdata",
                "head -n 50 /etc/nginx/nginx.conf",
                "head -n 50 /etc/ssh/sshd_config",
                "head -n 50 /etc/fstab",
                "head -n 50 /etc/hosts",
                "head -n 50 /etc/resolv.conf",
                "head -n 50 /etc/crontab",
                "grep -n '^[^#]' /etc/nginx/nginx.conf | head -20",
                "grep -n '^[^#]' /etc/ssh/sshd_config | head -20",
                "grep -n '^[^#]' /etc/fstab | head -20",
                "grep -n '^[^#]' /etc/hosts | head -20",
                "sha256sum /etc/nginx/nginx.conf",
                "sha256sum /etc/ssh/sshd_config",
                "sha256sum /etc/fstab",
                "sha256sum /etc/hosts",
                "tail -n 100 /var/log/syslog",
                "tail -n 100 /var/log/auth.log",
                "tail -n 100 /var/log/nginx/access.log",
                "tail -n 100 /var/log/nginx/error.log",
                "head -n 50 /var/log/syslog",
                "head -n 50 /var/log/auth.log",
                "grep -i error /var/log/syslog | tail -20",
                "grep -i error /var/log/nginx/error.log | tail -20",
                "journalctl -u kubelet --since today | tail -100",
                "journalctl -u docker --since today | tail -100",
                "sar -u 1 3",
                "vmstat 1 3",
                "iostat -xz 1 3",
                "dmesg | tail -50",
                "last | head -20",
                "lastlog | head -20",
                "passwd -S root",
                "id deploy",
                "groups deploy",
            ]
        )

        overflow_commands = []
        for bucket_name in [
            "linux_admin",
            "logs",
            "filesystem_matrix",
            "config_review",
            "linux_service_matrix",
            "package_managers",
            "developer_workflows",
            "cloud_cli",
            "database_admin",
            "networking_benign",
            "docker",
            "kubernetes",
        ]:
            overflow_commands.extend(expanded_by_bucket[bucket_name])
        overflow_commands.extend(row["command"] for row in sanity_buckets["trivial_benign"])
        overflow_commands.extend(row["command"] for row in sanity_buckets["routine_admin"])
        overflow_commands.extend(generated_overflow)
        overflow_commands = _ordered_commands(overflow_commands, rng)
        for command in overflow_commands:
            if len(selected) >= target_rows:
                break
            add_row(
                command,
                "benign_operational_overflow",
                "routine_operational:benign_operational_overflow",
                "overflow_routine_operational",
            )

    if not (target_rows <= len(selected) <= max(target_rows, 2000)):
        raise RuntimeError(f"Expected {target_rows}-{max(target_rows, 2000)} benign patch rows, built {len(selected)}")

    manifest = {
        "seed": seed,
        "target_rows": target_rows,
        "rows": len(selected),
        "source_family_counts": dict(family_counts),
        "provenance_source_counts": dict(provenance_counts),
        "source_family_targets": family_targets,
        "source_family_actual": actual_family_counts,
    }
    return selected, manifest


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest = build_rows(args.seed, args.target_rows)
    output_path = output_dir / f"{args.output_stem}.jsonl"
    manifest_path = output_dir / f"{args.output_stem}_manifest.json"
    write_jsonl(output_path, rows)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        **manifest,
    }, indent=2))


if __name__ == "__main__":
    main()