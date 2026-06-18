#!/usr/bin/env python3
"""Redact obvious secrets from shell history before provenance ingestion.

This is intentionally conservative: it preserves command structure while masking
common secret-bearing values such as tokens, credentials, internal domains, and
private IPs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PRIVATE_IPV4_RE = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|127(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
URL_CREDENTIAL_RE = re.compile(r"(?P<scheme>https?://)(?P<user>[^\s:/@]+):(?P<password>[^\s@]+)@")
LONG_TOKEN_RE = re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_\-.]{20,})\b")
ASSIGNMENT_RE = re.compile(
    r"\b(?P<name>(?:api|access|auth|bearer|db|deploy|github|gitlab|jwt|passwd|password|secret|session|token|webhook)[A-Za-z0-9_\-]*)=(?P<value>[^\s'\"]+)\b",
    flags=re.IGNORECASE,
)
FLAG_VALUE_RE = re.compile(
    r"(?P<flag>--?(?:token|password|passwd|secret|api[-_]?key|access[-_]?key|client[-_]?secret))\s+(?P<value>[^\s]+)",
    flags=re.IGNORECASE,
)
HEADER_BEARER_RE = re.compile(r"(Authorization:\s*Bearer\s+)([^\s'\"]+)", flags=re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+(?:corp|internal|local|lan|home|cluster|svc)\b", flags=re.IGNORECASE)
HOME_USER_RE = re.compile(r"(/home/)([^/\s]+)")
SSH_TARGET_RE = re.compile(r"\b([A-Za-z0-9._-]+)@([A-Za-z0-9._-]+)\b")


def redact_line(line: str, *, redact_private_ips: bool, redact_internal_domains: bool, redact_usernames: bool) -> str:
    redacted = line.rstrip("\n")
    redacted = URL_CREDENTIAL_RE.sub(r"\g<scheme><redacted-user>:<redacted-secret>@", redacted)
    redacted = ASSIGNMENT_RE.sub(lambda m: f"{m.group('name')}=<redacted-secret>", redacted)
    redacted = FLAG_VALUE_RE.sub(lambda m: f"{m.group('flag')} <redacted-secret>", redacted)
    redacted = HEADER_BEARER_RE.sub(r"\1<redacted-secret>", redacted)
    redacted = LONG_TOKEN_RE.sub("<redacted-secret>", redacted)
    if redact_private_ips:
        redacted = PRIVATE_IPV4_RE.sub("<redacted-private-ip>", redacted)
    if redact_internal_domains:
        redacted = DOMAIN_RE.sub("<redacted-internal-domain>", redacted)
    if redact_usernames:
        redacted = HOME_USER_RE.sub(r"\1<redacted-user>", redacted)
        redacted = SSH_TARGET_RE.sub("<redacted-user>@<redacted-host>", redacted)
    return redacted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="Path to the raw shell history or command text file.")
    parser.add_argument("output_path", help="Path to write the sanitized output.")
    parser.add_argument("--keep-private-ips", action="store_true", help="Do not redact private IPv4 addresses.")
    parser.add_argument("--keep-internal-domains", action="store_true", help="Do not redact internal-looking domains.")
    parser.add_argument("--keep-usernames", action="store_true", help="Do not redact usernames in /home paths or user@host targets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            sanitized = redact_line(
                line,
                redact_private_ips=not args.keep_private_ips,
                redact_internal_domains=not args.keep_internal_domains,
                redact_usernames=not args.keep_usernames,
            )
            if sanitized != line.rstrip("\n"):
                changed += 1
            dst.write(sanitized + "\n")

    print(
        f"Sanitized {total} lines from {input_path} -> {output_path} "
        f"(changed {changed} lines)."
    )


if __name__ == "__main__":
    main()