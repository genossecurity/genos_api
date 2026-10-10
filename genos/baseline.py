"""
baseline.py — Stateful baseline store, typed signature extraction,
novelty scoring with backoff, and poison-resistant admission rules.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Set, Tuple

SIGNATURE_SCHEMA_VERSION = "v1.0"
BASELINE_VERSION = "2026.10.1"


class BaselineMode(str, Enum):
    LEARN = "learn"
    AUDIT = "audit"
    ENFORCE = "enforce"


class BaselineStatus(str, Enum):
    KNOWN_STABLE = "known_stable"
    RARE_OR_NOVEL = "rare_or_novel"
    LEARNING = "learning"
    UNSCOPED = "unscoped"


SENSITIVE_EFFECT_TOKENS = {
    "persistence",
    "credential",
    "defense",
    "recovery",
    "shadow",
    "download",
    "execute",
    "service",
    "scheduled",
    "registry",
    "autorun",
    "remote",
    "bcd",
    "bitsadmin",
    "rundll",
    "schtasks",
    "wmic",
    "certutil",
}


@dataclass(frozen=True)
class ExecutionContext:
    host_id: str
    parent_process: str
    user: str
    timestamp: datetime
    role: Optional[str] = None
    fleet: str = "default"
    in_maintenance_window: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContext":
        ts = data.get("timestamp")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        elif isinstance(ts, datetime):
            dt = ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return cls(
            host_id=str(data.get("host_id", "unknown")),
            parent_process=str(data.get("parent_process", "unknown")).lower().strip(),
            user=str(data.get("user", "unknown")).lower().strip(),
            timestamp=dt.astimezone(timezone.utc),
            role=str(data.get("role", "")).lower().strip() if data.get("role") else None,
            fleet=str(data.get("fleet", "default")).lower().strip(),
            in_maintenance_window=bool(data.get("in_maintenance_window", False)),
        )


@dataclass(frozen=True)
class BaselineSignature:
    executable: str
    verb: Optional[str]
    flag_set: Tuple[str, ...]
    typed_slots: Tuple[str, ...]
    raw_signature: str

    @property
    def level0_exe(self) -> str:
        return self.executable

    @property
    def level1_verb(self) -> str:
        return f"{self.executable}:{self.verb or '*'}"

    @property
    def level2_flags(self) -> str:
        flags_fragment = ",".join(sorted(self.flag_set))
        return f"{self.executable}:{self.verb or '*'}:{flags_fragment}"

    @property
    def level3_full(self) -> str:
        return self.raw_signature


class SignatureExtractor:
    GUID_REGEX = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
    HEX_REGEX = re.compile(r"\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{32,64})\b")
    NUM_REGEX = re.compile(r"\b\d+\b")

    @staticmethod
    def _canonicalize_flag(flag: str) -> str:
        value = str(flag).strip().strip('"\'')
        if not value:
            return ""
        value = value.lower()
        if value.startswith(("-", "/")):
            value = value.lstrip("-/")
        return value

    @staticmethod
    def _classify_path(path: str) -> str:
        lowered = path.lower()
        if any(marker in lowered for marker in ("windows\\system32", "/system32/", "\\system32\\", "/syswow64/", "\\syswow64\\")):
            return "<PATH:system32>"
        if any(marker in lowered for marker in ("users\\public", "/public/", "\\public\\", "programdata", "temp", "/tmp/", "\\temp\\")):
            return "<PATH:users_public>"
        if lowered.startswith(("c:\\", "/", "~")):
            return "<PATH:abs>"
        return "<PATH:relative>"

    @classmethod
    def _extract_flag_values(cls, raw_command: str) -> Tuple[str, ...]:
        seen: Set[str] = set()
        tokens = re.findall(r'"[^"]*"|\S+', raw_command)
        for index, token in enumerate(tokens):
            norm = cls._canonicalize_flag(token)
            if not norm or not token.startswith(("-", "/")):
                continue
            seen.add(norm)
            if index + 1 < len(tokens):
                next_token = tokens[index + 1].strip('"\'')
                if next_token and not next_token.startswith(("-", "/")):
                    seen.add(f"{norm}:{cls._canonicalize_flag(next_token)}")
        return tuple(sorted(seen))

    @classmethod
    def extract(cls, parsed_cmd: Dict[str, Any]) -> BaselineSignature:
        raw_command = str(parsed_cmd.get("raw_command", "") or "")
        exe = (parsed_cmd.get("executable") or "unknown").lower()
        subcmd = parsed_cmd.get("subcommand")
        verb = subcmd.lower() if isinstance(subcmd, str) else None

        flags = tuple(sorted({f.lower() for f in cls._extract_flag_values(raw_command)}))

        slots: list[str] = []
        for path in parsed_cmd.get("file_paths", []) or []:
            slots.append(cls._classify_path(path))
        for _ in parsed_cmd.get("urls", []) or []:
            slots.append("<URL>")
        for _ in parsed_cmd.get("ips", []) or []:
            slots.append("<IP>")
        for _ in parsed_cmd.get("registry_paths", []) or []:
            slots.append("<REGKEY>")

        for arg in parsed_cmd.get("positional_args", []) or []:
            arg_raw = str(arg)
            arg_clean = cls.GUID_REGEX.sub("<GUID>", arg_raw)
            arg_clean = cls.HEX_REGEX.sub("<HEX>", arg_clean)
            arg_clean = cls.NUM_REGEX.sub("<NUM>", arg_clean)
            if arg_raw.lower().startswith(("http://", "https://")):
                slots.append("<URL>")
            elif arg_raw.lower().startswith(("powershell", "cmd", "python", "-encodedcommand", "-command")):
                slots.append("<CMD>")
            elif arg_clean != arg_raw:
                slots.append(arg_clean)
            else:
                slots.append(arg_raw)

        typed_slots = tuple(sorted({slot for slot in slots if slot}))
        raw_sig = "|".join([
            exe,
            verb or "*",
            ",".join(flags),
            ",".join(typed_slots),
        ])

        return BaselineSignature(
            executable=exe,
            verb=verb,
            flag_set=flags,
            typed_slots=typed_slots,
            raw_signature=raw_sig,
        )


@dataclass
class SignatureStats:
    seen_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    distinct_days: Set[str] = field(default_factory=set)
    parent_lineage: Dict[str, int] = field(default_factory=dict)
    hosts_observed: Set[str] = field(default_factory=set)
    is_promoted: bool = False
    flagged_count: int = 0
    max_malicious_probability: float = 0.0
    max_context_probability: float = 0.0
    last_rejection_reason: Optional[str] = None


@dataclass
class BaselineEvaluation:
    status: BaselineStatus
    scope: str
    seen_count: int
    first_seen: Optional[datetime]
    novelty_score: float
    lineage_novel: bool
    surprise_exe: float
    surprise_full: float
    should_run_genos: bool
    rejection_reason: Optional[str] = None


class BaselineStore:
    def __init__(
        self,
        mode: BaselineMode = BaselineMode.AUDIT,
        min_sightings: int = 5,
        min_distinct_days: int = 3,
        max_poison_mal_prob: float = 0.15,
        alert_budget_percentile: float = 0.99,
    ):
        self.mode = mode
        self.min_sightings = min_sightings
        self.min_distinct_days = min_distinct_days
        self.max_poison_mal_prob = max_poison_mal_prob
        self.alert_budget_percentile = alert_budget_percentile

        self._stores: Dict[str, Dict[str, SignatureStats]] = {"fleet": {}, "role": {}, "host": {}}
        self._scope_totals: Dict[str, Dict[str, int]] = {"fleet": {}, "role": {}, "host": {}}
        self._key_totals: Dict[str, Dict[str, int]] = {"fleet": {}, "role": {}, "host": {}}

    def _scope_id(self, scope_level: str, ctx: ExecutionContext) -> str:
        if scope_level == "host":
            return f"host:{ctx.host_id}"
        if scope_level == "role":
            return f"role:{ctx.role or f'user:{ctx.user}'}"
        return f"fleet:{ctx.fleet}"

    def _baseline_key_candidates(self, sig: BaselineSignature) -> Iterable[str]:
        return [sig.level3_full, sig.level2_flags, sig.level1_verb, sig.level0_exe]

    def _can_bypass(self, scope: str, stats: Optional[SignatureStats], ctx: ExecutionContext) -> bool:
        if scope == "host":
            return bool(stats and stats.is_promoted)
        if scope == "role":
            return bool(stats and stats.is_promoted and ctx.role and len(stats.hosts_observed) >= 2)
        return False

    def _has_sensitive_effect(self, genos_result: Dict[str, Any]) -> bool:
        text = " ".join([
            str(genos_result.get("reason", "")),
            " ".join(str(v) for v in genos_result.get("triggered_features", []) or []),
            str(genos_result.get("label", "")),
            str(genos_result.get("family", "")),
            str(genos_result.get("stage", "")),
        ]).lower()
        return any(token in text for token in SENSITIVE_EFFECT_TOKENS)

    def _record_total(self, level: str, scope_id: str, key: str) -> None:
        self._scope_totals.setdefault(level, {})
        self._scope_totals[level][scope_id] = self._scope_totals[level].get(scope_id, 0) + 1
        self._key_totals.setdefault(level, {})
        self._key_totals[level][f"{scope_id}:{key}"] = self._key_totals[level].get(f"{scope_id}:{key}", 0) + 1

    def _calculate_surprise(self, sig: BaselineSignature, ctx: ExecutionContext) -> float:
        host_scope = self._scope_id("host", ctx)
        role_scope = self._scope_id("role", ctx)
        fleet_scope = self._scope_id("fleet", ctx)

        for level, scope_id in (("host", host_scope), ("role", role_scope), ("fleet", fleet_scope)):
            for key in self._baseline_key_candidates(sig):
                count = self._key_totals.get(level, {}).get(f"{scope_id}:{key}", 0)
                total = self._scope_totals.get(level, {}).get(scope_id, 0)
                if total <= 0:
                    continue
                if count <= 0:
                    continue
                probability = (count + 0.5) / (total + 50.0)
                return -math.log2(probability)

        return 15.0

    def evaluate(self, sig: BaselineSignature, ctx: ExecutionContext) -> BaselineEvaluation:
        if ctx.in_maintenance_window:
            return BaselineEvaluation(
                status=BaselineStatus.LEARNING,
                scope="maintenance_window",
                seen_count=0,
                first_seen=None,
                novelty_score=0.0,
                lineage_novel=False,
                surprise_exe=0.0,
                surprise_full=0.0,
                should_run_genos=True,
                rejection_reason="maintenance_window",
            )

        scope_resolved = "fleet"
        best_stats: Optional[SignatureStats] = None
        for level in ("host", "role", "fleet"):
            scope_key = f"{self._scope_id(level, ctx)}:{sig.level3_full}"
            stats = self._stores[level].get(scope_key)
            if stats:
                best_stats = stats
                scope_resolved = level
                break

        parent_key = ctx.parent_process
        lineage_novel = bool(best_stats is None or best_stats.parent_lineage.get(parent_key, 0) == 0)

        surprise_full = self._calculate_surprise(sig, ctx)
        base_exe = BaselineSignature(executable=sig.executable, verb=sig.verb, flag_set=(), typed_slots=(), raw_signature=sig.executable)
        surprise_exe = self._calculate_surprise(base_exe, ctx)
        novelty_score = (0.7 * surprise_full) + (0.3 * surprise_exe)
        if lineage_novel:
            novelty_score += 2.5

        if best_stats and self._can_bypass(scope_resolved, best_stats, ctx) and not lineage_novel:
            status = BaselineStatus.KNOWN_STABLE
            should_run = self.mode == BaselineMode.AUDIT
        elif self.mode == BaselineMode.LEARN:
            status = BaselineStatus.LEARNING
            should_run = True
        else:
            status = BaselineStatus.RARE_OR_NOVEL
            should_run = True

        return BaselineEvaluation(
            status=status,
            scope=scope_resolved,
            seen_count=best_stats.seen_count if best_stats else 0,
            first_seen=best_stats.first_seen if best_stats else None,
            novelty_score=round(novelty_score, 3),
            lineage_novel=lineage_novel,
            surprise_exe=round(surprise_exe, 3),
            surprise_full=round(surprise_full, 3),
            should_run_genos=should_run,
            rejection_reason=None if status is BaselineStatus.KNOWN_STABLE else "novel_signature",
        )

    def observe_and_admit(
        self,
        sig: BaselineSignature,
        ctx: ExecutionContext,
        genos_result: Dict[str, Any],
    ) -> bool:
        day_str = ctx.timestamp.strftime("%Y-%m-%d")
        mal_prob = float((genos_result.get("class_probabilities", {}).get("Malicious") or 0.0) / 100.0)
        context_prob = float((genos_result.get("class_probabilities", {}).get("Context_Dependent") or 0.0) / 100.0)

        self._record_total("host", self._scope_id("host", ctx), sig.level3_full)
        self._record_total("role", self._scope_id("role", ctx), sig.level3_full)
        self._record_total("fleet", self._scope_id("fleet", ctx), sig.level3_full)

        for level in ("host", "role", "fleet"):
            scope_id = self._scope_id(level, ctx)
            key = f"{scope_id}:{sig.level3_full}"
            stats = self._stores[level].setdefault(key, SignatureStats())
            stats.seen_count += 1
            if stats.first_seen is None or ctx.timestamp < stats.first_seen:
                stats.first_seen = ctx.timestamp
            if stats.last_seen is None or ctx.timestamp > stats.last_seen:
                stats.last_seen = ctx.timestamp
            stats.distinct_days.add(day_str)
            stats.hosts_observed.add(ctx.host_id)
            stats.parent_lineage[ctx.parent_process] = stats.parent_lineage.get(ctx.parent_process, 0) + 1
            stats.max_malicious_probability = max(stats.max_malicious_probability, mal_prob)
            stats.max_context_probability = max(stats.max_context_probability, context_prob)

            if self._has_sensitive_effect(genos_result):
                stats.flagged_count += 1
                stats.last_rejection_reason = "sensitive_effect"
                stats.is_promoted = False
                return False

            if mal_prob >= self.max_poison_mal_prob or context_prob >= self.max_poison_mal_prob:
                stats.flagged_count += 1
                stats.last_rejection_reason = "flagged_sighting"
                stats.is_promoted = False
                return False

            if stats.flagged_count > 0:
                stats.is_promoted = False
                return False

            if (
                stats.seen_count >= self.min_sightings
                and len(stats.distinct_days) >= self.min_distinct_days
                and mal_prob <= self.max_poison_mal_prob
                and context_prob <= self.max_poison_mal_prob
                and level == "host"
            ):
                stats.is_promoted = True
                return True

        return False

    def revoke(self, sig: BaselineSignature, ctx: ExecutionContext) -> None:
        for level in ("host", "role", "fleet"):
            scope_id = self._scope_id(level, ctx)
            key = f"{scope_id}:{sig.level3_full}"
            if key in self._stores[level]:
                self._stores[level][key].is_promoted = False
                self._stores[level][key].last_rejection_reason = "revoked"
