"""
LLM Security Benchmark Runner
- Loads commands from a command file in either legacy format
    (label|command) or graded format (difficulty|label|command)
- Tests each command against Genos + 3 LLMs (OpenAI, Anthropic, Gemini)
- Round-robins to next model on rate-limit (429) with exponential backoff
- Writes all results to results.txt
"""
import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

# ── Prompt ──────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a security analysis engine.

Analyze the command below and return ONLY minified JSON with this exact schema:

{{"label":"Benign|Suspicious|Malicious","label_confidence":0,"MITRE_codes":[{{"code":"Txxxx","confidence":0}}]}}

Rules:
- Return JSON only — no markdown, no extra keys
- label must be exactly "Benign", "Suspicious", or "Malicious"
- label_confidence is 0-100
- MITRE_codes: up to 3 ranked technique predictions, each with a MITRE code and confidence (0-100)
- MITRE_codes must be an empty array for Benign classifications
- Focus on command behavior only

Command:
{command}
"""

# ── Provider registry ───────────────────────────────────────────────────────
# name → (api_key_env, model_env, base_url, token_param, token_budget)

OPENAI_LIKE: Dict[str, Tuple[str, str, str, str, int]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL",
               "https://api.openai.com/v1", "max_completion_tokens", 256),
}

LLM_PROVIDERS = ["openai", "anthropic"]

GENOS_RETRY_LIMIT = 2
GENOS_RETRY_DELAY = 1.5

# ── Rate-limit state ───────────────────────────────────────────────────────

@dataclass
class ProviderState:
    name: str
    cooldown_until: float = 0.0
    consecutive_429s: int = 0
    total_429s: int = 0
    total_success: int = 0
    total_errors: int = 0

# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class CommandCase:
    id: int
    difficulty: str
    label: str
    command: str

@dataclass
class TestResult:
    command_id: int
    difficulty: str
    command: str
    expected_label: str
    system: str
    model: str
    label: Optional[str] = None
    label_confidence: Optional[float] = None
    mitre_codes: str = "[]"
    correct: Optional[bool] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    tested_by: str = ""

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_commands(path: str) -> List[CommandCase]:
    cases: List[CommandCase] = []
    known_difficulties = {"easy", "medium", "hard", "unspecified"}
    known_labels = {"benign", "suspicious", "malicious"}
    with open(path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try graded format first: difficulty|label|command
            parts3 = line.split("|", 2)
            if (
                len(parts3) == 3
                and parts3[0].strip().lower() in known_difficulties
                and parts3[1].strip().lower() in known_labels
            ):
                difficulty = parts3[0].strip().lower()
                label = parts3[1].strip().lower()
                command = parts3[2].strip()
            else:
                # Legacy format: label|command (command may contain |)
                parts2 = line.split("|", 1)
                if len(parts2) != 2:
                    continue
                if parts2[0].strip().lower() not in known_labels:
                    continue
                difficulty = "unspecified"
                label = parts2[0].strip().lower()
                command = parts2[1].strip()
            idx += 1
            cases.append(
                CommandCase(
                    id=idx,
                    difficulty=difficulty,
                    label=label,
                    command=command,
                )
            )
    return cases


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            return json.loads(text[s : e + 1])
    raise ValueError("no JSON found")


def validate_label(data: dict) -> Tuple[Optional[str], Optional[float], str]:
    label = data.get("label")
    if label not in ("Benign", "Suspicious", "Malicious"):
        raise ValueError(f"bad label: {label!r}")
    conf = data.get("label_confidence", 0)
    codes = data.get("MITRE_codes", [])
    return label, float(conf), json.dumps(codes)


# ── API callers ─────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"429 rate-limited: {provider}")


async def call_genos(client: httpx.AsyncClient, command: str) -> Dict[str, Any]:
    for attempt in range(GENOS_RETRY_LIMIT + 1):
        resp = await client.post(
            os.environ["GENOS_URL"],
            json={"api_key": os.environ.get("GENOS_API_KEY", ""), "command": command},
        )
        if resp.status_code == 429:
            wait = GENOS_RETRY_DELAY * (2 ** attempt)
            print(f"  ⚠ genos rate-limited — retrying in {wait:.1f}s (attempt {attempt + 1}/{GENOS_RETRY_LIMIT + 1})")
            await asyncio.sleep(wait)
            continue
        if resp.status_code >= 500 and attempt < GENOS_RETRY_LIMIT:
            await asyncio.sleep(GENOS_RETRY_DELAY)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


async def call_openai_like(
    client: httpx.AsyncClient, command: str, provider: str,
) -> Dict[str, Any]:
    key_env, model_env, base_url, token_param, budget = OPENAI_LIKE[provider]
    prompt = PROMPT_TEMPLATE.format(command=command)
    resp = await client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ[key_env]}"},
        json={
            "model": os.environ[model_env],
            "temperature": 0,
            token_param: budget,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    if resp.status_code == 429:
        raise RateLimitError(provider)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return extract_json(content)


async def call_anthropic(client: httpx.AsyncClient, command: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(command=command)
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        json={
            "model": os.environ["ANTHROPIC_MODEL"],
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    if resp.status_code == 429:
        raise RateLimitError("anthropic")
    resp.raise_for_status()
    parts = resp.json().get("content", [])
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return extract_json(text)


# ── Dispatch with round-robin fallback on 429 ──────────────────────────────

async def call_llm(
    client: httpx.AsyncClient,
    command: str,
    preferred: str,
    states: Dict[str, ProviderState],
) -> Tuple[Dict[str, Any], str]:
    """Try preferred first; on 429 rotate through others. Returns (json, provider)."""
    order = [preferred] + [p for p in LLM_PROVIDERS if p != preferred]
    last_err: Optional[Exception] = None

    for provider in order:
        st = states[provider]
        now = time.time()
        if now < st.cooldown_until:
            continue

        try:
            if provider == "anthropic":
                raw = await call_anthropic(client, command)
            else:
                raw = await call_openai_like(client, command, provider)
            st.consecutive_429s = 0
            st.total_success += 1
            return raw, provider

        except RateLimitError:
            st.consecutive_429s += 1
            st.total_429s += 1
            backoff = min(5 * (2 ** (st.consecutive_429s - 1)), 120)
            st.cooldown_until = time.time() + backoff
            last_err = RateLimitError(provider)
            print(f"  ⚠ {provider} rate-limited — cooling {backoff}s, rotating…")
            continue

        except Exception as e:
            st.total_errors += 1
            last_err = e
            continue

    # All providers on cooldown — wait for the soonest to expire
    cooling = [st for st in states.values() if st.cooldown_until > time.time()]
    if cooling:
        soonest = min(st.cooldown_until for st in cooling)
        wait = max(soonest - time.time(), 0) + 1
        print(f"  ⏳ all providers cooling — waiting {wait:.0f}s")
        await asyncio.sleep(wait)
        return await call_llm(client, command, preferred, states)

    raise last_err or RuntimeError("all providers failed")


# ── Single test ─────────────────────────────────────────────────────────────

async def run_test(
    client: httpx.AsyncClient,
    case: CommandCase,
    system: str,
    model_name: str,
    states: Dict[str, ProviderState],
) -> TestResult:
    t0 = time.perf_counter()
    expected = {"benign": "Benign", "suspicious": "Suspicious", "malicious": "Malicious"}.get(case.label, "Malicious")
    try:
        if system == "genos":
            raw = await call_genos(client, case.command)
            tested_by = "genos"
        else:
            raw, tested_by = await call_llm(client, case.command, system, states)

        label, conf, mitre = validate_label(raw)
        elapsed = (time.perf_counter() - t0) * 1000
        correct = (label == expected)

        return TestResult(
            command_id=case.id, difficulty=case.difficulty, command=case.command, expected_label=expected,
            system=system, model=model_name, label=label,
            label_confidence=conf, mitre_codes=mitre, correct=correct,
            elapsed_ms=round(elapsed, 1), tested_by=tested_by,
        )
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return TestResult(
            command_id=case.id, difficulty=case.difficulty, command=case.command, expected_label=expected,
            system=system, model=model_name, elapsed_ms=round(elapsed, 1),
            error=str(e)[:200], tested_by="",
        )


# ── Results formatting ─────────────────────────────────────────────────────

def write_results_summary(
    f,
    all_results: Dict[str, List[TestResult]],
    states: Dict[str, ProviderState],
) -> None:
    # ── Summary comparison table ────────────────────────────────
    f.write(f"\n{'=' * 110}\n")
    f.write(f"  COMPARISON SUMMARY\n")
    f.write(f"{'=' * 110}\n\n")
    f.write(f"{'System':>12} │ {'Model':>30} │ {'Correct':>8} │ {'Total':>6} │ {'Accuracy':>9} │ {'Avg ms':>8} │ {'Errors':>7} │ {'429s':>5}\n")
    f.write(f"{'─' * 12}─┼─{'─' * 30}─┼─{'─' * 8}─┼─{'─' * 6}─┼─{'─' * 9}─┼─{'─' * 8}─┼─{'─' * 7}─┼─{'─' * 5}\n")

    for system, rows in all_results.items():
        model = rows[0].model if rows else "?"
        total = len(rows)
        correct = sum(1 for r in rows if r.correct)
        errs = sum(1 for r in rows if r.error)
        avg_ms = sum(r.elapsed_ms for r in rows) / max(total, 1)
        acc = correct / total * 100 if total else 0
        st = states.get(system)
        rate_429 = st.total_429s if st else 0
        f.write(f"{system:>12} │ {model:>30} │ {correct:>8} │ {total:>6} │ {acc:>8.1f}% │ {avg_ms:>7.0f}ms │ {errs:>7} │ {rate_429:>5}\n")

    # ── Rate-limit report ───────────────────────────────────────
    f.write(f"\n{'=' * 110}\n")
    f.write(f"  RATE-LIMIT REPORT\n")
    f.write(f"{'=' * 110}\n\n")
    for name, st in states.items():
        f.write(f"  {name}: {st.total_success} ok, {st.total_429s} rate-limited, {st.total_errors} errors\n")

    # ── Misclassification detail ────────────────────────────────
    f.write(f"\n{'=' * 110}\n")
    f.write(f"  MISCLASSIFICATIONS\n")
    f.write(f"{'=' * 110}\n\n")
    any_miss = False
    for system, rows in all_results.items():
        misses = [r for r in rows if r.correct is False]
        if misses:
            any_miss = True
            f.write(f"  [{system}]\n")
            for r in misses:
                f.write(f"    #{r.command_id} expected={r.expected_label} got={r.label} conf={r.label_confidence}  cmd={r.command[:80]}\n")
            f.write("\n")
    if not any_miss:
        f.write("  None — all classifications correct.\n")

    # ── Difficulty breakdown ──────────────────────────────────────
    f.write(f"\n{'=' * 110}\n")
    f.write("  DIFFICULTY BREAKDOWN\n")
    f.write(f"{'=' * 110}\n\n")
    for system, rows in all_results.items():
        grouped: Dict[str, List[TestResult]] = defaultdict(list)
        for row in rows:
            grouped[row.difficulty].append(row)
        f.write(f"  [{system}]\n")
        for difficulty in sorted(grouped.keys()):
            bucket = grouped[difficulty]
            total = len(bucket)
            correct = sum(1 for row in bucket if row.correct)
            acc = correct / total * 100 if total else 0.0
            f.write(f"    {difficulty:<12} {correct:>4}/{total:<4}  ({acc:>5.1f}%)\n")
        label_difficulty_counts = Counter((row.difficulty, row.expected_label) for row in rows)
        for difficulty, expected_label in sorted(label_difficulty_counts.keys()):
            bucket = [
                row for row in rows
                if row.difficulty == difficulty and row.expected_label == expected_label
            ]
            total = len(bucket)
            correct = sum(1 for row in bucket if row.correct)
            acc = correct / total * 100 if total else 0.0
            f.write(
                f"      {difficulty:<12} {expected_label:<10} {correct:>4}/{total:<4}  ({acc:>5.1f}%)\n"
            )
        f.write("\n")

    f.write(f"\n{'=' * 110}\n")
    f.write(f"  END OF REPORT\n")
    f.write(f"{'=' * 110}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Genos/LLM benchmark against a command corpus.")
    parser.add_argument(
        "--commands",
        default="test_commands.txt",
        help="Path to the command corpus. Supports label|command and difficulty|label|command.",
    )
    parser.add_argument(
        "--systems",
        default="genos,openai,anthropic",
        help="Comma-separated systems to test: genos, openai, anthropic.",
    )
    parser.add_argument(
        "--output",
        default="results.txt",
        help="Path to the output results file.",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    commands = load_commands(args.commands)
    benign = [c for c in commands if c.label == "benign"]
    suspicious = [c for c in commands if c.label == "suspicious"]
    malicious = [c for c in commands if c.label == "malicious"]
    print(f"Loaded {len(benign)} benign + {len(suspicious)} suspicious + {len(malicious)} malicious = {len(commands)} commands\n")

    available_systems = {
        "genos":     "genos-api",
        "openai":    os.environ.get("OPENAI_MODEL", ""),
        "anthropic": os.environ.get("ANTHROPIC_MODEL", ""),
    }
    selected_systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    systems = {name: available_systems[name] for name in selected_systems if name in available_systems}
    if not systems:
        raise SystemExit("No valid systems selected. Use --systems with any of: genos,openai,anthropic")

    states: Dict[str, ProviderState] = {
        name: ProviderState(name=name) for name in LLM_PROVIDERS
    }

    all_results: Dict[str, List[TestResult]] = {}
    total_tests = len(commands) * len(systems)
    done = 0
    rf = open(args.output, "w", encoding="utf-8")
    rf.write(f"{'=' * 110}\n")
    rf.write(f"  LLM SECURITY BENCHMARK — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    rf.write(f"{'=' * 110}\n\n")
    rf.flush()

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            for system, model in systems.items():
                print(f"━━━ {system.upper()} ({model}) ━━━")
                rf.write(f"━━━ {system.upper()} ({model}) ━━━\n")
                rf.write(
                    f"{'#':>3} │ {'Difficulty':>10} │ {'Expected':>10} │ {'Got':>10} │ {'Conf':>5} │ {'OK':>4} │ {'ms':>8} │ {'Tested By':>12} │ Command\n"
                )
                rf.write(f"{'─' * 110}\n")
                rf.flush()

                results: List[TestResult] = []

                for case in commands:
                    done += 1
                    tag = f"[{done}/{total_tests}]"
                    r = await run_test(client, case, system, model, states)
                    results.append(r)

                    ok_sym = "✓" if r.correct else ("✗" if r.correct is False else "ERR")
                    by = f" (via {r.tested_by})" if r.tested_by and r.tested_by != system else ""
                    print(f"  {tag} #{r.command_id:>3} {ok_sym} {r.label or 'ERR':>10} {r.elapsed_ms:>7.0f}ms{by}  {r.command[:60]}")

                    # Write this result immediately to results.txt
                    label_str = r.label or "—"
                    conf_str = f"{r.label_confidence:.0f}" if r.label_confidence is not None else "—"
                    by_str = r.tested_by or "—"
                    cmd_short = r.command[:50] + ("…" if len(r.command) > 50 else "")
                    rf.write(
                        f"{r.command_id:>3} │ {r.difficulty:>10} │ {r.expected_label:>10} │ {label_str:>10} │ {conf_str:>5} │ {ok_sym:>4} │ {r.elapsed_ms:>7.0f}ms │ {by_str:>12} │ {cmd_short}\n"
                    )
                    rf.flush()

                correct = sum(1 for r in results if r.correct)
                errs = sum(1 for r in results if r.error)
                avg_ms = sum(r.elapsed_ms for r in results) / max(len(results), 1)
                acc = correct / len(results) * 100 if results else 0
                rf.write(f"{'─' * 110}\n")
                rf.write(f"→ {system}: {correct}/{len(results)} correct ({acc:.1f}%)  |  Errors: {errs}  |  Avg: {avg_ms:.0f}ms\n\n")
                rf.flush()
                print(f"  → {system}: {correct}/{len(results)} correct ({acc:.1f}%)\n")
                all_results[system] = results

        # ── Write summary + misclassifications at the end ───────────
        write_results_summary(rf, all_results, states)
    finally:
        rf.close()

    print(f"\n✅ All {total_tests} tests complete. Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
