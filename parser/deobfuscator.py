"""
deobfuscator.py — standalone deobfuscation helpers extracted from GenosEngine.

No ML dependencies. Only stdlib: base64, json, math, re.
Optional: pyminusone (PowerShell-specific deobfuscation).
"""

import base64
import json
import math
import re

try:
    import pyminusone
except ImportError:
    pyminusone = None

MAX_LAYERS = 5


# ── entropy ──────────────────────────────────────────────────────────────────

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    entropy = 0.0
    for x in range(256):
        p_x = float(text.count(chr(x))) / len(text)
        if p_x > 0:
            entropy += -p_x * math.log(p_x, 2)
    return entropy


# ── detection ─────────────────────────────────────────────────────────────────

_ENCODED_CMD_RE = re.compile(
    r"(?i)-(?:enc(?:odedcommand)?)\s+([A-Za-z0-9+/=]{20,})"
)

def is_obfuscated(text: str) -> bool:
    patterns = [
        r"\[char\]",
        r"base64",
        r"frombase64",
        r"reverse\(",
        r"\+[ ]*'",
        r"\$[a-z0-9_]{10,}",
        r"\\x[0-9a-f]{2}",
    ]
    if any(re.search(p, text, re.I) for p in patterns):
        return True
    if _ENCODED_CMD_RE.search(text):
        return True
    if calculate_entropy(text) > 5.2:
        return True
    return False


# ── paren helper ──────────────────────────────────────────────────────────────

def _find_matching_paren(text: str, start_index: int) -> int:
    if start_index < 0 or start_index >= len(text) or text[start_index] != "(":
        return -1

    depth = 0
    in_single = False
    in_double = False
    i = start_index

    while i < len(text):
        ch = text[i]

        if ch == "`":
            i += 2
            continue

        if in_single:
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


# ── layer decoders ────────────────────────────────────────────────────────────

def decode_powershell_encoded_command(text: str) -> str:
    """Detect -Enc[odedCommand] <blob> and replace the whole command with the decoded payload."""
    match = _ENCODED_CMD_RE.search(text)
    if not match:
        return text
    blob = match.group(1)
    try:
        decoded = base64.b64decode(blob).decode("utf-8", errors="ignore")
        if len(decoded) > 3:
            return decoded
    except Exception:
        pass
    return text


def universal_decoder(text: str) -> str:
    """Attempt to base64-decode a bare base64 blob."""
    try:
        if re.match(
            r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$",
            text.strip(),
        ):
            decoded = base64.b64decode(text.strip()).decode("utf-8", errors="ignore")
            if len(decoded) > 3:
                return decoded
    except Exception:
        pass
    return text


def decode_embedded_base64(text: str) -> str:
    """Replace FromBase64String('...') calls with their decoded content."""
    pattern = re.compile(r"FromBase64String\(\s*['\"]([A-Za-z0-9+/=]{8,})['\"]\s*\)", re.I)

    def _decode(match):
        b64_payload = match.group(1)
        try:
            decoded = base64.b64decode(b64_payload).decode("utf-8", errors="ignore")
            return json.dumps(decoded)
        except Exception:
            return match.group(0)

    return pattern.sub(_decode, text)


def deobfuscate_char_constructions(text: str) -> str:
    """Resolve [char] range loops and single [char] casts."""
    range_loop_pattern = re.compile(
        r"\(\s*(\d{1,3})\s*\.\.\s*(\d{1,3})\s*\)\s*\|\s*%\s*\{\s*\[char\]\s*\$_\s*\}",
        re.I,
    )

    def _range_to_chars(match):
        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            start, end = end, start
        start = max(0, min(start, 255))
        end = max(0, min(end, 255))
        return "".join(chr(i) for i in range(start, end + 1))

    text = range_loop_pattern.sub(lambda m: json.dumps(_range_to_chars(m)), text)

    single_char_pattern = re.compile(r"\[char\]\s*\(?\s*(\d{1,3})\s*\)?", re.I)

    def _single_char(match):
        value = max(0, min(int(match.group(1)), 255))
        return json.dumps(chr(value))

    text = single_char_pattern.sub(_single_char, text)

    mixed_concat_pattern = re.compile(
        r"\(\s*(\d{1,3})\s*\.\.\s*(\d{1,3})\s*\)\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|\s*%\s*\{\s*\[char\]\s*\$_\s*\}",
        re.I,
    )

    def _mixed_concat(match):
        start = int(match.group(1))
        end = int(match.group(2))
        suffix = match.group(3)
        lead = chr(max(0, min(start, 255)))
        if abs(start - end) <= 32:
            return json.dumps(f"{lead}{suffix}")
        step = 1 if end >= start else -1
        decoded = "".join(chr(max(0, min(i, 255))) for i in range(start, end + step, step))
        return json.dumps(f"{decoded}{suffix}")

    return mixed_concat_pattern.sub(_mixed_concat, text)


def clean_concatenation(text: str) -> str:
    """Collapse adjacent string concatenations: "a" + "b" → "ab"."""
    quoted_join = re.compile(r"\"((?:\\.|[^\"\\])*)\"\s*\+\s*\"((?:\\.|[^\"\\])*)\"")
    while True:
        new_text = quoted_join.sub(lambda m: json.dumps(m.group(1) + m.group(2)), text)
        if new_text == text:
            break
        text = new_text

    q_plus_word = re.compile(r"\"((?:\\.|[^\"\\])*)\"\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)")
    text = q_plus_word.sub(lambda m: json.dumps(m.group(1) + m.group(2)), text)
    return text


def _extract_invocation_payload(text: str):
    """Extract the argument of a &(builder)(payload) PowerShell invocation."""
    s = text.strip()
    if not s.startswith("&("):
        return None

    builder_start = s.find("(")
    builder_end = _find_matching_paren(s, builder_start)
    if builder_end == -1:
        return None

    idx = builder_end + 1
    while idx < len(s) and s[idx].isspace():
        idx += 1

    if idx >= len(s) or s[idx] != "(":
        return None

    payload_end = _find_matching_paren(s, idx)
    if payload_end == -1:
        return None

    if s[payload_end + 1:].strip():
        return None

    payload = s[idx + 1:payload_end].strip()
    return payload or None


def extract_powershell_payload(text: str):
    """Strip PowerShell invocation wrappers and UTF8 encoding calls."""
    payload = _extract_invocation_payload(text)
    if payload is None:
        payload = text.strip()

    utf8_match = re.match(
        r"^\s*\[System\.Text\.Encoding\]::UTF8\.GetString\(\s*\[System\.Convert\]::(?P<quoted>(?:\"(?:\\.|[^\"\\])*\")|(?:'(?:\\.|[^'\\])*'))\s*\)\s*$",
        payload,
        re.I,
    )
    if utf8_match:
        quoted = utf8_match.group("quoted")
        if quoted.startswith('"'):
            try:
                return json.loads(quoted).strip()
            except Exception:
                return quoted.strip('"').strip()
        return quoted.strip("'").strip()

    return payload if payload != text.strip() else None


# ── public entry point ────────────────────────────────────────────────────────

def deobfuscate_layer(text: str) -> str:
    """Apply one round of deobfuscation transforms."""
    text = decode_powershell_encoded_command(text)
    text = universal_decoder(text)
    text = decode_embedded_base64(text)

    payload_only = extract_powershell_payload(text)
    if payload_only:
        text = payload_only

    text = deobfuscate_char_constructions(text)
    text = clean_concatenation(text)

    if pyminusone:
        try:
            text = pyminusone.deobfuscate(text, lang="powershell")
        except Exception:
            pass

    text = deobfuscate_char_constructions(text)
    text = clean_concatenation(text)

    return text


def deobfuscate(text: str, max_layers: int = MAX_LAYERS) -> str:
    """
    Iteratively deobfuscate until the text stabilises or max_layers is reached.
    Returns the clearest decoded form found.
    """
    current = text
    for _ in range(max_layers):
        if not is_obfuscated(current):
            break
        next_text = deobfuscate_layer(current)
        if next_text == current:
            break
        current = next_text
    return current
