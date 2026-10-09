"""
deobfuscator.py — Standalone deobfuscation engine for Genos.

Handles detection and iterative decoding of multi-layer command obfuscation:
- Bare Base64 payloads
- PowerShell -EncodedCommand / -e / -enc flags (UTF-16LE and UTF-8)
- Shell piped Base64 (e.g. echo ... | base64 -d)
- Embedded Base64 (FromBase64String)
- Character construction expressions ([char] range loops, casts, mixed concat)
- String concatenation resolution
- PowerShell execution wrapper stripping (&(builder)(payload))
- Shannon entropy measurement
- Optional pyminusone integration
"""

import base64
import json
import math
import re
from typing import Optional, Tuple

try:
    import pyminusone
except ImportError:
    pyminusone = None

DEFAULT_MAX_LAYERS = 5
DEFAULT_ENTROPY_THRESHOLD = 5.2
DEFAULT_ENTROPY_DELTA_STOP = 0.01


# ── Entropy ───────────────────────────────────────────────────────────────────

def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy for byte frequencies in text."""
    if not text:
        return 0.0
    entropy = 0.0
    text_len = len(text)
    for x in range(256):
        count = text.count(chr(x))
        if count > 0:
            p_x = float(count) / text_len
            entropy += -p_x * math.log(p_x, 2)
    return entropy


# ── Bare Base64 ───────────────────────────────────────────────────────────────

def decode_bare_base64(text: str) -> str:
    """Decode bare Base64 string if printable characters comprise >90% of output."""
    stripped = text.strip()
    if len(stripped) < 8 or len(stripped) % 4 != 0:
        return text
    if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", stripped):
        return text
    try:
        raw = base64.b64decode(stripped, validate=True)
        for encoding in ("utf-16-le", "utf-8"):
            try:
                decoded = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            printable = sum(1 for c in decoded if c in "\r\n\t" or " " <= c <= "~")
            if len(decoded) > 3 and (printable / len(decoded)) > 0.9:
                return decoded
    except ValueError:
        pass
    return text


# ── Obfuscation Detection ─────────────────────────────────────────────────────

_ENCODED_CMD_RE = re.compile(
    r"(?i)-(?:enc(?:odedcommand)?)\s+([A-Za-z0-9+/=]{20,})"
)

_OBFUSCATION_PATTERNS = [
    re.compile(r"\[char\]", re.I),
    re.compile(r"base64", re.I),
    re.compile(r"frombase64", re.I),
    re.compile(r"reverse\(", re.I),
    re.compile(r"\+[ ]*'", re.I),
    re.compile(r"\$[a-z0-9_]{10,}", re.I),
    re.compile(r"\\x[0-9a-f]{2}", re.I),
    _ENCODED_CMD_RE,
]


def is_obfuscated(text: str, entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD) -> bool:
    """Determine if text exhibits signs of obfuscation or encoding."""
    if not text:
        return False
    if decode_bare_base64(text) != text:
        return True
    if any(p.search(text) for p in _OBFUSCATION_PATTERNS):
        return True
    if calculate_entropy(text) > entropy_threshold:
        return True
    return False


# ── Parenthesis Matching Helper ───────────────────────────────────────────────

def find_matching_paren(text: str, start_index: int) -> int:
    """Find the closing parenthesis matching the open paren at start_index."""
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


# ── Decoding Transforms ───────────────────────────────────────────────────────

def decode_powershell_encoded_command(text: str) -> str:
    """Detect -Enc[odedCommand] <blob> and replace it with decoded command payload."""
    match = _ENCODED_CMD_RE.search(text)
    if not match:
        return text
    blob = match.group(1)
    try:
        raw = base64.b64decode(blob)
        try:
            utf16 = raw.decode("utf-16-le")
            ascii_printable = sum(1 for c in utf16 if '\x20' <= c <= '\x7e' or c in '\r\n\t')
            if ascii_printable > len(utf16) * 0.6 and len(utf16) > 3:
                return utf16
        except (UnicodeDecodeError, ValueError):
            pass
        decoded = raw.decode("utf-8", errors="ignore")
        if len(decoded) > 3:
            return decoded
    except Exception:
        pass
    return text


_SHELL_B64_PIPE_RE = re.compile(
    r"""(?:echo|printf|echo\s+-[neE]+)\s+
        ['"]?
        ([A-Za-z0-9+/]{20,}={0,2})
        ['"]?
        \s*\|\s*base64\s+-d""",
    re.X | re.I,
)


def decode_shell_base64_pipe(text: str) -> str:
    """Decode shell base64 piped execution (e.g. echo ... | base64 -d)."""
    def _repl(m):
        blob = m.group(1)
        try:
            decoded = base64.b64decode(blob).decode("utf-8", errors="ignore")
            printable = sum(1 for c in decoded if c.isprintable() or c in '\r\n\t')
            if printable > len(decoded) * 0.7 and len(decoded) > 3:
                return m.group(0).replace(blob, decoded)
        except Exception:
            pass
        return m.group(0)
    return _SHELL_B64_PIPE_RE.sub(_repl, text)


def universal_decoder(text: str) -> str:
    """Attempt bare base64 decoding for full string blobs."""
    try:
        stripped = text.strip()
        if re.match(
            r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$",
            stripped,
        ):
            decoded = base64.b64decode(stripped).decode("utf-8", errors="ignore")
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
    """Collapse adjacent string concatenations: 'a' + 'b' -> 'ab'."""
    quoted_join = re.compile(r"\"((?:\\.|[^\"\\])*)\"\s*\+\s*\"((?:\\.|[^\"\\])*)\"")
    while True:
        new_text = quoted_join.sub(lambda m: json.dumps(m.group(1) + m.group(2)), text)
        if new_text == text:
            break
        text = new_text

    q_plus_word = re.compile(r"\"((?:\\.|[^\"\\])*)\"\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)")
    text = q_plus_word.sub(lambda m: json.dumps(m.group(1) + m.group(2)), text)
    return text


def extract_powershell_payload(text: str) -> Optional[str]:
    """Strip PowerShell invocation wrappers (&(builder)(payload)) and UTF8 encoding calls."""
    s = text.strip()
    payload = None
    if s.startswith("&("):
        builder_start = s.find("(")
        builder_end = find_matching_paren(s, builder_start)
        if builder_end != -1:
            idx = builder_end + 1
            while idx < len(s) and s[idx].isspace():
                idx += 1
            if idx < len(s) and s[idx] == "(":
                payload_end = find_matching_paren(s, idx)
                if payload_end != -1 and not s[payload_end + 1:].strip():
                    payload = s[idx + 1:payload_end].strip()

    if payload is None:
        payload = s

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

    return payload if payload != s else None


# ── Layer Execution ───────────────────────────────────────────────────────────

def deobfuscate_layer(text: str) -> str:
    """Execute a single deobfuscation pass across all decoders."""
    text = decode_bare_base64(text)
    text = decode_powershell_encoded_command(text)
    text = decode_shell_base64_pipe(text)
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


def deobfuscate(
    text: str,
    max_layers: int = DEFAULT_MAX_LAYERS,
    entropy_delta_stop: float = DEFAULT_ENTROPY_DELTA_STOP,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
) -> str:
    """
    Iteratively deobfuscate until stable or entropy change is below threshold.
    Returns the deobfuscated text string.
    """
    cleaned, _, _ = deobfuscate_with_metadata(
        text,
        max_layers=max_layers,
        entropy_delta_stop=entropy_delta_stop,
        entropy_threshold=entropy_threshold,
    )
    return cleaned


def deobfuscate_with_metadata(
    text: str,
    max_layers: int = DEFAULT_MAX_LAYERS,
    entropy_delta_stop: float = DEFAULT_ENTROPY_DELTA_STOP,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
) -> Tuple[str, bool, int]:
    """
    Iteratively deobfuscate until stable or entropy change is below threshold.
    Returns (cleaned_text, was_obfuscated, layers_applied).
    """
    current = text.strip()
    was_obfuscated = is_obfuscated(current, entropy_threshold)
    if not was_obfuscated:
        return current, False, 0

    prev_entropy = calculate_entropy(current)
    layers_applied = 0

    for _ in range(max_layers):
        if not is_obfuscated(current, entropy_threshold):
            break
        new_text = deobfuscate_layer(current)
        if new_text == current:
            break
        current = new_text
        layers_applied += 1
        new_entropy = calculate_entropy(current)
        if abs(prev_entropy - new_entropy) < entropy_delta_stop:
            break
        prev_entropy = new_entropy

    return current, was_obfuscated, layers_applied


# ── Class Wrapper ─────────────────────────────────────────────────────────────

class Deobfuscator:
    """Configurable deobfuscation engine instance."""

    def __init__(
        self,
        max_layers: int = DEFAULT_MAX_LAYERS,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        entropy_delta_stop: float = DEFAULT_ENTROPY_DELTA_STOP,
    ):
        self.max_layers = max_layers
        self.entropy_threshold = entropy_threshold
        self.entropy_delta_stop = entropy_delta_stop

    def calculate_entropy(self, text: str) -> float:
        return calculate_entropy(text)

    def is_obfuscated(self, text: str) -> bool:
        return is_obfuscated(text, self.entropy_threshold)

    def deobfuscate_layer(self, text: str) -> str:
        return deobfuscate_layer(text)

    def deobfuscate(self, text: str) -> str:
        return deobfuscate(
            text,
            max_layers=self.max_layers,
            entropy_delta_stop=self.entropy_delta_stop,
            entropy_threshold=self.entropy_threshold,
        )

    def deobfuscate_with_metadata(self, text: str) -> Tuple[str, bool, int]:
        return deobfuscate_with_metadata(
            text,
            max_layers=self.max_layers,
            entropy_delta_stop=self.entropy_delta_stop,
            entropy_threshold=self.entropy_threshold,
        )

    def decode_bare_base64(self, text: str) -> str:
        return decode_bare_base64(text)
