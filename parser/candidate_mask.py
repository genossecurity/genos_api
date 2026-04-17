"""
candidate_mask.py — Map rule-engine output to specialist model index space.

Provides:
  - MITRE technique mapping from rule-engine class labels
  - Soft prior vector generation (inference-safe, no hard masking)
  - Candidate mask generation (diagnostics only, NOT used at inference)
  - Logit fusion helpers
  - Coverage statistics
"""

from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Rule-engine class label → primary MITRE technique IDs
# ─────────────────────────────────────────────────────────────────────────────

RULE_CLASS_TO_MITRE: Dict[str, List[str]] = {
    # Persistence
    "persistence:registry_run_key":            ["T1547", "T1060"],
    "persistence:scheduled_task":              ["T1053"],
    "persistence:create_modify_system_process": ["T1543"],
    "persistence:event_triggered_execution":   ["T1546"],
    "persistence:boot_logon_autostart_execution": ["T1547", "T1037"],
    "persistence:hijack_execution_flow":       ["T1574"],
    "persistence:account_manipulation":        ["T1098", "T1136"],

    # Execution
    "execution:command_and_scripting_interpreter": ["T1059"],
    "execution:windows_management_instrumentation": ["T1047"],
    "execution:native_api":                    ["T1106"],

    # Defense evasion
    "defense_evasion:signed_binary_proxy_execution": ["T1218"],
    "defense_evasion:obfuscated_files_or_information": ["T1027"],
    "defense_evasion:deobfuscate_decode_files":       ["T1140"],
    "defense_evasion:modify_registry":         ["T1112"],
    "defense_evasion:impair_defenses":         ["T1562"],
    "defense_evasion:file_permissions_modification": ["T1222"],
    "defense_evasion:indicator_removal":       ["T1070"],
    "defense_evasion:hide_artifacts":          ["T1564"],
    "defense_evasion:masquerading":            ["T1036"],

    # C2 / collection / exfil
    "command_and_control:ingress_tool_transfer": ["T1105"],
    "collection:archive_collected_data":         ["T1560"],
    "collection:data_from_local_system":         ["T1005"],
    "exfiltration:exfiltration_over_c2_channel": ["T1048"],

    # Credential access
    "credential_access:os_credential_dumping":  ["T1003", "T1555"],

    # Privilege escalation
    "privilege_escalation:abuse_elevation_control": ["T1548"],

    # Impact
    "impact:inhibit_system_recovery": ["T1490"],
    "impact:data_destruction":        ["T1485"],

    # Discovery
    "discovery:account_discovery":                       ["T1087"],
    "discovery:query_registry":                         ["T1012"],
    "discovery:system_network_configuration_discovery":  ["T1016"],
    "discovery:remote_system_discovery":                 ["T1018"],
    "discovery:file_and_directory_discovery":             ["T1083"],
    "discovery:system_information_discovery":             ["T1082"],
    "discovery:process_discovery":                       ["T1057"],

    # Lateral movement
    "lateral_movement:remote_services": ["T1021"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Conservative neighbour expansion
# Neighbours must exist in the specialist-map to be useful.
# These are ATT&CK techniques that frequently co-occur or are semantically
# adjacent to the primary mapping.
# ─────────────────────────────────────────────────────────────────────────────

MITRE_NEIGHBORS: Dict[str, List[str]] = {
    # Original neighbours (expanded with cross-cluster links)
    "T1547": ["T1546", "T1037", "T1060", "T1543"],
    "T1060": ["T1547", "T1546"],
    "T1053": ["T1546", "T1082"],
    "T1543": ["T1569", "T1547"],
    "T1059": ["T1202", "T1106", "T1562", "T1546", "T1005"],
    "T1047": ["T1059"],
    "T1218": ["T1216", "T1127", "T1105"],
    "T1027": ["T1140", "T1564"],
    "T1140": ["T1027"],
    "T1105": ["T1071", "T1218"],
    "T1560": ["T1074"],
    "T1048": ["T1567"],
    "T1490": ["T1485", "T1486", "T1489"],
    "T1087": ["T1033", "T1069", "T1078"],
    "T1016": ["T1049", "T1082"],
    "T1018": ["T1046"],
    "T1021": [],
    # Expanded neighbours (v1.1)
    "T1112": ["T1547", "T1562", "T1012"],
    "T1562": ["T1112", "T1070", "T1548", "T1222"],
    "T1222": ["T1548"],
    "T1070": ["T1485", "T1562"],
    "T1564": ["T1027", "T1036", "T1574"],
    "T1036": ["T1574", "T1564"],
    "T1106": ["T1059", "T1055"],
    "T1083": ["T1082", "T1057"],
    "T1082": ["T1083", "T1057", "T1518"],
    "T1057": ["T1082", "T1083"],
    "T1003": ["T1555", "T1552"],
    "T1555": ["T1003", "T1552"],
    "T1552": ["T1003", "T1555"],
    "T1546": ["T1547", "T1053", "T1543"],
    "T1574": ["T1036", "T1547", "T1564"],
    "T1098": ["T1136", "T1078"],
    "T1136": ["T1098"],
    "T1548": ["T1222", "T1134"],
    "T1485": ["T1070", "T1490"],
    "T1005": ["T1074", "T1560"],
    "T1037": ["T1547", "T1546"],
    # Cross-cluster links for under-covered techniques
    "T1055": ["T1106", "T1134"],
    "T1134": ["T1055", "T1548"],
    "T1518": ["T1082", "T1057"],
    "T1078": ["T1098", "T1087"],
    "T1012": ["T1112", "T1082"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Tactic-based expansion — when a rule fires in a tactic, add the most common
# MITRE techniques from that same tactic as candidates.  This ensures that
# even without exact rule coverage, the model can predict related techniques.
# ─────────────────────────────────────────────────────────────────────────────

TACTIC_EXPANSION: Dict[str, List[str]] = {
    "defense_evasion": [
        "T1027", "T1036", "T1070", "T1112", "T1140", "T1218",
        "T1222", "T1553", "T1562", "T1564", "T1574",
        "T1202", "T1216", "T1220", "T1480", "T1497", "T1622", "T1620",
    ],
    "execution": ["T1047", "T1059", "T1106", "T1129", "T1204", "T1559", "T1569"],
    "persistence": [
        "T1037", "T1053", "T1098", "T1136", "T1137", "T1505", "T1543",
        "T1546", "T1547", "T1556", "T1554", "T1078",
    ],
    "discovery": [
        "T1007", "T1012", "T1016", "T1018", "T1033", "T1040", "T1046",
        "T1049", "T1057", "T1069", "T1082", "T1083", "T1087", "T1124",
        "T1135", "T1201", "T1217", "T1482", "T1518", "T1526", "T1580",
        "T1614", "T1619", "T1652",
    ],
    "credential_access": ["T1003", "T1056", "T1110", "T1528", "T1539",
                          "T1550", "T1552", "T1555", "T1558"],
    "privilege_escalation": ["T1055", "T1134", "T1484", "T1548", "T1611"],
    "impact": ["T1485", "T1486", "T1489", "T1490", "T1491", "T1529", "T1531"],
    "collection": ["T1005", "T1074", "T1113", "T1114", "T1115", "T1119",
                    "T1530", "T1560"],
    "command_and_control": ["T1001", "T1071", "T1090", "T1105", "T1132",
                            "T1205", "T1219", "T1573"],
    "lateral_movement": ["T1021", "T1570"],
    "exfiltration": ["T1041", "T1048", "T1567"],
    "initial_access": ["T1060", "T1156", "T1566", "T1680"],
    "resource_development": ["T1127"],
}

MIN_CANDIDATE_SIZE = 3  # expand with neighbours below this threshold

# ─────────────────────────────────────────────────────────────────────────────
# Bucket-aware alpha scaling (configurable)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ALPHA = {
    "strong": 2.0,
    "weak":   1.5,
    "none":   0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Soft prior vector builder (inference-safe — no hard masking)
# ─────────────────────────────────────────────────────────────────────────────

def build_prior_vector(
    rule_result: dict,
    specialist_map: Dict[str, int],
    alpha_overrides: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Build a soft prior vector from rule-engine output.

    This is the PRIMARY inference-time interface.  It produces a prior vector
    that is *added* to model logits — no classes are ever zeroed out.

    Args:
        rule_result:     output of build_rule_result()
        specialist_map:  {mitre_id: int}  (num_classes entries)
        alpha_overrides: optional {bucket: alpha} to override DEFAULT_ALPHA

    Returns dict with:
        prior_vector    list[float]  len=num_classes, scaled by bucket alpha
        raw_prior       list[float]  len=num_classes, unscaled (weight from rules)
        alpha           float        the alpha used for this command
        rule_strength   str          "strong"|"weak"|"none"
        fired_rules     list[str]
    """
    alphas = {**DEFAULT_ALPHA, **(alpha_overrides or {})}
    num_classes = len(specialist_map)
    strength = rule_result.get("rule_strength", "none")
    alpha = alphas.get(strength, 0.0)

    # Build raw prior from rule-engine weights
    raw_prior = [0.0] * num_classes
    for cls, weight in rule_result.get("priors", {}).items():
        for tid in RULE_CLASS_TO_MITRE.get(cls, []):
            if tid in specialist_map:
                idx = specialist_map[tid]
                raw_prior[idx] = max(raw_prior[idx], weight)

    # Scale by bucket alpha
    prior_vector = [v * alpha for v in raw_prior]

    return {
        "prior_vector":  prior_vector,
        "raw_prior":     raw_prior,
        "alpha":         alpha,
        "rule_strength": strength,
        "fired_rules":   rule_result.get("fired_rules", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Candidate mask builder
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_mask(
    rule_result: dict,
    specialist_map: Dict[str, int],
    true_label: Optional[str] = None,
    expand_neighbors: bool = True,
) -> dict:
    """
    Build candidate mask and prior vector from rule-engine output.
    Uses rule_strength bucket to control expansion behavior:
      - strong: tight candidates, 1-hop neighbors, same-tactic expansion only
      - weak:   broader candidates, 2-hop neighbors, adjacent-tactic expansion
      - none:   full label space fallback for ML

    Returns dict with:
        candidate_mitre_ids, candidate_mask, prior_vector, num_candidates,
        true_label_in_candidates, used_fallback, rule_strength
    """
    num_classes = len(specialist_map)
    candidate_ids: set = set()
    banned_ids: set = set()
    strength = rule_result.get("rule_strength", "none")

    # Map rule-engine labels → MITRE IDs
    for cls in rule_result.get("candidate_classes", []):
        for tid in RULE_CLASS_TO_MITRE.get(cls, []):
            if tid in specialist_map:
                candidate_ids.add(tid)

    for cls in rule_result.get("banned_classes", []):
        for tid in RULE_CLASS_TO_MITRE.get(cls, []):
            if tid in specialist_map:
                banned_ids.add(tid)

    candidate_ids -= banned_ids

    # ── STRONG bucket: tight mask ─────────────────────────────────────────
    if strength == "strong" and expand_neighbors and len(candidate_ids) > 0:
        expanded = set(candidate_ids)
        # 1-hop only for strong rules
        for tid in candidate_ids:
            for nbr in MITRE_NEIGHBORS.get(tid, []):
                if nbr in specialist_map and nbr not in banned_ids:
                    expanded.add(nbr)
        candidate_ids = expanded

        # Same-tactic expansion only (no cross-tactic bleed)
        fired_tactics: set = set()
        for cls in rule_result.get("candidate_classes", []):
            fired_tactics.add(cls.split(":")[0])
        for tactic in fired_tactics:
            for tid in TACTIC_EXPANSION.get(tactic, []):
                if tid in specialist_map and tid not in banned_ids:
                    candidate_ids.add(tid)

    # ── WEAK bucket: broader mask ─────────────────────────────────────────
    elif strength == "weak" and expand_neighbors and len(candidate_ids) > 0:
        expanded = set(candidate_ids)
        # 2-hop neighbors
        hop1 = set()
        for tid in candidate_ids:
            for nbr in MITRE_NEIGHBORS.get(tid, []):
                if nbr in specialist_map and nbr not in banned_ids:
                    hop1.add(nbr)
        expanded |= hop1
        for tid in hop1:
            for nbr in MITRE_NEIGHBORS.get(tid, []):
                if nbr in specialist_map and nbr not in banned_ids:
                    expanded.add(nbr)
        candidate_ids = expanded

        # Same-tactic + adjacent tactics
        fired_tactics: set = set()
        for cls in rule_result.get("candidate_classes", []):
            fired_tactics.add(cls.split(":")[0])
        # Add adjacent tactics for weak signals
        _ADJACENT = {
            "execution": ["defense_evasion", "persistence"],
            "defense_evasion": ["execution", "persistence", "privilege_escalation"],
            "persistence": ["execution", "privilege_escalation", "defense_evasion"],
            "discovery": ["credential_access", "collection"],
            "credential_access": ["discovery", "privilege_escalation"],
            "privilege_escalation": ["defense_evasion", "persistence"],
            "collection": ["exfiltration", "discovery"],
            "lateral_movement": ["discovery", "credential_access"],
            "command_and_control": ["exfiltration"],
            "impact": ["defense_evasion"],
        }
        base_tactics = set(fired_tactics)
        for tactic in base_tactics:
            for adj in _ADJACENT.get(tactic, []):
                fired_tactics.add(adj)
        for tactic in fired_tactics:
            for tid in TACTIC_EXPANSION.get(tactic, []):
                if tid in specialist_map and tid not in banned_ids:
                    candidate_ids.add(tid)

    # ── NONE bucket: full label space ─────────────────────────────────────
    # (also covers edge case where strong/weak fired but produced no MITRE IDs)

    # Prior vector
    prior_vector = [0.0] * num_classes
    for cls, weight in rule_result.get("priors", {}).items():
        for tid in RULE_CLASS_TO_MITRE.get(cls, []):
            if tid in specialist_map:
                idx = specialist_map[tid]
                prior_vector[idx] = max(prior_vector[idx], weight)

    used_fallback = (strength == "none") or len(candidate_ids) == 0

    if used_fallback:
        # Full label space — let ML decide
        mask = [1] * num_classes
        candidate_ids = set(specialist_map.keys())
    else:
        mask = [0] * num_classes
        for tid in candidate_ids:
            mask[specialist_map[tid]] = 1

    # Check organic coverage BEFORE forcing true label
    true_in_candidates: Optional[bool] = None
    if true_label is not None and true_label in specialist_map:
        true_idx = specialist_map[true_label]
        true_in_candidates = mask[true_idx] == 1
        mask[true_idx] = 1  # force

    return {
        "candidate_mitre_ids":       sorted(candidate_ids),
        "candidate_mask":            mask,
        "prior_vector":              prior_vector,
        "num_candidates":            sum(mask),
        "true_label_in_candidates":  true_in_candidates,
        "used_fallback":             used_fallback,
        "rule_strength":             strength,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logit-level helpers (work with raw Python lists or torch tensors)
# ─────────────────────────────────────────────────────────────────────────────

def apply_candidate_mask(logits, mask, mask_value: float = -1e9):
    """
    Zero-out non-candidate logits.

    logits: (batch, C) or (C,)  — torch.Tensor
    mask:   same shape or (C,)  — torch.Tensor or list
    """
    import torch

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=logits.dtype, device=logits.device)
    if mask.dim() == 1 and logits.dim() == 2:
        mask = mask.unsqueeze(0)
    return logits * mask + (1.0 - mask) * mask_value


def fuse_logits_with_priors(logits, prior_vector, alpha: float = 1.0):
    """
    Soft additive fusion:  fused = logits + α × prior_vector.

    When using build_prior_vector(), alpha is already baked in, so pass alpha=1.0.
    When using raw priors, pass a custom alpha.

    No classes are eliminated — all logits remain accessible to the model.

    logits:       (batch, C) or (C,)
    prior_vector: (C,) or (batch, C)  — torch.Tensor or list
    alpha:        scalar weight (default 1.0 since build_prior_vector pre-scales)
    """
    import torch

    if not isinstance(prior_vector, torch.Tensor):
        prior_vector = torch.tensor(prior_vector, dtype=logits.dtype,
                                    device=logits.device)
    if prior_vector.dim() == 1 and logits.dim() == 2:
        prior_vector = prior_vector.unsqueeze(0)
    return logits + alpha * prior_vector


# ─────────────────────────────────────────────────────────────────────────────
# Coverage statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage_stats(mask_results: list) -> dict:
    """
    Aggregate stats from a list of build_candidate_mask() results.

    Reports organic coverage (before forcing the true label) so we can
    judge whether the rule engine is too aggressive.
    """
    total = len(mask_results)
    if total == 0:
        return {"total": 0}

    organic = sum(
        1 for r in mask_results
        if r["true_label_in_candidates"] is True  # includes fallback cases
    )
    fallback = sum(1 for r in mask_results if r["used_fallback"])
    sizes = [r["num_candidates"] for r in mask_results]
    sorted_sizes = sorted(sizes)

    return {
        "total":                       total,
        "organic_true_label_in_candidates": organic,
        "organic_coverage_pct":        round(organic / total * 100, 2),
        "fallback_count":              fallback,
        "fallback_pct":                round(fallback / total * 100, 2),
        "avg_candidate_size":          round(sum(sizes) / total, 2),
        "min_candidate_size":          sorted_sizes[0],
        "max_candidate_size":          sorted_sizes[-1],
        "median_candidate_size":       sorted_sizes[total // 2],
    }
