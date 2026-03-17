# boost.py
import torch

def apply_boosts(command: str, probs: torch.Tensor, s_map: dict):
    """
    Adjusts MITRE probabilities using domain-specific heuristics.
    probs: torch tensor (num_classes)
    s_map: index -> mitre_id
    """
    boosted = probs.clone()
    reverse_map = {v: k for k, v in s_map.items()}

    def boost(mid, amount):
        if mid in reverse_map:
            boosted[reverse_map[mid]] += amount

    def penalize(mid, amount):
        if mid in reverse_map:
            boosted[reverse_map[mid]] -= amount

    cmd = command.lower()

    # --- Boost rules ---
    if "-enc" in cmd or "-encodedcommand" in cmd:
        boost("T1027", 0.25)
        boost("T1059", 0.20)

    if ("curl" in cmd or "wget" in cmd) and ("sh" in cmd or "bash" in cmd):
        boost("T1105", 0.30)
        boost("T1059", 0.20)

    if "sekurlsa" in cmd or "mimikatz" in cmd:
        boost("T1003", 0.50)
        penalize("T1555", 0.30)

    if "rundll32" in cmd or "mshta" in cmd or "certutil" in cmd:
        boost("T1218", 0.30)

    if "disable" in cmd and ("defender" in cmd or "mpreference" in cmd):
        boost("T1562", 0.30)

    # Clamp negatives
    boosted = torch.clamp(boosted, min=0)
    # Renormalize
    boosted = boosted / boosted.sum()

    return boosted