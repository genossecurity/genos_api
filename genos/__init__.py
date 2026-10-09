"""
genos — Core security analysis engine package.

Modular architecture:
- engine.py              Main router: coordinates the full scan pipeline
- deobfuscator.py         Multi-layer deobfuscation (Base64, encoded commands, char constructs)
- gatekeeper.py           Tier 1 triage: Benign vs Suspicious classification
- specialist.py           Tier 2 specialist: 11 MITRE ATT&CK tactic families + behavior modeling
- evidence.py             IOC extraction and analyst evidence summaries
- baseline.py             Stateful baseline store, signature extraction, novelty scoring
- scientific_validation.py  Calibration, provenance hashing, and dataset split auditing
"""

from .engine import GenosEngine
from .gatekeeper import Gatekeeper, GATE_LABELS_BINARY, GATE_LABELS_3CLASS
from .specialist import Specialist, FAMILY_LABELS
from .deobfuscator import Deobfuscator, deobfuscate, is_obfuscated
from .evidence import EvidenceExtractor, collect_indicator_evidence, build_evidence
from .baseline import (
    BaselineStore,
    BaselineMode,
    BaselineStatus,
    ExecutionContext,
    SignatureExtractor,
)

__all__ = [
    "GenosEngine",
    "Gatekeeper",
    "GATE_LABELS_BINARY",
    "GATE_LABELS_3CLASS",
    "Specialist",
    "FAMILY_LABELS",
    "Deobfuscator",
    "deobfuscate",
    "is_obfuscated",
    "EvidenceExtractor",
    "collect_indicator_evidence",
    "build_evidence",
    "BaselineStore",
    "BaselineMode",
    "BaselineStatus",
    "ExecutionContext",
    "SignatureExtractor",
]
