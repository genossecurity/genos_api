"""Resolve the shared Hugging Face backbone for offline local serving."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BACKBONE = "microsoft/codebert-base"
DEFAULT_LOCAL_BACKBONE = BASE_DIR / "models" / "codebert-base"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cached_snapshot() -> Path | None:
    """Find an already downloaded CodeBERT snapshot in the HF cache."""
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo = hf_home / "hub" / "models--microsoft--codebert-base"
    ref = repo / "refs" / "main"
    if ref.is_file():
        snapshot = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snapshot.is_dir():
            return snapshot
    snapshots = sorted((repo / "snapshots").glob("*"))
    return snapshots[-1] if snapshots and snapshots[-1].is_dir() else None


def resolve_backbone(configured: str | None = None) -> tuple[str, bool]:
    """Return (pretrained source, local_files_only).

    A local directory is preferred when present. Set GENOS_CODEBERT_PATH to a
    local directory or HF cache snapshot, and GENOS_HF_LOCAL_ONLY=1 to prevent
    any network fallback.
    """
    requested = configured or os.getenv("GENOS_CODEBERT_PATH")
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if candidate.is_dir():
            return str(candidate), True
        return requested, _truthy(os.getenv("GENOS_HF_LOCAL_ONLY"))

    if DEFAULT_LOCAL_BACKBONE.is_dir():
        return str(DEFAULT_LOCAL_BACKBONE), True

    cached = _cached_snapshot()
    if cached is not None:
        return str(cached), True

    return DEFAULT_BACKBONE, _truthy(os.getenv("GENOS_HF_LOCAL_ONLY"))


def pretrained_kwargs(source: str, local_files_only: bool = False) -> dict:
    """Build consistent Transformers loading options for a resolved source."""
    return {"local_files_only": bool(local_files_only or Path(source).is_dir())}
