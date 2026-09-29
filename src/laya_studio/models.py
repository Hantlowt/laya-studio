from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelVariant:
    key: str
    family: str
    backend: str
    model_id: str
    title: str
    summary: str
    language: str
    context_tokens: int
    upstream_model: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


MODEL_VARIANTS = (
    ModelVariant(
        "english-mlx",
        "english",
        "mlx",
        "aac6fef/laya-mlx",
        "English · MLX",
        "ModernBERT-large, English, 512-token context; fastest native Apple Silicon option.",
        "English",
        512,
        "convaiinnovations/laya",
    ),
    ModelVariant(
        "multilingual-mlx",
        "multilingual",
        "mlx",
        "aac6fef/laya-multilingual-mlx",
        "Multilingual · MLX",
        "mmBERT-base, multilingual, 1024-token context; intended for non-English inputs.",
        "Multilingual",
        1024,
        "convaiinnovations/laya-multilingual",
    ),
    ModelVariant(
        "typed-decisions-mlx",
        "typed-decisions",
        "mlx",
        "aac6fef/laya-typed-decisions-mlx",
        "Typed Decisions · MLX",
        "ModernBERT-large, English, 1024-token context; specialized upstream decision checkpoint.",
        "English",
        1024,
        "convaiinnovations/laya-typed-decisions",
    ),
    ModelVariant(
        "english-pytorch",
        "english",
        "pytorch",
        "convaiinnovations/laya",
        "English · PyTorch",
        "Upstream ModernBERT-large checkpoint; reference PyTorch implementation.",
        "English",
        512,
    ),
    ModelVariant(
        "multilingual-pytorch",
        "multilingual",
        "pytorch",
        "convaiinnovations/laya-multilingual",
        "Multilingual · PyTorch",
        "Upstream mmBERT-base checkpoint with a 1024-token context.",
        "Multilingual",
        1024,
    ),
    ModelVariant(
        "typed-decisions-pytorch",
        "typed-decisions",
        "pytorch",
        "convaiinnovations/laya-typed-decisions",
        "Typed Decisions · PyTorch",
        "Upstream typed-decisions checkpoint with a 1024-token context.",
        "English",
        1024,
    ),
)


def model_catalog() -> list[dict[str, Any]]:
    return [variant.public() for variant in MODEL_VARIANTS]


def resolve_revision(model_id: str, requested: str | None = None) -> str | None:
    """Resolve a Hub branch/tag to an immutable commit without requiring hub support."""
    if (
        requested
        and len(requested) == 40
        and all(c in "0123456789abcdef" for c in requested.lower())
    ):
        return requested
    if Path(model_id).exists():
        return requested
    try:
        from huggingface_hub import model_info

        return str(model_info(model_id, revision=requested).sha)
    except Exception:
        # Custom local/runtime identifiers remain usable, but are visibly unpinned.
        return requested


def ensure_checkpoint(model_id: str, revision: str | None = None) -> str:
    """Download a missing Hub checkpoint and return its local snapshot path."""
    local = Path(model_id)
    if local.exists():
        return str(local.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface-hub to download checkpoints") from exc
    return snapshot_download(model_id, revision=revision)
