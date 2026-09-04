"""Cold-path materialization of MicroDuck robot binary assets.

Robot meshes are hosted on the Hugging Face dataset
``unilabsim/unilab-robots`` (the same dataset UniLab itself uses) and are
downloaded into this repository's ``assets/`` tree on first use so that XML
relative references (``assets/…``) resolve unchanged next to the scene XML.
This module must only run on init/materialization paths, never inside env
``step``/``reset``.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

_HF_ROBOTS_REPO_ID = "unilabsim/unilab-robots"

# <repo root>/assets — snapshots land at assets/robots/microduck/assets/ so
# the committed XMLs at assets/robots/microduck/*.xml resolve their relative
# mesh references without any path rewriting.
ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"

_MICRODUCK_PATTERNS = ("robots/microduck/assets/*",)
_MICRODUCK_MARKERS = (
    ASSETS_ROOT / "robots/microduck/assets/trunk_base.stl",
)


def ensure_microduck_assets() -> Path:
    """Download MicroDuck meshes next to the scene XML if incomplete."""
    if not all(marker.is_file() for marker in _MICRODUCK_MARKERS):
        snapshot_download(
            repo_id=_HF_ROBOTS_REPO_ID,
            repo_type="dataset",
            allow_patterns=list(_MICRODUCK_PATTERNS),
            local_dir=str(ASSETS_ROOT),
        )
    missing = [str(marker) for marker in _MICRODUCK_MARKERS if not marker.is_file()]
    if missing:
        raise FileNotFoundError(f"MicroDuck asset download incomplete; missing: {missing}")
    return ASSETS_ROOT / "robots/microduck"
