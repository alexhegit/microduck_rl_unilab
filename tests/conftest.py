"""Shared test setup for the external MicroDuck task package.

1. ``UNILAB_EXTRA_REGISTRY_PACKAGES`` must be set before any UniLab registry
   import so ``ensure_registries()`` (including collector subprocesses) picks
   up ``microduck_rl_unilab.tasks``.
2. The external ``conf/<algo>`` trees are appended to Hydra's config search
   path so tests can compose ``config`` from the ``unilab`` wheel conf while
   ``task=microduck_*`` owner YAMLs resolve from this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("UNILAB_EXTRA_REGISTRY_PACKAGES", "microduck_rl_unilab.tasks")

from microduck_rl_unilab.conf_searchpath import register_conf_search_path  # noqa: E402

register_conf_search_path()

REPO_ROOT = Path(__file__).resolve().parents[1]
