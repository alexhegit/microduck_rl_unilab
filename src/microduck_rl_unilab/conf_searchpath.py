"""Hydra config search path registration for the external conf trees.

Production training entrypoints append ``conf/<algo>`` via Hydra
``--config-dir`` (see :mod:`microduck_rl_unilab.cli`). Programmatic Hydra
composition (tests, alignment contract evaluation, alignment scripts) has no
CLI, so this module registers a ``SearchPathPlugin`` that appends the same
``conf/<algo>`` directories to every Hydra search path. UniLab's own conf
tree remains the primary config dir; only ``task=microduck_*`` owner YAMLs
resolve from this package.
"""

from __future__ import annotations

from pathlib import Path

CONF_ROOT = Path(__file__).resolve().parent / "conf"

_registered = False


def register_conf_search_path() -> None:
    """Append ``conf/<algo>`` dirs to Hydra's config search path (idempotent)."""
    global _registered
    if _registered:
        return

    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    algo_dirs = tuple(
        f"file://{algo_dir}" for algo_dir in sorted(CONF_ROOT.iterdir()) if algo_dir.is_dir()
    )

    class MicroduckConfSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path):  # noqa: ANN001, ANN202
            for path in algo_dirs:
                search_path.append("microduck_rl_unilab", path)

    Plugins.instance().register(MicroduckConfSearchPathPlugin)
    _registered = True
