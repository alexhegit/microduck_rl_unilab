"""Deploy-contract smoke tests for MicroDuck."""

from microduck_rl_unilab.tasks.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_OBS_SEGMENTS,
)
from microduck_rl_unilab.tasks.microduck.sim2real_notes import (
    POLLEN_RUNTIME_OBS_DIM,
    obs_layout_for_export,
)


def test_pollen_runtime_obs_dim_matches_actor_contract() -> None:
    assert POLLEN_RUNTIME_OBS_DIM == MICRODUCK_ACTOR_OBS_DIM == 61


def test_obs_layout_for_export_matches_segments() -> None:
    layout = obs_layout_for_export()
    assert layout == list(MICRODUCK_OBS_SEGMENTS)
    assert sum(dim for _, dim in layout) == MICRODUCK_ACTOR_OBS_DIM
