"""Checks for experimental TVIST screening and response ranking."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from me6404_openarm_shaping import tvist_selection
from me6404_openarm_shaping.configuration import (
    DEFAULT_CONFIG_PATH,
    parse_robot_config,
    robot_config_to_dict,
)
from me6404_openarm_shaping.mode_tracking import TrackedModes
from me6404_openarm_shaping.robot_model import load_model
from me6404_openarm_shaping.robot_sim import simulate
from me6404_openarm_shaping.tvist import TvistShaper
from me6404_openarm_shaping.tvist_selection import SubsetScore, select_subset_indices


@pytest.mark.parametrize(("velocity", "acceleration", "accepted"), [
    (4.5, 9.0, True), (5.1, 9.0, False), (4.5, 10.1, False),
])
def test_candidate_screen_uses_absolute_limits(
    monkeypatch: pytest.MonkeyPatch, velocity: float, acceleration: float, accepted: bool
) -> None:
    config = parse_robot_config(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))
    t = np.arange(5) * config.simulation.dt_s
    q = np.repeat((velocity * t + 0.5 * acceleration * t**2)[:, None], 7, axis=1)
    tracked = TrackedModes(np.full((5, 1), -1 + 3j), np.ones((5, 1)), np.ones(5), np.ones(5, dtype=bool))
    monkeypatch.setattr(tvist_selection, "apply_tvist_to_joint_trajectory", lambda *_: q)
    (candidate,) = tvist_selection.build_subset_candidates(
        q, config, tracked, TvistShaper.ZVD, config.input_shaping.time_varying.zvd
    )
    assert (candidate.trajectory is not None) is accepted
    assert candidate.trajectory_warning_samples == len(t)


def test_strict_benchmarks_and_independent_objectives() -> None:
    scores = [
        SubsetScore((1,), 0.8, 0.002, 2.0),  # Equal RMS cannot beat it.
        SubsetScore((2,), 0.8, 0.001, 3.0),
        SubsetScore((3,), 0.9, 0.0015, 2.5),
        SubsetScore((4,), 0.8, 0.003, 1.0),
        SubsetScore((1, 2), 0.9, 0.0005, None),
        SubsetScore((1, 3), 0.8, 0.0016, 2.5),  # Wins settling tie on command duration.
    ]
    assert select_subset_indices(scores, 0.002, 2.0) == {
        "min_residual": 4, "min_settling": 3, "beat_fixed_rms": 5,
        "beat_fixed_settling": 3,
    }
    assert "beat_fixed_rms" not in select_subset_indices(scores, 0.0005, 1.0)
    assert "beat_fixed_settling" not in select_subset_indices(scores, 0.0005, 1.0)


def test_experimental_config_is_opt_in_and_boolean() -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    varying = raw["input_shaping"]["time_varying"]
    varying.pop("experimental")
    assert not parse_robot_config(raw).input_shaping.time_varying.experimental
    varying["experimental"] = True
    config = parse_robot_config(raw)
    assert parse_robot_config(robot_config_to_dict(config)).input_shaping.time_varying.experimental
    varying["experimental"] = "true"
    with pytest.raises(TypeError, match=r"time_varying\.experimental"):
        parse_robot_config(raw)


def test_wide07_experimental_zvd_screens_and_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    # This regression reproduces the original wide-07 acceleration violation.
    raw["passive_damping"] = [0.4] * 7
    raw["added_armature"] = [0.0] * 7
    suite_path = Path(DEFAULT_CONFIG_PATH).with_name("trajectory_suite_focused.yaml")
    suite = yaml.safe_load(suite_path.read_text())
    raw["waypoints"] = next(row["waypoints"] for row in suite["trajectories"] if row["id"] == "wide-07")
    raw["input_shaping"]["time_varying"]["zv"]["enabled"] = False
    raw["input_shaping"]["time_varying"]["zvd"]["enabled"] = True
    raw["input_shaping"]["time_varying"]["ei"]["enabled"] = False
    config = parse_robot_config(raw)
    legacy = simulate(load_model(), config, headless=True)
    assert not legacy.tvist_selection
    assert np.max(np.abs(legacy.trajectories["tvist_zvd"]["ddq"])) > 10

    config.input_shaping.time_varying.experimental = True
    original_run_trial = tvist_selection.run_trial
    simulated = []

    def checked_run_trial(model, command, *args, **kwargs):
        assert np.all(np.abs(command["ddq"]) <= config.acceleration_limits)
        assert kwargs["headless"]
        simulated.append(command)
        return original_run_trial(model, command, *args, **kwargs)

    monkeypatch.setattr(tvist_selection, "run_trial", checked_run_trial)
    result = simulate(load_model(), config, headless=True)
    np.testing.assert_allclose(result.recordings["baseline"]["q"], legacy.recordings["baseline"]["q"])
    audit = cast(dict, result.tvist_selection["tvist_zvd"])
    rows = audit["candidates"]
    assert len(rows) == 15
    survivors = [row for row in rows if row["simulated"]]
    assert 0 < len(simulated) == len(survivors) < len(rows)
    for objective, metric in (("min_residual", "residual_rms_rad"), ("min_settling", "total_settling_s")):
        best = min(
            (row for row in survivors if row[metric] is not None),
            key=lambda row: (row[metric], row["command_duration_s"], row["branches"]),
        )
        assert audit["selected_branches"][objective] == best["branches"]
        label = f"tvist_zvd_{objective}"
        assert label in result.trajectories and label in result.recordings
        assert len(result.tvist_schedules[label]) == len(result.trajectories[label]["t"])
