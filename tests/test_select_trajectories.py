import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments import select_trajectories
from experiments.select_trajectories import (
    UNTRACKED_BAND,
    Candidate,
    entry,
    evaluate,
    screen_band,
    select_band,
    track_modes,
)
from me6404_openarm_shaping.configuration import load_robot_config
from me6404_openarm_shaping.robot_model import LEFT_ARM_JOINT_NAMES, load_model


def test_mode_tracking_preserves_identity_at_crossing_and_loss_of_higher_mode(
    monkeypatch,
):
    def state_matrix(_model, q, _kp, _kd, *, workspace=None):
        wn = np.array([10 + 2 * q[0], 12 - 2 * q[0], 20, 30, 40, 50, 60])
        damping = np.ones(7)
        if q[0] == 0.5:
            damping[2] = 50  # The 20 rad/s mode becomes overdamped.
        return np.block(
            [[np.zeros((7, 7)), np.eye(7)], [-np.diag(wn**2), -np.diag(damping)]]
        )

    monkeypatch.setattr("experiments.select_trajectories.state_matrix", state_matrix)
    q = np.zeros((3, 7))
    q[:, 0] = [0, 0.5, 1]
    result = track_modes(load_model(), q, load_robot_config())
    assert not isinstance(result, str)
    assert result.tracked_poles is not None
    assert result.tracked_poles.shape == (3, 2)
    assert [len(poles) for poles in result.sampled_poles] == [7, 6, 7]
    np.testing.assert_allclose(
        np.abs(result.tracked_poles[:, :2]), [[10, 12], [11, 11], [12, 10]]
    )


def test_screen_requires_two_fixed_zvd_components_only_for_medium_and_wide():
    assert screen_band(0.0, 0) == "narrow"
    assert screen_band(np.nextafter(0.40, 0.0), 0) == "narrow"
    assert screen_band(np.nextafter(0.75, 0.0), 2) == "medium"
    for beta, group in ((0.40, "medium"), (0.75, "wide"), (1.51, "wide")):
        assert screen_band(beta, 1) == "too_few_fixed_zvd_components"
        assert screen_band(beta, 2) == group
    assert screen_band(None, 3) == "tracking_failed_too_few_fixed_zvd_components"
    assert screen_band(None, 4) == UNTRACKED_BAND


def test_unstable_sample_after_tracking_failure_still_rejects(monkeypatch):
    matrices = iter(
        np.diag(diagonal)
        for diagonal in (
            [-1 + 2j, -1 - 2j, -1],
            [-1, -2, -3],  # Loss of the oscillatory pair.
            [-1, -2, 1],  # Later unstable pole must not be ignored.
        )
    )
    monkeypatch.setattr(
        "experiments.select_trajectories.state_matrix", lambda *_, **__: next(matrices)
    )
    result = track_modes(load_model(), np.zeros((3, 7)), load_robot_config())
    assert result == "unstable_or_marginal_modes"


def test_configured_path_tracks_low_modes_and_still_has_four_fixed_zvd_components():
    config = load_robot_config()
    model = load_model()
    limits = np.array([model.joint(name).range for name in LEFT_ARM_JOINT_NAMES])
    result = evaluate(model, config, limits, 0, config.waypoints, 0.10, 0.30)
    assert not isinstance(result, str)
    assert result.tracking_failure is None
    count = len(result.fixed_zvd_frequencies)
    assert count == 4
    group = screen_band(result.beta, count)
    assert group in ("narrow", "medium", "wide")
    benchmark = entry(result, group, 1, 0.10)
    assert result.frequencies is not None
    assert result.frequencies.shape == (result.modal_samples, 2)
    assert benchmark["beta"] == pytest.approx(max(result.spans))
    assert benchmark["fixed_zvd_component_count"] == 4


def test_band_selection_covers_beta_range_without_motion_matching():
    candidates = [
        Candidate(
            number=number,
            waypoints=np.zeros((2, 7)),
            duration=float(number),
            travel=float(number),
            clearance=np.zeros(7),
            events=[],
            frequencies=np.array([[1.0], [1.0 + beta]]),
            decay_times=np.ones((2, 1)),
            fixed_zvd_frequencies=np.array([1.0, 2.0]),
            minimum_mac=1.0,
            minimum_assignment_gap=1.0,
            modal_samples=2,
            tracking_failure=None,
        )
        for number, beta in enumerate(np.linspace(1.75, 0.75, 19), start=1)
    ]
    assert [path.number for path in select_band(candidates)] == list(range(19, 0, -2))
    assert [path.number for path in select_band(candidates[:2])] == [2, 1]
    assert select_band([]) == []


@pytest.mark.parametrize(
    ("component_counts", "expected_order"),
    [([], []), ([4, 5], [1, 0]), ([4] * 11 + [5], [11, *range(11)])],
)
def test_suite_archives_all_untracked_but_runs_only_top_ten(
    monkeypatch, tmp_path, component_counts, expected_order
):
    output = tmp_path / "suite.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_trajectories", "--candidates", "11", "--output", str(output)],
    )

    def evaluate_candidate(_model, _config, _limits, number, waypoints, *_):
        if number >= len(component_counts):
            return "joint_clearance"
        return Candidate(
            number=number,
            waypoints=waypoints,
            duration=1.0,
            travel=1.0,
            clearance=np.zeros(7),
            events=[],
            frequencies=None,
            decay_times=None,
            fixed_zvd_frequencies=np.arange(component_counts[number], dtype=float),
            minimum_mac=None,
            minimum_assignment_gap=None,
            modal_samples=2,
            tracking_failure="oscillatory_mode_count_changed",
        )

    monkeypatch.setattr(select_trajectories, "evaluate", evaluate_candidate)
    select_trajectories.main()

    suite = yaml.safe_load(output.read_text(encoding="utf-8"))
    archive = suite["ranked_untracked_benchmarks"]
    assert [item["candidate_number"] for item in archive] == expected_order
    assert suite["trajectories"] == archive[:10]
    assert suite["untracked_benchmark_count"] == len(expected_order)
    assert suite["runnable_untracked_benchmark_count"] == min(10, len(expected_order))


def test_pilot_paths_reproduce_fixed_zvd_component_counts():
    config = load_robot_config()
    model = load_model()
    model.opt.timestep = config.simulation.dt_s
    limits = np.array([model.joint(name).range for name in LEFT_ARM_JOINT_NAMES])
    cases = [
        (
            [
                -0.3578624,
                -1.1899304,
                0.6584042,
                1.6738175,
                -0.3899753,
                -0.0296799,
                -0.1138626,
            ],
            2,
            "medium",
        ),
        (
            [
                -0.2178409,
                -1.3919792,
                -0.8797611,
                0.6101853,
                0.3198062,
                0.5571291,
                1.0553377,
            ],
            4,
            "wide",
        ),
    ]
    for number, (target, count, group) in enumerate(cases, start=1):
        waypoints = np.array([[0.0] * 7, target])
        result = evaluate(model, config, limits, number, waypoints, 0.10, 0.30)
        assert not isinstance(result, str)
        assert len(result.fixed_zvd_frequencies) == count
        assert screen_band(result.beta, count) == group


def test_saved_suite_has_valid_trajectory_records():
    path = Path(__file__).resolve().parents[1] / "config" / "trajectory_suite.yaml"
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    trajectories = suite["trajectories"]
    assert len({item["id"] for item in trajectories}) == len(trajectories)
    for item in trajectories:
        assert item["waypoints"][0] == [0.0] * 7
        assert item["band"] in suite["bands"]
        assert item["contact_screen"]["disallowed_penetrating_pairs"] == 0
        if item["band"] == UNTRACKED_BAND:
            assert item["beta"] is None
            assert item["fixed_zvd_component_count"] >= 4
        else:
            assert item["beta"] == max(mode["beta"] for mode in item["modes"])
