import numpy as np
import pytest

from experiments.calibrate_joint_response import (
    add_joint_armature,
    first_target_crossing,
    overshoot_index,
    ring_down_fit,
)
from me6404_openarm_shaping.robot_model import load_model, mass_matrix


def test_armature_changes_only_target_inertia_and_not_damping() -> None:
    model = load_model()
    q = np.zeros(7)
    original = mass_matrix(model, q)
    damping = model.dof_damping.copy()
    add_joint_armature(model, 1, 0.08)
    expected = np.zeros((7, 7))
    expected[1, 1] = 0.08
    np.testing.assert_allclose(mass_matrix(model, q) - original, expected, atol=1e-14)
    np.testing.assert_array_equal(model.dof_damping, damping)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_first_target_crossing_interpolates_in_move_direction(direction: float) -> None:
    t = np.array([0.0, 0.2, 0.5, 0.6, 1.0])
    target = -1.0
    q = target + direction * np.array([-1.0, -0.25, 0.5, -1.0, 0.0])
    assert first_target_crossing(t, q, target, direction) == pytest.approx(0.3)
    assert first_target_crossing(t[3:], q[3:], target, direction) == 1.0
    assert first_target_crossing(t[:2], q[:2], target, direction) is None


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_peak_stays_in_first_lobe_after_crossing(direction: float) -> None:
    t = np.arange(8.0)
    target = -1.0
    q = target + direction * np.array([-0.3, 0.1, 0.2, 0.1, -0.1, 0.4, 0.5, -0.2])
    crossing = first_target_crossing(t, q, target, direction)
    assert crossing is not None
    assert overshoot_index(t, q, target, direction, crossing) == 2
    assert overshoot_index(t, q, target, direction, crossing, number=2) == 6
    assert overshoot_index(t[:4], q[:4], target, direction, crossing) is None
    assert (
        overshoot_index(t[:-1], q[:-1], target, direction, crossing, number=2) is None
    )


def test_ring_down_fit_recovers_damped_frequency_and_decay() -> None:
    time = np.linspace(0, 1.5, 301)
    signal = -1 + 0.2 * np.exp(-0.75 * time) * np.cos(2 * np.pi * 1.8 * time + 0.2)
    frequency, decay, fit = ring_down_fit(time + 12, signal)
    assert frequency == pytest.approx(1.8)
    assert decay == pytest.approx(0.75)
    assert fit == pytest.approx(1)
