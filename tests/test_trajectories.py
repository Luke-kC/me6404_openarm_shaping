"""Analytical and synchronization checks for fixed-period trajectories."""

import numpy as np
import pytest

from me6404_openarm_shaping.trajectories import synchronized_trajectory


@pytest.mark.parametrize(
    "velocity_limit, minimum_duration, accel_time, coast_time",
    [(2.0, 2.0, 1.0, 0.0), (0.5, 2.5, 0.5, 1.5)],
    ids=["triangular", "coast"],
)
def test_analytical_profile(velocity_limit, minimum_duration, accel_time, coast_time):
    qi = np.zeros(7)
    qf = np.array([1.0, 0, 0, 0, 0, 0, 0])
    dt = 0.3  # Neither analytical duration is an integer multiple of dt.
    t, q, dq, ddq = synchronized_trajectory(
        qi, qf, np.full(7, velocity_limit), np.ones(7), dt
    )
    expected_duration = np.ceil(minimum_duration / dt) * dt
    stretch = expected_duration / minimum_duration
    assert t[-1] == pytest.approx(expected_duration, abs=1e-14)
    np.testing.assert_allclose(np.diff(t), dt, rtol=0, atol=1e-14)
    assert q.shape == dq.shape == ddq.shape == (len(t), 7)
    np.testing.assert_allclose(q[0], qi, rtol=0, atol=1e-14)
    np.testing.assert_allclose(q[-1], qf, rtol=0, atol=1e-14)
    np.testing.assert_allclose(dq[[0, -1]], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ddq[[0, -1]], 0, rtol=0, atol=1e-14)

    # Unit displacement and acceleration: verify each branch analytically.
    for index, time in enumerate(t[1:-1], start=1):
        tau = time / stretch
        if tau < accel_time:
            expected = (0.5 * tau**2, tau, 1.0)
        elif tau < accel_time + coast_time:
            expected = (
                0.5 * accel_time**2 + velocity_limit * (tau - accel_time),
                velocity_limit,
                0.0,
            )
        else:
            remaining = minimum_duration - tau
            expected = (1 - 0.5 * remaining**2, remaining, -1.0)
        np.testing.assert_allclose(
            [q[index, 0], dq[index, 0], ddq[index, 0]],
            [expected[0], expected[1] / stretch, expected[2] / stretch**2],
            rtol=1e-12, atol=1e-14,
        )


def test_joint_synchronization_and_limits():
    qi = np.array([0.1, -0.3, 0.2, 0.1, -0.1, 0.2, 0.0])
    delta_q = np.array([1.0, -0.5, 0, 0.25, -0.1, 0, 0.3])
    qf = qi + delta_q
    velocity_limits = np.array([0.8, 0.2, 0.1, 0.4, 0.2, 0.1, 0.3])
    acceleration_limits = np.array([1.0, 0.8, 0.1, 0.1, 0.2, 0.1, 0.3])
    t, q, dq, ddq = synchronized_trajectory(
        qi, qf, velocity_limits, acceleration_limits, dt=0.01
    )
    moving = delta_q != 0
    for values in (q - qi, dq, ddq):
        progress = values[:, moving] / delta_q[moving]
        np.testing.assert_allclose(
            progress, np.broadcast_to(progress[:, :1], progress.shape),
            rtol=1e-12, atol=1e-14,
        )
    assert np.all(np.abs(dq) <= velocity_limits + 1e-12)
    assert np.all(np.abs(ddq) <= acceleration_limits + 1e-12)
    assert np.all(np.diff(q[:, 0]) >= -1e-14)
    np.testing.assert_allclose(q[:, ~moving], np.tile(qi[~moving], (len(t), 1)))
    np.testing.assert_allclose(dq[:, ~moving], 0, atol=1e-14)
    np.testing.assert_allclose(ddq[:, ~moving], 0, atol=1e-14)


def test_stationary_trajectory():
    qi = np.arange(7, dtype=np.float64) / 10
    t, q, dq, ddq = synchronized_trajectory(qi, qi, np.ones(7), np.ones(7), 0.01)
    np.testing.assert_allclose(t, [0.0], atol=1e-14)
    np.testing.assert_allclose(q, qi[None, :], atol=1e-14)
    np.testing.assert_allclose(dq, np.zeros((1, 7)), atol=1e-14)
    np.testing.assert_allclose(ddq, np.zeros((1, 7)), atol=1e-14)
