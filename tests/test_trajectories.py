"""Analytical and synchronization checks for fixed-period trajectories."""

import numpy as np
import pytest

from me6404_openarm_shaping.input_shaping import zvd_shaper
from me6404_openarm_shaping.trajectories import (
    apply_shaper_to_joint_trajectory,
    apply_shaper_to_progress,
    bang_coast_bang_progress,
    joint_trajectory_from_progress,
    waypoint_trajectory,
)


@pytest.mark.parametrize(
    "velocity_limit, minimum_duration, accel_time, coast_time",
    [(2.0, 2.0, 1.0, 0.0), (0.5, 2.5, 0.5, 1.5)],
    ids=["triangular", "coast"],
)
def test_analytical_profile(
    velocity_limit: float, minimum_duration: float, accel_time: float, coast_time: float
) -> None:
    qi = np.zeros(7)
    qf = np.array([1.0, 0, 0, 0, 0, 0, 0])
    dt = 0.3  # Neither analytical duration is an integer multiple of dt.
    t, s, ds, dds = bang_coast_bang_progress(
        qi, qf, np.full(7, velocity_limit), np.ones(7), dt
    )
    assert s.shape == ds.shape == dds.shape == t.shape
    np.testing.assert_allclose(s[[0, -1]], [0, 1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(ds[[0, -1]], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(dds[[0, -1]], 0, rtol=0, atol=1e-14)
    q, dq, ddq = joint_trajectory_from_progress(qi, qf, s, ds, dds)
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


def test_joint_synchronization_and_limits() -> None:
    qi = np.array([0.1, -0.3, 0.2, 0.1, -0.1, 0.2, 0.0])
    delta_q = np.array([1.0, -0.5, 0, 0.25, -0.1, 0, 0.3])
    qf = qi + delta_q
    velocity_limits = np.array([0.8, 0.2, 0.1, 0.4, 0.2, 0.1, 0.3])
    acceleration_limits = np.array([1.0, 0.8, 0.1, 0.1, 0.2, 0.1, 0.3])
    t, s, ds, dds = bang_coast_bang_progress(
        qi, qf, velocity_limits, acceleration_limits, dt=0.01
    )
    q, dq, ddq = joint_trajectory_from_progress(qi, qf, s, ds, dds)
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


def test_stationary_trajectory() -> None:
    qi = np.arange(7, dtype=np.float64) / 10
    t, s, ds, dds = bang_coast_bang_progress(
        qi, qi, np.ones(7), np.ones(7), 0.01
    )
    np.testing.assert_allclose(s, [1.0], rtol=0, atol=1e-14)
    q, dq, ddq = joint_trajectory_from_progress(qi, qi, s, ds, dds)
    np.testing.assert_allclose(t, [0.0], atol=1e-14)
    np.testing.assert_allclose(q, qi[None, :], atol=1e-14)
    np.testing.assert_allclose(dq, np.zeros((1, 7)), atol=1e-14)
    np.testing.assert_allclose(ddq, np.zeros((1, 7)), atol=1e-14)


def test_mapping_supplied_progress() -> None:
    qi = np.array([1, -1, 0.5, 0, 0, 0, 0], dtype=np.float64)
    qf = np.array([3, -2, 0.5, 0, 0, 0, 0], dtype=np.float64)
    # Interior progress endpoints must not be replaced with qi/qf.
    s = np.array([0.25, 0.75])
    ds = np.array([0.5, 0.25])
    dds = np.array([0.1, -0.2])
    q, dq, ddq = joint_trajectory_from_progress(qi, qf, s, ds, dds)
    np.testing.assert_allclose(
        q, [[1.5, -1.25, 0.5, 0, 0, 0, 0], [2.5, -1.75, 0.5, 0, 0, 0, 0]],
        rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(
        dq, [[1, -0.5, 0, 0, 0, 0, 0], [0.5, -0.25, 0, 0, 0, 0, 0]],
        rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(
        ddq, [[0.2, -0.1, 0, 0, 0, 0, 0], [-0.4, 0.2, 0, 0, 0, 0, 0]],
        rtol=0, atol=1e-14,
    )


def test_identity_progress_shaper() -> None:
    qi = np.zeros(7)
    qf = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    baseline = bang_coast_bang_progress(qi, qf, np.ones(7), np.ones(7), dt=0.1)
    _, s, ds, dds = baseline
    shaped = apply_shaper_to_progress(s, ds, dds, [1.0], [0.0], dt=0.1)
    for actual, expected in zip(shaped, baseline, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    "delays", [[0.0, 0.1, 0.1], [0.0, 0.06, 0.14]],
    ids=["duplicate-delays", "round-to-duplicate-samples"],
)
def test_delayed_progress_copies(delays: list[float]) -> None:
    # Toy samples isolate the delayed-copy arithmetic for all three channels.
    t, s, ds, dds = apply_shaper_to_progress(
        [0, 0.5, 1], [0, 2, 0], [0, -4, 0],
        [0.5, 0.25, 0.25], delays, dt=0.1,
    )
    np.testing.assert_allclose(t, [0, 0.1, 0.2, 0.3], rtol=0, atol=1e-14)
    np.testing.assert_allclose(s, [0, 0.25, 0.75, 1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(ds, [0, 1, 1, 0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(dds, [0, -2, -2, 0], rtol=0, atol=1e-14)


@pytest.mark.parametrize("progress_velocity", [2.0, 0.5], ids=["triangular", "coast"])
def test_shaped_joint_trajectory_limits_synchronization_and_endpoints(
    progress_velocity: float,
) -> None:
    qi = np.array([0.1, -0.3, 0.2, 0.1, -0.1, 0.2, -0.2])
    delta_q = np.array([1.0, -0.5, 0.25, -0.2, 0.1, -0.3, 0])
    qf = qi + delta_q
    moving = delta_q != 0
    # Moving joints give scalar limits v_s=progress_velocity and a_s=1.
    velocity_limits = np.where(moving, np.abs(delta_q) * progress_velocity, 1.0)
    acceleration_limits = np.where(moving, np.abs(delta_q), 1.0)
    dt = 0.01
    t, s, ds, dds = bang_coast_bang_progress(
        qi, qf, velocity_limits, acceleration_limits, dt
    )
    amplitudes, delays = zvd_shaper(2 * np.pi * 2.2, 0.1)
    assert np.all(amplitudes >= 0)
    assert amplitudes.sum() == pytest.approx(1.0, rel=0, abs=1e-14)
    t_shaped, s_shaped, ds_shaped, dds_shaped = apply_shaper_to_progress(
        s, ds, dds, amplitudes, delays, dt
    )
    q, dq, ddq = joint_trajectory_from_progress(
        qi, qf, s_shaped, ds_shaped, dds_shaped
    )

    assert np.all(np.abs(dq) <= velocity_limits + 1e-12)
    assert np.all(np.abs(ddq) <= acceleration_limits + 1e-12)
    for joint_values, progress in (
        (q - qi, s_shaped), (dq, ds_shaped), (ddq, dds_shaped)
    ):
        normalized = joint_values[:, moving] / delta_q[moving]
        np.testing.assert_allclose(
            normalized, np.broadcast_to(progress[:, None], normalized.shape),
            rtol=1e-12, atol=1e-14,
        )
    np.testing.assert_allclose(
        q[:, ~moving], np.tile(qi[~moving], (len(t_shaped), 1)),
        rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(dq[:, ~moving], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ddq[:, ~moving], 0, rtol=0, atol=1e-14)

    np.testing.assert_allclose(q[0], qi, rtol=0, atol=1e-14)
    np.testing.assert_allclose(q[-1], qf, rtol=0, atol=1e-14)
    np.testing.assert_allclose(dq[[0, -1]], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ddq[[0, -1]], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        s_shaped[[0, -1]], [0, 1], rtol=0, atol=1e-14
    )

    largest_delay_steps = int(np.rint(delays / dt).max())
    assert len(t_shaped) == len(t) + largest_delay_steps
    np.testing.assert_allclose(np.diff(t_shaped), dt, rtol=0, atol=1e-14)
    assert t_shaped[0] == pytest.approx(0, rel=0, abs=1e-14)
    assert t_shaped[-1] == pytest.approx(
        t[-1] + largest_delay_steps * dt, rel=0, abs=1e-14
    )


def test_waypoint_trajectory_joins_limits_and_synchronization() -> None:
    waypoints = np.array([
        [0.1, -0.3, 0.2, 0.1, -0.1, 0.2, 0.0],
        [0.5, -0.6, 0.2, 0.8, 0.2, -0.1, 0.3],
        [-0.2, -0.4, 0.1, 0.4, 0.3, 0.1, -0.2],
    ])
    velocity_limits = np.array([0.8, 0.2, 0.1, 0.4, 0.2, 0.1, 0.3])
    acceleration_limits = np.array([1.0, 0.8, 0.1, 0.1, 0.2, 0.1, 0.3])
    dt = 0.01
    t, q, dq, ddq = waypoint_trajectory(
        waypoints, velocity_limits, acceleration_limits, dt
    )
    np.testing.assert_allclose(np.diff(t), dt, rtol=0, atol=1e-14)
    assert np.all(np.diff(t) > 0)
    assert np.all(np.abs(dq) <= velocity_limits + 1e-12)
    assert np.all(np.abs(ddq) <= acceleration_limits + 1e-12)
    assert q.shape == dq.shape == ddq.shape == (len(t), 7)

    offset = 0
    for segment in range(len(waypoints) - 1):
        start, end = waypoints[segment:segment + 2]
        segment_t, s, ds, dds = bang_coast_bang_progress(
            start, end, velocity_limits, acceleration_limits, dt
        )
        stop = offset + len(segment_t) - 1
        for index, waypoint in ((offset, start), (stop, end)):
            np.testing.assert_allclose(q[index], waypoint, rtol=0, atol=1e-14)
            np.testing.assert_allclose(dq[index], 0, rtol=0, atol=1e-14)
            np.testing.assert_allclose(ddq[index], 0, rtol=0, atol=1e-14)
        delta_q = end - start
        moving = delta_q != 0
        for values, progress in (
            (q[offset:stop + 1] - start, s),
            (dq[offset:stop + 1], ds),
            (ddq[offset:stop + 1], dds),
        ):
            normalized = values[:, moving] / delta_q[moving]
            np.testing.assert_allclose(
                normalized, np.broadcast_to(progress[:, None], normalized.shape),
                rtol=1e-12, atol=1e-14,
            )
            np.testing.assert_allclose(values[:, ~moving], 0, rtol=0, atol=1e-14)
        offset = stop
    assert len(t) == offset + 1  # No extra sample or pause at the join.


def test_repeated_waypoints_add_no_duration() -> None:
    qi = np.zeros(7)
    qf = np.array([0.2, -0.3, 0.1, 0.4, 0, 0, 0])
    limits = np.ones(7)
    repeated = waypoint_trajectory([qi, qi, qf, qf], limits, limits, dt=0.01)
    direct = waypoint_trajectory([qi, qf], limits, limits, dt=0.01)
    for actual, expected in zip(repeated, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_all_stationary_waypoints() -> None:
    qi = np.arange(7, dtype=np.float64) / 10
    t, q, dq, ddq = waypoint_trajectory([qi, qi, qi], np.ones(7), np.ones(7), 0.01)
    np.testing.assert_allclose(t, [0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(q, qi[None, :], rtol=0, atol=1e-14)
    np.testing.assert_allclose(dq, np.zeros((1, 7)), rtol=0, atol=1e-14)
    np.testing.assert_allclose(ddq, np.zeros((1, 7)), rtol=0, atol=1e-14)


@pytest.mark.parametrize("use_zvd", [False, True], ids=["identity", "damped-zvd"])
def test_shape_complete_waypoint_trajectory(use_zvd: bool) -> None:
    waypoints = np.array([
        [0.1, -0.3, 0.2, 0.1, -0.1, 0.2, 0.0],
        [0.5, -0.6, 0.2, 0.8, 0.2, -0.1, 0.3],
        [-0.2, -0.4, 0.2, 0.4, 0.3, 0.1, -0.2],
    ])
    velocity_limits = np.array([0.8, 0.2, 0.1, 0.4, 0.2, 0.1, 0.3])
    acceleration_limits = np.array([1.0, 0.8, 0.1, 0.1, 0.2, 0.1, 0.3])
    dt = 0.01
    baseline = waypoint_trajectory(waypoints, velocity_limits, acceleration_limits, dt)
    t, q, dq, ddq = baseline
    amplitudes, delays = (
        zvd_shaper(2 * np.pi * 2.2, 0.1)
        if use_zvd else (np.array([1.0]), np.array([0.0]))
    )
    shaped = apply_shaper_to_joint_trajectory(q, dq, ddq, amplitudes, delays, dt)
    ts, qs, dqs, ddqs = shaped
    assert len(ts) == len(t) + int(np.rint(delays / dt).max())
    assert qs.shape == dqs.shape == ddqs.shape == (len(ts), 7)
    np.testing.assert_allclose(np.diff(ts), dt, rtol=0, atol=1e-14)
    np.testing.assert_allclose(qs[0], waypoints[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(qs[-1], waypoints[-1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(dqs[[0, -1]], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ddqs[[0, -1]], 0, rtol=0, atol=1e-14)
    assert np.all(np.abs(dqs) <= velocity_limits + 1e-12)
    assert np.all(np.abs(ddqs) <= acceleration_limits + 1e-12)
    # J3 never moves, even while the other joints change segment direction.
    np.testing.assert_allclose(qs[:, 2], 0.2, rtol=0, atol=1e-14)
    np.testing.assert_allclose(dqs[:, 2], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ddqs[:, 2], 0, rtol=0, atol=1e-14)
    if not use_zvd:
        for actual, expected in zip(shaped, baseline, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
