"""Synchronized offline joint trajectories."""

import numpy as np
from numpy.typing import ArrayLike

from me6404_openarm_shaping.data_types import FloatMatrix, FloatVector


def bang_coast_bang_progress(
    qi: ArrayLike,
    qf: ArrayLike,
    velocity_limits: ArrayLike,
    acceleration_limits: ArrayLike,
    dt: float,
) -> tuple[
    FloatVector,
    FloatVector,
    FloatVector,
    FloatVector,
]:
    """
    Return t, s, ds, dds, each with shape (N,).

    Positions are radians, symmetric positive limits are rad/s and rad/s^2,
    and dt is seconds.

    A shared bang-coast-bang progress profile synchronizes
    J1--J7. Short moves use a triangular profile without coast.
    Progress s is dimensionless; ds and dds have units 1/s and 1/s^2.
    Duration is stretched to a multiple of dt without exceeding joint limits.
    An unchanged configuration returns t=[0], s=[1], ds=dds=[0].
    """
    qi = np.asarray(qi, dtype=np.float64)
    qf = np.asarray(qf, dtype=np.float64)

    velocity_limits = np.asarray(velocity_limits, dtype=np.float64)
    acceleration_limits = np.asarray(acceleration_limits, dtype=np.float64)

    for values in (qi, qf, velocity_limits, acceleration_limits):
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Joint inputs must be finite arrays of shape (7,).")

    if np.any(velocity_limits <= 0) or np.any(acceleration_limits <= 0):
        raise ValueError("Velocity and acceleration limits must be positive.")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    delta_q = qf - qi

    moving = delta_q != 0

    if not np.any(moving):
        return np.array([0.0]), np.array([1.0]), np.zeros(1), np.zeros(1)

    displacement = np.abs(delta_q[moving])

    v_s = np.min(velocity_limits[moving] / displacement)
    a_s = np.min(acceleration_limits[moving] / displacement)

    peak_velocity = min(v_s, np.sqrt(a_s))

    accel_time = peak_velocity / a_s
    coast_time = max(0.0, (1.0 - peak_velocity**2 / a_s) / peak_velocity)

    minimum_duration = 2 * accel_time + coast_time

    num_steps = int(np.ceil(minimum_duration / dt))

    duration = num_steps * dt
    stretch = duration / minimum_duration

    t = np.arange(num_steps + 1, dtype=np.float64) * dt
    profile_time = t / stretch

    s = np.empty_like(t)
    ds = np.empty_like(t)
    dds = np.empty_like(t)

    accelerating = profile_time < accel_time
    coasting = (profile_time >= accel_time) & (profile_time < accel_time + coast_time)
    decelerating = ~(accelerating | coasting)

    tau = profile_time[accelerating]

    s[accelerating] = 0.5 * a_s * tau**2
    ds[accelerating] = a_s * tau
    dds[accelerating] = a_s

    tau = profile_time[coasting] - accel_time

    s[coasting] = 0.5 * a_s * accel_time**2 + peak_velocity * tau
    ds[coasting] = peak_velocity
    dds[coasting] = 0.0

    remaining = minimum_duration - profile_time[decelerating]
    s[decelerating] = 1.0 - 0.5 * a_s * remaining**2
    ds[decelerating] = a_s * remaining
    dds[decelerating] = -a_s

    ds /= stretch
    dds /= stretch**2
    s[0], s[-1] = 0.0, 1.0
    ds[0] = ds[-1] = 0.0
    dds[0] = dds[-1] = 0.0

    return t, s, ds, dds


def apply_shaper_to_progress(
    s: ArrayLike,
    ds: ArrayLike,
    dds: ArrayLike,
    amplitudes: ArrayLike,
    delays_s: ArrayLike,
    dt: float,
) -> tuple[
    FloatVector,
    FloatVector,
    FloatVector,
    FloatVector,
]:
    """Return shaped t, s, ds, dds on a fixed-dt grid starting at zero.

    Delays in seconds are rounded to the nearest sample.
    Duplicate delays accumulate. Amplitudes must sum to one; no normalization
    is applied.
    """
    s_array: FloatVector = np.asarray(s, dtype=np.float64)
    ds_array: FloatVector = np.asarray(ds, dtype=np.float64)
    dds_array: FloatVector = np.asarray(dds, dtype=np.float64)

    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    delays_s = np.asarray(delays_s, dtype=np.float64)

    if (
        s_array.ndim != 1 or s_array.size == 0
        or ds_array.shape != s_array.shape or dds_array.shape != s_array.shape
    ):
        raise ValueError(
            "Progress and derivatives must be matching nonempty 1-D arrays."
        )
    if (
        amplitudes.ndim != 1
        or amplitudes.size == 0
        or delays_s.shape != amplitudes.shape
        or not np.isfinite(amplitudes).all()
        or not np.isfinite(delays_s).all()
        or np.any(delays_s < 0)
    ):
        raise ValueError(
            "Shaper arrays must match, be finite, and have nonnegative delays."
        )
    if not np.isclose(amplitudes.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Shaper amplitudes must sum to one.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    delay_steps = np.rint(delays_s / dt).astype(np.intp)
    max_delay = int(delay_steps.max())

    size = len(s_array) + max_delay

    s_shaped = np.zeros(size)
    ds_shaped = np.zeros(size)
    dds_shaped = np.zeros(size)

    for amplitude, delay in zip(amplitudes, delay_steps, strict=True):
        padding = (int(delay), max_delay - int(delay))
        s_shaped += amplitude * np.pad(s_array, padding, mode="edge")
        ds_shaped += amplitude * np.pad(ds_array, padding, mode="constant")
        dds_shaped += amplitude * np.pad(dds_array, padding, mode="constant")

    t_shaped = np.arange(size, dtype=np.float64) * dt
    return t_shaped, s_shaped, ds_shaped, dds_shaped


def apply_shaper_to_joint_trajectory(
    q: ArrayLike,
    dq: ArrayLike,
    ddq: ArrayLike,
    amplitudes: ArrayLike,
    delays_s: ArrayLike,
    dt: float,
) -> tuple[
    FloatVector,
    FloatMatrix,
    FloatMatrix,
    FloatMatrix,
]:
    """Shape complete joint histories with the same impulses on all seven axes.

    Inputs are (N, 7) arrays in rad, rad/s, rad/s^2. Return t, q, dq, ddq,
    including the rounded-delay tail. Positions hold their endpoint values
    outside the input interval; velocities and accelerations extend with zeros.
    Intermediate waypoints/stops need not be preserved. Nonnegative unit-sum
    weights preserve componentwise velocity and acceleration magnitude bounds.
    """
    q_array: FloatMatrix = np.asarray(q, dtype=np.float64)
    dq_array: FloatMatrix = np.asarray(dq, dtype=np.float64)
    ddq_array: FloatMatrix = np.asarray(ddq, dtype=np.float64)
    if (
        q_array.ndim != 2
        or q_array.shape[1] != 7
        or len(q_array) == 0
        or dq_array.shape != q_array.shape
        or ddq_array.shape != q_array.shape
    ):
        raise ValueError(
            "Joint trajectories must have matching nonempty (N, 7) shapes."
        )
    shaped = [
        apply_shaper_to_progress(
            q_array[:, joint], dq_array[:, joint], ddq_array[:, joint],
            amplitudes, delays_s, dt
        )
        for joint in range(7)
    ]
    return (
        shaped[0][0],
        np.column_stack([result[1] for result in shaped]),
        np.column_stack([result[2] for result in shaped]),
        np.column_stack([result[3] for result in shaped]),
    )


def joint_trajectory_from_progress(
    qi: ArrayLike,
    qf: ArrayLike,
    s: ArrayLike,
    ds: ArrayLike,
    dds: ArrayLike,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    """Map unshaped or shaped progress to q, dq, ddq of shape (N, 7).

    Positions are radians, velocities rad/s, and accelerations rad/s^2.
    """
    qi = np.asarray(qi, dtype=np.float64)
    qf = np.asarray(qf, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    ds = np.asarray(ds, dtype=np.float64)
    dds = np.asarray(dds, dtype=np.float64)

    if qi.shape != (7,) or qf.shape != (7,):
        raise ValueError("Joint configurations must have shape (7,).")
    if s.ndim != 1 or ds.shape != s.shape or dds.shape != s.shape:
        raise ValueError("Progress and derivatives must be matching 1-D arrays.")

    delta_q = qf - qi

    q = qi + s[:, None] * delta_q
    dq = ds[:, None] * delta_q
    ddq = dds[:, None] * delta_q

    return q, dq, ddq


def waypoint_trajectory(
    waypoints: ArrayLike,
    velocity_limits: ArrayLike,
    acceleration_limits: ArrayLike,
    dt: float,
) -> tuple[
    FloatVector,
    FloatMatrix,
    FloatMatrix,
    FloatMatrix,
]:
    """Concatenate rest-to-rest segments through waypoints of shape (N+1, 7).

    Return t, q, dq, ddq on one fixed-dt grid, in seconds, rad, rad/s, rad/s^2.

    Every shared waypoint appears once with zero commanded velocity and
    acceleration. There is no dwell; repeated consecutive waypoints add no
    duration.

    Joints share progress within each segment, not over the whole path.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 7 or len(waypoints) < 2:
        raise ValueError("Provide at least two seven-joint waypoints, shape (N+1, 7).")

    positions: list[FloatMatrix] = []
    velocities: list[FloatMatrix] = []
    accelerations: list[FloatMatrix] = []

    for segment in range(len(waypoints) - 1):
        start, end = waypoints[segment : segment + 2]
        _, s, ds, dds = bang_coast_bang_progress(
            start, end, velocity_limits, acceleration_limits, dt
        )
        q, dq, ddq = joint_trajectory_from_progress(start, end, s, ds, dds)
        q[0], q[-1] = start, end
        # Retain the preceding endpoint rather than duplicating a join sample.
        keep = slice(None) if segment == 0 else slice(1, None)
        positions.append(q[keep])
        velocities.append(dq[keep])
        accelerations.append(ddq[keep])

    q = np.concatenate(positions)
    dq = np.concatenate(velocities)
    ddq = np.concatenate(accelerations)
    t = np.arange(len(q), dtype=np.float64) * dt

    return t, q, dq, ddq
