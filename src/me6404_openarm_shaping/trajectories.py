"""Synchronized offline joint trajectories."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def synchronized_trajectory(
    qi: ArrayLike,
    qf: ArrayLike,
    velocity_limits: ArrayLike,
    acceleration_limits: ArrayLike,
    dt: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Return t, q, dq, ddq with shapes (N,), (N, 7), (N, 7), (N, 7).

    Positions are radians, symmetric positive limits are rad/s and rad/s^2,
    and dt is seconds.

    A shared bang-coast-bang progress profile synchronizes
    J1--J7. Short moves use a triangular profile without coast.
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
        return np.array([0.0]), qi[None, :].copy(), np.zeros((1, 7)), np.zeros((1, 7))

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

    q = qi + s[:, None] * delta_q
    q[0], q[-1] = qi, qf
    dq = ds[:, None] * delta_q
    ddq = dds[:, None] * delta_q

    return t, q, dq, ddq
