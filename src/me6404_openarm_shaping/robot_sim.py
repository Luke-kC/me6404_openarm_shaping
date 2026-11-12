import time

import mujoco
import mujoco.viewer
import numpy as np

from me6404_openarm_shaping.robot_model import (
    load_model,
    nominal_configuration,
    resolve_joint_indices,
)
from me6404_openarm_shaping.trajectories import synchronized_trajectory


def main() -> None:
    model = load_model()
    data = mujoco.MjData(model)

    position_indices, velocity_indices = resolve_joint_indices(model)
    actuator_names = tuple(f"left_joint{i}_ctrl" for i in range(1, 8))
    actuator_indices = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in actuator_names
        ],
        dtype=np.intp,
    )
    if np.any(actuator_indices < 0):
        raise ValueError("Could not resolve all seven left-arm motor actuators.")

    Kp = np.array([70, 70, 70, 60, 10, 10, 10], dtype=np.float64)
    Kd = np.array([2.75, 2.50, 2.00, 2.00, 0.70, 0.60, 0.50])

    if not np.all(model.actuator_forcelimited[actuator_indices]):
        raise ValueError("Expected force-limited left-arm motor actuators.")

    torque_limits = model.actuator_forcerange[actuator_indices]

    qi = nominal_configuration(model)[position_indices]
    qf = np.array([-0.3, -0.7, -0.6, 1.8, 0, -0.4, 0.2])
    v_lim = np.ones(7) * 50.0
    a_lim = np.ones(7) * 100.0

    t, q, dq, ddq = synchronized_trajectory(qi, qf, v_lim, a_lim, dt=model.opt.timestep)

    data.qpos[position_indices] = qi
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_index = 0
        step_deadline = time.perf_counter()
        next_sync = step_deadline
        sync_interval = 1.0 / 60.0

        while viewer.is_running():
            target_index = min(step_index, len(q) - 1)

            q_measured = data.qpos[position_indices]
            dq_measured = data.qvel[velocity_indices]

            # torque = Kp * (q[target_index] - q_measured) + Kd * (
            #     dq[target_index] - dq_measured
            # )

            torque = Kp * (q[target_index] - q_measured) + Kd * (
                dq[target_index] * 0 - dq_measured
            )

            if not np.isfinite(torque).all():
                raise ValueError("Nonfinite left-arm torque; stopping simulation.")

            data.ctrl[actuator_indices] = np.clip(
                torque, torque_limits[:, 0], torque_limits[:, 1]
            )

            mujoco.mj_step(model, data)
            step_index += 1

            now = time.perf_counter()
            if now >= next_sync:
                viewer.sync()
                next_sync = now + sync_interval

            step_deadline += model.opt.timestep
            sleep_time = step_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
