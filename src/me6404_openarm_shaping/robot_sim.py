import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Simulate and plot the left-arm trajectory.")
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="save full-resolution recordings to a uniquely named CSV in sim_out/",
    )
    args = parser.parse_args()
    simulation_time = 1.5  # Simulated seconds; edit to change the run duration.
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
    Kd = np.array([2.75, 2.50, 2.00, 2.00, 0.70, 0.60, 0.50]) * 0.8

    if not np.all(model.actuator_forcelimited[actuator_indices]):
        raise ValueError("Expected force-limited left-arm motor actuators.")

    torque_limits = model.actuator_forcerange[actuator_indices]

    qi = nominal_configuration(model)[position_indices]
    qf = np.array([-0.3, -0.7, -0.6, 1.5, 0.2, -0.4, 0.2])
    v_lim = np.ones(7) * 50.0
    a_lim = np.ones(7) * 100.0

    t, q, dq, ddq = synchronized_trajectory(qi, qf, v_lim, a_lim, dt=model.opt.timestep)

    time_history = []
    angle_history = []
    requested_history = []
    feedback_history = []
    step_count = int(np.ceil(simulation_time / model.opt.timestep))

    data.qpos[position_indices] = qi
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_index = 0
        step_deadline = time.perf_counter()
        next_sync = step_deadline
        sync_interval = 1.0 / 60.0

        while step_index < step_count and viewer.is_running():
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

            # mj_step evaluates actuator force at the pre-integration state.
            sample_time = data.time
            sample_angles = q_measured.copy()
            mujoco.mj_step(model, data)
            time_history.append(sample_time)
            angle_history.append(sample_angles)
            requested_history.append(torque.copy())
            feedback_history.append(data.actuator_force[actuator_indices].copy())
            step_index += 1

            now = time.perf_counter()
            if now >= next_sync:
                viewer.sync()
                next_sync = now + sync_interval

            step_deadline += model.opt.timestep
            sleep_time = step_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    times = np.asarray(time_history)
    angles = np.asarray(angle_history)
    requested_values = np.asarray(requested_history)
    feedback_values = np.asarray(feedback_history)

    if args.save_csv and time_history:
        output_dir = Path(__file__).resolve().parents[2] / "sim_out"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"simulation_{timestamp}.csv"
        columns = ["time_s"]
        for quantity in ("q_rad", "torque_requested_Nm", "torque_feedback_Nm"):
            columns.extend(f"J{i}_{quantity}" for i in range(1, 8))
        with output_path.open("x") as output_file:
            np.savetxt(
                output_file,
                np.column_stack((times, angles, requested_values, feedback_values)),
                delimiter=",",
                header=",".join(columns),
                comments="",
            )
        print(f"Saved simulation results: {output_path}")

    # Decimate only the plots; recordings and CSV retain every simulation step.
    max_plot_points = 500
    trajectory_points = np.linspace(
        0, len(t) - 1, min(len(t), max_plot_points), dtype=int
    )
    joint_labels = [f"J{i}" for i in range(1, 8)]
    fig, axs = plt.subplots(3, 1, sharex=True, num="Calculated trajectory")
    for ax, values, ylabel in zip(
        axs,
        (q, dq, ddq),
        ("Position [rad]", "Velocity [rad/s]", "Acceleration [rad/s²]"),
    ):
        ax.plot(t[trajectory_points], values[trajectory_points], label=joint_labels)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axs[0].legend(ncol=7)
    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()

    if time_history:
        result_points = np.linspace(
            0, len(times) - 1, min(len(times), max_plot_points), dtype=int
        )
        result_fig, result_axs = plt.subplots(
            2, 1, sharex=True, num="Recorded left-arm joints"
        )
        angle_lines = result_axs[0].plot(
            times[result_points], angles[result_points], label=joint_labels
        )
        for i, line in enumerate(angle_lines):
            result_axs[1].plot(
                times[result_points],
                requested_values[result_points, i],
                color=line.get_color(),
                linestyle="--",
                label="Requested (dashed)" if i == 0 else "_nolegend_",
            )
            result_axs[1].plot(
                times[result_points],
                feedback_values[result_points, i],
                color=line.get_color(),
                label="Feedback (solid)" if i == 0 else "_nolegend_",
            )
        result_axs[0].set_ylabel("Position [rad]")
        result_axs[0].legend(ncol=7)
        result_axs[1].set_ylabel("Motor torque [N·m]")
        result_axs[1].legend()
        result_axs[-1].set_xlabel("Simulation time [s]")
        for ax in result_axs:
            ax.grid(True)
        result_fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
