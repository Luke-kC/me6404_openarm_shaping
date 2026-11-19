"""Compare the two lowest modal frequencies at matching command timestamps."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from me6404_openarm_shaping.configuration import RobotConfig, load_robot_config
from me6404_openarm_shaping.data_types import FloatMatrix, FloatVector, ModalProperties
from me6404_openarm_shaping.robot_model import load_model, modal_properties
from me6404_openarm_shaping.trajectories import waypoint_trajectory
from me6404_openarm_shaping.tvist import (
    TvistShaper,
    apply_tvist_to_joint_trajectory,
    select_tvist_modes,
)


def compare_frequencies(
    config: RobotConfig, iterations: int = 1
) -> tuple[FloatVector, list[FloatMatrix], float]:
    """Return the common time grid and frequencies at q_d, q_s^(0), q_s^(1), ..."""
    if iterations < 1:
        raise ValueError("At least one shaping pass is required.")
    print("Loading model and generating the unshaped trajectory...", flush=True)
    model = load_model()
    dt = model.opt.timestep = config.simulation.dt_s
    t_unshaped, q_unshaped, _, _ = waypoint_trajectory(
        config.waypoints,
        0.8 * config.velocity_limits,
        0.8 * config.acceleration_limits,
        dt=dt,
    )

    def sample_modes(positions: FloatMatrix, label: str) -> list[ModalProperties]:
        total = len(positions)
        print(f"Calculating {label} modal frequencies ({total} samples)...", flush=True)
        samples: list[ModalProperties] = []
        for index, pose in enumerate(positions):
            sample = modal_properties(model, pose, config.motor_kp, config.motor_kd)
            samples.append(sample)
            done = index + 1
            percent = 100 * done // total
            if percent // 10 > (100 * index // total) // 10:
                print(f"  {label}: {done}/{total} samples ({percent}%)", flush=True)
        return samples

    def lowest_frequencies(samples: list[ModalProperties]) -> FloatMatrix:
        # Keep the original two-rank diagnostic, independent of shaper selection.
        if any(len(sample["natural_frequency_hz"]) < 2 for sample in samples):
            raise ValueError("The diagnostic needs two oscillatory frequencies to plot.")
        return np.stack([sample["natural_frequency_hz"][:2] for sample in samples])

    samples = sample_modes(q_unshaped, "unshaped")
    frequencies = [lowest_frequencies(samples)]
    tau_min = config.input_shaping.time_varying.zvd.tau_min_s
    common_length = len(q_unshaped)
    for iteration in range(iterations):
        schedule = select_tvist_modes(samples, tau_min)

        # Always filter q_d. Holding its final pose lets the updated schedule
        # vary through the previous output's tail at the original timestamps.
        q_input = np.pad(
            q_unshaped, ((0, len(schedule) - len(q_unshaped)), (0, 0)), mode="edge"
        )
        print(f"Generating TVIST ZVD iteration {iteration}...", flush=True)
        q_shaped: FloatMatrix = apply_tvist_to_joint_trajectory(
            q=q_input,
            dt=dt,
            properties=schedule,
            shaper_type=TvistShaper.ZVD,
            lagrange_order=1,
        )
        if not np.allclose(q_shaped[-1], q_unshaped[-1], rtol=0, atol=1e-12):
            raise ValueError(f"TVIST iteration {iteration} did not reach the endpoint.")
        if iteration == 0:
            common_length = len(q_shaped)
        elif np.allclose(
            q_shaped[len(schedule) - 1 :], q_unshaped[-1], rtol=0, atol=1e-12
        ):
            # The filter appends a tail automatically; drop it only if redundant.
            q_shaped = q_shaped[: len(schedule)]
        samples = sample_modes(q_shaped, f"TVIST iteration {iteration}")
        frequencies.append(lowest_frequencies(samples))

    # Use the iteration-0 horizon for all comparisons so extra hold samples
    # in later iterations cannot dilute the reported RMS mismatch.
    frequencies[0] = np.pad(
        frequencies[0], ((0, common_length - len(q_unshaped)), (0, 0)), mode="edge"
    )
    return (
        np.arange(common_length) * dt,
        [values[:common_length] for values in frequencies],
        float(t_unshaped[-1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="number of shaping passes; 2 runs iterations 0 and 1 (default: 1)",
    )
    parser.add_argument(
        "--save-plot", type=Path, help="save the plot instead of showing it"
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1.")
    t, frequencies, unshaped_end = compare_frequencies(
        load_robot_config(), args.iterations
    )

    columns = 2 if args.iterations > 1 else 1
    fig, axes = plt.subplots(
        2,
        columns,
        sharex=True,
        squeeze=False,
        figsize=(12 if columns == 2 else 9, 6),
        layout="constrained",
    )
    print(f"Mismatch metrics compared over 0–{t[-1]:.3f} s (iteration-0 horizon).")
    for i, panels in enumerate(axes):
        ax = panels[0]
        ax.plot(t, frequencies[0][:, i], label="At unshaped command $q_d(t)$")
        ax.axvline(
            unshaped_end, color="0.5", linestyle=":", label="Unshaped command end"
        )
        ax.set_title(f"Frequency rank {i + 1}")
        ax.set_ylabel("Natural frequency [Hz]")
        ax.grid(True)
        for iteration in range(args.iterations):
            label = f"Iteration {iteration}"
            (line,) = ax.plot(t, frequencies[iteration + 1][:, i], "--", label=label)
            error = frequencies[iteration + 1][:, i] - frequencies[iteration][:, i]
            if columns == 2:
                panels[1].plot(t, error, color=line.get_color(), label=label)
            peak_index = int(np.argmax(np.abs(error)))
            rms = np.sqrt(np.mean(error**2))
            print(
                f"{label}, rank {i + 1}: peak |difference| {abs(error[peak_index]):.4f} Hz "
                f"at {t[peak_index]:.3f} s; RMS {rms:.4f} Hz"
            )
        ax.legend()
        if columns == 2:
            panels[1].set_title(f"Frequency rank {i + 1}: output minus schedule")
            panels[1].set_ylabel("Frequency mismatch [Hz]")
            panels[1].grid(True)
            panels[1].legend()
    fig.supxlabel("Time [s]")
    fig.suptitle("TVIST ZVD: modal frequencies at unshaped and shaped poses")
    if args.save_plot is not None:
        fig.savefig(args.save_plot)
        print(f"Saved plot: {args.save_plot}")
    else:
        print("Comparison complete. Opening the plot window...", flush=True)
        plt.show()


if __name__ == "__main__":
    main()
