"""Compare fixed and scheduled ZVD on a scalar oscillator, without robot dynamics."""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from me6404_openarm_shaping.data_types import FloatVector, ModalProperties
from me6404_openarm_shaping.robot_model import LEFT_ARM_JOINT_NAMES
from me6404_openarm_shaping.tvist import TvistShaper, apply_tvist_to_joint_trajectory

DT_S = 0.001
DAMPING_RATIO = 0.10
MOVE_DURATION_S = 0.5
RESIDUAL_WINDOW_S = 0.5


def simulate_oscillator(
    command: FloatVector, frequency_hz: FloatVector, damping: float, dt: float
) -> tuple[FloatVector, FloatVector]:
    """Integrate from rest at u(0) by RK4, linearly interpolating u and frequency."""
    x = np.empty_like(command)
    v = np.empty_like(command)
    x[0], v[0] = command[0], 0.0
    omega = 2 * np.pi * frequency_hz

    def acceleration(x: float, v: float, u: float, w: float) -> float:
        return w**2 * (u - x) - 2 * damping * w * v

    h = dt / 2
    for i in range(len(command) - 1):
        u0, u1 = command[i : i + 2]
        w0, w1 = omega[i : i + 2]
        um, wm = (u0 + u1) / 2, (w0 + w1) / 2
        k1x = v[i]
        k1v = acceleration(x[i], v[i], u0, w0)
        k2x = v[i] + h * k1v
        k2v = acceleration(x[i] + h * k1x, k2x, um, wm)
        k3x = v[i] + h * k2v
        k3v = acceleration(x[i] + h * k2x, k3x, um, wm)
        k4x = v[i] + dt * k3v
        k4v = acceleration(x[i] + dt * k3x, k4x, u1, w1)
        x[i + 1] = x[i] + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v[i + 1] = v[i] + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x, v


def shaped_command(
    command: FloatVector,
    schedule_hz: FloatVector,
    damping: float | FloatVector,
    dt: float,
    *,
    shaper: TvistShaper = TvistShaper.ZVD,
    lagrange_order: Literal[1, 2, 3] = 1,
) -> FloatVector:
    """Reuse the robot's fractional-delay filter on one channel, without a robot model."""
    schedule: list[ModalProperties] = []
    for f, zeta in zip(
        schedule_hz, np.broadcast_to(damping, schedule_hz.shape), strict=True
    ):
        damped_ratio = np.sqrt(1 - zeta**2)
        schedule.append(
            {
                "natural_frequency_hz": np.array([f]),
                "damped_frequency_hz": np.array([f * damped_ratio]),
                "damping_ratio": np.array([zeta]),
                "eigenvalues": np.array([2 * np.pi * f * (-zeta + 1j * damped_ratio)]),
                "real_eigenvalues": np.empty(0),
                "stable": bool(zeta > 0),
            }
        )
    channels = np.zeros((len(command), len(LEFT_ARM_JOINT_NAMES)))
    channels[:, 0] = command
    return apply_tvist_to_joint_trajectory(
        channels, dt, schedule, shaper, lagrange_order=lagrange_order
    )[:, 0]


def experiment_commands(
    final_frequency_hz: float,
    initial_frequency_hz: float = 2.0,
) -> tuple[FloatVector, dict[str, FloatVector]]:
    """Return the shared plant schedule during the move and each command."""
    progress = np.linspace(0.0, 1.0, round(MOVE_DURATION_S / DT_S) + 1)
    reference = progress**3 * (10 - 15 * progress + 6 * progress**2)
    variable_hz = (
        initial_frequency_hz + (final_frequency_hz - initial_frequency_hz) * reference
    )
    return variable_hz, {
        "Unshaped": reference,
        "Fixed-start ZVD": shaped_command(
            reference, np.full_like(reference, variable_hz[0]), DAMPING_RATIO, DT_S
        ),
        "Fixed-end ZVD": shaped_command(
            reference, np.full_like(reference, variable_hz[-1]), DAMPING_RATIO, DT_S
        ),
        "TVIST ZVD": shaped_command(reference, variable_hz, DAMPING_RATIO, DT_S),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("constant", "varying", "reverse"), default="varying"
    )
    parser.add_argument(
        "--save-plot", type=Path, help="save the plot instead of showing it"
    )
    args = parser.parse_args()
    cases = {"constant": (2.0, 2.0), "varying": (2.0, 3.0), "reverse": (3.0, 2.0)}
    start_hz, end_hz = cases[args.case]
    plant_hz, commands = experiment_commands(end_hz, initial_frequency_hz=start_hz)
    window_samples = round(RESIDUAL_WINDOW_S / DT_S)
    count = max(map(len, commands.values())) + window_samples
    t = np.arange(count) * DT_S
    plant_hz = np.pad(plant_hz, (0, count - len(plant_hz)), mode="edge")
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8), layout="constrained")
    axes[0].plot(t, plant_hz, color="black")
    print(
        f"{args.case.capitalize()} oscillator: RMS over "
        f"{RESIDUAL_WINDOW_S:g} s after each command ends"
    )
    for label, command in commands.items():
        # Ignore the filter's extra endpoint hold samples when locating motion end.
        end = int(np.flatnonzero(np.abs(command - 1.0) > 1e-12)[-1]) + 1
        assert end < len(command), f"{label} did not reach the target"
        u = np.pad(command, (0, count - len(command)), mode="edge")
        x, v = simulate_oscillator(u, plant_hz, DAMPING_RATIO, DT_S)
        window = slice(end + 1, end + 1 + window_samples)
        error_rms = np.sqrt(np.mean((x[window] - 1.0) ** 2))
        velocity_rms = np.sqrt(np.mean(v[window] ** 2))
        print(
            f"{label:15s} end={t[end]:.3f} s  "
            f"position error RMS={error_rms:.6g}  velocity RMS={velocity_rms:.6g} /s"
        )
        axes[1].plot(t, u, label=label)
        axes[2].plot(t, x, label=label)
        axes[3].plot(t, v, label=label)

    for ax, ylabel in zip(
        axes, ("Frequency [Hz]", "Command", "Position", "Velocity [1/s]"), strict=True
    ):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[1].legend()
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Oscillator: {start_hz:g} → {end_hz:g} Hz (ζ = {DAMPING_RATIO:g})")
    if args.save_plot is None:
        plt.show()
    else:
        fig.savefig(args.save_plot, dpi=150)
        print(f"Saved plot: {args.save_plot}")


if __name__ == "__main__":
    main()
