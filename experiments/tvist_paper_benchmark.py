"""Approximate Thomsen et al. (2019), §8.1, with added fixed-shaper comparisons."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from me6404_openarm_shaping.data_types import FloatVector
from me6404_openarm_shaping.tvist import TvistShaper

from .tvist_oscillator_experiment import shaped_command

DT_S = 1 / 500
TARGET_RAD = 0.5
MAX_VELOCITY = 0.375
MAX_ACCELERATION = 1.0
STIFFNESS = 2000.0
DAMPING = 22.1
RING_DOWN_S = 2.0


def inertia[T: (float, FloatVector)](angle: T) -> T:
    return 3.10 + 1.39 * angle


def reference() -> tuple[FloatVector, FloatVector]:
    """Sample the bang–coast–bang move on the paper's 500 Hz grid."""
    ramp_s = MAX_VELOCITY / MAX_ACCELERATION
    end_s = TARGET_RAD / MAX_VELOCITY + ramp_s
    t = np.arange(int(np.ceil(end_s / DT_S)) + 1) * DT_S
    ramp_up = np.minimum(t, ramp_s)
    time_left = np.clip(end_s - t, 0, ramp_s)
    q = np.where(
        t <= end_s - ramp_s,
        0.5 * MAX_ACCELERATION * ramp_up**2 + MAX_VELOCITY * (t - ramp_up),
        TARGET_RAD - 0.5 * MAX_ACCELERATION * time_left**2,
    )
    return t, q


def simulate(command: FloatVector) -> FloatVector:
    """Interpret Fig. 20 as J(x)x'' = k(u−x) + c(u'−x').

    Assumptions: no additional inertia-gradient torque; RK4 with linearly
    interpolated input and centered-difference command velocity.
    """
    velocity = np.gradient(command, DT_S, edge_order=2)
    x = np.zeros_like(command)
    v = 0.0

    def acceleration(position: float, speed: float, u: float, du: float) -> float:
        return (STIFFNESS * (u - position) + DAMPING * (du - speed)) / inertia(position)

    h = DT_S / 2
    for i in range(len(command) - 1):
        u0, u1 = command[i : i + 2]
        du0, du1 = velocity[i : i + 2]
        um, dum = (u0 + u1) / 2, (du0 + du1) / 2
        k1 = acceleration(x[i], v, u0, du0)
        v2 = v + h * k1
        k2 = acceleration(x[i] + h * v, v2, um, dum)
        v3 = v + h * k2
        k3 = acceleration(x[i] + h * v2, v3, um, dum)
        v4 = v + DT_S * k3
        k4 = acceleration(x[i] + DT_S * v3, v4, u1, du1)
        x[i + 1] = x[i] + DT_S / 6 * (v + 2 * v2 + 2 * v3 + v4)
        v += DT_S / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def benchmark() -> tuple[
    FloatVector, dict[str, tuple[FloatVector, FloatVector, int, float]]
]:
    """Return padded commands, responses, endpoint indices, and peak residuals."""
    _, q = reference()
    j = inertia(q)
    # The shaper estimates dynamics at the current unshaped input (Fig. 19).
    # The simulated plant evaluates inertia at its actual output (Fig. 20).
    fn = np.sqrt(STIFFNESS / j) / (2 * np.pi)
    zeta = DAMPING / (2 * np.sqrt(STIFFNESS * j))
    # Adapt §8.4's tuning rule to this narrower case: geometric mean of
    # endpoint *damped* frequencies, and mean of the endpoint damping ratios.
    fd = fn * np.sqrt(1 - zeta**2)
    midpoint_zeta = float((zeta[0] + zeta[-1]) / 2)
    midpoint_fn = np.sqrt(fd[0] * fd[-1]) / np.sqrt(1 - midpoint_zeta**2)
    commands = {
        "Unshaped": q,
        "Fixed-start ZV": shaped_command(
            q, np.full_like(q, fn[0]), float(zeta[0]), DT_S, shaper=TvistShaper.ZV
        ),
        "Fixed-end ZV": shaped_command(
            q, np.full_like(q, fn[-1]), float(zeta[-1]), DT_S, shaper=TvistShaper.ZV
        ),
        "Fixed-start ZVD": shaped_command(
            q, np.full_like(q, fn[0]), float(zeta[0]), DT_S
        ),
        "Fixed-end ZVD": shaped_command(
            q, np.full_like(q, fn[-1]), float(zeta[-1]), DT_S
        ),
        "Fixed-midpoint ZVD": shaped_command(
            q, np.full_like(q, midpoint_fn), midpoint_zeta, DT_S
        ),
        "OSA ZV, linear": shaped_command(
            q, fn, zeta, DT_S, shaper=TvistShaper.ZV, lagrange_order=1
        ),
        "FD-FIR ZV, cubic": shaped_command(
            q, fn, zeta, DT_S, shaper=TvistShaper.ZV, lagrange_order=3
        ),
    }
    ring_samples = round(RING_DOWN_S / DT_S)
    count = max(map(len, commands.values())) + ring_samples
    results = {}
    for label, command in commands.items():
        end = int(np.flatnonzero(np.abs(command - TARGET_RAD) > 1e-12)[-1]) + 1
        assert end < len(command), f"{label} did not reach the target"
        u = np.pad(command, (0, count - len(command)), mode="edge")
        x = simulate(u)
        # Use peak absolute position error after each command ends, normalized
        # to the unshaped peak. The paper does not specify its exact window.
        peak = float(np.max(np.abs(x[end : end + ring_samples] - TARGET_RAD)))
        results[label] = (u, x, end, peak)
    return np.arange(count) * DT_S, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-plot", type=Path)
    args = parser.parse_args()
    t, results = benchmark()
    baseline = results["Unshaped"][3]
    reported = {"OSA ZV, linear": 0.373, "FD-FIR ZV, cubic": 0.368}
    print("Approximate PRV: peak post-command position error / unshaped peak")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    for label, (u, x, end, peak) in results.items():
        prv = 100 * peak / baseline
        comparison = f" (paper: {reported[label]:.3f}%)" if label in reported else ""
        print(f"{label}: end={t[end]:.3f} s, PRV={prv:.3f}%{comparison}")
        axes[0].plot(t, u, label=label)
        axes[1].plot(t[end:] - t[end], x[end:] - TARGET_RAD, label=label)
    axes[0].set(xlabel="Time [s]", ylabel="Command [rad]")
    axes[1].set(xlabel="Time after command end [s]", ylabel="Position error [rad]")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Thomsen et al. §8.1 — approximate case-study reproduction")
    if args.save_plot is None:
        plt.show()
    else:
        fig.savefig(args.save_plot, dpi=150)
        print(f"Saved plot: {args.save_plot}")


if __name__ == "__main__":
    main()
