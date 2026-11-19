from collections.abc import Sequence
from hashlib import sha256
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from me6404_openarm_shaping.configuration import (
    experiment_fingerprint,
    load_robot_config,
)
from me6404_openarm_shaping.input_shaping import (
    ZVD_FREQUENCY_TOLERANCE,
    design_zvd_shaper,
    residual_vibration_factor,
    suggest_shaper_parameters,
)
from me6404_openarm_shaping.robot_model import (
    MODEL_PATH,
    MODEL_REVISION,
    check_joint_configuration,
    load_model,
    modal_properties,
)
from me6404_openarm_shaping.trajectories import waypoint_trajectory


def main() -> None:

    # ==================== SETUP AND NUMERICAL ANALYSIS ====================

    model = load_model()
    config = load_robot_config()
    model_sha256 = sha256(MODEL_PATH.read_bytes()).hexdigest()
    configuration_sha256 = experiment_fingerprint(config, model_sha256, MODEL_REVISION)
    print(f"\nExperiment configuration: {configuration_sha256}\n")

    if not config.input_shaping.time_invariant.zvd.enabled:
        print("Time-invariant ZVD is disabled for simulation.")
        print(
            "Modal analysis and candidate-shaper sensitivity are shown "
            "for design exploration only."
        )

    motor_kp, motor_kd = config.motor_kp, config.motor_kd

    model.opt.timestep = config.simulation.dt_s

    waypoints = config.waypoints

    zvd_settings = config.input_shaping.time_invariant.zvd

    for waypoint in waypoints:
        check_joint_configuration(model, waypoint)

    v_lim = config.velocity_limits
    a_lim = config.acceleration_limits

    t, q, _, _ = waypoint_trajectory(waypoints, v_lim, a_lim, dt=model.opt.timestep)

    # Pool both segments to design one fixed shaper for the complete trajectory.
    properties = [
        modal_properties(model, configuration, motor_kp, motor_kd)
        for configuration in q
    ]

    omega_n_hz = np.concatenate(
        [result["natural_frequency_hz"] for result in properties]
    )
    zeta = np.concatenate([result["damping_ratio"] for result in properties])

    # decay time constant, used for thresholding
    tau = 1 / (zeta * omega_n_hz * 2 * np.pi)
    tau_min = zvd_settings.tau_min_s
    retained = tau >= tau_min
    pooled_colors = np.where(retained, "green", "red")

    omega_n_hz_target = omega_n_hz[retained]
    zeta_target = zeta[retained]

    candidate_frequencies, candidate_damping, candidate_utilization = (
        suggest_shaper_parameters(
            properties, tau_min, ZVD_FREQUENCY_TOLERANCE, utilization_min=0.0
        )
    )

    print("\n-----INPUT SHAPING CANDIDATES-----")
    print("Candidate shaper natural frequencies [Hz]:", candidate_frequencies)
    print("Candidate shaper zeta:", candidate_damping)
    print("Candidate frequency-band utilization:", candidate_utilization)

    selected = candidate_utilization >= zvd_settings.utilization_min
    omega_n_nominal_hz_suggestion = candidate_frequencies[selected]
    zeta_nominal_suggestion = candidate_damping[selected]
    frequency_utilization = candidate_utilization[selected]

    print("\n-----INPUT SHAPING SUGGESTION-----")
    print("Suggested shaper natural frequencies [Hz]: ", omega_n_nominal_hz_suggestion)
    print("Suggested shaper zeta:", zeta_nominal_suggestion)
    print(
        "Suggested frequency-band utilization:",
        frequency_utilization,
    )
    print()

    if len(omega_n_nominal_hz_suggestion):
        combined_amplitudes, combined_times = design_zvd_shaper(
            omega_n_nominal_hz_suggestion, zeta_nominal_suggestion
        )

        # Use actual paired retained samples, not nominal/interpolated damping.
        residual_percent = 100 * np.array(
            [
                residual_vibration_factor(
                    combined_amplitudes,
                    combined_times,
                    2 * np.pi * frequency_hz,
                    damping,
                )
                for frequency_hz, damping in zip(
                    omega_n_hz_target, zeta_target, strict=True
                )
            ]
        )
        print("-----IDEAL COMBINED-SHAPER SENSITIVITY-----")
        print(f"Shaper duration: {np.max(combined_times):.6f} s")
        print(f"Maximum retained-sample residual: {residual_percent.max():.3f}%")
        print(
            "Retained samples exceeding 5%:",
            f"{np.count_nonzero(residual_percent > 5.0)} / {len(residual_percent)}",
        )
        print("Ideal continuous delays; frozen modes, not simulated vibration.")
        print()

        # ==================== RESIDUAL SENSITIVITY PLOTTING ====================
        residual_fig, residual_ax = plt.subplots(num="Residual vibration sensitivity")
        residual_ax.scatter(
            omega_n_hz_target,
            residual_percent,
            s=8,
            alpha=0.5,
            label="Retained modal samples",
        )
        residual_ax.axhline(5.0, linestyle="--", color="red", label="5% threshold")
        residual_ax.set_xlabel("Natural frequency [Hz]")
        residual_ax.set_ylabel("Residual vibration [%]")
        residual_ax.set_title("Suggested convolved ZVD shaper — ideal impulse delays")
        residual_ax.grid(True)
        residual_ax.legend()
        residual_fig.tight_layout()
    else:
        print("No shapers suggested; skipping residual vibration sensitivity.")

    # ==================== STABILITY DIAGNOSTICS ====================
    stable = np.array([result["stable"] for result in properties], dtype=bool)
    counts = np.array([len(result["eigenvalues"]) for result in properties])
    real_counts = [len(result["real_eigenvalues"]) for result in properties]

    print(f"Oscillatory pair count range: {counts.min()}--{counts.max()}")
    print(f"Real pole count range: {min(real_counts)}--{max(real_counts)}")
    print(f"Unstable or marginal samples: {np.count_nonzero(~stable)} / {len(t)}")

    if np.any(counts != counts[0]):
        print(
            "Oscillatory pair count changes; frequency ranks are not mode identities."
        )

    # ==================== MODAL AND POOLED PLOTTING ====================
    quantities = (
        ("natural_frequency_hz", "Natural frequency [Hz]"),
        ("damped_frequency_hz", "Damped frequency [Hz]"),
        ("damping_ratio", "Damping ratio [-]"),
    )
    fig, axes = plt.subplots(3, 1, sharex=True, num="Frozen-configuration modes")
    modal_axes = cast(Sequence[Axes], axes)
    for ax, (key, ylabel) in zip(modal_axes, quantities, strict=True):
        for rank in range(int(counts.max())):
            # Missing ranks remain gaps; scatter avoids implying branch tracking.
            values = np.array(
                [
                    result[key][rank] if len(result[key]) > rank else np.nan
                    for result in properties
                ]
            )
            ax.scatter(t, values, s=8, label=f"Frequency rank {rank + 1}")
        if np.any(~stable):
            ax.scatter(
                t[~stable],
                np.full(np.count_nonzero(~stable), 0.95),
                transform=ax.get_xaxis_transform(),
                marker="x",
                color="red",
                label="Unstable/marginal configuration",
            )
        ax.set_ylabel(ylabel)
        ax.grid(True)

    modal_axes[0].set_title(
        "Frozen-q, zero-velocity analysis; no physical mode tracking"
    )

    if counts.max() > 0 or np.any(~stable):
        modal_axes[0].legend()

    modal_axes[-1].set_xlabel("Trajectory time [s]")

    fig.tight_layout()

    pooled_fig, pooled_axes = plt.subplots(2, 1, num="Pooled modal properties")
    for ax, values, xlabel in zip(
        cast(Sequence[Axes], pooled_axes),
        (omega_n_hz, zeta),
        ("Natural frequency [Hz]", "Damping ratio [-]"),
        strict=True,
    ):
        ax.scatter(values, np.full_like(values, 0.15), c=pooled_colors, s=8, alpha=0.3)
        ax.spines["bottom"].set_position(("data", 0))
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_xlabel(xlabel)

    pooled_fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
