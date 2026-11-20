"""Compare ZVD suggestions across reproducible random offline waypoint paths.

Paths are joint-limit-valid, not collision-checked or hardware-safe. Analysis
uses a subset of frozen configurations, not moving-system linearization.
"""

import argparse
from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from me6404_openarm_shaping.configuration import load_robot_config
from me6404_openarm_shaping.data_types import FloatVector
from me6404_openarm_shaping.input_shaping import (
    ZVD_FREQUENCY_TOLERANCE,
    suggest_shaper_parameters,
)
from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    MODEL_REVISION,
    load_model,
    modal_properties,
    nominal_configuration,
    resolve_joint_indices,
)
from me6404_openarm_shaping.trajectories import waypoint_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="maximum modal samples per complete trajectory",
    )
    parser.add_argument("--seed", type=int, default=6404)
    args = parser.parse_args()
    if args.trajectories < 1 or args.samples < 2:
        parser.error("Require at least one trajectory and at least two modal samples.")

    config = load_robot_config()
    model = load_model(config.passive_damping)
    model.opt.timestep = config.simulation.dt_s
    position_indices, _ = resolve_joint_indices(model)
    qi = nominal_configuration(model)[position_indices]
    joint_ranges = np.array([model.joint(name).range for name in LEFT_ARM_JOINT_NAMES])
    # The existing stiffness calculation perturbs angles by +/-0.001 rad.
    margin = 0.002  # rad; random target positions stay away from the limits.
    lower = joint_ranges[:, 0] + margin
    upper = joint_ranges[:, 1] - margin
    rng = np.random.default_rng(args.seed)
    zvd_settings = config.input_shaping.time_invariant.zvd
    tau_min = zvd_settings.tau_min_s
    utilization_min = zvd_settings.utilization_min

    print(f"Model revision: {MODEL_REVISION}")
    print(f"Seed: {args.seed}; trajectories: {args.trajectories}; targets per path: 2")
    print(f"Maximum modal samples per path: {args.samples}; dt: {model.opt.timestep} s")
    print(f"tau_min: {tau_min} s; utilization_min: {utilization_min}")
    print(f"Joint-limit inset: {margin} rad; initial J1--J7 configuration: {qi}")
    print()
    for key, values in (
        ("motor_kp", config.motor_kp), ("motor_kd", config.motor_kd),
        ("velocity_limits", config.velocity_limits),
        ("acceleration_limits", config.acceleration_limits),
    ):
        print(f"{key}: {values}")
    print("Configured waypoints are not used; random paths start at nominal qi.")
    print()
    print(
        "Targets uniform within inset MJCF limits; no collision/safety certification."
    )

    results: list[tuple[FloatVector, FloatVector, FloatVector] | None] = []
    for trial in range(args.trajectories):
        waypoints = np.vstack((qi, rng.uniform(lower, upper, size=(2, 7))))
        print(f"\n----- TRAJECTORY {trial + 1} -----\n")
        print("# J1--J7 positions in radians")
        print("waypoints = np.array([")
        for waypoint in waypoints:
            # Python float repr preserves round-trip precision without truncation.
            print(f"    {waypoint.tolist()},")
        print("])")
        print()
        t, q, _, _ = waypoint_trajectory(
            waypoints,
            config.velocity_limits,
            config.acceleration_limits,
            dt=model.opt.timestep,
        )
        indices = np.linspace(0, len(t) - 1, min(len(t), args.samples), dtype=int)
        properties = [
            modal_properties(model, q[index], config.motor_kp, config.motor_kd)
            for index in indices
        ]
        unstable_count = sum(not sample["stable"] for sample in properties)
        if unstable_count:
            results.append(None)
            print(
                f"Trajectory {trial + 1}: {unstable_count}/{len(indices)} sampled "
                "configurations unstable/marginal; excluded from shaper design."
            )
            print()
            continue
        frequencies, damping, utilization = suggest_shaper_parameters(
            properties, tau_min, ZVD_FREQUENCY_TOLERANCE, utilization_min
        )
        results.append((frequencies, damping, utilization))
        print(
            f"Trajectory {trial + 1}: duration={t[-1]:.3f} s, samples={len(indices)}, "
            f"ZVD components={len(frequencies)}\n"
            f"  frequencies [Hz]: {frequencies}\n"
            f"  damping ratios: {damping}\n"
            f"  utilization: {utilization}"
        )
        print()

    valid = [result for result in results if result is not None]
    invalid_ids = [index + 1 for index, result in enumerate(results) if result is None]
    print("\n----- SUMMARY -----\n")
    print(f"Valid sampled paths: {len(valid)} / {len(results)}")
    print(f"Excluded unstable/marginal path IDs: {invalid_ids}")
    if valid:
        counts, occurrences = np.unique(
            [len(result[0]) for result in valid], return_counts=True
        )
        print(
            "ZVD component-count distribution:",
            dict(zip(counts.tolist(), occurrences.tolist(), strict=True)),
        )
    max_rank = max((len(result[0]) for result in valid), default=0)
    print()
    for rank in range(max_rank):
        frequencies = [result[0][rank] for result in valid if len(result[0]) > rank]
        damping = [result[1][rank] for result in valid if len(result[0]) > rank]
        print(
            f"Frequency rank {rank + 1}: "
            f"{min(frequencies):.4f}--{max(frequencies):.4f} Hz; "
            f"zeta {min(damping):.4f}--{max(damping):.4f}"
        )
    print()
    print("Ranks are sorted suggestions, not tracked physical mode identities.")
    print("Similar suggestions alone do not validate one shared shaper for all paths.")

    fig, raw_axes = plt.subplots(
        4, 1, sharex=True, figsize=(10, 11), num="Random-trajectory ZVD suggestions"
    )
    axes = cast(Sequence[Axes], raw_axes)
    ids = [index + 1 for index, result in enumerate(results) if result is not None]
    axes[0].scatter(ids, [len(result[0]) for result in valid], color="tab:blue")
    axes[0].set_ylabel("ZVD components")
    for rank in range(max_rank):
        available = [
            (index + 1, result)
            for index, result in enumerate(results)
            if result is not None and len(result[0]) > rank
        ]
        for quantity, ax in enumerate(axes[1:]):
            ax.scatter(
                [index for index, _ in available],
                [result[quantity][rank] for _, result in available],
                color=f"C{rank % 10}",
                label=f"Frequency rank {rank + 1}",
            )
    for ax, ylabel in zip(
        axes[1:],
        ("Nominal frequency [Hz]", "Damping ratio [-]", "Band utilization"),
        strict=True,
    ):
        ax.set_ylabel(ylabel)
    for ax in axes:
        if invalid_ids:
            ax.scatter(
                invalid_ids,
                np.full(len(invalid_ids), 0.95),
                transform=ax.get_xaxis_transform(),
                color="red",
                marker="x",
                label="Unstable/marginal path (excluded)",
            )
        ax.grid(True)
    if max_rank or invalid_ids:
        axes[1].legend()
    axes[-1].set_xlabel("Trajectory number")
    fig.suptitle(f"Random waypoint paths; seed={args.seed}; frozen sampled modes")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
