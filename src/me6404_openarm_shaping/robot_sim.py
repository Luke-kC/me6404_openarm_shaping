import argparse
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from numpy.typing import NDArray

from me6404_openarm_shaping.configuration import (
    RobotConfig,
    TvistEIConfig,
    experiment_fingerprint,
    load_robot_config,
)
from me6404_openarm_shaping.data_types import (
    FloatMatrix,
    FloatVector,
    JointTrajectory,
    ModalProperties,
    RunProvenance,
    ShaperDesign,
    TrialRecording,
)
from me6404_openarm_shaping.input_shaping import (
    ZV_FREQUENCY_TOLERANCE,
    ZVD_FREQUENCY_TOLERANCE,
    design_ei_shaper,
    design_zv_shaper,
    design_zvd_shaper,
    ei_frequency_tolerance,
    suggest_shaper_parameters,
)
from me6404_openarm_shaping.mode_tracking import track_modes
from me6404_openarm_shaping.robot_model import (
    MODEL_PATH,
    MODEL_REVISION,
    ModalWorkspace,
    check_joint_configuration,
    load_model,
    modal_properties,
    set_added_armature,
    set_passive_damping,
)
from me6404_openarm_shaping.simulation_helpers import (
    capture_run_provenance,
    plot_comparison,
    run_trial,
    save_experiment,
)
from me6404_openarm_shaping.simulation_safety import print_simulation_checks
from me6404_openarm_shaping.trajectories import (
    apply_shaper_to_joint_trajectory,
    waypoint_trajectory,
)
from me6404_openarm_shaping.tvist import (
    TvistShaper,
    apply_tvist_to_joint_trajectory,
    select_tvist_modes,
)
from me6404_openarm_shaping.tvist_selection import (
    RMS_WINDOW_S,
    SELECTION_OBJECTIVES,
    build_subset_candidates,
    evaluate_subset_candidates,
    response_scores,
)


@dataclass
class SimulationResult:
    trajectories: dict[str, JointTrajectory]
    recordings: dict[str, TrialRecording]
    shaper_designs: dict[str, ShaperDesign]
    duration_s: float
    tvist_schedules: dict[str, list[ModalProperties]]
    tvist_selection: dict[str, object] = field(default_factory=dict)


def simulate(
    model: mujoco.MjModel, config: RobotConfig, *, headless: bool
) -> SimulationResult:
    """Run the baseline and enabled shapers for a single waypoint path."""
    Kp = config.motor_kp
    Kd = config.motor_kd

    simulation = config.simulation

    shaping = config.input_shaping

    model.opt.timestep = simulation.dt_s
    set_passive_damping(model, config.passive_damping)
    set_added_armature(model, config.added_armature)

    simulation_time: float = simulation.minimum_duration_s

    waypoints = config.waypoints

    qi = waypoints[0]

    for waypoint in waypoints:
        check_joint_configuration(model, waypoint)

    t: FloatVector
    q: FloatMatrix
    dq: FloatMatrix
    ddq: FloatMatrix
    t, q, dq, ddq = waypoint_trajectory(
        waypoints, config.velocity_limits, config.acceleration_limits, dt=model.opt.timestep
    )
    trajectories: dict[str, JointTrajectory] = {
        "baseline": {"t": t, "q": q, "dq": dq, "ddq": ddq}
    }
    shaper_designs: dict[str, ShaperDesign] = {}
    fixed_shapers = (
        ("zv", shaping.time_invariant.zv, ZV_FREQUENCY_TOLERANCE, design_zv_shaper),
        ("zvd", shaping.time_invariant.zvd, ZVD_FREQUENCY_TOLERANCE, design_zvd_shaper),
        (
            "ei",
            shaping.time_invariant.ei,
            ei_frequency_tolerance(shaping.time_invariant.ei.residual_tolerance),
            partial(
                design_ei_shaper,
                residual_tolerance=shaping.time_invariant.ei.residual_tolerance,
            ),
        ),
    )
    tvist_shapers = [
        (kind, settings)
        for kind, settings in (
            (TvistShaper.ZV, shaping.time_varying.zv),
            (TvistShaper.ZVD, shaping.time_varying.zvd),
            (TvistShaper.EI, shaping.time_varying.ei),
        )
        if settings.enabled
    ]
    tvist_schedules: dict[str, list[ModalProperties]] = {}
    experimental = shaping.time_varying.experimental
    properties: list[ModalProperties] = []
    if any(settings.enabled for _, settings, _, _ in fixed_shapers) or (
        tvist_shapers and not experimental
    ):
        workspace = ModalWorkspace(model)
        properties = [
            modal_properties(model, configuration, Kp, Kd, workspace=workspace)
            for configuration in q
        ]

    for kind, settings, tolerance, design in fixed_shapers:
        if not settings.enabled:
            continue
        label = f"time_invariant_{kind}"
        # One fixed design from every sample across all baseline segments.
        frequencies_hz: FloatVector
        damping: FloatVector
        utilization: FloatVector
        frequencies_hz, damping, utilization = suggest_shaper_parameters(
            properties,
            tau_min=settings.tau_min_s,
            frequency_tolerance=tolerance,
            utilization_min=settings.utilization_min,
        )
        print(f"\n----- {label.upper()} SHAPER DESIGN -----\n")
        print("Suggested shaper natural frequencies [Hz]:", frequencies_hz)
        print("Suggested shaper damping ratios:", damping)
        print("Suggested frequency-band utilization:", utilization)
        print()
        amplitudes: FloatVector
        delays_s: FloatVector
        amplitudes, delays_s = design(frequencies_hz, damping)
        delay_steps: NDArray[np.intp] = np.rint(delays_s / model.opt.timestep).astype(
            np.intp
        )
        shaper_designs[label] = {
            "type": kind,
            "frequency_tolerance": tolerance,
            "natural_frequencies_hz": frequencies_hz.tolist(),
            "damping_ratios": damping.tolist(),
            "band_utilization": utilization.tolist(),
            "amplitudes": amplitudes.tolist(),
            "ideal_delays_s": delays_s.tolist(),
            "rounded_delay_steps": delay_steps.tolist(),
            "applied_delays_s": (delay_steps * model.opt.timestep).tolist(),
        }
        if kind == "ei":
            shaper_designs[label]["residual_tolerance"] = (
                shaping.time_invariant.ei.residual_tolerance
            )
        t_shaped: FloatVector
        q_shaped: FloatMatrix
        dq_shaped: FloatMatrix
        ddq_shaped: FloatMatrix
        t_shaped, q_shaped, dq_shaped, ddq_shaped = apply_shaper_to_joint_trajectory(
            q, dq, ddq, amplitudes, delays_s, model.opt.timestep
        )
        trajectories[label] = {
            "t": t_shaped,
            "q": q_shaped,
            "dq": dq_shaped,
            "ddq": ddq_shaped,
        }

    dt = model.opt.timestep
    for shaper, settings in (() if experimental else tvist_shapers):
        label = f"tvist_{shaper.value.lower()}"
        schedule = select_tvist_modes(properties, settings.tau_min_s)
        apply = (
            partial(
                apply_tvist_to_joint_trajectory,
                residual_tolerance=settings.residual_tolerance,
            )
            if isinstance(settings, TvistEIConfig)
            else apply_tvist_to_joint_trajectory
        )
        q_tvist = apply(
            q=q, dt=dt, properties=schedule, shaper_type=shaper, lagrange_order=1
        )
        # The filter holds the last modal sample throughout its appended tail.
        tvist_schedules[label] = schedule + [schedule[-1]] * (
            len(q_tvist) - len(schedule)
        )
        dq_tvist = np.gradient(q_tvist, dt, axis=0, edge_order=2)
        trajectories[label] = {
            "t": np.arange(len(q_tvist)) * dt,
            "q": q_tvist,
            "dq": dq_tvist,
            "ddq": np.gradient(dq_tvist, dt, axis=0, edge_order=2),
        }

    simulation_time = max(
        simulation_time,
        max(trajectory["t"][-1] for trajectory in trajectories.values())
        + simulation.settling_time_s,
    )

    recordings: dict[str, TrialRecording] = {}
    ranked_recordings: dict[str, TrialRecording] = {}
    selection: dict[str, object] = {}
    if experimental and tvist_shapers:
        tracked = track_modes(model, q, Kp, Kd)
        by_shaper = {
            shaper: build_subset_candidates(q, config, tracked, shaper, settings)
            for shaper, settings in tvist_shapers
        }
        # Compare all survivors on a common horizon, including a full RMS window.
        max_end = max(
            (
                float(candidate.trajectory["t"][-1])
                for candidates in by_shaper.values()
                for candidate in candidates
                if candidate.trajectory is not None
            ),
            default=0.0,
        )
        max_end = max(max_end, max(float(command["t"][-1]) for command in trajectories.values()))
        simulation_time = max(
            simulation_time, max_end + max(RMS_WINDOW_S + dt, simulation.settling_time_s)
        )
        fixed_scores: list[tuple[float | None, float | None]] = []
        for label in ("time_invariant_zvd", "time_invariant_ei"):
            if label in trajectories:
                command = trajectories[label]
                recording = run_trial(
                    model, command, qi, Kp, Kd, simulation_time, config=config, headless=True
                )
                ranked_recordings[label] = recording
                fixed_scores.append(response_scores(recording, float(command["t"][-1])))
        fixed_rms = fixed_settling = None
        if len(fixed_scores) == 2:
            (r_zvd, t_zvd), (r_ei, t_ei) = fixed_scores
            if r_zvd is not None and r_ei is not None:
                fixed_rms = min(r_zvd, r_ei)
            if t_zvd is not None and t_ei is not None:
                fixed_settling = min(t_zvd, t_ei)
        for shaper, candidates in by_shaper.items():
            prefix = f"tvist_{shaper.value.lower()}"
            winners, rows = evaluate_subset_candidates(
                model, config, candidates, simulation_time, fixed_rms, fixed_settling
            )
            selected: dict[str, list[int] | None] = {}
            for objective in SELECTION_OBJECTIVES:
                label = f"{prefix}_{objective}"
                if objective not in winners:
                    print(f"{label}: no eligible winner")
                    selected[objective] = None
                    continue
                candidate, recording = winners[objective]
                command = candidate.trajectory
                assert command is not None
                selected[objective] = list(candidate.branches)
                print(f"{label}: selected branches {candidate.branches}")
                trajectories[label] = command
                tvist_schedules[label] = candidate.schedule + [candidate.schedule[-1]] * (
                    len(command["t"]) - len(candidate.schedule)
                )
                ranked_recordings[label] = recording
            selection[prefix] = {
                "candidates": rows, "selected_branches": selected,
                "fixed_rms_rad": fixed_rms, "fixed_settling_s": fixed_settling,
            }

    for label, trajectory in trajectories.items():
        print(f"\nRunning {label} trial\n")
        if headless and label in ranked_recordings:
            result = ranked_recordings[label]
        else:
            result = run_trial(
                model,
                trajectory,
                qi,
                Kp,
                Kd,
                simulation_time,
                config=config,
                headless=headless,
            )
        recordings[label] = result
        print_simulation_checks(label, result["simulation_checks"])
        if not result["completed"]:
            print(
                "Viewer closed early; stopping comparison and retaining collected data."
            )
            break

    return SimulationResult(
        trajectories,
        recordings,
        shaper_designs,
        simulation_time,
        {
            label: schedule
            for label, schedule in tvist_schedules.items()
            if label in recordings
        },
        selection,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate and plot the left-arm trajectory."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="save results and metadata in a new uniquely named folder in sim_out/",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="skip the MuJoCo viewer and real-time pacing; show plots at the end",
    )
    args = parser.parse_args()
    provenance: RunProvenance | None = (
        capture_run_provenance() if args.save_csv else None
    )
    model = load_model()
    config = load_robot_config()
    model_sha256 = (
        provenance["model_sha256"]
        if provenance is not None
        else sha256(MODEL_PATH.read_bytes()).hexdigest()
    )
    configuration_sha256 = experiment_fingerprint(config, model_sha256, MODEL_REVISION)
    print(f"\nExperiment configuration: {configuration_sha256}\n")
    result = simulate(model, config, headless=args.headless)

    if provenance is not None and any(
        len(trial["t"]) for trial in result.recordings.values()
    ):
        output_dir: Path = Path(__file__).resolve().parents[2] / "sim_out"
        run_dir: Path = save_experiment(
            result.recordings,
            config,
            result.shaper_designs,
            result.duration_s,
            output_dir,
            provenance=provenance,
            configuration_fingerprint=configuration_sha256,
            tvist_schedules=result.tvist_schedules,
            tvist_selection=result.tvist_selection,
        )
        print("\nSaved simulation results and metadata:")
        print(run_dir)
        print()

    plot_comparison(result.trajectories, result.recordings)
    plt.show()


if __name__ == "__main__":
    main()
