import argparse
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from numpy.typing import NDArray

from me6404_openarm_shaping.configuration import (
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
    design_zvd_shaper,
    suggest_zvd_parameters,
)
from me6404_openarm_shaping.robot_model import (
    MODEL_PATH,
    MODEL_REVISION,
    check_joint_configuration,
    eval_gravity_torques,
    load_model,
    modal_properties,
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


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Simulate and plot the left-arm trajectory."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="save results and metadata in a new uniquely named folder in sim_out/",
    )
    args: argparse.Namespace = parser.parse_args()
    provenance: RunProvenance | None = (
        capture_run_provenance() if args.save_csv else None
    )

    model: mujoco.MjModel = load_model()
    config = load_robot_config()
    model_sha256 = (
        provenance["model_sha256"]
        if provenance is not None
        else sha256(MODEL_PATH.read_bytes()).hexdigest()
    )
    configuration_sha256 = experiment_fingerprint(config, model_sha256, MODEL_REVISION)
    print(f"\nExperiment configuration: {configuration_sha256}\n")

    Kp = config.motor_kp
    Kd = config.motor_kd

    simulation = config.simulation

    shaping = config.input_shaping

    model.opt.timestep = simulation.dt_s

    simulation_time: float = simulation.minimum_duration_s

    waypoints = config.waypoints

    qi = waypoints[0]

    for waypoint in waypoints:
        check_joint_configuration(model, waypoint)
    qf = waypoints[-1]

    v_lim = config.velocity_limits
    a_lim = config.acceleration_limits

    t: FloatVector
    q: FloatMatrix
    dq: FloatMatrix
    ddq: FloatMatrix
    t, q, dq, ddq = waypoint_trajectory(waypoints, v_lim, a_lim, dt=model.opt.timestep)
    trajectories: dict[str, JointTrajectory] = {
        "unshaped": {"t": t, "q": q, "dq": dq, "ddq": ddq}
    }
    shaper_design: ShaperDesign | None = None

    if shaping.enabled:
        # One fixed design from every sample across all unshaped segments.
        properties: list[ModalProperties] = [
            modal_properties(model, configuration, Kp, Kd) for configuration in q
        ]

        frequencies_hz: FloatVector
        damping: FloatVector
        utilization: FloatVector
        frequencies_hz, damping, utilization = suggest_zvd_parameters(
            properties,
            tau_min=shaping.zvd.tau_min_s,
            utilization_min=shaping.zvd.utilization_min,
        )
        print("\n----- SHAPER DESIGN -----\n")
        print("Suggested shaper natural frequencies [Hz]:", frequencies_hz)
        print("Suggested shaper damping ratios:", damping)
        print("Suggested frequency-band utilization:", utilization)
        print()
        amplitudes: FloatVector
        delays_s: FloatVector
        amplitudes, delays_s = design_zvd_shaper(frequencies_hz, damping)
        delay_steps: NDArray[np.intp] = np.rint(delays_s / model.opt.timestep).astype(
            np.intp
        )
        shaper_design = {
            "type": shaping.type,
            "natural_frequencies_hz": frequencies_hz.tolist(),
            "damping_ratios": damping.tolist(),
            "band_utilization": utilization.tolist(),
            "amplitudes": amplitudes.tolist(),
            "ideal_delays_s": delays_s.tolist(),
            "rounded_delay_steps": delay_steps.tolist(),
            "applied_delays_s": (delay_steps * model.opt.timestep).tolist(),
        }
        t_shaped: FloatVector
        q_shaped: FloatMatrix
        dq_shaped: FloatMatrix
        ddq_shaped: FloatMatrix
        t_shaped, q_shaped, dq_shaped, ddq_shaped = apply_shaper_to_joint_trajectory(
            q, dq, ddq, amplitudes, delays_s, model.opt.timestep
        )
        trajectories["shaped"] = {
            "t": t_shaped,
            "q": q_shaped,
            "dq": dq_shaped,
            "ddq": ddq_shaped,
        }

    simulation_time = max(
        simulation_time,
        max(trajectory["t"][-1] for trajectory in trajectories.values())
        + simulation.settling_time_s,
    )

    recordings: dict[str, TrialRecording] = {}

    for label, trajectory in trajectories.items():
        print(f"\nRunning {label} trial\n")
        result = run_trial(
            model, trajectory, qi, Kp, Kd, simulation_time, config=config
        )
        recordings[label] = result
        print_simulation_checks(label, result["simulation_checks"])
        if not result["completed"]:
            print(
                "Viewer closed early; stopping comparison and retaining collected data."
            )
            break

    if provenance is not None and any(
        len(result["t"]) for result in recordings.values()
    ):
        output_dir: Path = Path(__file__).resolve().parents[2] / "sim_out"
        run_dir: Path = save_experiment(
            recordings,
            config,
            shaper_design,
            simulation_time,
            output_dir,
            provenance=provenance,
            configuration_fingerprint=configuration_sha256,
        )
        print("\nSaved simulation results and metadata:")
        print(run_dir)
        print()

    plot_comparison(trajectories, recordings)
    plt.show()


if __name__ == "__main__":
    main()
