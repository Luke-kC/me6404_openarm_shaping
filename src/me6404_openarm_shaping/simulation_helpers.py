"""Small helpers for sequential offline simulation comparisons."""

import csv
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal, cast
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import yaml
from matplotlib.axes import Axes
from numpy.typing import NDArray

from me6404_openarm_shaping.configuration import RobotConfig, robot_config_to_dict
from me6404_openarm_shaping.data_types import (
    ContactEvent,
    FloatMatrix,
    FloatVector,
    GitState,
    JointTrajectory,
    ModalProperties,
    RunProvenance,
    ShaperDesign,
    TrialRecording,
    TrialSamples,
)
from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    MODEL_PATH,
    MODEL_REVISION,
    ModalWorkspace,
    eval_gravity_torques,
    resolve_joint_indices,
)
from me6404_openarm_shaping.simulation_safety import (
    assess_trial,
    record_penetrating_contacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_trial(
    model: mujoco.MjModel,
    trajectory: JointTrajectory,
    qi: FloatVector,
    Kp: FloatVector,
    Kd: FloatVector,
    simulation_time: float,
    *,
    config: RobotConfig,
    headless: bool = False,
) -> TrialRecording:
    """Run one trial with fresh data; headless mode skips rendering and pacing.

    Preserve the existing controller:
    PD with zero desired velocity and gravity feedforward based on commanded joint position.

    Only the selected arm is commanded.
    """
    data = mujoco.MjData(model)
    position_indices, velocity_indices = resolve_joint_indices(model)
    actuator_indices = np.array(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_joint{i}_ctrl"
            )
            for i in range(1, 8)
        ],
        dtype=np.intp,
    )
    if np.any(actuator_indices < 0):
        raise ValueError("Could not resolve all seven left-arm motor actuators.")
    if not np.all(model.actuator_forcelimited[actuator_indices]):
        raise ValueError("Expected force-limited left-arm motor actuators.")

    torque_limits = model.actuator_forcerange[actuator_indices]
    data.qpos[position_indices] = qi
    mujoco.mj_forward(model, data)
    contact_events: dict[tuple[int, ...], ContactEvent] = {}
    record_penetrating_contacts(model, data, contact_events, data.time)

    times: list[float] = []
    angles: list[NDArray[np.float64]] = []
    measured_velocities: list[NDArray[np.float64]] = []
    measured_accelerations: list[NDArray[np.float64]] = []
    requested: list[NDArray[np.float64]] = []
    feedback: list[NDArray[np.float64]] = []
    gravity_feedforward: list[NDArray[np.float64]] = []
    commanded_positions: list[NDArray[np.float64]] = []
    reference_velocities: list[NDArray[np.float64]] = []
    reference_accelerations: list[NDArray[np.float64]] = []
    step_count = int(np.ceil(simulation_time / model.opt.timestep))

    # Feedforward depends only on the known commanded pose, not measured state.
    gravity_workspace = ModalWorkspace(model)
    trajectory_gravity = np.empty_like(trajectory["q"])
    for index, q_command in enumerate(trajectory["q"]):
        trajectory_gravity[index] = eval_gravity_torques(
            model, q_command, workspace=gravity_workspace
        )

    viewer_context = (
        nullcontext() if headless else mujoco.viewer.launch_passive(model, data)
    )
    with viewer_context as viewer:
        if viewer is not None:
            # Start each trial framed on the left arm; mouse controls remain free.
            shoulder = data.body("openarm_left_link1").xpos
            hand = data.body("openarm_left_hand").xpos
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = (shoulder + hand) / 2
                viewer.cam.distance = 1.2
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -20.0

        step_index = 0
        step_deadline = time.perf_counter()
        next_sync = step_deadline
        sync_interval = 1.0 / 60.0
        while step_index < step_count and (viewer is None or viewer.is_running()):
            target_index = min(step_index, len(trajectory["q"]) - 1)
            q_measured = data.qpos[position_indices]
            dq_measured = data.qvel[velocity_indices]

            q_command = trajectory["q"][target_index]

            gravity_ff = trajectory_gravity[target_index]

            torque = (
                Kp * (trajectory["q"][target_index] - q_measured)
                - Kd * dq_measured
                + gravity_ff
            )
            if not np.isfinite(torque).all():
                raise ValueError("Nonfinite left-arm torque; stopping simulation.")
            data.ctrl[actuator_indices] = np.clip(
                torque, torque_limits[:, 0], torque_limits[:, 1]
            )

            # Keep pre-integration positions aligned with actuator-force samples.
            sample_time = data.time
            sample_angles = q_measured.copy()
            sample_velocities = dq_measured.copy()
            q_command = trajectory["q"][target_index].copy()
            # Log generated derivatives, not controller feedforward inputs.
            if step_index < len(trajectory["q"]):
                dq_reference = trajectory["dq"][target_index].copy()
                ddq_reference = trajectory["ddq"][target_index].copy()
            else:
                dq_reference = np.zeros(7)
                ddq_reference = np.zeros(7)
            mujoco.mj_step(model, data)
            # For the pinned model's Euler integrator, qacc corresponds to the
            # pre-integration state and applied control at sample_time.
            sample_accelerations = data.qacc[velocity_indices].copy()
            # mj_step's contacts describe its pre-integration state. The initial
            # state was already checked above; avoid counting it twice.
            if step_index > 0:
                record_penetrating_contacts(model, data, contact_events, sample_time)
            times.append(sample_time)
            angles.append(sample_angles)
            measured_velocities.append(sample_velocities)
            measured_accelerations.append(sample_accelerations)
            requested.append(torque.copy())
            feedback.append(data.actuator_force[actuator_indices].copy())
            gravity_feedforward.append(gravity_ff.copy())
            commanded_positions.append(q_command)
            reference_velocities.append(dq_reference)
            reference_accelerations.append(ddq_reference)
            step_index += 1

            if viewer is not None:
                now = time.perf_counter()
                if now >= next_sync:
                    viewer.sync()
                    next_sync = now + sync_interval
                step_deadline += model.opt.timestep
                sleep_time = step_deadline - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # Refresh contacts at the last integrated state, absent from the CSV's
    # pre-integration position rows. This does not advance simulation time.
    mujoco.mj_forward(model, data)
    if step_index > 0:
        record_penetrating_contacts(model, data, contact_events, data.time)
    samples: TrialSamples = {
        "t": np.asarray(times),
        "q": np.asarray(angles).reshape(-1, 7),
        "dq_measured": np.asarray(measured_velocities).reshape(-1, 7),
        "ddq_measured": np.asarray(measured_accelerations).reshape(-1, 7),
        "torque_requested": np.asarray(requested).reshape(-1, 7),
        "completed": step_index == step_count,
        "final_time_s": float(data.time),
        "final_q": data.qpos[position_indices].copy(),
        "final_dq": data.qvel[velocity_indices].copy(),
        "final_ddq": data.qacc[velocity_indices].copy(),
        "contact_events": list(contact_events.values()),
    }
    command_end_time_s = float(cast(np.float64, trajectory["t"][-1]))
    return {
        **samples,
        "torque_feedback": np.asarray(feedback).reshape(-1, 7),
        "torque_gravity_ff": np.asarray(gravity_feedforward).reshape(-1, 7),
        "q_command": np.asarray(commanded_positions).reshape(-1, 7),
        "dq_reference": np.asarray(reference_velocities).reshape(-1, 7),
        "ddq_reference": np.asarray(reference_accelerations).reshape(-1, 7),
        "simulation_checks": assess_trial(
            model,
            samples,
            config,
            command_end_time_s=command_end_time_s,
            trajectory=trajectory,
        ),
    }


def save_recordings_csv(
    recordings: dict[str, TrialRecording], output_path: Path
) -> Path:
    """Write all recorded trials to one labeled, full-resolution CSV."""
    columns = ["trajectory", "time_s"]
    for quantity in (
        "q_rad",
        "torque_requested_Nm",
        "torque_feedback_Nm",
        "q_command_rad",
        "dq_reference_rad_s",
        "ddq_reference_rad_s2",
        "dq_measured_rad_s",
        "ddq_measured_rad_s2",
        "torque_gravity_ff_Nm",
    ):
        columns.extend(f"J{i}_{quantity}" for i in range(1, 8))
    with output_path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for label, result in recordings.items():
            values = np.column_stack(
                (
                    result["t"],
                    result["q"],
                    result["torque_requested"],
                    result["torque_feedback"],
                    result["q_command"],
                    result["dq_reference"],
                    result["ddq_reference"],
                    result["dq_measured"],
                    result["ddq_measured"],
                    result["torque_gravity_ff"],
                )
            )
            for row in values:
                writer.writerow([label, *row])
    return output_path


def capture_run_provenance(
    project_root: Path = PROJECT_ROOT,
    model_path: Path = MODEL_PATH,
) -> RunProvenance:
    """Capture read-only provenance and lockfile bytes before running a trial."""
    lockfile_path = project_root / "uv.lock"
    lockfile_bytes = lockfile_path.read_bytes()
    model_hash = sha256(model_path.read_bytes()).hexdigest()
    git_info: GitState = {"commit_sha": None, "dirty": None}
    errors: dict[str, str] = {}
    for key, arguments in (
        ("commit_sha", ["git", "rev-parse", "HEAD"]),
        (
            "dirty",
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--ignore-submodules=none",
            ],
        ),
    ):
        try:
            result = subprocess.run(
                arguments, cwd=project_root, capture_output=True, text=True, check=True
            )
            if key == "dirty":
                git_info["dirty"] = bool(result.stdout.strip())
            else:
                git_info["commit_sha"] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            errors[key] = (
                error.stderr.strip()
                if isinstance(error, subprocess.CalledProcessError) and error.stderr
                else str(error)
            )
    if errors:
        git_info["errors"] = errors
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "project_git": git_info,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "lockfile_source": str(lockfile_path),
        "lockfile_sha256": sha256(lockfile_bytes).hexdigest(),
        "lockfile_bytes": lockfile_bytes,
    }


def save_experiment(
    recordings: dict[str, TrialRecording],
    config: RobotConfig,
    shaper_designs: dict[str, ShaperDesign],
    simulation_time: float,
    output_dir: Path,
    *,
    provenance: RunProvenance,
    configuration_fingerprint: str,
    tvist_schedules: dict[str, list[ModalProperties]] | None = None,
    directory_name: str | None = None,
) -> Path:
    """Save paired result/metadata files in an atomically unique run directory.

    Snapshot the config object actually loaded at startup; do not reread YAML.
    Exact reproduction also requires matching source code and dependencies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now(UTC)
    local_saved_at = saved_at.astimezone(ZoneInfo("America/New_York"))
    prefix = f"run_{local_saved_at.strftime('%Y%m%d_%H%M%S_%f')}_"
    if directory_name is None:
        run_dir = Path(mkdtemp(prefix=prefix, dir=output_dir))
    else:
        # Batch callers supply a trajectory ID inside an already unique batch.
        run_dir = output_dir / directory_name
        run_dir.mkdir()
    run_id = run_dir.name
    results_path = run_dir / f"{run_id}_results.csv"
    metadata_path = run_dir / f"{run_id}_metadata.yaml"
    lockfile_path = run_dir / f"{run_id}_uv.lock"
    config_snapshot = robot_config_to_dict(config)
    metadata = {
        "run_id": run_id,
        "saved_at_utc": saved_at.isoformat(),
        "results_file": results_path.name,
        "config": config_snapshot,
        "experiment_configuration_sha256": configuration_fingerprint,
        "shaper_designs": shaper_designs,
        "simulation_duration_requested_s": float(simulation_time),
        "trials": {
            label: {
                "samples": len(result["t"]),
                "completed": bool(result["completed"]),
                "recorded_duration_s": len(result["t"]) * config.simulation.dt_s,
                "simulation_checks": result["simulation_checks"],
            }
            for label, result in recordings.items()
        },
        "provenance_captured_at_utc": provenance["captured_at_utc"],
        "project_git": provenance["project_git"],
        "model": {
            "path": provenance["model_path"],
            "upstream_revision": MODEL_REVISION,
            "sha256": provenance["model_sha256"],
        },
        "dependency_lockfile": {
            "file": lockfile_path.name,
            "source": provenance["lockfile_source"],
            "sha256": provenance["lockfile_sha256"],
        },
        "joint_order": list(LEFT_ARM_JOINT_NAMES),
        "units": {
            "position": "rad",
            "velocity": "rad/s",
            "acceleration": "rad/s^2",
            "torque": "N*m",
            "time": "s",
            "natural_frequency": "Hz",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
    }
    if tvist_schedules:
        modes_path = run_dir / "tvist_modes.csv"
        metadata["tvist_modes_file"] = modes_path.name
        metadata["tvist_mode_count_histograms"] = {
            label: dict(
                sorted(
                    Counter(
                        len(sample["natural_frequency_hz"]) for sample in schedule
                    ).items()
                )
            )
            for label, schedule in tvist_schedules.items()
        }
        with modes_path.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["trajectory", "time_s", "frequency_rank", "natural_frequency_hz", "damping_ratio"]
            )
            for label, schedule in tvist_schedules.items():
                for index, sample in enumerate(schedule):
                    time_s = index * config.simulation.dt_s
                    frequencies = sample["natural_frequency_hz"]
                    if not len(frequencies):
                        writer.writerow([label, time_s, None, None, None])
                    for rank, (frequency, damping) in enumerate(
                        zip(frequencies, sample["damping_ratio"], strict=True), start=1
                    ):
                        writer.writerow([label, time_s, rank, frequency, damping])
    with lockfile_path.open("xb") as stream:
        stream.write(provenance["lockfile_bytes"])
    save_recordings_csv(recordings, results_path)
    with metadata_path.open("x") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return run_dir


def _plot_joint_panels(
    series: Sequence[tuple[str, FloatVector, FloatMatrix, str, str]],
    title: str,
    ylabel: str,
) -> None:
    """Plot (label, time, values, color, linestyle) series, one panel per joint."""
    fig, axes = plt.subplots(4, 2, sharex=True, figsize=(12, 10), num=title)
    panels = cast(Sequence[Axes], axes.flat)
    for joint, ax in enumerate(panels):
        if joint == 7:
            ax.set_visible(False)
            continue
        for label, times, values, color, linestyle in series:
            if not len(times):
                continue
            # Decimate only display data, never the recordings or CSV.
            points = np.linspace(0, len(times) - 1, min(len(times), 500), dtype=int)
            ax.plot(
                times[points],
                values[points, joint],
                color=color,
                linestyle=linestyle,
                label=label,
            )
        ax.set_title(f"J{joint + 1}")
        ax.set_ylabel(ylabel)
        ax.grid(True)
    panels[0].legend()
    fig.suptitle(title)
    fig.supxlabel("Time [s]")
    fig.tight_layout()


def plot_comparison(
    trajectories: dict[str, JointTrajectory], recordings: dict[str, TrialRecording]
) -> None:
    """Build commanded and recorded comparisons without launching a viewer."""
    colors = {
        "baseline": "tab:blue",
        "time_invariant_zv": "tab:red",
        "time_invariant_zvd": "tab:orange",
        "tvist_zv": "tab:purple",
        "tvist_zvd": "tab:green",
    }
    quantities: tuple[tuple[Literal["q", "dq", "ddq"], str, str], ...] = (
        ("q", "Commanded position", "Position [rad]"),
        ("dq", "Commanded velocity", "Velocity [rad/s]"),
        ("ddq", "Commanded acceleration", "Acceleration [rad/s²]"),
    )
    series: list[tuple[str, FloatVector, FloatMatrix, str, str]]
    for key, title, units in quantities:
        series = [
            (label, values["t"], values[key], colors[label], "-")
            for label, values in trajectories.items()
        ]
        _plot_joint_panels(series, title, units)

    if not any(len(values["t"]) for values in recordings.values()):
        return
    series = [
        (label, values["t"], values["q"], colors[label], "-")
        for label, values in recordings.items()
    ]
    _plot_joint_panels(series, "Recorded position", "Position [rad]")
    series = []
    for label, values in recordings.items():
        series.extend(
            [
                (
                    f"{label} requested",
                    values["t"],
                    values["torque_requested"],
                    colors[label],
                    "--",
                ),
                (
                    f"{label} applied",
                    values["t"],
                    values["torque_feedback"],
                    colors[label],
                    "-",
                ),
            ]
        )
    _plot_joint_panels(series, "Motor torque", "Torque [N·m]")
