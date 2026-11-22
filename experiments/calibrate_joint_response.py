"""Replay a recorded single-joint OpenArm move and sweep damping or armature.

From the repo root, source ROS Jazzy, then run this module with the project
Python environment. Reading the MCAP files does not start a ROS node.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from me6404_openarm_shaping.configuration import load_robot_config
from me6404_openarm_shaping.data_types import FloatMatrix, FloatVector, JointTrajectory
from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    load_model,
    resolve_joint_indices,
)
from me6404_openarm_shaping.simulation_helpers import run_trial

ROOT = Path(__file__).resolve().parents[1] / "hardware_out/unshaped/model_calibration"
STATE = "/left_joint_trajectory_controller/controller_state"
KP = "/left_motor_gains/kp"
KD = "/left_motor_gains/kd"


@dataclass
class Replay:
    time: FloatVector
    output_q: FloatMatrix
    effort: FloatMatrix
    reference_dq: FloatMatrix
    reference_ddq: FloatMatrix
    measured_q: FloatMatrix
    measured_dq: FloatMatrix
    kp: FloatVector
    kd: FloatVector
    motion_end_s: float


def read_replay(run: Path, joint: int | None = None) -> Replay:
    """Load recorded motor inputs; with no joint, use all joints to find motion end."""
    # Optional ROS dependencies are needed only by this offline experiment.
    try:
        import rosbag2_py as bag
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "Source /opt/ros/jazzy/setup.bash or setup.zsh first."
        ) from exc

    trial = json.loads((run / "trial.json").read_text())
    goal_duration = trial["trajectory_duration_s"]
    reader = bag.SequentialReader()
    reader.open(
        bag.StorageOptions(uri=str(run / "bag"), storage_id="mcap"),
        bag.ConverterOptions("", ""),
    )
    types = {
        topic.name: get_message(topic.type)
        for topic in reader.get_all_topics_and_types()
        if topic.name in (STATE, KP, KD)
    }
    reader.set_filter(bag.StorageFilter(topics=[STATE, KP, KD]))
    stamps: list[int] = []
    elapsed: list[float] = []
    rows: list[tuple[FloatVector, ...]] = []
    gains: dict[str, list[FloatVector]] = {KP: [], KD: []}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        msg = deserialize_message(raw, types[topic])
        if topic != STATE:
            gains[topic].append(np.asarray(msg.data, dtype=np.float64))
            continue
        if tuple(msg.joint_names) != LEFT_ARM_JOINT_NAMES:
            raise ValueError("Controller state joint order does not match left J1–J7.")
        stamp = msg.header.stamp
        stamps.append(stamp.sec * 10**9 + stamp.nanosec)
        duration = msg.reference.time_from_start
        elapsed.append(duration.sec + duration.nanosec * 1e-9)
        rows.append(
            tuple(
                np.asarray(field, dtype=np.float64)
                for field in (
                    msg.output.positions,
                    msg.output.effort,
                    msg.reference.velocities,
                    msg.reference.accelerations,
                    msg.feedback.positions,
                    msg.feedback.velocities,
                )
            )
        )

    # Controller trajectory time resets on goal acceptance; the previous
    # hold in these recordings has a timestamp older than the new goal duration.
    resets = [
        i
        for i in range(1, len(elapsed))
        if elapsed[i - 1] >= goal_duration and 0 <= elapsed[i] < 0.01
    ]
    if not resets or not goal_duration > 0:
        raise ValueError("Could not find the recorded goal start in controller state.")
    start = resets[0]
    stop = resets[1] if len(resets) > 1 else len(rows)
    time = (np.asarray(stamps[start:stop], dtype=np.int64) - stamps[start]) * 1e-9
    commands, effort, dq, ddq, measured, measured_dq = (
        np.stack(fields) for fields in zip(*rows[start:stop], strict=True)
    )
    if np.any(np.diff(time) <= 0) or any(
        x.shape != (len(time), 7) or not np.isfinite(x).all()
        for x in (commands, effort, dq, ddq, measured, measured_dq)
    ):
        raise ValueError("Replay requires increasing timestamps and finite J1–J7 data.")
    moving_joints = np.abs(dq) > 1e-5
    moving = np.flatnonzero(
        np.any(moving_joints, axis=1) if joint is None else moving_joints[:, joint]
    )
    if moving.size == 0 or moving[-1] + 1 >= len(time):
        raise ValueError("The selected recording has no complete move for this replay.")
    motion_end_s = float(time[moving[-1] + 1])
    kp, kd = (constant_gains(gains[topic], topic) for topic in (KP, KD))
    return Replay(
        time, commands, effort, dq, ddq, measured, measured_dq, kp, kd, motion_end_s
    )


def constant_gains(values: list[FloatVector], topic: str) -> FloatVector:
    if not values or any(
        v.shape != (7,)
        or not np.isfinite(v).all()
        or np.any(v < 0)
        or not np.array_equal(v, values[0])
        for v in values
    ):
        raise ValueError(f"Expected one consistent seven-joint gain vector on {topic}.")
    return values[0]


def ring_down_fit(
    time: FloatVector, position: FloatVector
) -> tuple[float, float, float]:
    """Fit offset + exp(-alpha*t)*(a*cos(w*t) + b*sin(w*t)) by grid search."""
    time = time - time[0]
    best = (float("inf"), 0.0, 0.0)
    for frequency in np.arange(0.75, 6.01, 0.05):
        phase = 2 * np.pi * frequency * time
        sin, cos = np.sin(phase), np.cos(phase)
        for decay in np.arange(0.0, 8.01, 0.25):
            envelope = np.exp(-decay * time)
            basis = np.column_stack(
                (np.ones_like(time), envelope * cos, envelope * sin)
            )
            weights = np.linalg.lstsq(basis, position, rcond=None)[0]
            error = float(np.mean((basis @ weights - position) ** 2))
            if error < best[0]:
                best = (error, float(frequency), float(decay))
    error, frequency, decay = best
    variance = float(np.var(position))
    return frequency, decay, 1 - error / variance if variance > 0 else 0.0


def add_joint_armature(model: mujoco.MjModel, joint: int, added: float) -> None:
    """Add effective inertia [kg·m²] to one left-arm hinge (zero-based index)."""
    _, dofs = resolve_joint_indices(model)
    model.dof_armature[dofs[joint]] += added
    mujoco.mj_setConst(model, mujoco.MjData(model))


def first_target_crossing(
    time: FloatVector, position: FloatVector, target: float, direction: float
) -> float | None:
    """Interpolate the first arrival at the final target in the move direction."""
    signed_error = direction * (position - target)
    brackets = np.flatnonzero((signed_error[:-1] < 0) & (signed_error[1:] >= 0))
    if not len(brackets):
        return None
    i = brackets[0]
    fraction = -signed_error[i] / (signed_error[i + 1] - signed_error[i])
    return float(time[i] + fraction * (time[i + 1] - time[i]))


def overshoot_index(
    time: FloatVector,
    position: FloatVector,
    target: float,
    direction: float,
    crossing: float,
    *,
    number: int = 1,
) -> int | None:
    """Find the extremum of the requested same-side overshoot lobe."""
    start = int(np.searchsorted(time, crossing, side="right"))
    error = direction * (position - target)
    for lobe in range(1, number + 1):
        returns = np.flatnonzero(error[start:] <= 0)
        if not len(returns) or returns[0] == 0:
            return None
        end = start + int(returns[0])
        if lobe == number:
            return start + int(np.argmax(error[start:end]))
        next_lobe = np.flatnonzero(error[end:] > 0)
        if not len(next_lobe):
            return None
        start = end + int(next_lobe[0])
    return None


def compare(
    joint: int,
    run: Path,
    candidates: list[float],
    ring_s: float,
    *,
    armature: bool,
    fixed_armature: float = 0.0,
) -> None:
    replay = read_replay(run, joint)
    config = load_robot_config()
    dt = config.simulation.dt_s
    end = replay.motion_end_s + ring_s
    if end + dt > replay.time[-1]:
        raise ValueError(
            "The hardware recording is too short for the requested ring-down window."
        )
    # Replay the most recent controller output on each MuJoCo tick, not a
    # fit to the measured motion. Hardware timestamps stay on their native grid.
    t = np.arange(int(np.ceil(end / dt)) + 1) * dt
    i = np.searchsorted(replay.time, t, side="right") - 1
    command: JointTrajectory = {
        "t": t,
        "q": replay.output_q[i],
        "dq": replay.reference_dq[i],
        "ddq": replay.reference_ddq[i],
    }
    nominal = 0.0 if armature else float(config.passive_damping[joint])
    by_joint = list(dict.fromkeys([nominal, *candidates]))
    full = replay.time <= end
    ring = full & (replay.time >= replay.motion_end_s)
    analysis_t = replay.time[full]
    measured = replay.measured_q[full]

    grid, axes = plt.subplots(4, 2, sharex=True, figsize=(12, 9), layout="constrained")
    ring_fig, ax_ring = plt.subplots(figsize=(9, 4), layout="constrained")
    peak_fig, ax_peak = plt.subplots(figsize=(9, 4), layout="constrained")
    grid.suptitle(
        f"J{joint + 1} excitation — controller command and measured positions"
    )
    ring_fig.suptitle(f"J{joint + 1} ring-down — position relative to final command")
    peak_number = 2 if joint == 1 else 1
    peak_name = "second" if peak_number == 2 else "first"
    peak_fig.suptitle(f"J{joint + 1} ring-down — aligned at {peak_name} overshoot")
    print(
        f"\n{run.name}: controller motion end {replay.motion_end_s:.3f} s; Kp={replay.kp}, Kd={replay.kd}"
    )
    if armature:
        print(
            f"Added armature in kg·m²; passive damping fixed at {config.passive_damping}"
        )
    else:
        print(
            f"Passive damping in N·m·s/rad; added armature fixed at {fixed_armature:g} kg·m²"
        )
    print("Damped frequency in Hz; decay in 1/s; RMS position error in rad")
    f, decay, fit = ring_down_fit(replay.time[ring], replay.measured_q[ring, joint])
    print(f"Hardware ring fit: {f:.2f} Hz, decay {decay:.2f}, R²={fit:.3f}")
    for j, ax in enumerate(axes.flat):
        if j == 7:
            ax.set_visible(False)
            break
        ax.plot(
            analysis_t, replay.output_q[full, j], ":", color="0.55", label="Command"
        )
        ax.plot(analysis_t, measured[:, j], color="black", label="Hardware")
        ax.axvline(replay.motion_end_s, color="0.75", linewidth=1)
        ax.set(ylabel=f"J{j + 1} [rad]")
        ax.grid(alpha=0.25)
    target = replay.output_q[i[-1], joint]
    direction = float(np.sign(target - replay.output_q[0, joint]))
    hw_crossing = first_target_crossing(
        analysis_t, measured[:, joint], target, direction
    )
    hw_peak = None
    if hw_crossing is not None:
        print(f"Hardware first crossing of {target:g} rad: t={hw_crossing:.4f} s")
        hw_peak = overshoot_index(
            analysis_t,
            measured[:, joint],
            target,
            direction,
            hw_crossing,
            number=peak_number,
        )
    if hw_peak is None:
        print("Hardware: no complete overshoot lobe; peak alignment unavailable.")
        plt.close(peak_fig)
    else:
        ax_peak.plot(
            analysis_t[hw_peak:] - analysis_t[hw_peak],
            measured[hw_peak:, joint] - target,
            color="black",
            label="Hardware",
        )
    ax_ring.plot(
        replay.time[ring] - replay.motion_end_s,
        replay.measured_q[ring, joint] - target,
        color="black",
        label="Hardware",
    )
    colors = plt.get_cmap("tab10").colors
    for index, value in enumerate(by_joint):
        candidate = config.passive_damping.copy()
        if not armature:
            candidate[joint] = value
        model = load_model(candidate)
        model.opt.timestep = dt
        added_inertia = value if armature else fixed_armature
        if added_inertia:
            add_joint_armature(model, joint, added_inertia)
        result = run_trial(
            model,
            command,
            replay.measured_q[0],
            replay.kp,
            replay.kd,
            t[-1] + dt,
            config=config,
            headless=True,
            initial_velocity=replay.measured_dq[0],
            gravity_feedforward_override=replay.effort[i],
        )
        sim = np.column_stack(
            [np.interp(analysis_t, result["t"], result["q"][:, j]) for j in range(7)]
        )
        sim_ring = sim[ring[full], joint]
        f, decay, fit = ring_down_fit(replay.time[ring], sim_ring)
        rms = np.sqrt(np.mean((sim - measured) ** 2, axis=0))
        ring_rms = np.sqrt(np.mean((sim_ring - replay.measured_q[ring, joint]) ** 2))
        setting = f"Added inertia={value:g}" if armature else f"D={value:g}"
        print(
            f"{setting}: {f:.2f} Hz, decay {decay:.2f}, R²={fit:.3f}; "
            f"ring RMS J{joint + 1}={ring_rms:.5f}; full J1–J7 RMS={np.round(rms, 5)}"
        )
        label = f"Sim {setting}" + (" (nominal)" if index == 0 else "")
        color = colors[index % len(colors)]
        for j, ax in enumerate(list(axes.flat)[:7]):
            ax.plot(analysis_t, sim[:, j], color=color, label=label)
        ax_ring.plot(
            replay.time[ring] - replay.motion_end_s,
            sim_ring - target,
            color=color,
            label=label,
        )
        native_window = result["t"] <= end
        sim_time = result["t"][native_window]
        sim_position = result["q"][native_window, joint]
        sim_crossing = first_target_crossing(sim_time, sim_position, target, direction)
        if sim_crossing is not None and hw_crossing is not None:
            offset_ms = 1000 * (hw_crossing - sim_crossing)
            print(
                f"  First crossing: t={sim_crossing:.4f} s; "
                f"hardware minus sim crossing time={offset_ms:+.1f} ms"
            )
        if hw_peak is not None:
            sim_peak = (
                overshoot_index(
                    sim_time,
                    sim_position,
                    target,
                    direction,
                    sim_crossing,
                    number=peak_number,
                )
                if sim_crossing is not None
                else None
            )
            if sim_peak is None:
                print(
                    f"  {setting}: no complete overshoot lobe; omitted from peak plot."
                )
            else:
                ax_peak.plot(
                    sim_time[sim_peak:] - sim_time[sim_peak],
                    sim_position[sim_peak:] - target,
                    color=color,
                    label=label,
                )
    grid.supxlabel("Time since first controller sample of the recorded goal [s]")
    axes[0, 0].legend(fontsize="small")
    ax_ring.set(
        xlabel="Time since reference motion ended [s]", ylabel="Position error [rad]"
    )
    ax_ring.grid(alpha=0.25)
    ax_ring.legend()
    if hw_peak is not None:
        ax_peak.set(
            xlabel=f"Time since {peak_name} overshoot peak [s]",
            ylabel="Position error [rad]",
        )
        ax_peak.grid(alpha=0.25)
        ax_peak.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joints", type=int, choices=range(1, 8), nargs="+", default=[1, 2]
    )
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument(
        "--damping", type=float, nargs="+", help="absolute passive damping in N·m·s/rad"
    )
    sweep.add_argument(
        "--armature",
        type=float,
        nargs="*",
        help="added joint inertia in kg·m²; omit values for a 0.03–0.15 sweep",
    )
    parser.add_argument(
        "--fixed-armature",
        type=float,
        nargs="+",
        help="added inertia in kg·m² during a damping sweep, one value per --joints entry",
    )
    parser.add_argument("--ring-seconds", type=float, default=1.5)
    args = parser.parse_args()
    armature = args.armature is not None
    if args.fixed_armature is not None and (
        armature or len(args.fixed_armature) != len(args.joints)
    ):
        parser.error("--fixed-armature requires one value per joint and no --armature.")
    fixed = args.fixed_armature or [0.0] * len(args.joints)
    if armature:
        candidates = args.armature or [0.03, 0.06, 0.09, 0.12, 0.15]
    else:
        candidates = args.damping or (
            [i / 20 for i in range(9)]
            if args.fixed_armature is not None
            else [0.0, 0.1, 0.2, 0.3]
        )
    if (
        not np.isfinite(candidates + fixed).all()
        or min(candidates + fixed) < 0
        or not np.isfinite(args.ring_seconds)
        or args.ring_seconds <= 0
    ):
        parser.error(
            "Sweep values must be finite and nonnegative; ring duration must be positive."
        )
    for joint, added_inertia in zip(args.joints, fixed, strict=True):
        runs = sorted(ROOT.glob(f"*_j{joint}_*"))
        if len(runs) != 1:
            parser.error(f"Expected exactly one recorded calibration run for J{joint}.")
        compare(
            joint - 1,
            runs[0],
            candidates,
            args.ring_seconds,
            armature=armature,
            fixed_armature=added_inertia,
        )
    plt.show()


if __name__ == "__main__":
    main()
