import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from me6404_openarm_shaping.data_types import FloatMatrix, FloatVector
    from me6404_openarm_shaping.robot_model import LEFT_ARM_JOINT_NAMES

    return (
        Figure,
        FloatMatrix,
        FloatVector,
        LEFT_ARM_JOINT_NAMES,
        NDArray,
        Path,
        dataclass,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def hardware_timing_intro(mo):
    mo.md(r"""
    # Recorded hardware trajectories

    Select a dataset to load its three baseline, three ZVD, and three EI trials.

    - **Start:** the controller's trajectory clock resets when the goal begins.
      Controller samples are aligned to this timestamp, so the start is **0 s**.
    - **Commanded motion:** $0 \le t < t_f$. The endpoint $t_f$ is the first
      recorded command sample after which all seven positions stay at their final
      values (within $10^{-12}$ rad).
    - **Post-command response:** $t \ge t_f$. This includes the trailing goal hold
      and subsequent recorded response. Pre-roll ($t < 0$) is also retained.
    - **Steady-state estimate:** the mean measured J1–J7 positions over the final
      0.5 s of the recording, using only samples at or after $t_f$.
      This window does not test whether the arm has settled.
    """)
    return


@app.cell
def hardware_dataset_selector(Path, mo):
    hardware_root = Path(__file__).resolve().parents[1] / "hardware_out"
    # The two unnamed simulation exports are identified by their recorded waypoints.
    dataset_exports = {
        "j2_only": "run_20261120_212941_939644_ke3y59b0",
        "j2+j4": "run_20261120_214830_775097_scis0reb",
        "medium-09": "medium-09",
        "wide-02": "wide-02",
        "wide-07": "wide-07",
    }
    group_folders = {
        "baseline": "baseline",
        "ZVD": "time_invariant_zvd",
        "EI": "time_invariant_ei",
    }

    dataset_selector = mo.ui.dropdown(
        options=list(dataset_exports), value="j2_only", label="Hardware dataset"
    )
    mo.output.replace(dataset_selector)
    return dataset_exports, dataset_selector, group_folders, hardware_root


@app.cell
def hardware_run_reader(
    FloatMatrix,
    FloatVector,
    LEFT_ARM_JOINT_NAMES,
    NDArray,
    Path,
    dataclass,
    np,
):
    @dataclass
    class HardwareRun:
        run_id: str
        t0_ns: int
        tf_s: float
        goal_end_s: float
        time_s: FloatVector
        command_time_s: FloatVector
        command_q: FloatMatrix
        reference_q: FloatMatrix
        reference_dq: FloatMatrix
        measured_q: FloatMatrix
        measured_dq: FloatMatrix
        steady_state_mask: NDArray[np.bool_]
        q_measured_steady_state: FloatVector
        error_q: FloatMatrix
        goal_error_q: FloatMatrix
        steady_state_goal_error_q: FloatVector
        steady_state_flagged_joints: tuple[str, ...]
        tracking_pos_rms: FloatVector
        settling_band_rad: FloatVector
        joint_settling_times_s: FloatVector
        ts_s: float | None
        post_command_settling_s: float | None
        joint_equilibrium_settling_times_s: FloatVector
        equilibrium_settling_total_s: float | None
        equilibrium_settling_post_s: float | None
        rms_window_length_s: float = 0.8

        @property
        def motion_mask(self) -> NDArray[np.bool_]:
            return (self.time_s >= 0) & (self.time_s < self.tf_s)

        @property
        def post_command_mask(self) -> NDArray[np.bool_]:
            return self.time_s >= self.tf_s

        @property
        def post_command_rms_window_mask(self) -> NDArray[np.bool_]:
            return (self.time_s >= self.tf_s) & (
                self.time_s <= self.tf_s + self.rms_window_length_s
            )

        @property
        def post_command_pos_rms(self) -> FloatVector | None:
            return residual_position_rms(
                self.time_s, self.error_q, self.tf_s, self.rms_window_length_s
            )

    def residual_position_rms(
        time_s: FloatVector,
        error: FloatMatrix,
        start_s: float,
        duration_s: float,
    ) -> FloatVector | None:
        """Integrate squared residual on the complete window, or flag missing coverage."""
        end_s = start_s + duration_s
        if time_s[0] > start_s or time_s[-1] < end_s:
            return None
        interior = (time_s > start_s) & (time_s < end_s)
        time = np.r_[start_s, time_s[interior], end_s]
        edges = np.array(
            [np.interp([start_s, end_s], time_s, trace) for trace in error.T]
        ).T
        window_error = np.vstack((edges[0], error[interior], edges[1]))
        return np.sqrt(
            np.trapezoid(window_error**2, x=time, axis=0) / duration_s
        )

    def flagged_steady_state_joints(
        time_s: FloatVector, measured_q: FloatMatrix, tf_s: float
    ) -> tuple[str, ...]:
        """Screen the final half-second for drift and large peak-to-peak motion."""
        end_s = time_s[-1]
        early = (time_s >= end_s - 0.5) & (time_s < end_s - 0.25)
        late = time_s >= end_s - 0.25
        if end_s - tf_s < 0.5 or not early.any() or not late.any():
            return LEFT_ARM_JOINT_NAMES
        drift = np.abs(
            measured_q[late].mean(axis=0) - measured_q[early].mean(axis=0)
        )
        peak_to_peak = np.ptp(measured_q[early | late], axis=0)
        flagged = (drift > 0.002) | (peak_to_peak > 0.005)
        return tuple(
            name
            for name, flag in zip(LEFT_ARM_JOINT_NAMES, flagged, strict=True)
            if flag
        )

    def tracking_position_rms(
        time_s: FloatVector, error: FloatMatrix, tf_s: float
    ) -> FloatVector:
        """Per-joint RMS over [0, tf], using the recorded controller timestamps."""
        start = int(np.searchsorted(time_s, 0.0))
        end = int(np.searchsorted(time_s, tf_s))
        # The controller sample at t0 is exactly zero. Interpolate at tf if needed.
        if time_s[end] == tf_s:
            time = time_s[start : end + 1]
            window_error = error[start : end + 1]
        else:
            fraction = (tf_s - time_s[end - 1]) / (
                time_s[end] - time_s[end - 1]
            )
            end_error = error[end - 1] + fraction * (
                error[end] - error[end - 1]
            )
            time = np.append(time_s[start:end], tf_s)
            window_error = np.vstack((error[start:end], end_error))
        return np.sqrt(np.trapezoid(window_error**2, x=time, axis=0) / tf_s)

    def settling_metrics(
        post_time: FloatVector, error: FloatMatrix, band: FloatVector
    ) -> tuple[FloatVector, float | None]:
        # A sample qualifies only if it and every later sample are in band.
        in_band = np.abs(error) <= band
        stays_in_band = np.logical_and.accumulate(in_band[::-1], axis=0)[::-1]
        joint_times = np.full(7, np.nan)
        for joint in range(7):
            candidates = np.flatnonzero(stays_in_band[:, joint])
            if candidates.size > 0:
                joint_times[joint] = post_time[candidates[0]]
        latest = float(joint_times.max())
        return joint_times, None if np.isnan(latest) else latest

    def read_hardware_run(
        run: Path, tolerance: float, *, absolute: bool = False
    ) -> HardwareRun:
        import rosbag2_py as bag
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        state_topic = "/left_joint_trajectory_controller/controller_state"
        command_topic = "/me6404/commanded_trajectory"
        reader = bag.SequentialReader()
        reader.open(
            bag.StorageOptions(uri=str(run / "bag"), storage_id="mcap"),
            bag.ConverterOptions("", ""),
        )
        types = {
            topic.name: get_message(topic.type)
            for topic in reader.get_all_topics_and_types()
            if topic.name in (state_topic, command_topic)
        }
        reader.set_filter(
            bag.StorageFilter(topics=[state_topic, command_topic])
        )

        stamps: list[int] = []
        elapsed: list[int] = []
        states = []
        commands = []
        while reader.has_next():
            topic, raw, _ = reader.read_next()
            msg = deserialize_message(raw, types[topic])
            if tuple(msg.joint_names) != LEFT_ARM_JOINT_NAMES:
                raise ValueError(
                    f"{run.name}: expected left-arm joints in J1–J7 order."
                )
            if topic == command_topic:
                commands.append(msg)
            else:
                stamp = msg.header.stamp
                stamps.append(stamp.sec * 10**9 + stamp.nanosec)
                duration = msg.reference.time_from_start
                elapsed.append(duration.sec * 10**9 + duration.nanosec)
                states.append(
                    (
                        msg.reference.positions,
                        msg.reference.velocities,
                        msg.feedback.positions,
                        msg.feedback.velocities,
                    )
                )

        if len(commands) != 1 or not stamps:
            raise ValueError(
                f"{run.name}: expected one recorded goal and controller states."
            )
        points = commands[0].points
        command_q = np.array(
            [point.positions for point in points], dtype=np.float64
        )
        command_ns = np.array(
            [
                point.time_from_start.sec * 10**9
                + point.time_from_start.nanosec
                for point in points
            ],
            dtype=np.int64,
        )
        goal_end_ns = command_ns[-1]
        clock = np.asarray(elapsed, dtype=np.int64)
        # The first reset from an expired hold is the recorded goal start.
        # Some bags also reset to a new hold after the goal finishes.
        resets = (
            np.flatnonzero((clock[:-1] >= goal_end_ns) & (clock[1:] == 0)) + 1
        )
        not_final = np.flatnonzero(
            np.any(np.abs(command_q - command_q[-1]) > 1e-12, axis=1)
        )
        if resets.size == 0 or not_final.size == 0:
            raise ValueError(
                f"{run.name}: could not locate the goal start or motion end."
            )
        t0_ns = stamps[int(resets[0])]
        tf_s = float(command_ns[not_final[-1] + 1]) * 1e-9
        time_s = (np.asarray(stamps, dtype=np.int64) - t0_ns) * 1e-9
        state_array = np.asarray(states, dtype=np.float64)
        if (
            state_array.shape != (time_s.size, 4, 7)
            or not np.isfinite(state_array).all()
            or np.any(np.diff(time_s) <= 0)
            or not 0 < tf_s < time_s[-1]
        ):
            raise ValueError(
                f"{run.name}: invalid joint data or incomplete motion window."
            )

        measured_q = state_array[:, 2]
        reference_q = state_array[:, 0]
        tracking_pos_rms = tracking_position_rms(
            time_s, measured_q - reference_q, tf_s
        )
        # Estimate steady state from the final 0.5 s, excluding commanded motion.
        steady_state_mask = time_s >= max(tf_s, time_s[-1] - 0.5)

        q_measured_steady_state = measured_q[steady_state_mask].mean(axis=0)

        error_q = measured_q - q_measured_steady_state
        goal_q = command_q[-1]
        goal_error_q = measured_q - goal_q
        steady_state_flags = flagged_steady_state_joints(
            time_s, measured_q, tf_s
        )

        post = time_s >= tf_s
        post_time = time_s[post]
        if absolute:
            settling_band_rad = np.full_like(goal_q, tolerance)
        else:
            # Scale by the commanded goal magnitude, with a floor near zero.
            settling_band_rad = np.maximum(
                0.001, (tolerance / 100.0) * np.abs(goal_q)
            )
        joint_times, ts_s = settling_metrics(
            post_time, goal_error_q[post], settling_band_rad
        )
        equilibrium_band = (
            settling_band_rad
            if absolute
            else np.maximum(
                0.001, (tolerance / 100.0) * np.abs(q_measured_steady_state)
            )
        )
        joint_equilibrium_settling, equilibrium_settling_s = settling_metrics(
            post_time, error_q[post], equilibrium_band
        )

        return HardwareRun(
            run_id=run.name,
            t0_ns=t0_ns,
            tf_s=tf_s,
            goal_end_s=float(goal_end_ns) * 1e-9,
            time_s=time_s,
            command_time_s=command_ns * 1e-9,
            command_q=command_q,
            reference_q=reference_q,
            reference_dq=state_array[:, 1],
            measured_q=measured_q,
            measured_dq=state_array[:, 3],
            steady_state_mask=steady_state_mask,
            q_measured_steady_state=q_measured_steady_state,
            error_q=error_q,
            goal_error_q=goal_error_q,
            steady_state_goal_error_q=q_measured_steady_state - goal_q,
            steady_state_flagged_joints=steady_state_flags,
            tracking_pos_rms=tracking_pos_rms,
            settling_band_rad=settling_band_rad,
            joint_settling_times_s=joint_times,
            ts_s=ts_s,
            post_command_settling_s=None if ts_s is None else ts_s - tf_s,
            joint_equilibrium_settling_times_s=joint_equilibrium_settling,
            equilibrium_settling_total_s=equilibrium_settling_s,
            equilibrium_settling_post_s=None
            if equilibrium_settling_s is None
            else equilibrium_settling_s - tf_s,
        )

    return HardwareRun, read_hardware_run


@app.cell(hide_code=True)
def hardware_dataset_loader(
    HardwareRun,
    dataset_exports,
    group_folders,
    hardware_root,
    read_hardware_run,
):
    def load_hardware_dataset(
        dataset: str, tolerance: float, *, absolute: bool = False
    ) -> dict[str, list[HardwareRun]]:
        export = dataset_exports[dataset]
        runs: dict[str, list[HardwareRun]] = {}
        for group, folder in group_folders.items():
            metadata_files = sorted(
                (hardware_root / folder).glob(f"*/{export}_metadata.yaml")
            )
            if len(metadata_files) != 3:
                raise ValueError(
                    f"{dataset} / {group}: expected three runs, found {len(metadata_files)}."
                )
            runs[group] = [
                read_hardware_run(path.parent, tolerance, absolute=absolute)
                for path in metadata_files
            ]
        return runs

    return (load_hardware_dataset,)


@app.cell
def hardware_settling_control(mo):
    settling_tolerance_mode = mo.ui.dropdown(
        options=["Percentage", "Absolute"],
        value="Absolute",
        label="Settling tolerance mode",
    )
    settling_tolerance = mo.ui.number(
        start=0.0, step=0.1, value=1.0, label="Percentage tolerance [%]"
    )
    settling_tolerance_rad = mo.ui.number(
        start=0.0, step=0.001, value=0.01, label="Absolute tolerance [rad]"
    )
    mo.output.replace(
        mo.vstack(
            [
                settling_tolerance_mode,
                mo.hstack(
                    [settling_tolerance, settling_tolerance_rad],
                    justify="start",
                ),
                mo.md(
                    "**Absolute** uses the same half-width in radians for every joint. "
                    "**Percentage** scales by the magnitude of the band center, with a 0.001 rad floor. "
                    "The center is the measured equilibrium for equilibrium settling and the final command for goal settling."
                ),
            ]
        )
    )
    return settling_tolerance, settling_tolerance_mode, settling_tolerance_rad


@app.cell
def hardware_selected_runs(
    HardwareRun,
    dataset_selector,
    load_hardware_dataset,
    mo,
    settling_tolerance,
    settling_tolerance_mode,
    settling_tolerance_rad,
):
    _runs: dict[str, list[HardwareRun]] = {}
    _absolute = settling_tolerance_mode.value == "Absolute"
    _tolerance = (
        settling_tolerance_rad.value if _absolute else settling_tolerance.value
    )
    try:
        _runs = load_hardware_dataset(
            dataset_selector.value, _tolerance, absolute=_absolute
        )
    except ImportError:
        mo.output.replace(
            mo.md("""
        **ROS Jazzy is not available to this notebook kernel.** Restart marimo
        from the repository root after sourcing `/opt/ros/jazzy/setup.bash`.
        """)
        )

    hardware_runs = _runs
    return (hardware_runs,)


@app.cell(hide_code=True)
def hardware_joint_grid(Figure, HardwareRun, plt):
    def plot_joint_grid(
        runs: dict[str, list[HardwareRun]], window: str
    ) -> Figure:
        titles = {
            "motion": "Commanded-motion window: start to motion end",
            "recording": "Full recording from command start",
            "response": "Post-command response: motion end to recording end",
        }
        colors = {"baseline": "tab:gray", "ZVD": "tab:blue", "EI": "tab:orange"}
        fig, axes = plt.subplots(
            4, 2, sharex=True, figsize=(9, 8), layout="constrained"
        )
        fig.delaxes(axes.flat[7])
        for joint, ax in enumerate(axes.flat[:7]):
            for group, repeats in runs.items():
                for repeat, run in enumerate(repeats):
                    if window == "motion":
                        mask = run.motion_mask
                    elif window == "response":
                        mask = run.post_command_mask
                    else:
                        mask = run.time_s >= 0
                    ax.plot(
                        run.time_s[mask],
                        run.measured_q[mask, joint],
                        color=colors[group],
                        alpha=0.65,
                        label=group if repeat == 0 else None,
                    )
            ax.set_ylabel(f"J{joint + 1} [rad]")
            ax.grid(alpha=0.25)
            ax.tick_params(labelbottom=True)
        if window != "response":
            axes.flat[0].set_xlim(left=0)
        axes.flat[0].legend(ncols=3, fontsize="small")
        fig.suptitle(titles[window])
        fig.supxlabel("Time since command start, t [s]")
        return fig

    return (plot_joint_grid,)


@app.cell
def hardware_plot_controls(mo):
    first_plot_end = mo.ui.dropdown(
        options={"Motion end (tf)": "motion", "Recording end": "recording"},
        value="Recording end",
        label="End first plot at",
    )
    mo.output.replace(first_plot_end)
    return (first_plot_end,)


@app.cell
def hardware_first_joint_plot(
    first_plot_end,
    hardware_runs,
    mo,
    plot_joint_grid,
):
    if hardware_runs:
        _fig = plot_joint_grid(hardware_runs, first_plot_end.value)
        mo.output.replace(_fig)
    return


@app.cell
def hardware_post_command_plot(hardware_runs, mo, plot_joint_grid):
    if hardware_runs:
        _fig = plot_joint_grid(hardware_runs, "response")
        mo.output.replace(_fig)
    return


@app.cell
def hardware_focused_joint_control(mo):
    focused_joint = mo.ui.dropdown(
        options={f"J{joint + 1}": joint for joint in range(7)},
        value="J2",
        label="Joint for focused plots",
    )
    focused_residual_window = mo.ui.dropdown(
        options={"RMS window": "rms", "Full post-command response": "full"},
        value="RMS window",
        label="Focused residual response window",
    )
    mo.output.replace(
        mo.hstack([focused_joint, focused_residual_window], justify="start")
    )
    return focused_joint, focused_residual_window


@app.cell
def hardware_focused_joint_plots(
    focused_joint,
    focused_residual_window,
    hardware_runs,
    mo,
    plt,
):
    if hardware_runs:
        _joint = focused_joint.value
        _rms_only = focused_residual_window.value == "rms"
        _fig, (_full, _residual, _tracking) = plt.subplots(
            3, 1, figsize=(8, 9), layout="constrained"
        )
        _colors = {
            "baseline": "tab:gray",
            "ZVD": "tab:blue",
            "EI": "tab:orange",
        }
        for _group, _runs in hardware_runs.items():
            for _repeat, _run in enumerate(_runs):
                _color = _colors[_group]
                _label = _group if _repeat == 0 else None
                _t = _run.time_s
                _full_mask = _t >= 0
                _post = (
                    _run.post_command_rms_window_mask
                    if _rms_only
                    else _run.post_command_mask
                )
                _motion = _run.motion_mask
                _measured = _run.measured_q[:, _joint]
                _reference = _run.reference_q[:, _joint]
                _full.plot(
                    _t[_full_mask],
                    _measured[_full_mask],
                    color=_color,
                    alpha=0.7,
                    label=_label,
                )
                _full.plot(
                    _t[_full_mask],
                    _reference[_full_mask],
                    "--",
                    color=_color,
                    alpha=0.65,
                )
                _residual.plot(
                    _t[_post] - _run.tf_s,
                    _run.error_q[_post, _joint],
                    color=_color,
                    alpha=0.7,
                )
                _tracking.plot(
                    _t[_motion],
                    (_measured - _reference)[_motion],
                    color=_color,
                    alpha=0.7,
                )
        _full.set(
            title="Full trajectory: measured (solid), reference (dashed)",
            ylabel="Position [rad]",
            xlabel="Time since command start [s]",
        )
        _residual.set(
            title="Residual response relative to measured steady state",
            ylabel="Residual [rad]",
            xlabel="Time since motion end [s]",
        )
        _tracking.set(
            title="In-motion tracking: measured minus reference",
            ylabel="Tracking error [rad]",
            xlabel="Time since command start [s]",
        )
        _full.legend(ncols=3)
        for _ax in (_full, _residual, _tracking):
            _ax.set_xlim(left=0)
            _ax.grid(alpha=0.25)
        if _rms_only:
            _residual.set_xlim(
                right=max(
                    run.rms_window_length_s
                    for runs in hardware_runs.values()
                    for run in runs
                )
            )
        _fig.suptitle(f"J{_joint + 1}: focused hardware comparison")
        mo.output.replace(_fig)
        plt.close(_fig)
    return


@app.cell(hide_code=True)
def hardware_tradeoff_summary(
    HardwareRun,
    dataset_exports,
    dataset_selector,
    hardware_runs,
    load_hardware_dataset,
    np,
    settling_tolerance,
    settling_tolerance_mode,
    settling_tolerance_rad,
):
    def summarize_vibration_tradeoffs(
        datasets: dict[str, dict[str, list[HardwareRun]]],
    ) -> tuple[
        dict[str, dict[str, list[tuple[float, float]]]],
        dict[str, dict[str, list[tuple[float, float]]]],
        dict[str, dict[str, list[tuple[float, float]]]],
        list[str],
        list[str],
    ]:
        equilibrium_settling: dict[
            str, dict[str, list[tuple[float, float]]]
        ] = {}
        command_duration: dict[str, dict[str, list[tuple[float, float]]]] = {}
        tracking: dict[str, dict[str, list[tuple[float, float]]]] = {}
        equilibrium_settling_notes: list[str] = []
        duration_notes: list[str] = []
        for trajectory, groups in datasets.items():
            equilibrium_settling[trajectory] = {}
            command_duration[trajectory] = {}
            tracking[trajectory] = {}
            for method, repeats in groups.items():
                stable_points: list[tuple[float, float]] = []
                duration_points: list[tuple[float, float]] = []
                tracking_points: list[tuple[float, float]] = []
                for repeat, run in enumerate(repeats, start=1):
                    label = f"{trajectory} / {method}, repeat {repeat}"
                    total = run.equilibrium_settling_total_s
                    if total is None:
                        equilibrium_settling_notes.append(
                            f"{label}: did not settle around equilibrium within recording."
                        )
                    else:
                        tracking_rms = float(
                            np.sqrt(np.mean(run.tracking_pos_rms**2))
                        )
                        tracking_points.append((total, tracking_rms))
                    residual = run.post_command_pos_rms
                    if residual is None:
                        duration_notes.append(
                            f"{label}: incomplete residual window."
                        )
                        continue
                    arm_rms = float(np.linalg.norm(residual))
                    duration_points.append(
                        (run.tf_s, arm_rms)
                    )  # The normalized t0 is zero.
                    if total is not None:
                        stable_points.append((total, arm_rms))
                equilibrium_settling[trajectory][method] = stable_points
                command_duration[trajectory][method] = duration_points
                tracking[trajectory][method] = tracking_points
        return (
            equilibrium_settling,
            command_duration,
            tracking,
            equilibrium_settling_notes,
            duration_notes,
        )

    vibration_equilibrium_settling_points = {}
    vibration_duration_points = {}
    vibration_tracking_points = {}
    vibration_equilibrium_settling_notes = []
    vibration_duration_notes = []
    command_durations_s = {}
    hardware_baseline_runs = {}
    if hardware_runs:
        _absolute = settling_tolerance_mode.value == "Absolute"
        _tolerance = (
            settling_tolerance_rad.value
            if _absolute
            else settling_tolerance.value
        )
        _datasets = {
            _trajectory: hardware_runs
            if _trajectory == dataset_selector.value
            else load_hardware_dataset(
                _trajectory, _tolerance, absolute=_absolute
            )
            for _trajectory in dataset_exports
        }
        (
            vibration_equilibrium_settling_points,
            vibration_duration_points,
            vibration_tracking_points,
            vibration_equilibrium_settling_notes,
            vibration_duration_notes,
        ) = summarize_vibration_tradeoffs(_datasets)
        hardware_baseline_runs = {
            trajectory: groups["baseline"][0]
            for trajectory, groups in _datasets.items()
        }
        # Duration is available even when a repeat lacks a complete residual window.
        command_durations_s = {
            trajectory: {
                method: [run.tf_s for run in repeats]
                for method, repeats in groups.items()
            }
            for trajectory, groups in _datasets.items()
        }
    return (
        command_durations_s,
        hardware_baseline_runs,
        vibration_duration_notes,
        vibration_duration_points,
        vibration_equilibrium_settling_notes,
        vibration_equilibrium_settling_points,
        vibration_tracking_points,
    )


@app.cell(hide_code=True)
def hardware_comparison_plot_helper(Figure, np, plt):
    def plot_hardware_comparison(
        groups_by_trajectory: dict[str, dict[str, list[tuple[float, float]]]],
        xlabel: str,
        ylabel: str,
        *,
        yscale: str = "linear",
    ) -> Figure:
        colors = {"baseline": "tab:gray", "ZVD": "tab:blue", "EI": "tab:orange"}
        fig, axes = plt.subplots(
            2,
            3,
            sharex=True,
            sharey=True,
            figsize=(11, 6.5),
            layout="constrained",
        )
        axes.flat[0].set_yscale(yscale)
        for ax, (trajectory, groups) in zip(
            axes.flat, groups_by_trajectory.items()
        ):
            ax.set_title(trajectory)
            ax.grid(alpha=0.25)
            ax.tick_params(labelbottom=True, labelleft=True)
            for method, repeats in groups.items():
                if not repeats:
                    continue
                values = np.asarray(repeats)
                visible = (
                    values[values[:, 1] > 0] if yscale == "log" else values
                )
                ax.scatter(
                    visible[:, 0],
                    visible[:, 1],
                    color=colors[method],
                    s=25,
                    alpha=0.45,
                )
                if len(repeats) == 3:
                    x, y = values.mean(axis=0)
                    x_sd, y_sd = values.std(axis=0, ddof=1)
                    if yscale == "log" and y <= 0:
                        continue
                    ax.errorbar(
                        x,
                        y,
                        xerr=x_sd,
                        yerr=y_sd,
                        fmt="D",
                        color=colors[method],
                        capsize=3,
                    )
        legend_ax = axes.flat[-1]
        legend_ax.set_axis_off()
        for method, color in colors.items():
            legend_ax.scatter([], [], color=color, marker="D", label=method)
        legend_ax.legend(loc="center", title="Method")
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        return fig

    return (plot_hardware_comparison,)


@app.cell(hide_code=True)
def hardware_frequency_variation_model_control(mo):
    hardware_beta_model_selector = mo.ui.dropdown(
        options=["Nominal", "Calibrated"],
        value="Calibrated",
        label="MuJoCo model for β table",
    )
    mo.vstack(
        [
            mo.md("### Trajectory frequency variation (β)"),
            hardware_beta_model_selector,
            mo.md(
                "**Nominal:** 0.40 N·m·s/rad passive damping per joint, no added armature. "
                "**Calibrated:** passive damping and added armature from `robot_config.yaml`."
            ),
        ]
    )
    return (hardware_beta_model_selector,)


@app.cell(hide_code=True)
def hardware_frequency_variation_table(
    Path,
    hardware_baseline_runs,
    hardware_beta_model_selector,
    mo,
    np,
):
    import yaml

    from me6404_openarm_shaping.configuration import load_robot_config
    from me6404_openarm_shaping.mode_tracking import track_modes
    from me6404_openarm_shaping.robot_model import load_model

    hardware_frequency_variation_rows = []
    if hardware_baseline_runs:
        _suite_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "trajectory_suite_focused.yaml"
        )
        _suite = yaml.safe_load(_suite_path.read_text(encoding="utf-8"))
        _screening = {_entry["id"]: _entry for _entry in _suite["trajectories"]}
        _config = load_robot_config()
        if hardware_beta_model_selector.value == "Calibrated":
            _model = load_model(
                _config.passive_damping, added_armature=_config.added_armature
            )
        else:
            _model = load_model(np.full(7, 0.4))
        for _trajectory, _run in hardware_baseline_runs.items():
            _q = _run.command_q[_run.command_time_s <= _run.tf_s]
            _tracked = track_modes(
                _model, _q, _config.motor_kp, _config.motor_kd
            )
            _frequencies = _tracked.natural_frequency_hz[:, :2]
            _mode_1 = _mode_2 = _beta = None
            if not _tracked.stable.all():
                _status = "Unstable or marginal model poles"
            elif (
                _frequencies.shape[1] < 2 or not np.isfinite(_frequencies).all()
            ):
                _status = (
                    "Two oscillatory modes could not be tracked throughout"
                )
            else:
                _spans = 100 * np.ptp(_frequencies, axis=0) / _frequencies[0]
                _mode_1, _mode_2 = (round(float(_span), 2) for _span in _spans)
                _beta = max(_mode_1, _mode_2)
                _status = (
                    "Provisional mode match"
                    if (_tracked.mac[:, :2] < 0.80).any()
                    or (_tracked.assignment_gap < 0.02).any()
                    else "Tracked"
                )
            _original = _screening.get(_trajectory)
            hardware_frequency_variation_rows.append(
                {
                    "Trajectory": _trajectory,
                    "Mode 1 β [%]": _mode_1,
                    "Mode 2 β [%]": _mode_2,
                    "Recorded-command β [%]": _beta,
                    "Original screening β [%]": (
                        None
                        if _original is None
                        else round(100 * _original["beta"], 2)
                    ),
                    "Screening band": None
                    if _original is None
                    else _original["band"],
                    "Tracking": _status,
                }
            )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(r"""
                    The selected MuJoCo model tracks the two lowest initial
                    oscillatory modes along the first recorded baseline command
                    for each trajectory, through motion end. For each mode,
                    $\beta = (f_{\max} - f_{\min}) / f_{\mathrm{initial}}$;
                    the trajectory β is the larger of the two. These are model
                    estimates, not measured hardware frequencies.

                    Original screening β and bands are copied from the focused
                    trajectory suite and refer to the original screening paths
                    and model. The two J2 tests were not in that suite.
                    """),
                    mo.ui.table(
                        hardware_frequency_variation_rows,
                        selection=None,
                        pagination=False,
                    ),
                ]
            )
        )
    return


@app.cell
def hardware_equilibrium_settling_table(hardware_runs, mo):
    if hardware_runs:
        _rows = [
            {
                "Method": _method,
                "Run": _run.run_id,
                "Post-command equilibrium settling [s]": None
                if _run.equilibrium_settling_post_s is None
                else round(_run.equilibrium_settling_post_s, 3),
                "Total equilibrium settling [s]": None
                if _run.equilibrium_settling_total_s is None
                else round(_run.equilibrium_settling_total_s, 3),
                "Status": "Settled around equilibrium"
                if _run.equilibrium_settling_total_s is not None
                else "Not settled around equilibrium within recording",
            }
            for _method, _runs in hardware_runs.items()
            for _run in _runs
        ]
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        "### Settling around measured equilibrium\n\n"
                        "The last joint to enter and stay within its band through recording end "
                        "sets the time. This captures oscillation and other transients, not steady state goal accuracy."
                    ),
                    mo.ui.table(_rows, selection=None, pagination=False),
                ]
            )
        )
    return


@app.cell
def hardware_residual_rms_scale(mo):
    residual_rms_scale = mo.ui.dropdown(
        options={"Linear": "linear", "Log": "log"},
        value="Linear",
        label="Residual tradeoff Y-axis",
    )
    mo.output.replace(residual_rms_scale)
    return (residual_rms_scale,)


@app.cell
def hardware_equilibrium_settling_tradeoff_plot(
    mo,
    plot_hardware_comparison,
    plt,
    residual_rms_scale,
    vibration_duration_notes,
    vibration_equilibrium_settling_notes,
    vibration_equilibrium_settling_points,
):
    if vibration_equilibrium_settling_points:
        _fig = plot_hardware_comparison(
            vibration_equilibrium_settling_points,
            "Total time to settle around measured equilibrium [s]",
            "Whole-arm residual position RMS [rad]",
            yscale=residual_rms_scale.value,
        )
        _outputs = [
            mo.md(
                "### Vibration vs. equilibrium settling time\n\n"
                "Circles are individual repeats; diamonds and bars show mean ± sample SD "
                "when all three repeats are available. Compare methods within each trajectory. "
                "Residual RMS is the joint-space norm over the first 0.8 s after motion ends."
            ),
            mo.as_html(_fig),
        ]
        plt.close(_fig)
        if residual_rms_scale.value == "log":
            _outputs.append(
                mo.md(
                    "Zero RMS points and nonpositive error-bar bounds cannot be shown on a log axis."
                )
            )
        if vibration_equilibrium_settling_notes:
            _outputs.append(
                mo.md(
                    "**Omitted from A and the tracking comparison:**\n\n"
                    + "\n".join(
                        f"- {_note}"
                        for _note in vibration_equilibrium_settling_notes
                    )
                )
            )
        if vibration_duration_notes:
            _outputs.append(
                mo.md(
                    "**Omitted from both residual plots:**\n\n"
                    + "\n".join(
                        f"- {_note}" for _note in vibration_duration_notes
                    )
                )
            )
        mo.output.replace(mo.vstack(_outputs))
    return


@app.cell(hide_code=True)
def hardware_percentage_chart_helper(Figure, np, plt):
    def plot_percentage_changes(
        rows: list[dict[str, str]], column_suffix: str, ylabel: str
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
        x = np.arange(len(rows))
        width = 0.34
        for method, color, offset in (
            ("ZVD", "tab:blue", -width / 2),
            ("EI", "tab:orange", width / 2),
        ):
            column = f"{method} {column_suffix} vs. baseline [%]"
            available = np.array([row[column] != "Unavailable" for row in rows])
            heights = [
                float(row[column])
                for row in rows
                if row[column] != "Unavailable"
            ]
            ax.bar(
                x[available] + offset,
                heights,
                width=width,
                color=color,
                label=method,
            )
        ax.set_xticks(x, [row["Trajectory"] for row in rows])
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend()
        return fig

    return (plot_percentage_changes,)


@app.cell
def hardware_residual_reduction_table(
    mo,
    np,
    plot_percentage_changes,
    plt,
    vibration_duration_points,
):
    def residual_reduction_percent(
        baseline: float | None, shaped: float | None
    ) -> float | None:
        if baseline is None or baseline <= 0 or shaped is None:
            return None
        return 100.0 * (1.0 - shaped / baseline)

    residual_reduction_rows = []
    for _trajectory, _groups in vibration_duration_points.items():
        _means = {
            _method: float(np.mean([_rms for _, _rms in _repeats]))
            if len(_repeats) == 3
            else None
            for _method, _repeats in _groups.items()
        }
        _baseline = _means["baseline"]
        _row = {"Trajectory": _trajectory}
        for _method, _label in (
            ("baseline", "Baseline"),
            ("ZVD", "ZVD"),
            ("EI", "EI"),
        ):
            _mean = _means[_method]
            _row[f"{_label} mean RMS [rad]"] = (
                "Unavailable" if _mean is None else f"{_mean:.6f}"
            )
            if _method != "baseline":
                _change = residual_reduction_percent(_baseline, _mean)
                _row[f"{_label} reduction vs. baseline [%]"] = (
                    "Unavailable" if _change is None else f"{_change:.1f}"
                )
        residual_reduction_rows.append(_row)

    if residual_reduction_rows:
        _fig = plot_percentage_changes(
            residual_reduction_rows,
            "reduction",
            "Residual vibration reduction [%]",
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        "### Residual vibration reduction vs. baseline\n\n"
                        "Each mean uses three whole-arm residual RMS values from the first 0.8 s after motion ends. "
                        "Reductions compare methods within the same trajectory; negative values mean vibration increased. "
                    ),
                    mo.ui.table(
                        residual_reduction_rows,
                        selection=None,
                        pagination=False,
                    ),
                    mo.as_html(_fig),
                ]
            )
        )
        plt.close(_fig)
    return


@app.cell
def hardware_duration_tradeoff_plot(
    mo,
    plot_hardware_comparison,
    plt,
    residual_rms_scale,
    vibration_duration_notes,
    vibration_duration_points,
):
    if vibration_duration_points:
        _fig = plot_hardware_comparison(
            vibration_duration_points,
            "Commanded motion duration (tf − t0) [s]",
            "Whole-arm residual position RMS [rad]",
            yscale=residual_rms_scale.value,
        )
        _outputs = [
            mo.md(
                "### Vibration vs. command duration\n\n"
                "Circles are individual repeats; diamonds and bars show mean ± sample SD "
                "when all three repeats are available. This plot does not use the settling threshold."
            ),
            mo.as_html(_fig),
        ]
        plt.close(_fig)
        if residual_rms_scale.value == "log":
            _outputs.append(
                mo.md(
                    "Zero RMS points and nonpositive error-bar bounds cannot be shown on a log axis."
                )
            )
        if vibration_duration_notes:
            _outputs.append(
                mo.md(
                    "**Omitted due to incomplete residual windows:**\n\n"
                    + "\n".join(
                        f"- {_note}" for _note in vibration_duration_notes
                    )
                )
            )
        mo.output.replace(mo.vstack(_outputs))
    return


@app.cell
def hardware_command_duration_increase_table(
    command_durations_s,
    mo,
    np,
    plot_percentage_changes,
    plt,
):
    command_duration_increase_rows = []
    for _trajectory, _groups in command_durations_s.items():
        _means = {
            _method: float(np.mean(_repeats)) if len(_repeats) == 3 else None
            for _method, _repeats in _groups.items()
        }
        _baseline = _means["baseline"]
        _row = {"Trajectory": _trajectory}
        for _method, _label in (
            ("baseline", "Baseline"),
            ("ZVD", "ZVD"),
            ("EI", "EI"),
        ):
            _mean = _means[_method]
            _row[f"{_label} mean duration [s]"] = (
                "Unavailable" if _mean is None else f"{_mean:.3f}"
            )
            if _method != "baseline":
                _row[f"{_label} increase vs. baseline [%]"] = (
                    "Unavailable"
                    if _baseline is None or _baseline <= 0 or _mean is None
                    else f"{100.0 * (_mean / _baseline - 1.0):.1f}"
                )
        command_duration_increase_rows.append(_row)

    if command_duration_increase_rows:
        _fig = plot_percentage_changes(
            command_duration_increase_rows,
            "increase",
            "Command duration increase [%]",
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        "### Command duration increase vs. baseline\n\n"
                        "Mean commanded motion duration (tf − t0) across three repeats, excluding the trailing hold. "
                    ),
                    mo.ui.table(
                        command_duration_increase_rows,
                        selection=None,
                        pagination=False,
                    ),
                    mo.as_html(_fig),
                ]
            )
        )
        plt.close(_fig)
    return


@app.cell
def hardware_residual_rms_tables(hardware_runs, mo, np):
    if hardware_runs:
        _run_rows = []
        _stats = {}
        _labels = [f"J{_joint}" for _joint in range(1, 8)] + ["Whole arm"]
        for _group, _runs in hardware_runs.items():
            _metrics = []
            for _run in _runs:
                _residual = _run.post_command_pos_rms
                _values = (
                    None
                    if _residual is None
                    else np.r_[_residual, np.linalg.norm(_residual)]
                )
                if _values is not None:
                    _metrics.append(_values)
                _flagged = ", ".join(
                    "J" + _name[-1]
                    for _name in _run.steady_state_flagged_joints
                )
                _run_rows.append(
                    {
                        "Method": _group,
                        "Run": _run.run_id,
                        **{
                            _label: None
                            if _values is None
                            else round(float(_values[_index]), 6)
                            for _index, _label in enumerate(_labels)
                        },
                        "0.8 s window": "Incomplete"
                        if _values is None
                        else "Complete",
                        "Tail warnings": _flagged or "None",
                    }
                )
            _stats[_group] = (
                np.stack(_metrics) if _metrics else np.empty((0, len(_labels)))
            )
        _summary_rows = []
        for _index, _label in enumerate(_labels):
            _row = {"Joint": _label}
            for _group, _values in _stats.items():
                _n = len(_values)
                if _n >= 2:
                    _trace = _values[:, _index]
                    _row[_group] = (
                        f"{_trace.mean():.6f} ± {_trace.std(ddof=1):.6f} (n={_n})"
                    )
                elif _n == 1:
                    _row[_group] = (
                        f"{_values[0, _index]:.6f} (n=1; SD unavailable)"
                    )
                else:
                    _row[_group] = "Unavailable (n=0)"
            _summary_rows.append(_row)
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("""
            ### Residual position RMS [rad]

            Exact 0.8 s window from motion end. Whole-arm RMS is the per-run
            Euclidean norm of the seven joint RMS values (joint space, not
            end-effector displacement). Summary values are mean ± sample SD.
            Incomplete windows are excluded. Tail warnings flag a short tail,
            drift above 0.002 rad, or peak-to-peak range above 0.005 rad in the
            final 0.5 s. Flagged estimates are retained but should be interpreted cautiously.
            """),
                    mo.ui.table(
                        _summary_rows, selection=None, pagination=False
                    ),
                    mo.md("**Individual runs**"),
                    mo.ui.table(_run_rows, selection=None, pagination=False),
                ]
            )
        )
    return


@app.cell
def hardware_tracking_rms_table(hardware_runs, mo):
    if hardware_runs:
        _rows = [
            {
                "Group": _group,
                "Run": _run.run_id,
                **{
                    f"J{_joint}": round(float(_rms), 6)
                    for _joint, _rms in enumerate(
                        _run.tracking_pos_rms, start=1
                    )
                },
            }
            for _group, _runs in hardware_runs.items()
            for _run in _runs
        ]
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(r"""
                ### Position tracking RMS [rad]

                Measured position minus the controller reference, per joint, over
                $[t_0,t_f]$ only. Squared error is integrated using the recorded
                timestamps and normalized by the commanded motion duration.
                """),
                    mo.ui.table(_rows, selection=None, pagination=False),
                ]
            )
        )
    return


@app.cell
def hardware_tracking_tradeoff_plot(
    mo,
    plot_hardware_comparison,
    plt,
    vibration_tracking_points,
):
    if vibration_tracking_points:
        _fig = plot_hardware_comparison(
            vibration_tracking_points,
            "Total time to settle around measured equilibrium [s]",
            "In-motion position tracking RMS [rad]",
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        "### Equilibrium settling time vs. position tracking error\n\n"
                        "Per-repeat tracking RMS combines the seven joints with equal weight. "
                        "Circles are repeats; diamonds and bars show mean ± sample SD for complete groups."
                    ),
                    mo.as_html(_fig),
                ]
            )
        )
        plt.close(_fig)
    return


@app.cell
def hardware_timing_table(hardware_runs, mo, np):
    settling_results = []
    for _group, _runs in hardware_runs.items():
        for _run in _runs:
            _ts = _run.ts_s
            _post_command_s = _run.post_command_settling_s
            _limiting_joints = ", ".join(
                f"J{_joint}"
                for _joint, _time in enumerate(
                    _run.joint_settling_times_s, start=1
                )
                if (np.isnan(_time) if _ts is None else _time == _ts)
            )
            settling_results.append(
                {
                    "Group": _group,
                    "Run": _run.run_id,
                    "Motion end tf [s]": round(_run.tf_s, 3),
                    "Full goal end [s]": round(_run.goal_end_s, 3),
                    "Last recorded sample [s]": round(
                        float(_run.time_s[-1]), 3
                    ),
                    "Post-command goal settling [s]": None
                    if _post_command_s is None
                    else round(_post_command_s, 3),
                    "Total goal settling [s]": None
                    if _ts is None
                    else round(_ts, 3),
                    "Limiting joint(s)": _limiting_joints,
                    **{
                        f"J{_joint} steady-state error [rad]": round(
                            float(_error), 6
                        )
                        for _joint, _error in enumerate(
                            _run.steady_state_goal_error_q, start=1
                        )
                    },
                    "Status": "Settled"
                    if _ts is not None
                    else "Not settled within recording",
                }
            )
    if settling_results:
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        "### Commanded-goal settling and steady-state error\n\n"
                        "Signed steady-state errors are measured equilibrium minus final commanded position."
                    ),
                    mo.ui.table(
                        settling_results, selection=None, pagination=False
                    ),
                ]
            )
        )
    return


if __name__ == "__main__":
    app.run()
