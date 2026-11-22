import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from matplotlib.figure import Figure

    from me6404_openarm_shaping.configuration import load_robot_config
    from me6404_openarm_shaping.data_types import (
        FloatMatrix,
        FloatVector,
        JointTrajectory,
    )
    from me6404_openarm_shaping.robot_model import load_model
    from me6404_openarm_shaping.simulation_helpers import run_trial

    # The experiment script lives in the checkout, outside the installed package.
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from experiments.calibrate_joint_response import (
        ROOT,
        Replay,
        add_joint_armature,
        read_replay,
    )

    return (
        Figure,
        FloatMatrix,
        FloatVector,
        JointTrajectory,
        Path,
        ROOT,
        Replay,
        add_joint_armature,
        load_model,
        load_robot_config,
        mo,
        np,
        read_replay,
        run_trial,
    )


@app.cell(hide_code=True)
def overview(mo):
    mo.md("""
    # OpenArm effective model calibration — J1–J7

    Each recorded excitation is replayed with the recorded motor gains,
    controller output, and feedforward effort. The nominal comparison uses fixed
    pre-calibration damping of **0.40 N·m·s/rad on every joint** and **zero added
    armature**, regardless of the project YAML defaults. A tuned single-joint
    replay changes only its excited joint; all other joints retain nominal
    parameters.

    The shared ring-down duration sets the cutoff for all calibration plots. A positive
    offset moves only that joint’s tuned ring-down trace later; it does
    not change the simulation.

    No parameters or time offsets are fitted.

    The FFTs use the unshifted ring-down samples. All three traces are resampled
    onto the same uniform grid, mean-subtracted, and Hann-windowed before plotting
    single-sided amplitude from 0 to 20 Hz.
    """)
    return


@app.cell(hide_code=True)
def load_recordings(ROOT, Replay, load_robot_config, read_replay):
    def load_recording(joint: int) -> Replay:
        runs = list(ROOT.glob(f"*_j{joint + 1}_*"))
        if len(runs) != 1:
            raise ValueError(
                f"Expected one calibration recording for J{joint + 1}."
            )
        return read_replay(runs[0], joint)

    config = load_robot_config()
    calibration_replays = tuple(load_recording(joint) for joint in range(7))
    return calibration_replays, config


@app.cell(hide_code=True)
def plot_duration(calibration_replays, config, mo, np):
    # Leave one MuJoCo step of headroom for the final replay tick.
    _available_s = (
        min(
            replay.time[-1] - replay.motion_end_s
            for replay in calibration_replays
        )
        - config.simulation.dt_s
    )
    _max_duration_s = float(np.floor(_available_s * 100) / 100)
    ring_duration = mo.ui.slider(
        start=0.01,
        stop=_max_duration_s,
        step=0.01,
        value=3.5,
        show_value=True,
        include_input=True,
        label="Shared calibration ring-down duration [s]",
    )
    mo.output.replace(ring_duration)
    return (ring_duration,)


@app.cell(hide_code=True)
def controls(config, mo):
    def joint_controls(armature: float, damping: float, offset_ms: float):
        return (
            mo.ui.number(
                start=0,
                value=armature,
                step=0.001,
                label="Added armature [kg·m²]",
            ),
            mo.ui.number(
                start=0,
                value=damping,
                step=0.01,
                label="Passive damping [N·m·s/rad]",
            ),
            mo.ui.number(
                value=offset_ms, step=1, label="Ring-down time offset [ms]"
            ),
        )

    j1_armature, j1_damping, j1_offset = joint_controls(0.11, 0.35, 80)
    j2_armature, j2_damping, j2_offset = joint_controls(0.09, 0.25, 90)
    j3_armature, j3_damping, j3_offset = joint_controls(0.2, 0.6, 0)
    j4_armature, j4_damping, j4_offset = joint_controls(
        0.075, float(config.passive_damping[3]), 150
    )
    j5_armature, j5_damping, j5_offset = joint_controls(0, 0.2, 0)
    j6_armature, j6_damping, j6_offset = joint_controls(0.02, 0.2, 0)
    j7_armature, j7_damping, j7_offset = joint_controls(0.025, 0.3, 0)
    return (
        j1_armature,
        j1_damping,
        j1_offset,
        j2_armature,
        j2_damping,
        j2_offset,
        j3_armature,
        j3_damping,
        j3_offset,
        j4_armature,
        j4_damping,
        j4_offset,
        j5_armature,
        j5_damping,
        j5_offset,
        j6_armature,
        j6_damping,
        j6_offset,
        j7_armature,
        j7_damping,
        j7_offset,
    )


@app.cell
def write_calibration_yaml_values(Path):
    import re
    from tempfile import NamedTemporaryFile

    import yaml

    from me6404_openarm_shaping.configuration import (
        DEFAULT_CONFIG_PATH,
        parse_robot_config,
    )

    def write_calibration_yaml(
        path: Path, damping: list[float], armature: list[float]
    ) -> None:
        """Validate and replace only the two top-level calibration arrays."""
        source = path.read_text(encoding="utf-8")
        updated = source
        for key, values in (
            ("passive_damping", damping),
            ("added_armature", armature),
        ):
            # Keep the rest of the project YAML, including its comments, unchanged.
            pattern = rf"^({key}:[ \t]*)\[[^\n\]]*\]([ \t]*(?:#[^\n]*)?)$"
            line = "[" + ", ".join(str(value) for value in values) + "]"
            updated, count = re.subn(
                pattern,
                lambda match, line=line: match[1] + line + match[2],
                updated,
                flags=re.MULTILINE,
            )
            if count != 1:
                raise ValueError(
                    f"Expected exactly one top-level flow array for {key} in {path.name}."
                )
        _ = parse_robot_config(yaml.safe_load(updated))
        if updated == source:
            return

        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                _ = stream.write(updated)
            temporary.chmod(path.stat().st_mode & 0o777)
            _ = temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return DEFAULT_CONFIG_PATH, write_calibration_yaml, yaml


@app.cell(hide_code=True)
def calibration_yaml_export_controls(
    DEFAULT_CONFIG_PATH,
    j1_armature,
    j1_damping,
    j2_armature,
    j2_damping,
    j3_armature,
    j3_damping,
    j4_armature,
    j4_damping,
    j5_armature,
    j5_damping,
    j6_armature,
    j6_damping,
    j7_armature,
    j7_damping,
    mo,
    write_calibration_yaml,
    yaml,
):
    calibration_export_message, set_calibration_export_message = mo.state("")
    _dampings = (
        j1_damping,
        j2_damping,
        j3_damping,
        j4_damping,
        j5_damping,
        j6_damping,
        j7_damping,
    )
    _armatures = (
        j1_armature,
        j2_armature,
        j3_armature,
        j4_armature,
        j5_armature,
        j6_armature,
        j7_armature,
    )

    def _save_calibration(_event):
        try:
            damping = [
                float(widget.value)
                for widget in _dampings
                if widget.value is not None
            ]
            armature = [
                float(widget.value)
                for widget in _armatures
                if widget.value is not None
            ]
            if len(damping) != 7 or len(armature) != 7:
                raise ValueError(
                    "Fill in all seven damping and armature fields."
                )
            write_calibration_yaml(DEFAULT_CONFIG_PATH, damping, armature)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
            set_calibration_export_message(f"**Not saved:** {error}")
        else:
            set_calibration_export_message(
                "**Saved to `config/robot_config.yaml`.** Reload other notebooks to use the new values. Existing recordings are unchanged."
            )

    calibration_export_button = mo.ui.button(
        label="Save calibration to project YAML",
        kind="success",
        on_click=_save_calibration,
    )
    mo.vstack(
        [
            mo.md("""### Export current calibration
        Saves only the J1–J7 passive damping and added armature arrays in
        `config/robot_config.yaml`, which batch simulation also uses. Plot offsets
        are not exported. Editing a value does not save it automatically.
        """),
            mo.ui.table(
                [
                    {
                        "Joint": f"J{i}",
                        "Passive damping [N·m·s/rad]": damping.value,
                        "Added armature [kg·m²]": armature.value,
                    }
                    for i, (damping, armature) in enumerate(
                        zip(_dampings, _armatures, strict=True), 1
                    )
                ],
                selection=None,
                pagination=False,
            ),
            calibration_export_button,
        ]
    )
    return (calibration_export_message,)


@app.cell(hide_code=True)
def calibration_yaml_export_status(calibration_export_message, mo):
    mo.output.replace(mo.md(calibration_export_message()))
    return


@app.cell(hide_code=True)
def replay_simulation(
    FloatMatrix,
    JointTrajectory,
    Replay,
    add_joint_armature,
    config,
    load_model,
    np,
    run_trial,
):
    def simulate_motor_replay(
        replay: Replay,
        added_armature: tuple[float, ...],
        damping: tuple[float, ...],
        end_s: float,
    ) -> FloatMatrix:
        dt = config.simulation.dt_s
        # Hold the most recent motor input on each tick. The last tick
        # brackets the comparison window for interpolation.
        t = np.arange(int(np.ceil(end_s / dt)) + 1) * dt
        i = np.searchsorted(replay.time, t, side="right") - 1
        command: JointTrajectory = {
            "t": t,
            "q": replay.output_q[i],
            "dq": replay.reference_dq[i],
            "ddq": replay.reference_ddq[i],
        }
        model = load_model(np.asarray(damping))
        model.opt.timestep = dt
        for joint, inertia in enumerate(added_armature):
            if inertia != 0.0:
                add_joint_armature(model, joint, inertia)
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
        plot_time = replay.time[replay.time <= end_s]
        return np.column_stack(
            [
                np.interp(plot_time, result["t"], result["q"][:, joint])
                for joint in range(7)
            ]
        )

    # Keep the pre-calibration comparison independent of project YAML defaults.
    nominal_damping = (0.4,) * 7
    return nominal_damping, simulate_motor_replay


@app.cell(hide_code=True)
def nominal_simulations(
    FloatMatrix,
    calibration_replays,
    mo,
    nominal_damping,
    ring_duration,
    simulate_motor_replay,
):
    calibration_ends = tuple(
        replay.motion_end_s + ring_duration.value
        for replay in calibration_replays
    )
    calibration_nominal = tuple(
        simulate_motor_replay(replay, (0.0,) * 7, nominal_damping, end)
        for replay, end in zip(
            calibration_replays, calibration_ends, strict=True
        )
    )

    def simulate_calibration(
        joint: int, armature: float | None, damping: float | None
    ) -> FloatMatrix:
        mo.stop(armature is None or damping is None)
        assert armature is not None and damping is not None
        inertias = [0.0] * 7
        inertias[joint] = armature
        dampings = list(nominal_damping)
        dampings[joint] = damping
        return simulate_motor_replay(
            calibration_replays[joint],
            tuple(inertias),
            tuple(dampings),
            calibration_ends[joint],
        )

    return calibration_nominal, simulate_calibration


@app.cell(hide_code=True)
def j1_tuned_simulation(j1_armature, j1_damping, simulate_calibration):
    j1_tuned = simulate_calibration(0, j1_armature.value, j1_damping.value)
    return (j1_tuned,)


@app.cell(hide_code=True)
def j2_tuned_simulation(j2_armature, j2_damping, simulate_calibration):
    j2_tuned = simulate_calibration(1, j2_armature.value, j2_damping.value)
    return (j2_tuned,)


@app.cell(hide_code=True)
def j3_tuned_simulation(j3_armature, j3_damping, simulate_calibration):
    j3_tuned = simulate_calibration(2, j3_armature.value, j3_damping.value)
    return (j3_tuned,)


@app.cell(hide_code=True)
def j4_tuned_simulation(j4_armature, j4_damping, simulate_calibration):
    j4_tuned = simulate_calibration(3, j4_armature.value, j4_damping.value)
    return (j4_tuned,)


@app.cell(hide_code=True)
def j5_tuned_simulation(j5_armature, j5_damping, simulate_calibration):
    j5_tuned = simulate_calibration(4, j5_armature.value, j5_damping.value)
    return (j5_tuned,)


@app.cell(hide_code=True)
def j6_tuned_simulation(j6_armature, j6_damping, simulate_calibration):
    j6_tuned = simulate_calibration(5, j6_armature.value, j6_damping.value)
    return (j6_tuned,)


@app.cell(hide_code=True)
def j7_tuned_simulation(j7_armature, j7_damping, simulate_calibration):
    j7_tuned = simulate_calibration(6, j7_armature.value, j7_damping.value)
    return (j7_tuned,)


@app.cell(hide_code=True)
def calibration_plot_helpers(
    Figure,
    FloatMatrix,
    calibration_nominal,
    calibration_replays,
):
    def calibration_plots(
        joints: tuple[int, ...],
        tuned_runs: tuple[FloatMatrix, ...],
        offsets_ms: tuple[float | None, ...],
    ) -> tuple[Figure, Figure]:
        count = len(joints)
        width = 10 if count == 2 else 5
        motion = Figure(figsize=(width, 3.5), layout="constrained")
        motion_axes = motion.subplots(1, count, squeeze=False)[0]
        ring = Figure(figsize=(width, 3.5), layout="constrained")
        ring_axes = ring.subplots(1, count, squeeze=False)[0]
        for column, (excited, tuned, offset_ms) in enumerate(
            zip(joints, tuned_runs, offsets_ms, strict=True)
        ):
            replay = calibration_replays[excited]
            nominal = calibration_nominal[excited]
            n = len(nominal)
            time = replay.time[:n]
            ax = motion_axes[column]
            ax.plot(
                time,
                replay.output_q[:n, excited],
                ":",
                color="0.5",
                label="Command",
            )
            ax.plot(
                time,
                replay.measured_q[:n, excited],
                color="black",
                label="Hardware",
            )
            ax.plot(
                time,
                nominal[:, excited],
                color="tab:blue",
                label="Nominal sim",
            )
            ax.plot(
                time,
                tuned[:, excited],
                "--",
                color="tab:orange",
                label="Tuned sim",
            )
            ax.axvline(replay.motion_end_s, color="0.75", linewidth=1)
            ax.set(
                title=f"J{excited + 1} excitation",
                ylabel=f"J{excited + 1} position [rad]",
            )
            ax.grid(alpha=0.25)

            post = time >= replay.motion_end_s
            t = time[post] - replay.motion_end_s
            target = replay.output_q[n - 1, excited]
            shift_s = 0.0 if offset_ms is None else offset_ms * 1e-3
            ax = ring_axes[column]
            ax.axhline(0, color="0.75", linewidth=1)
            ax.plot(
                t,
                replay.measured_q[:n, excited][post] - target,
                color="black",
                label="Hardware",
            )
            ax.plot(
                t,
                nominal[post, excited] - target,
                color="tab:blue",
                label="Nominal sim",
            )
            ax.plot(
                t + shift_s,
                tuned[post, excited] - target,
                "--",
                color="tab:orange",
                label="Tuned sim",
            )
            ax.set(
                title=f"J{excited + 1} ring-down",
                ylabel="Position − final command [rad]",
            )
            ax.grid(alpha=0.25)
        motion_axes[0].legend(fontsize="small")
        motion.suptitle("Unshifted command and response")
        motion.supxlabel("Time since recorded goal start [s]")
        ring_axes[0].legend(fontsize="small")
        ring.supxlabel("Time since recorded command motion ended [s]")
        return motion, ring

    return (calibration_plots,)


@app.cell(hide_code=True)
def calibration_spectrum_helpers(
    Figure,
    FloatMatrix,
    FloatVector,
    calibration_nominal,
    calibration_replays,
    np,
):
    def ring_down_spectrum(
        time: FloatVector, traces: FloatMatrix
    ) -> tuple[FloatVector, FloatMatrix]:
        # Resample hardware and both simulations onto the same uniform grid.
        uniform_time, dt = np.linspace(
            time[0], time[-1], time.size, retstep=True
        )
        uniform_traces = np.column_stack(
            [np.interp(uniform_time, time, trace) for trace in traces.T]
        )
        window = np.hanning(time.size)
        centered = uniform_traces - uniform_traces.mean(axis=0)
        amplitude = np.abs(np.fft.rfft(centered * window[:, None], axis=0))
        # Correct for the window gain; only bins with negative-frequency partners double.
        amplitude /= window.sum()
        amplitude[1 : -1 if time.size % 2 == 0 else None] *= 2
        return np.fft.rfftfreq(time.size, d=float(dt)), amplitude

    def plot_calibration_spectra(
        joints: tuple[int, ...], tuned_runs: tuple[FloatMatrix, ...]
    ) -> Figure:
        fig = Figure(figsize=(5 * len(joints), 3.5), layout="constrained")
        axes = fig.subplots(1, len(joints), squeeze=False)[0]
        for ax, joint, tuned in zip(axes, joints, tuned_runs, strict=True):
            replay = calibration_replays[joint]
            nominal = calibration_nominal[joint]
            time = replay.time[: len(nominal)]
            post = time >= replay.motion_end_s
            ax.set(
                title=f"J{joint + 1} ring-down FFT",
                xlim=(0, 20),
                ylabel="Amplitude [rad]",
            )
            ax.grid(alpha=0.25)
            if np.count_nonzero(post) >= 4:
                traces = np.column_stack(
                    (
                        replay.measured_q[: len(nominal), joint][post],
                        nominal[post, joint],
                        tuned[post, joint],
                    )
                )
                frequency, amplitude = ring_down_spectrum(time[post], traces)
                if frequency[1] <= 20:
                    for trace, label, color, style in zip(
                        amplitude.T,
                        ("Hardware", "Nominal sim", "Tuned sim"),
                        ("black", "tab:blue", "tab:orange"),
                        ("-", "-", "--"),
                        strict=True,
                    ):
                        # DC is not an oscillatory component; keep the first positive bin.
                        ax.plot(
                            frequency[1:],
                            trace[1:],
                            style,
                            color=color,
                            label=label,
                        )
                    ax.legend(fontsize="small")
                    continue
            ax.text(
                0.5,
                0.5,
                "Ring-down too short for 0–20 Hz",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        fig.supxlabel("Frequency [Hz]")
        return fig

    return plot_calibration_spectra, ring_down_spectrum


@app.cell(hide_code=True)
def calibration_j1_j2(
    calibration_plots,
    j1_armature,
    j1_damping,
    j1_offset,
    j1_tuned,
    j2_armature,
    j2_damping,
    j2_offset,
    j2_tuned,
    mo,
):
    _joints = (0, 1)
    _controls = (
        (j1_armature, j1_damping, j1_offset),
        (j2_armature, j2_damping, j2_offset),
    )
    _motion, _ring = calibration_plots(
        _joints,
        (
            j1_tuned,
            j2_tuned,
        ),
        (
            j1_offset.value,
            j2_offset.value,
        ),
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack(
                            [mo.md(f"**J{joint + 1} excitation**"), *widgets]
                        )
                        for joint, widgets in zip(
                            _joints, _controls, strict=True
                        )
                    ],
                    widths="equal",
                    align="start",
                    wrap=True,
                ),
                _motion,
                _ring,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def calibration_fft_j1_j2(j1_tuned, j2_tuned, mo, plot_calibration_spectra):
    mo.output.replace(
        plot_calibration_spectra(
            (0, 1),
            (
                j1_tuned,
                j2_tuned,
            ),
        )
    )
    return


@app.cell(hide_code=True)
def calibration_j3_j4(
    calibration_plots,
    j3_armature,
    j3_damping,
    j3_offset,
    j3_tuned,
    j4_armature,
    j4_damping,
    j4_offset,
    j4_tuned,
    mo,
):
    _joints = (2, 3)
    _controls = (
        (j3_armature, j3_damping, j3_offset),
        (j4_armature, j4_damping, j4_offset),
    )
    _motion, _ring = calibration_plots(
        _joints,
        (
            j3_tuned,
            j4_tuned,
        ),
        (
            j3_offset.value,
            j4_offset.value,
        ),
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack(
                            [mo.md(f"**J{joint + 1} excitation**"), *widgets]
                        )
                        for joint, widgets in zip(
                            _joints, _controls, strict=True
                        )
                    ],
                    widths="equal",
                    align="start",
                    wrap=True,
                ),
                _motion,
                _ring,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def calibration_fft_j3_j4(j3_tuned, j4_tuned, mo, plot_calibration_spectra):
    mo.output.replace(
        plot_calibration_spectra(
            (2, 3),
            (
                j3_tuned,
                j4_tuned,
            ),
        )
    )
    return


@app.cell(hide_code=True)
def calibration_j5_j6(
    calibration_plots,
    j5_armature,
    j5_damping,
    j5_offset,
    j5_tuned,
    j6_armature,
    j6_damping,
    j6_offset,
    j6_tuned,
    mo,
):
    _joints = (4, 5)
    _controls = (
        (j5_armature, j5_damping, j5_offset),
        (j6_armature, j6_damping, j6_offset),
    )
    _motion, _ring = calibration_plots(
        _joints,
        (
            j5_tuned,
            j6_tuned,
        ),
        (
            j5_offset.value,
            j6_offset.value,
        ),
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack(
                            [mo.md(f"**J{joint + 1} excitation**"), *widgets]
                        )
                        for joint, widgets in zip(
                            _joints, _controls, strict=True
                        )
                    ],
                    widths="equal",
                    align="start",
                    wrap=True,
                ),
                _motion,
                _ring,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def calibration_fft_j5_j6(j5_tuned, j6_tuned, mo, plot_calibration_spectra):
    mo.output.replace(
        plot_calibration_spectra(
            (4, 5),
            (
                j5_tuned,
                j6_tuned,
            ),
        )
    )
    return


@app.cell(hide_code=True)
def calibration_j7(
    calibration_plots,
    j7_armature,
    j7_damping,
    j7_offset,
    j7_tuned,
    mo,
):
    _joints = (6,)
    _controls = ((j7_armature, j7_damping, j7_offset),)
    _motion, _ring = calibration_plots(_joints, (j7_tuned,), (j7_offset.value,))
    mo.output.replace(
        mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack(
                            [mo.md(f"**J{joint + 1} excitation**"), *widgets]
                        )
                        for joint, widgets in zip(
                            _joints, _controls, strict=True
                        )
                    ],
                    widths="equal",
                    align="start",
                    wrap=True,
                ),
                _motion,
                _ring,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def calibration_fft_j7(j7_tuned, mo, plot_calibration_spectra):
    mo.output.replace(plot_calibration_spectra((6,), (j7_tuned,)))
    return


@app.cell(hide_code=True)
def calibration_comparison_helpers(
    FloatMatrix,
    FloatVector,
    Replay,
    calibration_nominal,
    calibration_replays,
    np,
    ring_down_spectrum,
):
    def replay_rmse(time: FloatVector, error: FloatVector) -> float | None:
        if time.size < 2:
            return None
        # Weight by elapsed time because the recorded timestamps are not uniform.
        return float(
            np.sqrt(np.trapezoid(error**2, x=time) / (time[-1] - time[0]))
        )

    def interpolated_fft_peak(
        frequency: FloatVector, spectrum: FloatVector
    ) -> tuple[float, float]:
        peak = int(np.argmax(spectrum))
        if 0 < peak < spectrum.size - 1:
            neighbors = spectrum[peak - 1 : peak + 2]
            if np.all(neighbors > 0):
                # Fit a parabola to log amplitude around the Hann-windowed peak.
                left, center, right = np.log(neighbors)
                curvature = left - 2 * center + right
                if curvature < 0:
                    offset_bins = 0.5 * (left - right) / curvature
                    peak_hz = frequency[peak] + offset_bins * (
                        frequency[1] - frequency[0]
                    )
                    peak_rad = np.exp(
                        center + 0.25 * (right - left) * offset_bins
                    )
                    return float(peak_hz), float(peak_rad)
        return float(frequency[peak]), float(spectrum[peak])

    def dominant_ring_peaks(
        time: FloatVector, traces: FloatMatrix
    ) -> tuple[tuple[float | None, float | None], ...]:
        if time.size < 4:
            return ((None, None),) * traces.shape[1]
        frequency, amplitude = ring_down_spectrum(time, traces)
        band = (frequency > 0) & (frequency <= 20)
        if not band.any():
            return ((None, None),) * traces.shape[1]
        return tuple(
            (None, None)
            if np.ptp(trace) == 0
            else interpolated_fft_peak(frequency[band], spectrum[band])
            for trace, spectrum in zip(traces.T, amplitude.T, strict=True)
        )

    def comparison_row(
        joint: int,
        metric: str,
        nominal_error: float | None,
        tuned_error: float | None,
        *,
        peaks: tuple[float | None, ...] = (None, None, None),
    ) -> dict[str, str | float | None]:
        improvement = (
            None
            if nominal_error is None
            or nominal_error == 0
            or tuned_error is None
            else 100 * (1 - tuned_error / nominal_error)
        )
        return {
            "Joint": f"J{joint + 1}",
            "Metric": metric,
            "Hardware peak": peaks[0],
            "Nominal peak": peaks[1],
            "Tuned peak": peaks[2],
            "Nominal error": nominal_error,
            "Tuned error": tuned_error,
            "Improvement [%]": improvement,
        }

    def build_calibration_comparison(
        j1_tuned_run: FloatMatrix,
        j2_tuned_run: FloatMatrix,
        test_replay: Replay,
        test_nominal: FloatMatrix,
        test_tuned: FloatMatrix,
    ) -> list[dict[str, str | float | None]]:
        rows: list[dict[str, str | float | None]] = []
        for joint, tuned in enumerate((j1_tuned_run, j2_tuned_run)):
            replay = calibration_replays[joint]
            nominal = calibration_nominal[joint]
            assert nominal.shape == tuned.shape
            n = len(nominal)
            time = replay.time[:n]
            post = time >= replay.motion_end_s
            hardware_q = replay.measured_q[:n, joint][post]
            nominal_q, tuned_q = nominal[post, joint], tuned[post, joint]
            peaks = dominant_ring_peaks(
                time[post], np.column_stack((hardware_q, nominal_q, tuned_q))
            )
            for component, metric in enumerate(
                (
                    "Dominant ring-down frequency [Hz]",
                    "FFT peak amplitude [rad]",
                )
            ):
                measured, nominal_peak, tuned_peak = (
                    peak[component] for peak in peaks
                )
                rows.append(
                    comparison_row(
                        joint,
                        metric,
                        None
                        if measured is None or nominal_peak is None
                        else abs(nominal_peak - measured),
                        None
                        if measured is None or tuned_peak is None
                        else abs(tuned_peak - measured),
                        peaks=(measured, nominal_peak, tuned_peak),
                    )
                )
            rows.append(
                comparison_row(
                    joint,
                    "Ring-down position RMSE [rad]",
                    replay_rmse(time[post], nominal_q - hardware_q),
                    replay_rmse(time[post], tuned_q - hardware_q),
                )
            )

        assert (
            test_nominal.shape
            == test_tuned.shape
            == test_replay.measured_q.shape
        )
        time = test_replay.time
        nominal_error = test_nominal - test_replay.measured_q
        tuned_error = test_tuned - test_replay.measured_q
        for joint in range(2, 7):
            for metric, mask in (
                (
                    "Commanded-motion position RMSE [rad]",
                    time <= test_replay.motion_end_s,
                ),
                (
                    "Post-command position RMSE [rad]",
                    time >= test_replay.motion_end_s,
                ),
            ):
                rows.append(
                    comparison_row(
                        joint,
                        metric,
                        replay_rmse(time[mask], nominal_error[mask, joint]),
                        replay_rmse(time[mask], tuned_error[mask, joint]),
                    )
                )
        return rows

    return build_calibration_comparison, comparison_row, replay_rmse


@app.cell(hide_code=True)
def hardware_replay_overview(mo):
    mo.output.replace(
        mo.md("""
    ## Replay a recorded hardware test

    Replay one recording using its controller position output, feedforward
    effort, motor gains, and measured initial state.  

    The tuned simulation
    applies the current J1–J7 settings together.
    The grid runs from recorded goal start to the last sample before any
    subsequent hold reset (or recording end).  

    A positive shared offset shifts both
    simulation traces later without rerunning either simulation.
    """)
    )
    return


@app.cell(hide_code=True)
def hardware_replay_controls(mo):
    # Match the dataset names and export identifiers in hardware_analysis.py.
    hardware_exports = {
        "j2_only": "run_20261120_212941_939644_ke3y59b0",
        "j2+j4": "run_20261120_214830_775097_scis0reb",
        "medium-09": "medium-09",
        "wide-02": "wide-02",
        "wide-07": "wide-07",
    }
    hardware_case = mo.ui.dropdown(
        options=list(hardware_exports),
        value="medium-09",
        label="Hardware test case",
    )
    hardware_method = mo.ui.dropdown(
        options={
            "Baseline": "baseline",
            "ZVD": "time_invariant_zvd",
            "EI": "time_invariant_ei",
        },
        value="Baseline",
        label="Shaping method",
    )
    hardware_repeat = mo.ui.dropdown(
        options={"1": 1, "2": 2, "3": 3}, value="1", label="Repeat"
    )
    show_hardware_nominal = mo.ui.checkbox(
        value=True, label="Show nominal simulation"
    )
    mo.output.replace(
        mo.hstack(
            [
                hardware_case,
                hardware_method,
                hardware_repeat,
                show_hardware_nominal,
            ],
            justify="start",
            align="center",
            wrap=True,
        )
    )
    return (
        hardware_case,
        hardware_exports,
        hardware_method,
        hardware_repeat,
        show_hardware_nominal,
    )


@app.cell(hide_code=True)
def hardware_replay_offset(mo):
    hardware_sim_offset = mo.ui.number(
        value=60, step=1, label="Shared simulation time offset [ms]"
    )
    mo.output.replace(hardware_sim_offset)
    return (hardware_sim_offset,)


@app.cell(hide_code=True)
def hardware_replay_recording(
    ROOT,
    hardware_case,
    hardware_exports,
    hardware_method,
    hardware_repeat,
    read_replay,
):
    _export = hardware_exports[hardware_case.value]
    _folder = ROOT.parents[1] / hardware_method.value
    _runs = sorted(_folder.glob(f"*/{_export}_metadata.yaml"))
    if len(_runs) != 3:
        raise ValueError(
            f"Expected three repeats for {_export} in {_folder.name}; found {len(_runs)}."
        )
    hardware_run_path = _runs[hardware_repeat.value - 1].parent
    hardware_replay = read_replay(hardware_run_path)
    return (hardware_replay,)


@app.cell(hide_code=True)
def hardware_replay_window(hardware_replay):
    hardware_end_s = float(hardware_replay.time[-1])
    return (hardware_end_s,)


@app.cell(hide_code=True)
def hardware_nominal_replay(
    hardware_end_s,
    hardware_replay,
    nominal_damping,
    simulate_motor_replay,
):
    hardware_nominal = simulate_motor_replay(
        hardware_replay,
        (0.0,) * 7,
        nominal_damping,
        hardware_end_s,
    )
    return (hardware_nominal,)


@app.cell(hide_code=True)
def hardware_tuned_replay(
    hardware_end_s,
    hardware_replay,
    j1_armature,
    j1_damping,
    j2_armature,
    j2_damping,
    j3_armature,
    j3_damping,
    j4_armature,
    j4_damping,
    j5_armature,
    j5_damping,
    j6_armature,
    j6_damping,
    j7_armature,
    j7_damping,
    mo,
    simulate_motor_replay,
):
    _armatures = (
        j1_armature.value,
        j2_armature.value,
        j3_armature.value,
        j4_armature.value,
        j5_armature.value,
        j6_armature.value,
        j7_armature.value,
    )
    _dampings = (
        j1_damping.value,
        j2_damping.value,
        j3_damping.value,
        j4_damping.value,
        j5_damping.value,
        j6_damping.value,
        j7_damping.value,
    )
    mo.stop(None in _armatures or None in _dampings)
    hardware_tuned = simulate_motor_replay(
        hardware_replay,
        tuple(value for value in _armatures if value is not None),
        tuple(value for value in _dampings if value is not None),
        hardware_end_s,
    )
    return (hardware_tuned,)


@app.cell(hide_code=True)
def hardware_replay_joint_grid(
    Figure,
    hardware_case,
    hardware_nominal,
    hardware_replay,
    hardware_sim_offset,
    hardware_tuned,
    mo,
    show_hardware_nominal,
):
    hardware_grid = Figure(figsize=(10, 8), layout="constrained")
    _axes = hardware_grid.subplots(4, 2, sharex=True, squeeze=False)
    hardware_grid.delaxes(_axes.flat[7])
    _t = hardware_replay.time
    _offset_ms = hardware_sim_offset.value
    _shift_s = 0.0 if _offset_ms is None else _offset_ms * 1e-3
    _sim_t = _t + _shift_s
    _axes.flat[0].set_xlim(min(0.0, _shift_s), max(_t[-1], _sim_t[-1]))
    for _joint, _ax in enumerate(_axes.flat[:7]):
        _ax.plot(
            _t,
            hardware_replay.output_q[:, _joint],
            ":",
            color="0.5",
            label="Motor command",
        )
        _ax.plot(
            _t,
            hardware_replay.measured_q[:, _joint],
            color="black",
            label="Hardware",
        )
        if show_hardware_nominal.value:
            _ax.plot(
                _sim_t,
                hardware_nominal[:, _joint],
                color="tab:blue",
                label="Nominal sim",
            )
        _ax.plot(
            _sim_t,
            hardware_tuned[:, _joint],
            "--",
            color="tab:orange",
            label="Tuned sim",
        )
        _ax.axvline(hardware_replay.motion_end_s, color="0.75", linewidth=1)
        _ax.set(ylabel=f"J{_joint + 1} [rad]")
        _ax.grid(alpha=0.25)
        _ax.tick_params(labelbottom=True)
    _axes.flat[0].legend(fontsize="small")
    hardware_grid.suptitle(
        f"Recorded motor-command replay: {hardware_case.value}"
    )
    hardware_grid.supxlabel("Time since recorded goal start [s]")
    mo.output.replace(hardware_grid)
    return


@app.cell(hide_code=True)
def calibration_model_comparison(
    build_calibration_comparison,
    hardware_case,
    hardware_method,
    hardware_nominal,
    hardware_repeat,
    hardware_replay,
    hardware_tuned,
    j1_tuned,
    j2_tuned,
    mo,
):
    calibration_comparison = build_calibration_comparison(
        j1_tuned, j2_tuned, hardware_replay, hardware_nominal, hardware_tuned
    )

    def _error_columns(row, unit):
        return {
            "Nominal error [" + unit + "]": row["Nominal error"],
            "Tuned error [" + unit + "]": row["Tuned error"],
            "Improvement [%]": row["Improvement [%]"],
        }

    def _peak_rows(metric_prefix, unit):
        return [
            {
                "Joint": row["Joint"],
                f"Hardware peak [{unit}]": row["Hardware peak"],
                f"Nominal peak [{unit}]": row["Nominal peak"],
                f"Tuned peak [{unit}]": row["Tuned peak"],
                **_error_columns(row, unit),
            }
            for row in calibration_comparison
            if str(row["Metric"]).startswith(metric_prefix)
        ]

    _ring_rows = [
        {"Joint": row["Joint"], **_error_columns(row, "rad")}
        for row in calibration_comparison
        if str(row["Metric"]).startswith("Ring-down")
    ]
    _test_rows = [
        {
            "Joint": row["Joint"],
            "Window": str(row["Metric"]).split(" position", 1)[0],
            **_error_columns(row, "rad"),
        }
        for row in calibration_comparison
        if row["Joint"] not in ("J1", "J2")
    ]

    def _table(rows):
        return mo.ui.table(
            rows, selection=None, pagination=False, show_data_types=False
        )

    mo.output.replace(
        mo.vstack(
            [
                mo.md(r"""
        ## Nominal vs. calibrated model

        Both simulations are compared with hardware at the same recorded
        timestamps from goal start, without manual plot offsets. Improvement
        is $100(1-E_{\mathrm{tuned}}/E_{\mathrm{nominal}})$ percent.
        Negative means worse; a blank means the nominal error is zero or
        the metric is unavailable.
        """),
                mo.md("""
        ### J1–J2: Dominant ring-down frequency

        The peak is estimated by interpolating the largest FFT bin and its
        neighbors on a log-amplitude scale within 0–20 Hz. At a band edge,
        the peak bin is used directly. Error is the absolute frequency
        difference from hardware.
        """),
                _table(_peak_rows("Dominant", "Hz")),
                mo.md("""
        ### J1–J2: FFT peak amplitude

        Uses the amplitude at each trace's own interpolated dominant peak.
        Improvement is the reduction in absolute amplitude error relative to
        hardware, not a reduction in vibration amplitude.
        """),
                _table(_peak_rows("FFT peak amplitude", "rad")),
                mo.md("""
        ### J1–J2: Ring-down position RMSE

        Uses each joint's calibration recording and the shared ring-down duration.
        """),
                _table(_ring_rows),
                mo.md(
                    "### J3–J7: Hardware replay position RMSE\n\n"
                    f"**Test:** `{hardware_case.value}` / `{hardware_method.selected_key}` "
                    f"/ repeat {hardware_repeat.value}\n\n"
                    "RMSE is weighted by elapsed time. The windows split at command motion end "
                    "and cover the full usable recording."
                ),
                _table(_test_rows),
            ]
        )
    )
    return


@app.cell(hide_code=True)
def shifted_comparison_helpers(FloatVector, comparison_row, np, replay_rmse):
    def shifted_replay_rmse(
        time: FloatVector,
        measured: FloatVector,
        nominal: FloatVector,
        tuned: FloatVector,
        start_s: float,
        end_s: float,
        *,
        nominal_shift_s: float = 0.0,
        tuned_shift_s: float = 0.0,
    ) -> tuple[float | None, float | None]:
        nominal_time = time + nominal_shift_s
        tuned_time = time + tuned_shift_s
        start = max(start_s, time[0], nominal_time[0], tuned_time[0])
        end = min(end_s, time[-1], nominal_time[-1], tuned_time[-1])
        if end <= start:
            return None, None
        # Include the exact overlap boundaries without extrapolating any trace.
        grid = np.r_[start, time[(time > start) & (time < end)], end]
        hardware_q = np.interp(grid, time, measured)
        return (
            replay_rmse(
                grid, np.interp(grid, nominal_time, nominal) - hardware_q
            ),
            replay_rmse(grid, np.interp(grid, tuned_time, tuned) - hardware_q),
        )

    def shifted_rmse_row(
        joint: int,
        window: str,
        offset_ms: float,
        errors: tuple[float | None, float | None],
    ) -> dict[str, str | float | None]:
        row = comparison_row(joint, window, *errors)
        return {
            "Joint": row["Joint"],
            "Window": window,
            "Offset [ms]": offset_ms,
            "Nominal RMSE [rad]": row["Nominal error"],
            "Tuned RMSE [rad]": row["Tuned error"],
            "Improvement [%]": row["Improvement [%]"],
        }

    return shifted_replay_rmse, shifted_rmse_row


@app.cell(hide_code=True)
def shifted_model_comparison(
    calibration_nominal,
    calibration_replays,
    hardware_case,
    hardware_method,
    hardware_nominal,
    hardware_repeat,
    hardware_replay,
    hardware_sim_offset,
    hardware_tuned,
    j1_offset,
    j1_tuned,
    j2_offset,
    j2_tuned,
    mo,
    shifted_replay_rmse,
    shifted_rmse_row,
):
    shifted_ring_rows: list[dict[str, str | float | None]] = []
    for _joint, (_tuned, _offset) in enumerate(
        (
            (j1_tuned, j1_offset.value),
            (j2_tuned, j2_offset.value),
        )
    ):
        _replay = calibration_replays[_joint]
        _nominal = calibration_nominal[_joint]
        _n = len(_nominal)
        _post = _replay.time[:_n] >= _replay.motion_end_s
        _time = _replay.time[:_n][_post]
        _ms = 0.0 if _offset is None else float(_offset)
        _errors = shifted_replay_rmse(
            _time,
            _replay.measured_q[:_n, _joint][_post],
            _nominal[_post, _joint],
            _tuned[_post, _joint],
            _replay.motion_end_s,
            float(_time[-1]),
            tuned_shift_s=_ms * 1e-3,
        )
        shifted_ring_rows.append(
            shifted_rmse_row(_joint, "Ring-down", _ms, _errors)
        )

    shifted_hardware_rows: list[dict[str, str | float | None]] = []
    _time = hardware_replay.time
    _end = hardware_replay.motion_end_s
    _offset = hardware_sim_offset.value
    _ms = 0.0 if _offset is None else float(_offset)
    for _joint in range(2, 7):
        for _window, _lo, _hi in (
            ("Commanded motion", float(_time[0]), _end),
            ("Post-command", _end, float(_time[-1])),
        ):
            _errors = shifted_replay_rmse(
                _time,
                hardware_replay.measured_q[:, _joint],
                hardware_nominal[:, _joint],
                hardware_tuned[:, _joint],
                _lo,
                _hi,
                nominal_shift_s=_ms * 1e-3,
                tuned_shift_s=_ms * 1e-3,
            )
            shifted_hardware_rows.append(
                shifted_rmse_row(_joint, _window, _ms, _errors)
            )

    mo.output.replace(
        mo.vstack(
            [
                mo.md("""
        ## Nominal vs. calibrated model — with manual time shifts

        Each row compares both models with hardware over the same overlapping
        times, without extrapolation. Positive offsets move simulation traces
        later. The shorter overlap can also affect RMSE, so these are not
        directly comparable with the unshifted full-window scores above.
        """),
                mo.md("""
        ### J1–J2: Shifted ring-down RMSE

        Only the tuned trace is shifted, using its joint's offset.
        """),
                mo.ui.table(
                    shifted_ring_rows,
                    selection=None,
                    pagination=False,
                    show_data_types=False,
                ),
                mo.md(
                    "### J3–J7: Shifted hardware replay RMSE\n\n"
                    f"**Test:** `{hardware_case.value}` / `{hardware_method.selected_key}` "
                    f"/ repeat {hardware_repeat.value}\n\n"
                    "Both nominal and tuned traces use the shared simulation offset."
                ),
                mo.ui.table(
                    shifted_hardware_rows,
                    selection=None,
                    pagination=False,
                    show_data_types=False,
                ),
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
