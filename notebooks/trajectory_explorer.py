import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np

    from me6404_openarm_shaping.configuration import load_robot_config
    from me6404_openarm_shaping.input_shaping import (
        design_ei_shaper,
        design_zv_shaper,
        design_zvd_shaper,
        residual_vibration_factor,
        suggest_shaper_parameters,
    )
    from me6404_openarm_shaping.robot_model import (
        LEFT_ARM_JOINT_NAMES,
        load_model,
        modal_properties,
        resolve_joint_indices,
    )
    from me6404_openarm_shaping.trajectories import waypoint_trajectory

    return (
        LEFT_ARM_JOINT_NAMES,
        design_ei_shaper,
        design_zv_shaper,
        design_zvd_shaper,
        load_model,
        load_robot_config,
        mo,
        modal_properties,
        mujoco,
        np,
        plt,
        residual_vibration_factor,
        resolve_joint_indices,
        suggest_shaper_parameters,
        waypoint_trajectory,
    )


@app.cell
def _(LEFT_ARM_JOINT_NAMES, load_model, load_robot_config, mo, np):
    config = load_robot_config()
    _limit_model = load_model()

    config.waypoints

    joint_min = np.empty(7)
    joint_max = np.empty(7)

    for i, name in enumerate(LEFT_ARM_JOINT_NAMES):
        lower, upper = _limit_model.joint(name).range
        joint_min[i] = lower
        joint_max[i] = upper

    default_joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    get_data, set_data = mo.state(
        [default_joint_pos, default_joint_pos], allow_self_loops=True
    )
    return config, default_joint_pos, get_data, joint_max, joint_min, set_data


@app.cell
def _(mo):
    mo.md("""
    # Trajectory Selection
    - Choose a **Trajectory preset** and click **Load preset** to replace the rows. You can still edit them manually.
    - Each column is restricted to the upper and lower limits of the joint.
    - Columns 0–6 represent J1–J7.
    - Each row is a waypoint. Drag the values to adjust.
    - Use **Waypoint to preview** to display the corresponding row (numbered from 1).
    - The preview defaults to the last row when you add or remove waypoints.

    Presets load waypoints only. Gains and reference limits come from the active
    `robot_config.yaml`. The model dropdown changes the simulation and modal
    analysis, not the waypoint editor. Reproducing a hardware design requires its
    recorded model and shaper settings.
    """)
    return


@app.cell(hide_code=True)
def trajectory_model_controls(mo):
    trajectory_model_selector = mo.ui.dropdown(
        options=["Nominal", "Calibrated"],
        value="Calibrated",
        label="MuJoCo model",
    )
    reload_model_calibration = mo.ui.button(
        label="Reload model values from YAML"
    )
    mo.vstack(
        [
            mo.hstack(
                [trajectory_model_selector, reload_model_calibration],
                justify="start",
                wrap=True,
            ),
            mo.md(
                "**Nominal:** 0.40 N·m·s/rad on each joint, no added armature.  "
                "**Calibrated:** passive damping and added armature from `robot_config.yaml`. Reload after exporting changes from the calibration notebook."
            ),
        ]
    )
    return reload_model_calibration, trajectory_model_selector


@app.cell(hide_code=True)
def trajectory_model_selection(
    load_model,
    load_robot_config,
    np,
    reload_model_calibration,
    trajectory_model_selector,
):
    _ = reload_model_calibration.value  # Re-read model parameters on demand.
    if trajectory_model_selector.value == "Calibrated":
        _model_settings = load_robot_config()
        model = load_model(
            _model_settings.passive_damping,
            added_armature=_model_settings.added_armature,
        )
    else:
        model = load_model(np.full(7, 0.4))
    return (model,)


@app.cell(hide_code=True)
def active_trajectory_model_parameters(
    load_model,
    mo,
    model,
    np,
    plt,
    resolve_joint_indices,
    trajectory_model_selector,
):
    _nominal = load_model()
    _, _dofs = resolve_joint_indices(model)
    _, _nominal_dofs = resolve_joint_indices(_nominal)
    _damping = model.dof_damping[_dofs]
    _base_armature = _nominal.dof_armature[_nominal_dofs]
    _armature = model.dof_armature[_dofs]
    _added_armature = _armature - _base_armature
    _joints = [f"J{i}" for i in range(1, 8)]

    _fig, (_damping_ax, _armature_ax) = plt.subplots(
        1, 2, figsize=(9, 3.4), layout="constrained"
    )
    _damping_bars = _damping_ax.bar(_joints, _damping, color="C0")
    _damping_ax.bar_label(
        _damping_bars, fmt="%.3g", padding=3, fontsize="small"
    )
    _damping_ax.set_ylabel("Passive damping [N·m·s/rad]")
    _armature_ax.bar(_joints, _base_armature, color="0.7", label="MJCF nominal")
    _armature_ax.bar(
        _joints,
        _added_armature,
        bottom=_base_armature,
        color="C1",
        label="Added armature",
    )
    _armature_ax.set_ylabel("Effective armature [kg·m²]")
    _armature_ax.legend(frameon=False, fontsize="small")
    if not np.any(_armature):
        _armature_ax.text(
            0.5,
            0.5,
            "Zero armature on all joints",
            transform=_armature_ax.transAxes,
            ha="center",
        )
        _armature_ax.set_ylim(0, 0.03)
    for _ax in (_damping_ax, _armature_ax):
        _ax.margins(y=0.2)
        _ax.grid(axis="y", alpha=0.25)
    _fig.suptitle(
        f"Active model parameters — {trajectory_model_selector.value}"
    )
    mo.output.replace(_fig)
    plt.close(_fig)
    return


@app.cell
def trajectory_preset_control(default_joint_pos, mo, set_data):
    _destinations = {
        "J2 only": [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "J2 + J4": [[0.0, -1.0, 0.0, 1.5, 0.0, 0.0, 0.0]],
        "medium-09": [
            [
                -1.034834,
                -1.521773,
                -0.701482,
                1.627131,
                -0.089773,
                -0.575398,
                0.09553,
            ],
            [
                0.528537,
                -1.855768,
                -0.398703,
                0.21,
                0.043395,
                -0.429131,
                0.935294,
            ],
        ],
        "wide-02": [
            [
                -1.277914,
                -0.734612,
                -1.360796,
                2.019373,
                -0.758611,
                -0.575398,
                -0.168337,
            ],
            [
                1.171412,
                -2.707822,
                -0.642116,
                0.267584,
                0.237352,
                -0.432009,
                0.974402,
            ],
        ],
        "wide-07": [
            [
                0.638348,
                -2.487847,
                -1.280695,
                0.21,
                0.397228,
                0.122155,
                1.244187,
            ],
        ],
    }
    trajectory_preset = mo.ui.dropdown(
        options=list(_destinations), value="J2 only", label="Trajectory preset"
    )

    def _load_preset(_click_count: int) -> None:
        rows = _destinations[trajectory_preset.value]
        set_data([default_joint_pos.copy(), *(row.copy() for row in rows)])

    load_preset_button = mo.ui.button(
        label="Load preset", on_click=_load_preset
    )
    mo.output.replace(
        mo.hstack([trajectory_preset, load_preset_button], justify="start")
    )
    return


@app.cell
def _(default_joint_pos, get_data, joint_max, joint_min, mo, set_data):
    data = get_data()
    num_rows = len(data)

    add_row_btn = mo.ui.button(
        label="Add Row",
        on_click=lambda _: set_data(
            lambda _: [*matrix_ui.value, list(default_joint_pos)]
        ),
    )

    remove_row_btn = mo.ui.button(
        label="Remove Last Row",
        on_click=lambda _: set_data(
            lambda _: (
                matrix_ui.value[:-1]
                if len(matrix_ui.value) > 1
                else matrix_ui.value
            )
        ),
    )

    matrix_ui = mo.ui.matrix(
        get_data(),
        min_value=[list(joint_min) for _ in range(num_rows)],
        max_value=[list(joint_max) for _ in range(num_rows)],
        step=0.01,
        precision=3,
    )
    mo.vstack(
        [mo.hstack([add_row_btn, remove_row_btn], justify="start"), matrix_ui]
    )
    return matrix_ui, num_rows


@app.cell
def _(config, matrix_ui, np, waypoint_trajectory):
    config.waypoints = np.asarray(matrix_ui.value)

    t, q, dq, ddq = waypoint_trajectory(
        config.waypoints,
        config.velocity_limits,
        config.acceleration_limits,
        dt=config.simulation.dt_s,
    )
    return ddq, dq, q, t


@app.cell
def waypoint_preview_control(mo, num_rows):
    waypoint_preview_selector = mo.ui.dropdown(
        options={f"Waypoint {index + 1}": index for index in range(num_rows)},
        value=f"Waypoint {num_rows}",
        label="Waypoint to preview",
    )
    mo.output.replace(waypoint_preview_selector)
    return (waypoint_preview_selector,)


@app.cell
def waypoint_pose_preview(
    matrix_ui,
    mo,
    model,
    mujoco,
    plt,
    resolve_joint_indices,
    waypoint_preview_selector,
):
    # Set the selected left-arm pose without stepping the simulation.
    _waypoint_index = waypoint_preview_selector.value
    _preview_data = mujoco.MjData(model)
    _qpos_indices, _ = resolve_joint_indices(model)
    _preview_data.qpos[_qpos_indices] = matrix_ui.value[_waypoint_index]
    mujoco.mj_forward(model, _preview_data)

    # Frame the camera around the displayed pose.
    _shoulder = _preview_data.body("openarm_left_link1").xpos
    _hand = _preview_data.body("openarm_left_hand").xpos
    _camera = mujoco.MjvCamera()
    _camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    _camera.lookat[:] = (_shoulder + _hand) / 3
    _camera.distance = 1.5
    _camera.azimuth = 150.0
    _camera.elevation = -20.0

    with mujoco.Renderer(model, height=480, width=640) as _renderer:
        _renderer.update_scene(_preview_data, camera=_camera)
        _rgb = _renderer.render()

    _fig, _ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    _ax.imshow(_rgb)
    _ax.set_axis_off()
    _ax.set_title(f"Waypoint {_waypoint_index + 1} — OpenArm")
    mo.output.replace(_fig)
    plt.close(_fig)
    return


@app.cell
def trajectory_frequencies(mo):
    mo.md("""
    # Trajectory Modal Properties
    Each branch starts in natural-frequency order, then follows the best
        one-to-one match between displacement mode shapes (complex MAC).
        Gaps indicate absent oscillatory modes. Open markers flag provisional
        matches: MAC below 0.80 or an assignment-score gap below 0.02.
        Tracking changes the plot labels, not the frozen model or shaper design.
    """)
    return


@app.cell
def track_trajectory_modes(config, model, q):
    from me6404_openarm_shaping.mode_tracking import track_modes

    tracked_modes = track_modes(model, q, config.motor_kp, config.motor_kd)
    return (tracked_modes,)


@app.cell
def tracked_trajectory_modal_plot(mo, np, plt, t, tracked_modes):
    _fig, (_natural_ax, _zeta_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), layout="constrained"
    )

    _unstable_times = t[~tracked_modes.stable]
    _branch_count = tracked_modes.poles.shape[1]
    _weak = (tracked_modes.mac < 0.80) | (
        (tracked_modes.assignment_gap[:, None] < 0.02)
        & np.isfinite(tracked_modes.mac)
    )

    for _ax, _samples, _ylabel in (
        (
            _natural_ax,
            tracked_modes.natural_frequency_hz,
            "Natural frequency [Hz]",
        ),
        (_zeta_ax, tracked_modes.damping_ratio, "Damping ratio ζ [−]"),
    ):
        for _branch, _values in enumerate(_samples.T):
            _color = f"C{_branch % 10}"
            _flagged = _weak[:, _branch] & np.isfinite(_values)
            _regular = np.isfinite(_values) & ~_flagged
            _ax.scatter(
                t[_regular],
                _values[_regular],
                s=8,
                color=_color,
                label=f"Branch {_branch + 1}",
            )
            _ax.scatter(
                t[_flagged],
                _values[_flagged],
                s=32,
                facecolors="none",
                edgecolors=_color,
            )
        if _weak.any():
            _ax.scatter(
                [],
                [],
                s=32,
                facecolors="none",
                edgecolors="0.4",
                label="Provisional match",
            )
        if len(_unstable_times) > 0:
            _ax.scatter(
                _unstable_times,
                np.full_like(_unstable_times, 0.95),
                transform=_ax.get_xaxis_transform(),
                marker="x",
                color="red",
                label="Unstable/marginal",
            )
        _ax.set_ylabel(_ylabel)
        _ax.grid(True)

    _natural_ax.set_title(
        "Frozen configuration, zero velocity — MAC-tracked modes"
    )
    if _branch_count > 0 or len(_unstable_times) > 0:
        _natural_ax.legend(title="Tracked branch", ncols=3, fontsize="small")
    _zeta_ax.set_xlabel("Trajectory time [s]")
    mo.output.replace(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    # Input Shaping
    ZV, ZVD, and EI shapers are available for time-invariant methods.
    """)
    return


@app.cell
def _(config, mo, modal_properties, model, np, q):
    model.opt.timestep = config.simulation.dt_s

    properties = [
        modal_properties(model, configuration, config.motor_kp, config.motor_kd)
        for configuration in q
    ]

    omega_n_hz = np.concatenate(
        [result["natural_frequency_hz"] for result in properties]
    )

    zeta = np.concatenate([result["damping_ratio"] for result in properties])

    tau = 1 / (zeta * omega_n_hz * 2 * np.pi)

    shaper_selection_dropdown = mo.ui.dropdown(
        options=["ZV", "ZVD", "EI"],
        label="Select a time-invariant shaper",
        value="ZVD",
    )
    shaper_selection_dropdown
    return omega_n_hz, properties, shaper_selection_dropdown, tau, zeta


@app.cell
def _(
    config,
    mo,
    properties,
    shaper_selection_dropdown,
    suggest_shaper_parameters,
):
    tau_min = 0.0
    frequency_tolerance = 0.0
    utilization_min = 0.0

    if shaper_selection_dropdown.value == "ZVD":
        tau_min = config.input_shaping.time_invariant.zvd.tau_min_s
        frequency_tolerance = 0.14
        utilization_min = (
            config.input_shaping.time_invariant.zvd.utilization_min
        )
    elif shaper_selection_dropdown.value == "ZV":
        tau_min = config.input_shaping.time_invariant.zv.tau_min_s
        frequency_tolerance = 0.03
        utilization_min = config.input_shaping.time_invariant.zv.utilization_min
    else:
        tau_min = config.input_shaping.time_invariant.ei.tau_min_s
        frequency_tolerance = 0.2
        utilization_min = config.input_shaping.time_invariant.ei.utilization_min

    candidate_frequencies, candidate_damping, candidate_utilization = (
        suggest_shaper_parameters(
            modal_samples=properties,
            tau_min=tau_min,
            frequency_tolerance=frequency_tolerance,
            utilization_min=0.0,
        )
    )

    mo.md(f"""
    - Input shaping candidates:
        - Shaper natural frequencies [Hz]: `{candidate_frequencies}`
        - Shaper zeta: `{candidate_damping}`
        - Frequency-band utilization: `{candidate_utilization}`
    """)
    return (
        candidate_damping,
        candidate_frequencies,
        candidate_utilization,
        frequency_tolerance,
        tau_min,
        utilization_min,
    )


@app.cell
def _(
    candidate_damping,
    candidate_frequencies,
    candidate_utilization,
    utilization_min,
):
    selected = candidate_utilization >= utilization_min
    omega_n_nominal_hz_suggestion = candidate_frequencies[selected]
    zeta_nominal_suggestion = candidate_damping[selected]
    frequency_utilization = candidate_utilization[selected]
    return (
        frequency_utilization,
        omega_n_nominal_hz_suggestion,
        selected,
        zeta_nominal_suggestion,
    )


@app.cell
def pooled_modal_number_lines(
    candidate_frequencies,
    frequency_tolerance,
    mo,
    np,
    omega_n_hz,
    plt,
    selected,
    tau,
    tau_min,
    zeta,
):
    _retained = tau >= tau_min
    _fig, (_frequency_ax, _zeta_ax) = plt.subplots(
        2, 1, figsize=(8, 3.5), layout="constrained"
    )

    for _ax, _values, _xlabel in (
        (_frequency_ax, omega_n_hz, "Natural frequency [Hz]"),
        (_zeta_ax, zeta, "Damping ratio ζ [−]"),
    ):
        for _mask, _color, _label in (
            (_retained, "green", "Meets threshold"),
            (~_retained, "red", "Below threshold"),
        ):
            _positions = _values[_mask]
            _ax.scatter(
                _positions,
                np.full_like(_positions, 0.15),
                color=_color,
                s=8,
                alpha=0.3,
                label=_label,
            )
        _ax.spines["bottom"].set_position(("data", 0))
        for _side in ("left", "right", "top"):
            _ax.spines[_side].set_visible(False)
        _ax.set_yticks([])
        _ax.set_ylim(-1, 1)
        _ax.set_xlabel(_xlabel)

    # Alternate heights in frequency order so nearby end ticks stay distinct.
    _band_heights = np.where(
        np.arange(candidate_frequencies.size) % 2 == 0, 0.4, 0.65
    )
    for _mask, _color, _label in (
        (selected, "green", "Suggested band"),
        (~selected, "black", "Other candidate band"),
    ):
        _centers = candidate_frequencies[_mask]
        if _centers.size == 0:
            continue
        _frequency_ax.errorbar(
            _centers,
            _band_heights[_mask],
            xerr=_centers * frequency_tolerance,
            fmt="none",
            ecolor=_color,
            capsize=4,
            label=_label,
        )

    _frequency_ax.set_title(
        f"Pooled modal samples — decay-time threshold: {tau_min:g} s"
    )
    _frequency_ax.legend(
        loc="upper right", ncols=4, fontsize="small", frameon=False
    )
    mo.output.replace(_fig)
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def selected_shaper_design(
    config,
    design_ei_shaper,
    design_zv_shaper,
    design_zvd_shaper,
    omega_n_nominal_hz_suggestion,
    shaper_selection_dropdown,
    zeta_nominal_suggestion,
):
    selected_shaper = shaper_selection_dropdown.value
    _frequencies = omega_n_nominal_hz_suggestion
    _damping = zeta_nominal_suggestion

    if selected_shaper == "ZV":
        shaper_amplitudes, shaper_delays_s = design_zv_shaper(
            _frequencies, _damping
        )
    elif selected_shaper == "ZVD":
        shaper_amplitudes, shaper_delays_s = design_zvd_shaper(
            _frequencies, _damping
        )
    else:
        shaper_amplitudes, shaper_delays_s = design_ei_shaper(
            _frequencies,
            _damping,
            residual_tolerance=config.input_shaping.time_invariant.ei.residual_tolerance,
        )
    return selected_shaper, shaper_amplitudes, shaper_delays_s


@app.cell
def residual_vibration_summary(
    frequency_utilization,
    mo,
    np,
    omega_n_hz,
    omega_n_nominal_hz_suggestion,
    residual_vibration_factor,
    selected_shaper,
    shaper_amplitudes,
    shaper_delays_s,
    tau,
    tau_min,
    zeta,
    zeta_nominal_suggestion,
):
    _retained = tau >= tau_min
    residual_frequency_hz = omega_n_hz[_retained]
    residual_percent = np.empty(0)
    _mode_count = omega_n_nominal_hz_suggestion.size
    _impulse_count = 0
    _delay = "—"
    _max_residual = "—"
    _note = ""

    _explanation = r"""
    ## Theoretical residual vibration

    For each retained frozen mode, the residual at the final impulse is

    $$
    R_{\%} = 100\left|\sum_i A_i
        e^{-\zeta\omega_n(T-t_i) + \mathrm{i}\omega_d t_i}\right|,
    \qquad \omega_n=2\pi f_n,\quad
    \omega_d=\omega_n\sqrt{1-\zeta^2},\quad T=\max_i t_i.
    $$

    Here $A_i$ and $t_i$ are the combined filter's impulse weights and times, and
    $f_n$ and $\zeta$ come from the same retained modal sample. This uses ideal,
    continuous impulse times; it is not a vibration simulation.
    """

    if _mode_count == 0:
        _note = "No shaper was suggested, so there is no residual plot or filter delay to report."
    else:
        residual_percent = 100 * np.array(
            [
                residual_vibration_factor(
                    shaper_amplitudes,
                    shaper_delays_s,
                    2 * np.pi * frequency,
                    damping,
                )
                for frequency, damping in zip(
                    residual_frequency_hz, zeta[_retained], strict=True
                )
            ]
        )

        _impulse_count = shaper_amplitudes.size
        _delay = f"{shaper_delays_s.max():.6f} s"
        _max_residual = f"{residual_percent.max():.3f}%"

    _frequencies = ", ".join(
        f"{frequency:.3f}" for frequency in omega_n_nominal_hz_suggestion
    )
    _ratios = ", ".join(f"{ratio:.4f}" for ratio in zeta_nominal_suggestion)
    _utilizations = ", ".join(
        f"{utilization:.1%}" for utilization in frequency_utilization
    )
    _summary = f"""
    ## Shaper design summary

    | Property | Value |
    | :--- | :--- |
    | Shaper | {selected_shaper} |
    | Selected nominal frequencies [Hz] | {_frequencies or "—"} |
    | Selected damping ratios ζ | {_ratios or "—"} |
    | Frequency-band utilization | {_utilizations or "—"} |
    | Single-mode shapers convolved | {_mode_count} |
    | Resulting impulse count (before merging) | {_impulse_count} |
    | Total shaper delay | {_delay} |
    | Maximum theoretical residual | {_max_residual} |

    {_note}
    """

    mo.md(_summary + "\n\n" + _explanation)
    return residual_frequency_hz, residual_percent


@app.cell
def residual_vibration_plot(mo, plt, residual_frequency_hz, residual_percent):
    if residual_percent.size > 0:
        _fig, _ax = plt.subplots(figsize=(8, 4), layout="constrained")
        _ax.scatter(
            residual_frequency_hz,
            residual_percent,
            s=8,
            alpha=0.5,
            label="Retained modal samples",
        )
        _ax.axhline(5.0, linestyle="--", color="red", label="5% threshold")
        _ax.set_xlabel("Natural frequency [Hz]")
        _ax.set_ylabel("Residual vibration [%]")
        _ax.set_title("Suggested combined shaper — ideal impulse delays")
        _ax.grid(True)
        _ax.legend()
        mo.output.replace(_fig)
        plt.close(_fig)
    return


@app.cell(hide_code=True)
def simulation_comparison_heading(mo):
    mo.md("""
    # MuJoCo simulation comparison

    The baseline uses the same command as the modal analysis above. The shaped
    command uses the selected filter with impulse delays rounded to the simulation
    time step. Both trials start at the first waypoint and run for the same time,
    with a settling window after the longer command ends.

    Offline simulation only; these results do not authorize a hardware trial.
    """)
    return


@app.cell(hide_code=True)
def simulate_selected_shaper(
    config,
    ddq,
    dq,
    model,
    omega_n_nominal_hz_suggestion,
    q,
    selected_shaper,
    shaper_amplitudes,
    shaper_delays_s,
    t,
):
    from me6404_openarm_shaping.data_types import JointTrajectory
    from me6404_openarm_shaping.simulation_helpers import run_trial
    from me6404_openarm_shaping.trajectories import (
        apply_shaper_to_joint_trajectory,
    )

    _commands: dict[str, JointTrajectory] = {
        "baseline": {"t": t, "q": q, "dq": dq, "ddq": ddq}
    }
    _settings = config.simulation
    if omega_n_nominal_hz_suggestion.size > 0:
        _t_shaped, _q_shaped, _dq_shaped, _ddq_shaped = (
            apply_shaper_to_joint_trajectory(
                q, dq, ddq, shaper_amplitudes, shaper_delays_s, _settings.dt_s
            )
        )
        _commands[f"time_invariant_{selected_shaper.lower()}"] = {
            "t": _t_shaped,
            "q": _q_shaped,
            "dq": _dq_shaped,
            "ddq": _ddq_shaped,
        }

    simulation_duration_s = max(
        _settings.minimum_duration_s,
        max(float(_command["t"][-1]) for _command in _commands.values())
        + _settings.settling_time_s,
    )
    simulation_recordings = {
        _label: run_trial(
            model,
            _command,
            q[0],
            config.motor_kp,
            config.motor_kd,
            simulation_duration_s,
            config=config,
            headless=True,
        )
        for _label, _command in _commands.items()
    }

    simulation_trajectories = _commands
    return (
        simulation_duration_s,
        simulation_recordings,
        simulation_trajectories,
    )


@app.cell(hide_code=True, expand_output=True)
def simulation_comparison_plots(
    mo,
    plt,
    selected_shaper,
    simulation_duration_s,
    simulation_recordings,
    simulation_trajectories,
):
    from me6404_openarm_shaping.simulation_helpers import plot_comparison

    _lines = [f"Common simulation duration: **{simulation_duration_s:.3f} s**."]
    if len(simulation_recordings) == 1:
        _lines.append(
            f"No {selected_shaper} modes qualified; showing baseline only."
        )
    for _label, _recording in simulation_recordings.items():
        _checks = _recording["simulation_checks"]
        _status = "pass" if _checks["passing"] else "fail"
        _issues = ", ".join(
            sorted({_issue["kind"] for _issue in _checks["reasons"]})
        )
        _details = f" ({_issues})" if len(_issues) > 0 else ""
        _name = "Baseline" if _label == "baseline" else selected_shaper
        _end_s = float(simulation_trajectories[_label]["t"][-1])
        _lines.append(
            f"- **{_name}:** command ends at {_end_s:.3f} s; simulation checks: {_status}{_details}."
        )

    # The CLI helper opens pyplot figures; display only these new figures.
    _existing = set(plt.get_fignums())
    plot_comparison(simulation_trajectories, simulation_recordings)
    _figures = [
        plt.figure(_num) for _num in plt.get_fignums() if _num not in _existing
    ]
    try:
        _tabs = {_fig.get_label(): mo.as_html(_fig) for _fig in _figures}
        mo.output.replace(
            mo.vstack([mo.md("\n\n".join(_lines)), mo.ui.tabs(_tabs)])
        )
    finally:
        for _fig in _figures:
            plt.close(_fig)
    return


@app.cell(hide_code=True)
def simulation_percentage_metrics(
    np,
    selected_shaper,
    simulation_recordings,
    simulation_trajectories,
):
    from me6404_openarm_shaping.data_types import TrialRecording

    def _whole_arm_residual_rms(
        recording: TrialRecording, command_end_s: float
    ) -> float | None:
        """Joint-space RMS about the final 0.5 s mean, using an exact 0.8 s window."""
        time = recording["t"]
        position = recording["q"]
        window_s = 0.8
        end_s = command_end_s + window_s
        if time.size == 0 or time[0] > command_end_s or time[-1] < end_s:
            return None

        equilibrium = position[time >= time[-1] - 0.5].mean(axis=0)
        interior = (time > command_end_s) & (time < end_s)
        sample_time = np.r_[command_end_s, time[interior], end_s]
        edges = np.array(
            [
                np.interp([command_end_s, end_s], time, trace)
                for trace in position.T
            ]
        ).T
        residual = (
            np.vstack((edges[0], position[interior], edges[1])) - equilibrium
        )
        squared_norm = np.sum(residual**2, axis=1)
        return float(
            np.sqrt(np.trapezoid(squared_norm, x=sample_time) / window_s)
        )

    _shaped_label = f"time_invariant_{selected_shaper.lower()}"
    _baseline_duration = float(simulation_trajectories["baseline"]["t"][-1])
    _baseline_rms = _whole_arm_residual_rms(
        simulation_recordings["baseline"], _baseline_duration
    )
    _shaped_duration = None
    _shaped_rms = None
    if _shaped_label in simulation_recordings:
        _shaped_duration = float(
            simulation_trajectories[_shaped_label]["t"][-1]
        )
        _shaped_rms = _whole_arm_residual_rms(
            simulation_recordings[_shaped_label], _shaped_duration
        )

    _duration_increase = (
        None
        if _baseline_duration <= 0 or _shaped_duration is None
        else 100.0 * (_shaped_duration / _baseline_duration - 1.0)
    )
    _residual_reduction = (
        None
        if _baseline_rms is None or _baseline_rms <= 0 or _shaped_rms is None
        else 100.0 * (1.0 - _shaped_rms / _baseline_rms)
    )
    simulation_residual_comparison = (
        _baseline_rms,
        _shaped_rms,
        _residual_reduction,
    )
    simulation_duration_comparison = (
        _baseline_duration,
        _shaped_duration,
        _duration_increase,
    )
    return simulation_duration_comparison, simulation_residual_comparison


@app.cell(hide_code=True, expand_output=True)
def simulation_percentage_tables(
    mo,
    selected_shaper,
    simulation_duration_comparison,
    simulation_residual_comparison,
):
    def _display_value(value: float | None, decimals: int) -> str:
        return "Unavailable" if value is None else f"{value:.{decimals}f}"

    _residual_caption = (
        "Whole-arm position RMS over the first 0.8 s after each command ends, "
        "relative to the mean simulated joint positions over the final 0.5 s. "
        "Negative values mean more vibration. A missing window or zero baseline RMS "
        "makes the percentage unavailable."
    )

    _duration_caption = (
        "Commanded motion only, excluding the settling hold. "
        "Negative values mean a shorter command. A zero baseline duration "
        "makes the percentage unavailable."
    )

    _outputs = [mo.md("## Simulated shaping tradeoff")]
    for _title, _units, _decimals, _metrics, _caption in (
        (
            "Residual vibration reduction",
            "rad",
            6,
            simulation_residual_comparison,
            _residual_caption,
        ),
        (
            "Command duration increase",
            "s",
            3,
            simulation_duration_comparison,
            _duration_caption,
        ),
    ):
        _baseline, _shaped, _change = _metrics
        _outputs.extend(
            [
                mo.md(f"### {_title} vs. baseline\n\n{_caption}"),
                mo.ui.table(
                    [
                        {
                            f"Baseline [{_units}]": _display_value(
                                _baseline, _decimals
                            ),
                            f"{selected_shaper} [{_units}]": _display_value(
                                _shaped, _decimals
                            ),
                            "Change vs. baseline [%]": _display_value(
                                _change, 1
                            ),
                        }
                    ],
                    selection=None,
                    pagination=False,
                ),
            ]
        )
    mo.output.replace(mo.vstack(_outputs))
    return


if __name__ == "__main__":
    app.run()
