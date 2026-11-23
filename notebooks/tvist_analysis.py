import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import csv
    from pathlib import Path
    from runpy import run_path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from me6404_openarm_shaping.input_shaping import zvd_shaper

    return Path, csv, mo, np, plt, run_path, yaml, zvd_shaper


@app.cell(hide_code=True)
def tvist_formulation(Path, mo):
    _figure_dir = Path(__file__).resolve().parents[1] / "figures"
    mo.vstack(
        [
            mo.md(r"""
    # Time-varying input shaping (TVIST)

    ## Formulation: output-side fractional-delay filtering

    ### 1. Time-varying impulse generation

    Thomsen et al. [1, §2.1, Eqs. (4)–(8)] give the conventional ZV design.
    TVIST updates the same parameters using the scheduled natural frequency
    $\omega_n(t)$ in rad/s and the damping ratio $\zeta(t)$:

    $$
    \omega_d(t)=\omega_n(t)\sqrt{1-\zeta^2(t)},\qquad
    K(t)=e^{-\zeta(t)\pi/\sqrt{1-\zeta^2(t)}},\qquad
    h(t)=\frac{\pi}{\omega_d(t)}.
    $$

    Here $h(t)$ is the damped half-period.

    $$
    \begin{aligned}
    A_{\mathrm{ZV}} &= \frac{[1,K]}{1+K},
    &\Delta_{\mathrm{ZV}} &= [0,h],\\
    A_{\mathrm{ZVD}} &= \frac{[1,2K,K^2]}{(1+K)^2},
    &\Delta_{\mathrm{ZVD}} &= [0,h,2h].
    \end{aligned}
    $$

    The paper's numerical examples use ZV. The OpenArm implementation
    schedules the conventional ZV and ZVD design.

    ### 2. Shape position, then differentiate

    Following [1, §2.2, Fig. 6], shape the planned position and derive the other
    references from that same command:

    $$
    q_d(t)\xrightarrow{\mathrm{TVIST}}q_s(t)
    \xrightarrow{\mathrm{differentiate}}\dot q_s(t)
    \xrightarrow{\mathrm{differentiate}}\ddot q_s(t).
    $$

    Position is already available from the trajectory planner. Starting there
    avoids reconstructing position by integrating a shaped acceleration. 

    More importantly, time-varying filtering and differentiation generally do not
    commute: independently shaping position, velocity, and acceleration usually do not produce consistent traces. Differentiating $q_s$ preserves that relationship
    numerically, but does not guarantee bounded acceleration.
    """),
            mo.image(
                _figure_dir / "paper_fig6.png",
                alt="Thomsen et al. Fig. 6: TVIST shapes the position reference before differentiation in the proposed robot control architecture.",
                width="50%",
                style={"display": "block", "margin-inline": "auto"},
            ),
            mo.md(r"""
    ### 3. Output-side algorithm and fractional delays

    For an integer-delay FIR filter, the output-side algorithm (OSA) is
    [1, §6, Eq. (24)]:

    $$
    x^*[i] = \sum_{k=0}^{M} I^i[k] \, x[i-k].
    $$

    At output sample $i$, $I^i[k]$ come from the **current**
    shaper and act on the unshaped input history $x[i-k]$. 

    The input-side algorithm (ISA) instead schedules contributions from each input sample using the shaper at that input time. 

    Contributions generated under different shapers can then
    mix in one output sample, which can introduce irregularities as parameters
    change [1, §§5–6]. 

    OSA avoids that mixing, but does not by itself ensure a
    smooth command when the schedule changes abruptly.

    Desired delays are generally not integer multiples of the sample period $T_s$.
    Write $d=\Delta/T_s$, $k=\lfloor d\rfloor$, and $D=d-k$. With
    `lagrange_order=1`, `sample_delayed()` uses two history samples:

    $$
    \hat q_d(t_i-\Delta)=(1-D)q_d[i-k]+Dq_d[i-k-1].
    $$

    """),
            mo.hstack(
                [
                    mo.md(r"""
    **Table 1.** Coefficients of Lagrange interpolation (Laakso et al., 1996;
    as reproduced by Thomsen et al. [1, Table 1]).

    | Order | $k$ | $k+1$ | $k+2$ | $k+3$ |
    | --- | --- | --- | --- | --- |
    | 1st | $1-D$ | $D$ | — | — |
    | 2nd | $(D-1)(D-2)/2$ | $-D(D-2)$ | $D(D-1)/2$ | — |
    | 3rd | $-(D-1)(D-2)(D-3)/6$ | $D(D-2)(D-3)/2$ | $-D(D-1)(D-3)/2$ | $D(D-1)(D-2)/6$ |
    """).style({"max-width": "100%", "overflow-x": "auto"}),
                ],
                justify="center",
            ),
            mo.md(r"""
    The OpenArm comparisons use first-order interpolation currently, but supports up to third-order.

    For example, a 0.253 s delay at $T_s=0.002$ s is 126.5 samples, so the lookup
    averages $q_d[i-126]$ and $q_d[i-127]$ instead of rounding the delay. See
    [1, Fig. 19] for the fractional-delay processing pipeline.
    """),
            mo.image(
                _figure_dir / "paper_fig19.png",
                alt="Thomsen et al. Fig. 19: the fractional-delay filter interpolates buffered input and combines it with the current impulse weights and delays.",
                width="50%",
                style={"display": "block", "margin-inline": "auto"},
            ),
            mo.md(r"""
    ### 4. OpenArm TVIST

    1. `robot_sim.py` computes modal frequencies and damping offline along the
       **unshaped reference**. `select_tvist_modes()` selects eligible modes (defaults to all modes) at each
       sample. The scheduled frequencies may differ from those along the shaped path.
    2. At each output sample, `apply_tvist_to_joint_trajectory()` designs a shaper
       for each selected mode and convolves them into one set of impulses.
       The same weights and delays are applied separately to each joint's position
       history. The lookup is causal given the schedule.
    3. The filter holds the initial position for lookups before the first sample,
       and holds the final position and modal schedule through the filter tail.
       `robot_sim.py` applies `np.gradient()` twice to obtain the velocity and
       acceleration references.

    Combining the current weights and fractional-delay lookups gives the actual
    position update used in the OpenArm simulations:

    $$
    q_s(t_i)=\sum_{j=1}^{N_i} A_j(t_i)\,\hat q_d\bigl(t_i-\Delta_j(t_i)\bigr).
    $$

    **[1]** Thomsen et al. (2019), *Smooth Online Time-Varying Input Shaping with
    Fractional Delay FIR Filtering*, Control Engineering Practice 88, 21–37.
    [DOI: 10.1016/j.conengprac.2019.04.003](https://doi.org/10.1016/j.conengprac.2019.04.003).
    """),
        ]
    )
    return


@app.cell
def oscillator_experiments(Path, np, run_path, zvd_shaper):
    # Load the existing experiment by path without invoking its CLI.
    _experiment = run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "tvist_oscillator_experiment.py"
        )
    )
    _dt = _experiment["DT_S"]
    oscillator_damping = _experiment["DAMPING_RATIO"]
    oscillator_rms_window = _experiment["RESIDUAL_WINDOW_S"]
    _window_samples = round(oscillator_rms_window / _dt)
    oscillator_cases = {}
    oscillator_metrics = []
    for _final_hz in (2.0, 3.0):
        _plant_hz, _commands = _experiment["experiment_commands"](_final_hz)
        if _final_hz == 2.0:
            # Independent fixed fractional-delay reference on the same output grid.
            _input = _commands["Unshaped"]
            _output_t = np.arange(_commands["TVIST ZVD"].size) * _dt
            _input_t = np.arange(_input.size) * _dt
            _weights, _delays = zvd_shaper(2 * np.pi * 2.0, oscillator_damping)
            _fixed = sum(
                weight * np.interp(_output_t - delay, _input_t, _input)
                for weight, delay in zip(_weights, _delays, strict=True)
            )
            constant_zvd_error = float(
                np.max(np.abs(_fixed - _commands["TVIST ZVD"]))
            )
            _commands["Fixed-start ZVD"] = _fixed
            del _commands[
                "Fixed-end ZVD"
            ]  # The two fixed designs coincide at 2 Hz.

        _count = max(map(len, _commands.values())) + _window_samples
        _t = np.arange(_count) * _dt
        _frequency = np.pad(
            _plant_hz, (0, _count - len(_plant_hz)), mode="edge"
        )
        _traces = {}
        _case = f"2 → {_final_hz:g} Hz"
        for _method, _command in _commands.items():
            # Match the experiment: exclude extra endpoint holds from command duration.
            _end = int(np.flatnonzero(np.abs(_command - 1.0) > 1e-12)[-1]) + 1
            _padded = np.pad(_command, (0, _count - len(_command)), mode="edge")
            _position, _ = _experiment["simulate_oscillator"](
                _padded, _frequency, oscillator_damping, _dt
            )
            _residual = _position - 1.0
            _window = _residual[_end + 1 : _end + 1 + _window_samples]
            oscillator_metrics.append(
                {
                    "Schedule": _case,
                    "Method": _method,
                    "Command duration [s]": float(_t[_end]),
                    "Residual position RMS": float(
                        np.sqrt(np.mean(_window**2))
                    ),
                }
            )
            _traces[_method] = (_padded, _residual, _end)
        oscillator_cases[_case] = (_t, _frequency, _traces)
    return (
        constant_zvd_error,
        oscillator_cases,
        oscillator_damping,
        oscillator_metrics,
        oscillator_rms_window,
    )


@app.cell(hide_code=True)
def oscillator_verification(
    constant_zvd_error,
    mo,
    np,
    oscillator_cases,
    oscillator_damping,
    oscillator_metrics,
    oscillator_rms_window,
    plt,
):
    _colors = {
        "Unshaped": "C0",
        "Fixed-start ZVD": "C1",
        "Fixed-end ZVD": "C2",
        "TVIST ZVD": "C4",
    }
    _fig, _axes = plt.subplots(
        4, 2, figsize=(11, 8), sharex="col", layout="constrained"
    )
    for _col, (_case, (_t, _frequency, _traces)) in enumerate(
        oscillator_cases.items()
    ):
        _axes[0, _col].set_title(_case)
        _axes[0, _col].plot(_t, _frequency, color="0.25")
        _axes[0, _col].set_ylim(1.85, 3.15)
        _period = 1 / (_frequency * np.sqrt(1 - oscillator_damping**2))
        _axes[1, _col].plot(_t, _period / 2, label="ZV", color="C3")
        _axes[1, _col].plot(_t, _period, label="ZVD", color="C4")
        _axes[1, _col].set_ylim(0.13, 0.54)
        for _method, (_command, _residual, _end) in _traces.items():
            _style = "--" if _method == "TVIST ZVD" else "-"
            _axes[2, _col].plot(
                _t, _command, _style, color=_colors[_method], label=_method
            )
            _axes[3, _col].plot(
                _t[_end:], _residual[_end:], _style, color=_colors[_method]
            )
    for _ax, _label in zip(
        _axes[:, 0],
        (
            "Natural frequency [Hz]",
            "Scheduled max delay [s]",
            "Position command",
            "Post-command error",
        ),
        strict=True,
    ):
        _ax.set_ylabel(_label)
    for _ax in _axes.flat:
        _ax.grid(alpha=0.25)
    _axes[1, 1].legend(loc="upper right", frameon=False)
    _axes[2, 1].legend(loc="lower right", frameon=False, fontsize="small")
    _fig.supxlabel("Time from command start [s]")
    mo.vstack(
        [
            mo.md(rf"""## Numerical verification: scalar oscillator

        Both experiments applies a 1 unit move with the same second-order plant model,
        $\ddot x + 2\zeta\omega_n(t)\dot x + \omega_n^2(t)(x-u)=0$.
        The damping ratio is {oscillator_damping:g}. The residual curves start when
        each command reaches its endpoint; RMS uses the next {oscillator_rms_window:g} s.

        **Constant frequency (2 Hz):** TVIST ZVD agrees with an independently
        evaluated fixed fractional-delay sum to a maximum command difference of
        **{constant_zvd_error:.2e}**. The two shaped traces overlap. This verifies
        the fractional-delay comparison.

        **Varying frequency (2→3 Hz):** TVIST ZVD reduces residual RMS relative to
        the unshaped move, but fixed-end ZVD does better in this example.
        """),
            _fig,
            mo.ui.table(
                [
                    {
                        **row,
                        "Residual position RMS": f"{row['Residual position RMS']:.6g}",
                    }
                    for row in oscillator_metrics
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def overview(mo):
    mo.md("""
    ## OpenArm simulation comparison

    Five trajectories, each with five methods (baseline, time-invariant ZVD, time-invariant EI, TVIST-ZV, TVIST-ZVD). Only simulation results are used below.

    - **Residual RMS:** whole-arm joint-space RMS about each trial's measured
      equilibrium (mean position over the last 0.5 s). Integrate over the first
      0.8 s after that trial's command ends.
    - **Total equilibrium settling:** seconds from command start to the first
      post-command sample after which every joint stays within ±0.01 rad of
      equilibrium through the end of the recording. “Not settled” means no
      sample met that condition in the recording.
    - **Acceleration violations:** number of recorded timestamps at which any
      joint's absolute commanded acceleration exceeds its saved limit (10 rad/s²
      here).
    """)
    return


@app.cell
def saved_comparison_data(Path, csv, np, yaml):
    data_dir = (
        Path(__file__).resolve().parents[1]
        / "sim_out"
        / "tvist_analysis"
        / "comparison"
    )
    with (data_dir / "summary.csv").open(newline="") as _stream:
        trial_summary = list(csv.DictReader(_stream))

    trace_rows_by_path = {}
    for _path in dict.fromkeys(row["results_csv"] for row in trial_summary):
        with (data_dir / _path).open(newline="") as _stream:
            trace_rows_by_path[_path] = list(csv.DictReader(_stream))

    limits_by_path = {}
    for _path in dict.fromkeys(row["metadata_yaml"] for row in trial_summary):
        with (data_dir / _path).open() as _stream:
            _metadata = yaml.safe_load(_stream)
        limits_by_path[_path] = np.asarray(
            _metadata["config"]["acceleration_limits"]
        )
    return data_dir, limits_by_path, trace_rows_by_path, trial_summary


@app.cell
def residual_metric_definitions(np):
    from me6404_openarm_shaping.data_types import FloatMatrix, FloatVector

    def calculate_residual_metrics(
        time: FloatVector,
        position: FloatMatrix,
        command_end: float,
    ) -> tuple[FloatVector, FloatMatrix, FloatVector | None, float | None]:
        """Return measured equilibrium, residual, joint RMS, and total settling time."""
        equilibrium = position[time >= max(command_end, time[-1] - 0.5)].mean(
            axis=0
        )
        residual = position - equilibrium
        rms_end = command_end + 0.8
        joint_rms = None
        if time[0] <= command_end and rms_end <= time[-1]:
            interior = (time > command_end) & (time < rms_end)
            window_time = np.r_[command_end, time[interior], rms_end]
            edges = np.array(
                [
                    np.interp([command_end, rms_end], time, joint)
                    for joint in residual.T
                ]
            ).T
            window_residual = np.vstack(
                (edges[0], residual[interior], edges[1])
            )
            joint_rms = np.sqrt(
                np.trapezoid(window_residual**2, x=window_time, axis=0) / 0.8
            )

        # Allow only for roundoff in the accumulated MuJoCo timestamps.
        post_command = time >= command_end - 1e-9
        within_band = (np.abs(residual[post_command]) <= 0.01).all(axis=1)
        stays_within_band = np.logical_and.accumulate(within_band[::-1])[::-1]
        settled_samples = time[post_command][stays_within_band]
        settled = (
            max(command_end, float(settled_samples[0]))
            if settled_samples.size
            else None
        )
        return equilibrium, residual, joint_rms, settled

    return (calculate_residual_metrics,)


@app.cell
def comparison_metrics(
    calculate_residual_metrics,
    limits_by_path,
    np,
    trace_rows_by_path,
    trial_summary,
):
    method_labels = {
        "baseline": "Unshaped",
        "time_invariant_zvd": "Fixed ZVD",
        "time_invariant_ei": "Fixed EI",
        "tvist_zv": "TVIST ZV",
        "tvist_zvd": "TVIST ZVD",
    }
    comparison = []
    for _run in trial_summary:
        _trial = _run["trial"]
        _command_end = float(_run["command_duration_s"])
        _rows = [
            row
            for row in trace_rows_by_path[_run["results_csv"]]
            if row["trajectory"] == _trial
        ]
        _limits = limits_by_path[_run["metadata_yaml"]]
        _t = np.array([float(row["time_s"]) for row in _rows])
        _q = np.array(
            [
                [float(row[f"J{joint}_q_rad"]) for joint in range(1, 8)]
                for row in _rows
            ]
        )
        _ddq = np.array(
            [
                [
                    float(row[f"J{joint}_ddq_reference_rad_s2"])
                    for joint in range(1, 8)
                ]
                for row in _rows
            ]
        )

        _, _, _joint_rms, _settled = calculate_residual_metrics(
            _t, _q, _command_end
        )
        _rms = (
            float(np.linalg.norm(_joint_rms))
            if _joint_rms is not None
            else None
        )

        _abs_ddq = np.abs(_ddq)
        comparison.append(
            {
                "Trajectory": _run["trajectory_id"],
                "Method": method_labels[_trial],
                "Command duration [s]": _command_end,
                "Residual RMS [rad]": _rms,
                "Total equilibrium settling [s]": _settled,
                "Peak commanded acceleration [rad/s²]": float(_abs_ddq.max()),
                "Samples over acceleration limit": int(
                    (_abs_ddq > _limits).any(axis=1).sum()
                ),
            }
        )
    return comparison, method_labels


@app.cell(hide_code=True)
def comparison_table(comparison, mo):
    _display_rows = [
        {
            **row,
            "Residual RMS [rad]": (
                f"{row['Residual RMS [rad]']:.6f}"
                if row["Residual RMS [rad]"] is not None
                else "Incomplete window"
            ),
            "Total equilibrium settling [s]": (
                f"{row['Total equilibrium settling [s]']:.3f}"
                if row["Total equilibrium settling [s]"] is not None
                else "Not settled"
            ),
            "Peak commanded acceleration [rad/s²]": round(
                row["Peak commanded acceleration [rad/s²]"], 3
            ),
        }
        for row in comparison
    ]
    mo.ui.table(
        _display_rows, selection=None, pagination=False, show_download=False
    )
    return


@app.cell(hide_code=True)
def comparison_plot(comparison, method_labels, mo, plt):
    _fig, _axes = plt.subplots(
        2, 3, figsize=(10, 5.5), sharex=True, sharey=True, layout="constrained"
    )
    _trajectories = list(dict.fromkeys(row["Trajectory"] for row in comparison))
    _colors = {
        method: f"C{i}" for i, method in enumerate(method_labels.values())
    }
    for _ax, _trajectory in zip(_axes.flat, _trajectories, strict=False):
        _ax.set_title(_trajectory)
        _ax.set_yscale("log")
        _ax.grid(alpha=0.25)
        for _row in comparison:
            if _row["Trajectory"] != _trajectory:
                continue
            _settling = _row["Total equilibrium settling [s]"]
            _rms = _row["Residual RMS [rad]"]
            if _settling is None or _rms is None or _rms <= 0:
                continue
            _color = _colors[_row["Method"]]
            _ax.scatter(
                _settling,
                _rms,
                edgecolors=_color,
                facecolors="none"
                if _row["Samples over acceleration limit"]
                else _color,
            )
    _key = _axes.flat[-1]
    _key.axis("off")
    for _method, _color in _colors.items():
        _key.plot(
            [], [], marker="o", linestyle="none", color=_color, label=_method
        )
    _key.plot(
        [],
        [],
        marker="o",
        linestyle="none",
        color="0.35",
        markerfacecolor="none",
        label="Acceleration limit violation",
    )
    _key.legend(loc="center", frameon=False)
    _fig.supxlabel("Total equilibrium settling time [s]")
    _fig.supylabel("Post-command residual RMS [rad]")
    mo.vstack(
        [
            mo.md("""### Residual vibration vs. settling time
        Open markers exceed the commanded acceleration limit. Runs without a
        settling time or positive RMS remain in the table but are not plotted.
        """),
            _fig,
        ]
    )
    return


@app.cell
def wide07_acceleration_data(
    limits_by_path,
    np,
    trace_rows_by_path,
    trial_summary,
):
    wide07_trials = ("baseline", "tvist_zvd")
    _wide07_runs = [
        row
        for row in trial_summary
        if row["trajectory_id"] == "wide-07" and row["trial"] in wide07_trials
    ]
    wide07_command_end = {
        row["trial"]: float(row["command_duration_s"]) for row in _wide07_runs
    }
    wide07_limits = limits_by_path[_wide07_runs[0]["metadata_yaml"]]
    _wide07_rows = trace_rows_by_path[_wide07_runs[0]["results_csv"]]
    wide07_acceleration = {}
    for _trial in wide07_trials:
        _rows = [row for row in _wide07_rows if row["trajectory"] == _trial]
        _time = np.array([float(row["time_s"]) for row in _rows])
        _ddq = np.array(
            [
                [
                    float(row[f"J{joint}_ddq_reference_rad_s2"])
                    for joint in range(1, 8)
                ]
                for row in _rows
            ]
        )
        wide07_acceleration[_trial] = (_time, _ddq)
    return (
        wide07_acceleration,
        wide07_command_end,
        wide07_limits,
        wide07_trials,
    )


@app.cell
def wide07_modal_schedule(csv, data_dir, np):
    _schedule_path = data_dir.parent / "wide-07" / "tvist_modes.csv"
    _samples = {}
    with _schedule_path.open(newline="") as _stream:
        for _row in csv.DictReader(_stream):
            if _row["trajectory"] != "tvist_zvd":
                continue
            _modes = _samples.setdefault(float(_row["time_s"]), [])
            if _row["frequency_rank"]:
                _modes.append(
                    (
                        int(_row["frequency_rank"]),
                        float(_row["natural_frequency_hz"]),
                        float(_row["damping_ratio"]),
                    )
                )

    wide07_schedule_time = np.array(sorted(_samples))
    wide07_mode_count = np.array(
        [len(_samples[t]) for t in wide07_schedule_time]
    )
    wide07_frequency_by_rank = np.full(
        (wide07_schedule_time.size, int(wide07_mode_count.max())), np.nan
    )
    wide07_damping_by_rank = np.full_like(wide07_frequency_by_rank, np.nan)
    wide07_max_delay = np.zeros(wide07_schedule_time.size)
    for _i, _time in enumerate(wide07_schedule_time):
        for _rank, _f, _zeta in _samples[_time]:
            wide07_frequency_by_rank[_i, _rank - 1] = _f
            wide07_damping_by_rank[_i, _rank - 1] = _zeta
            # Latest impulse of a convolution is the sum of the modal latest impulses.
            wide07_max_delay[_i] += 1 / (_f * np.sqrt(1 - _zeta**2))
    return (
        wide07_damping_by_rank,
        wide07_frequency_by_rank,
        wide07_max_delay,
        wide07_mode_count,
        wide07_schedule_time,
    )


@app.cell
def wide07_acceleration_and_schedule(
    method_labels,
    mo,
    np,
    plt,
    wide07_acceleration,
    wide07_command_end,
    wide07_damping_by_rank,
    wide07_frequency_by_rank,
    wide07_limits,
    wide07_max_delay,
    wide07_mode_count,
    wide07_schedule_time,
    wide07_trials,
):
    _fig, _axes = plt.subplots(
        4, 2, figsize=(12, 9), sharex=True, layout="constrained"
    )
    (
        _accel_ax,
        _frequency_ax,
        _damping_ax,
        _rate_ax,
        _tau_ax,
        _tau_rate_ax,
        _count_ax,
        _delay_ax,
    ) = _axes.flat
    _end = max(wide07_command_end.values())
    _tvist_t, _tvist_ddq = wide07_acceleration["tvist_zvd"]
    _index = 1  # J2 in zero-based joint order
    _limit = wide07_limits[_index]
    for _trial, _color in zip(wide07_trials, ("C0", "C4"), strict=True):
        _time, _ddq = wide07_acceleration[_trial]
        # Include the command end even when the recorded time differs by roundoff.
        _window = _time <= _end + 1e-9
        _accel_ax.plot(
            _time[_window],
            _ddq[_window, _index],
            color=_color,
            label=method_labels[_trial],
        )
    _accel_ax.axhline(
        _limit, color="0.35", linestyle="--", label="Acceleration limits"
    )
    _accel_ax.axhline(-_limit, color="0.35", linestyle="--")
    _over = (np.abs(_tvist_ddq[:, _index]) > _limit) & (_tvist_t <= _end + 1e-9)
    if _over.any():
        _accel_ax.scatter(
            _tvist_t[_over],
            _tvist_ddq[_over, _index],
            s=8,
            color="C3",
            label="TVIST ZVD exceedance",
            zorder=3,
        )
    _accel_ax.set_ylabel("J2 acceleration\n[rad/s²]")
    _accel_ax.grid(alpha=0.25)
    _accel_ax.legend(loc="lower right", frameon=False)
    _fig.supxlabel("Time from command start [s]")
    # wide-07 only: frequency ranks 3 and 4 reverse over this sampled interval.
    _plot_frequency = wide07_frequency_by_rank.copy()
    _plot_damping = wide07_damping_by_rank.copy()
    _swap = (wide07_schedule_time >= 0.724) & (wide07_schedule_time <= 0.912)
    for _values in (_plot_frequency, _plot_damping):
        _values[_swap, 2:4] = _values[_swap, 2:4][:, ::-1]

    for _rank, _frequencies in enumerate(_plot_frequency.T, start=1):
        _frequency_ax.scatter(
            wide07_schedule_time,
            _frequencies,
            s=3,
            color=f"C{_rank - 1}",
            label=f"Rank {_rank}" if _rank < 3 else f"Branch {_rank}",
        )
    _frequency_ax.set_ylabel("Natural frequency [Hz]")
    _frequency_ax.legend(loc="upper right", ncols=4, frameon=False)
    for _rank, _damping in enumerate(_plot_damping.T, start=1):
        _damping_ax.scatter(
            wide07_schedule_time, _damping, s=3, color=f"C{_rank - 1}"
        )
    _damping_ax.set_ylabel("Damping ratio ζ")

    _rate_window = wide07_schedule_time <= 0.8
    _rate_t = wide07_schedule_time[_rate_window]
    _rate_mode_count = wide07_mode_count[_rate_window]
    _rate_frequency = _plot_frequency[_rate_window]
    _rate_damping = _plot_damping[_rate_window]
    _omega_d_for_rate = (
        2 * np.pi * _rate_frequency * np.sqrt(1 - _rate_damping**2)
    )

    _schedule_dt = np.diff(_rate_t)
    _midpoints = _rate_t[:-1] + _schedule_dt / 2
    _omega_d_rate = np.diff(_omega_d_for_rate, axis=0) / _schedule_dt[:, None]
    # Ranks may shift when a mode enters or leaves the selected set.
    _omega_d_rate[np.diff(_rate_mode_count) != 0] = np.nan
    for _rank, _rate in enumerate(_omega_d_rate.T, start=1):
        _rate_ax.plot(_midpoints, _rate, color=f"C{_rank - 1}")
    _rate_ax.axhline(0, color="0.5", linewidth=0.7)
    _rate_ax.set_ylabel(r"$d\omega_d/dt$" + "\n[rad/s²]")
    # The amplitude envelope of a frozen mode decays as exp(-ζ ω_n t).
    _tau = 1 / (2 * np.pi * _plot_frequency * _plot_damping)
    _tau_dt = np.diff(wide07_schedule_time)
    _tau_midpoints = wide07_schedule_time[:-1] + _tau_dt / 2
    _tau_rate = np.diff(_tau, axis=0) / _tau_dt[:, None]
    _tau_rate[np.diff(wide07_mode_count) != 0] = np.nan
    for _rank, (_decay, _decay_rate) in enumerate(
        zip(_tau.T, _tau_rate.T, strict=True)
    ):
        _tau_ax.plot(wide07_schedule_time, _decay, color=f"C{_rank}")
        _tau_rate_ax.plot(_tau_midpoints, _decay_rate, color=f"C{_rank}")
    _tau_ax.set_ylabel(r"Decay constant $\tau$ [s]")
    _tau_rate_ax.set_ylabel(r"$d\tau/dt$ [s/s]")
    _tau_rate_ax.axhline(0, color="0.5", linewidth=0.7)
    _count_ax.step(
        wide07_schedule_time, wide07_mode_count, where="post", color="C4"
    )
    _count_ax.set_ylabel("Selected modes")
    _count_ax.set_yticks(np.unique(wide07_mode_count))
    _delay_ax.plot(wide07_schedule_time, wide07_max_delay, color="C4")
    _delay_ax.set_ylabel("Maximum delay [s]")
    for _ax in _axes.flat[1:]:
        _ax.grid(alpha=0.25)

    # Mark the largest-magnitude sample in each contiguous exceedance interval.
    _j2_ddq = _tvist_ddq[:, _index]
    _over_samples = np.flatnonzero(_over)
    _intervals = np.split(
        _over_samples, np.flatnonzero(np.diff(_over_samples) > 1) + 1
    )
    _peak_samples = [
        interval[np.abs(_j2_ddq[interval]).argmax()]
        for interval in _intervals
        if interval.size
    ]
    for _sample in _peak_samples:
        for _ax in _axes.flat:
            _ax.axvline(
                _tvist_t[_sample], color="C3", linestyle=":", alpha=0.75
            )
    _peak_summary = ", ".join(
        f"**{_j2_ddq[sample]:+.2f} rad/s² at {_tvist_t[sample]:.3f} s**"
        for sample in _peak_samples
    )
    _mode_changes = wide07_schedule_time[1:][np.diff(wide07_mode_count) != 0]
    _change_times = ", ".join(f"{t:.3f}" for t in _mode_changes)
    mo.vstack(
        [
            mo.md(rf"""### `wide-07`: commanded acceleration and modal schedule
        The dashed lines are the saved J2 limits; red dots mark TVIST ZVD
        samples beyond them. Dotted vertical lines mark the peak magnitude in each
        exceedance interval across all panels: {_peak_summary}.

        The saved TVIST ZVD schedule changes mode count at **{_change_times} s**,
        not at either acceleration peak. The frequency, damping, and rate panels
        share a manual relabeling: saved ranks 3 and 4 are swapped from **0.724 to
        0.912 s**, the interval where rank 3 has the higher damping ratio. This
        keeps the higher-damping branch in the branch-4 color when both modes are
        present. It is a plot-only correction, not general mode tracking. The rate
        $d\omega_d/dt$ panel still stops at **0.8 s**. The decay constant
        $\tau=1/(2\pi f_n\zeta)$ and its rate use the same relabeling and cover the
        full schedule. Both derivative panels skip differences across mode-count
        changes.

        Maximum delay is the sum of the selected modes’ damped periods (the last
        impulse of each ZVD shaper). A nearby schedule change does not establish
        what caused an acceleration spike.
        """),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def wide07_ablation_controls(mo):
    include_branch_1 = mo.ui.checkbox(label="Branch 1", value=True)
    include_branch_2 = mo.ui.checkbox(label="Branch 2", value=False)
    include_branch_3 = mo.ui.checkbox(label="Branch 3", value=True)
    include_branch_4 = mo.ui.checkbox(label="Branch 4", value=True)
    mo.vstack(
        [
            mo.md("""### Exploratory simulation: `wide-07` mode selection
        Check the branches to **include** in TVIST ZVD.
        """),
            mo.hstack(
                [
                    include_branch_1,
                    include_branch_2,
                    include_branch_3,
                    include_branch_4,
                ],
                justify="start",
                wrap=True,
                gap=1,
            ),
        ]
    )
    return (
        include_branch_1,
        include_branch_2,
        include_branch_3,
        include_branch_4,
    )


@app.cell
def wide07_ablation_original(
    data_dir,
    np,
    trial_summary,
    wide07_acceleration,
    yaml,
):
    from contextlib import redirect_stdout as redirect_sim_stdout
    from io import StringIO as SimulationOutput

    from me6404_openarm_shaping import robot_sim
    from me6404_openarm_shaping.configuration import parse_robot_config

    wide07_ablation_trial = "tvist_zvd"
    _run = next(
        row
        for row in trial_summary
        if row["trajectory_id"] == "wide-07"
        and row["trial"] == wide07_ablation_trial
    )
    _config_data = yaml.safe_load(
        (data_dir / _run["metadata_yaml"]).read_text()
    )["config"]
    for _group, _settings in _config_data["input_shaping"].items():
        for _kind, _entry in _settings.items():
            _entry["enabled"] = (_group, _kind) == ("time_varying", "zvd")

    def parse_archived_tvist_config(raw):
        """Replay the pre-split batch: planning used 80% of the saved ceilings."""
        replay = raw.copy()
        for name in ("velocity_limits", "acceleration_limits"):
            if f"trajectory_{name}" not in replay:
                replay[f"trajectory_{name}"] = [
                    0.8 * limit for limit in replay[name]
                ]
                replay[f"absolute_{name}"] = replay[name]
        return parse_robot_config(replay)

    wide07_ablation_config = parse_archived_tvist_config(_config_data)
    with redirect_sim_stdout(SimulationOutput()):
        _original = robot_sim.simulate(
            robot_sim.load_model(), wide07_ablation_config, headless=True
        )

    _saved_t, _saved_accel = wide07_acceleration[wide07_ablation_trial]
    _recording = _original.recordings[wide07_ablation_trial]
    assert _recording["completed"]
    np.testing.assert_allclose(_recording["t"], _saved_t)
    np.testing.assert_allclose(
        _recording["ddq_reference"], _saved_accel, rtol=0, atol=1e-5
    )
    wide07_ablation_original = (
        _original.trajectories[wide07_ablation_trial],
        _recording,
    )
    return (
        SimulationOutput,
        parse_robot_config,
        redirect_sim_stdout,
        robot_sim,
        wide07_ablation_config,
        wide07_ablation_original,
        wide07_ablation_trial,
    )


@app.cell
def wide07_ablation_experiment(
    SimulationOutput,
    include_branch_1,
    include_branch_2,
    include_branch_3,
    include_branch_4,
    redirect_sim_stdout,
    robot_sim,
    wide07_ablation_config,
    wide07_ablation_original,
    wide07_ablation_trial,
):
    from unittest.mock import patch as _patch

    wide07_included_branches = tuple(
        branch
        for branch, checked in enumerate(
            (
                include_branch_1.value,
                include_branch_2.value,
                include_branch_3.value,
                include_branch_4.value,
            ),
            start=1,
        )
        if checked
    )
    _select_modes = robot_sim.select_tvist_modes

    def _select_remaining(properties, tau_min_s):
        selected = _select_modes(properties, tau_min_s)
        for i, sample in enumerate(selected):
            count = len(sample["natural_frequency_hz"])
            time_s = i * wide07_ablation_config.simulation.dt_s
            # Map saved frequency ranks to the same branches shown in the plots.
            branches = (
                (1, 2, 4, 3)
                if count == 4 and 0.724 <= time_s <= 0.912
                else range(1, count + 1)
            )
            keep = [
                index
                for index, branch in enumerate(branches)
                if branch in wide07_included_branches
            ]
            selected[i] = {
                **sample,
                "natural_frequency_hz": sample["natural_frequency_hz"][keep],
                "damped_frequency_hz": sample["damped_frequency_hz"][keep],
                "damping_ratio": sample["damping_ratio"][keep],
                "eigenvalues": sample["eigenvalues"][keep],
            }
        return selected

    if len(wide07_included_branches) < 4:
        with (
            redirect_sim_stdout(SimulationOutput()),
            _patch.object(robot_sim, "select_tvist_modes", _select_remaining),
        ):
            _modified = robot_sim.simulate(
                robot_sim.load_model(), wide07_ablation_config, headless=True
            )
        _result = (
            _modified.trajectories[wide07_ablation_trial],
            _modified.recordings[wide07_ablation_trial],
        )
        _included = ", ".join(map(str, wide07_included_branches)) or "none"
        _modified_label = f"Included: {_included}"
    else:
        _result = wide07_ablation_original
        _modified_label = "All branches included"

    assert _result[1]["completed"]
    wide07_ablation_comparison = {
        "Original TVIST ZVD": wide07_ablation_original,
        _modified_label: _result,
    }
    return (wide07_ablation_comparison,)


@app.cell(hide_code=True)
def wide07_ablation_plot(
    mo,
    np,
    plt,
    wide07_ablation_comparison,
    wide07_limits,
):
    _fig, (_accel_ax, _delta_ax) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True, layout="constrained"
    )
    _end = max(
        trajectory["t"][-1]
        for trajectory, _ in wide07_ablation_comparison.values()
    )
    _metrics = []
    for (_label, (_trajectory, _recording)), _style in zip(
        wide07_ablation_comparison.items(), ("-", "--"), strict=True
    ):
        _time = _recording["t"]
        _ddq = _recording["ddq_reference"]
        _window = _time <= _end + 1e-9
        _accel_ax.plot(_time[_window], _ddq[_window, 1], _style, label=_label)
        _over = np.abs(_ddq) > wide07_limits
        _peak_joint = (
            int(np.unravel_index(np.abs(_ddq).argmax(), _ddq.shape)[1]) + 1
        )
        _metrics.append(
            {
                "Method": _label,
                "Command duration [s]": round(float(_trajectory["t"][-1]), 3),
                "Peak |J2 acceleration| [rad/s²]": round(
                    float(np.abs(_ddq[:, 1]).max()), 3
                ),
                "J2 samples over limit": int(_over[:, 1].sum()),
                "Peak across all joints [rad/s²]": round(
                    float(np.abs(_ddq).max()), 3
                ),
                "Peak joint": f"J{_peak_joint}",
                "Samples with any joint over limit": int(
                    _over.any(axis=1).sum()
                ),
            }
        )

    _old, _new = [
        recording for _, recording in wide07_ablation_comparison.values()
    ]
    np.testing.assert_allclose(_old["t"], _new["t"], rtol=0, atol=1e-9)
    _window = _old["t"] <= _end + 1e-9
    _delta = _new["ddq_reference"][:, 1] - _old["ddq_reference"][:, 1]
    _delta_ax.plot(_old["t"][_window], _delta[_window], color="C2")
    _delta_ax.axhline(0, color="0.5", linewidth=0.8)
    for _limit in (-wide07_limits[1], wide07_limits[1]):
        _accel_ax.axhline(_limit, color="0.35", linestyle=":")
    _accel_ax.set_ylabel("J2 commanded acceleration\n[rad/s²]")
    _delta_ax.set_ylabel("Change in J2 acceleration\n[rad/s²]")
    _delta_ax.set_xlabel("Time from command start [s]")
    _accel_ax.legend(frameon=False)
    for _ax in (_accel_ax, _delta_ax):
        _ax.grid(alpha=0.25)
    mo.vstack(
        [
            mo.md("""The lower panel is **modified minus original** on the same clock.
        Branch labels use the manual rank 3/4 swap from 0.724 to 0.912 s.
        """),
            _fig,
            mo.ui.table(_metrics, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def wide07_saved_response_references(
    method_labels,
    np,
    trace_rows_by_path,
    trial_summary,
):
    wide07_saved_responses = {}
    for _run in trial_summary:
        if _run["trajectory_id"] != "wide-07":
            continue
        _trial = _run["trial"]
        _rows = [
            row
            for row in trace_rows_by_path[_run["results_csv"]]
            if row["trajectory"] == _trial
        ]
        _time = np.array([float(row["time_s"]) for row in _rows])
        _position = np.array(
            [
                [float(row[f"J{joint}_q_rad"]) for joint in range(1, 8)]
                for row in _rows
            ]
        )
        _goal = np.array(
            [
                float(_rows[-1][f"J{joint}_q_command_rad"])
                for joint in range(1, 8)
            ]
        )
        _label = (
            f"Original {method_labels[_trial]}"
            if _trial.startswith("tvist_")
            else method_labels[_trial]
        )
        wide07_saved_responses[_label] = (
            _time,
            _position,
            float(_run["command_duration_s"]),
            _goal,
        )
    return (wide07_saved_responses,)


@app.cell
def wide07_ablation_response_metrics(
    calculate_residual_metrics,
    np,
    wide07_ablation_comparison,
    wide07_saved_responses,
):
    wide07_response_metrics = []
    wide07_joint_rms = []
    wide07_response_traces = {}
    _responses = dict(wide07_saved_responses)
    for _label, (_command, _recording) in wide07_ablation_comparison.items():
        if _label in ("Original TVIST ZVD", "All branches included"):
            continue  # Already represented by the saved original response.
        _responses[_label] = (
            _recording["t"],
            _recording["q"],
            float(_command["t"][-1]),
            _command["q"][-1],
        )
    for _label, (_time, _position, _end, _goal) in _responses.items():
        _equilibrium, _residual, _joint_rms, _settled = (
            calculate_residual_metrics(_time, _position, _end)
        )
        _rms = (
            float(np.linalg.norm(_joint_rms))
            if _joint_rms is not None
            else None
        )
        wide07_response_metrics.append(
            {
                "Method": _label,
                "Command duration [s]": _end,
                "Residual RMS [rad]": _rms,
                "Total equilibrium settling [s]": _settled,
                "Post-command settling [s]": _settled - _end
                if _settled is not None
                else None,
                "Max equilibrium-to-goal error [rad]": float(
                    np.abs(_equilibrium - _goal).max()
                ),
            }
        )
        wide07_joint_rms.append(
            {
                "Method": _label,
                **{
                    f"J{joint} [rad]": float(_joint_rms[joint - 1])
                    if _joint_rms is not None
                    else None
                    for joint in range(1, 8)
                },
            }
        )
        wide07_response_traces[_label] = (
            _time,
            _position,
            _end,
            _residual,
            _settled,
        )
    return wide07_joint_rms, wide07_response_metrics, wide07_response_traces


@app.cell(hide_code=True)
def wide07_ablation_response_plots(
    mo,
    np,
    plt,
    wide07_joint_rms,
    wide07_response_metrics,
    wide07_response_traces,
):
    _position_fig, _position_axes = plt.subplots(
        4, 2, figsize=(9, 7), sharex=True, layout="constrained"
    )
    _residual_fig, _residual_axes = plt.subplots(
        4, 2, figsize=(9, 7), sharex=True, layout="constrained"
    )
    for _joint, (_pos_ax, _res_ax) in enumerate(
        zip(_position_axes.flat, _residual_axes.flat, strict=True)
    ):
        if _joint == 7:
            _pos_ax.axis("off")
            _res_ax.axis("off")
            break
        _res_ax.axvspan(0, 0.8, color="0.90", zorder=0)
        for _limit in (-0.01, 0.01):
            _res_ax.axhline(_limit, color="0.5", linestyle=":", linewidth=0.8)
        for _index, (
            _label,
            (_time, _position, _end, _residual, _settled),
        ) in enumerate(wide07_response_traces.items()):
            _selected = _label.startswith("Included:")
            _color = f"C{_index}"
            _style = "--" if _selected else "-"
            _width = 2 if _selected else 1.2
            _pos_ax.plot(
                _time,
                _position[:, _joint],
                _style,
                color=_color,
                linewidth=_width,
                label=_label,
            )
            _pos_ax.axvline(_end, color=_color, linestyle=":", alpha=0.45)
            _post = _time >= _end - 1e-9
            _res_ax.plot(
                np.maximum(0, _time[_post] - _end),
                _residual[_post, _joint],
                _style,
                color=_color,
                linewidth=_width,
                label=_label,
            )
        for _ax in (_pos_ax, _res_ax):
            _ax.set_title(f"J{_joint + 1}", loc="left", fontsize="small")
            _ax.grid(alpha=0.2)
    for _axes in (_position_axes, _residual_axes):
        _handles, _labels = _axes.flat[0].get_legend_handles_labels()
        _axes.flat[-1].legend(_handles, _labels, loc="center", frameon=False)
    _position_fig.supxlabel("Time from command start [s]")
    _position_fig.supylabel("Measured joint position [rad]")
    _residual_fig.supxlabel("Time since each command ended [s]")
    _residual_fig.supylabel("Position relative to measured equilibrium [rad]")

    _summary_display = [
        {
            key: (
                "Incomplete window"
                if key == "Residual RMS [rad]"
                else "Not settled"
            )
            if value is None
            else round(value, 6)
            if isinstance(value, float)
            else value
            for key, value in row.items()
        }
        for row in wide07_response_metrics
    ]
    mo.vstack(
        [
            mo.md("""### `wide-07`: joint responses and settling by mode selection
        The dashed response is the current checkbox selection. The other curves
        are saved simulation responses; selecting all branches coincides with
        original TVIST ZVD. Residual RMS uses the first **0.8 s after each command ends** (shaded), relative to each
        run’s measured equilibrium. Settling is the first post-command sample
        after which all seven joints stay within **±0.01 rad** of equilibrium
        through the end of the recording.  
        Equilibrium-to-goal error is reported
        separately: a small residual is not the same as reaching the target.
        """),
            mo.ui.table(_summary_display, selection=None, pagination=False),
            mo.ui.tabs(
                {
                    "Post-command residuals": _residual_fig,
                    "Measured joint positions": mo.vstack(
                        [
                            mo.md(
                                "Dotted vertical lines mark the end of each command."
                            ),
                            _position_fig,
                        ]
                    ),
                    "Per-joint residual RMS": mo.ui.table(
                        wide07_joint_rms, selection=None, pagination=False
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def wide07_screening_metrics(
    np,
    trial_summary,
    wide07_ablation_config,
    wide07_damping_by_rank,
    wide07_frequency_by_rank,
    wide07_mode_count,
    wide07_schedule_time,
    zvd_shaper,
):
    _reference_row = next(
        row
        for row in trial_summary
        if row["trajectory_id"] == "wide-07" and row["trial"] == "baseline"
    )
    _dt = wide07_ablation_config.simulation.dt_s
    _sample_count = round(float(_reference_row["command_duration_s"]) / _dt) + 1
    wide07_screening_time = wide07_schedule_time[:_sample_count]
    wide07_screening_frequency = wide07_frequency_by_rank[:_sample_count].copy()
    wide07_screening_damping = wide07_damping_by_rank[:_sample_count].copy()
    _swap = (wide07_screening_time >= 0.724) & (wide07_screening_time <= 0.912)
    for _values in (wide07_screening_frequency, wide07_screening_damping):
        _values[_swap, 2:4] = _values[_swap, 2:4][:, ::-1]

    # Last dimension is the ZVD impulse index (0, 1, 2).
    _shape = (*wide07_screening_frequency.shape, 3)
    _weights = np.full(_shape, np.nan)
    _delays = np.full(_shape, np.nan)
    for _i, _j in np.argwhere(np.isfinite(wide07_screening_frequency)):
        _weights[_i, _j], _delays[_i, _j] = zvd_shaper(
            2 * np.pi * wide07_screening_frequency[_i, _j],
            wide07_screening_damping[_i, _j],
        )

    _invalid = ~np.isfinite(_weights[:-1, :, 0]) | ~np.isfinite(
        _weights[1:, :, 0]
    )
    _invalid[np.diff(wide07_mode_count[:_sample_count]) != 0] = True
    _invalid[np.diff(_swap), 2:4] = True
    _step = np.diff(wide07_screening_time)[:, None]
    _delay_rate = np.abs(np.diff(_delays, axis=0) / _step[:, :, None])
    _delay_rate[_invalid] = np.nan
    _a0 = _weights[:, :, 0]
    _a0_rate = np.abs(np.diff(_a0, axis=0) / _step)
    _a0_rate[_invalid] = np.nan
    wide07_weighted_delay_rate = np.sum(
        _weights[:-1, :, 1:] * _delay_rate[:, :, 1:], axis=2
    )
    wide07_amplitude_curvature = np.abs(
        (_a0[2:] - 2 * _a0[1:-1] + _a0[:-2]) / _dt**2
    )
    wide07_amplitude_curvature[_invalid[:-1] | _invalid[1:]] = np.nan

    wide07_screening_summary = []
    for _j in range(_shape[1]):
        _row = {"Branch": _j + 1}
        for _label, _values, _times in (
            ("|dD/dt| [s/s]", _delay_rate[:, :, 1], wide07_screening_time[:-1]),
            ("|dA₀/dt| [1/s]", _a0_rate, wide07_screening_time[:-1]),
            (
                "Weighted delay rate J_D [s/s]",
                wide07_weighted_delay_rate,
                wide07_screening_time[:-1],
            ),
            (
                "Amplitude curvature J_A [1/s²]",
                wide07_amplitude_curvature,
                wide07_screening_time[1:-1],
            ),
        ):
            _peak_index = int(np.nanargmax(_values[:, _j]))
            _row[f"Max {_label}"] = float(_values[_peak_index, _j])
            _row[f"Time of max {_label.split(' [')[0]} [s]"] = float(
                _times[_peak_index]
            )
        wide07_screening_summary.append(_row)
    return (
        wide07_amplitude_curvature,
        wide07_screening_damping,
        wide07_screening_frequency,
        wide07_screening_summary,
        wide07_screening_time,
        wide07_weighted_delay_rate,
    )


@app.cell
def wide07_screening_freeze_replay(
    np,
    trace_rows_by_path,
    trial_summary,
    wide07_ablation_config,
    wide07_ablation_original,
    wide07_limits,
    wide07_screening_damping,
    wide07_screening_frequency,
    wide07_screening_time,
):
    from me6404_openarm_shaping.tvist import (
        TvistShaper,
        apply_tvist_to_joint_trajectory,
    )

    _run = next(
        row
        for row in trial_summary
        if row["trajectory_id"] == "wide-07" and row["trial"] == "baseline"
    )
    _count = len(wide07_screening_time)
    _rows = [
        row
        for row in trace_rows_by_path[_run["results_csv"]]
        if row["trajectory"] == "baseline"
    ][:_count]
    np.testing.assert_allclose(
        [float(row["time_s"]) for row in _rows],
        wide07_screening_time,
        rtol=0,
        atol=1e-12,
    )
    _reference = np.array(
        [
            [float(row[f"J{joint}_q_command_rad"]) for joint in range(1, 8)]
            for row in _rows
        ]
    )
    _dt = wide07_ablation_config.simulation.dt_s
    wide07_freeze_summary = []
    for _label, _freeze_frequency, _freeze_damping in (
        ("Original: both varying", False, False),
        ("Branch 2 frequency fixed", True, False),
        ("Branch 2 damping fixed", False, True),
        ("Branch 2 frequency and damping fixed", True, True),
    ):
        _frequency = wide07_screening_frequency.copy()
        _damping = wide07_screening_damping.copy()
        _branch2_present = np.isfinite(_frequency[:, 1])
        if _freeze_frequency:
            _frequency[_branch2_present, 1] = _frequency[0, 1]
        if _freeze_damping:
            _damping[_branch2_present, 1] = _damping[0, 1]
        _properties = []
        for _f, _zeta in zip(_frequency, _damping, strict=True):
            _present = np.isfinite(_f)
            _f, _zeta = _f[_present], _zeta[_present]
            _fd = _f * np.sqrt(1 - _zeta**2)
            _properties.append(
                {
                    "natural_frequency_hz": _f,
                    "damped_frequency_hz": _fd,
                    "damping_ratio": _zeta,
                    "eigenvalues": 2 * np.pi * (-_f * _zeta + 1j * _fd),
                    "real_eigenvalues": np.empty(0),
                    "stable": True,
                }
            )
        _q = apply_tvist_to_joint_trajectory(
            _reference, _dt, _properties, TvistShaper.ZVD, lagrange_order=1
        )
        if not (_freeze_frequency or _freeze_damping):
            np.testing.assert_allclose(
                _q, wide07_ablation_original[0]["q"], rtol=0, atol=1e-12
            )
        _dq = np.gradient(_q, _dt, axis=0, edge_order=2)
        _ddq = np.gradient(_dq, _dt, axis=0, edge_order=2)
        _violations = np.abs(_ddq) > wide07_limits
        wide07_freeze_summary.append(
            {
                "Branch 2 parameters": _label,
                "Peak |J2 acceleration| [rad/s²]": float(
                    np.abs(_ddq[:, 1]).max()
                ),
                "J2 samples over limit": int(_violations[:, 1].sum()),
                "Peak across all joints [rad/s²]": float(np.abs(_ddq).max()),
                "Samples with any joint over limit": int(
                    _violations.any(axis=1).sum()
                ),
            }
        )
    return TvistShaper, apply_tvist_to_joint_trajectory, wide07_freeze_summary


@app.cell(hide_code=True)
def wide07_screening_plots(
    mo,
    np,
    plt,
    wide07_acceleration,
    wide07_amplitude_curvature,
    wide07_freeze_summary,
    wide07_limits,
    wide07_screening_summary,
    wide07_screening_time,
    wide07_weighted_delay_rate,
):
    _fig, (_delay_ax, _amplitude_ax) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True, layout="constrained"
    )
    for _branch, (_weighted, _curvature) in enumerate(
        zip(
            wide07_weighted_delay_rate.T,
            wide07_amplitude_curvature.T,
            strict=True,
        ),
        start=1,
    ):
        _delay_ax.plot(
            wide07_screening_time[:-1], _weighted, label=f"Branch {_branch}"
        )
        _amplitude_ax.plot(wide07_screening_time[1:-1], _curvature)
    _time, _acceleration = wide07_acceleration["tvist_zvd"]
    _violating = np.flatnonzero(np.abs(_acceleration[:, 1]) > wide07_limits[1])
    for _interval in np.split(
        _violating, np.flatnonzero(np.diff(_violating) > 1) + 1
    ):
        if not _interval.size:
            continue
        _peak_time = _time[
            _interval[np.abs(_acceleration[_interval, 1]).argmax()]
        ]
        for _ax in (_delay_ax, _amplitude_ax):
            _ax.axvline(_peak_time, linestyle=":", color="0.35")
    _delay_ax.set_ylabel(r"Weighted delay rate [s/s]")
    _amplitude_ax.set_ylabel(r"$|\ddot A_0|$ [1/s²]")
    _amplitude_ax.set_xlabel("Time from command start [s]")
    _delay_ax.legend(ncols=4, frameon=False)
    for _ax in (_delay_ax, _amplitude_ax):
        _ax.grid(alpha=0.25)
    _summary = [
        {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in wide07_screening_summary
    ]
    _freeze = [
        {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in wide07_freeze_summary
    ]
    mo.vstack(
        [
            mo.md(r"""### `wide-07`: screening modes by impulse-parameter variation
        For each branch, $D=\pi/\omega_d$ is the ZVD half-period delay, **not**
        the modal decay time constant. The histories below are the terms whose
        maxima define the two candidate scores:

        $$
        J_D=\max_t\bigl(A_1|\dot D|+A_2|2\dot D|\bigr),\qquad
        J_A=\max_t|\ddot A_0|.
        $$

        Delay and weight rates use forward differences at the left endpoint;
        weighted rates use the impulse amplitudes at that same sample. Amplitude
        curvature uses a centered three-point difference. Derivatives do not
        cross missing samples, mode-count changes, or the manual relabeling
        boundaries for branches 3 and 4. Dotted lines mark the two original J2
        acceleration peaks. These scores are diagnostics, not acceleration bounds
        or automatic mode-selection rules.
        """),
            _fig,
            mo.ui.table(_summary, selection=None, pagination=False),
            mo.md("""#### Branch 2 parameter-freeze replay
        Each fixed parameter is held at its initial value for Branch 2 only.
        All other branches retain their original schedules. This compares offline
        acceleration commands, not measured vibration or settling.
        """),
            mo.ui.table(_freeze, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def cross_trajectory_screening_mode_shapes(
    data_dir,
    np,
    parse_robot_config,
    trace_rows_by_path,
    trial_summary,
    yaml,
):
    from hashlib import sha256 as _sha256
    from itertools import permutations as _permutations

    from me6404_openarm_shaping import robot_model as _model_api

    def _match_mode_shapes(model, config, reference):
        """Match adjacent displacement shapes; retain the last shape through absences."""
        workspace = _model_api.ModalWorkspace(model)
        previous = None
        candidates = {}
        branches = []
        quality = []
        gaps = []
        counts = []
        for position in reference:
            matrix = _model_api.state_matrix(
                model,
                position,
                config.motor_kp,
                config.motor_kd,
                workspace=workspace,
            )
            poles, vectors = np.linalg.eig(matrix)
            eligible = np.flatnonzero((poles.imag > 1e-10) & (poles.real < 0))
            eligible = eligible[np.argsort(np.abs(poles[eligible]))]
            shapes = vectors[:7, eligible]
            shapes = shapes / np.linalg.norm(shapes, axis=0)
            count = len(eligible)
            if previous is None:
                previous = shapes.copy()
                assignment = np.arange(count)
                overlap = np.ones(count)
                gap = np.inf
            else:
                assert 0 < count <= previous.shape[1], (
                    "Unexpected number of oscillatory modes"
                )
                if count not in candidates:
                    candidates[count] = np.array(
                        list(_permutations(range(previous.shape[1]), count))
                    )
                possible = candidates[count]
                # Complex modal assurance criterion: invariant to vector phase and scale.
                mac = np.abs(previous.conj().T @ shapes) ** 2
                overlaps = mac[possible, np.arange(count)]
                scores = overlaps.sum(axis=1)
                best, second = np.argsort(scores)[-2:][::-1]
                assignment = possible[best]
                overlap = overlaps[best]
                gap = float(scores[best] - scores[second])
                previous[:, assignment] = shapes
            tracked = np.full(previous.shape[1], np.nan + 1j * np.nan)
            tracked[assignment] = poles[eligible]
            confidence = np.full(tracked.size, np.nan)
            confidence[assignment] = overlap
            branches.append(tracked)
            quality.append(confidence)
            gaps.append(gap)
            counts.append(count)
        return (
            np.array(branches),
            np.array(quality),
            np.array(gaps),
            np.array(counts),
        )

    screening_cross_inputs = {}
    _model_hash = _sha256(_model_api.MODEL_PATH.read_bytes()).hexdigest()
    for _run in trial_summary:
        _path = _run["trajectory_id"]
        if _run["trial"] != "baseline" or _path == "wide-07":
            continue  # The preceding section already covers wide-07.
        _metadata = yaml.safe_load(
            (data_dir / _run["metadata_yaml"]).read_text()
        )
        assert _metadata["model"]["sha256"] == _model_hash
        _config = parse_robot_config(_metadata["config"])
        _dt = _config.simulation.dt_s
        _count = round(float(_run["command_duration_s"]) / _dt) + 1
        _rows = trace_rows_by_path[_run["results_csv"]]
        _baseline = [row for row in _rows if row["trajectory"] == "baseline"][
            :_count
        ]
        _reference = np.array(
            [
                [float(row[f"J{joint}_q_command_rad"]) for joint in range(1, 8)]
                for row in _baseline
            ]
        )
        _model = _model_api.load_model(_config.passive_damping)
        _model.opt.timestep = _dt
        _poles, _mac, _gap, _counts = _match_mode_shapes(
            _model, _config, _reference
        )
        _saved = np.array(
            [
                [
                    float(row[f"J{joint}_ddq_reference_rad_s2"])
                    for joint in range(1, 8)
                ]
                for row in _rows
                if row["trajectory"] == "tvist_zvd"
            ]
        )
        screening_cross_inputs[_path] = (
            _reference,
            _dt,
            _poles,
            _mac,
            _gap,
            _counts,
            _config.acceleration_limits,
            _saved,
        )
    return (screening_cross_inputs,)


@app.cell
def build_tvist_zvd_from_poles(
    TvistShaper,
    apply_tvist_to_joint_trajectory,
    np,
):
    def build_tvist_zvd_command(reference, dt, poles):
        properties = []
        for sample in poles:
            active = sample[np.isfinite(sample)]
            # Match the original filter's accumulation order for retained modes.
            active = active[np.argsort(np.abs(active))]
            omega_n = np.abs(active)
            properties.append(
                {
                    "natural_frequency_hz": omega_n / (2 * np.pi),
                    "damped_frequency_hz": active.imag / (2 * np.pi),
                    "damping_ratio": -active.real / omega_n,
                    "eigenvalues": active,
                    "real_eigenvalues": np.empty(0),
                    "stable": True,
                }
            )
        q = apply_tvist_to_joint_trajectory(
            reference, dt, properties, TvistShaper.ZVD, lagrange_order=1
        )
        dq = np.gradient(q, dt, axis=0, edge_order=2)
        ddq = np.gradient(dq, dt, axis=0, edge_order=2)
        return {"t": np.arange(len(q)) * dt, "q": q, "dq": dq, "ddq": ddq}

    return (build_tvist_zvd_command,)


@app.cell
def cross_trajectory_screening_replays(
    build_tvist_zvd_command,
    np,
    screening_cross_inputs,
):
    def _score_branches(poles, dt, invalid):
        omega_n = np.abs(poles)
        damping = -poles.real / omega_n
        k = np.exp(-np.pi * damping / np.sqrt(1 - damping**2))
        a0 = 1 / (1 + k) ** 2
        delay = np.pi / poles.imag
        # For ZVD, A1 + 2*A2 = 2*K/(1+K). Evaluate weights at left endpoints.
        weighted = (
            2 * k[:-1] / (1 + k[:-1]) * np.abs(np.diff(delay, axis=0) / dt)
        )
        weighted[invalid] = np.nan
        curvature = np.abs(np.diff(a0, n=2, axis=0) / dt**2)
        curvature[invalid[:-1] | invalid[1:]] = np.nan
        return np.nanmax(weighted, axis=0), np.nanmax(curvature, axis=0)

    cross_screening_summary = []
    for _path, (
        _reference,
        _dt,
        _poles,
        _mac,
        _gap,
        _counts,
        _limits,
        _saved,
    ) in screening_cross_inputs.items():
        _present = np.isfinite(_poles)
        # Use the confidence cutoffs from the existing trajectory-selection experiment.
        _weak = (_mac < 0.80) | ((_gap < 0.02)[:, None] & _present)
        _invalid = ~(_present[:-1] & _present[1:]) | _weak[1:]
        _invalid[np.diff(_counts) != 0] = True
        _jd, _ja = _score_branches(_poles, _dt, _invalid)
        _original = build_tvist_zvd_command(_reference, _dt, _poles)["ddq"]
        np.testing.assert_allclose(
            _original, _saved[: len(_original)], rtol=0, atol=1e-5
        )
        _original_peak = float(np.abs(_original).max())
        _original_over = int((np.abs(_original) > _limits).any(axis=1).sum())
        for _j in range(_poles.shape[1]):
            _ablation = _poles.copy()
            _ablation[:, _j] = np.nan
            _modified = build_tvist_zvd_command(_reference, _dt, _ablation)[
                "ddq"
            ]
            _peak = float(np.abs(_modified).max())
            _over = int((np.abs(_modified) > _limits).any(axis=1).sum())
            cross_screening_summary.append(
                {
                    "Trajectory": _path,
                    "Branch removed": _j + 1,
                    "Weak links": int(_weak[:, _j].sum()),
                    "Min shape overlap (MAC)": float(np.nanmin(_mac[:, _j])),
                    "J_D [s/s]": float(_jd[_j]),
                    "J_A [1/s²]": float(_ja[_j]),
                    "Original peak [rad/s²]": _original_peak,
                    "Peak after removal [rad/s²]": _peak,
                    "Peak reduction [rad/s²]": _original_peak - _peak,
                    "Original violation samples": _original_over,
                    "Violation samples after removal": _over,
                }
            )
    return (cross_screening_summary,)


@app.cell(hide_code=True)
def cross_trajectory_screening_comparison(cross_screening_summary, mo, plt):
    _fig, _axes = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True, layout="constrained"
    )
    _paths = list(
        dict.fromkeys(row["Trajectory"] for row in cross_screening_summary)
    )
    for _ax, _key in zip(_axes, ("J_D [s/s]", "J_A [1/s²]"), strict=True):
        for _index, _path in enumerate(_paths):
            _rows = [
                row
                for row in cross_screening_summary
                if row["Trajectory"] == _path
            ]
            _color = f"C{_index}"
            _ax.scatter(
                [row[_key] for row in _rows],
                [row["Peak reduction [rad/s²]"] for row in _rows],
                edgecolors=_color,
                facecolors=[
                    "none" if row["Weak links"] else _color for row in _rows
                ],
                label=_path,
            )
            for _row in _rows:
                _ax.annotate(
                    str(_row["Branch removed"]),
                    (_row[_key], _row["Peak reduction [rad/s²]"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize="x-small",
                )
        _ax.axhline(0, color="0.4", linestyle=":")
        _ax.set_xlabel(_key)
        _ax.grid(alpha=0.25)
    _axes[0].set_ylabel("Whole-arm peak acceleration reduction [rad/s²]")
    _axes[1].legend(frameon=False, fontsize="small")
    mo.vstack(
        [
            mo.md("""### Do the screening scores generalize to the other four trajectories?
        Branch IDs start at the initial natural-frequency ranks. At each subsequent
        sample, branches are matched by displacement mode-shape overlap, not by
        frequency rank. A reappearing mode is compared with the last shape seen
        for each branch. Derivative stencils skip weak matches, missing samples,
        and mode-count changes. Open markers flag branches with at least one
        provisional match (MAC below 0.80 or assignment-score gap below 0.02).

        Each branch is removed for the entire offline replay, wherever present.
        A **positive** reduction means removal lowered the largest absolute
        commanded acceleration across all seven joints. This is a command-level
        comparison only: it does not measure vibration suppression after removal
        and does not set a screening threshold.

        In this set, the highest-$J_D$ branch is useful to flag for inspection,
        not safe to remove automatically: on `wide-02`, removing that branch
        introduces acceleration-limit violations where the original command had none.
        """),
            _fig,
            mo.ui.table(
                [
                    {
                        key: round(value, 4)
                        if isinstance(value, float)
                        else value
                        for key, value in row.items()
                    }
                    for row in cross_screening_summary
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def branch_sweep_controls(mo):
    branch_sweep_run = mo.ui.run_button(
        label="Run all 75 subset simulations", kind="success"
    )
    branch_sweep_path = mo.ui.dropdown(
        options=[
            "All trajectories",
            "j2-only",
            "j2-j4",
            "medium-09",
            "wide-02",
            "wide-07",
        ],
        value="wide-07",
        label="Show results for",
    )
    mo.vstack(
        [
            mo.md("""### TVIST ZVD branch-combination sweep
        Test every single branch, pair, triple, and the full four-branch shaper
        for each trajectory. The sweep is offline and headless. The table filter
        does not rerun simulations. Saved unshaped, fixed ZVD, and fixed EI trials
        are comparison references.
        """),
            mo.hstack(
                [branch_sweep_run, branch_sweep_path],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return branch_sweep_path, branch_sweep_run


@app.cell
def branch_sweep_simulations(
    branch_sweep_run,
    build_tvist_zvd_command,
    calculate_residual_metrics,
    data_dir,
    method_labels,
    mo,
    np,
    parse_robot_config,
    robot_sim,
    screening_cross_inputs,
    trace_rows_by_path,
    trial_summary,
    wide07_screening_damping,
    wide07_screening_frequency,
    yaml,
):
    from itertools import combinations as _combinations

    from me6404_openarm_shaping.simulation_helpers import (
        run_trial as _run_trial,
    )

    mo.stop(
        not branch_sweep_run.value,
        mo.md("Click **Run all 75 subset simulations** to populate the table."),
    )

    def _saved_channels(rows, suffix):
        return np.array(
            [
                [float(row[f"J{joint}_{suffix}"]) for joint in range(1, 8)]
                for row in rows
            ]
        )

    def _summarize_response(
        path,
        method,
        time,
        measured,
        acceleration,
        end,
        goal,
        limits,
        provisional=False,
    ):
        equilibrium, _, joint_rms, settled = calculate_residual_metrics(
            time, measured, end
        )
        return {
            "Trajectory": path,
            "Included branches / reference": method,
            "Provisional mode match": "Yes" if provisional else "",
            "Residual RMS [rad]": float(np.linalg.norm(joint_rms))
            if joint_rms is not None
            else None,
            "Command duration [s]": end,
            "Total equilibrium settling [s]": settled,
            "Post-command settling [s]": max(0.0, settled - end)
            if settled is not None
            else None,
            "Max equilibrium-to-goal error [rad]": float(
                np.abs(equilibrium - goal).max()
            ),
            "Peak commanded acceleration [rad/s²]": float(
                np.abs(acceleration).max()
            ),
            "Samples over acceleration limit": int(
                (np.abs(acceleration) > limits).any(axis=1).sum()
            ),
        }

    branch_sweep_results = []
    _branch_ids = (1, 2, 3, 4)
    _subsets = [
        subset
        for size in range(1, 5)
        for subset in _combinations(_branch_ids, size)
    ]
    _reference_methods = ("baseline", "time_invariant_zvd", "time_invariant_ei")
    for _path in dict.fromkeys(run["trajectory_id"] for run in trial_summary):
        _runs = {
            run["trial"]: run
            for run in trial_summary
            if run["trajectory_id"] == _path
        }
        _baseline = _runs["baseline"]
        _metadata = yaml.safe_load(
            (data_dir / _baseline["metadata_yaml"]).read_text()
        )
        _config = parse_robot_config(_metadata["config"])
        _rows = trace_rows_by_path[_baseline["results_csv"]]
        _dt = _config.simulation.dt_s
        _limits = _config.acceleration_limits
        if _path == "wide-07":
            # These are the manually relabeled frequency/damping pairs from the diagnostic.
            _wn = 2 * np.pi * wide07_screening_frequency
            _zeta = wide07_screening_damping
            _poles = _wn * (-_zeta + 1j * np.sqrt(1 - _zeta**2))
            _weak_branches = np.zeros(4, dtype=bool)
            _base_rows = [
                row for row in _rows if row["trajectory"] == "baseline"
            ]
            _reference = _saved_channels(
                _base_rows[: len(_poles)], "q_command_rad"
            )
        else:
            _reference, _, _poles, _mac, _gap, _, _, _ = screening_cross_inputs[
                _path
            ]
            _weak_branches = (
                ((_mac < 0.80) | (_gap[:, None] < 0.02)) & np.isfinite(_poles)
            ).any(axis=0)

        for _method in _reference_methods:
            _saved_rows = [row for row in _rows if row["trajectory"] == _method]
            _time = np.array([float(row["time_s"]) for row in _saved_rows])
            _end = float(_runs[_method]["command_duration_s"])
            _goal = np.array(
                [
                    float(_saved_rows[-1][f"J{joint}_q_command_rad"])
                    for joint in range(1, 8)
                ]
            )
            branch_sweep_results.append(
                _summarize_response(
                    _path,
                    method_labels[_method],
                    _time,
                    _saved_channels(_saved_rows, "q_rad"),
                    _saved_channels(_saved_rows, "ddq_reference_rad_s2"),
                    _end,
                    _goal,
                    _limits,
                )
            )

        _model = robot_sim.load_model(_config.passive_damping)
        _model.opt.timestep = _dt
        # Every subset ends no later than the full shaper. Use one observation horizon.
        _saved_zvd = [row for row in _rows if row["trajectory"] == "tvist_zvd"]
        _full_end = float(_runs["tvist_zvd"]["command_duration_s"])
        _horizon = max(
            _config.simulation.minimum_duration_s,
            _full_end + _config.simulation.settling_time_s,
        )
        for _subset in _subsets:
            _columns = np.array(_subset) - 1
            _command = build_tvist_zvd_command(
                _reference, _dt, _poles[:, _columns]
            )
            _end = float(_command["t"][-1])
            assert _end <= _full_end + 1e-9
            _recording = _run_trial(
                _model,
                _command,
                _reference[0],
                _config.motor_kp,
                _config.motor_kd,
                _horizon,
                config=_config,
                headless=True,
            )
            assert _recording["completed"]
            if _subset == _branch_ids:
                np.testing.assert_allclose(_end, _full_end, rtol=0, atol=1e-9)
                np.testing.assert_allclose(
                    _recording["t"],
                    [float(row["time_s"]) for row in _saved_zvd],
                    rtol=0,
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    _recording["ddq_reference"],
                    _saved_channels(_saved_zvd, "ddq_reference_rad_s2"),
                    rtol=0,
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    _recording["q"],
                    _saved_channels(_saved_zvd, "q_rad"),
                    rtol=0,
                    atol=1e-7,
                )
            branch_sweep_results.append(
                _summarize_response(
                    _path,
                    " + ".join(map(str, _subset)),
                    _recording["t"],
                    _recording["q"],
                    _recording["ddq_reference"],
                    _end,
                    _command["q"][-1],
                    _limits,
                    provisional=bool(_weak_branches[_columns].any()),
                )
            )
        print(f"{_path}: finished {len(_subsets)} headless trials")
    assert len(branch_sweep_results) == 90
    return (branch_sweep_results,)


@app.cell(hide_code=True)
def branch_sweep_table(branch_sweep_path, branch_sweep_results, mo):
    _selected = branch_sweep_path.value
    _rows = [
        {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in branch_sweep_results
        if _selected == "All trajectories" or row["Trajectory"] == _selected
    ]
    mo.vstack(
        [
            mo.md("""**How to read the table.** Residual RMS is measured over the first 0.8 s
        after each command ends, relative to the run’s equilibrium. Settling means
        staying within ±0.01 rad of equilibrium on all joints through the end of the
        recording. A missing settling time means the run did not settle within that
        recording. A provisional match uses at least one weakly matched branch;
        `wide-07` instead uses the case-specific manual rank swap. Acceleration-limit
        counts assess the command only, not overall simulation or hardware safety.
        """),
            mo.ui.table(_rows, selection=None, page_size=18),
        ]
    )
    return


@app.cell(hide_code=True)
def wide07_branch_sweep_tradeoff(branch_sweep_results, mo, plt):
    _fig, _ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    _ax.set_yscale("log")
    _ax.set_xlabel("Total equilibrium settling time [s]")
    _ax.set_ylabel("Post-command residual RMS [rad]")
    _ax.set_title("wide-07: TVIST ZVD branch combinations")
    _ax.grid(alpha=0.25, which="both")
    _ax.margins(x=0.16, y=0.24)

    # Label offsets separate subsets with almost identical settling times and RMS.
    _offsets = {
        "3": (-24, 11),
        "4": (10, -10),
        "3 + 4": (6, -12),
        "1": (-12, 10),
        "1 + 3": (-5, 13),
        "1 + 4": (-6, -16),
        "1 + 3 + 4": (-8, -18),
        "1 + 2": (10, 16),
        "1 + 2 + 3": (10, -7),
        "1 + 2 + 4": (10, 5),
        "1 + 2 + 3 + 4": (10, -16),
        "2": (12, 22),
        "2 + 4": (12, 8),
        "2 + 3": (12, -5),
        "2 + 3 + 4": (12, -12),
    }
    _references = {"Fixed ZVD": ("s", "C1"), "Fixed EI": ("D", "C2")}
    _method_key = "Included branches / reference"
    for _row in branch_sweep_results:
        _method = _row[_method_key]
        if _row["Trajectory"] != "wide-07" or _method == "Unshaped":
            continue
        _time = _row["Total equilibrium settling [s]"]
        _rms = _row["Residual RMS [rad]"]
        if _time is None or _rms is None:
            continue
        _feasible = _row["Samples over acceleration limit"] == 0
        _highlight = _method == "1 + 3 + 4"
        _marker, _color = _references.get(_method, ("o", "C0"))
        _ax.scatter(
            _time,
            _rms,
            marker=_marker,
            edgecolors=_color,
            facecolors=_color if _feasible else "none",
            s=90 if _highlight else 45,
            linewidths=1.4,
            zorder=3,
            label=_method if _method in _references else "_nolegend_",
        )
        if _method in _references:
            continue
        _dx, _dy = _offsets[_method]
        _ax.annotate(
            _method,
            (_time, _rms),
            xytext=(_dx, _dy),
            textcoords="offset points",
            ha="left" if _dx >= 0 else "right",
            va="center",
            fontsize=8,
            fontweight="bold" if _highlight else "normal",
            arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.7},
        )
    for _filled, _label in (
        (True, "TVIST ZVD: no acceleration-limit violations"),
        (False, "TVIST ZVD: acceleration-limit violation"),
    ):
        _ax.scatter(
            [],
            [],
            marker="o",
            edgecolors="C0",
            facecolors="C0" if _filled else "none",
            label=_label,
        )
    _ax.legend(loc="upper left", frameon=False, fontsize="small")
    mo.vstack(
        [
            _fig,
            mo.md("""**Branches 1 + 3 + 4** gave the lowest residual RMS of the TVIST
        subsets with no commanded acceleration-limit violations. They settled
        faster than either fixed reference, but left more residual vibration.
        Filled markers indicate commanded acceleration feasibility only, not
        overall simulation or hardware safety.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def experimental_tvist_controls(mo, trial_summary):
    experimental_model = mo.ui.dropdown(
        options=["Nominal", "Calibrated"],
        value="Nominal",
        label="Model dataset",
    )
    experimental_case = mo.ui.dropdown(
        options=list(
            dict.fromkeys(row["trajectory_id"] for row in trial_summary)
        ),
        value="wide-07",
        label="Trajectory for joint responses",
    )
    mo.vstack(
        [
            mo.md("""## Experimental TVIST

        Generate every nonempty MAC-tracked modal subset. Reject commands that
        exceed the absolute velocity or acceleration limits, then simulate the
        survivors. Apply the four criteria **separately to each enabled TVIST shaper**.
        """),
            mo.hstack(
                [
                    mo.md("""
        | Criterion | Which subset wins? |
        | :--- | :--- |
        | **1. Minimum residual RMS** | The one with the lowest residual RMS. |
        | **2. Minimum stabilization time** | The one that settles earliest, measured from command start. |
        | **3. Beat fixed RMS** | Among subsets with strictly lower RMS than the best fixed reference, choose the fastest to settle. |
        | **4. Beat fixed settling** | Among subsets that settle strictly sooner than the fastest fixed reference, choose the one with the lowest RMS. |
        """)
                ],
                justify="start",
            ),
            mo.md("""
        The RMS benchmark is the lower RMS of **fixed ZVD and fixed EI**. The time
        benchmark is their shorter total settling time. The two benchmarks can
        come from different shapers.

        **Nominal** uses the archived results; **Calibrated** uses the
        saved calibrated sweep. Switching models updates the Pareto and response
        plots. Both dropdowns load saved results without running simulations.
        """),
            mo.hstack(
                [experimental_model, experimental_case],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return experimental_case, experimental_model


@app.cell
def experimental_tvist_load_responses(
    calculate_residual_metrics,
    csv,
    experimental_case,
    experimental_model_label,
    experimental_nominal_zip,
    experimental_objective_labels,
    experimental_saved_metrics,
    experimental_sweep_dir,
    experimental_tradeoff_winners,
    method_labels,
    np,
    yaml,
):
    from io import TextIOWrapper as _TextIOWrapper
    from zipfile import ZipFile as _ZipFile

    _case = experimental_case.value
    _nominal = experimental_model_label == "Nominal"
    _metrics = [
        row for row in experimental_saved_metrics if row["trajectory"] == _case
    ]

    def _read_rows(path):
        if _nominal:
            with (
                _ZipFile(experimental_nominal_zip) as bundle,
                _TextIOWrapper(
                    bundle.open(path), encoding="utf-8", newline=""
                ) as stream,
            ):
                return list(csv.DictReader(stream))
        with (experimental_sweep_dir / path).open(newline="") as stream:
            return list(csv.DictReader(stream))

    experimental_selected_branches = [
        {key: value for key, value in row.items() if key != "Trajectory"}
        for row in experimental_tradeoff_winners
        if row["Trajectory"] == _case
    ]
    _items = []
    for _trial in ("baseline", "time_invariant_zvd", "time_invariant_ei"):
        _row = next(
            row
            for row in _metrics
            if row["source"] == "experimental_run" and row["trial"] == _trial
        )
        _items.append((method_labels[_trial], _row))
    for _row in experimental_selected_branches:
        if _row["Branches"] == "No qualifying subset":
            continue
        _kind = "tvist_" + _row["Shaper"].lower()
        _objective = next(
            key
            for key, label in experimental_objective_labels.items()
            if label == _row["Criterion"]
        )
        _suffix = _row["Branches"].replace(" ", "").replace("+", "_")
        _trial = (
            f"{_kind}_branches_{_suffix}"
            if _nominal
            else f"{_kind}_{_objective}"
        )
        _source = "experimental_candidate" if _nominal else "experimental_run"
        _saved = next(
            row
            for row in _metrics
            if row["source"] == _source and row["trial"] == _trial
        )
        _items.append(
            (f"Experimental {_row['Shaper']}: {_row['Criterion']}", _saved)
        )

    if _nominal:
        _refs = [
            row
            for row in _metrics
            if row["source"] == "saved_original"
            and row["trial"] in ("tvist_zv", "tvist_zvd")
        ]
    else:
        # Use only references saved alongside this calibrated sweep.
        _batch = experimental_sweep_dir / "reference_batch"
        with (_batch / "summary.csv").open(newline="") as _file:
            _refs = [
                dict(
                    row,
                    recording_csv=str(_batch / row["results_csv"]),
                    metadata_yaml=str(_batch / row["metadata_yaml"]),
                )
                for row in csv.DictReader(_file)
                if row["trajectory_id"] == _case
                and row["trial"] in ("tvist_zv", "tvist_zvd")
            ]
        assert len(_refs) == 2
    _items.extend(
        ("Original " + method_labels[row["trial"]], row) for row in _refs
    )

    experimental_response_data = {}
    experimental_response_metrics = []
    _trace_cache = {}
    for _label, _row in _items:
        _path = _row["recording_csv"]
        if _path not in _trace_cache:
            _trace_cache[_path] = _read_rows(_path)
        _samples = [
            s for s in _trace_cache[_path] if s["trajectory"] == _row["trial"]
        ]
        assert _samples, f"Missing trace for {_row['trial']}"
        _time = np.array([float(s["time_s"]) for s in _samples])
        _q = np.array(
            [[float(s[f"J{j}_q_rad"]) for j in range(1, 8)] for s in _samples]
        )
        _end = float(_row["command_duration_s"])
        if "residual_rms_rad" in _row:
            _rms = (
                float(_row["residual_rms_rad"])
                if _row["residual_rms_rad"]
                else None
            )
            _settled = (
                float(_row["total_settling_s"])
                if _row["total_settling_s"]
                else None
            )
            _peak = float(_row["peak_commanded_acceleration_rad_s2"])
            _violations = int(_row["acceleration_violation_samples"])
        else:
            _, _, _joint_rms, _settled = calculate_residual_metrics(
                _time, _q, _end
            )
            _rms = (
                float(np.linalg.norm(_joint_rms))
                if _joint_rms is not None
                else None
            )
            _config = yaml.safe_load(
                (experimental_sweep_dir / _row["metadata_yaml"]).read_text()
            )["config"]
            _ddq = np.abs(
                np.array(
                    [
                        [
                            float(s[f"J{j}_ddq_reference_rad_s2"])
                            for j in range(1, 8)
                        ]
                        for s in _samples
                    ]
                )
            )
            _peak = float(_ddq.max())
            _violations = int(
                (_ddq > _config["absolute_acceleration_limits"])
                .any(axis=1)
                .sum()
            )
        experimental_response_data[_label] = (_time, _q, _end)
        experimental_response_metrics.append(
            {
                "Method": _label,
                "Command duration [s]": _end,
                "Residual RMS [rad]": _rms,
                "Total equilibrium settling [s]": _settled,
                "Post-command settling [s]": max(0.0, _settled - _end)
                if _settled is not None
                else None,
                "Peak commanded acceleration [rad/s²]": _peak,
                "Samples over acceleration limit": _violations,
            }
        )
    return (
        experimental_response_data,
        experimental_response_metrics,
        experimental_selected_branches,
    )


@app.cell(hide_code=True)
def experimental_tvist_curve_controls(
    experimental_model_label,
    experimental_response_data,
    experimental_selected_branches,
    mo,
):
    experimental_curves = mo.ui.dictionary(
        {
            label: mo.ui.checkbox(value=True, label=label)
            for label in experimental_response_data
        }
    )
    mo.vstack(
        [
            mo.md(f"### Selected branches — {experimental_model_label} model"),
            mo.ui.table(
                experimental_selected_branches, selection=None, pagination=False
            )
            if experimental_selected_branches
            else mo.md("No TVIST shapers are enabled for this case."),
            mo.md(
                "### Measured joint responses\nToggle curves from the saved recordings."
            ),
            mo.hstack(
                list(experimental_curves.elements.values()),
                justify="start",
                wrap=True,
            ),
        ]
    )
    return (experimental_curves,)


@app.cell(hide_code=True)
def experimental_tvist_response_plot(
    experimental_curves,
    experimental_response_data,
    mo,
    plt,
):
    _fig, _axes = plt.subplots(
        4, 2, sharex=True, figsize=(11, 8), layout="constrained"
    )
    for _j, _ax in enumerate(_axes.flat[:7]):
        for _i, (_label, (_time, _q, _end)) in enumerate(
            experimental_response_data.items()
        ):
            if experimental_curves.value[_label]:
                _color = f"C{_i % 10}"
                _ax.plot(_time, _q[:, _j], color=_color, label=_label)
                _ax.axvline(_end, color=_color, linestyle=":", alpha=0.4)
        _ax.set_title(f"J{_j + 1}", loc="left")
        _ax.grid(alpha=0.25)
    _axes.flat[-1].axis("off")
    _handles, _labels = _axes.flat[0].get_legend_handles_labels()
    if _handles:
        _axes.flat[-1].legend(
            _handles, _labels, loc="center", fontsize="small", frameon=False
        )
    _fig.supxlabel("Time from command start [s]")
    _fig.supylabel("Simulated joint position [rad]")
    mo.vstack([mo.md("Dotted lines mark command end."), _fig])
    return


@app.cell(hide_code=True)
def experimental_tvist_final_table(experimental_response_metrics, mo):
    mo.vstack(
        [
            mo.md("""### Residual vibration and settling
        RMS covers the first 0.8 s after command end. Total settling is measured
        from command start; the settling band is ±0.01 rad about each measured
        joint equilibrium. Blank values indicate no measured score. 
        """),
            mo.ui.table(
                [
                    {
                        key: round(value, 6)
                        if isinstance(value, float)
                        else value
                        for key, value in row.items()
                    }
                    for row in experimental_response_metrics
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell
def experimental_tvist_tradeoff_data(csv, data_dir, experimental_model, mo):
    from io import TextIOWrapper
    from zipfile import ZipFile

    from me6404_openarm_shaping.tvist_selection import (
        SELECTION_OBJECTIVES,
        SubsetScore,
        select_subset_indices,
    )

    experimental_objective_labels = {
        "min_residual": "Minimum RMS",
        "min_settling": "Minimum stabilization time",
        "beat_fixed_rms": "Beat fixed RMS, then minimize time",
        "beat_fixed_settling": "Beat fixed time, then minimize RMS",
    }
    experimental_model_label = experimental_model.value
    experimental_nominal_zip = (
        data_dir.parent / "tvist-experimental-all-trajectories.zip"
    )
    experimental_calibrated_dir = data_dir.parent.parent / "tvist_calibrated"
    experimental_sweep_dir = experimental_calibrated_dir / "experimental_sweep"
    if experimental_model_label == "Nominal":
        mo.stop(
            not experimental_nominal_zip.is_file(),
            mo.md("The saved nominal sweep ZIP is missing."),
        )
        with ZipFile(experimental_nominal_zip) as _bundle:
            with TextIOWrapper(
                _bundle.open("subset_audit.csv"), encoding="utf-8", newline=""
            ) as _file:
                _audit_rows = list(csv.DictReader(_file))
            with TextIOWrapper(
                _bundle.open("final_characteristics.csv"),
                encoding="utf-8",
                newline="",
            ) as _file:
                experimental_saved_metrics = list(csv.DictReader(_file))
    else:
        mo.stop(
            not (experimental_sweep_dir / "README.md").is_file(),
            mo.md(
                "The calibrated sweep is not available yet. Select Nominal or switch models again after the batch finishes."
            ),
        )
        with (experimental_sweep_dir / "subset_audit.csv").open(
            newline=""
        ) as _file:
            _audit_rows = list(csv.DictReader(_file))
        with (experimental_sweep_dir / "final_characteristics.csv").open(
            newline=""
        ) as _file:
            experimental_saved_metrics = list(csv.DictReader(_file))
    _fixed_rows = [
        row
        for row in experimental_saved_metrics
        if row["source"] == "experimental_run"
        and row["trial"] in ("time_invariant_zvd", "time_invariant_ei")
    ]

    def _number(value):
        return float(value) if value else None

    def _benchmark(fixed, key):
        values = [float(row[key]) for row in fixed.values() if row[key]]
        return min(values) if len(values) == 2 else None

    experimental_tradeoff_data = {}
    experimental_tradeoff_winners = []
    for _case in dict.fromkeys(row["trajectory"] for row in _audit_rows):
        _fixed = {
            row["trial"]: row
            for row in _fixed_rows
            if row["trajectory"] == _case
        }

        _fixed_rms = _benchmark(_fixed, "residual_rms_rad")
        _fixed_time = _benchmark(_fixed, "total_settling_s")
        _methods = {}
        for _method in ("tvist_zv", "tvist_zvd"):
            _rows = [
                row
                for row in _audit_rows
                if row["trajectory"] == _case
                and row["shaper"] == _method
                and row["simulated"] == "True"
                and int(row["acceleration_violation_samples"]) == 0
            ]
            _scores = [
                SubsetScore(
                    tuple(map(int, row["branches"].split("+"))),
                    float(row["command_duration_s"]),
                    _number(row["residual_rms_rad"]),
                    _number(row["total_settling_s"]),
                )
                for row in _rows
            ]
            _selected = select_subset_indices(_scores, _fixed_rms, _fixed_time)
            _methods[_method] = (_scores, _selected)
            for _objective in SELECTION_OBJECTIVES:
                _index = _selected.get(_objective)
                _winner = _scores[_index] if _index is not None else None
                experimental_tradeoff_winners.append(
                    {
                        "Trajectory": _case,
                        "Shaper": _method.removeprefix("tvist_").upper(),
                        "Criterion": experimental_objective_labels[_objective],
                        "Branches": " + ".join(map(str, _winner.branches))
                        if _winner
                        else "No qualifying subset",
                        "Residual RMS [mrad]": round(1000 * _winner.rms, 4)
                        if _winner and _winner.rms is not None
                        else None,
                        "Total stabilization [s]": round(_winner.settling, 4)
                        if _winner and _winner.settling is not None
                        else None,
                        "Peak acceleration [rad/s²]": round(
                            float(
                                _rows[_index][
                                    "peak_commanded_acceleration_rad_s2"
                                ]
                            ),
                            4,
                        )
                        if _index is not None
                        else None,
                        "Provisional MAC match": _rows[_index][
                            "provisional_match"
                        ]
                        if _index is not None
                        else "",
                    }
                )
        experimental_tradeoff_data[_case] = (
            _fixed,
            _fixed_rms,
            _fixed_time,
            _methods,
        )
    return (
        experimental_model_label,
        experimental_nominal_zip,
        experimental_objective_labels,
        experimental_saved_metrics,
        experimental_sweep_dir,
        experimental_tradeoff_data,
        experimental_tradeoff_winners,
    )


@app.cell(hide_code=True)
def experimental_tvist_tradeoff_controls(mo):
    tradeoff_zv = mo.ui.checkbox(value=True, label="TVIST ZV")
    tradeoff_zvd = mo.ui.checkbox(value=True, label="TVIST ZVD")
    tradeoff_highlight = mo.ui.checkbox(
        value=True, label="Highlight selected subsets"
    )
    tradeoff_pareto = mo.ui.checkbox(
        value=True, label="Overall Pareto frontier"
    )
    mo.vstack(
        [
            mo.md("""### Experimental TVIST: vibration vs. stabilization across trajectories
        Each point is an acceleration-feasible subset from the selected saved dataset.
        The dashed lines show the best fixed RMS and best fixed stabilization time. Lower and farther left is better. The thin line connects the overall non-dominated points among feasible fixed ZVD, fixed EI, and both TVIST strategies. A frontier point offers a different tradeoff;
        it need not beat both fixed benchmarks.
        """),
            mo.hstack(
                [
                    tradeoff_zv,
                    tradeoff_zvd,
                    tradeoff_highlight,
                    tradeoff_pareto,
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return tradeoff_highlight, tradeoff_pareto, tradeoff_zv, tradeoff_zvd


@app.cell
def experimental_tvist_global_pareto(experimental_tradeoff_data):
    experimental_pareto_points = {}
    experimental_pareto_table = []
    for _case, (_fixed, _, _, _methods) in experimental_tradeoff_data.items():
        _points = []
        for _label, _row in _fixed.items():
            if (
                int(_row["acceleration_violation_samples"]) == 0
                and _row["total_settling_s"]
                and _row["residual_rms_rad"]
            ):
                _points.append(
                    (
                        float(_row["total_settling_s"]),
                        1000 * float(_row["residual_rms_rad"]),
                        "Fixed ZVD"
                        if _label == "time_invariant_zvd"
                        else "Fixed EI",
                        "—",
                    )
                )
        for _method, (_scores, _) in _methods.items():
            for _score in _scores:
                if _score.settling is not None and _score.rms is not None:
                    _points.append(
                        (
                            _score.settling,
                            1000 * _score.rms,
                            _method.replace("_", " ").upper(),
                            " + ".join(map(str, _score.branches)),
                        )
                    )
        _front = [
            point
            for point in sorted(_points)
            if not any(
                other[0] <= point[0]
                and other[1] <= point[1]
                and (other[0] < point[0] or other[1] < point[1])
                for other in _points
            )
        ]
        experimental_pareto_points[_case] = sorted(
            {(time, rms) for time, rms, _, _ in _front}
        )
        experimental_pareto_table.extend(
            {
                "Trajectory": _case,
                "Method": method,
                "Branches": branches,
                "Total stabilization [s]": round(time, 4),
                "Residual RMS [mrad]": round(rms, 4),
            }
            for time, rms, method, branches in _front
        )
    return experimental_pareto_points, experimental_pareto_table


@app.cell(hide_code=True)
def experimental_tvist_tradeoff_plot(
    experimental_model_label,
    experimental_pareto_points,
    experimental_tradeoff_data,
    plt,
    tradeoff_highlight,
    tradeoff_pareto,
    tradeoff_zv,
    tradeoff_zvd,
):
    from matplotlib.lines import Line2D as _Line2D

    _fig, _axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    _styles = {
        "tvist_zv": ("o", "tab:blue", tradeoff_zv.value),
        "tvist_zvd": ("^", "tab:orange", tradeoff_zvd.value),
    }
    for _ax, (_case, (_fixed, _rms, _time, _methods)) in zip(
        _axes.flat, experimental_tradeoff_data.items(), strict=False
    ):
        _ax.set_title(_case)
        if _rms is not None:
            _ax.axhline(
                1000 * _rms, color="0.45", linestyle="--", linewidth=0.8
            )
        if _time is not None:
            _ax.axvline(_time, color="0.45", linestyle="--", linewidth=0.8)
        for _kind, _marker in (
            ("time_invariant_zvd", "D"),
            ("time_invariant_ei", "s"),
        ):
            _row = _fixed.get(_kind)
            if (
                not _row
                or not _row["total_settling_s"]
                or not _row["residual_rms_rad"]
            ):
                continue
            _ax.scatter(
                float(_row["total_settling_s"]),
                1000 * float(_row["residual_rms_rad"]),
                marker=_marker,
                color="black",
                zorder=4,
            )
        for _method, (_scores, _winners) in _methods.items():
            _marker, _color, _show = _styles[_method]
            if not _show:
                continue
            _points = {
                i: (score.settling, 1000 * score.rms)
                for i, score in enumerate(_scores)
                if score.settling is not None and score.rms is not None
            }
            if _points:
                _ax.scatter(
                    *zip(*_points.values(), strict=True),
                    marker=_marker,
                    color=_color,
                    alpha=0.35,
                    s=24,
                )
            if tradeoff_highlight.value:
                for _index in set(_winners.values()) & _points.keys():
                    _ax.scatter(
                        *_points[_index],
                        marker=_marker,
                        color=_color,
                        edgecolors="black",
                        linewidths=0.6,
                        s=75,
                        zorder=5,
                    )
        _frontier = experimental_pareto_points[_case]
        if tradeoff_pareto.value and len(_frontier) > 1:
            _ax.plot(
                *zip(*_frontier, strict=True),
                color="0.25",
                linewidth=0.85,
                zorder=2,
            )
        _ax.grid(alpha=0.2)
    _key = _axes.flat[-1]
    _key.axis("off")
    _handles = [
        _Line2D(
            [],
            [],
            marker="D",
            color="black",
            linestyle="none",
            label="Fixed ZVD",
        ),
        _Line2D(
            [],
            [],
            marker="s",
            color="black",
            linestyle="none",
            label="Fixed EI",
        ),
        _Line2D(
            [], [], color="0.45", linestyle="--", label="Best fixed benchmarks"
        ),
    ]
    for _method, (_marker, _color, _show) in _styles.items():
        if _show:
            _handles.append(
                _Line2D(
                    [],
                    [],
                    marker=_marker,
                    color=_color,
                    linestyle="none",
                    label=_method.replace("_", " ").upper(),
                )
            )
    if tradeoff_pareto.value:
        _handles.append(
            _Line2D(
                [],
                [],
                color="0.25",
                linewidth=0.85,
                label="Overall Pareto frontier",
            )
        )
    _key.legend(handles=_handles, loc="center", frameon=False)
    _fig.suptitle(f"{experimental_model_label} model — saved subset sweep")
    _fig.supxlabel("Total stabilization time [s]")
    _fig.supylabel("Post-command residual RMS [mrad]")
    _fig
    return


@app.cell(hide_code=True)
def experimental_tvist_tradeoff_winners(
    experimental_pareto_table,
    experimental_tradeoff_winners,
    mo,
):
    mo.ui.tabs(
        {
            "Four selection criteria": mo.ui.table(
                experimental_tradeoff_winners, selection=None, page_size=8
            ),
            "Overall Pareto frontier": mo.ui.table(
                experimental_pareto_table, selection=None, page_size=12
            ),
        }
    )
    return


@app.cell(hide_code=True)
def tvist_findings_and_conclusion(mo):
    mo.md("""
    ## Findings

    1. Fixed ZVD and EI provide strong vibration suppression and often achieve favorable stabilization times, even as the robot’s modal properties change along the trajectory.

    2. On some trajectories, TVIST ZVD achieves lower residual RMS than fixed ZVD. TVIST ZV and reduced-mode TVIST ZVD can also stabilize sooner, at the cost of more residual vibration.

    3. All-mode TVIST sometimes produced excessive commanded acceleration and poor vibration suppression. Screening subsets against the acceleration limits recovered competitive commands. On
    `wide-07`, TVIST ZVD with branches **1 + 3 + 4** met the commanded acceleration
    limits and stabilized in **1.930 s**, faster than either fixed reference, with
    **3.61 mrad** residual RMS. The fixed references left less residual vibration.

    ### Shaper choice

    Choose the shaper based on the application’s vibration and completion-time
    requirements, not just on whether the dynamics vary.

    If very low residual vibration matters most, compare fixed ZVD and TVIST ZVD. If task completion
    time matters more and some vibration is acceptable, reduced-mode TVIST ZV or ZVD may be useful. If fixed EI already meets both requirements, TVIST’s added complexity may not be justified.
    """)
    return


if __name__ == "__main__":
    app.run()
