import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from me6404_openarm_shaping.configuration import load_robot_config
    from me6404_openarm_shaping.trajectories import waypoint_trajectory
    from me6404_openarm_shaping.robot_model import (
        load_model,
        modal_properties,
    )

    return (
        load_model,
        load_robot_config,
        mo,
        modal_properties,
        np,
        plt,
        waypoint_trajectory,
    )


@app.cell
def _(load_robot_config, waypoint_trajectory):
    config = load_robot_config()

    t, q, dq, ddq = waypoint_trajectory(
        config.waypoints,
        config.velocity_limits,
        config.acceleration_limits,
        dt=config.simulation.dt_s,
    )

    config.waypoints
    return config, q, t


@app.cell
def _(mo):
    joint = mo.ui.slider(
        start=1,
        stop=7,
        step=1,
        value=2,
        label="Joint",
    )

    joint
    return (joint,)


@app.cell
def _(joint, plt, q, t):
    _fig, _ax = plt.subplots()

    _ax.plot(t, q[:, joint.value - 1])

    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Joint Position [rad]")
    _ax.set_title(f"J{joint.value} Reference Trajectory")
    _ax.grid(True)

    _fig
    return


@app.cell
def _(config, load_model, modal_properties, np, q, t):
    model = load_model(config.passive_damping)
    model.opt.timestep = config.simulation.dt_s

    sample_stride = 10
    sample_indices = np.unique(
        np.r_[np.arange(0, len(t), sample_stride), len(t) - 1]
    )

    sample_times = t[sample_indices]

    modal_results = [
        modal_properties(
            model,
            q[i],
            config.motor_kp,
            config.motor_kd,
        )
        for i in sample_indices
    ]

    print(f"Analyzed {len(modal_results)} of {len(t)} trajectory samples")
    return modal_results, sample_times


@app.cell
def _(mo, modal_results):
    max_rank = max(
        len(result["natural_frequency_hz"]) for result in modal_results
    )

    frequency_rank = mo.ui.slider(
        start=1,
        stop=max_rank,
        step=1,
        value=1,
        label="Frequency rank",
    )

    frequency_rank
    return (frequency_rank,)


@app.cell
def _(frequency_rank, modal_results, np, plt, sample_times):
    rank_index = frequency_rank.value - 1

    selected_frequencies = np.array(
        [
            result["natural_frequency_hz"][rank_index]
            if len(result["natural_frequency_hz"]) > rank_index
            else np.nan
            for result in modal_results
        ]
    )

    _fig_modal, _ax_modal = plt.subplots(figsize=(9, 4))

    _ax_modal.scatter(
        sample_times,
        selected_frequencies,
        s=8,
    )

    _ax_modal.set_xlabel("Trajectory time [s]")
    _ax_modal.set_ylabel("Natural frequency [Hz]")
    _ax_modal.set_title(
        f"Frozen-configuration frequency rank {frequency_rank.value}"
    )
    _ax_modal.grid(True)

    _fig_modal.tight_layout()
    _fig_modal
    return


@app.cell
def _():
    nani=1
    asdf=2
    what=3
    could=4
    be=5
    wrong=6.0
    asdf=3
    comeon=4
    whyisdotis=6
    fixed=7
    asakjshdfkljah=10
    return


if __name__ == "__main__":
    app.run()
