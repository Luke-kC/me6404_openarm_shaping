"""Trial selection, fresh-state, and CSV checks without a graphical viewer."""

import csv
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from me6404_openarm_shaping import robot_sim, simulation_helpers
from me6404_openarm_shaping.configuration import (
    InputShapingConfig,
    RobotConfig,
    SimulationAssessmentConfig,
    SimulationConfig,
    ZVDConfig,
    experiment_fingerprint,
    load_robot_config,
)
from me6404_openarm_shaping.data_types import (
    ContactEvent,
    JointTrajectory,
    RunProvenance,
    ShaperDesign,
    TrialRecording,
)
from me6404_openarm_shaping.robot_model import (
    MODEL_REVISION,
    eval_gravity_torques,
    load_model,
    nominal_configuration,
    resolve_joint_indices,
)


def _recording(samples: int) -> TrialRecording:
    return {
        "t": np.arange(samples, dtype=np.float64) * 0.002,
        "q": np.full((samples, 7), 0.1),
        "dq_measured": np.full((samples, 7), 0.005),
        "ddq_measured": np.full((samples, 7), 0.6),
        "torque_requested": np.full((samples, 7), 2.0),
        "torque_feedback": np.full((samples, 7), 1.0),
        "q_command": np.full((samples, 7), 0.2),
        "dq_reference": np.full((samples, 7), 0.3),
        "ddq_reference": np.full((samples, 7), 0.4),
        "completed": True,
        "final_time_s": samples * 0.002,
        "final_q": np.full(7, 0.1),
        "final_dq": np.full(7, 0.005),
        "final_ddq": np.full(7, 0.6),
        "contact_events": [],
        "simulation_checks": {
            "passing": True,
            "reasons": [],
            "informational_contacts": [],
            "measured_acceleration": {
                "peak_abs_rad_s2": [0.6] * 7,
                "reference_limits_rad_s2": [10.0] * 7,
                "exceedances": [],
            },
            "categories": {
                "reference_limits": True,
                "position_limits": True,
                "measured_velocity_limits": True,
                "torque_clipping": True,
                "collision_checks": True,
                "settling": True,
                "data_validity": True,
            },
            "settling": {
                "evaluated": True,
                "passing": True,
                "command_end_time_s": 0.0,
                "window_start_s": 0.0,
                "window_end_s": samples * 0.002,
                "position_tolerance_rad": 0.01,
                "velocity_tolerance_rad_s": 0.01,
                "target_q_rad": [0.1] * 7,
                "final_position_error_rad": [0.0] * 7,
                "final_velocity_rad_s": [0.005] * 7,
                "max_abs_position_error_rad": [0.0] * 7,
                "max_abs_velocity_rad_s": [0.005] * 7,
            },
            "scope": "Test recording",
            "notice": "Test recording; no hardware authorization.",
        },
    }


def _provenance() -> RunProvenance:
    lockfile_bytes = b"version = 1\n"
    return {
        "captured_at_utc": "2026-11-15T11:00:00+00:00",
        "project_git": {"commit_sha": "a" * 40, "dirty": True},
        "model_path": "/project/model.xml",
        "model_sha256": sha256(b"<mujoco/>").hexdigest(),
        "lockfile_source": "/project/uv.lock",
        "lockfile_sha256": sha256(lockfile_bytes).hexdigest(),
        "lockfile_bytes": lockfile_bytes,
    }


def _fake_trial(*_args: object, **_kwargs: object) -> TrialRecording:
    return _recording(2)


@pytest.mark.parametrize("command_duration", [0.1, 6.0])
def test_unshaped_waypoint_trial(
    monkeypatch: pytest.MonkeyPatch,
    command_duration: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["robot_sim", "--save-csv"])
    provenance = _provenance()
    capture = MagicMock(return_value=provenance)
    monkeypatch.setattr(robot_sim, "capture_run_provenance", capture)
    qi = np.zeros(7)
    monkeypatch.setattr(robot_sim, "load_model", lambda: MagicMock())
    config = RobotConfig(
        motor_kp=np.ones(7),
        motor_kd=np.ones(7),
        velocity_limits=np.full(7, 10.0),
        acceleration_limits=np.full(7, 20.0),
        waypoints=np.array([qi + 0.02, qi + 0.03, qi + 0.04]),
        simulation=SimulationConfig(
            dt_s=0.004, minimum_duration_s=4.5, settling_time_s=0.25
        ),
        simulation_assessment=SimulationAssessmentConfig(0.25, 0.01, 0.01),
        input_shaping=InputShapingConfig(
            enabled=False, type="zvd", zvd=ZVDConfig(tau_min_s=0.2, utilization_min=0.4)
        ),
    )
    monkeypatch.setattr(robot_sim, "load_robot_config", lambda: config)
    validate = MagicMock()
    monkeypatch.setattr(robot_sim, "check_joint_configuration", validate)
    monkeypatch.setattr(robot_sim, "eval_gravity_torques", lambda model, q: qi)
    trajectory = (
        np.array([0, command_duration]),
        np.zeros((2, 7)),
        np.zeros((2, 7)),
        np.zeros((2, 7)),
    )
    generator = MagicMock(return_value=trajectory)
    monkeypatch.setattr(robot_sim, "waypoint_trajectory", generator)
    trial = MagicMock(side_effect=_fake_trial)
    monkeypatch.setattr(robot_sim, "run_trial", trial)
    save = MagicMock()
    monkeypatch.setattr(robot_sim, "save_experiment", save)
    plot = MagicMock()
    monkeypatch.setattr(robot_sim, "plot_comparison", plot)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    robot_sim.main()

    np.testing.assert_array_equal(generator.call_args.args[1], config.velocity_limits)
    np.testing.assert_array_equal(
        generator.call_args.args[2], config.acceleration_limits
    )
    np.testing.assert_array_equal(generator.call_args.args[0], config.waypoints)
    assert generator.call_args.kwargs["dt"] == 0.004
    assert validate.call_count == len(config.waypoints)
    expected = ["unshaped"]
    trajectories, recordings = plot.call_args.args
    assert list(trajectories) == list(recordings) == expected
    assert trial.call_count == len(expected)
    for call, label in zip(trial.call_args_list, expected, strict=True):
        assert call.args[1] is trajectories[label]
        assert call.kwargs["config"] is config
    np.testing.assert_array_equal(trial.call_args.args[2], config.waypoints[0])
    assert trial.call_args.args[-1] == max(4.5, command_duration + 0.25)
    save.assert_called_once()
    assert save.call_args.args[0] is recordings
    assert save.call_args.args[1] is config
    assert save.call_args.args[2] is None
    assert save.call_args.args[3] == max(4.5, command_duration + 0.25)
    capture.assert_called_once_with()
    assert save.call_args.kwargs["provenance"] is provenance
    fingerprint = experiment_fingerprint(
        config, provenance["model_sha256"], MODEL_REVISION
    )
    assert f"Experiment configuration: {fingerprint}" in capsys.readouterr().out
    assert save.call_args.kwargs["configuration_fingerprint"] == fingerprint


def test_whole_trajectory_shaper_design(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["robot_sim", "--save-csv"])
    monkeypatch.setattr(robot_sim, "capture_run_provenance", _provenance)
    model = MagicMock()
    model.opt.timestep = 0.01
    monkeypatch.setattr(robot_sim, "load_model", lambda: model)
    config = RobotConfig(
        motor_kp=np.ones(7),
        motor_kd=np.ones(7),
        velocity_limits=np.full(7, 10.0),
        acceleration_limits=np.full(7, 20.0),
        waypoints=np.zeros((3, 7)),
        simulation=SimulationConfig(
            dt_s=0.01, minimum_duration_s=5.0, settling_time_s=1.25
        ),
        simulation_assessment=SimulationAssessmentConfig(0.25, 0.01, 0.01),
        input_shaping=InputShapingConfig(
            enabled=True, type="zvd", zvd=ZVDConfig(tau_min_s=0.2, utilization_min=0.4)
        ),
    )
    monkeypatch.setattr(robot_sim, "load_robot_config", lambda: config)
    monkeypatch.setattr(robot_sim, "check_joint_configuration", lambda *a: None)
    monkeypatch.setattr(robot_sim, "eval_gravity_torques", lambda *a: np.zeros(7))
    t = np.array([0.0, 0.01, 0.02])
    q = np.arange(21, dtype=np.float64).reshape(3, 7) / 100
    dq, ddq = np.zeros_like(q), np.zeros_like(q)
    monkeypatch.setattr(
        robot_sim, "waypoint_trajectory", lambda *a, **kw: (t, q, dq, ddq)
    )
    modal_results = [{"sample": index} for index in range(len(t))]
    modal = MagicMock(side_effect=modal_results)
    monkeypatch.setattr(robot_sim, "modal_properties", modal)
    frequencies, damping, utilization = (
        np.array([2.2]),
        np.array([0.1]),
        np.array([0.8]),
    )
    suggest = MagicMock(return_value=(frequencies, damping, utilization))
    monkeypatch.setattr(robot_sim, "suggest_zvd_parameters", suggest)
    amplitudes, delays = np.array([1.0]), np.array([0.0])
    design = MagicMock(return_value=(amplitudes, delays))
    monkeypatch.setattr(robot_sim, "design_zvd_shaper", design)
    shaped = (
        np.array([0.0, 5.5]),
        np.zeros((2, 7)),
        np.zeros((2, 7)),
        np.zeros((2, 7)),
    )
    apply = MagicMock(return_value=shaped)
    monkeypatch.setattr(robot_sim, "apply_shaper_to_joint_trajectory", apply)
    trial = MagicMock(side_effect=_fake_trial)
    monkeypatch.setattr(robot_sim, "run_trial", trial)
    save = MagicMock()
    monkeypatch.setattr(robot_sim, "save_experiment", save)
    plot = MagicMock()
    monkeypatch.setattr(robot_sim, "plot_comparison", plot)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    robot_sim.main()

    assert modal.call_count == len(q)
    for call, configuration in zip(modal.call_args_list, q, strict=True):
        np.testing.assert_array_equal(call.args[1], configuration)
    suggest.assert_called_once_with(modal_results, tau_min=0.2, utilization_min=0.4)
    design.assert_called_once()
    assert design.call_args.args[0] is frequencies
    assert design.call_args.args[1] is damping
    apply.assert_called_once()
    for value, expected in zip(
        apply.call_args.args, (q, dq, ddq, amplitudes, delays, 0.01), strict=True
    ):
        np.testing.assert_array_equal(value, expected)
    trajectories, recordings = plot.call_args.args
    assert list(trajectories) == list(recordings) == ["unshaped", "shaped"]
    assert trial.call_count == 2
    for call, label in zip(trial.call_args_list, trajectories, strict=True):
        assert call.args[1] is trajectories[label]
        assert call.args[-1] == pytest.approx(
            6.75
        )  # Includes configured settling time.
    save.assert_called_once()
    saved_design = save.call_args.args[2]
    assert saved_design["natural_frequencies_hz"] == [2.2]
    assert saved_design["damping_ratios"] == [0.1]
    assert saved_design["amplitudes"] == [1.0]
    assert saved_design["rounded_delay_steps"] == [0]
    assert saved_design["applied_delays_s"] == [0.0]


def test_trials_start_from_identical_fresh_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = load_model()
    position_indices, velocity_indices = resolve_joint_indices(model)
    qi = nominal_configuration(model)[position_indices]
    config = load_robot_config()
    Kp, Kd = config.motor_kp, config.motor_kd
    trajectory: JointTrajectory = {
        "t": np.arange(2, dtype=np.float64) * model.opt.timestep,
        "q": np.vstack((qi, qi + 0.01)),
        "dq": np.array([[0.5] * 7, [0.1] * 7]),
        "ddq": np.array([[0.2] * 7, [-0.2] * 7]),
    }
    initial_states: list[
        tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    data_instances: list[mujoco.MjData] = []
    step_velocities: list[NDArray[np.float64]] = []
    step_accelerations: list[NDArray[np.float64]] = []
    step_times: list[float] = []
    original_step = mujoco.mj_step

    def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        step_times.append(float(data.time))
        step_velocities.append(data.qvel[velocity_indices].copy())
        original_step(model, data)
        step_accelerations.append(data.qacc[velocity_indices].copy())

    def launch(_model: mujoco.MjModel, data: mujoco.MjData) -> MagicMock:
        data_instances.append(data)
        initial_states.append(
            (data.time, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
        )
        viewer = MagicMock()
        viewer.__enter__.return_value = viewer
        viewer.is_running.return_value = True
        return viewer

    monkeypatch.setattr("mujoco.viewer.launch_passive", launch)
    monkeypatch.setattr("time.sleep", lambda duration: None)
    monkeypatch.setattr(mujoco, "mj_step", step)
    first = simulation_helpers.run_trial(
        model, trajectory, qi, Kp, Kd, 4 * model.opt.timestep, config=config
    )
    second = simulation_helpers.run_trial(
        model, trajectory, qi, Kp, Kd, 4 * model.opt.timestep, config=config
    )
    assert data_instances[0] is not data_instances[1]
    unselected = np.setdiff1d(np.arange(model.nq), position_indices)
    for initial in initial_states:
        assert initial[0] == 0
        np.testing.assert_allclose(initial[1][position_indices], qi, atol=1e-14)
        np.testing.assert_allclose(
            initial[1][unselected], model.qpos0[unselected], rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(initial[2], 0, atol=1e-14)
        np.testing.assert_allclose(initial[3], 0, atol=1e-14)
    for left, right in zip(initial_states[0], initial_states[1], strict=True):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)
    for key in (
        "t",
        "q",
        "torque_requested",
        "torque_feedback",
        "q_command",
        "dq_reference",
        "ddq_reference",
        "dq_measured",
        "ddq_measured",
        "final_dq",
        "final_ddq",
    ):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-12, atol=1e-14)
    assert first["completed"] and second["completed"]
    assert first["t"][0] == second["t"][0] == 0
    np.testing.assert_allclose(
        first["q_command"],
        np.vstack((qi, qi + 0.01, qi + 0.01, qi + 0.01)),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        first["dq_reference"],
        [[0.5] * 7, [0.1] * 7, [0] * 7, [0] * 7],
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        first["ddq_reference"],
        [[0.2] * 7, [-0.2] * 7, [0] * 7, [0] * 7],
        rtol=0,
        atol=1e-14,
    )
    # At zero position error and measured speed, only the existing commanded-
    # pose gravity feedforward remains; velocity references do not drive PD.
    np.testing.assert_allclose(
        first["torque_requested"][0],
        eval_gravity_torques(model, qi),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(first["dq_measured"][:1], 0, atol=1e-14)
    # qacc is the solver acceleration, not a finite difference: Euler handles
    # joint damping implicitly when updating velocity.
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    np.testing.assert_allclose(first["t"], step_times[:4], rtol=0, atol=0)
    np.testing.assert_allclose(
        first["dq_measured"], step_velocities[:4], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        first["ddq_measured"],
        step_accelerations[:4],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        first["final_ddq"],
        data_instances[0].qacc[velocity_indices],
    )
    assert not np.allclose(first["dq_measured"], first["dq_reference"])


def test_real_trial_metadata_serializes_command_end_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = load_model()
    config = load_robot_config()
    position_indices, _ = resolve_joint_indices(model)
    qi = nominal_configuration(model)[position_indices]
    trajectory: JointTrajectory = {
        "t": np.arange(2, dtype=np.float64) * model.opt.timestep,
        "q": np.vstack((qi, qi)),
        "dq": np.zeros((2, 7)),
        "ddq": np.zeros((2, 7)),
    }
    viewer = MagicMock(is_running=MagicMock(return_value=True))
    viewer.configure_mock(**{"__enter__.return_value": viewer})

    def launch(_model: mujoco.MjModel, _data: mujoco.MjData) -> MagicMock:
        return viewer

    def no_sleep(_duration: float) -> None:
        pass

    monkeypatch.setattr("mujoco.viewer.launch_passive", launch)
    monkeypatch.setattr("time.sleep", no_sleep)
    recording = simulation_helpers.run_trial(
        model,
        trajectory,
        qi,
        config.motor_kp,
        config.motor_kd,
        2 * model.opt.timestep,
        config=config,
    )
    provenance = _provenance()
    fingerprint = experiment_fingerprint(
        config, provenance["model_sha256"], MODEL_REVISION
    )
    folder = simulation_helpers.save_experiment(
        {"unshaped": recording},
        config,
        None,
        2 * model.opt.timestep,
        tmp_path,
        provenance=provenance,
        configuration_fingerprint=fingerprint,
    )
    metadata_text = (folder / f"{folder.name}_metadata.yaml").read_text()
    assert (
        type(recording["simulation_checks"]["settling"]["command_end_time_s"]) is float
    )
    assert f"command_end_time_s: {model.opt.timestep}\n" in metadata_text
    assert yaml.safe_dump(
        recording["simulation_checks"]["measured_acceleration"]
    ).strip()
    assert "measured_acceleration:" in metadata_text
    assert "peak_abs_rad_s2:" in metadata_text
    assert "categories:" in metadata_text
    assert (folder / f"{folder.name}_results.csv").is_file()


def test_combined_csv_retains_both_trials(tmp_path: Path) -> None:
    # Different lengths model an interrupted second trial.
    recordings = {"unshaped": _recording(3), "shaped": _recording(2)}
    recordings["shaped"]["q_command"][:] = 0.9
    path = simulation_helpers.save_recordings_csv(recordings, tmp_path / "results.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert [row["trajectory"] for row in rows] == ["unshaped"] * 3 + ["shaped"] * 2
    assert float(rows[0]["time_s"]) == float(rows[3]["time_s"]) == 0
    assert float(rows[2]["time_s"]) == pytest.approx(0.004)
    assert float(rows[0]["J1_q_rad"]) == pytest.approx(0.1)
    assert float(rows[0]["J7_torque_requested_Nm"]) == pytest.approx(2.0)
    assert float(rows[-1]["J7_torque_feedback_Nm"]) == pytest.approx(1.0)
    assert float(rows[0]["J1_q_command_rad"]) == pytest.approx(0.2)
    assert float(rows[3]["J1_q_command_rad"]) == pytest.approx(0.9)
    assert float(rows[-1]["J7_dq_reference_rad_s"]) == pytest.approx(0.3)
    assert float(rows[-1]["J7_ddq_reference_rad_s2"]) == pytest.approx(0.4)
    assert float(rows[-1]["J7_dq_measured_rad_s"]) == pytest.approx(0.005)
    assert float(rows[-1]["J7_ddq_measured_rad_s2"]) == pytest.approx(0.6)


def test_save_experiment_unique_files_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unique suffixes must distinguish saves even with an identical timestamp.
    clock = MagicMock()
    clock.now.return_value = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(simulation_helpers, "datetime", clock)
    config = load_robot_config()
    recordings = {"unshaped": _recording(3), "shaped": _recording(2)}
    recordings["shaped"]["completed"] = False
    recordings["shaped"]["simulation_checks"]["passing"] = False
    recordings["shaped"]["simulation_checks"]["reasons"] = [
        {"kind": "incomplete_run", "message": "Test interrupted run"}
    ]
    # Synthetic event verifies that expected contacts survive YAML serialization.
    contact: ContactEvent = {
        "kind": "penetrating_contact",
        "bodies": ["openarm_left_right_finger", "openarm_left_left_finger"],
        "geoms": [
            "openarm_left_right_finger_collision",
            "openarm_left_left_finger_collision",
        ],
        "first_time_s": 0.0,
        "first_separation_m": -0.001,
        "worst_time_s": 0.002,
        "minimum_separation_m": -0.002,
        "sample_count": 2,
    }
    recordings["shaped"]["simulation_checks"]["informational_contacts"] = [contact]
    design: ShaperDesign = {
        "type": "zvd",
        "natural_frequencies_hz": [2.2],
        "damping_ratios": [0.1],
        "band_utilization": [0.8],
        "amplitudes": [0.5, 0.5],
        "ideal_delays_s": [0.0, 0.003],
        "rounded_delay_steps": [0, 2],
        "applied_delays_s": [0.0, 0.004],
    }
    fingerprint = experiment_fingerprint(
        config, _provenance()["model_sha256"], MODEL_REVISION
    )
    folders = [
        simulation_helpers.save_experiment(
            recordings,
            config,
            design,
            5.0,
            tmp_path,
            provenance=_provenance(),
            configuration_fingerprint=fingerprint,
        )
        for _ in range(2)
    ]
    assert folders[0] != folders[1]
    for folder in folders:
        results = folder / f"{folder.name}_results.csv"
        metadata_file = folder / f"{folder.name}_metadata.yaml"
        lockfile = folder / f"{folder.name}_uv.lock"
        assert sorted(path.name for path in folder.iterdir()) == sorted(
            [
                results.name,
                metadata_file.name,
                lockfile.name,
            ]
        )
        metadata = yaml.safe_load(metadata_file.read_text())
        assert metadata["run_id"] == folder.name
        assert metadata["results_file"] == results.name
        assert metadata["saved_at_utc"] == "2026-11-15T12:00:00+00:00"
        assert metadata["config"]["waypoints"] == config.waypoints.tolist()
        assert metadata["config"]["motor_kd"] == config.motor_kd.tolist()
        assert metadata["config"]["simulation"] == asdict(config.simulation)
        assert metadata["config"]["input_shaping"] == asdict(config.input_shaping)
        assert metadata["config"]["simulation_assessment"] == asdict(
            config.simulation_assessment
        )
        assert (
            metadata["trials"]["unshaped"]["simulation_checks"]["settling"]
            == (recordings["unshaped"]["simulation_checks"]["settling"])
        )
        for key in ("measured_acceleration", "categories"):
            assert (
                metadata["trials"]["unshaped"]["simulation_checks"][key]
                == (recordings["unshaped"]["simulation_checks"][key])
            )
        assert metadata["experiment_configuration_sha256"] == fingerprint
        assert metadata["shaper_design"] == design
        assert metadata["trials"]["shaped"]["completed"] is False
        assert metadata["trials"]["shaped"]["simulation_checks"]["passing"] is False
        assert metadata["trials"]["shaped"]["simulation_checks"]["reasons"] == [
            {"kind": "incomplete_run", "message": "Test interrupted run"}
        ]
        assert metadata["trials"]["shaped"]["simulation_checks"][
            "informational_contacts"
        ] == [contact]
        assert metadata["trials"]["unshaped"]["samples"] == 3
        assert metadata["trials"]["shaped"]["recorded_duration_s"] == pytest.approx(
            2 * config.simulation.dt_s
        )
        assert metadata["model"]["upstream_revision"]
        assert metadata["model"]["sha256"] == _provenance()["model_sha256"]
        assert metadata["project_git"] == _provenance()["project_git"]
        assert (
            metadata["provenance_captured_at_utc"] == _provenance()["captured_at_utc"]
        )
        assert lockfile.read_bytes() == _provenance()["lockfile_bytes"]
        assert metadata["dependency_lockfile"]["file"] == lockfile.name
        assert (
            metadata["dependency_lockfile"]["sha256"]
            == sha256(lockfile.read_bytes()).hexdigest()
        )
        assert set(metadata["software"]) == {"python", "numpy", "mujoco"}
        with results.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 5
        assert rows[0]["trajectory"] == "unshaped"
        assert rows[-1]["trajectory"] == "shaped"
        assert float(rows[-1]["J7_ddq_reference_rad_s2"]) == pytest.approx(0.4)


@pytest.mark.parametrize(
    "status", ["", " M robot.py\n", "M  robot.py\n", "?? new.py\n"]
)
def test_capture_run_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_bytes(b"version = 1\n")
    model_path = tmp_path / "model.xml"
    model_path.write_bytes(b"<mujoco/>")
    git = MagicMock(
        side_effect=[
            CompletedProcess([], 0, stdout="b" * 40 + "\n"),
            CompletedProcess([], 0, stdout=status),
        ]
    )
    monkeypatch.setattr(simulation_helpers.subprocess, "run", git)
    captured = simulation_helpers.capture_run_provenance(tmp_path, model_path)
    assert captured["project_git"] == {"commit_sha": "b" * 40, "dirty": bool(status)}
    assert captured["model_sha256"] == sha256(b"<mujoco/>").hexdigest()
    assert captured["lockfile_sha256"] == sha256(b"version = 1\n").hexdigest()
    assert git.call_args_list[0].args[0] == ["git", "rev-parse", "HEAD"]
    assert "--untracked-files=normal" in git.call_args_list[1].args[0]
    assert "--ignore-submodules=none" in git.call_args_list[1].args[0]
    assert all(call.kwargs["cwd"] == tmp_path for call in git.call_args_list)
    # Later edits cannot change the captured bytes or hashes.
    lockfile.write_bytes(b"edited after startup\n")
    model_path.write_bytes(b"edited model\n")
    assert captured["lockfile_bytes"] == b"version = 1\n"
    assert captured["model_sha256"] == sha256(b"<mujoco/>").hexdigest()


def test_capture_unavailable_git_is_not_reported_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"version = 1\n")
    model_path = tmp_path / "model.xml"
    model_path.write_bytes(b"<mujoco/>")
    monkeypatch.setattr(
        simulation_helpers.subprocess,
        "run",
        MagicMock(side_effect=FileNotFoundError("git")),
    )
    captured = simulation_helpers.capture_run_provenance(tmp_path, model_path)
    assert captured["project_git"]["commit_sha"] is None
    assert captured["project_git"]["dirty"] is None
    assert "errors" in captured["project_git"]
    assert set(captured["project_git"]["errors"]) == {"commit_sha", "dirty"}
