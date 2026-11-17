"""Analytical limit checks and tiny MuJoCo contact cases; no graphical viewer."""

from typing import Literal, cast
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest

from me6404_openarm_shaping.configuration import (
    RobotConfig,
    SimulationAssessmentConfig,
    load_robot_config,
)
from me6404_openarm_shaping.data_types import (
    ContactEvent,
    JointTrajectory,
    SimulationChecks,
    TrialSamples,
)
from me6404_openarm_shaping.robot_model import (
    load_model,
    nominal_configuration,
    resolve_joint_indices,
)
from me6404_openarm_shaping.simulation_helpers import run_trial
from me6404_openarm_shaping.simulation_safety import (
    assess_trial as assess_recording,
)
from me6404_openarm_shaping.simulation_safety import (
    print_simulation_checks,
    record_penetrating_contacts,
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return load_model()


def _config() -> RobotConfig:
    config = load_robot_config()
    config.waypoints = np.zeros((2, 7))
    config.simulation_assessment = SimulationAssessmentConfig(0.002, 0.01, 0.01, 0.05, 0.05)
    return config


def assess_trial(model: mujoco.MjModel, result: TrialSamples) -> SimulationChecks:
    return assess_recording(
        model, result, _config(), command_end_time_s=0.0, trajectory=_reference()
    )


def _reference(end_time_s: float = 0.0) -> JointTrajectory:
    return {
        "t": np.array([end_time_s]),
        "q": np.zeros((1, 7)),
        "dq": np.zeros((1, 7)),
        "ddq": np.zeros((1, 7)),
    }


def _result() -> TrialSamples:
    return {
        "t": np.array([0.0, 0.002, 0.004]),
        "q": np.zeros((3, 7)),
        "dq_measured": np.zeros((3, 7)),
        "ddq_measured": np.zeros((3, 7)),
        "torque_requested": np.zeros((3, 7)),
        "completed": True,
        "final_time_s": 0.006,
        "final_q": np.zeros(7),
        "final_dq": np.zeros(7),
        "final_ddq": np.zeros(7),
        "contact_events": [],
    }


@pytest.mark.parametrize("margin", [0.0, 0.05, 0.1])
@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("excess", [-0.001, 0.0, 0.001])
def test_j4_measured_lower_position_tolerance(
    model: mujoco.MjModel, margin: float, final: bool, excess: float
) -> None:
    config = _config()
    config.simulation_assessment.j4_lower_position_tolerance_rad = margin
    result = _result()
    lower = model.joint("openarm_left_joint4").range[0] - margin
    if final:
        result["final_q"][3] = lower - excess
    else:
        result["q"][0, 3] = lower - excess
    report = assess_recording(
        model, result, config, command_end_time_s=0.0, trajectory=_reference()
    )
    # Settling is deliberately separate from the measured position check.
    assert report["categories"]["position_limits"] is (excess <= 0)
    if excess > 0:
        reason = next(r for r in report["reasons"] if r["kind"] == "position_limit")
        assert "lower_limit" in reason
        assert reason["lower_limit"] == lower
        assert reason["first_time_s"] == (result["final_time_s"] if final else 0.0)


@pytest.mark.parametrize("case", ["j4-reference-lower", "j4-measured-upper", "j1-measured-lower"])
def test_j4_allowance_does_not_relax_other_position_limits(
    model: mujoco.MjModel, case: str
) -> None:
    result = _result()
    reference = _reference()
    category: Literal["reference_limits", "position_limits"]
    if case == "j4-reference-lower":
        reference["q"][0, 3] = -0.001
        category = "reference_limits"
    elif case == "j4-measured-upper":
        result["q"][0, 3] = model.joint("openarm_left_joint4").range[1] + 0.001
        category = "position_limits"
    else:
        result["q"][0, 0] = model.joint("openarm_left_joint1").range[0] - 0.001
        category = "position_limits"
    report = assess_recording(
        model, result, _config(), command_end_time_s=0.0, trajectory=reference
    )
    assert not report["categories"][category]
    assert any(reason["kind"] == "position_limit" for reason in report["reasons"])


def test_clean_completed_trial_passes(model: mujoco.MjModel) -> None:
    report = assess_trial(model, _result())
    assert report["passing"]
    assert report["reasons"] == []
    assert report["informational_contacts"] == []
    assert report["settling"]["evaluated"]
    assert report["settling"]["passing"]
    assert all(report["categories"].values())


@pytest.mark.parametrize("quantity", ["q", "dq", "ddq"])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_reference_limits_are_hard_checks(
    model: mujoco.MjModel,
    quantity: Literal["q", "dq", "ddq"],
    nonfinite: bool,
) -> None:
    config = _config()
    reference = _reference()
    if nonfinite:
        reference[quantity][0, 0] = np.nan
    elif quantity == "q":
        reference["q"][0, 0] = 100.0  # Beyond the pinned arm's joint range.
    elif quantity == "dq":
        reference["dq"][0, 0] = config.velocity_limits[0] + 0.01
    else:
        reference["ddq"][0, 0] = -config.acceleration_limits[0] - 0.1
    report = assess_recording(
        model,
        _result(),
        config,
        command_end_time_s=0.0,
        trajectory=reference,
    )
    assert not report["passing"]
    assert not report["categories"]["reference_limits"]
    assert report["categories"]["position_limits"]
    assert report["categories"]["measured_velocity_limits"]
    (reason,) = report["reasons"]
    assert "joint" in reason
    assert reason.get("source") == "reference"
    assert reason["joint"] == "openarm_left_joint1"
    assert report["categories"]["data_validity"] is not nonfinite


def test_complete_reference_is_checked_for_interrupted_run(
    model: mujoco.MjModel,
) -> None:
    reference: JointTrajectory = {
        "t": np.array([0.0, 0.1]),
        "q": np.zeros((2, 7)),
        "dq": np.zeros((2, 7)),
        "ddq": np.zeros((2, 7)),
    }
    reference["ddq"][1, 0] = 11.0
    result = _result()
    result["completed"] = False
    report = assess_recording(
        model,
        result,
        _config(),
        command_end_time_s=0.1,
        trajectory=reference,
    )
    assert not report["categories"]["reference_limits"]
    assert any(issue["kind"] == "acceleration_limit" for issue in report["reasons"])


def test_measured_acceleration_is_diagnostic_in_console_and_metadata(
    model: mujoco.MjModel,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _result()
    result["ddq_measured"][0] = [-65.05, 48.84, -63.16, 1, 2, 3, 4]
    report = assess_trial(model, result)
    assert report["passing"]
    assert report["reasons"] == []
    assert report["measured_acceleration"]["peak_abs_rad_s2"] == [
        65.05,
        48.84,
        63.16,
        1,
        2,
        3,
        4,
    ]
    assert len(report["measured_acceleration"]["exceedances"]) == 3
    print_simulation_checks("unshaped", report)
    output = capsys.readouterr().out
    assert "unshaped: PASSING SIM — selected safety checks" in output
    assert "J1 peak: 65.05 rad/s²" in output
    assert "J2 peak: 48.84 rad/s²" in output
    assert "J7 peak: 4.00 rad/s²" in output
    for category in (
        "Reference limits",
        "Position limits",
        "Measured velocity limits",
        "Torque clipping",
        "Collision checks",
        "Settling",
        "Data validity",
    ):
        assert f"{category}: PASS" in output
    assert "reference-limit exceedances (informational)" in output
    assert "Worst: -65.05 rad/s^2 at 0.000000 s" in output


def test_nonfinite_acceleration_alone_still_fails(model: mujoco.MjModel) -> None:
    result = _result()
    result["final_ddq"][0] = np.nan
    report = assess_trial(model, result)
    assert not report["passing"]
    assert not report["categories"]["data_validity"]
    assert report["measured_acceleration"]["peak_abs_rad_s2"][0] is None
    (reason,) = report["reasons"]
    assert reason["kind"] == "nonfinite_acceleration_limit"


@pytest.mark.parametrize("quantity", ["velocity", "acceleration"])
def test_measured_motion_limits_include_final_state(
    model: mujoco.MjModel, quantity: str
) -> None:
    config = _config()
    result = _result()
    limit: float
    if quantity == "velocity":
        limit = cast(float, config.velocity_limits[1])
        result["dq_measured"][:, 1] = [limit, limit + 1, -limit - 2]
        result["final_dq"][1] = limit + 3
    else:
        limit = cast(float, config.acceleration_limits[1])
        result["ddq_measured"][:, 1] = [limit, limit + 1, -limit - 2]
        result["final_ddq"][1] = limit + 3
    report = assess_recording(
        model, result, config, command_end_time_s=0.0, trajectory=_reference()
    )
    if quantity == "velocity":
        assert not report["passing"]
        issues = report["reasons"]
    else:
        assert report["passing"]
        issues = report["measured_acceleration"]["exceedances"]
        assert report["measured_acceleration"]["peak_abs_rad_s2"][1] == limit + 3
    reason = next(item for item in issues if item["kind"] == f"{quantity}_limit")
    assert "lower_limit" in reason
    assert reason["joint"] == "openarm_left_joint2"
    assert reason["first_time_s"] == pytest.approx(0.002)
    assert reason["first_value"] == limit + 1
    assert reason["worst_time_s"] == pytest.approx(0.006)
    assert reason["worst_value"] == limit + 3
    margin = (
        config.simulation_assessment.measured_velocity_tolerance_rad_s
        if quantity == "velocity"
        else 0.0
    )
    assert reason["lower_limit"] == -limit - margin
    assert reason["upper_limit"] == limit + margin
    assert reason["sample_count"] == 3


@pytest.mark.parametrize("margin", [0.0, 0.05, 0.12])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("excess", [-0.001, 0.0, 0.001])
def test_measured_velocity_tolerance_boundary(
    model: mujoco.MjModel,
    margin: float,
    direction: int,
    final: bool,
    excess: float,
) -> None:
    config = _config()
    config.simulation_assessment.measured_velocity_tolerance_rad_s = margin
    result = _result()
    speed = direction * (config.velocity_limits[0] + margin + excess)
    if final:
        result["final_dq"][0] = speed
    else:
        result["dq_measured"][0, 0] = speed
    report = assess_recording(
        model, result, config, command_end_time_s=0.0, trajectory=_reference()
    )
    # Settling remains a separate check, especially for the final-state case.
    assert report["categories"]["measured_velocity_limits"] is (excess <= 0)
    issues = [issue for issue in report["reasons"] if issue["kind"] == "velocity_limit"]
    assert len(issues) == int(excess > 0)
    if issues:
        reason = issues[0]
        assert "lower_limit" in reason
        assert reason["lower_limit"] == -config.velocity_limits[0] - margin
        assert reason["upper_limit"] == config.velocity_limits[0] + margin
        assert reason["first_time_s"] == (result["final_time_s"] if final else 0.0)


@pytest.mark.parametrize("final", [False, True])
def test_nonfinite_derivatives_fail(model: mujoco.MjModel, final: bool) -> None:
    result = _result()
    if final:
        result["final_dq"][1] = np.nan
        result["final_ddq"][2] = np.inf
    else:
        result["dq_measured"][1, 1] = np.nan
        result["ddq_measured"][1, 2] = np.inf
    report = assess_trial(model, result)
    assert not report["passing"]
    assert {"nonfinite_velocity_limit", "nonfinite_acceleration_limit"} <= {
        reason["kind"] for reason in report["reasons"]
    }


def test_settling_uses_target_and_entire_approved_window(model: mujoco.MjModel) -> None:
    config = _config()
    config.simulation_assessment = SimulationAssessmentConfig(0.25, 0.01, 0.01, 0.05, 0.05)
    config.waypoints[-1] = 0.1
    result = _result()
    result["t"] = np.array([0.0, 0.25, 0.5])
    result["final_time_s"] = 0.75
    result["q"][:] = 0.1
    result["final_q"][:] = 0.1
    # Earlier motion outside the final window is allowed.
    result["q"][1, 0] = 0.2
    report = assess_recording(
        model, result, config, command_end_time_s=0.5, trajectory=_reference(0.5)
    )
    assert report["passing"]
    assert report["settling"]["window_start_s"] == 0.5
    assert report["settling"]["max_abs_position_error_rad"] == [0.0] * 7
    assert report["settling"]["max_abs_velocity_rad_s"] == [0.0] * 7

    # Good final values cannot hide an earlier bad sample in the window.
    result["q"][2, 0] += 0.02
    result["dq_measured"][2, 6] = -0.02
    report = assess_recording(
        model, result, config, command_end_time_s=0.5, trajectory=_reference(0.5)
    )
    assert not report["passing"]
    assert not report["settling"]["passing"]
    assert report["settling"]["final_position_error_rad"] == [0.0] * 7
    assert report["settling"]["final_velocity_rad_s"] == [0.0] * 7
    assert {reason["kind"] for reason in report["reasons"]} == {
        "settling_position_error",
        "settling_velocity",
    }


def test_settling_includes_final_state_and_accepts_tolerance_boundary(
    model: mujoco.MjModel,
) -> None:
    result = _result()
    result["final_q"][0] = 0.01
    result["final_dq"][0] = -0.01
    assert assess_trial(model, result)["passing"]
    result["final_q"][0] = 0.011
    result["final_dq"][0] = -0.011
    report = assess_trial(model, result)
    assert {reason["kind"] for reason in report["reasons"]} == {
        "settling_position_error",
        "settling_velocity",
    }
    for reason in report["reasons"]:
        assert "lower_limit" in reason
        assert reason["first_time_s"] == pytest.approx(0.006)
        assert reason["sample_count"] == 1


def test_insufficient_post_command_window_fails(model: mujoco.MjModel) -> None:
    report = assess_recording(
        model,
        _result(),
        _config(),
        command_end_time_s=0.005,
        trajectory=_reference(0.005),
    )
    assert not report["passing"]
    assert not report["settling"]["evaluated"]
    assert report["settling"]["max_abs_velocity_rad_s"] is None
    assert report["reasons"][0]["kind"] == "insufficient_settling_window"


def test_closed_finger_contact_is_recorded_but_informational(
    model: mujoco.MjModel, capsys: pytest.CaptureFixture[str]
) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    events: dict[tuple[int, ...], ContactEvent] = {}
    record_penetrating_contacts(model, data, events, 0.0)
    expected_pair = {
        "openarm_left_right_finger_collision",
        "openarm_left_left_finger_collision",
    }
    finger_events = [
        event for event in events.values() if set(event["geoms"]) == expected_pair
    ]
    assert len(finger_events) == 1
    event = finger_events[0]
    assert event["minimum_separation_m"] < 0
    assert event["sample_count"] == 1

    result = _result()
    result["contact_events"] = finger_events
    report = assess_trial(model, result)
    assert report["passing"]
    assert report["reasons"] == []
    assert report["informational_contacts"] == finger_events
    print_simulation_checks("unshaped", report)
    output = capsys.readouterr().out
    assert "PASSING SIM" in output
    assert "Expected finger contact (informational; not a failure)" in output
    assert "Minimum separation:" in output
    assert "openarm_left_left_finger_collision" in output

    # Pair order does not change classification.
    reversed_event: ContactEvent = {**event, "geoms": event["geoms"][::-1]}
    result["contact_events"] = [reversed_event]
    assert assess_trial(model, result)["passing"]

    # A finger contacting any other geometry remains a collision failure.
    other_event: ContactEvent = {
        **event,
        "geoms": [
            "openarm_left_right_finger_collision",
            "openarm_left_link7_collision",
        ],
    }
    result["contact_events"] = [event, other_event]
    report = assess_trial(model, result)
    assert not report["passing"]
    assert report["informational_contacts"] == [event]
    assert report["reasons"] == [other_event]

    # Expected contact must not hide a joint-position violation either.
    result["contact_events"] = [event]
    result["final_q"][3] = -0.06  # Beyond the 0.05 rad measured J4 allowance.
    report = assess_trial(model, result)
    assert not report["passing"]
    assert report["informational_contacts"] == [event]
    assert report["reasons"][0]["kind"] == "position_limit"


def test_position_violation_reports_first_and_worst(model: mujoco.MjModel) -> None:
    result = _result()
    upper = model.joint("openarm_left_joint1").range[1]
    result["q"][1, 0] = upper + 0.1
    result["q"][2, 0] = upper + 0.3
    report = assess_trial(model, result)
    assert not report["passing"]
    reason = next(
        item for item in report["reasons"] if item["kind"] == "position_limit"
    )
    assert reason["kind"] == "position_limit"
    assert reason["joint"] == "openarm_left_joint1"
    assert reason["first_time_s"] == pytest.approx(0.002)
    assert reason["worst_time_s"] == pytest.approx(0.004)
    assert reason["worst_value"] == pytest.approx(upper + 0.3)
    assert reason["sample_count"] == 2


def test_torque_clipping_not_just_reaching_limit(
    model: mujoco.MjModel, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _result()
    actuator = model.actuator("left_joint1_ctrl")
    lower, upper = actuator.forcerange
    result["torque_requested"][:, 0] = [upper, upper + 1, lower - 2]
    report = assess_trial(model, result)
    assert not report["passing"]
    (reason,) = report["reasons"]
    assert reason["kind"] == "torque_clipping"
    assert reason["first_time_s"] == pytest.approx(0.002)
    assert reason["worst_value"] == pytest.approx(lower - 2)
    assert reason["sample_count"] == 2
    print_simulation_checks("shaped", report)
    output = capsys.readouterr().out
    assert "NOT PASSING SIM" in output
    assert "openarm_left_joint1 torque_clipping" in output
    assert "0.002000 s" in output
    assert "Torque clipping: FAIL" in output
    assert "\n\n  openarm_left_joint1 torque_clipping:\n" in output
    assert "\n    First:" in output
    assert "\n    Worst:" in output
    assert "\n    Allowed:" in output
    assert "\n    Samples: 2\n" in output


def test_final_integrated_position_is_checked(model: mujoco.MjModel) -> None:
    result = _result()
    lower = model.joint("openarm_left_joint7").range[0]
    result["final_q"][6] = lower - 0.1
    reason = next(
        item
        for item in assess_trial(model, result)["reasons"]
        if item["kind"] == "position_limit"
    )
    assert reason["kind"] == "position_limit"
    assert reason["joint"] == "openarm_left_joint7"
    assert reason["first_time_s"] == pytest.approx(0.006)


def test_incomplete_and_empty_runs_do_not_pass(model: mujoco.MjModel) -> None:
    result = _result()
    result["completed"] = False
    result["t"] = np.empty(0)
    result["q"] = np.empty((0, 7))
    result["dq_measured"] = np.empty((0, 7))
    result["ddq_measured"] = np.empty((0, 7))
    result["torque_requested"] = np.empty((0, 7))
    report = assess_trial(model, result)
    assert not report["passing"]
    assert {reason["kind"] for reason in report["reasons"]} == {
        "incomplete_run",
        "empty_recording",
    }


def test_nonfinite_recorded_data_does_not_pass(model: mujoco.MjModel) -> None:
    result = _result()
    result["q"][1, 2] = np.nan
    result["torque_requested"][2, 3] = np.inf
    report = assess_trial(model, result)
    assert not report["passing"]
    assert {reason["kind"] for reason in report["reasons"]} == {
        "nonfinite_position_limit",
        "nonfinite_torque_clipping",
    }


@pytest.mark.parametrize("body_name", ["openarm_left_link7", "other_arm"])
def test_penetrating_contacts_and_left_arm_scope(
    model: mujoco.MjModel, body_name: str
) -> None:
    contact_model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <worldbody>
            <body name="{body_name}">
              <freejoint/>
              <geom name="moving_sphere" type="sphere" size="0.1"/>
            </body>
            <body name="obstacle" pos="0.15 0 0">
              <geom name="obstacle_sphere" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(contact_model)
    mujoco.mj_forward(contact_model, data)
    assert data.ncon > 0
    events: dict[tuple[int, ...], ContactEvent] = {}
    record_penetrating_contacts(contact_model, data, events, 0.0)
    data.qpos[0] = 0.02  # Deeper overlap at the second sample.
    mujoco.mj_forward(contact_model, data)
    record_penetrating_contacts(contact_model, data, events, 0.1)
    if body_name == "other_arm":
        assert not events
        return
    (event,) = events.values()
    assert event["first_time_s"] == 0.0
    assert event["worst_time_s"] == 0.1
    assert event["minimum_separation_m"] == pytest.approx(-0.07)
    assert event["sample_count"] == 2
    assert set(event["geoms"]) == {"moving_sphere", "obstacle_sphere"}
    result = _result()
    result["contact_events"] = list(events.values())
    report = assess_trial(model, result)
    assert not report["passing"]
    assert report["reasons"][0]["kind"] == "penetrating_contact"

    data.qpos[0] = -1.0  # Separated geometries must not produce a violation.
    mujoco.mj_forward(contact_model, data)
    separated: dict[tuple[int, ...], ContactEvent] = {}
    record_penetrating_contacts(contact_model, data, separated, 0.2)
    assert not separated


def test_trial_assessment_includes_last_integration(
    model: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    position_indices, _ = resolve_joint_indices(model)
    qi = nominal_configuration(model)[position_indices]
    trajectory: JointTrajectory = {
        "t": np.array([0.0]),
        "q": qi[None, :],
        "dq": np.zeros((1, 7)),
        "ddq": np.zeros((1, 7)),
    }
    viewer = MagicMock()
    viewer.__enter__.return_value = viewer
    viewer.is_running.return_value = True
    monkeypatch.setattr(
        "me6404_openarm_shaping.simulation_helpers.mujoco.viewer.launch_passive",
        lambda *a: viewer,
    )
    monkeypatch.setattr(
        "me6404_openarm_shaping.simulation_helpers.time.sleep", lambda *a: None
    )

    def integrate(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        joint = model.joint("openarm_left_joint1")
        data.qpos[joint.qposadr[0]] = joint.range[1] + 0.1
        data.time += model.opt.timestep

    monkeypatch.setattr(
        "me6404_openarm_shaping.simulation_helpers.mujoco.mj_step", integrate
    )
    result = run_trial(
        model,
        trajectory,
        qi,
        np.zeros(7),
        np.zeros(7),
        model.opt.timestep,
        config=_config(),
    )
    assert result["completed"]
    np.testing.assert_allclose(result["q"][0], qi, rtol=0, atol=1e-14)
    assert not result["simulation_checks"]["passing"]
    reason = next(
        item
        for item in result["simulation_checks"]["reasons"]
        if item["kind"] == "position_limit"
    )
    assert reason["first_time_s"] == pytest.approx(model.opt.timestep)
