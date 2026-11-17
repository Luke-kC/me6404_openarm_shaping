"""Small checks for the shared YAML configuration."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from me6404_openarm_shaping.configuration import (
    DEFAULT_CONFIG_PATH,
    InputShapingConfig,
    RobotConfig,
    SimulationAssessmentConfig,
    SimulationConfig,
    ZVDConfig,
    experiment_fingerprint,
    load_robot_config,
    robot_config_to_dict,
)


def test_load_robot_config(tmp_path: Path) -> None:
    path = tmp_path / "robot.yaml"
    path.write_text(
        "motor_kp: [1, 2, 3, 4, 5, 6, 7]\n"
        "motor_kd: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]\n"
        "velocity_limits: [1, 2, 3, 4, 5, 6, 7]\n"
        "acceleration_limits: [2, 4, 6, 8, 10, 12, 14]\n"
        "waypoints: [[0, 0, 0, 0, 0, 0, 0], [0.1, -0.2, 0, 0.3, 0, 0, 0]]\n"
        "simulation: {dt_s: 0.004, minimum_duration_s: 3, settling_time_s: 0.5}\n"
        "simulation_assessment: {settling_window_s: 0.25, position_tolerance_rad: 0.01, velocity_tolerance_rad_s: 0.01, measured_velocity_tolerance_rad_s: 0.05, j4_lower_position_tolerance_rad: 0.05}\n"
        "input_shaping:\n"
        "  enabled: false\n"
        "  type: zvd\n"
        "  zvd: {tau_min_s: 0.2, utilization_min: 0.4}\n"
    )
    config = load_robot_config(path)
    assert isinstance(config, RobotConfig)
    kp, kd = config.motor_kp, config.motor_kd
    for values in (kp, kd, config.velocity_limits, config.acceleration_limits, config.waypoints):
        assert values.dtype == np.float64
    np.testing.assert_allclose(kp, [1, 2, 3, 4, 5, 6, 7], rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        kd, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        config.velocity_limits, [1, 2, 3, 4, 5, 6, 7], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        config.acceleration_limits, [2, 4, 6, 8, 10, 12, 14], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        config.waypoints[1], [0.1, -0.2, 0, 0.3, 0, 0, 0], rtol=0, atol=1e-14
    )
    assert config.simulation == SimulationConfig(
        dt_s=0.004, minimum_duration_s=3.0, settling_time_s=0.5
    )
    assert config.input_shaping == InputShapingConfig(
        enabled=False, type="zvd", zvd=ZVDConfig(tau_min_s=0.2, utilization_min=0.4)
    )
    snapshot = robot_config_to_dict(config)
    assert config.simulation_assessment == SimulationAssessmentConfig(0.25, 0.01, 0.01, 0.05, 0.05)
    assert yaml.safe_load(yaml.safe_dump(snapshot)) == yaml.safe_load(path.read_text())
    cast(list[list[float]], snapshot["waypoints"])[0][0] = 99.0
    cast(dict[str, float], snapshot["simulation"])["dt_s"] = 1.0
    assert config.waypoints[0, 0] == 0.0
    assert config.simulation.dt_s == 0.004


def test_reject_incorrect_gain_count(tmp_path: Path) -> None:
    path = tmp_path / "gains.yaml"
    path.write_text("motor_kp: [1, 2]\nmotor_kd: [1, 2, 3, 4, 5, 6, 7]\n")
    with pytest.raises(ValueError, match="motor_kp"):
        load_robot_config(path)


@pytest.mark.parametrize("key", ["velocity_limits", "acceleration_limits"])
def test_limits_must_be_positive(tmp_path: Path, key: str) -> None:
    path = tmp_path / "robot.yaml"
    config = {
        "motor_kp": [0] * 7,
        "motor_kd": [0] * 7,
        "velocity_limits": [1] * 7,
        "acceleration_limits": [1] * 7,
    }
    config[key][0] = 0
    path.write_text("\n".join(f"{name}: {values}" for name, values in config.items()))
    with pytest.raises(ValueError, match=key):
        _ = load_robot_config(path)


@pytest.mark.parametrize("case", ["zero-dt", "wrong-waypoint-shape", "unsupported-shaper"])
def test_reject_invalid_experiment_settings(tmp_path: Path, case: str) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    if case == "zero-dt":
        config["simulation"]["dt_s"] = 0
        match = "dt_s"
    elif case == "wrong-waypoint-shape":
        config["waypoints"] = [[0] * 6, [0] * 6]
        match = "waypoints"
    else:
        config["input_shaping"]["type"] = "other"
        match = "zvd"
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        load_robot_config(path)


def test_fingerprint_is_deterministic_and_independent_of_yaml_format(tmp_path: Path) -> None:
    snapshot = robot_config_to_dict(load_robot_config())
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    _ = first.write_text(yaml.safe_dump(snapshot, sort_keys=True))
    _ = second.write_text(
        "# Different comments, key order, path, and formatting.\n"
        + yaml.safe_dump(snapshot, sort_keys=False, default_flow_style=True)
    )
    fingerprint = experiment_fingerprint(load_robot_config(first), "1" * 64, "a" * 40)
    assert fingerprint == experiment_fingerprint(load_robot_config(first), "1" * 64, "a" * 40)
    assert fingerprint == experiment_fingerprint(load_robot_config(second), "1" * 64, "a" * 40)
    assert len(fingerprint) == 64
    assert all(character in "0123456789abcdef" for character in fingerprint)


@pytest.mark.parametrize(
    "change", [
        "gain", "waypoint", "timestep", "shaping-enabled", "model-hash",
         "model-revision", "settling-tolerance", "measured-velocity-tolerance",
         "j4-lower-position-tolerance",
    ]
)
def test_fingerprint_changes_with_configuration_or_model_identity(change: str) -> None:
    config = load_robot_config()
    model_hash, model_revision = "1" * 64, "a" * 40
    original = experiment_fingerprint(config, model_hash, model_revision)
    if change == "gain":
        config.motor_kp[0] += 1
    elif change == "waypoint":
        config.waypoints[1, 0] += 0.01
    elif change == "timestep":
        config.simulation.dt_s *= 2
    elif change == "shaping-enabled":
        config.input_shaping.enabled = not config.input_shaping.enabled
    elif change == "model-hash":
        model_hash = "2" * 64
    elif change == "settling-tolerance":
        config.simulation_assessment.position_tolerance_rad *= 2
    elif change == "measured-velocity-tolerance":
        config.simulation_assessment.measured_velocity_tolerance_rad_s += 0.01
    elif change == "j4-lower-position-tolerance":
        config.simulation_assessment.j4_lower_position_tolerance_rad += 0.01
    else:
        model_revision = "b" * 40
    assert experiment_fingerprint(config, model_hash, model_revision) != original


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("settling_window_s", 0.0), ("settling_window_s", -0.1),
        ("position_tolerance_rad", -0.01), ("velocity_tolerance_rad_s", -0.01),
        ("position_tolerance_rad", float("nan")),
        ("velocity_tolerance_rad_s", float("inf")),
        ("measured_velocity_tolerance_rad_s", -0.01),
        ("measured_velocity_tolerance_rad_s", float("nan")),
        ("j4_lower_position_tolerance_rad", -0.01),
        ("j4_lower_position_tolerance_rad", float("inf")),
    ],
)
def test_invalid_assessment_settings(tmp_path: Path, key: str, value: float) -> None:
    config = robot_config_to_dict(load_robot_config())
    cast(dict[str, float], config["simulation_assessment"])[key] = value
    path = tmp_path / "robot.yaml"
    _ = path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=key):
        _ = load_robot_config(path)
