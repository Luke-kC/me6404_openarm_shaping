"""Load joint gains and motion limits for offline analysis and simulation."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import numpy as np
import yaml
from numpy.typing import ArrayLike

from me6404_openarm_shaping.data_types import FloatMatrix, FloatVector

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/robot_config.yaml"


@dataclass
class SimulationConfig:
    dt_s: float
    minimum_duration_s: float
    settling_time_s: float


@dataclass
class SimulationAssessmentConfig:
    settling_window_s: float
    position_tolerance_rad: float
    velocity_tolerance_rad_s: float


@dataclass
class ZVDConfig:
    tau_min_s: float
    utilization_min: float


@dataclass
class InputShapingConfig:
    enabled: bool
    type: Literal["zvd"]
    zvd: ZVDConfig


@dataclass
class RobotConfig:
    motor_kp: FloatVector
    motor_kd: FloatVector
    velocity_limits: FloatVector
    acceleration_limits: FloatVector
    waypoints: FloatMatrix
    simulation: SimulationConfig
    input_shaping: InputShapingConfig
    simulation_assessment: SimulationAssessmentConfig


def load_robot_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> RobotConfig:
    """Return joint arrays, complete waypoints, and simulation/shaper settings.

    Gains have units N*m/rad and N*m*s/rad; velocity and acceleration limits
    have units rad/s and rad/s^2. Limits are symmetric in both directions.
    Joint arrays and waypoint rows are float64 in left J1--J7 order.
    Load once at program startup; edits take effect on the next run.
    """
    with Path(path).open() as stream:
        config = cast(dict[str, object], yaml.safe_load(stream))

    arrays: dict[str, FloatVector] = {}
    for name in ("motor_kp", "motor_kd", "velocity_limits", "acceleration_limits"):
        values = np.asarray(cast(ArrayLike, config[name]), dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain seven finite values.")
        if name in ("motor_kp", "motor_kd"):
            if np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative.")
        elif np.any(values <= 0):
            raise ValueError(f"{name} must be strictly positive.")
        arrays[name] = values

    waypoints = np.asarray(cast(ArrayLike, config["waypoints"]), dtype=np.float64)
    if (
        waypoints.ndim != 2
        or waypoints.shape[1] != 7
        or len(waypoints) < 2
        or not np.isfinite(waypoints).all()
    ):
        raise ValueError("waypoints must contain at least two finite J1--J7 rows.")

    simulation = cast(dict[str, float], config["simulation"]).copy()
    for name in ("dt_s", "minimum_duration_s", "settling_time_s"):
        value = float(simulation[name])
        if not np.isfinite(value) or value < 0 or (name == "dt_s" and value == 0):
            raise ValueError(f"Invalid simulation setting: {name}")
        simulation[name] = value

    shaping = cast(dict[str, object], config["input_shaping"]).copy()
    if not isinstance(shaping["enabled"], bool):
        raise TypeError("input_shaping.enabled must be a YAML boolean.")
    if shaping["type"] != "zvd":
        raise ValueError("Only input_shaping.type: zvd is currently supported.")
    zvd = cast(dict[str, float], shaping["zvd"]).copy()
    tau_min = float(zvd["tau_min_s"])
    utilization_min = float(zvd["utilization_min"])
    if not np.isfinite(tau_min) or tau_min < 0:
        raise ValueError("zvd.tau_min_s must be finite and nonnegative.")
    if not np.isfinite(utilization_min) or not 0 <= utilization_min <= 1:
        raise ValueError("zvd.utilization_min must be within [0, 1].")
    assessment = cast(dict[str, float], config["simulation_assessment"]).copy()
    for name in (
        "settling_window_s",
        "position_tolerance_rad",
        "velocity_tolerance_rad_s",
    ):
        value = float(assessment[name])
        if (
            not np.isfinite(value)
            or value < 0
            or (name == "settling_window_s" and value == 0)
        ):
            raise ValueError(f"Invalid simulation assessment setting: {name}")
        assessment[name] = value
    return RobotConfig(
        motor_kp=arrays["motor_kp"],
        motor_kd=arrays["motor_kd"],
        velocity_limits=arrays["velocity_limits"],
        acceleration_limits=arrays["acceleration_limits"],
        waypoints=waypoints,
        simulation=SimulationConfig(
            dt_s=simulation["dt_s"],
            minimum_duration_s=simulation["minimum_duration_s"],
            settling_time_s=simulation["settling_time_s"],
        ),
        input_shaping=InputShapingConfig(
            enabled=shaping["enabled"],
            type="zvd",
            zvd=ZVDConfig(tau_min_s=tau_min, utilization_min=utilization_min),
        ),
        simulation_assessment=SimulationAssessmentConfig(
            settling_window_s=assessment["settling_window_s"],
            position_tolerance_rad=assessment["position_tolerance_rad"],
            velocity_tolerance_rad_s=assessment["velocity_tolerance_rad_s"],
        ),
    )


def robot_config_to_dict(config: RobotConfig) -> dict[str, object]:
    """Return a detached, plain YAML-compatible snapshot with the same schema."""
    return {
        "motor_kp": config.motor_kp.tolist(),
        "motor_kd": config.motor_kd.tolist(),
        "velocity_limits": config.velocity_limits.tolist(),
        "acceleration_limits": config.acceleration_limits.tolist(),
        "waypoints": config.waypoints.tolist(),
        "simulation": asdict(config.simulation),
        "input_shaping": asdict(config.input_shaping),
        "simulation_assessment": asdict(config.simulation_assessment),
    }


def experiment_fingerprint(
    config: RobotConfig, model_sha256: str, model_revision: str
) -> str:
    """Identify validated configuration and model identity, independent of runs.

    YAML formatting, paths, timestamps, and output directories are excluded.
    This does not fingerprint project source, dependencies, or mesh contents.
    """
    payload = {
        "config": robot_config_to_dict(config),
        "model": {"sha256": model_sha256, "upstream_revision": model_revision},
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(canonical.encode("utf-8")).hexdigest()
