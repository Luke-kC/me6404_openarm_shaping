from pathlib import Path

import numpy as np
import yaml

from me6404_openarm_shaping.batch_sim import DEFAULT_BATCH_CONFIG, load_batch
from me6404_openarm_shaping.configuration import DEFAULT_CONFIG_PATH, load_robot_config


def test_batch_shaping_and_waypoints_replace_single_run_settings(
    tmp_path: Path,
) -> None:
    batch = yaml.safe_load(DEFAULT_BATCH_CONFIG.read_text())
    batch["input_shaping"]["time_invariant"]["ei"].update(
        enabled=True, residual_tolerance=0.03
    )
    robot = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    # Deliberately change both single-run options; only the batch block counts.
    robot["input_shaping"]["time_invariant"]["zv"]["enabled"] = True
    robot["input_shaping"]["time_varying"]["zvd"]["enabled"] = False
    robot_path = tmp_path / "robot.yaml"
    robot_path.write_text(yaml.safe_dump(robot))
    batch["robot_config"] = robot_path.name
    batch["trajectory_suite"] = str(
        DEFAULT_BATCH_CONFIG.parent / batch["trajectory_suite"]
    )
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(yaml.safe_dump(batch))

    _, suite, jobs = load_batch(config_path)
    assert len(jobs) == len(suite["trajectories"]) > 0
    single = load_robot_config(robot_path)
    assert single.input_shaping.time_invariant.zv.enabled
    assert not single.input_shaping.time_varying.zvd.enabled
    assert not single.input_shaping.time_invariant.ei.enabled
    for entry, config in jobs:
        assert not config.input_shaping.time_invariant.zv.enabled
        assert config.input_shaping.time_varying.zvd.enabled
        assert config.input_shaping.time_invariant.ei.enabled
        assert config.input_shaping.time_invariant.ei.residual_tolerance == 0.03
        np.testing.assert_array_equal(config.waypoints, entry["waypoints"])
        np.testing.assert_array_equal(config.motor_kp, single.motor_kp)
