import numpy as np

from me6404_openarm_shaping.configuration import load_robot_config
from me6404_openarm_shaping.mode_tracking import match_mode_shapes, track_modes
from me6404_openarm_shaping.robot_model import load_model, modal_properties


def test_mac_preserves_branch_ids_through_frequency_crossing() -> None:
    shapes = np.eye(2, dtype=np.complex128)
    poles, mac, gap = match_mode_shapes(
        [
            (np.array([-1 + 2j, -1 + 4j]), shapes),
            (np.array([-1 + 2j, -1 + 4j]), shapes[:, ::-1] * [2j, -3]),
        ]
    )
    np.testing.assert_array_equal(poles, [[-1 + 2j, -1 + 4j], [-1 + 4j, -1 + 2j]])
    np.testing.assert_allclose(mac[1], 1)
    np.testing.assert_allclose(gap[1], 2)


def test_missing_mode_leaves_gap_and_can_reappear() -> None:
    shapes = np.eye(2, dtype=np.complex128)
    two = (np.array([-1 + 2j, -1 + 4j]), shapes)
    one = (two[0][1:], shapes[:, 1:])
    poles, mac, _ = match_mode_shapes([one, two, one, two])
    np.testing.assert_array_equal(poles[:, 0], -1 + 4j)
    np.testing.assert_allclose(poles[:, 1], [np.nan, -1 + 2j, np.nan, -1 + 2j])
    np.testing.assert_allclose(mac[3], 1)


def test_tracking_preserves_openarm_modal_values() -> None:
    config = load_robot_config()
    model = load_model(config.passive_damping)
    tracked = track_modes(model, config.waypoints, config.motor_kp, config.motor_kd)
    for i, (q, poles) in enumerate(zip(config.waypoints, tracked.poles, strict=True)):
        original = modal_properties(model, q, config.motor_kp, config.motor_kd)
        present = poles[np.isfinite(poles)]
        order = np.argsort(np.abs(present))
        np.testing.assert_allclose(present[order], original["eigenvalues"])
        assert tracked.stable[i] == original["stable"]
