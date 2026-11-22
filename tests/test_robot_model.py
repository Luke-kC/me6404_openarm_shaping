"""Inertia checks and gravity verification using the pinned OpenArm model."""

import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from me6404_openarm_shaping.configuration import load_robot_config
from me6404_openarm_shaping.robot_model import (
    FINGER_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    ModalWorkspace,
    eval_gravity_torques,
    gravitational_stiffness,
    load_model,
    mass_matrix,
    modal_properties,
    nominal_configuration,
    resolve_joint_indices,
    set_added_armature,
    state_matrix,
)


def test_calibrated_armature_is_additive_and_idempotent() -> None:
    config = load_robot_config()
    nominal = load_model()
    model = load_model(config.passive_damping, added_armature=config.added_armature)
    _, dofs = resolve_joint_indices(model)
    expected = nominal.dof_armature.copy()
    expected[dofs] += config.added_armature
    np.testing.assert_array_equal(model.dof_damping[dofs], config.passive_damping)
    for _ in range(2):
        set_added_armature(model, config.added_armature)
        np.testing.assert_array_equal(model.dof_armature, expected)
    q = config.waypoints[0]
    np.testing.assert_allclose(
        mass_matrix(model, q) - mass_matrix(nominal, q),
        np.diag(config.added_armature),
        atol=1e-12,
    )
    set_added_armature(model, np.zeros(7))
    np.testing.assert_array_equal(model.dof_armature, nominal.dof_armature)


def test_reused_modal_workspace_matches_fresh_state_matrices() -> None:
    model = load_model()
    config = load_robot_config()
    workspace = ModalWorkspace(model)
    for q in (*config.waypoints, *reversed(config.waypoints)):
        fresh = state_matrix(model, q, config.motor_kp, config.motor_kd)
        reused = state_matrix(
            model, q, config.motor_kp, config.motor_kd, workspace=workspace
        )
        np.testing.assert_array_equal(reused, fresh)


@pytest.fixture(scope="module")
def matrix() -> NDArray[np.float64]:
    model = load_model()
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)
    # Verify the test posture against the loaded MJCF before evaluating it.
    for name, angle in zip(LEFT_ARM_JOINT_NAMES, q, strict=True):
        joint = model.joint(name)
        lower, upper = joint.range
        assert joint.limited[0]
        assert lower <= angle <= upper, (name, angle, lower, upper)
    return mass_matrix(model, q)


def test_matrix_shape(matrix: NDArray[np.float64]) -> None:
    assert matrix.shape == (7, 7)


def test_matrix_symmetry(matrix: NDArray[np.float64]) -> None:
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12)


def test_matrix_positive_definiteness(matrix: NDArray[np.float64]) -> None:
    np.linalg.cholesky(matrix)


def test_nominal_configuration_matches_model_defaults() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    reference = model.qpos0.copy()
    nominal = nominal_configuration(model)
    assert nominal.dtype == np.float64
    np.testing.assert_allclose(nominal, data.qpos, rtol=0, atol=1e-14)
    for name in FINGER_JOINT_NAMES:
        joint = model.joint(name)
        position = nominal[joint.qposadr[0]]
        assert position == pytest.approx(0.0, rel=0, abs=1e-14)
        assert joint.range[0] <= position <= joint.range[1]
    nominal[:] = 1.0
    np.testing.assert_array_equal(model.qpos0, reference)
    np.testing.assert_allclose(data.qpos, reference, rtol=0, atol=1e-14)


def _gravity_torques_from_potential_energy(
    model: mujoco.MjModel, q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Approximate dV/dq in N*m using central differences of energy in joules.

    Fingers stay fixed, so their elastic potential cancels in the difference.
    The caller enables energy computation on its test-local model.
    """
    position_indices, _ = resolve_joint_indices(model)
    epsilon = 0.001  # rad
    data = mujoco.MjData(model)
    data.qpos[:] = nominal_configuration(model)
    dV_dq: list[float] = []

    for i in position_indices:
        data.qpos[position_indices] = q
        data.qpos[i] += epsilon
        mujoco.mj_forward(model, data)
        V2 = data.energy[0]

        data.qpos[position_indices] = q
        data.qpos[i] -= epsilon
        mujoco.mj_forward(model, data)
        V1 = data.energy[0]

        dV_dq.append((V2 - V1) / (2 * epsilon))

    return np.asarray(dV_dq)


def test_gravity_torques_match_potential_gradient() -> None:
    model = load_model()
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)
    np.testing.assert_allclose(
        eval_gravity_torques(model, q),
        _gravity_torques_from_potential_energy(model, q),
        rtol=1e-5,
        atol=1e-6,
    )


def test_gravitational_stiffness_finite_difference_convergence() -> None:
    model = load_model()
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)
    steps = (0.001, 0.0005, 0.00025)  # rad

    # Every stencil position must remain inside the loaded MJCF joint limits.
    for step in steps:
        for name, angle in zip(LEFT_ARM_JOINT_NAMES, q, strict=True):
            joint = model.joint(name)
            lower, upper = joint.range
            assert joint.limited[0]
            assert lower <= angle - step <= angle + step <= upper

    coarse, medium, fine = (
        gravitational_stiffness(model, q, epsilon=step) for step in steps
    )
    coarse_difference = np.linalg.norm(coarse - medium, ord="fro")
    fine_difference = np.linalg.norm(medium - fine, ord="fro")
    # Halving h should reduce the O(h^2) central-difference error by about four.
    assert fine_difference > 0
    ratio = coarse_difference / fine_difference
    assert 3.5 <= ratio <= 4.5, f"Observed convergence ratio: {ratio}"


@pytest.mark.parametrize("gain_scale", [1.0, 1.2])
def test_state_matrix_residual(gain_scale: float) -> None:
    """Check assembly for state ordering [delta q; delta qdot]."""
    passive = np.array([0.02, 0.12, 0.4, 0.4, 0.4, 0.4, 0.4])
    model = load_model(passive)
    _, dofs = resolve_joint_indices(model)
    np.testing.assert_array_equal(model.dof_damping[dofs], passive)
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)

    M = mass_matrix(model, q)
    Kg = gravitational_stiffness(model, q)
    config = load_robot_config()
    motor_kp, motor_kd = config.motor_kp, config.motor_kd
    motor_kp = motor_kp * gain_scale
    motor_kd = motor_kd * gain_scale
    Kp = np.diag(motor_kp)
    Kd = np.diag(motor_kd)
    Keff = Kp + Kg
    Deff = Kd + np.diag(passive)

    A = state_matrix(model, q, motor_kp, motor_kd)

    assert A.shape == (14, 14)
    np.testing.assert_allclose(A[:7, :7], 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(A[:7, 7:], np.eye(7), rtol=0, atol=1e-10)
    np.testing.assert_allclose(M @ A[7:, :7] + Keff, 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(M @ A[7:, 7:] + Deff, 0, rtol=0, atol=1e-10)


def test_modal_properties() -> None:
    model = load_model()
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)
    config = load_robot_config()
    motor_kp, motor_kd = config.motor_kp, config.motor_kd
    result = modal_properties(model, q, motor_kp, motor_kd)
    poles, _ = np.linalg.eig(state_matrix(model, q, motor_kp, motor_kd))
    selected = poles[poles.imag > 1e-10]
    selected = selected[np.argsort(np.abs(selected))]

    np.testing.assert_allclose(result["eigenvalues"], selected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        result["natural_frequency_hz"],
        np.abs(selected) / (2 * np.pi),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["damped_frequency_hz"],
        selected.imag / (2 * np.pi),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["damping_ratio"],
        -selected.real / np.abs(selected),
        rtol=1e-10,
        atol=1e-12,
    )
    real_poles = np.sort(poles.real[np.abs(poles.imag) <= 1e-10])
    np.testing.assert_allclose(
        result["real_eigenvalues"], real_poles, rtol=1e-10, atol=1e-10
    )
    assert 2 * len(selected) + len(real_poles) == 14
    assert result["stable"] == bool(np.all(poles.real < 0))
    assert np.all(np.diff(result["natural_frequency_hz"]) >= 0)
