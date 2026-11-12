"""Inertia checks and gravity verification using the pinned OpenArm model."""

import mujoco
import numpy as np
import pytest

from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    eval_gravity_torques,
    gravitational_stiffness,
    load_model,
    mass_matrix,
    nominal_configuration,
    resolve_joint_indices,
    state_matrix,
)


@pytest.fixture(scope="module")
def matrix():
    model = load_model()
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)
    # Verify the test posture against the loaded MJCF before evaluating it.
    for name, angle in zip(LEFT_ARM_JOINT_NAMES, q, strict=True):
        joint = model.joint(name)
        lower, upper = joint.range
        assert joint.limited[0]
        assert lower <= angle <= upper, (name, angle, lower, upper)
    return mass_matrix(model, q)


def test_matrix_shape(matrix):
    assert matrix.shape == (7, 7)


def test_matrix_symmetry(matrix):
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12)


def test_matrix_positive_definiteness(matrix):
    np.linalg.cholesky(matrix)


def _gravity_torques_from_potential_energy(model, q):
    """Approximate dV/dq in N*m using central differences of energy in joules.

    Fingers stay fixed, so their elastic potential cancels in the difference.
    The caller enables energy computation on its test-local model.
    """
    position_indices, _ = resolve_joint_indices(model)
    epsilon = 0.001  # rad
    data = mujoco.MjData(model)
    data.qpos[:] = nominal_configuration(model)
    dV_dq = []

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


def test_gravity_torques_match_potential_gradient():
    model = load_model()
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)
    np.testing.assert_allclose(
        eval_gravity_torques(model, q),
        _gravity_torques_from_potential_energy(model, q),
        rtol=1e-5,
        atol=1e-6,
    )


def test_gravitational_stiffness_finite_difference_convergence():
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


def test_state_matrix_residual():
    """Check assembly for state ordering [delta q; delta qdot]."""
    model = load_model()
    q = np.array([0, -0.3, 0, 0.012, 0, 0, 0], dtype=np.float64)

    M = mass_matrix(model, q)
    Kg = gravitational_stiffness(model, q)
    Kp = np.diag([70, 70, 70, 60, 10, 10, 10])
    Kd = np.diag([2.75, 2.50, 2.00, 2.00, 0.70, 0.60, 0.50])
    Keff = Kp + Kg
    Deff = Kd + 0.4 * np.eye(7)

    A = state_matrix(model, q)

    assert A.shape == (14, 14)
    np.testing.assert_allclose(A[:7, :7], 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(A[:7, 7:], np.eye(7), rtol=0, atol=1e-10)
    np.testing.assert_allclose(M @ A[7:, :7] + Keff, 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(M @ A[7:, 7:] + Deff, 0, rtol=0, atol=1e-10)
