"""Minimal inertia checks using the pinned OpenArm model."""

import numpy as np
import pytest

from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    load_model,
    mass_matrix,
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
