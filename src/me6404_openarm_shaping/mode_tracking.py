"""Offline tracking of frozen-configuration oscillatory modes."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from me6404_openarm_shaping.data_types import ComplexVector, FloatMatrix, FloatVector
from me6404_openarm_shaping.robot_model import (
    ModalWorkspace,
    check_joint_configuration,
    state_matrix,
)


@dataclass
class TrackedModes:
    """Rows are samples; columns are branch IDs. Absent poles are NaN."""

    poles: NDArray[np.complex128]
    mac: FloatMatrix
    assignment_gap: FloatVector
    stable: NDArray[np.bool_]

    @property
    def natural_frequency_hz(self) -> FloatMatrix:
        return np.abs(self.poles) / (2 * np.pi)

    @property
    def damping_ratio(self) -> FloatMatrix:
        return -self.poles.real / np.abs(self.poles)


def match_mode_shapes(
    samples: Sequence[tuple[ComplexVector, NDArray[np.complex128]]],
) -> tuple[NDArray[np.complex128], FloatMatrix, FloatVector]:
    """Match by complex MAC, maximizing total overlap over distinct assignments.

    Shapes are columns of displacement eigenvectors, paired with the poles.
    New branches start in frequency order. An absent branch keeps its last
    shape for a possible reappearance, but its output has a gap.
    """
    n_branches = max((len(poles) for poles, _ in samples), default=0)
    tracked: NDArray[np.complex128] = np.full(
        (len(samples), n_branches), complex(np.nan, np.nan)
    )
    quality: FloatMatrix = np.full(tracked.shape, np.nan)
    gaps: FloatVector = np.full(len(samples), np.nan)
    previous: NDArray[np.complex128] | None = None
    assignments: dict[tuple[int, int], NDArray[np.intp]] = {}

    for i, (poles, shapes) in enumerate(samples):
        if poles.size == 0:
            continue
        order = np.argsort(np.abs(poles))
        poles, shapes = poles[order], shapes[:, order]
        shapes = shapes / np.linalg.norm(shapes, axis=0)
        count = len(poles)
        if previous is None:
            tracked[i, :count] = poles
            previous = shapes.copy()
            continue

        old_count = int(previous.shape[1])
        n_match, n_choices = sorted((old_count, count))
        key = (n_match, n_choices)
        if key not in assignments:
            assignments[key] = np.array(
                list(permutations(range(n_choices), n_match)), dtype=np.intp
            )
        possible = assignments[key]
        mac = np.abs(previous.conj().T @ shapes) ** 2
        indices = np.arange(n_match)
        overlaps = mac[possible, indices] if old_count >= count else mac[indices, possible]
        scores = overlaps.sum(axis=1)
        best = int(scores.argmax())
        if scores.size > 1:
            gaps[i] = scores[best] - np.partition(scores, -2)[-2]

        if old_count >= count:
            branch_ids, mode_indices = possible[best], indices
        else:
            branch_ids, mode_indices = indices, possible[best]
        tracked[i, branch_ids] = poles[mode_indices]
        quality[i, branch_ids] = overlaps[best]
        previous[:, branch_ids] = shapes[:, mode_indices]

        if count > old_count:
            new_modes = np.setdiff1d(np.arange(count), mode_indices)
            tracked[i, old_count:count] = poles[new_modes]
            previous = np.column_stack((previous, shapes[:, new_modes]))

    return tracked, quality, gaps


def track_modes(
    model: mujoco.MjModel,
    configurations: ArrayLike,
    motor_kp: ArrayLike,
    motor_kd: ArrayLike,
    tol: float = 1e-10,
) -> TrackedModes:
    """Track the same oscillatory pairs as ``modal_properties`` along a path."""
    workspace = ModalWorkspace(model)
    samples: list[tuple[ComplexVector, NDArray[np.complex128]]] = []
    stability: list[bool] = []
    path: FloatMatrix = np.asarray(configurations, dtype=np.float64)
    for q in path:
        check_joint_configuration(model, q)
        poles, vectors = np.linalg.eig(
            state_matrix(model, q, motor_kp, motor_kd, workspace=workspace)
        )
        oscillatory = poles.imag > tol
        # The state is [joint displacement, joint velocity].
        samples.append((poles[oscillatory], vectors[: len(q), oscillatory]))
        stability.append(bool(np.all(poles.real < 0)))
    poles, mac, gap = match_mode_shapes(samples)
    return TrackedModes(poles, mac, gap, np.array(stability, dtype=bool))
