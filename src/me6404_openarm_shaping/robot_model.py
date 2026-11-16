from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from me6404_openarm_shaping.data_types import FloatMatrix, FloatVector, ModalProperties

MODEL_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"
)


MODEL_REVISION = "a8c979629f2591ad035d99d338ce114969e6cddc"

LEFT_ARM_JOINT_NAMES = tuple(f"openarm_left_joint{i}" for i in range(1, 8))


FINGER_JOINT_NAMES = (
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
)


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def check_joint_configuration(model: mujoco.MjModel, q: ArrayLike) -> None:

    q = np.asarray(q)

    for name, qi in zip(LEFT_ARM_JOINT_NAMES, q, strict=True):
        lower, upper = model.joint(name).range

        if not np.isfinite(qi) or not lower <= qi <= upper:
            raise ValueError(
                f"{name}: q={qi} must be finite and within [{lower}, {upper}] rad"
            )


def resolve_joint_indices(
    model: mujoco.MjModel,
    joint_names: Sequence[str] = LEFT_ARM_JOINT_NAMES,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:

    joint_ids: list[int] = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids.append(joint_id)

    return (
        np.array(model.jnt_qposadr[joint_ids], dtype=np.intp),
        np.array(model.jnt_dofadr[joint_ids], dtype=np.intp),
    )


def nominal_configuration(model: mujoco.MjModel) -> FloatVector:
    """Return a copy of qpos0 with no gripper overrides.

    The pinned model defaults all four finger positions to zero metres.
    This matches fresh MjData initialization; it does not lock the fingers.
    """
    return np.array(model.qpos0, dtype=np.float64, copy=True)


def mass_matrix(model: mujoco.MjModel, q: ArrayLike) -> FloatMatrix:
    """
    Calculates the mass inertia matrix, M(q), at joint configuration q.
    For one 7-DoF OpenArm, this result is 7x7
    """
    position_indices, velocity_indices = resolve_joint_indices(model)

    data = mujoco.MjData(model)

    data.qpos[:] = nominal_configuration(model)
    data.qpos[position_indices] = q

    mujoco.mj_forward(model, data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)

    return M[np.ix_(velocity_indices, velocity_indices)]


def eval_gravity_torques(model: mujoco.MjModel, q: ArrayLike) -> FloatVector:
    """
    Calculates the constant gravity torques, g(q), at joint configuration q.
    For one 7-DoF OpenArm, this result is 7x1
    """

    position_indices, velocity_indices = resolve_joint_indices(model)

    data = mujoco.MjData(model)

    data.qpos[:] = nominal_configuration(model)
    data.qpos[position_indices] = q

    data.qvel[: model.nv] = 0.0
    data.qacc[: model.nv] = 0.0

    mujoco.mj_forward(model, data)

    full_gravity_torques = data.qfrc_bias[: model.nv]

    return full_gravity_torques[velocity_indices]


def gravitational_stiffness(
    model: mujoco.MjModel, q: ArrayLike, epsilon: float = 0.001
) -> FloatMatrix:
    """
    Calculates the gravitational stiffness matrix, Kg(q), at joint configuraion q.
    Kg(q) = dg(q)/dq
    For one 7-DoF OpenArm, this result is 7x7
    epsilon is the central-difference step in radians.
    """
    position_indices, _ = resolve_joint_indices(model)

    n = len(position_indices)
    Kg = np.zeros((n, n))

    for i in range(n):
        q_plus = np.array(q, dtype=np.float64)
        q_minus = np.array(q, dtype=np.float64)

        q_plus[i] += epsilon
        q_minus[i] -= epsilon

        g_plus = np.asarray(eval_gravity_torques(model, q_plus))
        g_minus = np.asarray(eval_gravity_torques(model, q_minus))

        Kg[:, i] = (g_plus - g_minus) / (2 * epsilon)

    return Kg


def state_matrix(
    model: mujoco.MjModel,
    q: ArrayLike,
    motor_kp: ArrayLike,
    motor_kd: ArrayLike,
) -> FloatMatrix:
    position_indices, _ = resolve_joint_indices(model)

    n = len(position_indices)

    A1 = np.zeros((n, n))
    A2 = np.identity(n)

    M = mass_matrix(model, q)

    Kg = gravitational_stiffness(model, q)

    Keff = np.diag(motor_kp) + Kg

    passive_damping = 0.4

    Deff = np.diag(motor_kd) + passive_damping * np.identity(n)

    A3 = -np.linalg.solve(M, Keff)
    A4 = -np.linalg.solve(M, Deff)

    A = np.block([[A1, A2], [A3, A4]])

    return A


def modal_properties(
    model: mujoco.MjModel,
    q: ArrayLike,
    motor_kp: ArrayLike,
    motor_kd: ArrayLike,
    tol: float = 1e-10,
) -> ModalProperties:
    """Return frozen-q, zero-velocity properties of the closed-loop state matrix.

    Oscillatory arrays contain one pole per conjugate pair, sorted by natural
    frequency (not tracked physical mode identity).
    Frequencies are Hz and damping ratios dimensionless.
    Real poles are returned separately in 1/s.
    The oscillatory mode count may change with q.
    """
    check_joint_configuration(model, q)

    poles = np.linalg.eigvals(state_matrix(model, q, motor_kp, motor_kd))

    oscillatory = poles[np.imag(poles) > tol]

    oscillatory = oscillatory[np.argsort(np.abs(oscillatory))]

    omega_n = np.abs(oscillatory)

    return {
        "eigenvalues": oscillatory,
        "natural_frequency_hz": omega_n / (2 * np.pi),
        "damped_frequency_hz": np.imag(oscillatory) / (2 * np.pi),
        "damping_ratio": -np.real(oscillatory) / omega_n,
        "real_eigenvalues": np.sort(np.real(poles[np.abs(np.imag(poles)) <= tol])),
        "stable": bool(np.all(np.real(poles) < 0.0)),
    }


def modal_report(A: FloatMatrix, tol: float = 1e-10) -> None:

    eigenvalues, eigenvectors = np.linalg.eig(A)

    if bool(np.all(np.real(eigenvalues) < 0.0)):
        print("All eigenvalues have negative real part")
    else:
        print("One or more eigenvalues have >= 0.0 real part")

    complex_count = np.sum(np.abs(np.imag(eigenvalues)) > tol)
    complex_count = int(complex_count // 2)

    real_count = len(eigenvalues) - 2 * complex_count

    print("Number of complex-conjugate pairs: ", complex_count)
    print("Number of purely real eigenvalues: ", real_count)

    mode = 1
    for i, lam in enumerate(eigenvalues):
        if np.imag(lam) > tol:
            sigma = np.real(lam)
            omega_d = np.imag(lam)
            omega_n = np.abs(lam)
            zeta = -sigma / omega_n
            damped_freq_hz = omega_d / (2 * np.pi)
            natural_freq_hz = omega_n / (2 * np.pi)
            eigenvector = eigenvectors[:, i]

            joint_space_mode_shape = eigenvector[:7]

            # Normalize displacement magnitudes separately from velocity entries
            relative_amplitudes = np.abs(joint_space_mode_shape)
            relative_amplitudes /= np.max(relative_amplitudes)

            print(f"Mode {mode}:")
            print(f"    Eigenvalue:               {lam:.2f}")
            print(f"    Damped Frequency (Hz):    {damped_freq_hz:.3f} Hz")
            print(f"    Natural Frequency (Hz):   {natural_freq_hz:.3f} Hz")
            print(f"    Damping Ratio:            {zeta:.3f}")
            print("    Relative joint amplitudes (largest = 1):")
            for name, amplitude in zip(
                LEFT_ARM_JOINT_NAMES, relative_amplitudes, strict=True
            ):
                print(f"        {name}: {amplitude:.3f}")

            mode += 1
