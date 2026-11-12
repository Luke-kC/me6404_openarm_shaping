from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

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


def resolve_joint_indices(
    model: mujoco.MjModel,
    joint_names: Sequence[str] = LEFT_ARM_JOINT_NAMES,
):

    joint_ids = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids.append(joint_id)

    return (
        np.array(model.jnt_qposadr[joint_ids], dtype=np.intp),
        np.array(model.jnt_dofadr[joint_ids], dtype=np.intp),
    )


def nominal_configuration(model: mujoco.MjModel) -> NDArray[np.float64]:

    nominal = np.array(model.qpos0, dtype=np.float64, copy=True)

    # Set the fingers to midpoints
    for name in FINGER_JOINT_NAMES:
        joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        lower, upper = model.jnt_range[joint_id]

        nominal[model.jnt_qposadr[joint_id]] = (lower + upper) / 2

    return nominal


def mass_matrix(model: mujoco.MjModel, q: ArrayLike) -> NDArray[np.float64]:
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


def eval_gravity_torques(model: mujoco.MjModel, q: ArrayLike):
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
):
    """
    Calculates the gravitational stiffness matrix, Kg(q), at joint configuraion q.
    Kg(q) = dg(q)/dq
    For one 7-DoF OpenArm, this result is 7x7
    epsilon is the central-difference step in radians.
    """
    position_indices, velocity_indices = resolve_joint_indices(model)

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
):
    position_indices, velocity_indices = resolve_joint_indices(model)

    n = len(position_indices)

    A1 = np.zeros((n, n))
    A2 = np.identity(n)

    M = mass_matrix(model, q)

    Kp = [70, 70, 70, 60, 10, 10, 10]
    Kg = gravitational_stiffness(model, q)

    Keff = np.diag(Kp) + Kg

    motor_Kd = [2.75, 2.5, 2.00, 2.00, 0.70, 0.60, 0.50]
    passive_damping = 0.4

    Deff = np.diag(motor_Kd) + passive_damping * np.identity(n)

    A3 = -np.linalg.solve(M, Keff)
    A4 = -np.linalg.solve(M, Deff)

    A = np.block([[A1, A2], [A3, A4]])

    return A


def modal_report(A: NDArray[np.float64], tol: float = 1e-10):

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
