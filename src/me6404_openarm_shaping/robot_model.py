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


def _nominal_configuration(model: mujoco.MjModel) -> NDArray[np.float64]:

    nominal = np.array(model.qpos0, dtype=np.float64, copy=True)

    # Set the fingers to midpoints
    for name in FINGER_JOINT_NAMES:
        joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        lower, upper = model.jnt_range[joint_id]

        nominal[model.jnt_qposadr[joint_id]] = (lower + upper) / 2

    return nominal


def mass_matrix(model: mujoco.MjModel, q: ArrayLike) -> NDArray[np.float64]:
    position_indices, velocity_indices = resolve_joint_indices(model)

    data = mujoco.MjData(model)

    data.qpos[:] = _nominal_configuration(model)
    data.qpos[position_indices] = q

    mujoco.mj_forward(model, data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)

    return M[np.ix_(velocity_indices, velocity_indices)]


def eval_gravity_torques(model: mujoco.MjModel, q: ArrayLike):

    position_indices, velocity_indices = resolve_joint_indices(model)

    data = mujoco.MjData(model)

    data.qpos[:] = _nominal_configuration(model)
    data.qpos[position_indices] = q

    data.qvel[: model.nv] = 0.0
    data.qacc[: model.nv] = 0.0

    mujoco.mj_forward(model, data)

    full_gravity_torques = data.qfrc_bias[: model.nv]

    return full_gravity_torques[velocity_indices]


def gravity_torques_from_potential_energy(model: mujoco.MjModel, q: ArrayLike):
    position_indices, velocity_indices = resolve_joint_indices(model)

    epsilon = 0.01

    data = mujoco.MjData(model)

    data.qvel[: model.nv] = 0.0
    data.qacc[: model.nv] = 0.0

    data.qpos[:] = _nominal_configuration(model)

    dV_dq = []

    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    for i in position_indices:
        data.qpos[position_indices] = q
        data.qpos[i] += epsilon

        mujoco.mj_forward(model, data)

        V2 = data.energy[0]

        data.qpos[position_indices] = q
        data.qpos[i] -= epsilon
        mujoco.mj_forward(model, data)

        V1 = data.energy[0]

        dV_dqi = (V2 - V1) / (2 * epsilon)

        dV_dq.append(dV_dqi)

    return dV_dq
