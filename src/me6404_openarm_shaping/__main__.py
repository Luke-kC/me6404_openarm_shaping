"""Minimal command-line report for the pinned OpenArm left-arm inertia."""

import argparse

import numpy as np

from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    MODEL_PATH,
    MODEL_REVISION,
    eval_gravity_torques,
    gravity_torques_from_potential_energy,
    load_model,
    mass_matrix,
    resolve_joint_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q",
        type=float,
        nargs=7,
        required=True,
        metavar="RAD",
        help="left-arm joint positions J1 through J7 in radians",
    )
    args = parser.parse_args()

    try:
        model = load_model()
        position_indices, velocity_indices = resolve_joint_indices(model)
        matrix = mass_matrix(model, args.q)
        np.linalg.cholesky(matrix)
        gravity_torques = eval_gravity_torques(model, args.q)
        gravity_torques_from_pot = gravity_torques_from_potential_energy(model, args.q)

    except (ValueError, OSError, np.linalg.LinAlgError) as error:
        parser.error(str(error))

    print("----------INFO----------")
    print(f"Model: {MODEL_PATH}")
    print(f"Pinned upstream revision: {MODEL_REVISION}")
    print(f"Selected joints: {', '.join(LEFT_ARM_JOINT_NAMES)}")
    print(f"Position indices: {position_indices.tolist()}")
    print(f"Velocity indices: {velocity_indices.tolist()}")
    print("Nominal: arms at qpos0 (zero rad); all four fingers at 0.022 m.")

    print("\n----------MASS MATRIX----------")
    print(f"Mass matrix shape: {matrix.shape}; units: kg m^2")
    print(np.array2string(matrix, precision=10))
    print(f"Symmetry error (max absolute): {np.max(np.abs(matrix - matrix.T)):.10e}")
    print("Positive definite: Cholesky succeeded")
    print(f"Minimum eigenvalue: {np.linalg.eigvalsh(matrix)[0]:.10e}")

    print("\n----------GRAVITY TORQUES----------")
    print("Gravity torques [N*m]:", gravity_torques)
    print("Gravity torques from potential [N*m]:", gravity_torques_from_pot)


if __name__ == "__main__":
    main()
