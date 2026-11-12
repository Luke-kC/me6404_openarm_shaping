"""Minimal command-line report for the pinned OpenArm left-arm inertia."""

import argparse

import numpy as np

from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    MODEL_PATH,
    MODEL_REVISION,
    eval_gravity_torques,
    gravitational_stiffness,
    load_model,
    mass_matrix,
    modal_report,
    resolve_joint_indices,
    state_matrix,
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

        mass_inertia_matrix = mass_matrix(model, args.q)

        np.linalg.cholesky(mass_inertia_matrix)

        gravity_torques = eval_gravity_torques(model, args.q)

        grav_stiffness_matrix = gravitational_stiffness(model, args.q)

        A = state_matrix(model, args.q)

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
    print(f"Mass matrix shape: {mass_inertia_matrix.shape}; units: kg m^2")
    print(np.array2string(mass_inertia_matrix, precision=10))
    print(
        f"Symmetry error (max absolute): {np.max(np.abs(mass_inertia_matrix - mass_inertia_matrix.T)):.10e}"
    )
    print("Positive definite: Cholesky succeeded")
    print(f"Minimum eigenvalue: {np.linalg.eigvalsh(mass_inertia_matrix)[0]:.10e}")

    print("\n----------GRAVITY TORQUES----------")
    print("Gravity torques [N*m]:", gravity_torques)

    print("\n----------GRAVITATIONAL STIFFNESS MATRIX----------")
    print(f"Shape: {grav_stiffness_matrix.shape}; units: N*m/rad")
    print(np.array2string(grav_stiffness_matrix, precision=10))
    print(
        f"Symmetry error (max absolute): {np.max(np.abs(grav_stiffness_matrix - grav_stiffness_matrix.T)):.10e}"
    )

    print("\n----------STATE MATRIX----------")
    print(f"Shape: {A.shape}")
    print(np.array2string(A, precision=10))

    print("\n----------MODAL ANALYSIS----------")
    modal_report(A)


if __name__ == "__main__":
    main()
