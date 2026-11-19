from enum import StrEnum
from math import modf
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from me6404_openarm_shaping.data_types import (
    FloatMatrix,
    FloatVector,
    IntVector,
    ModalProperties,
)
from me6404_openarm_shaping.input_shaping import (
    convolve_shapers,
    zv_shaper,
    zvd_shaper,
)
from me6404_openarm_shaping.robot_model import LEFT_ARM_JOINT_NAMES


class TvistShaper(StrEnum):
    ZV = "ZV"
    ZVD = "ZVD"


def select_tvist_modes(
    properties: list[ModalProperties], tau_min_s: float
) -> list[ModalProperties]:
    """Keep stable oscillatory modes with decay time at least tau_min_s."""
    selected: list[ModalProperties] = []
    for sample in properties:
        damping = sample["damping_ratio"]
        decay_rate = 2 * np.pi * sample["natural_frequency_hz"] * damping
        keep = (damping > 0) & (decay_rate * tau_min_s <= 1)
        selected.append(
            {
                **sample,
                "natural_frequency_hz": sample["natural_frequency_hz"][keep],
                "damping_ratio": damping[keep],
                "damped_frequency_hz": sample["damped_frequency_hz"][keep],
                "eigenvalues": sample["eigenvalues"][keep],
            }
        )
    return selected


def sample_delayed(
    history: FloatVector,
    delay_samples: float,
    lagrange_order: Literal[1, 2, 3],
) -> float:
    """
    Uses Lagrange interpolation to estimate the value at the requested `delay_samples`
    from `history`.

    `history[0]` is latest/now, `history[1]` is one sample in the past, ...

    `lagrange_order` specifies the order of interpolation.
    1 requires two samples
    2 requires three samples
    3 requires four samples
    """

    D, base = modf(delay_samples)

    min_history_length: int = int(base) + 1

    lagrange_indices: IntVector
    lagrange_coefficient: FloatVector

    match lagrange_order:
        case 1:
            lagrange_coefficient = np.array(
                [
                    1.0 - D,
                    D,
                ],
                dtype=np.float64,
            )
            min_history_length += 1
            lagrange_indices = np.array([base, base + 1], dtype=np.int32)

        case 2:
            lagrange_coefficient = np.array(
                [
                    (D - 1.0) * (D - 2.0) / 2.0,
                    -D * (D - 2.0),
                    D * (D - 1) / 2.0,
                ],
                dtype=np.float64,
            )
            min_history_length += 2
            lagrange_indices = np.array([base, base + 1, base + 2], dtype=np.int32)

        case 3:
            lagrange_coefficient = np.array(
                [
                    -(D - 1.0) * (D - 2.0) * (D - 3.0) / 6.0,
                    D * (D - 2.0) * (D - 3.0) / 2.0,
                    -D * (D - 1.0) * (D - 3.0) / 2.0,
                    D * (D - 1.0) * (D - 2.0) / 6.0,
                ],
                dtype=np.float64,
            )
            min_history_length += 3
            lagrange_indices = np.array(
                [base, base + 1, base + 2, base + 3], dtype=np.int32
            )

    if len(history) < min_history_length:
        raise ValueError(
            f"Length of history [{len(history)}] must be >= min_history_length [{min_history_length}]"
        )

    lagrange_samples: FloatVector = history[lagrange_indices]

    return np.sum(lagrange_coefficient * lagrange_samples)


def apply_tvist_to_joint_trajectory(
    q: ArrayLike,
    dt: float,
    properties: list[ModalProperties],
    shaper_type: TvistShaper,
    lagrange_order: Literal[1, 2, 3] = 1,
):
    q_array: FloatMatrix = np.asarray(q, dtype=np.float64)
    curr_history: FloatMatrix

    last_natural_frequencies = properties[-1]["natural_frequency_hz"]
    last_zetas = properties[-1]["damping_ratio"]

    # Determine how much to extend the input trajectory based on shaper
    delta_max_final: float = 0.0
    for index, (last_natural_frequency, last_zeta) in enumerate(
        zip(last_natural_frequencies, last_zetas, strict=True)
    ):
        if shaper_type == TvistShaper.ZV:
            _, next_times = zv_shaper(2 * np.pi * last_natural_frequency, last_zeta)
        else:
            _, next_times = zvd_shaper(2 * np.pi * last_natural_frequency, last_zeta)

        delta_max_final += float(np.max(next_times))

    # For 3rd order lagrange interpolation
    required_input_extension = int(delta_max_final / dt) + 4

    # Extend the input trajectory matrix
    q_array = np.pad(
        q_array,
        ((0, required_input_extension), (0, 0)),
        mode="edge",
    )

    # Likewise extend the modal properties
    extended_properties = properties + [properties[-1]] * required_input_extension

    # Allocate for shaped output trajectory
    qs = np.empty_like(q_array)

    for n, _ in enumerate(q_array):
        curr_history = q_array[: n + 1]
        curr_history = curr_history[::-1]

        curr_modal_property: ModalProperties = extended_properties[n]

        natural_frequencies_hz = curr_modal_property["natural_frequency_hz"]
        damping_ratios = curr_modal_property["damping_ratio"]

        amplitudes, times_s = np.array([1.0]), np.array([0.0])

        for index, (frequency_hz, zeta) in enumerate(
            zip(natural_frequencies_hz, damping_ratios, strict=True)
        ):
            if shaper_type == TvistShaper.ZV:
                next_amplitudes, next_times = zv_shaper(2 * np.pi * frequency_hz, zeta)
            else:
                next_amplitudes, next_times = zvd_shaper(2 * np.pi * frequency_hz, zeta)

            if index == 0:
                amplitudes, times_s = next_amplitudes, next_times
            else:
                amplitudes, times_s = convolve_shapers(
                    amplitudes, times_s, next_amplitudes, next_times
                )

        # for 3rd order lagrange interpolation
        required_history_length = int(np.max(times_s) / dt) + 4

        if len(curr_history) < required_history_length:
            padding_required = required_history_length - len(curr_history)
            curr_history = np.pad(
                curr_history,
                ((0, padding_required), (0, 0)),
                mode="edge",
            )

        for joint_number in range(len(LEFT_ARM_JOINT_NAMES)):
            qi_history = curr_history[:, joint_number]

            qi_s: float = 0.0

            for impulse, delay_s in zip(amplitudes, times_s):
                delay_samples = delay_s / dt
                qi_hat = sample_delayed(qi_history, delay_samples, lagrange_order)
                qi_s += impulse * qi_hat

            qs[n, joint_number] = qi_s

    return qs


def main() -> None:
    history = np.array([10.0, 8.0, 6.0, 4.0])

    assert np.isclose(sample_delayed(history, 2.5, 1), 5.0)


if __name__ == "__main__":
    main()
