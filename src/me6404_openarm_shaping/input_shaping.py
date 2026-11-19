from collections.abc import Callable, Sequence
from math import exp, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike

from me6404_openarm_shaping.data_types import FloatVector, ModalSample

ZV_FREQUENCY_TOLERANCE = 0.03
ZVD_FREQUENCY_TOLERANCE = 0.14


def zv_shaper(
    omega_n_rad_s: float, zeta: float
) -> tuple[
    FloatVector,
    FloatVector,
]:
    """
    omega_n_rad_s: positive natural frequency (rad/s)
    zeta: damping, 0 <= zeta < 1

    Returns A, t_s for a ZV input shaper

    A: impulse amplitudes
    t_s: times for each impulse
    """

    if not np.isfinite(omega_n_rad_s) or not np.isfinite(zeta):
        raise ValueError("Omega_n_rad_s and zeta must both be finite")

    if omega_n_rad_s <= 0.0:
        raise ValueError("Omega_n_rad_s must be > 0.0")

    if zeta < 0.0 or zeta >= 1.0:
        raise ValueError("zeta must be within [0, 1.0)")

    K = exp(-(zeta * pi) / sqrt(1 - zeta**2))

    w_d = omega_n_rad_s * sqrt(1 - zeta**2)

    Td = 2 * pi / w_d

    A = np.array([1.0 / (1.0 + K), K / (1.0 + K)])
    t_s = np.array([0.0, Td / 2.0])

    return A, t_s


def zvd_shaper(
    omega_n_rad_s: float, zeta: float
) -> tuple[
    FloatVector,
    FloatVector,
]:
    """
    omega_n_rad_s: positive natural frequency (rad/s)
    zeta: damping, 0 <= zeta < 1

    Returns A, t_s for a ZVD input shaper

    A: impulse amplitudes
    t_s: times for each impulse
    """
    if not np.isfinite(omega_n_rad_s) or not np.isfinite(zeta):
        raise ValueError("Omega_n_rad_s and zeta must both be finite")

    if omega_n_rad_s <= 0.0:
        raise ValueError("Omega_n_rad_s must be > 0.0")

    if zeta < 0.0 or zeta >= 1.0:
        raise ValueError("zeta must be within [0, 1.0)")

    K = exp(-(zeta * pi) / sqrt(1 - zeta**2))

    w_d = omega_n_rad_s * sqrt(1 - zeta**2)

    Td = 2 * pi / w_d

    t_s = np.array([0, Td / 2, Td])
    A = np.array([1, 2 * K, K**2]) / ((1 + K) ** 2)

    return A, t_s


def convolve_shapers(
    A1: FloatVector,
    t1_s: FloatVector,
    A2: FloatVector,
    t2_s: FloatVector,
) -> tuple[
    FloatVector,
    FloatVector,
]:
    """
    Convolves two shapers.

    Returns amp, t_s
    amp: amplitudes of the convolved shaper
    t_s: times of the convolved shaper

    Returned values are not sorted. Duplicates are not merged.
    """

    amp = np.array([], dtype=np.float64)
    t_s = np.array([], dtype=np.float64)

    for i, A1_curr in enumerate(A1):
        t1_s_curr = t1_s[i]

        for j, A2_curr in enumerate(A2):
            t2_s_curr = t2_s[j]

            amp = np.append(amp, A1_curr * A2_curr)
            t_s = np.append(t_s, t1_s_curr + t2_s_curr)

    return amp, t_s


def suggest_shaper_parameters(
    modal_samples: Sequence[ModalSample],
    tau_min: float,
    frequency_tolerance: float,
    utilization_min: float = 0.5,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """Return suggested natural frequencies [Hz], damping ratios, utilization.

    Pool paired modal samples and retain decay times >= tau_min [s].

    Greedily cover sorted frequencies with bands of ±frequency_tolerance
    relative to the nominal frequency, interpolate nominal damping, and retain bands with
    span utilization >= utilization_min (default 0.5).

    Set utilization_min=0 to return all candidates.

    Utilization is not density or occupied bandwidth.
    """
    if not 0 < frequency_tolerance < 1:
        raise ValueError("frequency_tolerance must be between 0 and 1 (exclusive).")
    if not modal_samples:
        return np.empty(0), np.empty(0), np.empty(0)

    frequencies_hz = np.concatenate(
        [sample["natural_frequency_hz"] for sample in modal_samples]
    )

    damping_ratios = np.concatenate(
        [sample["damping_ratio"] for sample in modal_samples]
    )

    # Undamped samples have infinite decay time and remain eligible.
    with np.errstate(divide="ignore"):
        tau = 1 / (damping_ratios * frequencies_hz * 2 * np.pi)

    retained = tau >= tau_min

    frequencies_hz = frequencies_hz[retained]
    damping_ratios = damping_ratios[retained]

    order = np.argsort(frequencies_hz)

    frequencies_hz = frequencies_hz[order]
    damping_ratios = damping_ratios[order]

    suggestions: list[float] = []
    damping: list[float] = []
    utilization: list[float] = []
    next_uncovered = 0

    while next_uncovered < len(frequencies_hz):
        lower_hz = frequencies_hz[next_uncovered]
        nominal_hz = lower_hz / (1 - frequency_tolerance)
        upper_hz = nominal_hz * (1 + frequency_tolerance)
        next_uncovered = np.searchsorted(frequencies_hz, upper_hz, side="right")
        covered_width = frequencies_hz[next_uncovered - 1] - lower_hz
        band_utilization = covered_width / (upper_hz - lower_hz)
        if band_utilization >= utilization_min:
            suggestions.append(nominal_hz)
            damping.append(np.interp(nominal_hz, frequencies_hz, damping_ratios))
            utilization.append(band_utilization)

    return (
        np.asarray(suggestions, dtype=np.float64),
        np.asarray(damping, dtype=np.float64),
        np.asarray(utilization, dtype=np.float64),
    )


def design_zvd_shaper(
    nominal_frequencies_hz: ArrayLike,
    nominal_damping_ratios: ArrayLike,
) -> tuple[FloatVector, FloatVector]:
    """Design/convolve ZVDs from paired frequencies [Hz] and damping ratios."""
    return _design_shaper(nominal_frequencies_hz, nominal_damping_ratios, zvd_shaper)


def design_zv_shaper(
    nominal_frequencies_hz: ArrayLike,
    nominal_damping_ratios: ArrayLike,
) -> tuple[FloatVector, FloatVector]:
    """Design/convolve ZVs from paired frequencies [Hz] and damping ratios."""
    return _design_shaper(nominal_frequencies_hz, nominal_damping_ratios, zv_shaper)


def _design_shaper(
    nominal_frequencies_hz: ArrayLike,
    nominal_damping_ratios: ArrayLike,
    design_mode: Callable[[float, float], tuple[FloatVector, FloatVector]],
) -> tuple[FloatVector, FloatVector]:
    """Return amplitudes and unrounded impulse times [s]; empty gives identity."""
    frequencies = np.asarray(nominal_frequencies_hz, dtype=np.float64)
    damping = np.asarray(nominal_damping_ratios, dtype=np.float64)

    if frequencies.ndim != 1 or damping.shape != frequencies.shape:
        raise ValueError("Nominal frequencies and damping must be matching 1-D arrays.")

    amplitudes, times_s = np.array([1.0]), np.array([0.0])

    for index, (frequency_hz, zeta) in enumerate(
        zip(frequencies, damping, strict=True)
    ):
        next_amplitudes, next_times = design_mode(2 * np.pi * frequency_hz, zeta)
        if index == 0:
            amplitudes, times_s = next_amplitudes, next_times
        else:
            amplitudes, times_s = convolve_shapers(
                amplitudes, times_s, next_amplitudes, next_times
            )

    return amplitudes, times_s


def residual_vibration_factor(
    amplitudes: FloatVector,
    times_s: FloatVector,
    omega_n_rad_s: float,
    zeta: float,
) -> float:
    """Return dimensionless residual amplitude at the final shaper impulse.

    Evaluate one underdamped modal pair using ideal continuous delays, not
    sample-rounded delays. A unit impulse at t=0 has residual factor 1.
    Unsorted or duplicate impulse times are allowed; contributions add.
    """
    if not np.isfinite(omega_n_rad_s) or omega_n_rad_s <= 0:
        raise ValueError("Natural frequency must be finite and positive in rad/s.")
    if not np.isfinite(zeta) or not 0 <= zeta < 1:
        raise ValueError("Damping ratio must be finite and within [0, 1).")

    omega_d = omega_n_rad_s * sqrt(1 - zeta**2)
    final_time = np.max(times_s)

    contributions = amplitudes * np.exp(
        -zeta * omega_n_rad_s * (final_time - times_s) + 1j * omega_d * times_s
    )

    return float(np.abs(np.sum(contributions)))
