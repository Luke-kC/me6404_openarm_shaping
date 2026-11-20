from collections.abc import Callable, Sequence
from functools import partial
from math import acos, exp, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike

from me6404_openarm_shaping.data_types import FloatVector, ModalSample

ZV_FREQUENCY_TOLERANCE = 0.03
ZVD_FREQUENCY_TOLERANCE = 0.14
DEFAULT_EI_RESIDUAL_TOLERANCE = 0.05


def ei_frequency_tolerance(residual_tolerance: float) -> float:
    """Greedy coverage half-band from undamped EI, not a damped residual bound."""
    if residual_tolerance == 0:
        return ZVD_FREQUENCY_TOLERANCE  # Zero-tolerance EI is exactly ZVD.
    if residual_tolerance == DEFAULT_EI_RESIDUAL_TOLERANCE:
        return 0.20  # Round the analytical 5% band to the usual ±20%.
    return acos(1 - 4 * residual_tolerance / (1 + residual_tolerance)) / pi


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


def ei_shaper(
    omega_n_rad_s: float,
    zeta: float,
    residual_tolerance: float = DEFAULT_EI_RESIDUAL_TOLERANCE,
    *,
    use_polynomial_fit: bool = True,
) -> tuple[FloatVector, FloatVector]:
    """Three-impulse EI with a damped polynomial fit through zeta = 0.3.

    Above the fit range, use a nonnegative half-period design targeting no
    more than the specified nominal residual, not an equal-ripple guarantee.
    Set use_polynomial_fit=False to use that design throughout a modal schedule
    and avoid a jump in amplitudes and delays at the fit boundary.
    """
    amplitudes, times_s = zvd_shaper(omega_n_rad_s, zeta)
    if not 0 <= residual_tolerance < 1:
        raise ValueError("EI residual_tolerance must be within [0, 1).")
    if residual_tolerance == 0:
        return amplitudes, times_s

    v = residual_tolerance
    if zeta == 0:
        return np.array([(1 + v) / 4, (1 - v) / 2, (1 + v) / 4]), times_s

    if use_polynomial_fit and zeta <= 0.3:
        z = zeta
        a1 = (
            0.24968
            + 0.24961 * v
            + (0.80008 + 1.23328 * v) * z
            + (0.49599 + 3.17316 * v) * z**2
        )
        a3 = (
            0.25149
            + 0.21474 * v
            + (-0.83249 + 1.41498 * v) * z
            + (0.85181 - 4.90094 * v) * z**2
        )
        middle_fraction = (
            0.49990
            + v * z * (0.46159 + 4.26169 * z + 1.75601 * z**2)
            + v**2 * z * (8.57843 - 108.644 * z + 336.989 * z**2)
        )
        fitted = np.array([a1, 1 - (a1 + a3), a3])
        if np.all(fitted >= 0) and 0 < middle_fraction < 1:
            times_s[1] = times_s[-1] * middle_fraction
            return fitted, times_s

    # Blend ZVD (zero nominal residual) with the in-phase outer impulses.
    # Their weights are proportional to [1, 0, K²], where K is the
    # half-period decay. The nominal signed residual is linear in the blend.
    outer = amplitudes.copy()
    outer[1] = 0
    outer /= outer.sum()
    outer_residual = 2 * outer[-1]
    fraction = 1.0 if v >= outer_residual else v / outer_residual
    return (1 - fraction) * amplitudes + fraction * outer, times_s


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


def design_ei_shaper(
    nominal_frequencies_hz: ArrayLike,
    nominal_damping_ratios: ArrayLike,
    residual_tolerance: float = DEFAULT_EI_RESIDUAL_TOLERANCE,
) -> tuple[FloatVector, FloatVector]:
    """Design/convolve EI components from paired frequencies [Hz] and damping ratios."""
    return _design_shaper(
        nominal_frequencies_hz,
        nominal_damping_ratios,
        partial(ei_shaper, residual_tolerance=residual_tolerance),
    )


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
