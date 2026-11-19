"""Small analytical checks for impulse shapers."""

import numpy as np
import pytest

from me6404_openarm_shaping.data_types import ModalSample
from me6404_openarm_shaping.input_shaping import (
    convolve_shapers,
    design_zv_shaper,
    design_zvd_shaper,
    residual_vibration_factor,
    suggest_shaper_parameters,
    zv_shaper,
    zvd_shaper,
)


def test_undamped_zvd() -> None:
    amplitudes, times = zvd_shaper(2 * np.pi, 0.0)
    np.testing.assert_allclose(
        amplitudes, [0.25, 0.5, 0.25], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(times, [0, 0.5, 1.0], rtol=0, atol=1e-14)
    assert amplitudes.sum() == pytest.approx(1.0, rel=0, abs=1e-14)


def test_convolution_aggregates_simultaneous_impulses() -> None:
    amplitudes = np.array([0.5, 0.5])
    times = np.array([0.0, 1.0])
    combined_amplitudes, combined_times = convolve_shapers(
        amplitudes, times, amplitudes, times
    )
    # Integer-second delays are exactly representable here. Sum contributions
    # independently of output order or whether the function merges duplicates.
    unique_times = np.unique(combined_times)
    aggregate = np.array([
        combined_amplitudes[combined_times == time].sum()
        for time in unique_times
    ])
    np.testing.assert_allclose(unique_times, [0, 1, 2], rtol=0, atol=1e-14)
    np.testing.assert_allclose(aggregate, [0.25, 0.5, 0.25], rtol=0, atol=1e-14)
    assert aggregate.sum() == pytest.approx(1.0, rel=0, abs=1e-14)


def test_zvd_rejects_critical_damping() -> None:
    with pytest.raises(ValueError, match="zeta"):
        zvd_shaper(2 * np.pi, 1.0)


def test_damped_zvd_cancellation() -> None:
    omega_n = 2 * np.pi * 2.2  # rad/s
    zeta = 0.1
    A, t = zvd_shaper(omega_n, zeta)

    omega_d = omega_n * np.sqrt(1 - zeta**2)
    lam = -zeta * omega_n + 1j * omega_d

    # ZVD places a double zero at the damped system pole.
    H = np.sum(A * np.exp(-lam * t))
    dH = np.sum(-t * A * np.exp(-lam * t))

    np.testing.assert_allclose(H, 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dH, 0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("zeta", [0.0, 0.1])
def test_identity_shaper_residual(zeta: float) -> None:
    residual = residual_vibration_factor(
        np.array([1.0]), np.array([0.0]), 2 * np.pi * 2.2, zeta
    )
    assert residual == pytest.approx(1.0, rel=0, abs=1e-14)


def test_undamped_quarter_period_residual() -> None:
    # Equal impulses 90 degrees apart give magnitude sqrt(0.5).
    residual = residual_vibration_factor(
        np.array([0.5, 0.5]), np.array([0.0, 0.25]), 2 * np.pi, 0.0
    )
    assert residual == pytest.approx(np.sqrt(0.5), rel=0, abs=1e-14)


def test_convolved_zvd_residual_at_both_design_pairs() -> None:
    design_pairs = [(2 * np.pi * 2.2, 0.1), (2 * np.pi * 4.0, 0.2)]
    A1, t1 = zvd_shaper(*design_pairs[0])
    A2, t2 = zvd_shaper(*design_pairs[1])
    amplitudes, times = convolve_shapers(A1, t1, A2, t2)
    for omega_n, zeta in design_pairs:
        residual = residual_vibration_factor(amplitudes, times, omega_n, zeta)
        np.testing.assert_allclose(residual, 0, rtol=0, atol=1e-12)


def test_suggest_shaper_parameters_zvd_band() -> None:
    # One full 0.86--1.14 Hz band, one isolated sample (zero span), and
    # one rapidly decaying sample that must be excluded before band selection.
    samples: list[ModalSample] = [
        {"natural_frequency_hz": np.array([3.0, 1.14]),
         "damping_ratio": np.array([0.1, 0.07])},
        {"natural_frequency_hz": np.array([0.86, 100.0]),
         "damping_ratio": np.array([0.05, 0.5])},
    ]
    frequencies, damping, utilization = suggest_shaper_parameters(samples, 0.2, 0.14)
    np.testing.assert_allclose(frequencies, [1.0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(damping, [0.06], rtol=0, atol=1e-14)
    np.testing.assert_allclose(utilization, [1.0], rtol=0, atol=1e-14)


@pytest.mark.parametrize("samples", [
    [],
    [{"natural_frequency_hz": np.array([100.0]), "damping_ratio": np.array([0.5])}],
    [{"natural_frequency_hz": np.array([1.0]), "damping_ratio": np.array([0.1])}],
])
def test_suggest_shaper_parameters_empty_result(samples: list[ModalSample]) -> None:
    for values in suggest_shaper_parameters(samples, 0.2, 0.14):
        assert values.shape == (0,)
        assert values.dtype == np.float64


def test_design_zvd_shaper_empty() -> None:
    amplitudes, times = design_zvd_shaper([], [])
    np.testing.assert_allclose(amplitudes, [1.0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(times, [0.0], rtol=0, atol=1e-14)


def test_suggest_zvd_returns_low_utilization_candidates_when_requested() -> None:
    samples: list[ModalSample] = [{
        "natural_frequency_hz": np.array([0.86, 1.14, 3.0]),
        "damping_ratio": np.array([0.05, 0.07, 0.1]),
    }]
    frequencies, damping, utilization = suggest_shaper_parameters(
        samples, 0.2, 0.14, utilization_min=0.0
    )
    np.testing.assert_allclose(frequencies, [1.0, 3.0 / 0.86], rtol=0, atol=1e-14)
    np.testing.assert_allclose(damping, [0.06, 0.1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(utilization, [1.0, 0.0], rtol=0, atol=1e-14)
    selected = utilization >= 0.5
    for candidates, suggestions in zip(
        (frequencies, damping, utilization),
        suggest_shaper_parameters(samples, 0.2, 0.14),
        strict=True,
    ):
        np.testing.assert_allclose(
            candidates[selected], suggestions, rtol=0, atol=1e-14
        )


def test_narrow_zv_band_requires_more_shapers() -> None:
    samples: list[ModalSample] = [{
        "natural_frequency_hz": np.array([0.97, 1.03, 1.067, 1.132]),
        "damping_ratio": np.full(4, 0.1),
    }]
    narrow, _, _ = suggest_shaper_parameters(samples, 0.2, 0.03, 0.3)
    wide, _, _ = suggest_shaper_parameters(samples, 0.2, 0.14, 0.3)
    np.testing.assert_allclose(narrow, [1.0, 1.1])
    assert len(wide) == 1


def test_design_zv_shaper_two_modes() -> None:
    actual = design_zv_shaper([2.2, 4.0], [0.1, 0.2])
    expected = convolve_shapers(
        *zv_shaper(2 * np.pi * 2.2, 0.1), *zv_shaper(2 * np.pi * 4.0, 0.2)
    )
    for values, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(values, reference)


def test_design_zvd_shaper_single_mode() -> None:
    actual = design_zvd_shaper([2.2], [0.1])
    expected = zvd_shaper(2 * np.pi * 2.2, 0.1)
    for values, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(values, reference, rtol=1e-12, atol=1e-14)


def test_design_zvd_shaper_two_modes() -> None:
    actual = design_zvd_shaper([2.2, 4.0], [0.1, 0.2])
    A1, t1 = zvd_shaper(2 * np.pi * 2.2, 0.1)
    A2, t2 = zvd_shaper(2 * np.pi * 4.0, 0.2)
    expected = convolve_shapers(A1, t1, A2, t2)
    for values, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(values, reference, rtol=1e-12, atol=1e-14)
    assert actual[0].sum() == pytest.approx(1.0, rel=0, abs=1e-14)
