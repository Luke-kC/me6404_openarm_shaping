"""Independent analytical checks for the scalar oscillator experiment."""

import numpy as np
import pytest

from experiments import tvist_paper_benchmark as paper
from experiments.tvist_oscillator_experiment import (
    DAMPING_RATIO,
    DT_S,
    MOVE_DURATION_S,
    experiment_commands,
    simulate_oscillator,
)


def test_constant_frequency_step_response() -> None:
    # The solver starts at u(0); a unit command from t=0 needs an initial
    # displacement of -1 relative to equilibrium. Use the equivalent
    # free response from rest after a very short linear command ramp.
    dt = 1e-4
    t = np.arange(5001) * dt
    u = np.ones_like(t)
    u[0] = 0.0
    zeta = 0.1
    wn = 2 * np.pi * 2.0
    wd = wn * np.sqrt(1 - zeta**2)
    # Exact unit-step response averaged over step times in [0, dt], the
    # superposition that gives a linear ramp. Gauss nodes avoid the ramp edge.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    s = t[1:, None] - (nodes + 1) * dt / 2
    envelope = np.exp(-zeta * wn * s)
    step_x = 1 - envelope * (np.cos(wd * s) + zeta * wn / wd * np.sin(wd * s))
    step_v = envelope * wn**2 / wd * np.sin(wd * s)
    x, v = simulate_oscillator(u, np.full_like(t, 2.0), zeta, dt)
    np.testing.assert_allclose(x[1:], step_x @ weights / 2, rtol=0, atol=1e-9)
    np.testing.assert_allclose(v[1:], step_v @ weights / 2, rtol=0, atol=1e-8)


def test_constant_frequency_shapers_agree_and_suppress_residual_motion() -> None:
    _, commands = experiment_commands(2.0)
    np.testing.assert_array_equal(commands["Fixed-start ZVD"], commands["TVIST ZVD"])
    np.testing.assert_array_equal(commands["Fixed-end ZVD"], commands["TVIST ZVD"])
    peaks = {}
    for label, command in commands.items():
        u = np.pad(command, (0, 500), mode="edge")
        _, v = simulate_oscillator(u, np.full_like(u, 2.0), DAMPING_RATIO, DT_S)
        peaks[label] = np.max(np.abs(v[-500:]))
    assert peaks["Fixed-start ZVD"] < 0.01 * peaks["Unshaped"]


@pytest.mark.parametrize(("start", "finish"), [(2.0, 3.0), (3.0, 2.0)])
def test_varying_case_completion_times(start: float, finish: float) -> None:
    plant_hz, commands = experiment_commands(finish, initial_frequency_hz=start)
    np.testing.assert_array_equal(plant_hz[[0, -1]], [start, finish])
    np.testing.assert_allclose(
        plant_hz, start + (finish - start) * commands["Unshaped"]
    )
    for label, design_hz in (
        ("Fixed-start ZVD", start),
        ("Fixed-end ZVD", finish),
        ("TVIST ZVD", finish),
    ):
        command = commands[label]
        np.testing.assert_allclose(command[[0, -1]], [0.0, 1.0], rtol=0, atol=1e-12)
        end = int(np.flatnonzero(np.abs(command - 1.0) > 1e-12)[-1]) + 1
        # TVIST and fixed-end use the same delays after the sweep has finished.
        expected_end = MOVE_DURATION_S + 1 / (design_hz * np.sqrt(1 - DAMPING_RATIO**2))
        assert abs(end * DT_S - expected_end) <= DT_S + 1e-12


def test_paper_benchmark_reference_and_residuals() -> None:
    t, q = paper.reference()
    np.testing.assert_array_equal(q[[0, -1]], [0.0, paper.TARGET_RAD])
    assert np.max(np.diff(q) / np.diff(t)) <= paper.MAX_VELOCITY + 1e-12
    _, results = paper.benchmark()
    unshaped_peak = results["Unshaped"][3]
    assert unshaped_peak > 0
    for label in ("OSA ZV, linear", "FD-FIR ZV, cubic"):
        command, response, end, peak = results[label]
        np.testing.assert_allclose(command[end:], paper.TARGET_RAD, atol=1e-12, rtol=0)
        assert np.isfinite(response).all()
        assert peak < unshaped_peak
