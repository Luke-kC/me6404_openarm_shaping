import numpy as np
import pytest

from me6404_openarm_shaping.data_types import ModalProperties
from me6404_openarm_shaping.input_shaping import zv_shaper
from me6404_openarm_shaping.trajectories import (
    apply_shaper_to_joint_trajectory,
)
from me6404_openarm_shaping.tvist import (
    TvistShaper,
    apply_tvist_to_joint_trajectory,
)


@pytest.mark.parametrize(
    "shaper_type, delay_factor",
    [
        (TvistShaper.ZV, 0.5),
        (TvistShaper.ZVD, 1.0),
    ],
)
def test_tvist_multimode_tail(shaper_type, delay_factor):
    dt = 0.01
    n_samples = 100

    # Nonconstant 7-DOF trajectory
    q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    q1 = q0 + np.array([0.5, -0.3, 0.2, 0.4, -0.2, 0.1, -0.1])

    s = np.linspace(0.0, 1.0, n_samples)
    q = q0 + s[:, None] * (q1 - q0)

    # Two vibration modes
    frequencies_hz = np.array([4.0, 5.0])
    zeta = np.array([0.0, 0.0])

    modal: ModalProperties = {
        "natural_frequency_hz": frequencies_hz,
        "damped_frequency_hz": frequencies_hz,
        "damping_ratio": zeta,
        "eigenvalues": 2j * np.pi * frequencies_hz,
        "real_eigenvalues": np.array([]),
        "stable": False,
    }

    properties = [modal] * n_samples

    qs = apply_tvist_to_joint_trajectory(
        q=q,
        dt=dt,
        properties=properties,
        shaper_type=shaper_type,
        lagrange_order=1,
    )

    # Combined maximum delay is the sum of modal delays.
    delta_max = delay_factor * np.sum(1.0 / frequencies_hz)

    # Current implementation adds 4 samples for interpolation.
    expected_extension = int(delta_max / dt) + 4

    # 1. Output includes the required tail.
    assert qs.shape == (n_samples + expected_extension, 7)

    # 2. Original trajectory has not finished at the original endpoint.
    assert not np.allclose(qs[n_samples - 1], q1)

    # 3. Extended trajectory reaches the desired endpoint.
    np.testing.assert_allclose(
        qs[-1],
        q1,
        rtol=0,
        atol=1e-12,
    )

    # 4. Input properties list was not modified.
    assert len(properties) == n_samples


def test_tvist_time_varying_frequency():
    dt = 0.01
    n_samples = 40

    # Linear trajectory for joint 1
    q = np.zeros((n_samples, 7))
    q[:, 0] = 0.01 * np.arange(n_samples)

    # Frequency changes from 4 Hz to 5 Hz at n=20
    properties = []

    for n in range(n_samples):
        frequency_hz = 4.0 if n < 20 else 5.0

        properties.append(
            {
                "natural_frequency_hz": np.array([frequency_hz]),
                "damped_frequency_hz": np.array([frequency_hz]),
                "damping_ratio": np.array([0.0]),
                "eigenvalues": np.array([1j * 2 * np.pi * frequency_hz]),
                "real_eigenvalues": np.array([]),
                "stable": False,
            }
        )

    q_tvist = apply_tvist_to_joint_trajectory(
        q=q,
        dt=dt,
        properties=properties,
        shaper_type=TvistShaper.ZV,
        lagrange_order=1,
    )

    # At n=19, frequency = 4 Hz, delay = 12.5 samples.
    # The delayed input is interpolated between q[7] and q[6].
    np.testing.assert_allclose(
        q_tvist[19, 0],
        0.1275,
        rtol=0,
        atol=1e-12,
    )

    # At n=20, frequency = 5 Hz, delay = 10 samples.
    # q_s[20] = 0.5*q[20] + 0.5*q[10] = 0.15
    np.testing.assert_allclose(
        q_tvist[20, 0],
        0.15,
        rtol=0,
        atol=1e-12,
    )


def test_tvist_fractional_delay():
    dt = 0.01
    n_samples = 40

    # Linear trajectory for joint 1; other joints remain stationary
    q = np.zeros((n_samples, 7))
    q[:, 0] = 0.01 * np.arange(n_samples)

    # Single vibration mode: 4 Hz, zero damping
    frequency_hz = 4.0
    zeta = 0.0

    modal: ModalProperties = {
        "natural_frequency_hz": np.array([frequency_hz]),
        "damped_frequency_hz": np.array([frequency_hz]),
        "damping_ratio": np.array([zeta]),
        "eigenvalues": np.array([1j * 2 * np.pi * frequency_hz]),
        "real_eigenvalues": np.array([]),
        "stable": False,
    }

    properties = [modal] * n_samples

    q_tvist = apply_tvist_to_joint_trajectory(
        q=q,
        dt=dt,
        properties=properties,
        shaper_type=TvistShaper.ZV,
        lagrange_order=1,
    )

    # ZV: A = [0.5, 0.5], delays = [0, 0.125] s
    # Second delay = 12.5 samples
    #
    # At n=20:
    # q_s[20] = 0.5*q[20] + 0.5*(0.5*q[8] + 0.5*q[7])
    #         = 0.5*0.20 + 0.5*(0.5*0.08 + 0.5*0.07)
    #         = 0.1375

    np.testing.assert_allclose(
        q_tvist[20, 0],
        0.1375,
        rtol=0,
        atol=1e-12,
    )

    # The other six joints should remain at zero
    np.testing.assert_allclose(
        q_tvist[:, 1:],
        0.0,
        rtol=0,
        atol=1e-12,
    )


def test_tvist_matches_fixed_zv():
    dt = 0.01
    t = np.arange(101) * dt

    # Smooth point-to-point joint trajectory
    q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    delta_q = np.array([0.5, -0.3, 0.2, 0.4, -0.2, 0.1, -0.1])

    s = 0.5 * (1 - np.cos(np.pi * t))
    ds = 0.5 * np.pi * np.sin(np.pi * t)
    dds = 0.5 * np.pi**2 * np.cos(np.pi * t)

    q = q0 + s[:, None] * delta_q
    dq = ds[:, None] * delta_q
    ddq = dds[:, None] * delta_q

    # Fixed modal properties: 5 Hz, zero damping
    # ZV delay = 0.1 s = exactly 10 samples
    frequency_hz = 5.0
    zeta = 0.0

    modal: ModalProperties = {
        "natural_frequency_hz": np.array([frequency_hz]),
        "damped_frequency_hz": np.array([frequency_hz]),
        "damping_ratio": np.array([zeta]),
        "eigenvalues": np.array([1j * 2 * np.pi * frequency_hz]),
        "real_eigenvalues": np.array([]),
        "stable": False,  # Undamped mode is marginally stable
    }
    properties = [modal] * len(q)

    # Existing fixed ZV implementation
    amplitudes, delays = zv_shaper(2 * np.pi * frequency_hz, zeta)

    _, q_fixed, _, _ = apply_shaper_to_joint_trajectory(
        q, dq, ddq, amplitudes, delays, dt
    )

    # New OSA-FD-FIR implementation
    q_tvist = apply_tvist_to_joint_trajectory(
        q=q,
        dt=dt,
        properties=properties,
        shaper_type=TvistShaper.ZV,
        lagrange_order=1,
    )

    # Compare the overlapping portion.
    # The fixed implementation includes the additional shaper tail.
    np.testing.assert_allclose(
        q_tvist[: len(q), :],
        q_fixed[: len(q), :],
        rtol=0,
        atol=1e-12,
    )


def test_tvist_unity_gain():
    dt = 0.002
    n_samples = 100

    # Constant, nonzero 7-DOF joint trajectory
    q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    q = np.tile(q0, (n_samples, 1))

    # Two vibration modes with time-varying modal properties
    properties = []

    for n in range(n_samples):
        properties.append(
            {
                "natural_frequency_hz": np.array(
                    [
                        2.0 + 0.5 * np.sin(n * 0.05),
                        4.0 + 0.5 * np.cos(n * 0.05),
                    ]
                ),
                "damping_ratio": np.array(
                    [
                        0.05 + 0.02 * np.sin(n * 0.05),
                        0.08 + 0.02 * np.cos(n * 0.05),
                    ]
                ),
            }
        )

    qs = apply_tvist_to_joint_trajectory(
        q=q,
        dt=dt,
        properties=properties,
        shaper_type=TvistShaper.ZV,
        lagrange_order=1,
    )

    assert qs.shape[0] > q.shape[0]
    assert qs.shape[1] == q.shape[1]

    # The entire shaped trajectory should remain constant.
    expected = np.tile(q0, (len(qs), 1))

    np.testing.assert_allclose(
        qs,
        expected,
        rtol=0,
        atol=1e-12,
    )
