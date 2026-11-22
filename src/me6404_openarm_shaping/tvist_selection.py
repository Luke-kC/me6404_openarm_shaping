"""Experimental offline branch-subset selection for TVIST."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import mujoco
import numpy as np

from me6404_openarm_shaping.configuration import (
    RobotConfig,
    TvistEIConfig,
    TvistShaperConfig,
)
from me6404_openarm_shaping.data_types import (
    FloatMatrix,
    FloatVector,
    JointTrajectory,
    ModalProperties,
    TrialRecording,
)
from me6404_openarm_shaping.mode_tracking import TrackedModes
from me6404_openarm_shaping.simulation_helpers import run_trial
from me6404_openarm_shaping.tvist import TvistShaper, apply_tvist_to_joint_trajectory

RMS_WINDOW_S = 0.8
EQUILIBRIUM_WINDOW_S = 0.5
SETTLING_BAND_RAD = 0.01
SELECTION_OBJECTIVES = ("min_residual", "min_settling", "beat_fixed_rms", "beat_fixed_settling")


@dataclass(frozen=True)
class SubsetScore:
    branches: tuple[int, ...]
    command_duration: float
    rms: float | None
    settling: float | None


def select_subset_indices(
    scores: Sequence[SubsetScore],
    fixed_rms: float | None = None,
    fixed_settling: float | None = None,
) -> dict[str, int]:
    """Rank one shaper's acceleration-feasible candidates, with strict benchmarks."""
    winners: dict[str, int] = {}
    best: dict[str, tuple[float, float, tuple[int, ...]]] = {}
    for index, score in enumerate(scores):
        rms, settling = score.rms, score.settling
        objectives = (
            rms,
            settling,
            settling if fixed_rms is not None and rms is not None and rms < fixed_rms else None,
            rms if fixed_settling is not None and settling is not None and settling < fixed_settling else None,
        )
        for objective, value in zip(SELECTION_OBJECTIVES, objectives, strict=True):
            if value is not None:
                key = (value, score.command_duration, score.branches)
                if objective not in best or key < best[objective]:
                    best[objective] = key
                    winners[objective] = index
    return winners


@dataclass
class SubsetCandidate:
    branches: tuple[int, ...]
    provisional_match: bool
    peak_acceleration: float | None
    violation_samples: int
    velocity_violation_samples: int
    trajectory_warning_samples: int
    trajectory: JointTrajectory | None
    schedule: list[ModalProperties]


def build_subset_candidates(
    q: FloatMatrix,
    config: RobotConfig,
    tracked: TrackedModes,
    shaper: TvistShaper,
    settings: TvistShaperConfig,
) -> list[SubsetCandidate]:
    """Build command references and reject absolute derivative-limit violations."""
    dt = config.simulation.dt_s
    decay_rate = -tracked.poles.real
    eligible = np.isfinite(tracked.poles) & (decay_rate > 0)
    if settings.tau_min_s > 0:
        eligible &= decay_rate * settings.tau_min_s <= 1
    branches = [int(i) for i in np.flatnonzero(eligible.any(axis=0))]
    weak = (tracked.mac < 0.80) | (tracked.assignment_gap[:, None] < 0.02)
    candidates: list[SubsetCandidate] = []
    for size in range(1, len(branches) + 1):
        for subset in combinations(branches, size):
            selected = eligible[:, subset]
            schedule: list[ModalProperties] = []
            for poles, keep, stable in zip(
                tracked.poles[:, subset], selected, tracked.stable, strict=True
            ):
                active = poles[keep]
                # Preserve the legacy order for summing convolved impulse contributions.
                active = active[np.argsort(np.abs(active))]
                omega_n = np.abs(active)
                schedule.append({
                    "natural_frequency_hz": omega_n / (2 * np.pi),
                    "damped_frequency_hz": active.imag / (2 * np.pi),
                    "damping_ratio": -active.real / omega_n,
                    "eigenvalues": active,
                    "real_eigenvalues": np.empty(0),
                    "stable": bool(stable),
                })
            if isinstance(settings, TvistEIConfig):
                shaped = apply_tvist_to_joint_trajectory(
                    q, dt, schedule, shaper, residual_tolerance=settings.residual_tolerance
                )
            else:
                shaped = apply_tvist_to_joint_trajectory(q, dt, schedule, shaper)
            dq: FloatMatrix = np.gradient(shaped, dt, axis=0, edge_order=2)
            ddq: FloatMatrix = np.gradient(dq, dt, axis=0, edge_order=2)
            finite = all(bool(np.isfinite(values).all()) for values in (shaped, dq, ddq))
            violations = int((np.abs(ddq) > config.acceleration_ceiling).any(axis=1).sum())
            velocity_violations = int((np.abs(dq) > config.velocity_ceiling).any(axis=1).sum())
            planning_exceedances = (np.abs(dq) > config.velocity_limits) | (
                np.abs(ddq) > config.acceleration_limits
            )
            trajectory: JointTrajectory | None = None
            # Legacy single-tier runs screened acceleration only.
            if finite and violations == 0 and (
                config.absolute_velocity_limits is None or velocity_violations == 0
            ):
                time: FloatVector = np.arange(len(shaped), dtype=np.float64) * dt
                trajectory = {"t": time, "q": shaped, "dq": dq, "ddq": ddq}
            candidates.append(SubsetCandidate(
                branches=tuple(i + 1 for i in subset),
                provisional_match=bool((selected & weak[:, subset]).any()),
                peak_acceleration=float(np.abs(ddq).max()) if finite else None,
                violation_samples=violations,
                velocity_violation_samples=velocity_violations,
                trajectory_warning_samples=int(planning_exceedances.any(axis=1).sum()),
                trajectory=trajectory,
                schedule=schedule if trajectory is not None else [],
            ))
    return candidates


def response_scores(
    recording: TrialRecording, command_end: float
) -> tuple[float | None, float | None]:
    """Whole-arm residual RMS and total equilibrium settling, as in the notebook."""
    t, q = recording["t"], recording["q"]
    if not recording["completed"] or t.size == 0 or not bool(np.isfinite(q).all()):
        return None, None
    post = t >= command_end - 1e-9  # MuJoCo time accumulation roundoff.
    if not bool(post.any()):
        return None, None
    equilibrium = q[t >= max(command_end - 1e-9, t[-1] - EQUILIBRIUM_WINDOW_S)].mean(axis=0)
    error = q - equilibrium
    rms_end = command_end + RMS_WINDOW_S
    rms = None
    if t[0] <= command_end and t[-1] >= rms_end:
        inside = (t > command_end) & (t < rms_end)
        window_t = np.r_[command_end, t[inside], rms_end]
        edges = np.array([np.interp([command_end, rms_end], t, joint) for joint in error.T]).T
        window_error = np.vstack((edges[0], error[inside], edges[1]))
        rms = float(np.sqrt(np.trapezoid((window_error**2).sum(axis=1), x=window_t) / RMS_WINDOW_S))
    in_band = (np.abs(error[post]) <= SETTLING_BAND_RAD).all(axis=1)
    stays = np.logical_and.accumulate(in_band[::-1])[::-1]
    candidates = t[post][stays]
    settling = max(command_end, float(candidates[0])) if candidates.size > 0 else None
    return rms, settling


def evaluate_subset_candidates(
    model: mujoco.MjModel,
    config: RobotConfig,
    candidates: list[SubsetCandidate],
    horizon: float,
    fixed_rms: float | None = None,
    fixed_settling: float | None = None,
) -> tuple[dict[str, tuple[SubsetCandidate, TrialRecording]], list[dict[str, object]]]:
    """Simulate survivors once and return winners plus an audit row for every subset."""
    scores: list[SubsetScore] = []
    survivors: list[tuple[SubsetCandidate, TrialRecording]] = []
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        command = candidate.trajectory
        row: dict[str, object] = {
            "branches": list(candidate.branches),
            "provisional_match": candidate.provisional_match,
            "peak_commanded_acceleration_rad_s2": candidate.peak_acceleration,
            "acceleration_violation_samples": candidate.violation_samples,
            "velocity_violation_samples": candidate.velocity_violation_samples,
            "trajectory_limit_exceedance_samples": candidate.trajectory_warning_samples,
            "simulated": command is not None,
        }
        rows.append(row)
        if command is None:
            continue
        end = float(command["t"][-1])
        recording = run_trial(
            model, command, config.waypoints[0], config.motor_kp, config.motor_kd,
            horizon, config=config, headless=True,
        )
        rms, settling = response_scores(recording, end)
        row.update({
            "command_duration_s": end,
            "residual_rms_rad": rms,
            "total_settling_s": settling,
            "simulation_checks_pass": recording["simulation_checks"]["passing"],
        })
        scores.append(SubsetScore(candidate.branches, end, rms, settling))
        survivors.append((candidate, recording))
    winners = {
        objective: survivors[index]
        for objective, index in select_subset_indices(scores, fixed_rms, fixed_settling).items()
    }
    return winners, rows
