"""Select reproducible OpenArm reference paths without running any shapers.

Run with ``python -m experiments.select_trajectories --output /tmp/opencode/trajectory_suite.yaml``.
Use ``--modal-stride 2`` to screen faster at the risk of missing candidates.
This checks frozen poses with the other joints at their model defaults, not
the measured dynamics.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
from pathlib import Path

import mujoco
import numpy as np
import yaml
from numpy.typing import NDArray

from me6404_openarm_shaping.batch_sim import DEFAULT_BATCH_CONFIG
from me6404_openarm_shaping.configuration import (
    RobotConfig,
    load_robot_config,
)
from me6404_openarm_shaping.data_types import (
    ContactEvent,
    FloatMatrix,
    FloatVector,
    ModalSample,
)
from me6404_openarm_shaping.input_shaping import (
    ZVD_FREQUENCY_TOLERANCE,
    suggest_shaper_parameters,
)
from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    MODEL_PATH,
    MODEL_REVISION,
    ModalWorkspace,
    load_model,
    resolve_joint_indices,
    state_matrix,
)
from me6404_openarm_shaping.simulation_safety import (
    EXPECTED_FINGER_CONTACT,
    record_penetrating_contacts,
)
from me6404_openarm_shaping.trajectories import waypoint_trajectory

SEED = 6404
MARGIN_RAD = 0.20
LIMIT_SCALE = 0.8  # Match robot_sim's unshaped reference.
MIN_MAC = 0.80
MIN_ASSIGNMENT_GAP = 0.02
BANDS = ("narrow", "medium", "wide")
MAX_PER_BAND = 10
BETA_MODE_COUNT = 2
MIN_FIXED_ZVD_COMPONENTS = 2
MIN_UNTRACKED_ZVD_COMPONENTS = 4
MAX_RUNNABLE_UNTRACKED = MAX_PER_BAND
UNTRACKED_BAND = "multimode_untracked"


@dataclass
class Candidate:
    number: int
    waypoints: FloatMatrix
    duration: float
    travel: float
    clearance: FloatVector
    events: list[ContactEvent]
    frequencies: FloatMatrix | None
    decay_times: FloatMatrix | None
    fixed_zvd_frequencies: FloatVector
    minimum_mac: float | None
    minimum_assignment_gap: float | None
    modal_samples: int
    tracking_failure: str | None

    @property
    def spans(self) -> FloatVector:
        assert self.frequencies is not None
        return np.ptp(self.frequencies, axis=0) / self.frequencies[0]

    @property
    def beta(self) -> float | None:
        return (
            None
            if self.frequencies is None
            else float(self.spans[:BETA_MODE_COUNT].max())
        )


def band(beta: float) -> str:
    if beta < 0.40:
        return "narrow"
    if beta < 0.75:
        return "medium"
    return "wide"


def screen_band(beta: float | None, fixed_zvd_count: int) -> str:
    if beta is None:
        return (
            UNTRACKED_BAND
            if fixed_zvd_count >= MIN_UNTRACKED_ZVD_COMPONENTS
            else "tracking_failed_too_few_fixed_zvd_components"
        )
    group = band(beta)
    if group in ("medium", "wide") and fixed_zvd_count < MIN_FIXED_ZVD_COMPONENTS:
        return "too_few_fixed_zvd_components"
    return group


def joint_clearance(
    q: FloatMatrix, waypoints: FloatMatrix, limits: FloatMatrix
) -> FloatVector | None:
    """Allow an initial entry into the inset, never a later exit."""
    tolerance = 1e-10
    lower, upper = limits.T
    clearance = np.minimum(q - lower, upper - q)
    waypoint_clearance = np.minimum(waypoints[1:] - lower, upper - waypoints[1:])
    if (
        not np.isfinite(q).all()
        or clearance.min() < -tolerance
        or waypoint_clearance.min() < MARGIN_RAD - tolerance
    ):
        return None
    inset = clearance >= MARGIN_RAD - tolerance
    if np.any(np.maximum.accumulate(inset, axis=0) & ~inset):
        return None
    return clearance.min(axis=0)


@dataclass
class ModalScan:
    sampled_poles: list[NDArray[np.complex128]]
    tracked_poles: NDArray[np.complex128] | None
    minimum_mac: float | None
    minimum_assignment_gap: float | None
    tracking_failure: str | None


type ModalCache = dict[bytes, tuple[NDArray[np.complex128], NDArray[np.complex128]]]


def track_modes(
    model: mujoco.MjModel,
    q: FloatMatrix,
    config: RobotConfig,
    cache: ModalCache | None = None,
) -> ModalScan | str:
    """Track the two lowest initial modes while checking all poles for stability."""
    sampled_poles = []
    tracked_poles = []
    tracking_failure = None
    minimum_mac = 1.0
    minimum_gap = np.inf
    previous = None
    rows = np.arange(BETA_MODE_COUNT)
    assignments_by_count: dict[int, NDArray[np.intp]] = {}
    workspace = ModalWorkspace(model)
    for position in q:
        key = position.tobytes() if cache is not None else b""
        cached = cache.get(key) if cache is not None else None
        if cached is None:
            try:
                matrix = state_matrix(
                    model,
                    position,
                    config.motor_kp,
                    config.motor_kd,
                    workspace=workspace,
                )
                if not np.isfinite(matrix).all():
                    return "invalid_modal_calculation"
                poles, vectors = np.linalg.eig(matrix)
            except np.linalg.LinAlgError:
                return "invalid_modal_calculation"
            if not np.isfinite(poles).all() or not np.isfinite(vectors).all():
                return "invalid_modal_calculation"
            if np.any(poles.real >= 0):
                return "unstable_or_marginal_modes"
            oscillatory = poles.imag > 1e-10
            oscillatory_poles = poles[oscillatory]
            shapes = vectors[:7, oscillatory]
            if cache is not None:
                cache[key] = (oscillatory_poles, shapes)
        else:
            oscillatory_poles, shapes = cached
        sampled_poles.append(oscillatory_poles)
        if tracking_failure is not None:
            continue
        count = len(oscillatory_poles)
        if count < BETA_MODE_COUNT:
            tracking_failure = (
                "too_few_oscillatory_modes"
                if previous is not None
                else "too_few_initial_oscillatory_modes"
            )
            continue
        norms = np.linalg.norm(shapes, axis=0)
        # A zero displacement vector cannot match either tracked mode.
        shapes = shapes / np.where(norms > 0, norms, 1.0)
        if previous is None:
            order = np.argsort(np.abs(oscillatory_poles))[:BETA_MODE_COUNT]
            if np.any(norms[order] == 0):
                tracking_failure = "missing_displacement_mode_shape"
                continue
        else:
            # Complex modal assurance criterion: invariant to phase and scale.
            # Match both tracked modes to distinct poles, even if other modes appear or disappear.
            if count not in assignments_by_count:
                assignments_by_count[count] = np.array(
                    list(permutations(range(count), BETA_MODE_COUNT)), dtype=np.intp
                )
            assignments = assignments_by_count[count]
            mac = np.abs(previous.conj().T @ shapes) ** 2
            overlaps = mac[rows, assignments]
            scores = overlaps.sum(axis=1)
            if not np.isfinite(scores).all():
                tracking_failure = "invalid_mode_overlap"
                continue
            best = int(scores.argmax())
            order = assignments[best]
            minimum_mac = min(minimum_mac, float(overlaps[best].min()))
            if len(scores) > 1:
                runner_up = np.partition(scores, -2)[-2]
                minimum_gap = min(minimum_gap, float(scores[best] - runner_up))
            if minimum_mac < MIN_MAC or minimum_gap < MIN_ASSIGNMENT_GAP:
                tracking_failure = "ambiguous_mode_assignment"
                continue
        tracked_poles.append(oscillatory_poles[order])
        previous = shapes[:, order]
    if tracking_failure is not None:
        return ModalScan(sampled_poles, None, None, None, tracking_failure)
    return ModalScan(
        sampled_poles, np.array(tracked_poles), minimum_mac, float(minimum_gap), None
    )


def fixed_zvd_frequencies(
    poles: list[NDArray[np.complex128]], tau_min: float, utilization_min: float
) -> FloatVector:
    # Sort each sample as robot_sim does before pooling natural frequency/damping pairs.
    samples: list[ModalSample] = []
    for sample in poles:
        magnitude = np.abs(sample)
        order = np.argsort(magnitude)
        samples.append(
            {
                "natural_frequency_hz": magnitude[order] / (2 * np.pi),
                "damping_ratio": -sample.real[order] / magnitude[order],
            }
        )
    nominal, _, _ = suggest_shaper_parameters(
        samples, tau_min, ZVD_FREQUENCY_TOLERANCE, utilization_min
    )
    return nominal


def generate_waypoints(
    rng: np.random.Generator, limits: FloatMatrix, number: int, reference: FloatMatrix
) -> FloatMatrix:
    lower, upper = limits.T
    # Leave a little extra room before rounding. Larger sweeps explore posture
    # changes. A common shoulder-motion scale helps match travel and duration
    # even when the elbow and wrist change the modal frequencies differently.
    lower, upper = lower + 0.21, upper - 0.21
    if number % 2 == 0:
        # Alternate broad sweeps and perturbations of the configured posture
        # excursion; include both single-leg and two-leg paths.
        waypoints = reference[1:]
        family = (number // 2 - 1) % 4
        if family == 0:
            targets = rng.uniform(lower, upper, size=(1, 7))
        elif family == 1:
            targets = rng.choice(waypoints, axis=0)[None, :] + rng.normal(
                0, 0.45, (1, 7)
            )
        elif family == 2:
            targets = waypoints[[0, -1]] + rng.normal(0, 0.30, (2, 7))
        else:
            first = rng.uniform(lower, upper)
            targets = np.array([first, first + rng.normal(0, 0.60, 7)])
        return np.vstack((np.zeros(7), np.clip(targets, lower, upper).round(6)))

    number = (number + 1) // 2
    elbow, wrist = (
        ((0.22, 0.32), (0.19, 0.43)),
        ((0.22, 0.34), (-0.22, 0.13)),
        ((0.22, 0.32), (-0.72, -0.43)),
        ((0.53, 0.88), (-0.10, 0.34)),
    )[(number - 1) % 4]
    target = np.array(
        [
            rng.choice([-1, 1]) * rng.uniform(0.83, 1.16),
            rng.uniform(-0.17, -0.04),
            rng.uniform(-0.40, 0.40),
            rng.uniform(*elbow),
            rng.uniform(-0.24, 0.24),
            rng.uniform(-0.10, 0.10),
            rng.uniform(*wrist),
        ]
    )
    targets = np.clip(target[None, :], lower, upper)
    if number % 8 in (1, 5):
        # Give some narrow controls more travel without requiring a large
        # elbow/wrist excursion: make a second leg mostly along J1.
        second = targets[0].copy()
        second[0] *= -rng.uniform(0.50, 1.0)
        targets = np.vstack((targets, second))
    elif number % 8 == 0:
        intermediate = np.clip(0.65 * target, lower, upper)
        targets = np.vstack((intermediate, targets))
    return np.vstack((np.zeros(7), targets.round(6)))


def evaluate(
    model: mujoco.MjModel,
    config: RobotConfig,
    limits: FloatMatrix,
    number: int,
    waypoints: FloatMatrix,
    tau_min: float,
    utilization_min: float,
    modal_stride: int = 1,
    cache: ModalCache | None = None,
) -> Candidate | str:
    t, q, _, _ = waypoint_trajectory(
        waypoints,
        LIMIT_SCALE * config.velocity_limits,
        LIMIT_SCALE * config.acceleration_limits,
        dt=config.simulation.dt_s,
    )
    clearance = joint_clearance(q, waypoints, limits)
    if clearance is None:
        return "joint_clearance"
    data = mujoco.MjData(model)
    positions, _ = resolve_joint_indices(model)
    events: dict[tuple[int, ...], ContactEvent] = {}
    for time, position in zip(t, q, strict=True):
        data.qpos[positions] = position
        mujoco.mj_fwdPosition(model, data)
        record_penetrating_contacts(model, data, events, float(time))
        if any(
            frozenset(event["geoms"]) != EXPECTED_FINGER_CONTACT
            for event in events.values()
        ):
            return "disallowed_contact"

    # Coarsen only modal sampling, after checking contacts on the full grid.
    modal_q = q
    if modal_stride > 1:
        keep = np.zeros(len(q), dtype=bool)
        keep[::modal_stride] = True
        keep[-1] = True
        for waypoint in waypoints:
            # The interpolator writes each endpoint exactly onto the dt grid.
            keep |= np.all(q == waypoint, axis=1)
        modal_q = q[keep]
    result = track_modes(model, modal_q, config, cache)
    if isinstance(result, str):
        return result
    poles = result.tracked_poles
    return Candidate(
        number,
        waypoints,
        float(t[-1]),
        float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()),
        clearance,
        list(events.values()),
        None if poles is None else np.abs(poles) / (2 * np.pi),
        None if poles is None else -1 / poles.real,
        fixed_zvd_frequencies(result.sampled_poles, tau_min, utilization_min),
        result.minimum_mac,
        result.minimum_assignment_gap,
        len(result.sampled_poles),
        result.tracking_failure,
    )


def select_band(candidates: list[Candidate]) -> list[Candidate]:
    """Cover the observed beta range with up to ten distinct paths."""
    by_beta: list[tuple[float, int, Candidate]] = []
    for candidate in candidates:
        beta = candidate.beta
        assert beta is not None
        by_beta.append((beta, candidate.number, candidate))
    by_beta.sort(key=lambda item: item[:2])
    if len(by_beta) <= MAX_PER_BAND:
        return [candidate for _, _, candidate in by_beta]

    # Reserve both endpoints before assigning interior targets so a sparse
    # band still spans its full observed range.
    selected = [by_beta[0], by_beta[-1]]
    remaining = by_beta[1:-1]
    for target in np.linspace(by_beta[0][0], by_beta[-1][0], MAX_PER_BAND)[1:-1]:
        nearest = min(
            range(len(remaining)),
            key=lambda i: (abs(remaining[i][0] - target), remaining[i][1]),
        )
        selected.append(remaining.pop(nearest))
    return [
        candidate for _, _, candidate in sorted(selected, key=lambda item: item[:2])
    ]


def entry(
    candidate: Candidate, group: str, band_index: int | None, tau_min: float
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": f"{group}-{band_index if band_index is not None else candidate.number:02d}",
        "triplet_id": None,
        "band": group,
        "candidate_number": candidate.number,
        "waypoints": candidate.waypoints.tolist(),
        "duration_s": candidate.duration,
        "joint_space_travel_rad": candidate.travel,
        "min_reference_clearance_by_joint_rad": candidate.clearance.tolist(),
        "contact_screen": {
            "disallowed_penetrating_pairs": 0,
            "informational_contacts": candidate.events,
        },
        "modal_samples": candidate.modal_samples,
        "tracking_failure": candidate.tracking_failure,
        "minimum_mode_match_mac": candidate.minimum_mac,
        "minimum_assignment_score_gap": (
            candidate.minimum_assignment_gap
            if candidate.minimum_assignment_gap is not None
            and np.isfinite(candidate.minimum_assignment_gap)
            else None
        ),
        "beta": candidate.beta,
        "beta_decay_eligible_modes": None,
        "fixed_zvd_component_count": len(candidate.fixed_zvd_frequencies),
        "fixed_zvd_nominal_frequencies_hz": candidate.fixed_zvd_frequencies.tolist(),
        "modes": [],
    }
    frequencies, decay_times = candidate.frequencies, candidate.decay_times
    if frequencies is not None and decay_times is not None:
        eligible = (decay_times >= tau_min).any(axis=0)
        result["beta_decay_eligible_modes"] = (
            float(candidate.spans[eligible].max()) if eligible.any() else None
        )
        result["modes"] = [
            {
                "mode_id": i + 1,
                "f_initial_hz": float(f[0]),
                "f_min_hz": float(f.min()),
                "f_max_hz": float(f.max()),
                "beta": float(beta),
                "decay_eligible_at_start": bool(decay_times[0, i] >= tau_min),
                "decay_eligible_sample_fraction": float(
                    np.mean(decay_times[:, i] >= tau_min)
                ),
            }
            for i, (f, beta) in enumerate(
                zip(frequencies.T, candidate.spans, strict=True)
            )
        ]
    return result


def rank_untracked(candidates: list[Candidate]) -> list[Candidate]:
    """Rank by descending fixed ZVD component count, then candidate number."""
    return sorted(
        candidates,
        key=lambda candidate: (-len(candidate.fixed_zvd_frequencies), candidate.number),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument(
        "--modal-stride",
        type=int,
        default=1,
        help="initial modal sample stride; qualifying paths are rechecked at full resolution",
    )
    parser.add_argument("--batch-config", type=Path, default=DEFAULT_BATCH_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidates < 1:
        parser.error("--candidates must be positive")
    if args.modal_stride < 1:
        parser.error("--modal-stride must be positive")

    batch_bytes = args.batch_config.read_bytes()
    batch = yaml.safe_load(batch_bytes)
    robot_path = args.batch_config.parent / batch["robot_config"]
    config = load_robot_config(robot_path)
    # Screen against the batch thresholds even if ZVD is disabled in the runner.
    zvd = batch["input_shaping"]["time_invariant"]["zvd"]
    tau_min = float(zvd["tau_min_s"])
    utilization_min = float(zvd["utilization_min"])
    model = load_model(config.passive_damping)
    model.opt.timestep = config.simulation.dt_s
    limits = np.array([model.joint(name).range for name in LEFT_ARM_JOINT_NAMES])
    rng = np.random.default_rng(SEED)
    counts: Counter[str] = Counter()
    groups: dict[str, list[Candidate]] = {name: [] for name in BANDS}
    untracked: list[Candidate] = []
    modal_rechecks = 0

    def status() -> str:
        totals = "/".join(str(counts[name]) for name in (*BANDS, UNTRACKED_BAND))
        return f"[{totals}]"

    # Screen the configured comparison path before the seeded search.
    for number in range(args.candidates + 1):
        if number == 0 and np.any(config.waypoints[0] != 0):
            counts["nonzero_start"] += 1
            print(f"{status()} 0 (configured path): nonzero_start", flush=True)
            continue
        waypoints = (
            config.waypoints
            if number == 0
            else generate_waypoints(rng, limits, number, config.waypoints)
        )
        stride = 1 if number == 0 else args.modal_stride
        # Reuse the coarse eigensolves, but redo mode tracking on the full grid.
        cache: ModalCache | None = {} if stride > 1 else None
        result = evaluate(
            model,
            config,
            limits,
            number,
            waypoints,
            tau_min,
            utilization_min,
            stride,
            cache,
        )
        key = (
            result
            if isinstance(result, str)
            else screen_band(result.beta, len(result.fixed_zvd_frequencies))
        )
        if stride > 1 and (key in groups or key == UNTRACKED_BAND):
            modal_rechecks += 1
            result = evaluate(
                model,
                config,
                limits,
                number,
                waypoints,
                tau_min,
                utilization_min,
                cache=cache,
            )
            key = (
                result
                if isinstance(result, str)
                else screen_band(result.beta, len(result.fixed_zvd_frequencies))
            )
        counts[key] += 1
        label = "0 (configured path)" if number == 0 else str(number)
        if isinstance(result, str):
            print(f"{status()} {label}: {key}", flush=True)
            continue
        if key == UNTRACKED_BAND:
            untracked.append(result)
        elif key in groups:
            groups[key].append(result)
        beta = result.beta
        beta_text = f"{beta:.4f}" if beta is not None else "unavailable"
        reason = f", {result.tracking_failure}" if result.tracking_failure else ""
        print(
            f"{status()} {label}: {key} (beta={beta_text}, ZVD components={len(result.fixed_zvd_frequencies)},"
            f" {result.duration:.3f} s, {result.travel:.3f} rad{reason})",
            flush=True,
        )

    selected_bands = {group: select_band(groups[group]) for group in BANDS}
    selected_counts = {group: len(paths) for group, paths in selected_bands.items()}
    ranked_untracked = [
        entry(candidate, UNTRACKED_BAND, None, tau_min)
        for candidate in rank_untracked(untracked)
    ]
    runnable_untracked = ranked_untracked[:MAX_RUNNABLE_UNTRACKED]
    manifest = {
        "scope": "Sampled unshaped reference only; not dynamic simulation or hardware clearance.",
        "seed": SEED,
        "model_revision": MODEL_REVISION,
        "model_xml_sha256": sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "robot_config_sha256": sha256(robot_path.read_bytes()).hexdigest(),
        "batch_config_sha256": sha256(batch_bytes).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "joint_order": list(LEFT_ARM_JOINT_NAMES),
        "dt_s": config.simulation.dt_s,
        "initial_modal_stride": args.modal_stride,
        "full_resolution_modal_rechecks": modal_rechecks,
        "motor_kp": config.motor_kp.tolist(),
        "motor_kd": config.motor_kd.tolist(),
        "velocity_limits_rad_s": (LIMIT_SCALE * config.velocity_limits).tolist(),
        "acceleration_limits_rad_s2": (
            LIMIT_SCALE * config.acceleration_limits
        ).tolist(),
        "inset_rad": MARGIN_RAD,
        "initial_inset_exception": "Until each initially out-of-inset joint first enters.",
        "contact_policy": (
            "Existing MuJoCo collision filters; penetrating pairs involving the left arm "
            "disallowed except the existing closed left gripper exception."
        ),
        "beta_definition": (
            "Maximum relative natural-frequency span of the two lowest-frequency "
            "oscillatory modes at the initial pose, tracked by mode shape."
        ),
        "bands": {
            "narrow": "beta < 0.40",
            "medium": "0.40 <= beta < 0.75",
            "wide": "beta >= 0.75",
            UNTRACKED_BAND: "Stable path with incomplete mode tracking; ranked separately.",
        },
        "fixed_zvd_screen": {
            "tau_min_s": tau_min,
            "utilization_min": utilization_min,
            "frequency_tolerance": ZVD_FREQUENCY_TOLERANCE,
            "minimum_components_medium_and_wide": MIN_FIXED_ZVD_COMPONENTS,
            "minimum_components_untracked": MIN_UNTRACKED_ZVD_COMPONENTS,
        },
        "minimum_allowed_mode_match_mac": MIN_MAC,
        "minimum_allowed_assignment_score_gap": MIN_ASSIGNMENT_GAP,
        "band_selection": "Observed beta endpoints and nearest distinct interior targets.",
        "maximum_trajectories_per_band": MAX_PER_BAND,
        "candidate_count": args.candidates + 1,
        "configured_path_candidate_number": 0,
        "candidate_outcomes": dict(counts),
        "selected_band_counts": selected_counts,
        "untracked_benchmark_count": len(untracked),
        "runnable_untracked_benchmark_count": len(runnable_untracked),
        "trajectories": [
            entry(candidate, group, i, tau_min)
            for group, paths in selected_bands.items()
            for i, candidate in enumerate(paths, start=1)
        ]
        + runnable_untracked,
        "ranked_untracked_benchmarks": ranked_untracked,
    }
    args.output.write_text(
        yaml.safe_dump(manifest, sort_keys=False, width=100), encoding="utf-8"
    )
    print(f"{status()} Selected by band: {selected_counts}; outcomes: {dict(counts)}.")
    print(
        f"{status()} Archived {len(untracked)} untracked benchmarks; "
        f"selected {len(runnable_untracked)} for the batch."
    )


if __name__ == "__main__":
    main()
