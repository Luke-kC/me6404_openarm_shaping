"""Conservative sampled simulation checks, not hardware-safety authorization."""

from typing import Literal, cast

import mujoco
import numpy as np

from me6404_openarm_shaping.configuration import RobotConfig
from me6404_openarm_shaping.data_types import (
    AccelerationDiagnostics,
    CheckCategories,
    ContactEvent,
    FloatMatrix,
    FloatVector,
    JointTrajectory,
    SettlingAssessment,
    SimulationChecks,
    SimulationIssue,
    TrialSamples,
)
from me6404_openarm_shaping.robot_model import LEFT_ARM_JOINT_NAMES

# Assessment exception for the pinned model's closed left gripper only.
# Do not alter MuJoCo contact generation or exempt any other geometry pair.
EXPECTED_FINGER_CONTACT = frozenset(
    {
        "openarm_left_right_finger_collision",
        "openarm_left_left_finger_collision",
    }
)


def record_penetrating_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    events: dict[tuple[int, ...], ContactEvent],
    time_s: float,
) -> None:
    """Accumulate penetrating geom pairs involving a left-arm body.

    Honor existing MuJoCo collision filters/exclusions. Count each geom pair
    once per sampled state, taking its deepest contact point. No clearance
    margin, new exclusions, or penetration tolerance is applied.
    """
    frame: dict[tuple[int, ...], tuple[float, list[str]]] = {}
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        bodies = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
            or f"body_{model.geom_bodyid[geom]}"
            for geom in pair
        ]
        if not any(name.startswith("openarm_left_") for name in bodies):
            continue
        if pair not in frame or contact.dist < frame[pair][0]:
            frame[pair] = (float(contact.dist), bodies)
    for pair, (separation, bodies) in frame.items():
        if pair not in events:
            events[pair] = {
                "kind": "penetrating_contact",
                "bodies": bodies,
                "geoms": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                    or f"geom_{geom}"
                    for geom in pair
                ],
                "first_time_s": float(time_s),
                "first_separation_m": separation,
                "worst_time_s": float(time_s),
                "minimum_separation_m": separation,
                "sample_count": 0,
            }
        event = events[pair]
        event["sample_count"] += 1
        if separation < event["minimum_separation_m"]:
            event["minimum_separation_m"] = separation
            event["worst_time_s"] = float(time_s)


def _limit_reasons(
    values: FloatMatrix,
    times: FloatVector,
    limits: FloatMatrix,
    kind: Literal[
        "position_limit",
        "torque_clipping",
        "velocity_limit",
        "acceleration_limit",
        "settling_position_error",
        "settling_velocity",
    ],
    units: str,
) -> list[SimulationIssue]:
    reasons: list[SimulationIssue] = []
    for joint, name in enumerate(LEFT_ARM_JOINT_NAMES):
        column = values[:, joint]
        finite = np.isfinite(column)
        invalid = np.flatnonzero(~finite)
        if len(invalid):
            nonfinite_kind: Literal[
                "nonfinite_position_limit",
                "nonfinite_torque_clipping",
                "nonfinite_velocity_limit",
                "nonfinite_acceleration_limit",
                "nonfinite_settling_position_error",
                "nonfinite_settling_velocity",
            ]
            match kind:
                case "position_limit":
                    nonfinite_kind = "nonfinite_position_limit"
                case "torque_clipping":
                    nonfinite_kind = "nonfinite_torque_clipping"
                case "velocity_limit":
                    nonfinite_kind = "nonfinite_velocity_limit"
                case "acceleration_limit":
                    nonfinite_kind = "nonfinite_acceleration_limit"
                case "settling_position_error":
                    nonfinite_kind = "nonfinite_settling_position_error"
                case "settling_velocity":
                    nonfinite_kind = "nonfinite_settling_velocity"
            reasons.append(
                {
                    "kind": nonfinite_kind,
                    "joint": name,
                    "first_time_s": float(times[invalid[0]]),
                    "sample_count": len(invalid),
                }
            )
        lower, upper = limits[joint]
        excess = np.maximum(lower - column, column - upper)
        violations = np.flatnonzero(finite & (excess > 0))
        if not len(violations):
            continue
        first = violations[0]
        worst = violations[np.argmax(excess[violations])]
        reasons.append(
            {
                "kind": kind,
                "joint": name,
                "first_time_s": float(times[first]),
                "first_value": float(column[first]),
                "worst_time_s": float(times[worst]),
                "worst_value": float(column[worst]),
                "lower_limit": float(lower),
                "upper_limit": float(upper),
                "units": units,
                "sample_count": len(violations),
            }
        )
    return reasons


def _assess_settling(
    result: TrialSamples,
    config: RobotConfig,
    command_end_time_s: float,
    times: FloatVector,
    positions: FloatMatrix,
    velocities: FloatMatrix,
) -> tuple[SettlingAssessment, list[SimulationIssue]]:
    settings = config.simulation_assessment
    end = result["final_time_s"]
    start = end - settings.settling_window_s
    target = cast(FloatVector, config.waypoints[-1])
    final_error: FloatVector = result["final_q"] - target
    summary: SettlingAssessment = {
        "evaluated": False,
        "passing": False,
        "command_end_time_s": command_end_time_s,
        "window_start_s": start,
        "window_end_s": end,
        "position_tolerance_rad": settings.position_tolerance_rad,
        "velocity_tolerance_rad_s": settings.velocity_tolerance_rad_s,
        "target_q_rad": target.tolist(),
        "final_position_error_rad": final_error.tolist(),
        "final_velocity_rad_s": result["final_dq"].tolist(),
        "max_abs_position_error_rad": None,
        "max_abs_velocity_rad_s": None,
    }
    # Incomplete/empty runs already have an explicit failure reason.
    if not result["completed"] or not len(result["t"]):
        return summary, []
    # Allow only floating-point timestamp roundoff, not missing physical time.
    time_slack = (
        16 * np.finfo(np.float64).eps * max(1.0, abs(end), abs(command_end_time_s))
    )
    if start < command_end_time_s - time_slack or start < times[0] - time_slack:
        return summary, [
            {
                "kind": "insufficient_settling_window",
                "message": (
                    f"Need {settings.settling_window_s:g} s after command end "
                    f"at {command_end_time_s:g} s; run ended at {end:g} s."
                ),
            }
        ]
    window = times >= start - time_slack
    errors: FloatMatrix = positions[window] - target
    speeds = velocities[window]
    position_limits = np.full((7, 2), settings.position_tolerance_rad)
    position_limits[:, 0] *= -1
    velocity_limits = np.full((7, 2), settings.velocity_tolerance_rad_s)
    velocity_limits[:, 0] *= -1
    reasons = _limit_reasons(
        errors, times[window], position_limits, "settling_position_error", "rad"
    )
    reasons.extend(
        _limit_reasons(
            speeds, times[window], velocity_limits, "settling_velocity", "rad/s"
        )
    )
    summary["evaluated"] = True
    summary["passing"] = not reasons
    # Nonfinite histories fail above; avoid writing NaN/Inf as useful metrics.
    if np.isfinite(errors).all():
        summary["max_abs_position_error_rad"] = np.max(np.abs(errors), axis=0).tolist()
    if np.isfinite(speeds).all():
        summary["max_abs_velocity_rad_s"] = np.max(np.abs(speeds), axis=0).tolist()
    return summary, reasons


def assess_trial(
    model: mujoco.MjModel,
    result: TrialSamples,
    config: RobotConfig,
    *,
    command_end_time_s: float,
    trajectory: JointTrajectory,
) -> SimulationChecks:
    """Assess recorded samples plus the final integrated arm configuration.

    Any requested torque outside its motor force range counts as clipping.
    Reference position/velocity/acceleration and measured position/velocity
    limits are hard checks. Finite measured acceleration exceedances are only
    diagnostic. Contacts must be collected during simulation.
    """
    reasons: list[SimulationIssue] = []
    if not result["completed"]:
        reasons.append(
            {
                "kind": "incomplete_run",
                "message": "Run ended before all steps completed.",
            }
        )
    if not len(result["t"]):
        reasons.append(
            {"kind": "empty_recording", "message": "No simulation steps were recorded."}
        )

    joint_limits = np.array([model.joint(name).range for name in LEFT_ARM_JOINT_NAMES])
    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_joint{i}_ctrl")
        for i in range(1, 8)
    ]
    torque_limits = model.actuator_forcerange[actuator_ids]
    position_times = np.append(result["t"], result["final_time_s"])
    positions = np.vstack((result["q"], result["final_q"]))
    velocities = np.vstack((result["dq_measured"], result["final_dq"]))
    accelerations = np.vstack((result["ddq_measured"], result["final_ddq"]))
    reasons.extend(
        _limit_reasons(positions, position_times, joint_limits, "position_limit", "rad")
    )
    reasons.extend(
        _limit_reasons(
            velocities,
            position_times,
            np.column_stack((-config.velocity_limits, config.velocity_limits)),
            "velocity_limit",
            "rad/s",
        )
    )
    acceleration_issues = _limit_reasons(
        accelerations,
        position_times,
        np.column_stack((-config.acceleration_limits, config.acceleration_limits)),
        "acceleration_limit",
        "rad/s^2",
    )
    # Invalid numerical data remains a failure, even for diagnostic quantities.
    reasons.extend(
        issue
        for issue in acceleration_issues
        if issue["kind"] == "nonfinite_acceleration_limit"
    )
    peaks: list[float | None] = []
    for joint in range(7):
        column = cast(FloatVector, accelerations[:, joint])
        peaks.append(
            float(cast(np.float64, np.max(np.abs(column))))
            if np.isfinite(column).all()
            else None
        )
    acceleration_diagnostics: AccelerationDiagnostics = {
        "peak_abs_rad_s2": peaks,
        "reference_limits_rad_s2": config.acceleration_limits.tolist(),
        "exceedances": [
            issue
            for issue in acceleration_issues
            if issue["kind"] == "acceleration_limit"
        ],
    }
    settling, settling_reasons = _assess_settling(
        result, config, command_end_time_s, position_times, positions, velocities
    )
    reasons.extend(settling_reasons)
    reasons.extend(
        _limit_reasons(
            result["torque_requested"],
            result["t"],
            torque_limits,
            "torque_clipping",
            "N*m",
        )
    )
    informational_contacts: list[ContactEvent] = []
    for event in result["contact_events"]:
        if (
            len(event["geoms"]) == 2
            and frozenset(event["geoms"]) == EXPECTED_FINGER_CONTACT
        ):
            informational_contacts.append(event)
        else:
            reasons.append(event)
    measured_kinds = {issue["kind"] for issue in reasons}
    # Assess the complete reference, including portions absent from an
    # interrupted recording. Do not rely only on the trajectory generator.
    reference_reasons = _limit_reasons(
        trajectory["q"], trajectory["t"], joint_limits, "position_limit", "rad"
    )
    reference_reasons.extend(
        _limit_reasons(
            trajectory["dq"],
            trajectory["t"],
            np.column_stack((-config.velocity_limits, config.velocity_limits)),
            "velocity_limit",
            "rad/s",
        )
    )
    reference_reasons.extend(
        _limit_reasons(
            trajectory["ddq"],
            trajectory["t"],
            np.column_stack((-config.acceleration_limits, config.acceleration_limits)),
            "acceleration_limit",
            "rad/s^2",
        )
    )
    for issue in reference_reasons:
        if "joint" in issue:
            issue["source"] = "reference"
    reasons.extend(reference_reasons)
    categories: CheckCategories = {
        "reference_limits": not reference_reasons,
        "position_limits": not measured_kinds.intersection(
            {
                "position_limit",
                "nonfinite_position_limit",
            }
        ),
        "measured_velocity_limits": not measured_kinds.intersection(
            {
                "velocity_limit",
                "nonfinite_velocity_limit",
            }
        ),
        "torque_clipping": not measured_kinds.intersection(
            {
                "torque_clipping",
                "nonfinite_torque_clipping",
            }
        ),
        "collision_checks": "penetrating_contact" not in measured_kinds,
        "settling": settling["passing"],
        "data_validity": not any(
            issue["kind"].startswith("nonfinite_")
            or issue["kind"] in {"incomplete_run", "empty_recording"}
            for issue in reasons
        ),
    }
    return {
        "passing": not reasons,
        "scope": (
            "Full reference motion limits; sampled measured position/velocity, torque clipping, "
            "settling, and penetrating contacts. Measured acceleration exceedances and the "
            "named closed left finger pair are informational only."
        ),
        "reasons": reasons,
        "informational_contacts": informational_contacts,
        "settling": settling,
        "measured_acceleration": acceleration_diagnostics,
        "categories": categories,
        "notice": "Not a hardware-safety authorization; no inter-sample clearance guarantee.",
    }


def print_simulation_checks(label: str, report: SimulationChecks) -> None:
    status = "PASSING SIM" if report["passing"] else "NOT PASSING SIM"
    print(f"\n{label}: {status} — selected safety checks")
    print("\nMeasured acceleration diagnostics (informational):")
    for joint, peak in enumerate(
        report["measured_acceleration"]["peak_abs_rad_s2"], start=1
    ):
        value = (
            f"{peak:.2f} rad/s²" if peak is not None else "unavailable (nonfinite data)"
        )
        print(f"  J{joint} peak: {value}")
    print()
    titles = {
        "reference_limits": "Reference limits",
        "position_limits": "Position limits",
        "measured_velocity_limits": "Measured velocity limits",
        "torque_clipping": "Torque clipping",
        "collision_checks": "Collision checks",
        "settling": "Settling",
        "data_validity": "Data validity",
    }
    for key, passing in report["categories"].items():
        outcome = "PASS" if passing else "FAIL"
        print(f"{titles[key]}: {outcome}")
    if report["measured_acceleration"]["exceedances"]:
        print("\nMeasured acceleration reference-limit exceedances (informational):")
        _print_issues(report["measured_acceleration"]["exceedances"])
    _print_issues(report["reasons"])
    for contact in report["informational_contacts"]:
        print()
        print("  Expected finger contact (informational; not a failure):")
        print("    Geometries:")
        for geom in contact["geoms"]:
            print(f"      {geom}")
        print(f"    First contact: {contact['first_time_s']:.6f} s")
        print(f"    Worst time: {contact['worst_time_s']:.6f} s")
        print(f"    Minimum separation: {contact['minimum_separation_m']:.6g} m")
        print(f"    Sampled states: {contact['sample_count']}")
    print()
    print(report["notice"])
    print()


def _print_issues(issues: list[SimulationIssue]) -> None:
    for reason in issues:
        print()
        if reason["kind"] == "penetrating_contact":
            print("  Collision:")
            print(f"    Bodies: {' / '.join(reason['bodies'])}")
            print("    Geometries:")
            for geom in reason["geoms"]:
                print(f"      {geom}")
            print(f"    First contact: {reason['first_time_s']:.6f} s")
            print(f"    Worst time: {reason['worst_time_s']:.6f} s")
            print(f"    Minimum separation: {reason['minimum_separation_m']:.6g} m")
            print(f"    Sampled states: {reason['sample_count']}")
        elif "lower_limit" in reason:
            prefix = "Reference " if reason.get("source") == "reference" else ""
            print(f"  {prefix}{reason['joint']} {reason['kind']}:")
            print(
                f"    First: {reason['first_value']:.6g} {reason['units']} "
                f"at {reason['first_time_s']:.6f} s"
            )
            print(
                f"    Worst: {reason['worst_value']:.6g} {reason['units']} "
                f"at {reason['worst_time_s']:.6f} s"
            )
            print(
                f"    Allowed: [{reason['lower_limit']:.6g}, {reason['upper_limit']:.6g}] "
                f"{reason['units']}"
            )
            print(f"    Samples: {reason['sample_count']}")
        elif "joint" in reason:
            prefix = "Reference " if reason.get("source") == "reference" else ""
            print(f"  {prefix}{reason['joint']} {reason['kind']}:")
            print(f"    First occurrence: {reason['first_time_s']:.6f} s")
            print(f"    Samples: {reason['sample_count']}")
        else:
            print(f"  {reason['kind']}: {reason['message']}")
