from typing import Literal, NotRequired, TypedDict

import numpy as np

type FloatVector = np.ndarray[tuple[int], np.dtype[np.float64]]
type FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type ComplexVector = np.ndarray[tuple[int], np.dtype[np.complex128]]


class ModalSample(TypedDict):
    natural_frequency_hz: FloatVector
    damping_ratio: FloatVector


class ModalProperties(ModalSample):
    eigenvalues: ComplexVector | FloatVector
    damped_frequency_hz: FloatVector
    real_eigenvalues: FloatVector
    stable: bool


class JointTrajectory(TypedDict):
    t: FloatVector
    q: FloatMatrix
    dq: FloatMatrix
    ddq: FloatMatrix


class ContactEvent(TypedDict):
    kind: Literal["penetrating_contact"]
    bodies: list[str]
    geoms: list[str]
    first_time_s: float
    first_separation_m: float
    worst_time_s: float
    minimum_separation_m: float
    sample_count: int


class LimitViolation(TypedDict):
    source: NotRequired[Literal["reference", "measured"]]
    kind: Literal[
        "position_limit", "torque_clipping", "velocity_limit", "acceleration_limit",
        "settling_position_error", "settling_velocity",
    ]
    joint: str
    first_time_s: float
    first_value: float
    worst_time_s: float
    worst_value: float
    lower_limit: float
    upper_limit: float
    units: str
    sample_count: int


class NonfiniteViolation(TypedDict):
    source: NotRequired[Literal["reference", "measured"]]
    kind: Literal[
        "nonfinite_position_limit", "nonfinite_torque_clipping",
        "nonfinite_velocity_limit", "nonfinite_acceleration_limit",
        "nonfinite_settling_position_error", "nonfinite_settling_velocity",
    ]
    joint: str
    first_time_s: float
    sample_count: int


class RunIssue(TypedDict):
    kind: Literal["incomplete_run", "empty_recording", "insufficient_settling_window"]
    message: str


type SimulationIssue = ContactEvent | LimitViolation | NonfiniteViolation | RunIssue


class SettlingAssessment(TypedDict):
    evaluated: bool
    passing: bool
    command_end_time_s: float
    window_start_s: float
    window_end_s: float
    position_tolerance_rad: float
    velocity_tolerance_rad_s: float
    target_q_rad: list[float]
    final_position_error_rad: list[float]
    final_velocity_rad_s: list[float]
    max_abs_position_error_rad: list[float] | None
    max_abs_velocity_rad_s: list[float] | None


class AccelerationDiagnostics(TypedDict):
    peak_abs_rad_s2: list[float | None]
    reference_limits_rad_s2: list[float]
    exceedances: list[SimulationIssue]


class CheckCategories(TypedDict):
    reference_limits: bool
    position_limits: bool
    measured_velocity_limits: bool
    torque_clipping: bool
    collision_checks: bool
    settling: bool
    data_validity: bool


class SimulationChecks(TypedDict):
    passing: bool
    scope: str
    reasons: list[SimulationIssue]
    informational_contacts: list[ContactEvent]
    settling: SettlingAssessment
    measured_acceleration: AccelerationDiagnostics
    categories: CheckCategories
    notice: str


class TrialSamples(TypedDict):
    t: FloatVector
    q: FloatMatrix
    dq_measured: FloatMatrix
    ddq_measured: FloatMatrix
    torque_requested: FloatMatrix
    completed: bool
    final_time_s: float
    final_q: FloatVector
    final_dq: FloatVector
    final_ddq: FloatVector
    contact_events: list[ContactEvent]


class TrialRecording(TrialSamples):
    torque_feedback: FloatMatrix
    q_command: FloatMatrix
    dq_reference: FloatMatrix
    ddq_reference: FloatMatrix
    simulation_checks: SimulationChecks


class GitState(TypedDict):
    commit_sha: str | None
    dirty: bool | None
    errors: NotRequired[dict[str, str]]


class RunProvenance(TypedDict):
    captured_at_utc: str
    project_git: GitState
    model_path: str
    model_sha256: str
    lockfile_source: str
    lockfile_sha256: str
    lockfile_bytes: bytes


class ShaperDesign(TypedDict):
    type: Literal["zvd"]
    natural_frequencies_hz: list[float]
    damping_ratios: list[float]
    band_utilization: list[float]
    amplitudes: list[float]
    ideal_delays_s: list[float]
    rounded_delay_steps: list[int]
    applied_delays_s: list[float]
