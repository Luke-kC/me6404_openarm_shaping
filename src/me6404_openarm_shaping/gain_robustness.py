"""Run all gain perturbations for one offline MuJoCo trajectory."""

from contextlib import redirect_stdout
from io import StringIO
from math import isfinite
from typing import Final, cast

import numpy as np

from me6404_openarm_shaping.configuration import RobotConfig
from me6404_openarm_shaping.data_types import FloatVector
from me6404_openarm_shaping.robot_model import load_model
from me6404_openarm_shaping.robot_sim import simulate
from me6404_openarm_shaping.simulation_helpers import run_trial
from me6404_openarm_shaping.tvist_selection import response_scores

GAIN_OFFSETS: Final = (-10, -5, 0, 5, 10)
GAIN_MODES: Final = {
    "Kp only": (True, False),
    "Kd only": (False, True),
    "Kp and Kd together": (True, True),
}
_CRITERIA: Final = {
    "min_residual": "Min RMS",
    "min_settling": "Min settling",
    "beat_fixed_rms": "Beat fixed RMS",
    "beat_fixed_settling": "Beat fixed time",
}


def sweep_trajectory(
    path: str, config: RobotConfig
) -> dict[str, dict[str, list[float]]]:
    """Design once at 100% gains and return RMS by gain mode, method, and offset."""
    model = load_model()
    with redirect_stdout(StringIO()):
        design = simulate(model, config, headless=True)
    initial_position = cast(FloatVector, config.waypoints[0])

    labels = {
        "baseline": ["Unshaped"],
        "time_invariant_zvd": ["Fixed ZVD"],
        "time_invariant_ei": ["Fixed EI"],
    }
    for shaper in ("zv", "zvd"):
        same_command: dict[int, str] = {}
        for criterion, criterion_label in _CRITERIA.items():
            trial = f"tvist_{shaper}_{criterion}"
            if trial not in design.trajectories:
                continue
            command_id = id(design.trajectories[trial])
            first_trial = same_command.setdefault(command_id, trial)
            labels.setdefault(first_trial, []).append(criterion_label)

    results: dict[str, dict[str, list[float]]] = {mode: {} for mode in GAIN_MODES}
    for trial, command_labels in labels.items():
        label = " / ".join(command_labels)
        if trial.startswith("tvist_"):
            label = f"TVIST {trial.split('_')[1].upper()}: {label}"
        command = design.trajectories[trial]
        end = float(cast(np.float64, command["t"][-1]))
        center_rms, _ = response_scores(design.recordings[trial], end)
        assert center_rms is not None and isfinite(center_rms), (
            f"{path}, {label}, 0%: missing residual RMS"
        )
        if trial == "baseline":
            assert center_rms > 0, f"{path}, 0%: zero unshaped RMS"

        for mode, (vary_kp, vary_kd) in GAIN_MODES.items():
            rms_values: list[float] = []
            for offset in GAIN_OFFSETS:
                if offset == 0:
                    rms_values.append(center_rms)
                    continue
                scale = 1 + offset / 100
                recording = run_trial(
                    model,
                    command,
                    initial_position,
                    (scale if vary_kp else 1.0) * config.motor_kp,
                    (scale if vary_kd else 1.0) * config.motor_kd,
                    design.duration_s,
                    config=config,
                    headless=True,
                )
                rms, _ = response_scores(recording, end)
                assert rms is not None and isfinite(rms), (
                    f"{path}, {mode}, {label}, {offset:+d}%: missing residual RMS"
                )
                if trial == "baseline":
                    assert rms > 0, f"{path}, {mode}, {offset:+d}%: zero unshaped RMS"
                rms_values.append(rms)
            results[mode][label] = rms_values
    return results
