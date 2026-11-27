"""Generate the calibrated P1–P3 command bundle without publishing to ROS."""

import json
import subprocess
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml

from me6404_openarm_shaping.configuration import (
    experiment_fingerprint,
    parse_robot_config,
)
from me6404_openarm_shaping.data_types import TrialRecording
from me6404_openarm_shaping.robot_model import MODEL_REVISION, load_model
from me6404_openarm_shaping.robot_sim import simulate
from me6404_openarm_shaping.simulation_helpers import (
    capture_run_provenance,
    save_experiment,
)
from me6404_openarm_shaping.tvist_selection import response_scores

_CASES = {"P1": "medium-09", "P2": "wide-07", "P3": "wide-02"}
_DESIGN_MODEL = "calibrated"
_GRAVITY_SCALE = 1.0
_CHECK_CSV = """
import json
import sys
from pathlib import Path
from runpy import run_path

script, csv_path, gravity_scale, *labels = sys.argv[1:]
load_trajectory = run_path(script)["load_trajectory"]
errors = {}
for label in labels:
    try:
        load_trajectory(Path(csv_path), float(gravity_scale), label)
    except ValueError as error:
        errors[label] = str(error)
print(json.dumps(errors))
"""


def _replay_errors(
    project_root: Path, csv_path: Path, labels: list[str]
) -> dict[str, str]:
    # Only import the publisher's offline loader, without invoking main() or ROS init.
    script = (
        project_root / "ros2_ws/src/me6404_bringup/scripts/publish_csv_trajectory.py"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source /opt/ros/jazzy/setup.bash && exec /usr/bin/python3 -c "$@"',
            "offline-replay-check",
            _CHECK_CSV,
            str(script),
            str(csv_path),
            str(_GRAVITY_SCALE),
            *labels,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Offline ROS Jazzy CSV check failed: {result.stderr.strip()}"
        )
    return cast(dict[str, str], json.loads(result.stdout))


def build_focused_trial_export(
    project_root: Path, calibrated_sweep_dir: Path
) -> tuple[bytes, list[dict[str, str | float | None]]]:
    """Return a ZIP containing accepted trials and a manifest of all selections."""
    provenance = capture_run_provenance(project_root)
    summary: list[dict[str, str | float | None]] = []
    output = BytesIO()
    with (
        TemporaryDirectory(prefix="focused-trials-") as temp,
        ZipFile(output, "w", compression=ZIP_DEFLATED) as bundle,
    ):
        staging = Path(temp)
        for priority, path in _CASES.items():
            config_path = calibrated_sweep_dir / path / f"{path}_metadata.yaml"
            metadata = cast(dict[str, object], yaml.safe_load(config_path.read_text()))
            config = parse_robot_config(cast(dict[str, object], metadata["config"]))
            with redirect_stdout(StringIO()):
                result = simulate(load_model(), config, headless=True)

            scores = {
                label: response_scores(
                    recording,
                    float(cast(np.float64, result.trajectories[label]["t"][-1])),
                )
                for label, recording in result.recordings.items()
            }
            selections: dict[str, str | None] = {
                "Unshaped": "baseline",
                "Fixed ZVD": "time_invariant_zvd",
                "Fixed EI": "time_invariant_ei",
            }
            for method, objective in (
                ("TVIST vibration-focused", "min_residual"),
                ("TVIST time-focused", "beat_fixed_settling"),
            ):
                ranked = [
                    (rms, label)
                    for label, (rms, _) in scores.items()
                    if label.startswith("tvist_")
                    and label.endswith(f"_{objective}")
                    and rms is not None
                ]
                selections[method] = min(ranked)[1] if len(ranked) > 0 else None

            labels: dict[int, str] = {}
            rows: list[dict[str, str | float | None]] = []
            eligible: dict[str, TrialRecording] = {}
            for method, label in selections.items():
                row: dict[str, str | float | None] = {
                    "Priority": priority,
                    "Path": path,
                    "Method": method,
                    "CSV label": None,
                    "Status": "No qualifying candidate",
                    "Residual RMS [rad]": None,
                    "Total settling [s]": None,
                }
                rows.append(row)
                if label is None:
                    continue
                canonical = labels.setdefault(id(result.trajectories[label]), label)
                recording = result.recordings[canonical]
                rms, settling = scores[canonical]
                row["CSV label"] = canonical
                row["Residual RMS [rad]"] = rms
                row["Total settling [s]"] = settling
                checks = recording["simulation_checks"]
                if recording["completed"] and checks["passing"]:
                    eligible[canonical] = recording
                    row["Status"] = (
                        "Exported" if label == canonical else "Shared command"
                    )
                else:
                    reasons = sorted({issue["kind"] for issue in checks["reasons"]})
                    detail = (
                        ", ".join(reasons) if len(reasons) > 0 else "incomplete trial"
                    )
                    row["Status"] = f"MuJoCo blocked: {detail}"

            if len(eligible) > 0:
                fingerprint = experiment_fingerprint(
                    config, provenance["model_sha256"], MODEL_REVISION
                )
                staged_dir = save_experiment(
                    eligible,
                    config,
                    result.shaper_designs,
                    result.duration_s,
                    staging / "validation",
                    provenance=provenance,
                    configuration_fingerprint=fingerprint,
                    tvist_selection=result.tvist_selection,
                    directory_name=path,
                )
                errors = _replay_errors(
                    project_root, staged_dir / f"{path}_results.csv", list(eligible)
                )
                for row in rows:
                    label = row["CSV label"]
                    if isinstance(label, str) and label in errors:
                        row["Status"] = f"CSV loader blocked: {errors[label]}"
                accepted = {
                    label: recording
                    for label, recording in eligible.items()
                    if label not in errors
                }
                if len(accepted) > 0:
                    official_dir = save_experiment(
                        accepted,
                        config,
                        result.shaper_designs,
                        result.duration_s,
                        staging / priority,
                        provenance=provenance,
                        configuration_fingerprint=fingerprint,
                        tvist_schedules={
                            label: result.tvist_schedules[label]
                            for label in accepted
                            if label in result.tvist_schedules
                        },
                        tvist_selection=result.tvist_selection,
                        directory_name=path,
                    )
                    metadata_path = official_dir / f"{path}_metadata.yaml"
                    exported_metadata = cast(
                        dict[str, object], yaml.safe_load(metadata_path.read_text())
                    )
                    exported_metadata["design_model"] = _DESIGN_MODEL
                    _ = metadata_path.write_text(
                        yaml.safe_dump(exported_metadata, sort_keys=False)
                    )
                    for file in sorted(official_dir.iterdir()):
                        bundle.write(file, file.relative_to(staging))
            summary.extend(rows)
            print(f"{priority} {path}: generation and offline checks complete")
        bundle.writestr(
            "selection.yaml",
            yaml.safe_dump(
                {
                    "design_model": _DESIGN_MODEL,
                    "gravity_scale_checked": _GRAVITY_SCALE,
                    "notice": "Offline validation only; separate authorization is required for physical trials.",
                    "trials": summary,
                },
                sort_keys=False,
            ),
        )
    return output.getvalue(), summary
