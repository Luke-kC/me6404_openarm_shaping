"""Run one robot simulation comparison per waypoint path in a suite."""

import argparse
import csv
import re
from contextlib import ExitStack
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import yaml

from me6404_openarm_shaping.configuration import (
    RobotConfig,
    experiment_fingerprint,
    parse_robot_config,
)
from me6404_openarm_shaping.robot_model import (
    LEFT_ARM_JOINT_NAMES,
    MODEL_REVISION,
    load_model,
)
from me6404_openarm_shaping.robot_sim import simulate
from me6404_openarm_shaping.simulation_helpers import (
    capture_run_provenance,
    save_experiment,
    save_joint_position_plot,
)

DEFAULT_BATCH_CONFIG = Path(__file__).resolve().parents[2] / "config" / "batch_sim.yaml"


def load_batch(
    path: Path,
) -> tuple[Path, dict[str, Any], list[tuple[dict[str, Any], RobotConfig]]]:
    batch = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = yaml.safe_load(
        (path.parent / batch["robot_config"]).read_text(encoding="utf-8")
    )
    suite = yaml.safe_load(
        (path.parent / batch["trajectory_suite"]).read_text(encoding="utf-8")
    )
    if suite.get("joint_order", list(LEFT_ARM_JOINT_NAMES)) != list(
        LEFT_ARM_JOINT_NAMES
    ):
        raise ValueError("Trajectory suite joint order must match left J1–J7.")

    # Replace the entire shaping block, never merge it with single-run settings.
    base["input_shaping"] = batch["input_shaping"]
    jobs = []
    ids: set[str] = set()
    for entry in suite["trajectories"]:
        name = entry["id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name) or name in ids:
            raise ValueError(f"Invalid or duplicate trajectory ID: {name!r}")
        ids.add(name)
        jobs.append(
            (entry, parse_robot_config({**base, "waypoints": entry["waypoints"]}))
        )
    if not jobs:
        raise ValueError("Trajectory suite is empty.")
    return (path.parent / batch["output_dir"]).resolve(), suite, jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BATCH_CONFIG)
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="save per-trajectory CSVs, metadata, and summary.csv in a new batch folder",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="save one measured joint-position grid PNG per trajectory",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="skip the MuJoCo viewer and real-time pacing; no interactive plots",
    )
    args = parser.parse_args()
    output_dir, suite, jobs = load_batch(args.config)
    model = load_model()
    provenance = capture_run_provenance() if args.save_csv else None
    batch_dir = None
    summary_stream = None
    with ExitStack() as stack:
        if args.save_csv or args.save_plots:
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_dir = Path(mkdtemp(prefix="batch_", dir=output_dir))
            print(f"Saving batch results to {batch_dir}")
            (batch_dir / "batch_sim.yaml").write_bytes(args.config.read_bytes())
            (batch_dir / "trajectory_suite.yaml").write_text(
                yaml.safe_dump(suite, sort_keys=False), encoding="utf-8"
            )
        if batch_dir is not None and args.save_csv:
            summary_stream = stack.enter_context(
                (batch_dir / "summary.csv").open("x", newline="", encoding="utf-8")
            )
            csv.writer(summary_stream).writerow(
                [
                    "trajectory_id",
                    "band",
                    "triplet_id",
                    "beta",
                    "trial",
                    "completed",
                    "passing_sim",
                    "samples",
                    "command_duration_s",
                    "results_csv",
                    "metadata_yaml",
                ]
            )
        for index, (entry, config) in enumerate(jobs, start=1):
            name = entry["id"]
            print(f"\n[{index}/{len(jobs)}] {name}")
            result = simulate(model, config, headless=args.headless)
            if (
                batch_dir is not None
                and provenance is not None
                and summary_stream is not None
            ):
                run_dir = save_experiment(
                    result.recordings,
                    config,
                    result.shaper_designs,
                    result.duration_s,
                    batch_dir,
                    provenance=provenance,
                    configuration_fingerprint=experiment_fingerprint(
                        config, provenance["model_sha256"], MODEL_REVISION
                    ),
                    tvist_schedules=result.tvist_schedules,
                    tvist_selection=result.tvist_selection,
                    directory_name=name,
                )
                summary = csv.writer(summary_stream)
                for label, trial in result.recordings.items():
                    summary.writerow(
                        [
                            name,
                            entry.get("band"),
                            entry.get("triplet_id"),
                            entry.get("beta"),
                            label,
                            trial["completed"],
                            trial["simulation_checks"]["passing"],
                            len(trial["t"]),
                            result.trajectories[label]["t"][-1],
                            run_dir.relative_to(batch_dir) / f"{name}_results.csv",
                            run_dir.relative_to(batch_dir) / f"{name}_metadata.yaml",
                        ]
                    )
                summary_stream.flush()
            if batch_dir is not None and args.save_plots:
                save_joint_position_plot(
                    result.recordings,
                    batch_dir / f"{name}_joint_positions.png",
                    trajectory_id=name,
                )
            if any(not trial["completed"] for trial in result.recordings.values()):
                print("Batch stopped early.")
                break
    if batch_dir is not None and args.save_csv:
        print(f"Batch summary: {batch_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
