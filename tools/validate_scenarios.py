"""Run the three stage-four scenes repeatedly and audit their saved facts.

This is a headless functional validation. CPU/wall durations measure only the
simulation logic, never a graphical frame rate. Each invocation creates its
own batch directory; failed and incomplete records are retained for inspection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_app.models import BehaviorState, Point, RunState, Team, UnitType
from sim_app.navigation import point_clear, segment_clear
from sim_app.recording import export_run
from sim_app.replay import load_replay
from sim_app.scene import load_scene
from sim_app.simulation import Simulation


SCENARIO_FILES = ("basic_scene.json", "obstacle_scene.json", "sharing_scene.json")
EXPECTED_REASONS = {
    "stage4_basic": "missions_complete",
    "stage4_obstacle": "missions_complete",
    "stage4_sharing": "score_limit",
    "stage4_blocked": "missions_complete",
}
SUMMARY_FIELDS = (
    "scenario", "variant", "round", "status", "source_scene_sha256",
    "effective_scene_sha256", "state", "step_count", "sim_time_seconds",
    "finish_reason", "winner", "red_score", "blue_score", "blocked_count",
    "event_count", "snapshot_count", "replay_checked_frames", "reset_reproduced",
    "repeated_run_reproduced", "logged_info_shared_transitions", "shared_receiver_count",
    "unique_shared_receiver_target_pairs", "fresh_shared_contact_samples",
    "shared_contact_samples", "logic_cpu_seconds", "logic_wall_seconds",
    "logic_steps_per_wall_second", "run_json", "events_csv", "states_csv", "errors",
)


def canonical_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment() -> dict:
    try:
        pygame_version = importlib.metadata.version("pygame")
    except importlib.metadata.PackageNotFoundError:
        pygame_version = None
    return {
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "pygame_version": pygame_version,
        "measurement": "headless simulation logic CPU/wall duration; excludes export/replay; not GUI FPS",
        "randomness": "No stochastic behavior in these scenes; configuration and integer logic steps are fixed.",
    }


def advance_to_end(sim: Simulation, max_steps: int) -> dict:
    """Run a fresh/reset simulation with a bounded number of fixed logic steps."""
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    sim.start()
    for _ in range(max_steps):
        if sim.finished:
            break
        sim.advance(sim.scene.fixed_dt)
    cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    return {"logic_cpu_seconds": cpu, "logic_wall_seconds": wall,
            "logic_steps_per_wall_second": sim.step_count / wall if wall else 0.0}


def knowledge_metrics(snapshots: list[dict], events: list[dict], fixed_dt: float) -> dict:
    """Count saved knowledge records, receivers and sources, not network sends.

    A shared contact held for ten steps contributes ten contact samples. An
    info_shared event records a knowledge/source transition and is not a count
    of every broadcast. Control snapshots at one step count only once here.
    """
    frames_by_step = {frame["step"]: frame for frame in snapshots}
    direct_pairs, shared_pairs, triples, receiver_steps = set(), set(), set(), set()
    fresh_direct = fresh_shared = shared_samples = 0
    peak_total = 0
    sources: dict[str, set[str]] = {}
    first_shared: dict[str, int] = {}
    peak_per_receiver: dict[str, int] = {}
    for step, frame in sorted(frames_by_step.items()):
        total = 0
        for unit in frame["units"]:
            contacts = unit.get("contacts", [])
            total += len(contacts)
            shared_count = 0
            for contact in contacts:
                pair = (unit["id"], contact["target_id"])
                fresh = contact["observed_step"] == step
                if contact["shared"]:
                    shared_samples += 1
                    fresh_shared += int(fresh)
                    shared_pairs.add(pair)
                    triples.add((contact["source_id"], *pair))
                    receiver_steps.add((unit["id"], step))
                    sources.setdefault(unit["id"], set()).add(contact["source_id"])
                    first_shared.setdefault(unit["id"], step)
                    shared_count += 1
                else:
                    direct_pairs.add(pair)
                    fresh_direct += int(fresh)
            peak_per_receiver[unit["id"]] = max(peak_per_receiver.get(unit["id"], 0), shared_count)
        peak_total = max(peak_total, total)
    counts = Counter(event["kind"] for event in events)
    receiver_ids = sorted(sources)
    return {
        "event_counts": dict(sorted(counts.items())),
        "logged_info_shared_transitions": counts["info_shared"],
        "direct_observation_samples": fresh_direct,
        "fresh_shared_contact_samples": fresh_shared,
        "shared_contact_samples": shared_samples,
        "unique_direct_observer_target_pairs": len(direct_pairs),
        "unique_shared_receiver_target_pairs": len(shared_pairs),
        "shared_receiver_count": len(receiver_ids), "shared_receiver_ids": receiver_ids,
        "shared_sources_by_receiver": {unit_id: sorted(sources[unit_id]) for unit_id in receiver_ids},
        "source_receiver_target_triples": [list(item) for item in sorted(triples)],
        "first_shared_step_by_receiver": dict(sorted(first_shared.items())),
        "peak_shared_contacts_by_receiver": dict(sorted(peak_per_receiver.items())),
        "shared_knowledge_unit_steps": len(receiver_steps),
        "shared_knowledge_unit_seconds": len(receiver_steps) * fixed_dt,
        "peak_total_contact_count": peak_total,
        "definitions": {
            "logged_info_shared_transitions": "New shared knowledge or source transitions recorded in events; not broadcast count.",
            "fresh_shared_contact_samples": "Receiver-target-step records refreshed by that step's direct observation; not physical/network packets.",
            "shared_contact_samples": "Saved shared receiver-target knowledge at each distinct logical step, including TTL memory.",
            "shared_knowledge_unit_seconds": "Sum of each receiver's logical time holding at least one shared contact; unit-seconds.",
        },
    }


def geometry_valid(sim: Simulation) -> bool:
    specs = {spec.id: spec for spec in sim.scene.units}
    for frame in sim.snapshots:
        for unit in frame["units"]:
            point = Point(unit["x"], unit["y"])
            spec = specs[unit["id"]]
            if not point_clear(sim.scene, point, spec.unit_type):
                return False
    # A step can travel across a path corner. Joining only its two snapshots
    # would invent a diagonal cutting that corner. The actual motion trail
    # preserves every visited route vertex, so audit those complete segments.
    for unit in sim.units:
        for start, end in zip(unit.trail, unit.trail[1:]):
            if not segment_clear(sim.scene, start, end, unit.unit_type):
                return False
    return True


def knowledge_sources_valid(sim: Simulation) -> bool:
    specs = {spec.id: spec for spec in sim.scene.units}
    for frame in sim.snapshots:
        for row in frame["units"]:
            receiver = specs[row["id"]]
            for contact in row.get("contacts", []):
                source, target = specs[contact["source_id"]], specs[contact["target_id"]]
                if source.team != receiver.team or target.team == receiver.team:
                    return False
                if contact["shared"] == (source.id == receiver.id):
                    return False
    return True


def audit_record(sim: Simulation, paths: dict[str, Path]) -> dict:
    for key, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty {key} recording: {path}")
    payload = json.loads(paths["run"].read_text(encoding="utf-8"))
    live = sim.snapshot()
    if payload["result"] != live or payload["events"] != sim.events:
        raise ValueError("exported result/events disagree with the original simulation")
    # A rules scene samples its recording, so the exported run may carry one
    # extra frame: the live state appended when it is newer than the tail.
    recorded = copy.deepcopy(sim.snapshots)
    if not recorded or recorded[-1] != live:
        recorded.append(live)
    if payload["snapshots"] != recorded:
        raise ValueError("exported snapshots disagree with the original simulation")
    with paths["events"].open(encoding="utf-8-sig", newline="") as source:
        event_rows = list(csv.DictReader(source))
    with paths["states"].open(encoding="utf-8-sig", newline="") as source:
        state_rows = list(csv.DictReader(source))
    state_count = len(state_rows)
    if len(event_rows) != len(sim.events) or state_count != sum(len(frame["units"]) for frame in recorded):
        raise ValueError("CSV recording row counts disagree with JSON facts")
    # Stage-four game records have the six-column event format. Details stores
    # all fields beyond the five basic columns, preserving nested details.
    for saved, original in zip(event_rows, sim.events):
        for key in ("step", "time", "kind", "unit_id", "message"):
            value = original[key]
            if saved[key] != ("" if value is None else str(value)):
                raise ValueError(f"event CSV field {key} disagrees with original event")
        extras = {key: value for key, value in original.items()
                  if key not in {"step", "time", "kind", "unit_id", "message"}}
        if json.loads(saved["details"]) != extras:
            raise ValueError("event CSV details lost original extra fields")
    expected_rows = []
    for frame in recorded:
        for unit in frame["units"]:
            # The state CSV carries counts where the snapshot carries rows; the
            # nested contact and tagged-target lists stay in run.json only.
            counted = {key: value for key, value in unit.items()
                       if key not in ("contacts", "tagged_targets")}
            counted["contact_count"] = len(unit["contacts"])
            counted["shared_count"] = sum(row["shared"] is True for row in unit["contacts"])
            counted["tagged_targets_count"] = len(unit["tagged_targets"])
            expected_rows.append({
                "step": frame["step"], "time": frame["time"], "state": frame["state"], **counted,
                "score_red": frame["scores"]["red"], "score_blue": frame["scores"]["blue"],
                "sharing_enabled": frame["sharing_enabled"], "finish_reason": frame["finish_reason"],
                "winner": frame["winner"], "event_count": frame["event_count"],
            })
    for index, (saved, expected) in enumerate(zip(state_rows, expected_rows)):
        if set(saved) != set(expected):
            raise ValueError(f"state CSV row {index} has missing/unknown columns")
        for key, value in expected.items():
            if saved[key] != str(value):
                raise ValueError(f"state CSV field {key} disagrees with snapshot at row {index}")
    replay = load_replay(paths["run"])
    if replay.count != len(recorded):
        raise ValueError("replay frame count disagrees with the saved simulation")
    for index, expected in enumerate(recorded):
        if not replay.seek(index) or replay.current_snapshot != expected:
            raise ValueError(f"replay frame {index} disagrees with its saved snapshot")
        if replay.source_state.value != expected["state"]:
            raise ValueError(f"replay source state differs at frame {index}")
        if replay.events != sim.events[:expected["event_count"]]:
            raise ValueError(f"replay event boundary differs at frame {index}")
        if replay.scores != {Team(key): value for key, value in expected["scores"].items()}:
            raise ValueError(f"replay scores differ at frame {index}")
    if replay.events != sim.events or replay.result_snapshot != sim.snapshot():
        raise ValueError("replay final events/result disagree with the original run")
    return {"replay_checked_frames": replay.count, "event_csv_rows": len(event_rows),
            "state_csv_rows": state_count, "effective_scene_sha256": canonical_hash(payload["scene"])}


def signature(sim: Simulation) -> dict:
    return {"terminal_sha256": canonical_hash(sim.snapshot()),
            "snapshots_sha256": canonical_hash(sim.snapshots), "events_sha256": canonical_hash(sim.events)}


def normal_checks(sim: Simulation) -> dict[str, bool]:
    specs = sim.scene.units
    full_types = {(unit.team, unit.unit_type) for unit in specs} == {
        (team, unit_type) for team in Team for unit_type in UnitType
    }
    counts = Counter(event["kind"] for event in sim.events)
    return_expected = all(spec.return_home for spec in specs)
    return {
        "four_units_both_teams_and_types": len(specs) == 4 and full_types,
        "finished_within_limit": sim.finished,
        "expected_finish_reason": sim.finish_reason == EXPECTED_REASONS[sim.scene.name],
        "no_blocked_units": all(unit.behavior != BehaviorState.BLOCKED for unit in sim.units),
        "all_units_stopped": all(unit.behavior == BehaviorState.STOPPED for unit in sim.units),
        "complete_routes_and_return": not return_expected or all(
            unit.position == sim.scene.return_points[unit.team]
            and unit.waypoint_index == len(unit.waypoints) for unit in sim.units
        ),
        "observed_opponents": counts["object_discovered"] > 0,
        "valid_virtual_points_when_required": sim.scene.name == "stage4_obstacle" or counts["virtual_tag"] > 0,
        "movement_segments_valid": geometry_valid(sim),
        "knowledge_sources_team_local": knowledge_sources_valid(sim),
    }


def validate_one(scene, source_path: Path, directory: Path, round_index: int, max_steps: int,
                 expected_signature: dict | None = None, variant: str = "normal",
                 expected_blocked: bool = False) -> tuple[dict, dict, Simulation]:
    sim = Simulation(scene)
    timing = advance_to_end(sim, max_steps)
    paths = export_run(sim, directory, source_path)
    audit = audit_record(sim, paths)
    sig = signature(sim)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    sig.update(source_scene_sha256=source_hash, effective_scene_sha256=audit["effective_scene_sha256"])
    metrics = knowledge_metrics(sim.snapshots, sim.events, scene.fixed_dt)
    if expected_blocked:
        ground = [unit for unit in sim.units if unit.unit_type == UnitType.GROUND]
        air = [unit for unit in sim.units if unit.unit_type == UnitType.AIR]
        checks = {
            "finished_within_limit": sim.finished,
            "expected_finish_reason": sim.finish_reason == "missions_complete",
            "exactly_two_blocked_ground_units": len(ground) == 2 and all(
                unit.behavior == BehaviorState.BLOCKED and unit.reason and unit.distance_travelled == 0
                for unit in ground),
            "two_air_units_returned": len(air) == 2 and all(
                unit.behavior == BehaviorState.STOPPED and unit.position == scene.return_points[unit.team]
                for unit in air),
            "two_explicit_path_blocked_events": metrics["event_counts"].get("path_blocked", 0) == 2,
            "movement_segments_valid": geometry_valid(sim),
            "knowledge_sources_team_local": knowledge_sources_valid(sim),
        }
    else:
        checks = normal_checks(sim)
    checks["recording_and_every_replay_frame_match"] = True
    checks["repeated_run_reproduced"] = expected_signature is None or sig == expected_signature
    original = copy.deepcopy((sim.snapshot(), sim.events, sim.snapshots))
    checks["reset_restores_ready_state"] = sim.reset() and sim.snapshot() == Simulation(scene).snapshot()
    reset_timing = advance_to_end(sim, max_steps)
    checks["reset_reproduces_entire_run"] = (sim.snapshot(), sim.events, sim.snapshots) == original
    ok = all(checks.values())
    status = "EXPECTED_BLOCKED" if ok and expected_blocked else "PASS" if ok else "FAIL"
    row = {
        "scenario": scene.name, "variant": variant, "round": round_index, "status": status,
        "source_scene": str(source_path.resolve()),
        "source_scene_sha256": source_hash,
        "state": sim.state.value, "step_count": sim.step_count, "sim_time_seconds": sim.sim_time,
        "fixed_dt": scene.fixed_dt, "finish_reason": sim.finish_reason, "winner": sim.winner,
        "scores": {team.value: sim.scores[team] for team in Team},
        "red_score": sim.scores[Team.RED], "blue_score": sim.scores[Team.BLUE],
        "sharing_enabled": sim.sharing_enabled,
        "blocked_count": sum(unit.behavior == BehaviorState.BLOCKED for unit in sim.units),
        "expected_failure_example": expected_blocked,
        "blocked_reasons": {unit.id: unit.reason for unit in sim.units if unit.behavior == BehaviorState.BLOCKED},
        "event_count": len(sim.events), "snapshot_count": len(sim.snapshots),
        "reset_reproduced": checks["reset_reproduces_entire_run"],
        "repeated_run_reproduced": checks["repeated_run_reproduced"],
        "checks": checks, "errors": [name for name, passed in checks.items() if not passed],
        "unit_distance_travelled": {unit.id: unit.distance_travelled for unit in sim.units},
        "knowledge": metrics, **{key: value for key, value in metrics.items() if key in SUMMARY_FIELDS},
        "run_json": str(paths["run"].resolve()), "events_csv": str(paths["events"].resolve()),
        "states_csv": str(paths["states"].resolve()),
        "reset_logic_cpu_seconds": reset_timing["logic_cpu_seconds"],
        "reset_logic_wall_seconds": reset_timing["logic_wall_seconds"],
        **timing, **audit, **sig,
    }
    return row, sig, sim


def sharing_comparison(on: dict, off: dict, on_sim: Simulation, off_sim: Simulation) -> dict:
    on_payload = json.loads(Path(on["run_json"]).read_text(encoding="utf-8"))
    off_payload = json.loads(Path(off["run_json"]).read_text(encoding="utf-8"))
    expected_off = copy.deepcopy(on_payload["scene"])
    expected_off["rules"]["sharing_enabled"] = False
    common_step = min(on_sim.step_count, off_sim.step_count)
    on_frame = next(frame for frame in reversed(on_sim.snapshots) if frame["step"] == common_step)
    off_frame = next(frame for frame in reversed(off_sim.snapshots) if frame["step"] == common_step)
    checks = {
        "only_configured_sharing_flag_differs": expected_off == off_payload["scene"],
        "same_source_configuration_hash": on["source_scene_sha256"] == off["source_scene_sha256"],
        "on_has_shared_receivers": on["knowledge"]["shared_receiver_count"] > 0,
        "off_has_no_shared_receivers": off["knowledge"]["shared_receiver_count"] == 0,
        "on_has_shared_contact_samples": on["knowledge"]["shared_contact_samples"] > 0,
        "off_has_no_shared_contact_samples": off["knowledge"]["shared_contact_samples"] == 0,
        "on_has_shared_transition_logs": on["knowledge"]["logged_info_shared_transitions"] > 0,
        "off_has_no_shared_transition_logs": off["knowledge"]["logged_info_shared_transitions"] == 0,
        "knowledge_differs_at_matched_logical_step": [unit["contacts"] for unit in on_frame["units"]]
                                                      != [unit["contacts"] for unit in off_frame["units"]],
        "both_rounds_finish_normally": on["status"] == off["status"] == "PASS",
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "matched_logical_step": common_step, "matched_sim_time_seconds": common_step * on_sim.scene.fixed_dt,
            "on_record": on["run_json"], "off_record": off["run_json"],
            "on": {"steps": on["step_count"], "scores": on["scores"], "winner": on["winner"], "knowledge": on["knowledge"]},
            "off": {"steps": off["step_count"], "scores": off["scores"], "winner": off["winner"], "knowledge": off["knowledge"]},
            "interpretation": "Fixed fictional software rules and information flow only; no claim about real-world tactics or network performance."}


def _write_reports(directory: Path, report: dict) -> None:
    (directory / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (directory / "summary.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in report["records"]:
            values = dict(row)
            values["errors"] = json.dumps(row.get("errors", []), ensure_ascii=False)
            writer.writerow(values)


def validate_batch(output_dir: str | Path, repeat: int = 3, max_steps: int = 7200,
                   config_dir: str | Path | None = None) -> dict:
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise ValueError("repeat must be a positive integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    config_dir = Path(config_dir) if config_dir else PROJECT_ROOT / "configs"
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    directory = Path(output_dir).resolve() / f"batch_{stamp}_{uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    rows, failures, signatures = [], [], {}
    sharing_on = sharing_sim = None

    def execute(path, scene, round_index, variant="normal", expected_blocked=False):
        key = (scene.name, variant)
        record_dir = directory / scene.name / (f"round_{round_index:02d}" if variant == "normal" else variant)
        try:
            row, sig, sim = validate_one(scene, path, record_dir, round_index, max_steps,
                                         signatures.get(key), variant, expected_blocked)
            signatures.setdefault(key, sig)
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
            row = {"scenario": scene.name, "variant": variant, "round": round_index,
                   "status": "FAIL", "errors": [f"{type(error).__name__}: {error}"],
                   "source_scene": str(path.resolve()), "record_directory": str(record_dir)}
            sim = None
        rows.append(row)
        if row["status"] == "FAIL":
            failures.append({"scenario": scene.name, "variant": variant,
                             "round": round_index, "errors": row["errors"]})
        print(f"SCENARIO_{row['status']} name={scene.name} variant={variant} round={round_index} "
              f"steps={row.get('step_count', '?')} reason={row.get('finish_reason', '?')}", flush=True)
        return row, sim

    for filename in SCENARIO_FILES:
        path = config_dir / filename
        try:
            scene = load_scene(path)
        except (OSError, ValueError) as error:
            failures.append({"source_scene": str(path.resolve()), "errors": [str(error)]})
            continue
        for round_index in range(1, repeat + 1):
            row, sim = execute(path, scene, round_index)
            if scene.name == "stage4_sharing" and round_index == 1:
                sharing_on, sharing_sim = row, sim
    comparison = None
    if sharing_on is not None and sharing_sim is not None:
        path = config_dir / "sharing_scene.json"
        off_scene = replace(sharing_sim.scene, rules=replace(sharing_sim.scene.rules, sharing_enabled=False))
        off, off_sim = execute(path, off_scene, 1, variant="sharing_off")
        if off_sim is not None and sharing_on["status"] == off["status"] == "PASS":
            comparison = sharing_comparison(sharing_on, off, sharing_sim, off_sim)
            if comparison["status"] != "PASS":
                failures.append({"sharing_comparison": comparison["checks"]})
    if comparison is None:
        failures.append({"sharing_comparison": "missing complete sharing on/off pair"})
    try:
        path = config_dir / "blocked_scene.json"
        blocked_scene = load_scene(path)
        execute(path, blocked_scene, 1, variant="expected_blocked", expected_blocked=True)
    except (OSError, ValueError) as error:
        failures.append({"expected_blocked_example": str(error)})
    business = [row for row in rows if row["variant"] == "normal"]
    expected_count = len(SCENARIO_FILES) * repeat
    if len(business) != expected_count:
        failures.append({"business_records": f"expected {expected_count}, produced {len(business)}"})
    summary = {}
    for name in ("stage4_basic", "stage4_obstacle", "stage4_sharing"):
        selected = [row for row in business if row["scenario"] == name and row["status"] == "PASS"]
        if not selected:
            continue
        summary[name] = {
            "completed_runs": len(selected), "step_count": selected[0]["step_count"],
            "sim_time_seconds": selected[0]["sim_time_seconds"], "scores": selected[0]["scores"],
            "finish_reason": selected[0]["finish_reason"], "winner": selected[0]["winner"],
            "all_repeated_runs_reproduced": all(row["repeated_run_reproduced"] for row in selected),
            "all_reset_runs_reproduced": all(row["reset_reproduced"] for row in selected),
            "replay_checked_frames": sum(row["replay_checked_frames"] for row in selected),
            "mean_logic_cpu_seconds": statistics.mean(row["logic_cpu_seconds"] for row in selected),
            "mean_logic_wall_seconds": statistics.mean(row["logic_wall_seconds"] for row in selected),
            "min_logic_wall_seconds": min(row["logic_wall_seconds"] for row in selected),
            "max_logic_wall_seconds": max(row["logic_wall_seconds"] for row in selected),
        }
    report = {
        "format_version": 1, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "FAIL" if failures else "PASS", "directory": str(directory),
        "parameters": {"repeat": repeat, "max_steps": max_steps, "config_dir": str(config_dir.resolve())},
        "environment": environment(), "expected_business_records": expected_count,
        "completed_business_records": sum(row["status"] == "PASS" for row in business),
        "business_run_count": len(business), "extra_record_count": len(rows) - len(business),
        "saved_record_count": sum("run_json" in row for row in rows),
        "reset_rerun_count": sum(row.get("reset_reproduced", False) for row in rows),
        "records": rows, "summary_by_scenario": summary, "sharing_comparison": comparison,
        "failures": failures,
        "scope": "Three scenes repeated headlessly; reset reruns; every saved replay frame/events; sharing-off pair; expected blocked helper. Does not establish GUI stability, FPS or cross-machine compatibility.",
    }
    _write_reports(directory, report)
    return report


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "stage4_validation",
                        help="Parent directory for a new independent batch_... folder.")
    parser.add_argument("--repeat", type=positive_integer, default=3)
    parser.add_argument("--max-steps", type=positive_integer, default=7200)
    args = parser.parse_args(argv)
    try:
        report = validate_batch(args.output_dir, args.repeat, args.max_steps)
    except (OSError, ValueError) as error:
        print(f"VALIDATION_ERROR: {error}", file=sys.stderr)
        return 1
    print(f"VALIDATION_{report['status']} business={report['completed_business_records']}/"
          f"{report['expected_business_records']} extras={report['extra_record_count']} "
          f"directory={report['directory']}", flush=True)
    if report["failures"]:
        print(json.dumps(report["failures"], ensure_ascii=False), file=sys.stderr)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
