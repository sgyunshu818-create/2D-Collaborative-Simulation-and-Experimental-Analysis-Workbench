"""Measure the real Pygame window with the original simulation and renderer.

Warmup is separate. The measured flip cadence includes pacing, events, full
simulation recording, rendering, flip, frame-CSV instrumentation and memory
samples. No delta clamp, snapshot truncation or waypoint hiding is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_app.models import BehaviorState, RunState
from sim_app.scene import scene_from_data
from sim_app.simulation import Simulation


FRAME_FIELDS = (
    "phase", "frame", "frame_start_monotonic_ns", "frame_end_monotonic_ns",
    "previous_flip_monotonic_ns", "flip_interval_ns", "elapsed_seconds", "logic_dt_seconds",
    "logic_steps", "sim_step", "sim_time_seconds", "event_ms", "pacing_ms", "update_ms",
    "render_ms", "flip_ms", "clock_tick_reported_ms", "window_active", "keyboard_focus",
    "position_changed_units", "memory_sample_monotonic_ns", "working_set_bytes", "private_usage_bytes",
)
SECOND_FIELDS = ("second", "start_monotonic_ns", "end_monotonic_ns", "actual_seconds",
                 "complete_one_second", "frames", "flip_rate_hz", "gui_fps")
NON_WINDOW_DRIVERS = {"dummy", "offscreen"}


def positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def check_parameters(units=4, duration=600, warmup=5, target_fps=60) -> dict:
    if isinstance(units, bool) or not isinstance(units, int) or units not in (4, 8, 12):
        raise ValueError("units must be 4, 8 or 12")
    if isinstance(target_fps, bool) or not isinstance(target_fps, int) or target_fps <= 0:
        raise ValueError("target_fps must be a positive integer")
    return {"units": units, "duration_seconds": positive_number(duration, "duration"),
            "warmup_seconds": positive_number(warmup, "warmup"), "target_fps": target_fps}


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def make_workload(units=4, duration=600, warmup=5, source_path=None):
    """Extend the obstacle template into long alternating routes at original speed."""
    params = check_parameters(units, duration, warmup)
    source_path = Path(source_path or PROJECT_ROOT / "configs" / "obstacle_scene.json")
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes.decode("utf-8-sig"))
    data = json.loads(json.dumps(source))
    # The same route horizon is used for ordinary 600/60-second measurements.
    # Euclidean length is a lower bound on the ground detour's duration.
    horizon = max(900.0, params["duration_seconds"] + params["warmup_seconds"] + 120.0)
    if not math.isfinite(horizon):
        raise ValueError("requested duration overflows the workload horizon")
    specs = []
    leg_counts = []
    for copy_index in range(units // 4):
        for original in source["units"]:
            item = json.loads(json.dumps(original))
            direction = 1 if original["team"] == "red" else -1
            dx, dy = direction * copy_index * 8, direction * copy_index * 10
            origin = {"x": original["x"] + dx, "y": original["y"] + dy}
            far = {"x": original["waypoints"][0]["x"] + dx,
                   "y": original["waypoints"][0]["y"] + dy}
            leg_seconds = math.hypot(far["x"] - origin["x"], far["y"] - origin["y"]) / original["speed"]
            count = math.ceil(horizon / leg_seconds) + 4
            if count > 10000:
                raise ValueError("requested duration requires over 10000 waypoints per unit")
            item.update(id=original["id"] if copy_index == 0 else f"{original['id']}_copy_{copy_index + 1}",
                        x=origin["x"], y=origin["y"],
                        waypoints=[dict(far if index % 2 == 0 else origin) for index in range(count)],
                        return_home=False)
            specs.append(item)
            leg_counts.append(count)
    data.update(name=f"stage5_benchmark_{units}_units", units=specs)
    data["rules"].update(score_limit=1000000, time_limit=horizon * 2 + 120)
    scene = scene_from_data(data, source_path)
    metadata = {
        "source_path": str(source_path.resolve()), "source_scene_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "effective_scene_sha256": hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest(),
        "minimum_route_horizon_seconds": horizon, "waypoints_per_unit": leg_counts,
        "generation": "Obstacle scene templates; paired ground/air retain speed/rules, translated copies, repeated far/origin legs.",
        "inter_unit_collisions": "Not implemented or measured; geometry checks static rectangles only.",
        "render_load": "Original Renderer draws ALL generated waypoint labels and full trails/knowledge; none hidden for benchmark.",
        "snapshot_policy": "Original Simulation stores every game logic step and controls; no truncation or recording disable.",
    }
    return scene, data, metadata


def percentile(values, fraction: float):
    """Linear interpolation at (N-1)*fraction, including the endpoints."""
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    left = int(math.floor(location))
    right = int(math.ceil(location))
    return ordered[left] + (ordered[right] - ordered[left]) * (location - left)


class FrameStats:
    def __init__(self, start_ns):
        self.start_ns = start_ns
        self.intervals_ns = []
        self.seconds = {}
        self.active_frames = self.focused_frames = self.moving_frames = 0
        self.changed_ids = set()
        self.total_logic_steps = 0
        self.parts = {name: 0.0 for name in ("event_ms", "pacing_ms", "update_ms", "render_ms", "flip_ms")}

    def add(self, row, changed_ids=()):
        self.intervals_ns.append(row["flip_interval_ns"])
        second = max(0, (row["frame_end_monotonic_ns"] - self.start_ns - 1) // 1000000000)
        self.seconds[second] = self.seconds.get(second, 0) + 1
        self.active_frames += int(row["window_active"])
        self.focused_frames += int(row["keyboard_focus"])
        self.moving_frames += int(bool(changed_ids))
        self.changed_ids.update(changed_ids)
        self.total_logic_steps += row["logic_steps"]
        for key in self.parts:
            self.parts[key] += row[key]

    def finish(self, end_ns, real_window):
        elapsed = max(0, end_ns - self.start_ns) / 1e9
        count = len(self.intervals_ns)
        second_rows = []
        complete_rates = []
        for second in range(math.ceil(elapsed)):
            start = self.start_ns + second * 1000000000
            end = min(end_ns, start + 1000000000)
            duration = max(0, end - start) / 1e9
            frames = self.seconds.get(second, 0)
            rate = frames / duration if duration else 0.0
            complete = end - start == 1000000000
            if complete:
                complete_rates.append(rate)
            second_rows.append({"second": second, "start_monotonic_ns": start, "end_monotonic_ns": end,
                                "actual_seconds": duration, "complete_one_second": complete,
                                "frames": frames, "flip_rate_hz": rate, "gui_fps": rate if real_window else None})
        milliseconds = [value / 1e6 for value in self.intervals_ns]
        rate = count / elapsed if elapsed else 0.0
        summary = {
            "total_frames": count, "actual_wall_seconds": elapsed,
            "actual_flip_rate_hz": rate, "gui_average_fps": rate if real_window else None,
            "flip_interval_p50_ms": percentile(milliseconds, 0.5),
            "flip_interval_p95_ms": percentile(milliseconds, 0.95),
            "flip_interval_p99_ms": percentile(milliseconds, 0.99),
            "slow_frames_over_33_33_ms": sum(value > 33.33 for value in milliseconds),
            "worst_complete_one_second_gui_fps": min(complete_rates) if complete_rates and real_window else None,
            "complete_one_second_window_count": len(complete_rates),
            "active_frame_fraction": self.active_frames / count if count else None,
            "keyboard_focus_frame_fraction": self.focused_frames / count if count else None,
            "frames_with_position_change": self.moving_frames,
            "units_with_observed_position_changes": sorted(self.changed_ids),
            "logic_steps_during_phase": self.total_logic_steps,
            "mean_components_ms": {key: value / count if count else None for key, value in self.parts.items()},
            "interval_percentile_method": "Linear interpolation at (N-1)*q; flip intervals include pacing and bookkeeping.",
            "one_second_method": "Fixed non-overlapping (start+k,start+k+1] windows; partial last window listed but excluded from worst 1s.",
        }
        return summary, second_rows


def sample_process_memory() -> dict:
    """Own-process working set/private committed bytes from Windows PSAPI."""
    sampled = time.perf_counter_ns()
    if os.name != "nt":
        return {"sample_monotonic_ns": sampled, "available": False,
                "reason": "Windows PSAPI memory counters unavailable on this platform"}
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"sample_monotonic_ns": sampled, "available": True,
                "working_set_bytes": counters.WorkingSetSize, "private_usage_bytes": counters.PrivateUsage,
                "os_peak_working_set_bytes": counters.PeakWorkingSetSize,
                "meaning": "WorkingSetSize resident own-process memory; PrivateUsage private committed bytes; OS peak includes setup/warmup."}
    except (OSError, AttributeError) as error:
        return {"sample_monotonic_ns": sampled, "available": False, "reason": str(error)}


def write_full_record(sim, data, scene_path, directory):
    """Stream the full format-3 record without deepcopy or one giant JSON string.

    Export is AFTER measurement. All original in-memory step snapshots survive;
    this avoids a large, unnecessary extra copy during ten-minute final export.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    header = {"format_version": 3, "source_scene": str(scene_path.resolve()), "scene": data,
              "result": sim.snapshot(), "events": sim.events}
    digest = hashlib.sha256()
    last_progress = time.perf_counter()
    with path.open("w", encoding="utf-8") as output:
        output.write("{")
        for key, value in header.items():
            output.write(json.dumps(key) + ":")
            json.dump(value, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            output.write(",")
        output.write('"snapshots":[')
        for index, snapshot in enumerate(sim.snapshots):
            if index:
                output.write(",\n")
            encoded = canonical_json(snapshot)
            output.write(encoded)
            digest.update((encoded + "\n").encode("utf-8"))
            now = time.perf_counter()
            if now - last_progress >= 30:
                print(f"BENCHMARK_EXPORT snapshots={index + 1}/{len(sim.snapshots)}", flush=True)
                last_progress = now
        output.write("]}")
    with (directory / "events.csv").open("w", encoding="utf-8-sig", newline="") as output:
        fields = ("step", "time", "kind", "unit_id", "message", "details")
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event in sim.events:
            row = dict(event)
            row["details"] = canonical_json({key: value for key, value in event.items() if key not in fields[:5]})
            writer.writerow(row)
    # The full state sequence is already in standard run.json. A compact
    # snapshot index makes its timing/event boundaries separately auditable.
    with (directory / "snapshot_index.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("index", "step", "time", "state", "event_count", "unit_count"))
        writer.writeheader()
        for index, frame in enumerate(sim.snapshots):
            writer.writerow({"index": index, "step": frame["step"], "time": frame["time"],
                             "state": frame["state"], "event_count": frame["event_count"], "unit_count": len(frame["units"])})
    return {"run_json": str(path.resolve()), "run_json_bytes": path.stat().st_size,
            "events_csv": str((directory / "events.csv").resolve()),
            "snapshot_index_csv": str((directory / "snapshot_index.csv").resolve()),
            "snapshot_count": len(sim.snapshots), "event_count": len(sim.events),
            "snapshot_content_sha256": digest.hexdigest(),
            "snapshot_digest_definition": "SHA256 of each canonical snapshot JSON plus LF, in recorded order."}


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(units=4, duration=600, warmup=5, target_fps=60, output_dir=None,
                  allow_non_window=False) -> tuple[int, dict]:
    params = check_parameters(units, duration, warmup, target_fps)
    if not isinstance(allow_non_window, bool):
        raise ValueError("allow_non_window must be a boolean")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    directory = Path(output_dir or PROJECT_ROOT / "artifacts" / "stage5_performance").resolve() / f"benchmark_{units}u_{stamp}_{uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    scene, data, workload = make_workload(units, duration, warmup)
    scene_path = directory / "scene.json"
    scene_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    sim = Simulation(scene)
    errors = []
    samples = []
    phase_results = {}
    real_window = False
    status = "ERROR"
    pygame = None
    surface = None
    env = {
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(), "process_id": os.getpid(),
        "SDL_VIDEODRIVER_environment": os.environ.get("SDL_VIDEODRIVER"),
        "benchmark_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "clock": {key: getattr(time.get_clock_info("perf_counter"), key)
                  for key in ("implementation", "monotonic", "adjustable", "resolution")},
    }
    print(f"BENCHMARK_START units={units} warmup={warmup}s measurement={duration}s directory={directory}", flush=True)
    try:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame as pygame_module
        from sim_app.renderer import Renderer, WINDOW_SIZE
        pygame = pygame_module
        pygame.display.init()
        pygame.font.init()
        surface = pygame.display.set_mode(WINDOW_SIZE)
        renderer = Renderer(surface)
        driver = pygame.display.get_driver()
        native_info = pygame.display.get_wm_info()
        native_window = driver not in NON_WINDOW_DRIVERS and bool(native_info.get("window"))
        real_window = native_window and not allow_non_window
        env.update(pygame_version=pygame.version.ver, display_driver=driver,
                   window_resolution=list(surface.get_size()), desktop_resolutions=pygame.display.get_desktop_sizes(),
                   native_window_present=bool(native_info.get("window")), real_window=real_window,
                   mechanism_test_requested=allow_non_window,
                   font_path=renderer.font_path)
        if not real_window and not allow_non_window:
            status = "INVALID_DISPLAY"
            raise RuntimeError(f"display driver={driver!r} does not prove a real native window; dummy/offscreen are mechanism tests only")
        sim.start()
        if sim.state != RunState.RUNNING:
            raise RuntimeError("workload did not start continuously RUNNING")
        clock = pygame.time.Clock()
        for phase, phase_duration in (("warmup", params["warmup_seconds"]), ("measurement", params["duration_seconds"])):
            clock.tick()  # Explicitly discard setup/PNG/phase-transition pacing history.
            phase_start = previous_flip = previous_paced = time.perf_counter_ns()
            stats = FrameStats(phase_start)
            phase_sim_start = sim.step_count
            last_progress_ns = phase_start
            next_memory_ns = phase_start
            memory = {}
            stopped = False
            with (directory / ("frames.csv" if phase == "measurement" else "warmup_frames.csv")).open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FRAME_FIELDS)
                writer.writeheader()
                last_flush_second = -1
                while True:
                    frame_start = time.perf_counter_ns()
                    tick_ms = clock.tick(target_fps)
                    paced = time.perf_counter_ns()
                    elapsed = (paced - phase_start) / 1e9
                    logic_dt = (paced - previous_paced) / 1e9
                    previous_paced = paced
                    t0 = time.perf_counter_ns()
                    for event in pygame.event.get():
                        abort = event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            abort = bool(renderer.action_at(event.pos, sim.state) or renderer.extra_action_at(event.pos, sim))
                        if abort:
                            raise RuntimeError("window exit/control requested before benchmark completed")
                    t1 = time.perf_counter_ns()
                    old_positions = {unit.id: unit.position for unit in sim.units}
                    steps = sim.advance(logic_dt)
                    changed = [unit.id for unit in sim.units if unit.position != old_positions[unit.id]]
                    t2 = time.perf_counter_ns()
                    remaining = max(0.0, phase_duration - elapsed)
                    title = f"Performance Benchmark | {units} units | {phase} {elapsed:.1f}s / {phase_duration:g}s"
                    pygame.display.set_caption(title)
                    notice = renderer.label(
                        f"性能基准 · {units} 单位 · {phase} {elapsed:.1f}s · 剩余 {remaining:.1f}s · Esc 中止",
                        f"Benchmark | {units} units | {phase} {elapsed:.1f}s | remaining {remaining:.1f}s | Esc abort")
                    renderer.draw(sim, notice, pygame.mouse.get_pos())
                    t3 = time.perf_counter_ns()
                    pygame.display.flip()
                    end = time.perf_counter_ns()
                    if end >= next_memory_ns:
                        memory = sample_process_memory()
                        samples.append({"phase": phase, "elapsed_seconds": (end - phase_start) / 1e9, **memory})
                        next_memory_ns = end + 1000000000
                    row = {
                        "phase": phase, "frame": len(stats.intervals_ns), "frame_start_monotonic_ns": frame_start,
                        "frame_end_monotonic_ns": end, "previous_flip_monotonic_ns": previous_flip,
                        "flip_interval_ns": end - previous_flip, "elapsed_seconds": (end - phase_start) / 1e9,
                        "logic_dt_seconds": logic_dt, "logic_steps": steps, "sim_step": sim.step_count,
                        "sim_time_seconds": sim.sim_time, "event_ms": (t1 - t0) / 1e6,
                        "pacing_ms": (paced - frame_start) / 1e6, "update_ms": (t2 - t1) / 1e6,
                        "render_ms": (t3 - t2) / 1e6, "flip_ms": (end - t3) / 1e6,
                        "clock_tick_reported_ms": tick_ms, "window_active": pygame.display.get_active(),
                        "keyboard_focus": pygame.key.get_focused(), "position_changed_units": len(changed),
                        "memory_sample_monotonic_ns": memory.get("sample_monotonic_ns"),
                        "working_set_bytes": memory.get("working_set_bytes"), "private_usage_bytes": memory.get("private_usage_bytes"),
                    }
                    stats.add(row, changed)
                    writer.writerow(row)
                    flush_second = int(row["elapsed_seconds"])
                    if flush_second != last_flush_second:
                        output.flush()
                        last_flush_second = flush_second
                    previous_flip = end
                    if end - last_progress_ns >= 30000000000:
                        print(f"BENCHMARK_PROGRESS phase={phase} units={units} elapsed={row['elapsed_seconds']:.1f}s "
                              f"frames={len(stats.intervals_ns)} steps={sim.step_count} snapshots={len(sim.snapshots)}", flush=True)
                        last_progress_ns = end
                    if sim.state != RunState.RUNNING:
                        stopped = True
                        errors.append("workload left RUNNING before its sustained measurement finished")
                    if any(unit.behavior == BehaviorState.BLOCKED for unit in sim.units):
                        stopped = True
                        errors.append("workload contains BLOCKED units")
                    if stopped or (end - phase_start) / 1e9 >= phase_duration:
                        break
                summary, seconds = stats.finish(previous_flip, real_window)
                summary.update(start_monotonic_ns=phase_start, end_monotonic_ns=previous_flip,
                               sim_start_step=phase_sim_start, sim_end_step=sim.step_count,
                               requested_seconds=phase_duration,
                               completed_duration=summary["actual_wall_seconds"] >= phase_duration,
                               continuous_running=not stopped and sim.state == RunState.RUNNING)
                phase_results[phase] = summary
                _write_csv(directory / ("seconds.csv" if phase == "measurement" else "warmup_seconds.csv"), SECOND_FIELDS, seconds)
            pygame.image.save(surface, str(directory / ("after_warmup.png" if phase == "warmup" else "finished.png")))
            if stopped:
                break
            print(f"BENCHMARK_PHASE_DONE phase={phase} frames={summary['total_frames']} "
                  f"actual_seconds={summary['actual_wall_seconds']:.6f}", flush=True)
        measurement = phase_results.get("measurement", {})
        complete = measurement.get("completed_duration", False) and measurement.get("continuous_running", False)
        status = ("COMPLETE" if real_window else "MECHANISM_ONLY") if complete and not errors else "PARTIAL"
    except (Exception, KeyboardInterrupt) as error:
        errors.append(f"{type(error).__name__}: {error}")
        if status != "INVALID_DISPLAY":
            status = "PARTIAL" if phase_results else "ERROR"
        # Frames already written remain available even when interrupted inside
        # a phase. Recover its statistics using the last successful flip.
        if "stats" in locals() and phase not in phase_results:
            summary, seconds = stats.finish(previous_flip, real_window)
            summary.update(start_monotonic_ns=phase_start, end_monotonic_ns=previous_flip,
                           sim_start_step=phase_sim_start, sim_end_step=sim.step_count,
                           requested_seconds=phase_duration, completed_duration=False, continuous_running=False)
            phase_results[phase] = summary
            try:
                _write_csv(directory / ("seconds.csv" if phase == "measurement" else "warmup_seconds.csv"), SECOND_FIELDS, seconds)
            except OSError as export_error:
                errors.append(f"partial seconds export: {export_error}")
        if surface is not None and pygame is not None:
            try:
                pygame.image.save(surface, str(directory / "partial.png"))
            except (OSError, pygame.error) as image_error:
                errors.append(f"partial screenshot: {image_error}")
    finally:
        if pygame is not None:
            pygame.quit()
    export_start = time.perf_counter()
    recording = {}
    try:
        recording = write_full_record(sim, data, scene_path, directory / "recording")
    except (OSError, ValueError, MemoryError) as error:
        errors.append(f"full recording export: {error}")
        status = "ERROR"
    available = [sample for sample in samples if sample.get("available") and sample["phase"] == "measurement"]
    measurement = phase_results.get("measurement", {})
    fps = measurement.get("gui_average_fps")
    report = {
        "format_version": 1, "status": status, "directory": str(directory),
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "parameters": params, "environment": env, "workload": workload, "phases": phase_results,
        "real_window_measurement": real_window,
        "performance_goal": {"minimum_average_gui_fps": 30, "met": fps >= 30 if fps is not None else None,
                             "meaning": "Measured average only; no preset success or long-duration claim for the 8/12-unit short tests."},
        "memory": {"samples": samples, "sample_interval_seconds": 1,
                   "measured_peak_working_set_bytes": max((sample["working_set_bytes"] for sample in available), default=None),
                   "measured_peak_private_usage_bytes": max((sample["private_usage_bytes"] for sample in available), default=None),
                   "measurement_available_samples": len(available),
                   "meaning": "Own-process Windows PSAPI; CSV carries latest sample, not a new memory query every frame; includes full in-memory simulation records."},
        "terminal": sim.snapshot(), "blocked_unit_ids": [unit.id for unit in sim.units if unit.behavior == BehaviorState.BLOCKED],
        "recording": recording, "recording_export_wall_seconds": time.perf_counter() - export_start,
        "errors": errors,
        "measurement_definition": {
            "overall": "Successful measurement flips divided by actual monotonic first-boundary-to-last-flip seconds; warmup/setup/screenshots/final exports excluded.",
            "included": "Pacing tick, events, unclamped actual delta, original Simulation and every-step records, original full Renderer, display.flip, CSV writes/flushes and 1s memory sampling between flips.",
            "breakdown": "event/update/render/flip regions exclude pacing; pacing_ms is separate. Bookkeeping after a flip appears in the next flip interval.",
            "boundary": "After warmup PNG, timing baseline is deliberately restarted; simulation state and all warmup snapshots are retained.",
            "ui": "Original business Renderer unchanged; benchmark title bar and existing notice show units/phase/elapsed/remaining.",
        },
        "limits": ["No inter-unit collision model", "Long generated route labels add genuine renderer workload",
                   "Memory includes original step snapshots and can grow for 600 seconds", "No per-frame PNG capture",
                   "The dedicated benchmark loop does not establish all interactive app flows", "Other computers and GPU/OS loads may differ"],
    }
    try:
        (directory / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"BENCHMARK_REPORT_ERROR: {error}", file=sys.stderr, flush=True)
        return 1, report
    code = 0 if status in ("COMPLETE", "MECHANISM_ONLY") and not errors else 1
    print(f"BENCHMARK_{status} units={units} frames={measurement.get('total_frames', 0)} "
          f"actual_seconds={measurement.get('actual_wall_seconds', 0):.6f} gui_fps={fps} directory={directory}", flush=True)
    if errors:
        print(json.dumps(errors, ensure_ascii=False), file=sys.stderr, flush=True)
    return code, report


def _number_argument(value):
    try:
        return positive_number(float(value), "value")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, choices=(4, 8, 12), default=4)
    parser.add_argument("--duration", type=_number_argument, default=600.0)
    parser.add_argument("--warmup", type=_number_argument, default=5.0)
    parser.add_argument("--target-fps", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "stage5_performance")
    parser.add_argument("--mechanism-test", action="store_true", help="Allow dummy/offscreen for mechanism testing only; GUI FPS is null.")
    args = parser.parse_args(argv)
    try:
        code, _ = run_benchmark(args.units, args.duration, args.warmup, args.target_fps,
                                args.output_dir, allow_non_window=args.mechanism_test)
        return code
    except (OSError, ValueError) as error:
        print(f"BENCHMARK_ARGUMENT_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
