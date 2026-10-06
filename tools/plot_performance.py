"""Independently audit saved frame CSVs and plot real window measurements."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def quantile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def audit(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "COMPLETE" or not report["real_window_measurement"]:
        raise ValueError(f"{path}: not a complete native-window measurement")
    m = report["phases"]["measurement"]
    with (path.parent / "frames.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"{path}: no measurement frames")
    begin, end = m["start_monotonic_ns"], m["end_monotonic_ns"]
    previous = begin
    bins = Counter()
    intervals, elapsed, memory = [], [], []
    for index, row in enumerate(rows):
        finish = int(row["frame_end_monotonic_ns"])
        interval = int(row["flip_interval_ns"])
        if int(row["frame"]) != index or row["phase"] != "measurement":
            raise ValueError("frame index/phase mismatch")
        if int(row["previous_flip_monotonic_ns"]) != previous or finish - previous != interval or interval <= 0:
            raise ValueError("frame interval chain mismatch")
        previous = finish
        seconds = (finish - begin) / 1e9
        if not math.isclose(float(row["elapsed_seconds"]), seconds, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("elapsed time mismatch")
        bins[max(0, (finish - begin - 1) // 10**9)] += 1
        intervals.append(interval / 1e6)
        elapsed.append(seconds)
        memory.append(int(row["working_set_bytes"]) / 2**20 if row["working_set_bytes"] else math.nan)
    if previous != end:
        raise ValueError("last frame differs from measurement end")
    wall = (end - begin) / 1e9
    values = {"total_frames": len(rows), "actual_wall_seconds": wall,
              "gui_average_fps": len(rows) / wall,
              "flip_interval_p50_ms": quantile(intervals, .5),
              "flip_interval_p95_ms": quantile(intervals, .95),
              "flip_interval_p99_ms": quantile(intervals, .99),
              "slow_frames_over_33_33_ms": sum(v > 33.33 for v in intervals),
              "worst_complete_one_second_gui_fps": min(bins[s] for s in range(math.floor(wall))),
              "logic_steps_during_phase": sum(int(row["logic_steps"]) for row in rows)}
    for key, value in values.items():
        if not math.isclose(value, m[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{path}: {key} does not match raw frames")
    if not m["completed_duration"] or not m["continuous_running"] or wall < report["parameters"]["duration_seconds"]:
        raise ValueError("duration/RUNNING check failed")
    if report["terminal"]["state"] != "RUNNING" or report["blocked_unit_ids"] or report["errors"]:
        raise ValueError("terminal/error check failed")
    if len(report["terminal"]["units"]) != report["parameters"]["units"]:
        raise ValueError("terminal unit count mismatch")
    with (path.parent / "seconds.csv").open(encoding="utf-8-sig", newline="") as source:
        second_rows = list(csv.DictReader(source))
    if len(second_rows) != math.ceil(wall) or sum(int(row["frames"]) for row in second_rows) != len(rows):
        raise ValueError("second bins do not account for all frames")
    for index, row in enumerate(second_rows):
        if int(row["second"]) != index or int(row["frames"]) != bins[index]:
            raise ValueError("second-bin count mismatch")
    evidence = {"report": str(path.resolve()), "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frames_sha256": hashlib.sha256((path.parent / "frames.csv").read_bytes()).hexdigest(),
                "units": report["parameters"]["units"], "status": "PASS", **values}
    return report, evidence, elapsed, memory, bins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing output is preserved")
    args = parser.parse_args(argv)
    try:
        audited = sorted([audit(path) for path in args.reports], key=lambda row: row[1]["units"])
        counts = [row[1]["units"] for row in audited]
        if len(set(counts)) != len(counts):
            raise ValueError("duplicate unit-count reports")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        args.output_dir.mkdir(parents=True, exist_ok=False)
        first = audited[0][0]
        host = first["environment"]["processor"]
        subtitle = f"1280 x 800 | target pacing 60 FPS | warmup 5s | {host}"
        fig, ax = plt.subplots(figsize=(10, 5.7))
        bars = ax.bar([str(n) for n in counts], [row[1]["gui_average_fps"] for row in audited], color="#347a9b")
        ax.axhline(30, color="#b75c45", linestyle="--", label="30 FPS project target")
        for bar, row in zip(bars, audited):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .6,
                    f"{row[1]['gui_average_fps']:.2f}\n{row[1]['actual_wall_seconds']:.2f}s, n={row[1]['total_frames']}", ha="center", fontsize=10)
        ax.set(xlabel="Total simulated units", ylabel="Measured native-window flips / wall seconds", ylim=(0, 74),
               title="Window throughput by workload size (different durations)")
        ax.legend(loc="lower right"); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        fig.text(.5, .02, subtitle, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(args.output_dir / "fps_by_units.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 5.7))
        for report, result, _, _, bins in audited:
            seconds = range(math.floor(result["actual_wall_seconds"]))
            ax.plot(list(seconds), [bins[s] for s in seconds], label=f"{result['units']} units, {result['actual_wall_seconds']:.2f}s", linewidth=1.0)
        ax.axhline(30, color="#b75c45", linestyle="--")
        ax.set(xlabel="Seconds after measurement baseline", ylabel="Frames in complete one-second window",
               title="Actual frame cadence over time (warmup excluded)")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(args.output_dir / "fps_over_time.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 5.7))
        for report, result, seconds, memory, _ in audited:
            ax.plot(seconds[::60], memory[::60], label=f"{result['units']} units", linewidth=1.2)
        ax.set(xlabel="Seconds after measurement baseline", ylabel="Own-process working set (MiB)",
               title="Sampled resident memory (includes full in-memory snapshots)")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(args.output_dir / "memory_over_time.png", dpi=160); plt.close(fig)
        result = {"status": "PASS", "records": [row[1] for row in audited],
                  "method": "Raw frame chain/count/wall time/linear quantiles/complete 1s bins/terminal independently checked before plotting",
                  "limits": "4-unit 600s and 8/12-unit 60s differ in duration; no scaling law or cross-computer guarantee; active/focus and source metadata remain in original reports",
                  "matplotlib": matplotlib.__version__}
        (args.output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"PERFORMANCE_PLOTS_OK records={len(audited)} output={args.output_dir.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, ImportError, TypeError) as error:
        print(f"PERFORMANCE_PLOT_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
