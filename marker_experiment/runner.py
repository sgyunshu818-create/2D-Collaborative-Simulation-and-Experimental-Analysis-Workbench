"""Development-only selection, frozen parameters, paired final evaluation."""

import csv
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

from . import ExperimentError, LABELS, PREDICTIONS
from .algorithms import Parameters, candidates, predict
from .dataset import generate_dataset, load_dataset, read_image, safe_path, sha256, write_json


ALGORITHMS = ("baseline", "improved")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "numpy": np.__version__,
            "opencv": cv2.__version__, "matplotlib": importlib.metadata.version("matplotlib"),
            "opencv_threads": cv2.getNumThreads(), "opencl_enabled": cv2.ocl.useOpenCL(),
            "packages": sorted(f"{dist.metadata['Name']}=={dist.version}"
                               for dist in importlib.metadata.distributions()),
            "source_sha256": {path.name: sha256(path.read_bytes())
                              for path in sorted(Path(__file__).parent.glob("*.py"))}}


def select_parameters(dev_rows, directory, config):
    """Select accuracy-maximizing candidates using only development labels.

    The predeclared candidate order breaks ties; timing and final-test results
    play no part in this selection. Every condition has equal sample counts.
    """
    if not dev_rows or any(row["split"] != "dev" for row in dev_rows):
        raise ExperimentError("parameter selection accepts development rows only")
    pixels = [(row, read_image(safe_path(directory, row["path"]))) for row in dev_rows]
    selected, trials = {}, []
    for algorithm in ALGORITHMS:
        best_correct = -1
        for parameters in candidates(config, algorithm):
            correct, by_condition = 0, defaultdict(lambda: {"n": 0, "correct": 0})
            for row, image in pixels:
                result = predict(image, parameters)
                success = int(result == row["label"])
                correct += success
                by_condition[row["condition"]]["n"] += 1
                by_condition[row["condition"]]["correct"] += success
            trials.append({"algorithm": algorithm, "parameters": parameters.data(),
                           "n": len(pixels), "correct": correct, "accuracy": correct / len(pixels),
                           "conditions": dict(by_condition)})
            if correct > best_correct:
                selected[algorithm], best_correct = parameters, correct
    return selected, trials


def evaluate_test(test_rows, directory, parameters, parameters_hash):
    if not test_rows or any(row["split"] != "test" for row in test_rows):
        raise ExperimentError("final evaluation accepts test rows only")
    predictions = []
    for index, row in enumerate(test_rows):
        pixels = read_image(safe_path(directory, row["path"]))
        # Alternate order to reduce a fixed first/second-call timing advantage.
        order = ALGORITHMS if index % 2 == 0 else tuple(reversed(ALGORITHMS))
        for algorithm in order:
            started = perf_counter_ns()
            prediction = predict(pixels, parameters[algorithm])
            elapsed_ns = perf_counter_ns() - started
            predictions.append({"split": "test", "image_id": row["image_id"],
                "original_id": row["original_id"], "condition": row["condition"],
                "strength": row["strength"], "label": row["label"],
                "prediction": prediction, "correct": int(prediction == row["label"]),
                "elapsed_ns": elapsed_ns, "elapsed_ms": elapsed_ns / 1_000_000,
                "pixel_sha256": row["pixel_sha256"], "png_sha256": row["png_sha256"],
                "algorithm": algorithm, "parameters_sha256": parameters_hash})
    return predictions


def summarize(predictions):
    """Compute all counts and timing from per-image measurements, including failures."""
    groups = defaultdict(list)
    for row in predictions:
        if row["label"] not in LABELS or row["prediction"] not in PREDICTIONS:
            raise ExperimentError("unknown label or prediction in measurements")
        if row["correct"] != int(row["prediction"] == row["label"]):
            raise ExperimentError("correct flag disagrees with prediction and label")
        if row["elapsed_ns"] < 0 or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0:
            raise ExperimentError("invalid processing time")
        groups[(row["algorithm"], row["condition"])].append(row)
    results = []
    for (algorithm, condition), rows in groups.items():
        matrix = {label: {prediction: 0 for prediction in PREDICTIONS} for label in LABELS}
        for row in rows:
            matrix[row["label"]][row["prediction"]] += 1
        durations = np.array([row["elapsed_ms"] for row in rows], dtype=np.float64)
        correct = sum(row["correct"] for row in rows)
        results.append({"algorithm": algorithm, "condition": condition, "n": len(rows),
                        "correct": correct, "errors": len(rows) - correct,
                        "accuracy": correct / len(rows), "mean_ms": float(np.mean(durations)),
                        "median_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95)),
                        "confusion_matrix": matrix})
    return results


def write_csv(path, rows):
    if not rows:
        raise ExperimentError(f"{path}: cannot write an empty table")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def make_figures(directory, manifest, results, predictions):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(directory)
    conditions = [condition for condition, _ in manifest["conditions"]]
    by_key = {(row["algorithm"], row["condition"]): row for row in results}
    colors = {"baseline": "#3675a6", "improved": "#dd8944"}
    for field, filename, ylabel in (("accuracy", "accuracy.png", "Correct classification (%)"),
                                    ("mean_ms", "processing_time.png", "Mean processing time (ms/image)")):
        fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
        locations = np.arange(len(conditions))
        for aindex, algorithm in enumerate(ALGORITHMS):
            values = [by_key[(algorithm, condition)][field] * (100 if field == "accuracy" else 1)
                      for condition in conditions]
            bars = axis.bar(locations + (aindex - 0.5) * 0.36, values, width=0.36,
                            label=algorithm, color=colors[algorithm])
            axis.bar_label(bars, fmt="%.1f" if field == "accuracy" else "%.3f", padding=3, fontsize=9)
        axis.set_xticks(locations, conditions)
        axis.set_ylabel(ylabel)
        axis.set_title(f"Synthetic circle / square / triangle — n={by_key[('baseline', conditions[0])]['n']} per condition")
        axis.set_ylim(0, 113 if field == "accuracy" else max(0.001, max(row[field] for row in results) * 1.25))
        axis.legend(loc="upper right" if field == "accuracy" else "upper left")
        axis.grid(axis="y", alpha=0.2)
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)
    lookup = {(row["image_id"], row["algorithm"]): row["prediction"] for row in predictions}
    images = {(row["label"], row["condition"]): row for row in reversed(manifest["images"]) if row["split"] == "test"}
    # A predeclared first original per class; selection is independent of success.
    for row in manifest["images"]:
        if row["split"] == "test" and row["original_id"].endswith("_0000"):
            images[(row["label"], row["condition"])] = row
    fig, axes = plt.subplots(3, len(conditions), figsize=(3.0 * len(conditions), 8.5), layout="constrained")
    for rindex, label in enumerate(LABELS):
        for cindex, condition in enumerate(conditions):
            row = images[(label, condition)]
            axis = axes[rindex, cindex]
            axis.imshow(read_image(safe_path(directory, row["path"])), cmap="gray", vmin=0, vmax=255)
            axis.set_title(condition + "\nB=" + lookup[(row["image_id"], "baseline")] +
                           "  I=" + lookup[(row["image_id"], "improved")], fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            if cindex == 0:
                axis.set_ylabel("True: " + label, fontsize=12)
    fig.suptitle("First test original of each class, same original across conditions\nB=baseline; I=improved; occlusion fraction refers to hidden marker area", fontsize=13)
    fig.savefig(directory / "examples.png", dpi=150)
    plt.close(fig)


def run_experiment(config, directory, *, make_plots=True):
    """Generate or verify data, freeze dev choices, then evaluate final pixels."""
    directory = Path(directory)
    if (directory / "summary.json").exists() or (directory / "predictions.csv").exists():
        raise ExperimentError(f"{directory}: final results already exist; use a new output path to preserve this run")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    manifest = (load_dataset(directory, config) if (directory / "manifest.json").exists()
                else generate_dataset(config, directory))
    audit = [{"kind": "dataset_verified", "utc": utc_now()}]
    dev_rows = [row for row in manifest["images"] if row["split"] == "dev"]
    test_rows = [row for row in manifest["images"] if row["split"] == "test"]
    audit.append({"kind": "development_selection_started", "utc": utc_now(), "image_count": len(dev_rows)})
    selected, trials = select_parameters(dev_rows, directory, config)
    frozen = {"format_version": 1, "frozen_utc": utc_now(),
              "selection_split": "dev", "selection_metric": "pooled dev accuracy; first predeclared candidate breaks ties",
              "config_sha256": config.sha256,
              "manifest_sha256": sha256((directory / "manifest.json").read_bytes()),
              "development_images": [{"image_id": row["image_id"], "pixel_sha256": row["pixel_sha256"]} for row in dev_rows],
              "parameters": {algorithm: selected[algorithm].data() for algorithm in ALGORITHMS},
              "development_trials": trials}
    frozen_path = directory / "frozen_parameters.json"
    write_json(frozen_path, frozen)
    frozen_hash = sha256(frozen_path.read_bytes())
    # Evaluation uses parameters read back from the frozen file, not search state.
    disk_frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selected = {algorithm: Parameters(**disk_frozen["parameters"][algorithm]) for algorithm in ALGORITHMS}
    audit.append({"kind": "parameters_frozen", "utc": utc_now(), "sha256": frozen_hash})
    for row in dev_rows[:10]:
        pixels = read_image(safe_path(directory, row["path"]))
        for algorithm in ALGORITHMS:
            predict(pixels, selected[algorithm])
    audit.append({"kind": "test_evaluation_started", "utc": utc_now(), "image_count": len(test_rows)})
    write_json(directory / "protocol.json", audit)
    predictions = evaluate_test(test_rows, directory, selected, frozen_hash)
    if sha256(frozen_path.read_bytes()) != frozen_hash:
        raise ExperimentError("frozen parameter file changed during evaluation")
    results = summarize(predictions)
    environment_data = environment()
    by_key = {(row["algorithm"], row["condition"]): row for row in results}
    comparisons = [{"condition": condition,
                    "accuracy_change": by_key[("improved", condition)]["accuracy"] - by_key[("baseline", condition)]["accuracy"],
                    "mean_ms_change": by_key[("improved", condition)]["mean_ms"] - by_key[("baseline", condition)]["mean_ms"]}
                   for condition, _ in config.conditions]
    summary = {"format_version": 1, "completed_utc": utc_now(), "config": config.data(),
               "config_sha256": config.sha256, "manifest_sha256": frozen["manifest_sha256"],
               "parameters_sha256": frozen_hash, "environment": environment_data,
               "test_images": len(test_rows), "prediction_count": len(predictions),
               "timing": "one warmed predictor call per algorithm/image; preprocessing + contours + classification + input check; PNG decoding, labels, CSV and plotting excluded; order alternated; OpenCV 1 thread; no inference-time averaging",
               "results": results, "comparisons": comparisons,
               "limitations": ["Synthetic geometric markers only; no real-object recognition claims.",
                               "Variants within a condition have distinct originals; the same original is paired across conditions.",
                               "Timing depends on this computer and load; no guaranteed improvement; all declines retained."]}
    write_csv(directory / "predictions.csv", predictions)
    write_csv(directory / "summary.csv", results)
    write_json(directory / "summary.json", summary)
    write_json(directory / "environment.json", environment_data)
    (directory / "environment_packages.txt").write_text("\n".join(environment_data["packages"]) + "\n", encoding="utf-8")
    if make_plots:
        make_figures(directory, manifest, results, predictions)
    audit.append({"kind": "evaluation_completed", "utc": utc_now(), "prediction_count": len(predictions)})
    write_json(directory / "protocol.json", audit)
    return summary
