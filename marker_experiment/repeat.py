"""Repeat a completed marker experiment with its unchanged, frozen parameters.

Usage: python -m marker_experiment.repeat --source artifacts/marker_experiment
       --output artifacts/marker_repeat5 --repeats 5

No generator or parameter search is called. A new batch directory preserves
each run. Accuracy repeats test the same images, adding no independent samples.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from . import ExperimentError, LABELS, PREDICTIONS
from .config import config_from_data


ALGORITHMS = ("baseline", "improved")
CORE_FILES = ("__init__.py", "__main__.py", "algorithms.py", "config.py", "dataset.py", "runner.py")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise ExperimentError(f"{path}: cannot read experiment evidence: {error}") from error
    if not isinstance(value, dict):
        raise ExperimentError(f"{path}: expected a JSON object")
    return value


def validate_repeats(repeats, order_seed):
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ExperimentError("repeats must be an integer from 1 to 100")
    if type(order_seed) is not int or not 0 <= order_seed < 2 ** 63:
        raise ExperimentError("order_seed must be a non-negative integer below 2**63")


def _percentile95(values):
    values = sorted(values)
    position = 0.95 * (len(values) - 1)
    lower = math.floor(position)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[min(lower + 1, len(values) - 1)] * fraction


def round_statistics(rows):
    """Recompute each round's condition counts and timing from raw predictions."""
    groups, identities = defaultdict(list), set()
    for row in rows:
        identity = (row["round"], row["image_id"], row["algorithm"])
        if identity in identities:
            raise ExperimentError("duplicate round/image/algorithm measurement")
        identities.add(identity)
        if row["label"] not in LABELS or row["prediction"] not in PREDICTIONS:
            raise ExperimentError("invalid measured class")
        if row["correct"] != int(row["label"] == row["prediction"]):
            raise ExperimentError("correct flag disagrees with prediction")
        if type(row["elapsed_ns"]) is not int or row["elapsed_ns"] < 0:
            raise ExperimentError("elapsed_ns must be a non-negative integer")
        if not math.isfinite(row["elapsed_ms"]) or not math.isclose(row["elapsed_ms"], row["elapsed_ns"] / 1e6, abs_tol=1e-12):
            raise ExperimentError("elapsed_ms must agree with elapsed_ns")
        if row["changed_prediction"] != int(row["prediction"] != row["first_prediction"]):
            raise ExperimentError("changed flag disagrees with first prediction")
        groups[(row["round"], row["condition"], row["algorithm"])].append(row)
    result = []
    for (round_index, condition, algorithm), group in sorted(groups.items()):
        correct = sum(row["correct"] for row in group)
        durations = [row["elapsed_ms"] for row in group]
        matrix = {label: {prediction: 0 for prediction in PREDICTIONS} for label in LABELS}
        for row in group:
            matrix[row["label"]][row["prediction"]] += 1
        result.append({"round": round_index, "condition": condition, "algorithm": algorithm,
                       "n": len(group), "correct": correct, "errors": len(group) - correct,
                       "unknown": sum(row["prediction"] == "unknown" for row in group),
                       "changed_predictions": sum(row["changed_prediction"] for row in group),
                       "accuracy": correct / len(group), "mean_ms": statistics.fmean(durations),
                       "median_ms": statistics.median(durations), "p95_ms": _percentile95(durations),
                       "confusion_matrix": matrix})
    return result


def aggregate_rounds(per_round, rows, repeats):
    """Use rounds as timing repetitions; keep 120 unique images as the denominator."""
    groups, raw_groups = defaultdict(list), defaultdict(list)
    for row in per_round:
        groups[(row["condition"], row["algorithm"])].append(row)
    for row in rows:
        raw_groups[(row["condition"], row["algorithm"])].append(row)
    result = []
    for (condition, algorithm), group in sorted(groups.items()):
        group.sort(key=lambda row: row["round"])
        if [row["round"] for row in group] != list(range(1, repeats + 1)) or len({row["n"] for row in group}) != 1:
            raise ExperimentError("each condition/algorithm must have equal samples in every round")
        means = [row["mean_ms"] for row in group]
        result.append({"condition": condition, "algorithm": algorithm,
            "unique_test_images": group[0]["n"], "repeats": repeats,
            "measured_calls": sum(row["n"] for row in group),
            "correct_per_round": [row["correct"] for row in group],
            "errors_per_round": [row["errors"] for row in group],
            "unknown_per_round": [row["unknown"] for row in group],
            "changed_predictions": sum(row["changed_predictions"] for row in group),
            "accuracy_mean": statistics.fmean(row["accuracy"] for row in group),
            "round_mean_ms": means, "round_mean_ms_mean": statistics.fmean(means),
            "round_mean_ms_sample_std": statistics.stdev(means) if repeats > 1 else 0.0,
            "round_mean_ms_min": min(means), "round_mean_ms_max": max(means),
            "pooled_call_p95_ms": _percentile95([row["elapsed_ms"] for row in raw_groups[(condition, algorithm)]]),
            "first_round_confusion_matrix": group[0]["confusion_matrix"]})
    return result


def paired_errors(rows):
    pairs = defaultdict(dict)
    for row in rows:
        key = (row["round"], row["condition"], row["image_id"])
        if row["algorithm"] in pairs[key]:
            raise ExperimentError("duplicate paired algorithm result")
        pairs[key][row["algorithm"]] = row
    groups = defaultdict(Counter)
    for (round_index, condition, image_id), pair in pairs.items():
        if set(pair) != set(ALGORITHMS) or pair["baseline"]["pixel_sha256"] != pair["improved"]["pixel_sha256"]:
            raise ExperimentError(f"{image_id}: both algorithms must use the same image")
        base, improved = bool(pair["baseline"]["correct"]), bool(pair["improved"]["correct"])
        outcome = ("both_correct" if base and improved else "baseline_only_correct" if base
                   else "improved_only_correct" if improved else "both_wrong")
        groups[(round_index, condition)][outcome] += 1
    result = []
    for (round_index, condition), counts in sorted(groups.items()):
        result.append({"round": round_index, "condition": condition,
                       "unique_test_images": sum(counts.values()),
                       **{name: counts[name] for name in ("both_correct", "baseline_only_correct", "improved_only_correct", "both_wrong")},
                       "net_gain_unique_images": counts["improved_only_correct"] - counts["baseline_only_correct"]})
    return result


def prepare_source(source):
    """Audit first-run evidence and current code before any new predictions."""
    from .algorithms import Parameters
    from .dataset import load_dataset

    source = Path(source).resolve()
    summary = read_json(source / "summary.json")
    frozen = read_json(source / "frozen_parameters.json")
    manifest_data = read_json(source / "manifest.json")
    config = config_from_data(manifest_data.get("config"), source / "manifest.json")
    manifest_hash = digest(source / "manifest.json")
    parameters_hash = digest(source / "frozen_parameters.json")
    if summary.get("manifest_sha256") != manifest_hash or frozen.get("manifest_sha256") != manifest_hash:
        raise ExperimentError(f"{source}: manifest hash differs from frozen/first-run evidence")
    if summary.get("parameters_sha256") != parameters_hash:
        raise ExperimentError(f"{source}: frozen parameter hash differs from the first run")
    if summary.get("config_sha256") != config.sha256 or frozen.get("config_sha256") != config.sha256 or frozen.get("selection_split") != "dev":
        raise ExperimentError(f"{source}: frozen development/configuration evidence is inconsistent")
    previous_code = summary.get("environment", {}).get("source_sha256", {})
    if not isinstance(previous_code, dict) or not set(CORE_FILES).issubset(previous_code):
        raise ExperimentError(f"{source}: first-run core source hashes are missing")
    code_hashes = {}
    for filename, expected in previous_code.items():
        if not isinstance(filename, str) or Path(filename).name != filename or ":" in filename or "\\" in filename:
            raise ExperimentError(f"{source}: invalid recorded core filename")
        current = digest(Path(__file__).parent / filename)
        if current != expected:
            raise ExperimentError(f"{filename}: current source differs from the first experiment; this repeat protocol cannot change the method")
        code_hashes[filename] = current
    manifest = load_dataset(source, config)
    test_rows = [row for row in manifest["images"] if row["split"] == "test"]
    dev_rows = [row for row in manifest["images"] if row["split"] == "dev"]
    parameters_data = frozen.get("parameters")
    if not isinstance(parameters_data, dict) or set(parameters_data) != set(ALGORITHMS):
        raise ExperimentError(f"{source}: frozen parameters must contain both algorithms")
    parameters = {}
    for algorithm in ALGORITHMS:
        try:
            parameter = Parameters(**parameters_data[algorithm])
        except (TypeError, ValueError) as error:
            raise ExperimentError(f"{source}: invalid frozen {algorithm} parameters: {error}") from error
        if (parameter.algorithm != algorithm or parameter.epsilon not in config.polygon_epsilons
                or parameter.circularity not in config.circularity_thresholds or parameter.min_area != 80.0
                or (algorithm == "baseline" and parameter.threshold not in config.baseline_thresholds)
                or (algorithm == "improved" and parameter.blur_kernel not in config.blur_kernels)):
            raise ExperimentError(f"{source}: parameters are outside the declared frozen development grid")
        parameters[algorithm] = parameter
    try:
        with (source / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            first_rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ExperimentError(f"{source}: cannot read first predictions: {error}") from error
    image_by_id = {row["image_id"]: row for row in test_rows}
    first = {}
    for row in first_rows:
        image = image_by_id.get(row.get("image_id"))
        algorithm = row.get("algorithm")
        if image is None or algorithm not in ALGORITHMS:
            raise ExperimentError(f"{source}: invalid first prediction identity")
        key = (row["image_id"], algorithm)
        if (key in first or row.get("label") != image["label"] or row.get("condition") != image["condition"]
                or row.get("pixel_sha256") != image["pixel_sha256"] or row.get("png_sha256") != image["png_sha256"]
                or row.get("parameters_sha256") != parameters_hash or row.get("prediction") not in PREDICTIONS
                or row.get("correct") != str(int(row["prediction"] == row["label"]))):
            raise ExperimentError(f"{source}: duplicate or inconsistent first prediction")
        first[key] = row
    if len(first) != len(test_rows) * 2:
        raise ExperimentError(f"{source}: first predictions do not cover both algorithms for every test image")
    first_summary = {(row.get("condition"), row.get("algorithm")): row for row in summary.get("results", [])}
    for condition, _ in config.conditions:
        for algorithm in ALGORITHMS:
            selected = [row for row in first_rows if row["condition"] == condition and row["algorithm"] == algorithm]
            recorded = first_summary.get((condition, algorithm))
            correct = sum(row["correct"] == "1" for row in selected)
            matrix = Counter((row["label"], row["prediction"]) for row in selected)
            if recorded is None or recorded.get("n") != len(selected) or recorded.get("correct") != correct:
                raise ExperimentError(f"{source}: first summary does not agree with raw predictions")
            for label in LABELS:
                for prediction in PREDICTIONS:
                    if recorded.get("confusion_matrix", {}).get(label, {}).get(prediction) != matrix[(label, prediction)]:
                        raise ExperimentError(f"{source}: first confusion matrix disagrees with raw predictions")
    evidence_names = ("summary.json", "manifest.json", "frozen_parameters.json", "predictions.csv",
                      "environment.json", "manifest.csv", "protocol.json")
    evidence_hashes = {name: digest(source / name) for name in evidence_names}
    return {"source": source, "config": config, "manifest": manifest, "summary": summary,
            "frozen": frozen, "parameters": parameters, "parameters_sha256": parameters_hash,
            "manifest_sha256": manifest_hash, "test_rows": test_rows, "dev_rows": dev_rows,
            "first": first, "code_sha256": code_hashes, "evidence_sha256": evidence_hashes}


def make_plots(source, directory, protocol, aggregate, paired, rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    from .dataset import read_image, safe_path

    conditions = [name for name, _ in protocol["conditions"]]
    lookup = {(row["condition"], row["algorithm"]): row for row in aggregate}
    figure, axis = plt.subplots(figsize=(11, 5.5), layout="constrained")
    positions = np.arange(len(conditions))
    for index, algorithm in enumerate(ALGORITHMS):
        selected = [lookup[(condition, algorithm)] for condition in conditions]
        means = [row["round_mean_ms_mean"] for row in selected]
        errors = [row["round_mean_ms_sample_std"] for row in selected]
        bars = axis.bar(positions + (index - 0.5) * 0.36, means, 0.36, yerr=errors,
                        capsize=4, label=algorithm, color=("#3675a6", "#dd8944")[index])
        axis.bar_label(bars, fmt="%.3f", padding=5, fontsize=9)
    axis.set_xticks(positions, conditions)
    axis.set_ylabel("Mean processing time (ms/image)")
    axis.set_title(f"{protocol['repeats']} real repeats, same frozen test pixels\nBars: mean of round means; error bars: sample SD across rounds")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    axis.set_ylim(0, max(row["round_mean_ms_mean"] + row["round_mean_ms_sample_std"] for row in aggregate) * 1.3)
    figure.savefig(Path(directory) / "repeat_processing_time.png", dpi=160)
    plt.close(figure)
    first = {(row["image_id"], row["algorithm"]): row for row in rows if row["round"] == 1}
    selected_ids = protocol["failure_example_selection"]["image_ids"]
    images = {row["image_id"]: row for row in protocol["test_image_identity"]}
    figure, axes = plt.subplots(2, 5, figsize=(15, 6.7), layout="constrained")
    for axis, image_id in zip(axes.flat, selected_ids + [None] * max(0, 10 - len(selected_ids))):
        axis.set_xticks([])
        axis.set_yticks([])
        if image_id is None:
            axis.text(0.5, 0.5, "No failure for this selection", ha="center", va="center")
            continue
        image = images[image_id]
        axis.imshow(read_image(safe_path(source, image["path"])), cmap="gray", vmin=0, vmax=255)
        axis.set_title(image["condition"] + " / true=" + image["label"] + "\nB=" + first[(image_id, "baseline")]["prediction"] +
                       "; I=" + first[(image_id, "improved")]["prediction"], fontsize=10)
    figure.suptitle("Predeclared failure audit: first-run lexicographically first failures\nSelection fixed before repeats; both algorithms and unchanged true class shown", fontsize=12)
    figure.savefig(Path(directory) / "failure_examples.png", dpi=150)
    plt.close(figure)


def repeat_experiment(source, directory, repeats=5, order_seed=510005, *, plots=True):
    """Write an independent repeat batch; existing batch directories are refused."""
    validate_repeats(repeats, order_seed)
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if directory == source or directory.is_relative_to(source):
        raise ExperimentError("repeat output must be outside the immutable source directory")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ExperimentError(f"{directory}: output already contains evidence; choose a new batch directory")
    import cv2
    import numpy as np
    from .algorithms import predict
    from .dataset import load_dataset, read_image, safe_path, write_json
    from .runner import environment, write_csv

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    context = prepare_source(source)
    current_environment = environment()
    for library in ("numpy", "opencv", "matplotlib"):
        if current_environment[library] != context["summary"]["environment"].get(library):
            raise ExperimentError(f"{library}: repeat environment differs from the frozen first-run protocol")
    orders = []
    for round_index in range(1, repeats + 1):
        seed = [order_seed, round_index]
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        order = [context["test_rows"][int(index)]["image_id"] for index in generator.permutation(len(context["test_rows"]))]
        orders.append({"round": round_index, "seed": seed, "image_ids": order,
                       "order_sha256": hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()})
    # Existing first-run failures are an audit target, not a new selection set.
    selected_failure_ids = []
    for condition, _ in context["config"].conditions:
        for algorithm in ALGORITHMS:
            failed = sorted(row["image_id"] for row in context["first"].values()
                            if row["condition"] == condition and row["algorithm"] == algorithm and row["correct"] == "0")
            if failed and failed[0] not in selected_failure_ids:
                selected_failure_ids.append(failed[0])
    protocol = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "repeats": repeats, "order_seed": order_seed,
        "conditions": context["config"].conditions, "test_images": len(context["test_rows"]),
        "expected_prediction_calls": len(context["test_rows"]) * 2 * repeats,
        "source_evidence_sha256": context["evidence_sha256"], "source_core_sha256": context["code_sha256"],
        "core_source_matches_first_run": True, "manifest_matches_first_run": True, "parameters_match_first_run": True,
        "frozen_parameters_sha256": context["parameters_sha256"], "manifest_sha256": context["manifest_sha256"],
        "parameters": context["frozen"]["parameters"], "environment": current_environment,
        "warmup_images_per_algorithm_per_round": min(10, len(context["dev_rows"])),
        "warmup_image_ids": [row["image_id"] for row in context["dev_rows"][:10]],
        "round_orders": orders,
        "algorithm_order": "AB when (zero-based image position + one-based round) is even, BA otherwise",
        "timing": "perf_counter_ns around exactly predict(image, frozen_parameters); input check/preprocessing/contours/classification included; decoding/evaluation/warmup/CSV/plots excluded; decoded uint8 arrays preloaded read-only; OpenCV 1 thread/OpenCL off",
        "test_image_identity": [{key: row[key] for key in ("image_id", "original_id", "condition", "label", "path", "pixel_sha256", "png_sha256")} for row in context["test_rows"]],
        "failure_example_selection": {"rule": "for each predeclared condition and baseline then improved, choose lexicographically first incorrect first-run image; de-duplicate; at most 10; selection frozen before repeats", "image_ids": selected_failure_ids},
        "statistics": "per-round condition/algorithm means; across-round mean/sample SD(ddof=1)/min/max; pooled call p95 uses linear percentile; repeated accuracy uses the same unique image denominator; no significance claim",
        "limitations": "600 condition variants come from 120 originals; repeats add no independent originals. Occlusion20 first-run net gain was one image, not five new successes."}
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", protocol)
    (directory / "frozen_parameters.json").write_bytes((source / "frozen_parameters.json").read_bytes())
    pixels = {row["image_id"]: read_image(safe_path(source, row["path"])) for row in context["test_rows"]}
    warmup = [read_image(safe_path(source, row["path"])) for row in context["dev_rows"][:10]]
    for image in pixels.values():
        image.flags.writeable = False
    image_by_id = {row["image_id"]: row for row in context["test_rows"]}
    rows = []
    measurement_started = perf_counter_ns()
    raw_path = directory / "predictions.csv"
    raw_fields = ["round", "round_order_seed", "image_order_index", "algorithm_order_index", "image_id", "original_id",
                  "condition", "label", "algorithm", "prediction", "first_prediction", "correct", "changed_prediction",
                  "elapsed_ns", "elapsed_ms", "pixel_sha256", "png_sha256", "parameters_sha256"]
    with raw_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        for order in orders:
            for image in warmup:
                for algorithm in ALGORITHMS:
                    predict(image, context["parameters"][algorithm])
            round_rows = []
            for position, image_id in enumerate(order["image_ids"]):
                original = image_by_id[image_id]
                algorithm_order = ALGORITHMS if (position + order["round"]) % 2 == 0 else tuple(reversed(ALGORITHMS))
                image = pixels[image_id]
                for algorithm_position, algorithm in enumerate(algorithm_order):
                    started = perf_counter_ns()
                    prediction = predict(image, context["parameters"][algorithm])
                    elapsed_ns = perf_counter_ns() - started
                    first = context["first"][(image_id, algorithm)]
                    round_rows.append({"round": order["round"], "round_order_seed": json.dumps(order["seed"]),
                        "image_order_index": position, "algorithm_order_index": algorithm_position,
                        "image_id": image_id, "original_id": original["original_id"], "condition": original["condition"],
                        "label": original["label"], "algorithm": algorithm, "prediction": prediction,
                        "first_prediction": first["prediction"], "correct": int(prediction == original["label"]),
                        "changed_prediction": int(prediction != first["prediction"]), "elapsed_ns": elapsed_ns,
                        "elapsed_ms": elapsed_ns / 1e6, "pixel_sha256": original["pixel_sha256"],
                        "png_sha256": original["png_sha256"], "parameters_sha256": context["parameters_sha256"]})
            writer.writerows(round_rows)
            stream.flush()
            rows.extend(round_rows)
            write_json(directory / "progress.json", {"completed_rounds": order["round"], "repeats": repeats,
                       "raw_predictions": len(rows), "utc": datetime.now(timezone.utc).isoformat()})
    measurement_wall_seconds = (perf_counter_ns() - measurement_started) / 1e9
    per_round = round_statistics(rows)
    aggregate = aggregate_rounds(per_round, rows, repeats)
    pairs = paired_errors(rows)
    write_csv(directory / "per_run.csv", per_round)
    write_csv(directory / "summary.csv", aggregate)
    write_csv(directory / "paired_errors.csv", pairs)
    failures = [row for row in rows if not row["correct"] or row["changed_prediction"]]
    with (directory / "failures.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(failures)
    # Recheck source evidence and all image hashes after measuring, not within calls.
    load_dataset(source, context["config"])
    after_hashes = {name: digest(source / name) for name in context["evidence_sha256"]}
    after_code = {name: digest(Path(__file__).parent / name) for name in context["code_sha256"]}
    unchanged = after_hashes == context["evidence_sha256"] and after_code == context["code_sha256"]
    changed_count = sum(row["changed_prediction"] for row in rows)
    summary = {"format_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "source": str(source), "output": str(directory), "repeats": repeats,
               "test_images": len(context["test_rows"]), "prediction_count": len(rows),
               "changed_predictions": changed_count, "source_and_core_unchanged": unchanged,
               "status": "ok" if unchanged and not changed_count else "prediction_or_source_changed",
               "measurement_loop_wall_seconds": measurement_wall_seconds,
               "measurement_loop_wall_scope": "round warmup, predictions, Python bookkeeping and per-round CSV/progress writes; excludes source audit, preload, final statistics/plots; not per-image predictor timing",
               "environment": current_environment, "protocol_sha256": digest(directory / "protocol.json"),
               "source_evidence_sha256_after": after_hashes, "source_core_sha256_after": after_code,
               "parameters_sha256": context["parameters_sha256"], "manifest_sha256": context["manifest_sha256"],
               "results": aggregate, "paired_errors_per_round": pairs,
               "failure_call_count": len(failures), "failure_unique_image_algorithm_pairs": len({(row["image_id"], row["algorithm"]) for row in failures}),
               "interpretation": "Accuracy repeats verify deterministic predictions, not generalization or statistical significance. Report same 120 unique images per condition. Occlusion20 net gain remains one unique image. Keep unknowns and method regressions in the denominator; timing is specific to this environment."}
    write_json(directory / "summary.json", summary)
    write_json(directory / "environment.json", current_environment)
    if plots:
        make_plots(source, directory, protocol, aggregate, pairs, rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Repeat frozen marker predictions without generation or parameter selection")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "artifacts" / "marker_experiment")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "marker_repeat5", help="parent directory for a new unique batch")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=510005)
    args = parser.parse_args(argv)
    try:
        validate_repeats(args.repeats, args.order_seed)
        batch = args.output / ("batch_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        summary = repeat_experiment(args.source, batch, args.repeats, args.order_seed)
        if summary["status"] != "ok":
            print(f"REPEAT_INCOMPLETE changed_predictions={summary['changed_predictions']} output={batch.resolve()}")
            return 1
        print(f"REPEAT_OK repeats={summary['repeats']} predictions={summary['prediction_count']} changed=0 output={batch.resolve()}")
        return 0
    except ImportError as error:
        print(f"REPEAT_DEPENDENCY_ERROR: {error}. Install: python -m pip install -r requirements-experiments.txt", file=sys.stderr)
        return 3
    except (ExperimentError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"REPEAT_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
