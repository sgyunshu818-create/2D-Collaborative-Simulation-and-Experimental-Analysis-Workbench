"""Frozen-method repetitions retain failures and can be audited from raw calls."""

import copy
import csv
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from marker_experiment import ExperimentError
from marker_experiment.config import config_from_data
from marker_experiment.repeat import (aggregate_rounds, digest, main, paired_errors,
                                      round_statistics, validate_repeats)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_IMAGE_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "cv2", "matplotlib"))
if HAS_IMAGE_DEPS:
    from marker_experiment.algorithms import predict
    from marker_experiment.dataset import pixel_hash
    from marker_experiment.repeat import prepare_source, repeat_experiment
    from marker_experiment.runner import run_experiment


def example_measurements():
    rows = []
    for round_index in (1, 2):
        for image_id, label, index in (("a", "circle", 1), ("b", "square", 3)):
            for algorithm in ("baseline", "improved"):
                prediction = "triangle" if image_id == "b" and algorithm == "improved" else label
                elapsed_ns = (index + int(algorithm == "improved")) * round_index * 1_000_000
                rows.append({"round": round_index, "image_id": image_id, "condition": "normal",
                             "algorithm": algorithm, "label": label, "prediction": prediction,
                             "first_prediction": prediction, "correct": int(prediction == label),
                             "changed_prediction": 0, "elapsed_ns": elapsed_ns,
                             "elapsed_ms": elapsed_ns / 1e6, "pixel_sha256": image_id * 64})
    return rows


class MarkerRepeatStatisticsTests(unittest.TestCase):
    def test_statistics_use_round_means_sample_sd_and_unique_images(self):
        raw = example_measurements()
        per_round = round_statistics(raw)
        results = aggregate_rounds(per_round, raw, repeats=2)
        baseline = next(row for row in results if row["algorithm"] == "baseline")
        self.assertEqual(baseline["round_mean_ms"], [2.0, 4.0])
        self.assertEqual(baseline["round_mean_ms_mean"], 3.0)
        self.assertAlmostEqual(baseline["round_mean_ms_sample_std"], math.sqrt(2))
        self.assertEqual((baseline["round_mean_ms_min"], baseline["round_mean_ms_max"]), (2, 4))
        self.assertEqual((baseline["unique_test_images"], baseline["measured_calls"]), (2, 4))
        self.assertEqual(baseline["correct_per_round"], [2, 2])
        self.assertAlmostEqual(baseline["pooled_call_p95_ms"], 5.55)

    def test_paired_errors_preserve_regressions_and_every_failure(self):
        paired = paired_errors(example_measurements())
        for row in paired:
            self.assertEqual(row["both_correct"], 1)
            self.assertEqual(row["baseline_only_correct"], 1)
            self.assertEqual(row["improved_only_correct"], 0)
            self.assertEqual(row["net_gain_unique_images"], -1)
            self.assertEqual(row["unique_test_images"], 2)

    def test_duplicate_rows_changed_flags_and_invalid_timings_are_rejected(self):
        original = example_measurements()
        cases = [lambda rows: rows.append(copy.deepcopy(rows[0])),
                 lambda rows: rows[0].update(correct=0),
                 lambda rows: rows[0].update(changed_prediction=1),
                 lambda rows: rows[0].update(elapsed_ns=-1),
                 lambda rows: rows[0].update(elapsed_ms=float("nan")),
                 lambda rows: rows[0].update(elapsed_ms=123)]
        for index, change in enumerate(cases):
            raw = copy.deepcopy(original)
            change(raw)
            with self.subTest(case=index), self.assertRaises(ExperimentError):
                round_statistics(raw)

    def test_missing_rounds_and_unpaired_images_are_rejected(self):
        raw = example_measurements()
        with self.assertRaises(ExperimentError):
            aggregate_rounds(round_statistics(raw[:-4]), raw[:-4], repeats=2)
        with self.assertRaises(ExperimentError):
            paired_errors(raw[:-1])
        raw[1]["pixel_sha256"] = "different"
        with self.assertRaises(ExperimentError):
            paired_errors(raw)

    def test_repeat_count_and_order_seed_are_strict(self):
        for value in (0, -1, True, 1.5, 101):
            with self.subTest(repeats=value), self.assertRaises(ExperimentError):
                validate_repeats(value, 1)
        for value in (-1, True, 2 ** 63):
            with self.subTest(seed=value), self.assertRaises(ExperimentError):
                validate_repeats(5, value)

    def test_cli_help_missing_dependencies_and_bad_count_are_readable(self):
        for arguments, expected in ((["--help"], 0), (["--repeats", "0"], 2), (["--repeats", "1"], 3)):
            result = subprocess.run([sys.executable, "-S", "-m", "marker_experiment.repeat", *arguments],
                                    cwd=PROJECT_ROOT, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            if expected == 3:
                self.assertIn("requirements-experiments.txt", result.stderr)

    def test_changed_prediction_status_exits_nonzero(self):
        with patch("marker_experiment.repeat.repeat_experiment", return_value={"status": "prediction_or_source_changed", "changed_predictions": 1}), \
             patch("builtins.print"):
            self.assertEqual(main(["--repeats", "1"]), 1)


@unittest.skipUnless(HAS_IMAGE_DEPS, "image repeat checks require .experiment-venv and requirements-experiments.txt")
class MarkerRepeatImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.config = config_from_data({"image_size": 64, "dev_seed": 771, "test_seed": 772,
                                       "dev_per_class": 2, "test_per_class": 34,
                                       "baseline_thresholds": [128], "polygon_epsilons": [0.035],
                                       "circularity_thresholds": [0.72], "blur_kernels": [5]})
        cls.source = cls.root / "first"
        run_experiment(cls.config, cls.source, make_plots=False)
        cls.output = cls.root / "repeat"
        cls.summary = repeat_experiment(cls.source, cls.output, repeats=2, order_seed=901, plots=True)
        cls.protocol = json.loads((cls.output / "protocol.json").read_text(encoding="utf-8"))
        with (cls.output / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            cls.raw = list(csv.DictReader(stream))

    def test_real_repeated_predictions_match_first_frozen_inputs_and_source_files(self):
        self.assertEqual(self.summary["prediction_count"], 2040)
        self.assertEqual(self.summary["changed_predictions"], 0)
        self.assertTrue(self.summary["source_and_core_unchanged"])
        self.assertTrue(all(row["prediction"] == row["first_prediction"] for row in self.raw))
        self.assertEqual(digest(self.output / "frozen_parameters.json"), digest(self.source / "frozen_parameters.json"))
        self.assertEqual(self.summary["source_evidence_sha256_after"], self.protocol["source_evidence_sha256"])
        self.assertEqual(self.summary["source_core_sha256_after"], self.protocol["source_core_sha256"])

    def test_protocol_and_explicit_seeded_image_orders_exist_before_any_predictor_call(self):
        target = self.root / "observed"
        count = 0
        def observed(image, parameters):
            nonlocal count
            count += 1
            if count == 1:
                protocol = json.loads((target / "protocol.json").read_text(encoding="utf-8"))
                self.assertEqual(protocol["repeats"], 1)
                self.assertEqual(len(protocol["round_orders"][0]["image_ids"]), 510)
            self.assertFalse(image.flags.writeable if count > 20 else False)
            return predict(image, parameters)
        with patch("marker_experiment.algorithms.predict", side_effect=observed), \
             patch("marker_experiment.runner.select_parameters", side_effect=AssertionError("must not choose parameters")), \
             patch("marker_experiment.runner.run_experiment", side_effect=AssertionError("must not run first pipeline")), \
             patch("marker_experiment.dataset.generate_dataset", side_effect=AssertionError("must not generate images")):
            result = repeat_experiment(self.source, target, repeats=1, order_seed=902, plots=False)
        self.assertEqual(count, 1020 + 20)
        self.assertEqual(result["prediction_count"], 1020)
        self.assertEqual(result["changed_predictions"], 0)

    def test_raw_times_independently_recompute_per_round_and_cross_round_statistics(self):
        groups = defaultdict(list)
        for row in self.raw:
            groups[(int(row["round"]), row["condition"], row["algorithm"])].append(row)
            self.assertAlmostEqual(float(row["elapsed_ms"]), int(row["elapsed_ns"]) / 1e6, places=12)
        with (self.output / "per_run.csv").open(encoding="utf-8-sig", newline="") as stream:
            per_run = list(csv.DictReader(stream))
        self.assertEqual(len(per_run), 20)
        for row in per_run:
            raw = groups[(int(row["round"]), row["condition"], row["algorithm"])]
            self.assertEqual(int(row["n"]), len(raw))
            self.assertEqual(int(row["correct"]), sum(item["label"] == item["prediction"] for item in raw))
            self.assertAlmostEqual(float(row["mean_ms"]), statistics.fmean(float(item["elapsed_ms"]) for item in raw), places=12)
            confusion = Counter((item["label"], item["prediction"]) for item in raw)
            for label, predictions in json.loads(row["confusion_matrix"]).items():
                for prediction, amount in predictions.items():
                    self.assertEqual(amount, confusion[(label, prediction)])
        for summary in self.summary["results"]:
            means = [statistics.fmean(float(row["elapsed_ms"]) for row in groups[(r, summary["condition"], summary["algorithm"])])
                     for r in (1, 2)]
            self.assertEqual(summary["unique_test_images"], 102)
            self.assertEqual(summary["measured_calls"], 204)
            self.assertAlmostEqual(summary["round_mean_ms_mean"], statistics.fmean(means), places=12)
            self.assertAlmostEqual(summary["round_mean_ms_sample_std"], statistics.stdev(means), places=12)

    def test_frozen_parameter_and_core_source_tampering_fail_before_creating_output(self):
        corrupt = self.root / "corrupt"
        shutil.copytree(self.source, corrupt)
        frozen_path = corrupt / "frozen_parameters.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen["parameters"]["baseline"]["threshold"] = 100
        frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
        target = self.root / "should_not_exist"
        with self.assertRaisesRegex(ExperimentError, "parameter hash"):
            repeat_experiment(corrupt, target, repeats=1, plots=False)
        self.assertFalse(target.exists())
        original_digest = digest
        def mismatched(path):
            return "0" * 64 if Path(path).name == "algorithms.py" else original_digest(path)
        with patch("marker_experiment.repeat.digest", side_effect=mismatched):
            with self.assertRaisesRegex(ExperimentError, "source differs"):
                prepare_source(self.source)

    def test_existing_batch_and_output_inside_source_are_refused_without_modifying_evidence(self):
        before = digest(self.output / "summary.json")
        with self.assertRaisesRegex(ExperimentError, "already contains"):
            repeat_experiment(self.source, self.output, repeats=1, plots=False)
        self.assertEqual(digest(self.output / "summary.json"), before)
        inside = self.source / "forbidden_repeat"
        with self.assertRaisesRegex(ExperimentError, "immutable source"):
            repeat_experiment(self.source, inside, repeats=1, plots=False)
        self.assertFalse(inside.exists())

    def test_changed_predictions_are_preserved_with_failure_analysis_instead_of_suppressed(self):
        manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        target_hash = next(row["pixel_sha256"] for row in manifest["images"]
                           if row["image_id"] == "test_circle_0000--normal")
        def altered(image, parameters):
            result = predict(image, parameters)
            return "triangle" if parameters.algorithm == "baseline" and pixel_hash(image) == target_hash else result
        with patch("marker_experiment.algorithms.predict", side_effect=altered):
            result = repeat_experiment(self.source, self.root / "changed", repeats=1, plots=False)
        self.assertEqual(result["changed_predictions"], 1)
        self.assertEqual(result["status"], "prediction_or_source_changed")
        self.assertTrue(result["source_and_core_unchanged"])
        with (self.root / "changed" / "failures.csv").open(encoding="utf-8-sig", newline="") as stream:
            failures = list(csv.DictReader(stream))
        self.assertEqual(sum(row["changed_prediction"] == "1" for row in failures), 1)

    def test_predeclared_failure_images_and_both_result_figures_are_saved(self):
        from marker_experiment.dataset import read_image
        rules = self.protocol["failure_example_selection"]
        self.assertIn("first-run", rules["rule"])
        self.assertLessEqual(len(rules["image_ids"]), 10)
        first_rows = {}
        with (self.source / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                first_rows.setdefault(row["image_id"], []).append(row)
        self.assertTrue(all(any(row["correct"] == "0" for row in first_rows[image_id]) for image_id in rules["image_ids"]))
        for filename in ("repeat_processing_time.png", "failure_examples.png"):
            image = read_image(self.output / filename)
            self.assertGreater(image.shape[0], 400)
            self.assertGreater(image.shape[1], 800)


if __name__ == "__main__":
    unittest.main()
