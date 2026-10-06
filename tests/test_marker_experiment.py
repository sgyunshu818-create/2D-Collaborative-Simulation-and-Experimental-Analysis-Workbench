"""Isolation, pixel-only interfaces, paired evaluation and independent audits.

Core simulator environments intentionally lack optional experiment libraries.
Those environments run configuration/CLI checks and explicitly skip image
checks. Run this module in .experiment-venv to execute every image check.
"""

import copy
import csv
import dataclasses
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from marker_experiment import ExperimentError
from marker_experiment.config import config_from_data, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_EXPERIMENT_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("numpy", "cv2", "matplotlib"))
if HAS_EXPERIMENT_DEPENDENCIES:
    import cv2
    import numpy as np
    from marker_experiment.algorithms import Parameters, predict
    from marker_experiment.dataset import (generate_dataset, load_dataset, pixel_hash,
                                          read_image, render_original, sha256, validate_manifest)
    from marker_experiment.runner import (evaluate_test, make_figures, run_experiment,
                                         select_parameters)


class MarkerConfigurationTests(unittest.TestCase):
    def test_default_configuration_declares_disjoint_seeds_and_at_least_100_test_images_per_condition(self):
        config = load_config(PROJECT_ROOT / "configs" / "marker_experiment.json")
        self.assertNotEqual(config.dev_seed, config.test_seed)
        self.assertGreaterEqual(config.test_per_class * 3, 100)
        self.assertEqual(len(config.conditions), 5)
        self.assertEqual(config.sha256, load_config(PROJECT_ROOT / "configs" / "marker_experiment.json").sha256)

    def test_invalid_parameters_are_rejected_with_a_field(self):
        cases = (("dev_seed", True), ("test_seed", 171104), ("test_per_class", 33),
                 ("image_size", 32), ("noise_sigmas", [25]), ("noise_sigmas", [25, 25]),
                 ("noise_sigmas", [25, float("nan")]), ("noise_sigmas", [25, 10 ** 1000]),
                 ("occlusion_fractions", [0, 0.2]), ("occlusion_fractions", [0.2, 1]),
                 ("blur_kernels", [4]), ("polygon_epsilons", [0]),
                 ("baseline_thresholds", [True]), ("extra", 1))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ExperimentError) as caught:
                    config_from_data({field: value}, "config.json")
                self.assertIn(field if field != "extra" else "unknown fields", str(caught.exception))

    def test_bad_json_missing_files_and_utf8_errors_are_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            for content in (b"{", b"\xff", b"[]"):
                path.write_bytes(content)
                with self.assertRaises(ExperimentError) as caught:
                    load_config(path)
                self.assertIn(str(path), str(caught.exception))
            with self.assertRaises(ExperimentError):
                load_config(Path(directory) / "absent.json")

    def test_cli_missing_optional_dependencies_has_install_instructions(self):
        result = subprocess.run([sys.executable, "-S", "-m", "marker_experiment", "run"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 3)
        self.assertIn("requirements-experiments.txt", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_help_and_bad_config_work_without_importing_image_dependencies(self):
        result = subprocess.run([sys.executable, "-S", "-m", "marker_experiment", "--help"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, "-S", "-m", "marker_experiment", "run",
                                 "--config", "missing_marker_config.json"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 2)
        self.assertIn("EXPERIMENT_ERROR", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


@unittest.skipUnless(HAS_EXPERIMENT_DEPENDENCIES,
                     "optional image checks require .experiment-venv and requirements-experiments.txt")
class MarkerImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        # Test-only seeds never inspect the real experiment's final-test pixels.
        cls.config = config_from_data({"image_size": 64, "dev_seed": 123, "test_seed": 456,
                                       "dev_per_class": 2, "test_per_class": 34,
                                       "baseline_thresholds": [128], "polygon_epsilons": [0.025],
                                       "circularity_thresholds": [0.72], "blur_kernels": [5]})
        cls.output = cls.root / "run"
        cls.summary = run_experiment(cls.config, cls.output, make_plots=False)
        cls.manifest = load_dataset(cls.output, cls.config)
        cls.parameters = {algorithm: Parameters(**data) for algorithm, data in
                          json.loads((cls.output / "frozen_parameters.json").read_text(encoding="utf-8"))["parameters"].items()}

    def test_originals_and_all_variants_are_isolated_by_split_seed_identity_and_hash(self):
        originals = self.manifest["originals"]
        self.assertEqual(len(originals), 108)
        self.assertEqual(len({row["original_id"] for row in originals}), 108)
        self.assertEqual(len({row["pixel_sha256"] for row in originals}), 108)
        groups = defaultdict(list)
        for row in self.manifest["images"]:
            groups[row["original_id"]].append(row)
        for variants in groups.values():
            self.assertEqual(len(variants), 5)
            self.assertEqual(len({row["split"] for row in variants}), 1)
            self.assertEqual(len({row["original_pixel_sha256"] for row in variants}), 1)
        hashes = {split: {row["pixel_sha256"] for row in self.manifest["images"] if row["split"] == split}
                  for split in ("dev", "test")}
        self.assertFalse(hashes["dev"] & hashes["test"])
        counts = Counter(row["condition"] for row in self.manifest["images"] if row["split"] == "test")
        self.assertTrue(all(count == 102 for count in counts.values()))

    def test_same_configuration_reproduces_png_and_pixel_hashes(self):
        second = generate_dataset(self.config, self.root / "repeat")
        self.assertEqual([(row["image_id"], row["png_sha256"], row["pixel_sha256"])
                          for row in second["images"]],
                         [(row["image_id"], row["png_sha256"], row["pixel_sha256"])
                          for row in self.manifest["images"]])

    def test_occlusion_records_actual_hidden_foreground_fraction(self):
        for row in self.manifest["images"]:
            if row["condition"].startswith("occlusion_"):
                self.assertLess(abs(row["actual_occlusion"] - row["strength"]), 0.01)

    def test_predictors_classify_clean_geometry_without_mutating_pixels_or_reading_files(self):
        self.assertEqual(tuple(inspect.signature(predict).parameters), ("image", "parameters"))
        for label in ("circle", "square", "triangle"):
            pixels, _, _ = render_original(label, 128, [789, 7])
            before = pixels.copy()
            pixels.flags.writeable = False
            with patch("pathlib.Path.read_bytes", side_effect=AssertionError("predictor must only receive pixels")):
                for algorithm in ("baseline", "improved"):
                    self.assertEqual(predict(pixels, self.parameters[algorithm]), label)
            np.testing.assert_array_equal(pixels, before)
        with self.assertRaises(ValueError):
            predict({"label": "circle", "path": "circle.png"}, self.parameters["baseline"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.parameters["baseline"].threshold = 42

    def test_development_selection_refuses_final_test_rows_and_reads_dev_pixels_only(self):
        dev = [row for row in self.manifest["images"] if row["split"] == "dev"]
        test = [row for row in self.manifest["images"] if row["split"] == "test"]
        with self.assertRaises(ExperimentError):
            select_parameters(test, self.output, self.config)
        observed_hashes = []
        def observed(image, parameters):
            observed_hashes.append(pixel_hash(image))
            return predict(image, parameters)
        with patch("marker_experiment.runner.predict", side_effect=observed):
            select_parameters(dev, self.output, self.config)
        self.assertEqual(set(observed_hashes), {row["pixel_sha256"] for row in dev})
        self.assertFalse(set(observed_hashes) & {row["pixel_sha256"] for row in test})

    def test_both_algorithms_receive_the_same_decoded_pixel_array_for_each_test_image(self):
        rows = [row for row in self.manifest["images"] if row["split"] == "test"][:3]
        received = []
        def observed(image, parameters):
            received.append((id(image), pixel_hash(image), parameters.algorithm))
            return predict(image, parameters)
        with patch("marker_experiment.runner.predict", side_effect=observed):
            results = evaluate_test(rows, self.output, self.parameters, "fixed_hash")
        self.assertEqual(len(results), len(rows) * 2)
        for index, row in enumerate(rows):
            pair = received[index * 2:index * 2 + 2]
            self.assertEqual(pair[0][:2], pair[1][:2])
            self.assertEqual(pair[0][1], row["pixel_sha256"])
            self.assertEqual({item[2] for item in pair}, {"baseline", "improved"})

    def test_raw_csv_independently_recomputes_counts_errors_confusion_and_mean_time(self):
        with (self.output / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            raw = list(csv.DictReader(stream))
        self.assertEqual(len(raw), 1020)
        grouped = defaultdict(list)
        for row in raw:
            grouped[(row["algorithm"], row["condition"])].append(row)
            self.assertEqual(int(row["correct"]), int(row["label"] == row["prediction"]))
        for result in self.summary["results"]:
            rows = grouped[(result["algorithm"], result["condition"])]
            correct = sum(row["label"] == row["prediction"] for row in rows)
            self.assertEqual((result["n"], result["correct"], result["errors"]), (len(rows), correct, len(rows) - correct))
            self.assertEqual(result["accuracy"], correct / len(rows))
            self.assertAlmostEqual(result["mean_ms"], math.fsum(float(row["elapsed_ms"]) for row in rows) / len(rows), places=12)
            confusion = Counter((row["label"], row["prediction"]) for row in rows)
            for label, column in result["confusion_matrix"].items():
                for prediction, count in column.items():
                    self.assertEqual(count, confusion[(label, prediction)])

    def test_frozen_parameters_precede_final_test_and_hash_links_every_prediction(self):
        frozen_path = self.output / "frozen_parameters.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.assertEqual(frozen["selection_split"], "dev")
        test_ids = {row["image_id"] for row in self.manifest["images"] if row["split"] == "test"}
        self.assertFalse(test_ids & {row["image_id"] for row in frozen["development_images"]})
        protocol = json.loads((self.output / "protocol.json").read_text(encoding="utf-8"))
        kinds = [event["kind"] for event in protocol]
        self.assertLess(kinds.index("parameters_frozen"), kinds.index("test_evaluation_started"))
        frozen_hash = sha256(frozen_path.read_bytes())
        self.assertEqual(self.summary["parameters_sha256"], frozen_hash)
        with (self.output / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertTrue(all(row["parameters_sha256"] == frozen_hash for row in csv.DictReader(stream)))

    def test_manifest_split_seed_path_and_image_tampering_are_detected(self):
        mutations = [lambda m: m["images"][0].update(split="test"),
                     lambda m: m["images"][0].update(seed=[999, 0, 0, 1]),
                     lambda m: m["images"][0].update(path="../escape.png"),
                     lambda m: m["originals"][0].update(split=[]),
                     lambda m: m["images"][0].update(original_id=[])]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.subTest(mutation=index):
                with self.assertRaises(ExperimentError):
                    validate_manifest(changed, self.output, self.config)
        path = self.output / self.manifest["images"][0]["path"]
        original = path.read_bytes()
        try:
            path.write_bytes(original[:-4] + b"edit")
            with self.assertRaisesRegex(ExperimentError, "hash mismatch"):
                load_dataset(self.output, self.config)
        finally:
            path.write_bytes(original)

    def test_existing_results_are_preserved_and_cli_verification_succeeds(self):
        before = (self.output / "summary.json").read_bytes()
        with self.assertRaises(ExperimentError):
            run_experiment(self.config, self.output)
        with self.assertRaises(ExperimentError):
            generate_dataset(self.config, self.output)
        self.assertEqual((self.output / "summary.json").read_bytes(), before)
        config_path = self.root / "test_config.json"
        config_path.write_text(json.dumps(self.config.data()), encoding="utf-8")
        result = subprocess.run([sys.executable, "-m", "marker_experiment", "verify",
                                 "--config", str(config_path), "--output", str(self.output)],
                                cwd=PROJECT_ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DATASET_OK", result.stdout)

    def test_result_figures_are_actual_pngs_with_accuracy_time_and_aligned_examples(self):
        with (self.output / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        make_figures(self.output, self.manifest, self.summary["results"], predictions)
        for filename in ("accuracy.png", "processing_time.png", "examples.png"):
            image = read_image(self.output / filename)
            self.assertGreater(image.shape[0], 400)
            self.assertGreater(image.shape[1], 800)


if __name__ == "__main__":
    unittest.main()
