"""A frozen business fingerprint must detect edits, omissions and unsafe paths."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.freeze_manifest import EXTRA_FILES, MARKER_MODULES, create_manifest, verify_manifest


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        names = ["sim_app/simulation.py", "sim_app/app.py", "configs/scene.json"]
        names.extend(f"marker_experiment/{name}" for name in MARKER_MODULES)
        names.extend(EXTRA_FILES)
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture " + name, encoding="utf-8")
        self.manifest = create_manifest(self.root, Path(self.temp.name) / "output")

    def test_fingerprint_is_portable_and_new_measurements_are_allowed(self):
        (self.root / "marker_experiment/repeat.py").write_text("measurement", encoding="utf-8")
        (self.root / "tools/benchmark_simulation.py").write_text("measurement", encoding="utf-8")
        (self.root / "README.md").write_text("updated report", encoding="utf-8")
        result = verify_manifest(self.manifest, self.root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["checked_files"], 13)

    def test_same_size_edit_removed_file_and_added_business_module_are_detected(self):
        path = self.root / "sim_app/app.py"
        text = path.read_text(encoding="utf-8")
        path.write_text("X" + text[1:], encoding="utf-8")
        (self.root / "configs/scene.json").unlink()
        (self.root / "sim_app/new_feature.py").write_text("feature", encoding="utf-8")
        result = verify_manifest(self.manifest, self.root)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["changed"], ["sim_app/app.py"])
        self.assertEqual(result["missing"], ["configs/scene.json"])
        self.assertEqual(result["added_business_files"], ["sim_app/new_feature.py"])

    def test_parent_path_in_manifest_is_rejected(self):
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        data["files"][0]["path"] = "../outside.txt"
        self.manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inside the project"):
            verify_manifest(self.manifest, self.root)

    def test_duplicate_paths_and_count_mismatch_are_rejected(self):
        original = json.loads(self.manifest.read_text(encoding="utf-8"))
        original["files"].append(original["files"][0])
        self.manifest.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_manifest(self.manifest, self.root)

    def test_malformed_structure_size_digest_and_count_are_rejected(self):
        original = json.loads(self.manifest.read_text(encoding="utf-8"))
        for value in ([], {"format_version": 1, "files": [None]}):
            with self.subTest(value=value):
                self.manifest.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_manifest(self.manifest, self.root)
        for key, value in (("bytes", True), ("bytes", -1), ("sha256", "z" * 64)):
            data = json.loads(json.dumps(original))
            data["files"][0][key] = value
            self.manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                verify_manifest(self.manifest, self.root)
        original["file_count"] += 1
        self.manifest.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "count"):
            verify_manifest(self.manifest, self.root)


if __name__ == "__main__":
    unittest.main()
