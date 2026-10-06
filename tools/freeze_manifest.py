"""Fingerprint the stage-four business implementation before stage-five measurements."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER_MODULES = ("__init__.py", "__main__.py", "algorithms.py", "config.py", "dataset.py", "runner.py")
EXTRA_FILES = ("requirements.txt", "requirements-experiments.txt",
               "tools/validate_scenarios.py", "tools/build_demo_gif.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(project_root: Path) -> list[str]:
    root = Path(project_root).resolve()
    paths = [*root.joinpath("sim_app").rglob("*.py"), *root.joinpath("configs").glob("*.json")]
    paths.extend(root / "marker_experiment" / name for name in MARKER_MODULES)
    paths.extend(root / name for name in EXTRA_FILES)
    names = []
    for path in paths:
        # A fingerprint must never read an out-of-project symlink target.
        path.resolve().relative_to(root)
        names.append(path.relative_to(root).as_posix())
    return sorted(set(names))


def create_manifest(project_root: Path, output_root: Path) -> Path:
    root = Path(project_root).resolve()
    names = inventory(root)
    if not names or not (root / "sim_app" / "simulation.py").is_file():
        raise ValueError("Expected the existing simulation project")
    entries = [{"path": name, "bytes": (root / name).stat().st_size,
                "sha256": sha256(root / name)} for name in names]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    directory = Path(output_root) / f"freeze_{stamp}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_at_creation": str(root), "file_count": len(entries), "files": entries,
        "scope": "Existing business simulation, historical/business scene JSON, six original pixel experiment modules, dependency lists and stage-four tools",
        "permitted_stage5_work": "New measurement tools/tests and benchmark artifacts; living README/report and stage-five documents. No new business behavior.",
        "limitations": "A content fingerprint is not a Git release, external publication, dependency installation audit, or cross-computer acceptance.",
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def verify_manifest(manifest_path: Path, project_root: Path = PROJECT_ROOT) -> dict:
    root = Path(project_root).resolve()
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("format_version")) is not int or data.get("format_version") != 1 or not isinstance(data.get("files"), list):
        raise ValueError("Unsupported fingerprint manifest")
    expected = {}
    for row in data["files"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid file entry")
        name = row.get("path")
        if not isinstance(name, str) or name in expected:
            raise ValueError("Invalid or duplicate manifest path")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Manifest path must be relative and remain inside the project")
        (root / name).resolve().relative_to(root)
        if not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("Invalid file digest")
        if type(row.get("bytes")) is not int or row["bytes"] < 0:
            raise ValueError("Invalid file size")
        expected[name] = row
    if type(data.get("file_count")) is not int or data.get("file_count") != len(expected) or not expected:
        raise ValueError("Manifest file count does not match")
    current = set(inventory(root))
    missing, changed = [], []
    for name, row in expected.items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            changed.append(name)
    added = sorted(current - set(expected))
    return {"status": "PASS" if not (missing or changed or added) else "FAIL",
            "manifest": str(Path(manifest_path).resolve()), "checked_files": len(expected),
            "changed": changed, "missing": missing, "added_business_files": added}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "stage5_freeze")
    args = parser.parse_args(argv)
    if args.command == "verify" and args.manifest is None:
        parser.error("verify requires --manifest")
    try:
        if args.command == "create":
            path = create_manifest(args.project_root, args.output_dir)
            print(f"FREEZE_CREATED: {path.resolve()}")
            return 0
        result = verify_manifest(args.manifest, args.project_root)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        print(f"FREEZE_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
