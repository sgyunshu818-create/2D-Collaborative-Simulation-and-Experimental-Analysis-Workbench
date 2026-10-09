"""Fingerprint every built-in scene so behaviour changes show up in review.

The digest covers recorded behaviour only: the scene input, the events, the
snapshots and the terminal result. Run metadata is deliberately excluded,
because it embeds a digest of the source tree and a wall-clock timestamp, so
including it would make the fingerprint change on every edit and hide the
differences this gate exists to surface.

Every scene runs through the same public recording path the application uses,
which keeps the gate honest: it measures what actually gets saved.

Usage:
    python tools/scene_fingerprints.py            # compare with the golden file
    python tools/scene_fingerprints.py --update   # rewrite the golden file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_app.recording import export_run
from sim_app.scene import SceneConfigError, load_scene
from sim_app.simulation import Simulation


CONFIG_DIR = PROJECT_ROOT / "configs"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden_fingerprints.json"

# Scenes without an end condition would otherwise advance forever; the horizon
# bounds them without making the comparison depend on a step budget.
HORIZON_SECONDS = 60.0
MAX_STEPS = 20000

# The image experiment is a separate pipeline with its own config schema.
EXCLUDED_CONFIGS = frozenset({"marker_experiment.json"})

# Fields that describe the machine or the source tree rather than the behaviour.
VOLATILE_KEYS = ("metadata", "source_scene")


def scene_files() -> list[Path]:
    """Return the built-in scene configs, in a stable order."""
    return sorted(
        path for path in CONFIG_DIR.glob("*.json") if path.name not in EXCLUDED_CONFIGS
    )


def advance_to_horizon(sim: Simulation) -> None:
    """Advance deterministically until the run ends or the horizon is reached."""
    fixed_dt = sim.scene.fixed_dt
    limit = min(MAX_STEPS, int(HORIZON_SECONDS / fixed_dt) + 1)
    for _ in range(limit):
        if sim.finished:
            break
        sim.advance(fixed_dt)


def recorded_payload(sim: Simulation, scene_path: Path) -> dict:
    """Save the run through the real exporter and return the parsed payload."""
    with tempfile.TemporaryDirectory() as directory:
        export_run(sim, directory, scene_path)
        return json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))


def behaviour_digest(payload: dict) -> str:
    """Digest only the facts a change in behaviour would move."""
    behavioural = {key: value for key, value in payload.items() if key not in VOLATILE_KEYS}
    canonical = json.dumps(
        behavioural, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def describe(sim: Simulation, scene_path: Path) -> dict:
    """Describe what a simulation would record, without running it further."""
    payload = recorded_payload(sim, scene_path)
    result = payload["result"]
    return {
        "sha256": behaviour_digest(payload),
        "steps": result["step"],
        "sim_time": round(result["time"], 9),
        "state": result["state"],
        "finish_reason": result.get("finish_reason", ""),
        "events": len(payload["events"]),
        "snapshots": len(payload["snapshots"]),
        "units": len(result["units"]),
    }


def measure(scene_path: Path) -> dict:
    """Run one scene from its config and describe the recorded result."""
    sim = Simulation(load_scene(scene_path))
    sim.start()
    advance_to_horizon(sim)
    return describe(sim, scene_path)


def measure_all() -> dict[str, dict]:
    return {path.name: measure(path) for path in scene_files()}


def load_golden() -> dict:
    if not GOLDEN_PATH.is_file():
        return {}
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def write_golden(measured: dict[str, dict]) -> None:
    payload = {"horizon_seconds": HORIZON_SECONDS, "scenes": measured}
    GOLDEN_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compare(measured: dict[str, dict], golden: dict) -> list[str]:
    """Return one readable line per scene whose recorded behaviour moved."""
    expected = golden.get("scenes", {})
    lines = []
    for name in sorted(set(measured) | set(expected)):
        if name not in expected:
            lines.append(f"{name}: new scene, not in the golden file")
        elif name not in measured:
            lines.append(f"{name}: gone from configs/ but still in the golden file")
        elif measured[name] != expected[name]:
            fields = [
                f"{key}: {expected[name].get(key)} -> {measured[name][key]}"
                for key in sorted(set(expected[name]) | set(measured[name]))
                if expected[name].get(key) != measured[name].get(key)
            ]
            lines.append(f"{name}: " + "; ".join(fields))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="rewrite the golden file from this run")
    arguments = parser.parse_args(argv)

    lines: list[str] = []
    measured: dict[str, dict] = {}
    for path in scene_files():
        try:
            measured[path.name] = measure(path)
        except SceneConfigError as error:
            lines.append(f"{path.name}: SCENE_ERROR {error}")

    if lines:
        for line in lines:
            print(line, file=sys.stderr)
        return 1

    if arguments.update:
        write_golden(measured)
        print(f"GOLDEN_UPDATED {GOLDEN_PATH} scenes={len(measured)}")
        for name, record in sorted(measured.items()):
            print(f"  {name:28s} {record['sha256'][:16]} steps={record['steps']:<6} state={record['state']}")
        return 0

    golden = load_golden()
    differences = compare(measured, golden)
    if differences:
        print("FINGERPRINTS_CHANGED", file=sys.stderr)
        for line in differences:
            print(f"  {line}", file=sys.stderr)
        print("Review each difference; if it is intended, run: python tools/scene_fingerprints.py --update",
              file=sys.stderr)
        return 1

    print(f"FINGERPRINTS_OK scenes={len(measured)} horizon={HORIZON_SECONDS:g}s")
    for name, record in sorted(measured.items()):
        print(f"  {name:28s} {record['sha256'][:16]} steps={record['steps']:<6} state={record['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
