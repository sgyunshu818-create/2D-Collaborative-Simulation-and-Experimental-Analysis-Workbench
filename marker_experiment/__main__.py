"""python -m marker_experiment generate|run|verify --config ... --output ..."""

import argparse
from pathlib import Path
import sys

from . import ExperimentError
from .config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent synthetic marker classification experiment")
    parser.add_argument("command", choices=("generate", "run", "verify"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "marker_experiment.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "marker_experiment")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        from .dataset import generate_dataset, load_dataset
        if args.command == "generate":
            manifest = generate_dataset(config, args.output)
        elif args.command == "verify":
            manifest = load_dataset(args.output, config)
        else:
            from .runner import run_experiment
            summary = run_experiment(config, args.output)
            print(f"EXPERIMENT_OK test_images={summary['test_images']} predictions={summary['prediction_count']} summary={(args.output / 'summary.json').resolve()}")
            return 0
        counts = {split: sum(row["split"] == split for row in manifest["images"]) for split in ("dev", "test")}
        print(f"DATASET_OK command={args.command} dev={counts['dev']} test={counts['test']} output={args.output.resolve()}")
        return 0
    except ImportError as error:
        print(f"EXPERIMENT_DEPENDENCY_ERROR: {error}. Install the separate environment with: python -m pip install -r requirements-experiments.txt", file=sys.stderr)
        return 3
    except (ExperimentError, OSError, ValueError) as error:
        print(f"EXPERIMENT_ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
