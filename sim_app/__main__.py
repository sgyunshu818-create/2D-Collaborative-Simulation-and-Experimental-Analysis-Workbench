"""python -m sim_app entry point."""

import argparse
from pathlib import Path

from .app import DEFAULT_SCENE, run, run_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="二维协同仿真与实验分析：运行、记录及回放")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Path to scene JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true", help="Run a mission and save preview/final screenshots")
    mode.add_argument("--headless", action="store_true", help="Run and export without importing Pygame")
    mode.add_argument("--replay", type=Path, help="Play a saved run.json without re-running the simulation")
    mode.add_argument("--replay-check", type=Path, help="Validate a recording and read its final snapshot without Pygame")
    parser.add_argument("--max-steps", type=int, default=7200, help="Logic step limit for headless/smoke runs (default: 7200)")
    parser.add_argument("--full-record", action="store_true",
                        help="Record every logic step of a rules scene instead of sampling; much larger run.json")
    parser.add_argument("--output-dir", type=Path, help="Parent directory for exported GUI/headless runs")
    parser.add_argument("--capture-dir", type=Path, help="Optional parent directory for recording this app's GUI frames")
    parser.add_argument("--replay-time", type=float, help="Open a recording at this simulation time (seconds)")
    parser.add_argument("--replay-event", type=int, help="Open a recording at this zero-based original event index")
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.replay or args.replay_check:
        return run_replay(args.replay or args.replay_check, check_only=bool(args.replay_check), capture_dir=args.capture_dir, seek_time=args.replay_time, event_index=args.replay_event)
    return run(args.scene, smoke_test=args.smoke_test, headless=args.headless,
               max_steps=args.max_steps, output_dir=args.output_dir, capture_dir=args.capture_dir,
               full_record=args.full_record)


if __name__ == "__main__":
    raise SystemExit(main())
