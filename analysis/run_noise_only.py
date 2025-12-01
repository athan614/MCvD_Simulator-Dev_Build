from pathlib import Path
import sys
import argparse
import yaml

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.run_final_analysis import (
    preprocess_config_full,
    ProgressManager,
    run_zero_signal_noise_analysis,
    DEFAULT_MODE_DISTANCES,
    DEFAULT_NM_RANGES,
)
from analysis.log_utils import setup_tee_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-signal noise measurements.")
    parser.add_argument("--mode", choices=["MoSK", "CSK", "Hybrid"], default="Hybrid", help="Modulation mode.")
    parser.add_argument("--noise-only-seeds", type=int, default=4, help="Number of seeds to measure.")
    parser.add_argument("--noise-only-seq-len", type=int, default=16, help="Sequence length for each noise run.")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration.")
    parser.add_argument("--skip-noise-sweep", action="store_true", help="Skip noise sweep.")
    parser.add_argument(
        "--progress",
        choices=["gui", "rich", "tqdm", "none"],
        default="tqdm",
        help="Progress backend (gui, rich, tqdm, none).",
    )
    parser.add_argument("--logdir", default=str(Path("results/logs")), help="Directory for log files.")
    parser.add_argument("--no-log", action="store_true", help="Disable file logging.")
    parser.add_argument("--fsync-logs", action="store_true", help="Force fsync on each write.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.no_log:
        setup_tee_logging(Path(args.logdir), prefix="run_noise_only", fsync=args.fsync_logs)
    else:
        print("[log] File logging disabled by --no-log")
    cfg_path = Path("config/default.yaml")
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = preprocess_config_full(yaml.safe_load(cfg_text))
    pm = ProgressManager(args.progress, gui_session_meta={"mode": args.mode, "resume": False})
    seeds = list(range(args.noise_only_seeds))
    run_zero_signal_noise_analysis(
        cfg=cfg,
        mode=args.mode,
        args=args,
        nm_values=DEFAULT_NM_RANGES[args.mode],
        lod_distance_grid=DEFAULT_MODE_DISTANCES[args.mode],
        seeds=seeds,
        data_dir=Path("results/data"),
        suffix="",
        pm=pm,
        hierarchy_supported=False,
        mode_key=None,
    )
    pm.stop()


if __name__ == "__main__":
    main()
