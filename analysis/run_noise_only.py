from pathlib import Path
import argparse
import yaml

from analysis.run_final_analysis import (
    preprocess_config_full,
    ProgressManager,
    run_zero_signal_noise_analysis,
    DEFAULT_MODE_DISTANCES,
    DEFAULT_NM_RANGES,
)

def main() -> None:
    cfg_path = Path("config/default.yaml")
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = preprocess_config_full(yaml.safe_load(cfg_text))
    mode = "Hybrid"  # or "MoSK"/"CSK"
    args = argparse.Namespace(
        skip_noise_sweep=False,
        noise_only_seeds=4,      # how many seeds to measure
        noise_only_seq_len=16,   # sequence length for each noise run
        recalibrate=False,
        progress="rich",
    )
    pm = ProgressManager(args.progress, gui_session_meta=None)
    seeds = list(range(args.noise_only_seeds))
    run_zero_signal_noise_analysis(
        cfg=cfg,
        mode=mode,
        args=args,
        nm_values=DEFAULT_NM_RANGES[mode],
        lod_distance_grid=DEFAULT_MODE_DISTANCES[mode],
        seeds=seeds,
        data_dir=Path("results/data"),
        suffix="",
        pm=pm,
        hierarchy_supported=False,
        mode_key=None,
    )

if __name__ == "__main__":
    main()
