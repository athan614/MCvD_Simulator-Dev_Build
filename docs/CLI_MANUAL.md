# CLI Manual

This reference enumerates every command-line flag available in the scripted workflows. Defaults match the current codebase (Python 3.11 build).

## `analysis/run_final_analysis.py`

| Flag(s) | Type | Default | Description |
| --- | --- | --- | --- |
| `--ablation-parallel` | flag | `false` | Hint from orchestrator: defer canonical CSV merges while both ablations run in parallel. |
| `--allow-ts-exceed` | flag | `false` | Allow Ts to exceed limits during LoD sweeps. |
| `--analytic-lod-bracket` | flag | `true` | Use Gaussian SER approximation for tighter LoD bracketing (experimental). |
| `--analytic-noise-all` | flag | `false` | Force analytic noise for both zero-signal and LoD stages (skip zero-signal measurements). |
| `--beast-mode` | flag | `false` | Use the aggressive P-core worker heuristic (P-cores minus safety margin). |
| `--cal-eps-rel` | float | `0.01` | Adaptive calibration convergence threshold (relative change, default: 0.01) |
| `--cal-min-samples` | int | `50` | Minimum samples per class for stable thresholds (default: 50) |
| `--cal-min-seeds` | int | `4` | Minimum seeds before early stopping can trigger (default: 4) |
| `--cal-patience` | int | `2` | Wait N iterations before stopping convergence (default: 2) |
| `--channel-profile` | choice | `tri` | Physical channel setup: tri (DA+SERO+CTRL), dual (DA+SERO), single (DA only). Choices: `tri, dual, single`. |
| `--csk-dual` | choice | `None` | Force dual-channel CSK combiner on/off for this run. Choices: `on, off`. |
| `--csk-level-scheme` | choice | `uniform` | CSK level mapping scheme. Choices: `uniform, zero-based`. |
| `--csk-target` | choice | `None` | Override CSK target channel (single-channel baselines). Choices: `DA, SERO`. |
| `--ctrl-auto` | flag | `false` | Enable adaptive CTRL on/off based on measured correlation. |
| `--ctrl-rho-min-abs` | float | `0.1` | Minimum absolute correlation threshold for CTRL (default: 0.10) |
| `--ctrl-snr-min-gain-db` | float | `0.0` | Minimum SNR gain in dB to keep CTRL enabled (default: 0.0) |
| `--debug-calibration` | flag | `false` | Print detailed calibration information. |
| `--decision-window-frac` | float | `None` | Decision window fraction for fraction_of_Ts policy. |
| `--decision-window-policy` | choice | `None` | Override decision window policy. Choices: `fixed, fraction_of_Ts, full_Ts`. |
| `--detector-mode` | choice | `zscore` | Detector statistic normalisation (default: zscore). Choices: `zscore, raw, whitened`. |
| `--disable-isi` | flag | `false` | Disable ISI (runtime). Default is enabled. |
| `--distance-sweep` | choice | `always` | Generate SER/SNR vs distance sweep: always (default), auto (only when distances available), or never. Choices: `always, auto, never`. |
| `--distance-sweep-nm` | float | `None` | Override Nm_per_symbol used when sweeping distance (default: median LoD Nm or config baseline). |
| `--distances` | list | `None` | Override LoD distance grid per mode. Example: --distances MoSK=25,35,45 --distances CSK=15,25. Use 'ALL=' to apply to all modes. |
| `--extreme-mode` | flag | `false` | Max out workers according to CPU detection (no safety margin). |
| `--force-noise-resample` | flag | `false` | Force rerunning zero-signal noise sweeps even if resume cache matches. |
| `--freeze-calibration` | flag | `false` | Reuse baseline thresholds/sigmas during parameter sweeps. |
| `--fsync-logs` | flag | `false` | Force fsync on each write. |
| `--guard-factor` | float | `None` | Override guard factor for ISI calculations. |
| `--guard-max-ts` | float | `0.0` | Cap guard-factor sweeps when symbol period exceeds this many seconds (0 disables). |
| `--guard-samples-cap` | float | `None` | Per-seed sample cap for guard-factor sweeps (0 disables cap) |
| `--inhibit-sleep` | flag | `false` | Prevent the OS from sleeping while the pipeline runs. |
| `--isi-memory-cap` | int | `None` | ISI memory cap in symbols. |
| `--isi-sweep` | choice | `always` | Run ISI trade-off sweep: always (default), auto (only when ISI enabled), or never. Choices: `auto, always, never`. |
| `--keep-display-on` | flag | `false` | Also keep the display awake (Windows/macOS) |
| `--lod-analytic-noise` | flag | `false` | Force analytic noise during LoD sweeps (ignore cached/frozen noise). |
| `--lod-debug` | flag | `false` | Enable verbose diagnostics for LoD sweeps (writes results/debug/*.jsonl) |
| `--lod-distance-concurrency` | int | `8` | How many distances to run concurrently in LoD sweep (default: 8). |
| `--lod-distance-timeout-s` | float | `7200.0` | Per-distance time budget during LoD analysis. <=0 disables timeout. |
| `--lod-max-nm` | int | `1000000` | Upper bound for Nm during LoD search (default: 1000000). |
| `--lod-num-seeds` | str | `<=100:6,<=150:8,>150:10` | LoD seed schedule: `N` uses fixed seeds; `min,max` interpolates from min at 25um to max at 200um; rule lists like '<=100:6,<=150:8,>150:10' are supported. Final LoD validation always uses the full seed set. |
| `--lod-screen-delta` | float | `0.0001` | Hoeffding screening significance (delta) for early-stop LoD tests. |
| `--lod-seq-len` | int | `250` | If set, temporarily override sequence_length during LoD search only. |
| `--lod-skip-retry` | flag | `false` | On resume, do not retry distances whose previous LoD attempt failed (keep NaN). |
| `--lod-validate-seq-len` | int | `None` | If set, override sequence_length during final LoD validation only (not search). |
| `--logdir` | value | `results/logs` | Directory for log files. |
| `--max-lod-validation-seeds` | int | `12` | Cap the number of seeds used for LoD validation (default: use all seeds). |
| `--max-symbol-duration-s` | float | `None` | Skip LoD search at distances where symbol period exceeds this limit (seconds). |
| `--max-ts-for-lod` | float | `None` | If set, skip LoD at distances whose dynamic Ts exceeds this (seconds). |
| `--max-workers` | int | `None` | Override the ProcessPool worker count (auto-heuristic otherwise). |
| `--merge-ablation-csvs` | flag | `false` | Merge per-ablation CSV branches into canonical outputs and exit. |
| `--merge-data-dir` | str | `None` | Override the data directory for --merge-ablation-csvs (expects results/data path). |
| `--merge-stages` | list | `None` | Limit --merge-ablation-csvs to specific stages (ser,lod,dist,isi). Choices: `ser, lod, dist, isi`. |
| `--min-ci-seeds` | int | `8` | Minimum seeds required before adaptive CI stopping can trigger. |
| `--min-decision-points` | int | `4` | Minimum time points for window guard (default: 4) |
| `--mode` | choice | `None` | Force a single modulation mode; defaults to MoSK when neither --mode nor --modes is provided. Choices: `MoSK, CSK, Hybrid, ALL`. |
| `--modes` | list | `None` | Run multiple modes in one command (e.g., `--modes CSK Hybrid`); use `all`/`*` for every modulation. Choices: `MoSK, CSK, Hybrid, all`. |
| `--nm-grid` | str | `""` | Comma-separated Nm values for SER sweeps (e.g., 200,500,1000,2000). If not provided, uses cfg['Nm_range'] from YAML. |
| `--no-log` | flag | `false` | Disable file logging. |
| `--noise-only-seeds` | int | `8` | Number of seeds for noise-only sweeps (default: 8). |
| `--noise-only-seq-len` | int | `200` | Sequence length for each noise-only run (default: 200). |
| `--nonlod-watchdog-secs` | int | `14400` | Timeout for non-LoD sweeps (guard/frontier, SER vs Nm, etc.); <=0 disables (default: 4h). |
| `--seed-timeout-retries` | int | `2` | Watchdog expirations allowed for a single seed before restarting the worker pool. |
| `--seed-timeout-restarts` | int | `2` | Maximum worker-pool restarts triggered by one seed before aborting the sweep. |
| `--nt-pairs` | str | `""` | CSV nt-pairs for CSK sweeps. |
| `--num-seeds` | int | `20` | Monte Carlo seed count per Nm/distance combination. |
| `--parallel-modes` | int | `1` | >1 to run MoSK/CSK/Hybrid concurrently (e.g., 3). |
| `--progress` | choice | `tqdm` | Progress UI backend. Choices: `tqdm, rich, gui, none`. |
| `--recalibrate` | flag | `false` | Rebuild detector thresholds even when cached JSON exists. |
| `--resume` | flag | `false` | Resume: skip finished values and append results as we go. |
| `--run-stages` | str | `all` | Comma/semicolon list of stages/aliases to run (1=SER/Nm+zero-noise, 2=LoD+SER/SNR vs distance, 3=Device FoM, 4=ISI/guard; 5-7 plots/tables via master). Use `all` or `*` for every stage; subsets run only the selected stages. |
| `--sequence-length` | int | `1000` | Symbols simulated per seed. |
| `--ser-refine` | flag | `false` | After coarse SER vs Nm sweep, auto-run a few Nm points that bracket the target SER. |
| `--ser-refine-points` | int | `4` | How many log-spaced Nm points to add between the bracketing Nm pair (default: 4). |
| `--ser-target` | float | `0.01` | Target SER for auto-refine (default: 0.01). |
| `--skip-noise-sweep` | flag | `false` | Skip zero-signal noise-only sweeps (use analytic noise instead). |
| `--store-calibration-stats` | flag | `false` | Persist per-symbol calibration statistics (raw/zscore/whitened, integrals) into the threshold cache. |
| `--target-ci` | float | `0.004` | If >0, stop adding seeds once Wilson 95% CI half-width <= target. 0 disables. |
| `--ts-cap-s` | float | `None` | Symbol period cap in seconds. |
| `--ts-warn-only` | flag | `false` | Issue warnings for long Ts instead of skipping (overrides all Ts limits) |
| `--variant` | str | `""` | Suffix appended to CSV basenames (e.g., _single_DA, _dual). |
| `--verbose` | flag | `false` | Enable verbose logging for pipeline workers. |
| `--watchdog-secs` | int | `1800` | Soft timeout for seed completion before retry hint (default: 1800s/30min) |
| `--with-ctrl, --no-ctrl` | flag | `true` | Use CTRL differential subtraction. |

## `analysis/run_master.py`

Flags below include master-specific controls and pass-through options forwarded to `run_final_analysis.py`.

| Flag(s) | Type | Default | Description |
| --- | --- | --- | --- |
| `--ablation` | choice | `both` | Run with CTRL (on), without CTRL (off), or both (default) Choices: `both, on, off`. |
| `--ablation-parallel` | flag | `false` | Launch CTRL-on and CTRL-off runs concurrently (use with care) |
| `--allow-ts-exceed` | flag | `false` | Allow Ts to exceed limits during LoD sweeps. |
| `--analytic-lod-bracket` | flag | `false` | Use Gaussian SER approximation for tighter LoD bracketing (pass-through) |
| `--baseline-isi` | choice | `off` | ISI state for baseline SER/LoD sweeps; ISI trade-off always runs ON. Choices: `off, on`. |
| `--beast-mode` | flag | `false` | Pass through to run_final_analysis (P-cores minus margin) |
| `--cal-eps-rel` | float | `0.01` | Adaptive calibration convergence threshold (pass-through) |
| `--cal-min-samples` | int | `50` | Minimum samples per class for stable thresholds (pass-through) |
| `--cal-min-seeds` | int | `4` | Minimum seeds before early stopping (pass-through) |
| `--cal-patience` | int | `2` | Calibration patience before stopping (pass-through) |
| `--channel-suite` | flag | `false` | Run physical channel baselines (single-channel DA and dual-channel DA+SERO). |
| `--clear-distance` | float | `None` | Remove LoD results for the given distance(s) in micrometres. |
| `--clear-lod-state` | list | `None` | Clear LoD caches/state. SPEC formats: 'all', 'mode', 'mode:distance', 'distance'. |
| `--clear-nm` | float | `None` | Remove SER vs Nm data for the specified molecule counts. |
| `--clear-seed-cache` | list | `None` | Clear cached seed payloads (e.g., 'lod_search', 'ser_vs_nm'). Use without values to clear all sweeps. |
| `--clear-threshold-cache` | list | `None` | Clear cached threshold JSON files (optionally filtered by MODE). |
| `--csk-baselines` | flag | `false` | Generate CSK single/dual baseline sweeps and comparison plots. |
| `--ctrl-auto` | flag | `false` | Enable adaptive CTRL on/off (pass-through) |
| `--ctrl-rho-min-abs` | float | `0.1` | Minimum correlation threshold for CTRL (pass-through) |
| `--ctrl-snr-min-gain-db` | float | `0.0` | Minimum SNR gain for CTRL (pass-through) |
| `--decision-window-frac` | float | `None` | Decision window fraction for fraction_of_Ts policy (0.1-1.0) |
| `--decision-window-policy` | choice | `None` | Override decision window policy (default: use YAML config) Choices: `fixed, fraction_of_Ts, full_Ts`. |
| `--distances` | list | `None` | Comma-separated distance grid in um for LoD (pass-through). Example: --distances MoSK=25,35,45 --distances CSK=15,25. |
| `--extreme-mode` | flag | `false` | Pass through to run_final_analysis (max P-core threads) |
| `--force-noise-resample` | flag | `false` | Force rerunning zero-signal noise sweeps even if resume cache exists. |
| `--fsync-logs` | flag | `false` | Force fsync on each write. |
| `--gain-step, --no-gain-step` | flag | `true` | Run the dI/dQ gain plot step (use --no-gain-step to skip). |
| `--guard-factor` | float | `None` | Override guard factor for ISI calculations. |
| `--guard-max-ts` | float | `0.0` | Cap guard-factor sweeps when symbol period exceeds this many seconds (0 disables; pass-through). |
| `--guard-samples-cap` | float | `None` | Per-seed sample cap for guard sweeps (0 disables cap) |
| `--inhibit-sleep` | flag | `false` | Prevent the OS from sleeping while the pipeline runs. |
| `--isi-memory-cap` | int | `None` | ISI memory cap in symbols (0 = no cap) |
| `--keep-display-on` | flag | `false` | Also keep the display awake (Windows/macOS) |
| `--list-maintenance` | list | `None` | List maintenance resources (choices: cache, cache-summary, data, device, figures, guard, lod, logs, overview, ser, stages, thresholds). |
| `--list-stages` | flag | `false` | Show maintenance stage numbering and descriptions. |
| `--lod-debug` | flag | `false` | Enable verbose LoD diagnostics (writes debug logs) |
| `--lod-distance-concurrency` | int | `8` | How many distances to run concurrently in LoD sweep (default: 8). |
| `--lod-distance-timeout-s` | float | `7200.0` | Per-distance time budget during LoD analysis. <=0 disables timeout (pass-through) |
| `--lod-max-nm` | int | `1000000` | Upper bound for Nm during LoD search (default: 1000000; pass-through) |
| `--lod-num-seeds` | str | `None` | LoD seed schedule. N \| min,max \| rules like '<=100:6,<=150:8,>150:10' (pass-through) |
| `--lod-screen-delta` | float | `0.0001` | Hoeffding screening significance for LoD binary search (pass-through) |
| `--lod-seq-len` | int | `None` | Override sequence_length during LoD search only (pass-through) |
| `--lod-skip-retry` | flag | `false` | On resume, do not retry LoD distances that previously failed (keep NaN). |
| `--lod-validate-seq-len` | int | `None` | Override sequence_length during final LoD validation only (pass-through) |
| `--logdir` | value | `results/logs` | Directory for log files. |
| `--maintenance-dry-run` | flag | `false` | Preview maintenance actions without deleting files. |
| `--maintenance-log` | str | `None` | Optional maintenance log path (e.g., results/logs/maintenance.log). Use 'none' or '-' to disable logging. |
| `--maintenance-only` | flag | `false` | Run maintenance actions then exit without executing the master workflow. |
| `--max-lod-validation-seeds` | int | `None` | Cap #seeds for final LoD validation (pass-through) |
| `--max-symbol-duration-s` | float | `None` | Skip LoD when dynamic Ts exceeds this (seconds; pass-through) |
| `--max-ts-for-lod` | float | `None` | Optional Ts cutoff to skip LoD at a distance (pass-through) |
| `--max-workers` | int | `None` | Override worker count in run_final_analysis. |
| `--min-ci-seeds` | int | `6` | Minimum seeds before CI stopping can trigger (pass-through) |
| `--min-decision-points` | int | `4` | Minimum time points for window guard (pass-through) |
| `--modes` | list | `all` | Modes to execute (e.g., `--modes MoSK CSK`; use `all` for MoSK+CSK+Hybrid). Choices: `MoSK, CSK, Hybrid, all`. |
| `--nm-grid` | str | `""` | Global Nm values for SER sweeps (pass-through to run_final_analysis). |
| `--nm-grid-csk` | str | `""` | Nm grid override for CSK only. |
| `--nm-grid-hybrid` | str | `""` | Nm grid override for Hybrid only. |
| `--nm-grid-mosk` | str | `""` | Nm grid override for MoSK only. |
| `--no-log` | flag | `false` | Disable file logging. |
| `--nonlod-watchdog-secs` | int | `14400` | Timeout for non-LoD sweeps (default: 4h; pass-through). |
| `--seed-timeout-retries` | int | `2` | Watchdog expirations allowed for a seed before restarting the worker pool (pass-through). |
| `--seed-timeout-restarts` | int | `2` | Maximum pool restarts triggered by a single seed before aborting (pass-through). |
| `--nt-pairs` | str | `""` | Comma-separated NT pairs for CSK sweeps, e.g. DA-5HT,DA-DA. |
| `--nuke-results` | flag | `false` | Delete the entire results/ directory before running the master workflow. |
| `--num-seeds` | int | `20` | Default Monte Carlo seeds forwarded to run_final_analysis. |
| `--organoid-study, --no-organoid-study` | flag | `true` | Run organoid sensitivity sweeps (alpha_H, bias, ionic strength). |
| `--parallel-modes` | int | `1` | Run modes concurrently within each ablation run (e.g., 3 for all three) |
| `--progress` | choice | `rich` | Progress UI backend for the master run (gui/rich/tqdm/none). Choices: `gui, rich, tqdm, none`. |
| `--realistic-onsi` | flag | `false` | Use cached simulation noise for ONSI calculation in hybrid benchmarks. |
| `--recalibrate` | flag | `false` | Force recalibration (ignore JSON cache) |
| `--reset` | choice | `None` | Reset simulator state. 'cache' removes caches/state only; 'all' clears results/* (applied when --reset is passed without a value). Choices: `cache, all`. |
| `--reset-stage` | list | `None` | Reset entire stages/sweeps by number or alias (1:SER, 2:LoD, 3:Device FoM, 4:Guard/ISI, 5:Main figures, 6:Supplementary, 7:Tables, 8:Sensitivity, 9:Organoid, 10:Ts sweeps, 11:Capacity). |
| `--resume` | flag | `false` | Resume completed steps. |
| `--run-stages` | str | `all` | Comma/semicolon list of stages/aliases to run (1=SER/Nm+zero-noise, 2=LoD+SER/SNR vs distance, 3=Device FoM, 4=ISI/guard; 5-7 plots/tables via master). Use `all` or `*` for every stage; subsets run only the selected stages. |
| `--sequence-length` | int | `1000` | Symbols per seed forwarded to run_final_analysis. |
| `--ser-refine` | flag | `false` | After coarse SER vs Nm sweep, auto-run a few Nm points that bracket the target SER. |
| `--ser-refine-points` | int | `4` | How many log-spaced Nm points to add between the bracketing Nm pair (default: 4). |
| `--ser-target` | float | `0.01` | Target SER for auto-refine (default: 0.01). |
| `--shared-pool` | flag | `false` | Run modes in shared process pool for maximum utilization. |
| `--store-calibration-stats` | flag | `false` | Forward to analysis runner to persist full calibration statistics. |
| `--studies` | str | `""` | Comma list from {sensitivity,capacity,isi-analytic}. Empty = none. |
| `--supplementary` | flag | `false` | Also generate supplementary figures. |
| `--supplementary-allow-fallback` | flag | `false` | Allow illustrative supplementary fallbacks when caches are missing. |
| `--supplementary-include-synthetic` | flag | `false` | Include synthetic supplementary panels (S3/S4/S6). |
| `--target-ci` | float | `0.0` | Stop adding seeds once Wilson 95%% CI half-width <= target; 0 disables (pass-through) |
| `--ts-cap-s` | float | `None` | Symbol period cap in seconds (0 = no cap) |
| `--ts-warn-only` | flag | `false` | Issue warnings for long Ts instead of skipping (overrides all Ts limits; pass-through) |
| `--validate-theory` | flag | `false` | Run analytic BER/SEP and diffusion validation figure scripts. |
| `--watchdog-secs` | int | `1800` | Soft timeout for seed completion before retry hint (default: 1800s/30min; pass-through) |
| `-preset, --preset` | choice | `None` | Apply preset configurations (ieee: publication-grade, verify: fast sanity, production: long-run batch) Choices: `ieee, verify, production`. |

## `analysis/sensitivity_study.py`

| Flag(s) | Type | Default | Description |
| --- | --- | --- | --- |
| `--target-ser` | float | `0.01` | Target SER for LoD search during sweeps. |
| `--seeds` | int | `4` | Number of seeds per LoD/metric point. |
| `--force` | flag | `false` | Recompute even if cached CSVs exist. |
| `--resume` | flag | `false` | Skip rows already present in the CSV. |
| `--progress` | choice | `none` | Placeholder for run_master compatibility. |
| `--freeze-calibration` | flag | `false` | Reuse baseline thresholds/noise during sweeps. |
| `--detector-mode` | choice | `None` | Optional detector override (raw, zscore, whitened); None = use config. |
| `--metric` | choice | `snr` | Metric for panels (lod, snr, ser). |
| `--dual-ctrl, --no-dual-ctrl` | flag | `true` | Run each sweep with CTRL on and off (default on; use --no-dual-ctrl to run CTRL-on only). |

## `analysis/organoid_sensitivity.py`

| Flag(s) | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | path | `config/default.yaml` | Configuration YAML. |
| `--sequence-length` | int | `400` | Symbols per Monte-Carlo run. |
| `--seeds` | int | `6` | Number of random seeds per sweep point. |
| `--alpha-values` | list | `[3e-4 ... 5.6e-3]` | 1/f noise (Hooge) values for alpha sweep. |
| `--bias-values` | list | `[4e-5 ... 5.5e-4]` | Drain current targets for bias sweep. |
| `--bias-nm` | int | `2000` | Nm per symbol for bias sweep. |
| `--bias-gm-scaling` | choice | `sqrt` | gm scaling vs I_dc: none, sqrt, linear. |
| `--qeff-scales` | list | `[1.25 ... 0.3]` | q_eff scaling factors. |
| `--qeff-nm` | int | `2000` | Nm per symbol for q_eff sweep. |
| `--nm-grid` | list | `None` | Optional Nm values for alpha sweep. |
| `--output-root` | path | `results` | Root directory for outputs. |
| `--recalibrate` | flag | `false` | Force threshold recalibration. |
| `--resume` | flag | `false` | Skip sweep points already present in CSVs. |
| `--dual-ctrl, --no-dual-ctrl` | flag | `true` | Run sweeps with CTRL on and off (default on; use --no-dual-ctrl for CTRL-on only). |
