# Tri-Channel OECT Molecular Communication Simulator

## Overview
- Research-grade simulator for the tri-channel organic electrochemical transistor (OECT) receiver covering dopamine (DA), serotonin (SERO), and control (CTRL) channels.
- End-to-end physics pipeline: finite molecular release, restricted diffusion, stochastic aptamer binding, OECT current generation, and digital detection for MoSK, CSK, and hybrid modulation.
- Vectorized Monte Carlo workflow with crash-safe resume, deterministic seeding, and automated aggregation of limit-of-detection and symbol error statistics.
- Analysis notebooks replaced by scripted figure/table builders that reproduce the IEEE manuscript assets under `analysis/`.

## Quickstart
```bash
# 1) create or reuse a Python 3.11+ virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools

# 2) install runtime + developer extras
pip install -e ".[dev]"
# 3) provision results/ tree and verify core packages
python setup_project.py

# 4) launch a small sanity run (4 seeds, shorter sequence)
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm
```
Use `python setup_env.py --extras dev --editable` for a guided installation and `python setup_env.py --use-freeze --editable` to replay the pinned snapshot in `requirements.latest.txt`.

## Maintenance Suite
Both `analysis/run_final_analysis.py` and `analysis/run_master.py` ship with a shared maintenance suite (`analysis/maintenance_suite.py`) so you can inspect, prune, or rebuild parts of the results tree without hand-editing files.

**Stage map** (used by `--reset-stage`):
1. SER vs Nm sweeps (`results/data/ser_vs_nm*.csv`, `results/cache/*/ser_vs_nm_*`)
2. LoD vs distance search/state (`lod_vs_distance`, `lod_* caches`, `dXXum` folders)
3. Device FoM sweeps (`device_fom` CSV + caches)
4. Guard-frontier & ISI trade-off (`guard_tradeoff/frontier` CSV + caches)
5. Main figures (publication outputs under `results/figures/`, `results/figures/notebook_replicas/`, `results/figures/publication/`)
6. Supplementary figures (`results/figures/supplementary/`, `results/figures/appendix/`)
7. Tables & summaries (`results/tables/`, `results/data/table_*`, `results/data/summary_*`)

**Listing resources**
- `--list-maintenance cache data` prints a short tree under `results/cache/` and `results/data/`.
- Additional categories: `thresholds`, `lod`, `ser`, `device`, `guard`, `figures`, `cache-summary`, `stages`.
- Combine with `--maintenance-only` to inspect the filesystem without kicking off simulations.

**Targeted cleanup flags**
- `--clear-threshold-cache [MODE ...]` remove cached detector thresholds (omit MODE to clear everything).
- `--clear-lod-state [SPEC ...]` prune LoD state/caches. `SPEC` accepts `all`, `mode`, `distance`, or `mode:distance[:wctrl|noctrl]`.
- `--clear-seed-cache [SWEEP ...]` drop cached seed payloads (e.g., `lod_search`, `ser_vs_nm`). With no values it clears every sweep.
- `--clear-distance DIST [DIST ...]` purge LoD rows/caches for specific distances (micrometres).
- `--clear-nm NM [NM ...]` purge SER vs Nm rows/caches for specific molecule counts.

**Operational helpers**
- `--maintenance-dry-run` preview maintenance actions without deleting files.
- `--maintenance-log PATH` write maintenance audit entries to the chosen file (default: `results/logs/maintenance.log`, use `none` or `-` to disable).
- `--maintenance-only` run maintenance commands and exit before any simulations.

**Stage resets & destructive options**
- `--reset-stage 2 4` wipes the listed stages (see map above) including CSVs and caches.
- `--nuke-results` removes *everything* under `results/` (the directory is recreated automatically).
- Legacy `run_master.py --reset cache/all` now forwards to the same helpers; you can mix old and new flags.

Pass `--maintenance-only` alongside any cleanup command to perform the action(s) and exit immediately.
**Quick stage runs**
- Stage map shared by `run_master.py` and `run_final_analysis.py`: `1=SER vs Nm`, `2=LoD vs distance`, `3=Device FoM`, `4=Guard/ISI trade-off`, `5=Main figures`, `6=Supplementary`, `7=Tables/Summaries`.
- `analysis/run_final_analysis.py --run-stages 2` runs only the LoD sweep (Stage 2) for the selected mode(s).
- `analysis/run_master.py --run-stages 1` replays just the SER vs Nm sweeps (Stage 1) and skips downstream plotting.
- `analysis/run_master.py --run-stages 5,7` rebuilds main figures (Stage 5) and tables (Stage 7) without rerunning simulations.


## Repository Layout
- `src/` core simulator modules (imported via `src.pipeline.run_sequence`).
- `analysis/` command-line workflows for calibration, Monte Carlo sweeps, plotting, and table generation.
- `config/` baseline YAML scenarios (`default.yaml`) consumed by the pipeline.
- `results/` cache, figures, tables, and logs created at runtime (generated on demand).
- `requirements*.txt`, `pyproject.toml`, `environment.yml` dependency manifests for different tooling preferences.

## Module Guide (src/)
- `pipeline.py` orchestrates the full transmit-propagate-bind-detect chain. Key helpers include `_resolve_decision_window`, `_csk_dual_channel_Q`, `_single_symbol_currents`, and `run_sequence/run_sequence_batch` for multiprocessing execution.
- `mc_channel/transport.py` implements restricted extracellular diffusion using analytical Green's functions and vectorized burst solvers (`finite_burst_concentration`, batch and time-offset variants).
- `mc_receiver/binding.py` models aptamer kinetics with Bernoulli/Poisson Monte Carlo steps, deterministic ODE solutions (`mean_binding`), and noise PSD utilities.
- `mc_receiver/oect.py` converts bound charge into correlated drain currents, adding thermal, flicker, and drift noise while enforcing channel-to-channel covariance.
- `mc_detection/algorithms.py` supplies integration, detector statistics, closed-form SER/BER expressions, and Monte Carlo placeholders for higher layer studies.
- `constants.py`, `analysis_utils.py`, `config_utils.py`, and small shims (`binding.py`, `detection.py`, `diffusion.py`, `oect.py`) centralize physical constants, numeric coercion, and legacy import paths.

## Analysis Workflows (analysis/)
- `run_final_analysis.py` primary entry point. Supports `--mode/--modes`, resume logic, calibration caching, ISI sweeps, guard-factor controls, CTRL subtraction toggles, and watchdogs for LoD searches.
- `run_master.py` orchestrates multi-mode batches, guard-frontier sweeps, resume-aware chunking, and optional parallel child processes.
- Plot/table builders (`plot_*`, `generate_*`, `rebuild_*`, `table_maker.py`) transform cached CSV outputs into IEEE-ready figures and LaTeX tables.
- Diagnostics (`capacity_analysis.py`, `sensitivity_study.py`, `noise_correlation.py`, `validate_*`) probe specific subsystems such as decision-window policies, transport validation, or hybrid combiner behavior.

## Simulator Deliverables
- **Per-seed cache**: `results/cache/<mode>/<distance>/<seed>.json` captures raw symbol outcomes, LoD brackets, detector statistics, RNG seeds, and timing metadata. Safe to interrupt/resume because writes are atomic.
- **Aggregated CSVs** (created once all seeds finish a sweep):
  - `ser_vs_nm_<mode>.csv` — SER/BER versus molecule budgets with confidence intervals, decision-window policy, and CTRL usage flags.
  - `lod_vs_distance_<mode>.csv` — limit-of-detection search results including Nm bracket, achieved SER, symbol period, guard factor, data rate, and retry counters.
  - `isi_tradeoff_<mode>.csv` — throughput versus guard-factor/ISI-memory settings.
  - `guard_frontier_<variant>.csv` — output from `run_master.py` guard-frontier campaigns summarizing CTRL states and achievable symbol periods.
- **Figures & tables**: IEEE-ready PNGs and TeX tables under `results/figures/` and `results/tables/`, regenerated by the plotting scripts using the cached CSVs.
- **Waveform traces**: optional diagnostic exports (`_calibration_trace`, channel integrals, decision statistics) embedded inside cache JSON files to support post-run forensic analysis.
- **Logs**: mirrored CLI invocation, parameter hashes, platform info, and progress snapshots under `results/logs/`.

## Supported Sweeps & Tests
- **LoD search**: automated Nm bracket refinement per distance with configurable validation seeds (`--lod-*` flags).
- **SER sweeps**: configurable Nm grids (`--nm-grid`, config-driven lists) and target refinement runs (`--ser-refine`).
- **ISI/guard studies**: toggle and bound guard factor, ISI memory, and decision-window settings (`--guard-factor`, `--guard-samples-cap`, `--disable-isi`, `--decision-window-policy`, `--decision-window-frac`).
- **Noise experiments**: isolate noise-only seeds (`--noise-only-seeds`, `--skip-noise-sweep`), force fresh noise characterization when needed (`--force-noise-resample`), switch detector statistics, and inspect correlated control subtraction (`--with-ctrl`, `--no-ctrl`, `--detector-mode`).
- **Hybrid modulation analysis**: vary CSK level mapping (`--csk-level-scheme`), tail integration fraction, and dual-channel combiners.
- **Parallel calibration stress tests**: `--beast-mode`, `--extreme-mode`, and watchdog settings vet long-running workloads, while `run_master.py` can drive cross-mode or guard-frontier matrices in parallel.
- **Specialized diagnostics**: scripts under `analysis/` provide capacity estimation, sensitivity sweeps to temperature/diffusion/binding, noise correlation heatmaps, and analytic-versus-simulation validation.

## Configuration and Results
- Primary scenario file: `config/default.yaml`. Sections cover neurotransmitter transport, aptamer counts, device parameters, modulation choices, ISI memory, and analysis guards. Use `src.config_utils.preprocess_config` to normalize numeric strings before passing to the pipeline.
- Key CLI flags in `analysis/run_final_analysis.py`:
  - `--num-seeds`, `--sequence-length`, `--noise-only-seeds`, `--noise-only-seq-len` tune Monte Carlo budgets.
  - `--resume`, `--progress {tqdm,rich,gui,none}`, `--logdir`, `--no-log`, `--fsync-logs` manage crash-safe execution and telemetry.
  - `--decision-window-policy {fixed,fraction_of_Ts,full_Ts}`, `--decision-window-frac`, `--guard-factor`, `--isi-memory-cap`, `--disable-isi` explore ISI mitigation strategies.
  - `--lod-*` family (distance concurrency, Nm bounds, timeout, validation budget) controls limit-of-detection searches.
  - `--detector-mode {zscore,raw,whitened}`, `--csk-level-scheme`, `--with-ctrl/--no-ctrl`, `--ctrl-auto` govern modulation and CTRL subtraction modes.

## Dependency Snapshot (2025-10-13)
- Runtime: numpy 2.3.3, scipy 1.16.2, pandas 2.3.3, matplotlib 3.10.7, seaborn 0.13.2, PyYAML 6.0.3, tqdm 4.67.1, rich 14.2.0, psutil 7.1.0, joblib 1.5.2, statsmodels 0.14.5, cycler 0.12.1, pyarrow 21.0.0.
- Tooling: pytest 8.4.2, pytest-cov 7.0.0, black 25.9.0, ruff 0.14.0, mypy 1.18.2, pip-tools 7.5.1, jupyter 1.1.1.
- `environment.yml`, `requirements.in`, and `requirements.latest.txt` mirror these versions for conda, pip-compile, and frozen installs respectively.

## Development Notes
- Automated tests are not bundled. When adding physics or analysis features, prefer authoring focused scripts in `analysis/` or integrate with an external pytest suite before relying on production runs.
- Static checks: `ruff check src analysis`, formatting: `black src analysis`, typing: `mypy src analysis`. Run these commands from the repository root after activating your environment.
- `setup_project.py --reset cache` clears cached Monte Carlo results; `--reset all` removes the entire `results/` tree (use with care).
- Large Monte Carlo sweeps can stress CPU and memory; keep BLAS thread counts at one (already enforced in `run_final_analysis.py`) and monitor system load when adjusting `--max-workers` or `--parallel-modes`.

## Citation
If this simulator underpins published work, cite the IEEE TMBMC article and/or the `CITATION.cff` metadata packaged with the repository.

## License
Released under the MIT License. See `LICENSE` for full terms.
