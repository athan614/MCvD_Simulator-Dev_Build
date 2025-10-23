# Tri-Channel OECT Molecular Communication Simulator

## Abstract
This repository delivers an IEEE Transactions on Molecular, Biological, and Multi-Scale Communications (TMBMC)-grade simulation environment for a tri-channel organic electrochemical transistor (OECT) molecular communication receiver. The platform models discrete molecule release, restricted extracellular diffusion, stochastic aptamer binding, triad OECT transduction (DA, SERO, CTRL), correlated device noise, and digital detection for MoSK, CSK, and hybrid dual-branch modulation. A fully vectorized Monte Carlo core, deterministic job orchestration, and scripted figure/table generators reproduce the complete manuscript pipeline without notebooks.

## Capability Highlights
- End-to-end physics stack (`src/pipeline.py`) coupling finite-duration release, Green’s function transport, stochastic/mean-field binding (`mc_receiver.binding`), correlated OECT noise synthesis (`mc_receiver.oect`), and detector statistics (`mc_detection.algorithms`).
- Multi-modulation workflow: MoSK, CSK (dual-channel combiner with leakage modelling), and Hybrid (MoSK gating with CSK amplitude refinement), each with LoD, SER, and SNR analytics.
- Crash-safe Monte Carlo execution with deterministic seeding, ProcessPool concurrency, watchdog timers, and resume-aware aggregation (`analysis/run_final_analysis.py`).
- Stage-gated orchestration (`analysis/run_master.py`, `analysis/maintenance_suite.py`) that rebuilds manuscript-quality figures, tables, and supplementary artefacts on demand.
- Extensive maintenance utilities (cache pruning, stage resets, per-category listings) and progress front-ends (`tqdm`, `rich`, and GUI overlays via `analysis/ui_progress.py`). 
- Physics validation suites (e.g., `analysis/validate_transport_against_fick.py`, `analysis/validate_analytics.py`) and analytic benchmarks (capacity, guard-factor frontiers, device FoM sweeps).

## Repository Layout
- `src/`: Simulator library (transport, binding, OECT, detection, shared utilities). The `pipeline.run_sequence` entry point accepts preprocessed YAML configs and emits per-seed statistics, noise metadata, and ISI traces.
- `analysis/`: CLI front-ends for Monte Carlo sweeps, plotting, table generation, device/guard studies, and validation. Each script is notebook-free and emits reproducible artefacts into `results/`.
- `config/`: Baseline YAML configurations; `config/default.yaml` encodes the calibrated manuscript scenario, including modulation grids and LoD search policies.
- `results/`: Generated tree containing caches, CSV exports, figures, tables, logs, and debug traces. Created automatically by `setup_project.py`.
- `requirements*.txt`, `pyproject.toml`, `environment.yml`: Dependency manifests (latest-version floors and a pinned snapshot).
- `setup_env.py`, `setup_project.py`, `Makefile`: Environment bootstrapper, results-tree provisioner, and canned workflows.
- `docs/CLI_MANUAL.md`: Complete CLI flag dictionary for `analysis/run_final_analysis.py` and `analysis/run_master.py`.

## Installation & Environment
```bash
# 1) create or reuse a Python 3.11+ virtual environment
python -m venv .venv
. .venv/Scripts/Activate.ps1    # Windows PowerShell
python -m pip install -U pip wheel setuptools

# 2) install runtime + developer extras
pip install -e ".[dev]"

# 3) provision results/ tree and verify third-party stack
python setup_project.py
```
Alternate automation:
- `python setup_env.py --extras dev --editable` creates the venv, installs latest deps, and places the package in editable mode.
- `python setup_env.py --use-freeze --editable` replays the pinned snapshot in `requirements.latest.txt` for archival builds.
- `make setup` mirrors the manual steps on POSIX shells (PowerShell users can run `pip install -e .[dev]` and `python setup_project.py` directly).

## Reproducible Workflow Overview
1. **Bootstrap** – `setup_project.py` ensures Python ≥ 3.11, confirms core libraries, and builds the canonical `results/` layout (`data/`, `cache/`, `figures/`, `tables/`, `logs/`, `figures/notebook_replicas/`). The script is idempotent and supports `--reset cache` or `--reset all` when a clean slate is required.
2. **Monte Carlo sweeps** – `analysis/run_final_analysis.py` is the authoritative driver for SER vs Nm, LoD vs distance, guard-factor sweeps, device figures of merit, and analytic noise benchmarks. Stage execution is deterministic under a fixed seed plan and safe to resume mid-run.
3. **Master orchestration** – `analysis/run_master.py` sequences the Monte Carlo engine, comparative plots, guard/ISI studies, hybrid benchmarks, tables, and supplementary figures. Presets (`--preset ieee`, `--preset verify`, `--preset production`) tune seed counts, guard policies, and plotting limits.
4. **Figure/Table generation** – Dedicated scripts under `analysis/` (e.g., `plot_snr_panels.py`, `generate_comparative_plots.py`, `table_maker.py`, `plot_guard_frontier.py`, `plot_device_fom.py`) ingest CSVs from `results/data/` and emit publication-ready assets.

### Stage Registry (shared by `run_final_analysis.py` & `run_master.py`)
| Stage | Outputs | Description |
|-------|---------|-------------|
| 1 | `results/data/ser_vs_nm*.csv`, `results/cache/*/ser_vs_nm_*` | SER vs Nm Monte Carlo sweeps with optional refinement and CTRL ablations |
| 2 | `results/data/lod_vs_distance_*.csv`, `results/cache/*/lod_*`, `results/cache/*/d*um/` | LoD grid search, analytic bracketing, and validation |
| 3 | `results/data/device_fom_*.csv`, `results/cache/*/device_fom*/` | Device gm/C sweeps and noise-limited FoM metrics |
| 4 | `results/data/guard_frontier_*.csv`, `results/data/guard_tradeoff_*.csv`, `results/cache/*/guard_frontier*/` | Guard-factor and ISI/frontier trade-off analysis |
| 5 | `results/figures/**/*.png|pdf|svg` | Main-text figures |
| 6 | `results/figures/supplementary/**` | Supplementary/appendix figures |
| 7 | `results/data/table_*.csv`, `results/data/summary_*.csv`, `results/tables/**/*.tex` | Tables, summary sheets, manuscript appendices |

`analysis/maintenance_suite.py` centralises stage definitions and exposes:
- `--list-maintenance {cache,data,figures,logs,stages,...}` to print scoped directory trees.
- `--maintenance-only` to execute maintenance commands without launching simulations.
- Targeted cleaners such as `--clear-threshold-cache`, `--clear-distance`, `--clear-lod-state`, `--clear-seed-cache`, plus stage reset hooks (`--reset-stage 2 4`) and full wipes (`--nuke-results`). A dry-run mode and audit logging (`--maintenance-log`) support reproducibility audits.

## Monte Carlo Driver – `analysis/run_final_analysis.py`
Key characteristics:
- **Crash-safe resume** via JSONL checkpoints and partial CSV append; re-running with `--resume` skips completed (mode, Nm, distance) tuples.
- **Seed planning** with ProcessPool execution, `--max-workers`, and CPU affinity presets (`--beast-mode`, `--extreme-mode`) tuned for workstation vs cluster usage.
- **LoD search** implementing analytic bracketing, adaptive guard/ISI policies, validation cap (`--max-lod-validation-seeds`), and watchdog timers (`--lod-distance-timeout-s`, `--watchdog-secs`).
- **Noise handling** including measured vs analytic noise toggles (`--lod-analytic-noise`, `--analytic-noise-all`), zero-signal noise sweeps (`--noise-only-seeds`, `--skip-noise-sweep`), CTRL residual gain thresholds (`--ctrl-auto`, `--ctrl-rho-min-abs`, `--ctrl-snr-min-gain-db`), and cached threshold reuse (`--force-noise-resample`, `--recalibrate`, `--freeze-calibration`). 
- **ISI and guard logic** with memory caps (`--isi-memory-cap`), guard-factor sweeps (`--guard-factor`, `--guard-max-ts`, `--guard-samples-cap`), and disable switches (`--disable-isi`, `--isi-sweep`). 
- **Progress front-ends** selectable via `--progress {tqdm,rich,gui,none}` (GUI overlay integrates with Tk progress manager). Logging can be redirected via `--logdir`, suppressed with `--no-log`, or fsync’d for crash resilience.
- **Flexible modulation grids**: `--mode/-modes`, per-modulation Nm overrides (`--nm-grid`, `--nm-grid-mosk`, etc.), dynamic distance sweeps (`--distance-sweep`), and dual-channel CSK combiner metadata capture.
- **Watchdogs and CI gating**: `--target-ci`, `--min-ci-seeds`, `--lod-screen-delta`, `--ser-refine`, `--ser-target` enforce statistical confidence before advancing to downstream stages.

Quick sanity run:
```bash
python analysis/run_final_analysis.py \
  --mode CSK \
  --num-seeds 4 \
  --sequence-length 200 \
  --recalibrate \
  --resume \
  --progress tqdm
```

## Master Driver – `analysis/run_master.py`
The Stage-11 driver sequences all manuscript artefacts. Highlights:
- Presets: `--preset ieee` (publication-grade, full seed counts), `--preset verify` (compact regression), `--preset production` (batch cluster). Presets configure guard factors, LoD depth, figure toggles, and progress UI defaults.
- Ablation control: `--ablation {both,on,off}`, `--ablation-parallel` for concurrent CTRL vs no-CTRL branches. 
- Supplementary gating: `--supplementary`, `--supplementary-include-synthetic`, `--supplementary-allow-fallback`.
- Study selectors: `--channel-suite`, `--organoid-study`, `--validate-theory`, `--csk-baselines`, `--realistic-onsi`, `--gain-step` to toggle optional figure families.
- Pass-through to Monte Carlo core for calibration, guard, LoD, and Nm grid parameters; inherits all major knobs from `run_final_analysis.py`.
- Power utilities: `--shared-pool`, `--parallel-modes`, `--max-workers`, `--watchdog-secs` coordinate with long-running jobs on shared servers.
- Logging and UX: identical progress/logging switches plus display inhibitors (`--keep-display-on`, `--inhibit-sleep`).

Shortcuts (POSIX shell):
```bash
make run-all           # all modes, resume, rich progress
make supplementary     # include supplementary figures
make verify            # fast integrity sweep (preset verify)
```

## Analysis & Plotting Modules
- **Comparative plots**: `analysis/generate_comparative_plots.py`, `analysis/plot_snr_panels.py`, `analysis/plot_input_output_gain.py`, `analysis/plot_csk_single_dual.py`, `analysis/plot_channel_profiles.py`.
- **Guard/ISI studies**: `analysis/plot_guard_frontier.py`, `analysis/plot_isi_tradeoff.py`, `analysis/guard_frontier` caches, and guard analytics produced during Stage 4.
- **Device characterization**: `analysis/plot_device_fom.py`, `analysis/rebuild_oect_figs.py`, `analysis/rebuild_pipeline_figs.py`, `analysis/rebuild_transport_figs.py` (OECT PSD breakdowns, gm/C sweeps, transport visualisations).
- **Noise and capacity**: `analysis/run_noise_only.py`, `analysis/noise_correlation.py`, `analysis/capacity_analysis.py`, `analysis/fig_onsi_vs_rho_cc.py`.
- **Hybrid benchmarks**: `analysis/plot_hybrid_multidim_benchmarks.py`, `analysis/organoid_sensitivity.py`, `analysis/sensitivity_study.py`.
- **Tables**: `analysis/table_maker.py`, `analysis/param_table.py` convert CSV aggregates into LaTeX-ready tables.
- **Validation**: `analysis/validate_transport_against_fick.py`, `analysis/validate_analytics.py` cross-check physics modules against analytic references and published baselines.

All plotting scripts apply IEEE styling via `analysis/ieee_plot_style.apply_ieee_style` (serif fonts, compact margins, vector-friendly output). Most accept `--write-summary`, `--combined-output`, and `--force` toggles to manage regeneration workflows.

## Physics Modules (src/)
- `mc_channel.transport`: Vectorized finite-burst concentration kernels (rectangular/gamma release), Green’s function solvers, and batch diffusion accelerators with clearance and tortuosity.
- `mc_receiver.binding`: Stochastic aptamer binding via Bernoulli/Poisson switching, Damköhler-limited on-rates, mean-field ODE solvers, and batch helpers.
- `mc_receiver.oect`: Correlated tri-channel noise synthesis (thermal, 1/f, drift) using FFT envelope shaping, Cholesky correlation, CTRL subtraction analytics, and deterministic fallbacks.
- `mc_detection.algorithms`: Detector statistics, MoSK/CSK thresholds, z-score and whitened detectors, and hybrid post-processing.
- `constants.py`, `config_utils.py`, `analysis_utils.py`: Shared constants (e.g., Avogadro, charge units), YAML preprocessing, calibration metadata normalization, and summary utilities.
- `pipeline.py`: The orchestrator binding transport, binding, noise, and detection with online statistics (`RunningStat`), ISI bookkeeping, hybrid branch metrics, cached threshold injection, and ProcessPool wrappers (`run_sequence_batch_cpu`).

## Configuration
`config/default.yaml` is the calibrated baseline for the manuscript:
- Specifies neurotransmitter kinetics, diffusion constants, device gm/C, noise correlation, symbolic guard policies, and modulation-specific distance/Nm grids.
- Pipeline parameters toggle ISI memory, dual-channel CSK combiner, guard factor caps, LoD bounds, and detector policies (`decision_window_policy`, `detector_mode`). 
- Distances and Nm ranges can be overridden via CLI without editing YAML; `config_utils.preprocess_config` normalizes numeric strings, injects derived parameters, and aligns detection windows with the symbol period.

## Results Tree Anatomy
- `results/data/`: CSV exports (`ser_vs_nm_*.csv`, `lod_vs_distance_*.csv`, `guard_frontier_*.csv`, `device_fom_*.csv`, `snr_vs_ts_*.csv`, panel summaries, tables).
- `results/cache/`: Per-seed checkpoints, noise calibrations (`lod_state`, `nm_refine`, `ser_refine`, `guard_frontier`, etc.), and resume metadata.
- `results/figures/`: Publication assets (`fig_*.png/pdf/svg`), notebook replicas, supplementary figures, and guard/SNR panel outputs.
- `results/tables/`: LaTeX/CSV tables, markdown parameter sheets.
- `results/logs/`: Rotating log files (`run_final_analysis.log`, maintenance audits, progress transcripts).
- `results/debug/`: Optional LoD debug JSONL dumps when `MCVD_LOD_DEBUG=1`.
All file writes are atomic (temp file + replace) to guard against partial writes.

## Development & QA
```bash
pytest                    # Monte Carlo regression tests / quick physics checks
pytest --maxfail=1 -q     # Faster failure-oriented run
ruff check src analysis   # Lint (PEP 8 + domain rules)
black src analysis        # Code formatting (line length 88)
mypy src analysis         # Static type checking
```
The `dev` extra installs `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `pip-tools`, and `jupyter`. `pip-compile` (via `pip-tools`) can regenerate lockfiles when bumping dependencies beyond the tracked snapshot.

## Dependency Baseline (minimum versions)
- NumPy ≥ 2.3.4
- SciPy ≥ 1.16.2
- pandas ≥ 2.3.3
- Matplotlib ≥ 3.10.7
- Seaborn ≥ 0.13.2
- PyYAML ≥ 6.0.3
- tqdm ≥ 4.67.1
- Rich ≥ 14.2.0
- psutil ≥ 7.1.1
- joblib ≥ 1.5.2
- statsmodels ≥ 0.14.5
- cycler ≥ 0.12.1
- pyarrow ≥ 21.0.0 (analysis extra)
- Dev tooling: pytest ≥ 8.4.2, pytest-cov ≥ 7.0.0, black ≥ 25.9.0, ruff ≥ 0.14.2, mypy ≥ 1.18.2, pip-tools ≥ 7.5.1, jupyter ≥ 1.1.1

Use `requirements.latest.txt` (snapshot dated 2025-10-24) for reproducible archival installs. `environment.yml` mirrors the same floors for Conda-based provisioning while delegating package resolution to pip (`-e .[dev]`).

## Citation
If this simulator underpins published work, cite the forthcoming IEEE TMBMC article accompanying the simulator and/or the repository’s `CITATION.cff` metadata.

## License
Released under the MIT License. See `LICENSE` for terms of use and redistribution.
