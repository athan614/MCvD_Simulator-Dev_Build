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

## Configuration and Results
- Primary scenario file: `config/default.yaml`. Sections cover neurotransmitter transport, aptamer counts, device parameters, modulation choices, ISI memory, and analysis guards. Use `src.config_utils.preprocess_config` to normalize numeric strings before passing to the pipeline.
- Key CLI flags in `analysis/run_final_analysis.py`:
  - `--num-seeds`, `--sequence-length`, `--noise-only-seeds`, `--noise-only-seq-len` tune Monte Carlo budgets.
  - `--resume`, `--progress {tqdm,rich,gui,none}`, `--logdir`, `--no-log`, `--fsync-logs` manage crash-safe execution and telemetry.
  - `--decision-window-policy {fixed,fraction_of_Ts,full_Ts}`, `--decision-window-frac`, `--guard-factor`, `--isi-memory-cap`, `--disable-isi` explore ISI mitigation strategies.
  - `--lod-*` family (distance concurrency, Nm bounds, timeout, validation budget) controls limit-of-detection searches.
  - `--detector-mode {zscore,raw,whitened}`, `--csk-level-scheme`, `--with-ctrl/--no-ctrl`, `--ctrl-auto` govern modulation and CTRL subtraction modes.
- Outputs live in `results/`:
  - `results/cache/.../seed_*.json` per-seed provenance and metrics (atomic renames support resume).
  - `results/data/*.csv` consolidated LoD, SER, ISI, and guard-frontier tables.
  - `results/figures/*.png` and `results/tables/*.tex` reproducible manuscript assets.
  - `results/logs/*.log` mirrored CLI arguments, Git hashes, timing, and warnings.

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
