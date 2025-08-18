# Tri‑Channel OECT Molecular Communication Simulator
**GLU • GABA • CTRL** receiver with MoSK, CSK‑4, and Hybrid (2 bits/symbol) modes.  
Single‑runner pipeline, crash‑safe resume, IEEE‑quality figures/tables, parallel execution, real-time logging, and a novel Hybrid multi‑dimensional benchmark suite (HDS • ONSI • IRT).

> This README reflects the finalized Stage 1–15 implementation with parallel execution and enhanced logging capabilities.

---

## 🔧 What's in this repo

```
analysis/
  run_master.py                     # Stage 11: end‑to‑end orchestrator (single command)
  run_final_analysis.py             # Core simulation engine with parallel mode support
  rebuild_oect_figs.py              # Notebook-replica OECT mechanism figures
  rebuild_binding_figs.py           # Notebook-replica binding kinetics figures  
  rebuild_transport_figs.py         # Notebook-replica transport/diffusion figures
  rebuild_pipeline_figs.py          # Notebook-replica end-to-end pipeline figures
  log_utils.py                      # Real-time tee logging system
  ui_progress.py                    # Stage 3 progress backends (tqdm/rich/gui)
src/
  mc_channel/transport.py
  mc_receiver/binding.py
  mc_receiver/oect.py
  pipeline.py                       # end‑to‑end simulator path (CTRL differential, vectorized ISI, etc.)
results/
  data/     figures/     tables/    cache/   logs/   # created on first run
  figures/notebook_replicas/        # mechanism detail figures
config/
  default.yaml                      # baseline configuration (editable)
```

---

## 🧱 Requirements

- **Python ≥ 3.11** (SciPy ≥ 1.16 requires 3.11–3.13; NumPy ≥ 2.3 supports 3.11–3.14).
- Linux/macOS/Windows (64‑bit). A C/Fortran toolchain is **not** required for normal use (we use prebuilt wheels).

### Runtime deps (pinned to current stable minimums)

- NumPy ≥ 2.3.2, SciPy ≥ 1.16.1, Pandas ≥ 2.3.1, Matplotlib ≥ 3.10.5, tqdm ≥ 4.67.1, Rich ≥ 14.1.0, PyYAML ≥ 6.0.2, psutil ≥ 7.0.0.  
  (Optional) PyArrow ≥ 21.0.0 for large CSV/feather workflows; Seaborn ≥ 0.13.2 for supplementary plots.

> **Why NumPy ≥ 2.x?** The runner uses `numpy.trapezoid` for charge integration; this API ships with NumPy 2.x (an alias of the classic `trapz`).

---

## 🚀 Install

```bash
# from a fresh Python 3.11+ environment
pip install -e .[dev]
# verify environment & create results tree
python setup_project.py
```

If you prefer Jupyter for inspection, install the optional extra and then `jupyter lab`.

---

## ✅ Quick sanity runs (minutes, not hours)

**Single mode (fast):**

```bash
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm
```

**Full paper set (default Hybrid ONSI proxy):**

```bash
python analysis/run_master.py --modes all --resume --progress rich
```

**Parallel execution (3x faster):**

```bash
python analysis/run_master.py --modes all --resume --progress rich --parallel-modes 3
```

This executes: simulations → comparative plots (Fig. 7/10/11) → ISI trade‑off → Hybrid multidimensional figure → notebook-replica mechanism panels → tables.

---

## 🧭 CLI Cheat‑Sheet (canonical runner)

```text
analysis/run_final_analysis.py
  --modes all|MoSK|CSK|Hybrid      # run one or all three modes
  --num-seeds INT                  # Monte Carlo seeds per sweep point
  --sequence-length INT            # symbols per sequence (SER precision)
  --resume                         # skip completed sweep values (crash-safe)
  --recalibrate                    # force threshold recalibration
  --parallel-modes INT             # run modes concurrently (e.g., 3 for max speed)
  --progress {tqdm,rich,gui,none}  # progress UI backend
  --target-ci FLOAT               # adaptive seed stopping (Wilson CI half-width)
  --logdir DIR                     # directory for log files
  --no-log                         # disable file logging
  --fsync-logs                     # force OS sync on each write
  --with-ctrl / --no-ctrl          # enable/disable CTRL differential subtraction
  --nt-pairs STR                   # CSK neurotransmitter pair sweeps
  --lod-screen-delta FLOAT         # Stage 13 Hoeffding early-stop for LoD
```

**Master command:** `analysis/run_master.py` adds `--supplementary`, `--reset {cache|all}`, and passes through common options including `--parallel-modes`.

---

## 📦 Canonical outputs

Main‑text figures (≥300 dpi, IEEE style):  
- `results/figures/fig7_comparative_ser.png`  
- `results/figures/fig10_comparative_lod.png`  
- `results/figures/fig11_comparative_data_rate.png`  
- `results/figures/fig_ctrl_ablation_ser.png`  
- `results/figures/fig_isi_tradeoff.png`  
- `results/figures/fig_nt_pairs_ser.png`  
- `results/figures/fig_hybrid_multidim_benchmarks.png`  (HDS • ONSI • IRT)

Notebook-replica mechanism figures:
- `results/figures/notebook_replicas/oect_differential_psd.png`
- `results/figures/notebook_replicas/oect_noise_breakdown.png`
- `results/figures/notebook_replicas/binding_mean_vs_mc.png`
- `results/figures/notebook_replicas/binding_psd_vs_analytic.png`
- `results/figures/notebook_replicas/concentration_profiles.png`
- `results/figures/notebook_replicas/delay_factors_analysis.png`
- `results/figures/notebook_replicas/distance_scaling_analysis.png`
- `results/figures/notebook_replicas/fixed_with_isi_v1_format.png`

Data & tables (CSV + LaTeX where applicable):  
- `results/data/ser_vs_nm_{mode}.csv` (includes `mosk_ser`,`csk_ser` for Hybrid)  
- `results/data/lod_vs_distance_{mode}.csv` (includes `data_rate_bps`,`symbol_period_s`)  
- `results/data/isi_tradeoff_{mode}.csv`  
- `results/data/performance_summary.csv`  
- `results/tables/table1.tex`, `results/tables/table_ii_performance.tex`

Real-time logging:
- `results/logs/run_master_YYYYMMDD-HHMMSS.log`
- `results/logs/run_final_analysis_YYYYMMDD-HHMMSS.log`

Crash‑safety & resume: per‑(param,seed) JSON blobs under `results/cache/<mode>/<sweep>/<value>/seed_*.json` and atomic `*.tmp→.csv` writes.

---

## 🚀 Performance Features

### Parallel Mode Execution
Run multiple modulation modes (MoSK, CSK, Hybrid) simultaneously for 3x speedup:

```bash
# Sequential (default): ~45 minutes for full suite
python analysis/run_master.py --modes all

# Parallel: ~15 minutes for full suite  
python analysis/run_master.py --modes all --parallel-modes 3
```

The parallel execution uses thread-based orchestration with a shared process pool to avoid CPU oversubscription while maximizing throughput.

### Real-Time Logging
All console output is automatically mirrored to timestamped log files:

```bash
# Logs to results/logs/run_master_20250818-143022.log
python analysis/run_master.py --modes all

# Custom log directory
python analysis/run_master.py --modes all --logdir ./my_logs

# Disable logging entirely
python analysis/run_master.py --modes all --no-log
```

Log files are written in real-time, so you can monitor progress with `tail -f` even during long runs.

---