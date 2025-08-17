# Tri‑Channel OECT Molecular Communication Simulator
**GLU • GABA • CTRL** receiver with MoSK, CSK‑4, and Hybrid (2 bits/symbol) modes.  
Single‑runner pipeline, crash‑safe resume, IEEE‑quality figures/tables, and a novel Hybrid multi‑dimensional benchmark suite (HDS • ONSI • IRT).

> This README reflects the finalized Stage 1–15 implementation. It replaces the older, single‑script prototype.

---

## 🔧 What’s in this repo

```
analysis/
  run_master.py                     # Stage 11: end‑to‑end orchestrator (single command)
  run_final_analysis.py             # Stage 1–7,13: canonical runner (modes, resume, ISI, CTRL, pairs, CI/LoD)
  generate_comparative_plots.py     # Fig.7/10/11 + CTRL ablation + NT‑pair panel + summary .tex
  plot_isi_tradeoff.py              # ISI trade‑off (Stage 6)
  plot_hybrid_multidim_benchmarks.py# HDS/ONSI/IRT panels (Stage 10)
  generate_supplementary_figures.py # Supplementary (gated, data‑driven confusions)
  ieee_plot_style.py                # Stage 9 unified IEEE style
  ui_progress.py                    # Stage 3 progress backends (tqdm/rich/gui)
src/
  mc_channel/transport.py
  mc_receiver/{binding.py,oect.py}
  mc_detection/algorithms.py
  pipeline.py                       # end‑to‑end simulator path (CTRL differential, vectorized ISI, etc.)
results/
  data/     figures/     tables/    cache/   # created on first run
config/
  default.yaml                      # baseline configuration (editable)
```

---

## 🧱 Requirements

- **Python ≥ 3.11** (SciPy ≥ 1.16 requires 3.11–3.13; NumPy ≥ 2.3 supports 3.11–3.14). citeturn9search2turn9search1turn9search9
- Linux/macOS/Windows (64‑bit). A C/Fortran toolchain is **not** required for normal use (we use prebuilt wheels).

### Runtime deps (pinned to current stable minimums)

- NumPy ≥ 2.3.2, SciPy ≥ 1.16.1, Pandas ≥ 2.3.1, Matplotlib ≥ 3.10.5, tqdm ≥ 4.66.6, Rich ≥ 14.1.0, PyYAML ≥ 6.0.2, psutil ≥ 7.0.0.  
  (Optional) PyArrow ≥ 21.0.0 for large CSV/feather workflows; Seaborn ≥ 0.13.2 for supplementary plots.  
  Latest versions as of Aug 2025: NumPy 2.3.2, SciPy 1.16.1, Pandas 2.3.1, Matplotlib 3.10.5, tqdm 4.66.6, Rich 14.1.0, PyYAML 6.0.2, PyArrow 21.0.0, Seaborn 0.13.2. citeturn0search0turn0search1turn0search6turn0search7turn2search1turn1search1turn1search18turn3search0turn5search1

> **Why NumPy ≥ 2.x?** The runner uses `numpy.trapezoid` for charge integration; this API ships with NumPy 2.x (an alias of the classic `trapz`). citeturn6search0

---

## 🚀 Install

```bash
# from a fresh Python 3.11+ environment
pip install -e .[viz,dev]
# verify environment & create results tree
python setup_project.py
```

If you prefer Jupyter for inspection, install the optional extra and then `jupyter lab`. citeturn7search0

---

## ✅ Quick sanity runs (minutes, not hours)

**Single mode (fast):**

```bash
python analysis/run_final_analysis.py   --mode CSK --num-seeds 4 --sequence-length 200   --recalibrate --resume --progress tqdm
```

**Full paper set (default Hybrid ONSI proxy):**

```bash
python analysis/run_master.py --modes all --resume --progress rich
```

This executes: simulations → comparative plots (Fig. 7/10/11) → ISI trade‑off → Hybrid multidimensional figure → tables. fileciteturn1file1

---

## 🧭 CLI Cheat‑Sheet (canonical runner)

```text
analysis/run_final_analysis.py
  --modes all|MoSK|CSK|Hybrid    # run one or all three modes
  --num-seeds INT                # seeds per sweep point (adaptive CI can stop early)
  --sequence-length INT          # symbols per seed
  --resume / --recalibrate       # crash‑safe resume; force calib refresh
  --progress tqdm|rich|gui|none  # Stage 3 progress backends
  --with-ctrl / --no-ctrl        # CTRL ablation (affects CSV 'use_ctrl')
  --disable-isi                  # turn off ISI model
  --nt-pairs GLU-GABA,GLU-DA,... # Stage 7 molecule‑pair sweep (CSK)
  --target-ci FLOAT --min-ci-seeds INT
                                 # Stage 13 adaptive seeds (Wilson 95% CI)
  --lod-screen-delta FLOAT       # Stage 13 Hoeffding early-stop for LoD
```
See implementation for exact defaults. fileciteturn1file4

**Master command:** `analysis/run_master.py` adds `--supplementary`, `--reset {cache|all}`, and passes through common options. fileciteturn1file11

---

## 📦 Canonical outputs

Main‑text figures (≥300 dpi, IEEE style):  
- `results/figures/fig7_comparative_ser.png`  
- `results/figures/fig10_comparative_lod.png`  
- `results/figures/fig11_comparative_data_rate.png`  
- `results/figures/fig_ctrl_ablation_ser.png`  
- `results/figures/fig_isi_tradeoff.png`  
- `results/figures/fig_nt_pairs_ser.png`  
- `results/figures/fig_hybrid_multidim_benchmarks.png`  (HDS • ONSI • IRT) fileciteturn1file17turn1file5turn1file0

Data & tables (CSV + LaTeX where applicable):  
- `results/data/ser_vs_nm_{mode}.csv` (includes `mosk_ser`,`csk_ser` for Hybrid)  
- `results/data/lod_vs_distance_{mode}.csv` (includes `data_rate_bps`,`symbol_period_s`)  
- `results/data/isi_tradeoff_{mode}.csv`  
- `results/data/performance_summary.csv`  
- `results/tables/table1.tex`, `results/tables/table_ii_performance.tex` fileciteturn1file17

Crash‑safety & resume: per‑(param,seed) JSON blobs under `results/cache/<mode>/<sweep>/<value>/seed_*.json` and atomic `*.tmp→.csv` writes. fileciteturn1file4

---

## 🖼️ Hybrid multi‑dimensional benchmarks (Stage 10)

`analysis/plot_hybrid_multidim_benchmarks.py` renders:  
- **HDS**: total SER + MoSK/CSK components vs Nm (and 2‑D grid when available).  
- **ONSI**: (ΔI_diff/σ_I_diff) / (g_m/C_tot) — normalized device sensitivity index.  
- **IRT**: ISI‑Robust Throughput heatmap or curve.  
Inputs are the canonical CSVs written by the runner. fileciteturn1file2

---

## 🧪 Supplementary (gated)

`analysis/generate_supplementary_figures.py --strict --only-data` builds data‑driven confusion matrices (S5). Synthetic panels (S3/S4/S6) stay gated unless explicitly enabled. fileciteturn1file19

---

## 🔒 Reproducibility
- Independent seeds via `SeedSequence`‑derived RNG streams in parallel workers (SER invariant to scheduling).  
- Calibration caching keyed on configuration; caches invalidated by Nm or distance changes.  
- Logged noise sigmas per decision window for reproducibility. (See runner for details.) fileciteturn1file4

---

## 🛠️ Troubleshooting
- **NumPy < 2.0** → `AttributeError: numpy has no attribute 'trapezoid'`. Upgrade NumPy (and SciPy). citeturn6search0  
- **GUI progress** requires Tkinter. If missing, use `--progress rich`.  
- **Reset** bad runs: `python analysis/run_master.py --reset all` (or `cache`). fileciteturn1file11

---

## 📄 Citation
If you use this simulator in academic work, please cite our TMBMC paper (pending) and this repository.

