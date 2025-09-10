# Tri‑Channel OECT Molecular Communication Simulator
**GLU • GABA • CTRL** receiver with MoSK, CSK‑4, and Hybrid (2 bits/symbol) modes.  
Single‑runner pipeline, crash‑safe resume, IEEE‑quality figures/tables, parallel execution, real‑time logging, and a Hybrid multi‑dimensional benchmark suite (HDS • ONSI • IRT).

> This README matches the finalized Stage 1–15 implementation and the *single‑source‑of‑truth* runner.

---

## 🔧 What’s in this repo

```
analysis/
  run_master.py                     # Stage 11: end‑to‑end orchestrator (single command)
  run_final_analysis.py             # Core simulation engine (resume, checkpoints, ISI, ablations)
  generate_comparative_plots.py     # Fig. 7/10/11 + CTRL ablation + NT‑pair panel
  plot_isi_tradeoff.py              # ISI trade‑off
  plot_hybrid_multidim_benchmarks.py# HDS / ONSI / IRT panels
  generate_supplementary_figures.py # (gated) confusion matrices & diagnostics
  ieee_plot_style.py                # IEEE TMBMC figure style (fonts, markers, CIs)
  ui_progress.py                    # Progress backends: tqdm / rich / tiny GUI
  log_utils.py                      # Real‑time tee logging
src/
  mc_channel/transport.py
  mc_receiver/binding.py
  mc_receiver/oect.py
  mc_detection/algorithms.py
  pipeline.py                       # end‑to‑end simulator path (CTRL differential; vectorized ISI)
results/                             # created on first run
  data/   figures/   tables/   cache/   logs/   figures/notebook_replicas/
config/
  default.yaml                      # baseline configuration (editable)
```

---

## 🧱 Requirements

- **Python ≥ 3.11** (SciPy≥1.16 & NumPy≥2.x toolchain).
- Linux/macOS/Windows (64‑bit). No custom C/Fortran build required (prebuilt wheels).

**Runtime (latest‑min pins)**  
NumPy ≥ 2.3.2, SciPy ≥ 1.16.1, Pandas ≥ 2.3.1, Matplotlib ≥ 3.10.5, tqdm ≥ 4.67.1, Rich ≥ 14.1.0, PyYAML ≥ 6.0.2, psutil ≥ 7.0.0.  
Optional: PyArrow ≥ 21.0.0 (large CSV/feather), Seaborn ≥ 0.13.2 (supplementary plots).

> Why NumPy ≥2.x? We rely on `numpy.trapezoid` (alias of `trapz`) in the analysis pipeline.

---

## 🚀 Install

**Option A (one‑liner, editable + dev tools):**
```bash
pip install -e .[dev]
python setup_project.py         # verify env & create results/ tree
```

**Option B (guided, with freeze or extras):**
```bash
# latest packages into .venv and editable install
python setup_env.py --extras dev --editable

# or reproduce the validated frozen set
python setup_env.py --use-freeze --editable

python setup_project.py
```

> Windows/macOS: the tiny GUI backend is optional; if unavailable the code falls back to `rich` automatically.

---

## ✅ Quick sanity (minutes, not hours)

Single mode (fast):
```bash
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm
```

Full paper set:
```bash
python analysis/run_master.py --modes all --resume --progress rich
```

Parallel modes (≈3× faster wall time):
```bash
python analysis/run_master.py --modes all --resume --progress rich --parallel-modes 3
```

---

## 🧭 CLI cheat‑sheets

### `analysis/run_final_analysis.py` (single source of truth)
```
--mode {MoSK,CSK,Hybrid,ALL} or --modes {MoSK,CSK,Hybrid,all}
--num-seeds INT                   # Monte Carlo seeds per sweep point
--sequence-length INT             # symbols per sequence (SER precision)
--resume                          # skip finished (param,seed) jobs; crash‑safe
--recalibrate                     # force threshold recalibration (ignore cache)
--progress {tqdm,rich,gui,none}
--with-ctrl / --no-ctrl           # toggle CTRL differential subtraction
--disable-isi                     # disable ISI for baseline (ISI sweep still available)
--nt-pairs STR                    # CSK NT-pair sweeps, e.g., GLU-GABA,GLU-DA,ACH-GABA
--target-ci FLOAT                 # adaptive seeds: stop when 95% CI half‑width ≤ target
--min-ci-seeds INT                # seeds required before CI stop can trigger
--lod-screen-delta FLOAT          # Hoeffding early‑stop for LoD binary search
--distances CSV                   # distance grid for LoD sweep (µm)
--lod-num-seeds RULE              # distance‑aware LoD seed schedule (e.g., '<=100:6,<=150:8,>150:10')
--lod-seq-len INT                 # shorter sequences during LoD search only
--lod-validate-seq-len INT        # override sequence length for final LoD validation
--analytic-lod-bracket            # Gaussian SER bracket for tighter LoD search
--max-ts-for-lod FLOAT            # skip LoD rows when Ts exceeds this (seconds)
--max-lod-validation-seeds INT    # cap validation retries
```

### `analysis/run_master.py` (single command for the paper)
```
--modes {MoSK,CSK,Hybrid,all}
--parallel-modes INT              # run modes concurrently per ablation
--ablation {both,on,off}          # CTRL on/off or both (default)
--ablation-parallel               # run CTRL on/off in parallel
--supplementary                   # build gated supplementary figures
--preset {ieee,verify}            # ready‑made parameter sets (publication/sanity)
--realistic-onsi                  # compute ONSI from cached device noise
# …plus pass‑throughs for LoD/ISI/CI flags listed above
```

---

## 📦 Canonical outputs

**Main‑text (≥300 dpi; IEEE style):**  
- `results/figures/fig7_comparative_ser.png`  
- `results/figures/fig10_comparative_lod.png`  
- `results/figures/fig11_comparative_data_rate.png`  
- `results/figures/fig_ctrl_ablation_ser.png`  
- `results/figures/fig_isi_tradeoff.png`  
- `results/figures/fig_nt_pairs_ser.png`  
- `results/figures/fig_hybrid_multidim_benchmarks.png`  (HDS • ONSI • IRT)

**Mechanism panels (notebook‑replicas):**  
`results/figures/notebook_replicas/*.png` (OECT PSD & noise breakdown; binding mean/PSD; transport concentration & scaling; end‑to‑end ISI).

**Data & tables (CSV + LaTeX):**  
`results/data/ser_vs_nm_{mode}.csv` (Hybrid includes `mosk_ser`, `csk_ser`)  
`results/data/lod_vs_distance_{mode}.csv` (includes `data_rate_bps`, `symbol_period_s`)  
`results/data/isi_tradeoff_{mode}.csv`, `results/data/performance_summary.csv`  
`results/tables/table1.tex`, `results/tables/table_ii_performance.tex`

---

## 🔬 Scientific metrics (Hybrid novelty panel)

- **Hybrid Decision Surface (HDS):** heat/contour from `ser`, plus marginal curves decomposing **MoSK** and **CSK** error components available in the Hybrid CSVs.  
- **OECT‑Normalized Sensitivity Index (ONSI):**  
  \[ \mathrm{ONSI} \triangleq \dfrac{\Delta I_{\mathrm{diff}}/\sigma_{I,\mathrm{diff}}}{g_m/C_{\mathrm{tot}}} \]  
  device‑normalized SNR proxy to compare molecule pairs & modes.
- **ISI‑Robust Throughput (IRT):**  
  \( R_{\text{eff}}(T_s,d) = \dfrac{\text{bits/symbol}}{T_s}\, [1-\mathrm{SER}(T_s,d)] \)  
  rendered as a ridge/heat map across \(T_s\) × distance.

These are visual summaries; see source comments for exact computation hooks.

---

## 🔁 Reproducibility & resume

- Independent RNG streams via `SeedSequence` per worker; SER does not depend on job order.  
- Checkpoints: each (param, seed) job writes `results/cache/<mode>/<sweep>_<ctrl>/<value>/seed_<id>.json`.  
- Aggregations use atomic `*.tmp → .csv` renames; `--resume` schedules only missing combinations.  
- Calibration caches are keyed by the *sweep value* and invalidate correctly when you change distance/Nm/guard.  
- CSV schema is stable and consumed by plotting scripts; IEEE style and 95% Wilson CIs are applied automatically.

---

## 🧪 Typical workflows

**CTRL ablation:** run twice (on/off) and overlay SER vs Nm in `fig_ctrl_ablation_ser.png`.  
**NT‑pair versatility (CSK):** `--nt-pairs GLU-GABA,GLU-DA,ACH-GABA` produces a small SER comparison panel.  
**ISI trade‑off:** writes `results/data/isi_tradeoff_{mode}.csv` and `fig_isi_tradeoff.png` with SER vs guard (Ts fraction).

---

## ⚙️ Configuration highlights (`config/default.yaml`)

- `pipeline.use_control_channel`: toggles differential subtraction (CTRL).  
- `pipeline.enable_isi`, `pipeline.guard_factor`, `pipeline.isi_memory[_cap_symbols]`: ISI model.  
- `pipeline.lod_nm_max`, `pipeline.Nm_per_symbol`, `pipeline.distance_um`: sweep anchors.  
- CSK/Hybrid: `pipeline.csk_levels`, `pipeline.csk_dual_channel`, `pipeline.csk_combiner`.  
- Detection window: `detection.decision_window_policy` and `decision_window_fraction` (fraction of `T_s`).

---

## 🧩 Troubleshooting

- **Tkinter GUI fails on macOS:** we fall back to `rich` automatically.  
- **Long LoD at far distances:** use `--analytic-lod-bracket`, `--lod-seq-len 250`, and `--max-ts-for-lod 180` to avoid infeasible tails.  
- **Slow runs:** enable `--parallel-modes 3` and `--extreme-mode` (master), or increase `--max-workers` in the runner.  
- **Disk churn:** CSVs are atomic; it’s safe to `Ctrl+C` and continue with `--resume`.

---

## 📄 How to cite

If you use this simulator, please cite the associated TMBMC paper (and/or this repository). See `CITATION.cff`.

---

## 📝 License

MIT (see `LICENSE`).
