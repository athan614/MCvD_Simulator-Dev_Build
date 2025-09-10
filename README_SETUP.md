# Setup (latest packages + enhanced features)

**Python**: 3.11+ (SciPy≥1.16 & NumPy≥2.x toolchain)  
**OS**: Linux, macOS, or Windows (64‑bit)

## Quick start (editable install)
```bash
# from repo root
pip install -e .[dev]
python setup_project.py
```

## Managed environment (venv helper)

Create a local virtualenv, install either always‑latest (`requirements.in`) or the frozen set (`requirements.latest.txt`), and perform an editable install:

```bash
# Latest packages + dev extras + editable install
python setup_env.py --extras dev --editable

# Reproduce the frozen set validated on 2025‑08‑17
python setup_env.py --use-freeze --editable
```

To activate the environment:
- **Linux/macOS:** `source .venv/bin/activate`  
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Then verify and create folders:
```bash
python setup_project.py
```

## Handy commands

Full paper suite (with resume & progress):
```bash
python analysis/run_master.py --modes all --resume --progress rich
```

Parallel (≈3× faster):
```bash
python analysis/run_master.py --modes all --resume --progress rich --parallel-modes 3
```

Fast sanity check (minutes):
```bash
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --resume --progress tqdm
```

Generate mechanism figures:
```bash
python analysis/rebuild_oect_figs.py
python analysis/rebuild_binding_figs.py
python analysis/rebuild_transport_figs.py
python analysis/rebuild_pipeline_figs.py
```

## Notes

- Real‑time logging mirrors console output to `results/logs/*.log`.  
- On macOS, the GUI backend may fall back to `rich` when running in background threads.
