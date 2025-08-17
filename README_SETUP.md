# Setup (Latest Packages)

**Python**: 3.11+ is required (SciPy≥1.16 requires Python≥3.11).  
**OS**: Linux, macOS, or Windows. (Optional Tk for GUI progress.)

## Quick start
```bash
# from repo root
python setup_env.py --extras dev --editable
# or, to reproduce the exact set we validated on 2025‑08‑17:
python setup_env.py --use-freeze --editable
```

This creates `.venv/`, installs the latest dependencies (or the frozen set),
and installs the package itself.

## Handy commands
```bash
# Full paper suite
python analysis/run_master.py --modes all --resume --progress rich --supplementary

# Fast sanity (reduced seeds)
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --progress tqdm --resume
```

See the top‑level README for more on the simulator, figures, and tables.
