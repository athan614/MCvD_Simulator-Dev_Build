# Setup (Latest Packages + Enhanced Features)

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

### Full paper suite (enhanced)
```bash
# Sequential execution (traditional)
python analysis/run_master.py --modes all --resume --progress rich --supplementary

# Parallel execution (3x faster)
python analysis/run_master.py --modes all --resume --progress rich --supplementary --parallel-modes 3

# With custom logging
python analysis/run_master.py --modes all --resume --progress rich --logdir ./paper_logs
```

### Fast sanity checks
```bash
# Single mode with reduced seeds
python analysis/run_final_analysis.py --mode CSK --num-seeds 4 --sequence-length 200 --recalibrate --progress tqdm --resume

# Parallel sanity check
python analysis/run_final_analysis.py --mode ALL --num-seeds 4 --sequence-length 200 --parallel-modes 3
```

### Mechanism figure generation
```bash
# Generate all notebook-replica mechanism figures
python analysis/rebuild_oect_figs.py
python analysis/rebuild_binding_figs.py  
python analysis/rebuild_transport_figs.py
python analysis/rebuild_pipeline_figs.py

# Or run via master (included in nb_replicas step)
python analysis/run_master.py --modes all
```

## New Features

### 🚀 Parallel Mode Execution
- `--parallel-modes N`: Run N modulation modes concurrently
- Thread-based orchestration with shared process pool
- 3x speedup for full simulation suite
- Automatic progress bar multiplexing with Rich backend

### 📝 Real-Time Logging  
- `--logdir DIR`: Custom log directory (default: results/logs/)
- `--no-log`: Disable file logging
- `--fsync-logs`: Force OS sync for critical systems
- All console output mirrored to timestamped files in real-time
- Monitor progress: `tail -f results/logs/run_master_*.log`

### 🖼️ Notebook-Replica Figures
- Detailed mechanism visualizations in `results/figures/notebook_replicas/`
- OECT noise analysis (differential PSD, noise breakdown)
- Binding kinetics (deterministic vs Monte Carlo, PSD analysis)
- Transport/diffusion (concentration profiles, scaling analysis)  
- End-to-end pipeline with ISI effects

See the top‑level README for more on the simulator, figures, and tables.