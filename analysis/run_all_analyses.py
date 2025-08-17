"""
(DEPRECATED wrapper) — Please use `analysis/run_master.py`.

This thin wrapper forwards all CLI flags to run_master for backward compatibility.

CLI (kept):
  python analysis/run_all_analyses.py --progress rich --resume --nt-pairs "GLU-GABA,GLU-DA,ACH-GABA"
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Reuse the real master entrypoint
from analysis.run_master import main as _master_main  # type: ignore

if __name__ == "__main__":
    _master_main()
