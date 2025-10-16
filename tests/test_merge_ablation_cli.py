import csv
import os
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_ser_merge_ablation_cli(tmp_path: Path) -> None:
    data_dir = tmp_path / "results" / "data"
    ctrl_rows = [
        {"pipeline_Nm_per_symbol": 1000, "ser": 1e-3, "use_ctrl": True},
        {"pipeline_Nm_per_symbol": 2000, "ser": 5e-4, "use_ctrl": True},
    ]
    noctrl_rows = [
        {"pipeline_Nm_per_symbol": 1500, "ser": 2e-3, "use_ctrl": False},
        {"pipeline_Nm_per_symbol": 2000, "ser": 1e-3, "use_ctrl": False},
    ]

    _write_csv(data_dir / "ser_vs_nm_mosk__ctrl.csv", ctrl_rows)
    _write_csv(data_dir / "ser_vs_nm_mosk__noctrl.csv", noctrl_rows)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    subprocess.run(
        [
            sys.executable,
            "analysis/run_final_analysis.py",
            "--merge-ablation-csvs",
            "--mode",
            "MoSK",
            "--merge-data-dir",
            str(data_dir),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    canonical = data_dir / "ser_vs_nm_mosk.csv"
    assert canonical.exists(), "Canonical SER CSV was not created"

    with canonical.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4, "Canonical CSV should contain union of both branches"
    nm_use_ctrl = {
        (float(row["pipeline_Nm_per_symbol"]), row["use_ctrl"].strip().lower())
        for row in rows
    }
    expected = {
        (1000.0, "true"),
        (2000.0, "true"),
        (1500.0, "false"),
        (2000.0, "false"),
    }
    assert nm_use_ctrl == expected
