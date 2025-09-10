# Contributing

Thanks for helping improve the Tri‑Channel OECT MC simulator!

## Ground rules
- Keep physics & interfaces stable (transport/binding/OECT/detection). Changes that alter equations belong in a new branch and a design doc.
- Favor vectorized NumPy; avoid per‑symbol Python loops in hot paths.
- Add tests for new logic; ensure `pytest -q` passes.

## Dev environment
```bash
pip install -e .[dev]
python setup_project.py
pytest -q
ruff check . && black --check . && mypy .
```

## Style
- Python ≥3.11, type-hinted, formatted with **black**, linted with **ruff**, type‑checked with **mypy**.
- Plotting: import and call `analysis/ieee_plot_style.apply_ieee_style()` for any new figures.
- Figures: ≥300 dpi, consistent fonts/markers; include 95% CI where applicable.

## Pull requests
1. Open an issue summarizing the change and why it’s needed.
2. Keep PRs focused; include benchmarks if runtime changes.
3. Update README/CLI help if you add flags or outputs.
4. Include before/after images for plot changes.

## Reproducibility checklist
- Independent RNG (`SeedSequence`) per worker.
- Resume & checkpoints verified with `--resume` (no duplicate work).
- Calibration cache keys update when Nm/distance/guard changes.

## License & attribution
- By contributing, you agree to license your contribution under the repository’s MIT license.
- If you add data/figures from external sources, include clear attribution.
