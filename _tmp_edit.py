from pathlib import Path
path = Path('analysis/run_final_analysis.py')
lines = path.read_text(encoding='utf-8').splitlines()
for idx, line in enumerate(lines):
    if line.strip() == "guard_analysis: List[Tuple[float, float]] = []":
        lines.insert(idx, '            df_guard_new = pd.DataFrame()')
        break
else:
    raise SystemExit('guard_analysis line not found')
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
