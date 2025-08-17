# analysis/table_maker.py
"""
Generate Table II: Performance comparison of modulation schemes.
Outputs both LaTeX and CSV formats.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def _load_csv_with_fallback(p: Path) -> pd.DataFrame:
    if p.exists():
        return pd.read_csv(p)
    # fallbacks for older runs (e.g. *_uniform.csv)
    cand1 = p.with_name(p.stem + "_uniform.csv")
    cand2 = p.with_name(p.stem + "_zero.csv")
    for cand in (cand1, cand2):
        if cand.exists():
            print(f"Loaded {cand.name} (fallback)")
            return pd.read_csv(cand)
    print(f"Warning: {p.name} not found")
    return pd.DataFrame()

def load_performance_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load performance data for all modulation schemes."""
    results = {}
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        lod_file = data_dir / f"lod_vs_distance_{mode.lower()}.csv"
        df = _load_csv_with_fallback(lod_file)
        if not df.empty:
            results[mode] = df
    return results

def calculate_performance_metrics(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate key performance metrics for Table II."""
    metrics = []
    for mode, df in results.items():
        if df.empty:
            continue
        df_valid = df.dropna(subset=['lod_nm'])
        if df_valid.empty:
            continue
        row = {
            'Modulation': mode,
            'Bits/Symbol': 1 if mode == 'MoSK' else 2,
            'LoD @ 100μm': np.nan,
            'LoD @ 200μm': np.nan,
            'Max Data Rate (bps)': df_valid['data_rate_bps'].max() if 'data_rate_bps' in df_valid else np.nan,
            'Distance @ Max Rate (μm)': np.nan,
        }
        for dist in [100, 200]:
            dist_data = df_valid[df_valid['distance_um'] == dist]
            if not dist_data.empty:
                row[f'LoD @ {dist}μm'] = dist_data['lod_nm'].iloc[0]
        if 'data_rate_bps' in df_valid:
            max_idx = df_valid['data_rate_bps'].idxmax()
            if not pd.isna(max_idx):
                row['Distance @ Max Rate (μm)'] = df_valid.loc[max_idx, 'distance_um']
        metrics.append(row)
    return pd.DataFrame(metrics)

def format_table_latex(df: pd.DataFrame) -> str:
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance Comparison of Modulation Schemes}\n"
    latex += "\\label{tab:performance}\n"
    # Fix 1: Change from 7 columns (lcccccr) to 6 columns (lccccc)
    latex += "\\begin{tabular}{lccccc}\n"
    latex += "\\toprule\n"
    latex += "Modulation & Bits/ & LoD @ 100μm & LoD @ 200μm & Max Rate & Distance @ \\\\\n"
    latex += "Scheme & Symbol & (molecules) & (molecules) & (bps) & Max Rate (μm) \\\\\n"
    latex += "\\midrule\n"
    
    # Fix 2: Proper row formatting with terminators
    for _, row in df.iterrows():
        cells = [
            f"{row['Modulation']}",
            f"{row['Bits/Symbol']}",
            f"{row['LoD @ 100μm']:.0f}" if pd.notna(row['LoD @ 100μm']) else "--",
            f"{row['LoD @ 200μm']:.0f}" if pd.notna(row['LoD @ 200μm']) else "--",
            f"{row['Max Data Rate (bps)']:.3f}" if pd.notna(row['Max Data Rate (bps)']) else "--",
            f"{row['Distance @ Max Rate (μm)']:.0f}" if pd.notna(row['Distance @ Max Rate (μm)']) else "--",
        ]
        latex += " & ".join(cells) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    return latex

def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    columns_order = [
        'Modulation', 'Bits/Symbol', 'LoD @ 100μm', 'LoD @ 200μm',
        'Max Data Rate (bps)', 'Distance @ Max Rate (μm)'
        ]
    df_formatted = df[columns_order].copy()
    if 'LoD @ 100μm' in df_formatted.columns:
        df_formatted['LoD @ 100μm'] = df_formatted['LoD @ 100μm'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "--")
    if 'LoD @ 200μm' in df_formatted.columns:
        df_formatted['LoD @ 200μm'] = df_formatted['LoD @ 200μm'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "--")
    if 'Max Data Rate (bps)' in df_formatted.columns:
        df_formatted['Max Data Rate (bps)'] = df_formatted['Max Data Rate (bps)'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "--")
    if 'Distance @ Max Rate (μm)' in df_formatted.columns:
        df_formatted['Distance @ Max Rate (μm)'] = df_formatted['Distance @ Max Rate (μm)'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "--")
    return df_formatted

def main():
    print("\n" + "="*60)
    print("Generating Table II: Performance Comparison")
    print("="*60)
    results_dir = project_root / "results"
    data_dir = results_dir / "data"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    print("\nLoading performance data...")
    results = load_performance_data(data_dir)
    if not results:
        print("Error: No performance data found. Run analysis first.")
        return
    print("Calculating performance metrics...")
    metrics_df = calculate_performance_metrics(results)
    if metrics_df.empty:
        print("Error: No valid metrics calculated.")
        return
    
    # Stage 15: Ensure stable modulation order (already implemented correctly)
    mode_order = {'MoSK': 0, 'CSK': 1, 'Hybrid': 2}
    metrics_df['sort_order'] = metrics_df['Modulation'].map(mode_order)
    metrics_df = metrics_df.sort_values('sort_order').drop('sort_order', axis=1)
    
    formatted_df = create_comparison_table(metrics_df)
    
    # Stage 15: Standardized output with both CSV and LaTeX
    out_dir = results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) CSV (machine-readable)
    csv_path = out_dir / "table_ii_performance.csv"
    formatted_df.to_csv(csv_path, index=False)
    
    # 2) Raw metrics CSV for analysis
    raw_path = out_dir / "table_ii_raw.csv"
    metrics_df.to_csv(raw_path, index=False)
    
    # 3) LaTeX (camera-ready)
    latex_content = format_table_latex(metrics_df)
    latex_path = out_dir / "table_ii_performance.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:  # Add encoding='utf-8'
        f.write(latex_content)
    
    # Stage 15: Consistent status reporting
    print(f"✓ Wrote Table II CSV/LaTeX to {out_dir}")
    print(f"  - CSV: {csv_path}")
    print(f"  - LaTeX: {latex_path}")
    print(f"  - Raw: {raw_path}")
    
    print("\nTable II: Performance Comparison")
    print("-" * 80)
    print(formatted_df.to_string(index=False))
    print("-" * 80)
    if 'Max Data Rate (bps)' in metrics_df.columns:
        best_rate_idx = metrics_df['Max Data Rate (bps)'].idxmax()
        if not pd.isna(best_rate_idx):
            best_mode = metrics_df.loc[best_rate_idx, 'Modulation']
            best_rate = metrics_df.loc[best_rate_idx, 'Max Data Rate (bps)']
            print(f"- Highest data rate: {best_mode} at {best_rate:.4f} bps")
    print("\n" + "="*60)
    print("Table generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
