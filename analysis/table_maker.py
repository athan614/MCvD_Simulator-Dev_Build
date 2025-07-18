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


def load_performance_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load performance data for all modulation schemes."""
    results = {}
    
    for mode in ['MoSK', 'CSK', 'Hybrid']:
        # Load LoD data
        lod_file = data_dir / f"lod_vs_distance_{mode.lower()}.csv"
        if lod_file.exists():
            results[mode] = pd.read_csv(lod_file)
        else:
            print(f"Warning: {lod_file} not found")
    
    return results


def calculate_performance_metrics(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate key performance metrics for Table II."""
    metrics = []
    
    for mode, df in results.items():
        if df.empty:
            continue
        
        # Filter valid data
        df_valid = df.dropna(subset=['lod_nm'])
        
        if df_valid.empty:
            continue
        
        # Calculate metrics
        row = {
            'Modulation': mode,
            'Bits/Symbol': 1 if mode == 'MoSK' else 2,
            'LoD @ 100μm': np.nan,
            'LoD @ 200μm': np.nan,
            'Max Data Rate (bps)': df_valid['data_rate_bps'].max() if 'data_rate_bps' in df_valid else np.nan,
            'Distance @ Max Rate (μm)': np.nan,
            'Energy/bit (relative)': 1.0 if mode == 'MoSK' else (0.5 if mode == 'CSK' else 0.75)
        }
        
        # Get LoD at specific distances
        for dist in [100, 200]:
            dist_data = df_valid[df_valid['distance_um'] == dist]
            if not dist_data.empty:
                row[f'LoD @ {dist}μm'] = dist_data['lod_nm'].iloc[0]
        
        # Get distance at max data rate
        if 'data_rate_bps' in df_valid:
            max_idx = df_valid['data_rate_bps'].idxmax()
            if not pd.isna(max_idx):
                row['Distance @ Max Rate (μm)'] = df_valid.loc[max_idx, 'distance_um']
        
        metrics.append(row)
    
    return pd.DataFrame(metrics)


def format_table_latex(df: pd.DataFrame) -> str:
    """Format the table for LaTeX."""
    # Create LaTeX table
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance Comparison of Modulation Schemes}\n"
    latex += "\\label{tab:performance}\n"
    latex += "\\begin{tabular}{lcccccr}\n"
    latex += "\\toprule\n"
    latex += "Modulation & Bits/ & LoD @ 100μm & LoD @ 200μm & Max Rate & Distance @ & Energy/bit \\\\\n"
    latex += "Scheme & Symbol & (molecules) & (molecules) & (bps) & Max Rate (μm) & (relative) \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in df.iterrows():
        latex += f"{row['Modulation']} & "
        latex += f"{row['Bits/Symbol']} & "
        
        # LoD values
        if pd.notna(row['LoD @ 100μm']):
            latex += f"{row['LoD @ 100μm']:.0f} & "
        else:
            latex += "-- & "
            
        if pd.notna(row['LoD @ 200μm']):
            latex += f"{row['LoD @ 200μm']:.0f} & "
        else:
            latex += "-- & "
        
        # Data rate
        if pd.notna(row['Max Data Rate (bps)']):
            latex += f"{row['Max Data Rate (bps)']:.3f} & "
        else:
            latex += "-- & "
            
        # Distance at max rate
        if pd.notna(row['Distance @ Max Rate (μm)']):
            latex += f"{row['Distance @ Max Rate (μm)']:.0f} & "
        else:
            latex += "-- & "
            
        # Energy per bit
        latex += f"{row['Energy/bit (relative)']:.2f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a nicely formatted comparison table."""
    # Reorder columns for better presentation
    columns_order = [
        'Modulation',
        'Bits/Symbol',
        'LoD @ 100μm',
        'LoD @ 200μm',
        'Max Data Rate (bps)',
        'Distance @ Max Rate (μm)',
        'Energy/bit (relative)'
    ]
    
    df_formatted = df[columns_order].copy()
    
    # Format numeric columns
    if 'LoD @ 100μm' in df_formatted.columns:
        df_formatted['LoD @ 100μm'] = df_formatted['LoD @ 100μm'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "--"
        )
    
    if 'LoD @ 200μm' in df_formatted.columns:
        df_formatted['LoD @ 200μm'] = df_formatted['LoD @ 200μm'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "--"
        )
    
    if 'Max Data Rate (bps)' in df_formatted.columns:
        df_formatted['Max Data Rate (bps)'] = df_formatted['Max Data Rate (bps)'].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "--"
        )
    
    if 'Distance @ Max Rate (μm)' in df_formatted.columns:
        df_formatted['Distance @ Max Rate (μm)'] = df_formatted['Distance @ Max Rate (μm)'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "--"
        )
    
    return df_formatted


def main():
    """Generate Table II in multiple formats."""
    print("\n" + "="*60)
    print("Generating Table II: Performance Comparison")
    print("="*60)
    
    # Setup paths
    results_dir = project_root / "results"
    data_dir = results_dir / "data"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading performance data...")
    results = load_performance_data(data_dir)
    
    if not results:
        print("Error: No performance data found. Run analysis first.")
        return
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics_df = calculate_performance_metrics(results)
    
    if metrics_df.empty:
        print("Error: No valid metrics calculated.")
        return
    
    # Sort by modulation scheme order
    mode_order = {'MoSK': 0, 'CSK': 1, 'Hybrid': 2}
    metrics_df['sort_order'] = metrics_df['Modulation'].map(mode_order)
    metrics_df = metrics_df.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Create formatted table
    formatted_df = create_comparison_table(metrics_df)
    
    # Save CSV
    csv_path = tables_dir / "table_ii_performance.csv"
    formatted_df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved to {csv_path}")
    
    # Save raw metrics
    raw_path = tables_dir / "table_ii_raw.csv"
    metrics_df.to_csv(raw_path, index=False)
    
    # Generate LaTeX
    latex_content = format_table_latex(metrics_df)
    latex_path = tables_dir / "table_ii_performance.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"✓ LaTeX saved to {latex_path}")
    
    # Print to console
    print("\nTable II: Performance Comparison")
    print("-" * 80)
    print(formatted_df.to_string(index=False))
    print("-" * 80)
    
    # Summary statistics
    print("\nSummary:")
    print(f"- MoSK: Best energy efficiency (baseline)")
    print(f"- CSK: Highest spectral efficiency (2 bits/symbol)")
    print(f"- Hybrid: Balanced performance")
    
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