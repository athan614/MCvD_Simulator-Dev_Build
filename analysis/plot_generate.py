#!/usr/bin/env python3
"""
Standalone script to generate plots from existing CSV files.
Place this in your analysis folder and run it to generate the missing plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional

def plot_ser_vs_nm(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    """Plot SER vs Nm for all available modulation schemes."""
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode, df in results_dict.items():
        if 'pipeline.Nm_per_symbol' in df.columns and 'ser' in df.columns:
            # Sort by Nm for proper line plotting
            df_sorted = df.sort_values('pipeline.Nm_per_symbol')
            
            plt.loglog(df_sorted['pipeline.Nm_per_symbol'], df_sorted['ser'],
                      color=colors.get(mode, 'black'),
                      marker=markers.get(mode, 'o'),
                      markersize=8,
                      label=mode,
                      linewidth=2,
                      linestyle='-',
                      alpha=0.8)
    
    plt.xlabel('Number of Molecules per Symbol (Nm)', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    plt.title('SER vs. Nm for All Modulation Schemes', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    plt.ylim(1e-4, 1)
    plt.xlim(1e2, 1e5)
    
    # Add target SER line
    plt.axhline(y=0.01, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    plt.text(3e2, 0.012, 'Target SER = 1%', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved SER vs Nm plot to {save_path}")
    plt.show()

def plot_lod_vs_distance(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    """Plot LoD vs Distance for all available modulation schemes."""
    plt.figure(figsize=(10, 6))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    for mode, df in results_dict.items():
        if 'distance_um' in df.columns and 'lod_nm' in df.columns:
            # Remove NaN values and sort
            df_valid = df.dropna(subset=['lod_nm'])
            df_sorted = df_valid.sort_values('distance_um')
            
            if len(df_sorted) > 0:
                plt.semilogy(df_sorted['distance_um'], df_sorted['lod_nm'],
                            color=colors.get(mode, 'black'),
                            marker=markers.get(mode, 'o'),
                            markersize=8,
                            label=mode,
                            linewidth=2,
                            linestyle='-',
                            alpha=0.8)
    
    plt.xlabel('Distance (μm)', fontsize=12)
    plt.ylabel('Limit of Detection (molecules)', fontsize=12)
    plt.title('Limit of Detection vs. Distance', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    
    # Set reasonable y-axis limits if data exists
    all_lods = []
    for mode, df in results_dict.items():
        if 'lod_nm' in df.columns:
            valid_lods = df['lod_nm'].dropna()
            if len(valid_lods) > 0:
                all_lods.extend(valid_lods.tolist())
    
    if all_lods:
        min_lod = min(all_lods) * 0.5
        max_lod = max(all_lods) * 2
        plt.ylim(min_lod, max_lod)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved LoD vs Distance plot to {save_path}")
    plt.show()

def plot_combined_metrics(results_dict: Dict[str, pd.DataFrame], save_path: Path):
    """Create a 2x2 subplot with additional metrics if available."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'MoSK': 'blue', 'CSK': 'green', 'Hybrid': 'red'}
    markers = {'MoSK': 'o', 'CSK': 's', 'Hybrid': '^'}
    
    # Plot 1: SER vs Nm
    ax1 = axes[0, 0]
    for mode, df in results_dict.items():
        if 'pipeline.Nm_per_symbol' in df.columns and 'ser' in df.columns:
            df_sorted = df.sort_values('pipeline.Nm_per_symbol')
            ax1.loglog(df_sorted['pipeline.Nm_per_symbol'], df_sorted['ser'],
                      color=colors.get(mode, 'black'), marker=markers.get(mode, 'o'),
                      markersize=6, label=mode, linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Number of Molecules (Nm)')
    ax1.set_ylabel('SER')
    ax1.set_title('(a) SER vs. Nm')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.axhline(y=0.01, color='k', linestyle=':', alpha=0.5)
    ax1.legend(loc='best')
    
    # Plot 2: SNR vs Nm (if available)
    ax2 = axes[0, 1]
    has_snr = False
    for mode, df in results_dict.items():
        if 'pipeline.Nm_per_symbol' in df.columns and 'snr_db' in df.columns:
            df_sorted = df.sort_values('pipeline.Nm_per_symbol')
            ax2.semilogx(df_sorted['pipeline.Nm_per_symbol'], df_sorted['snr_db'],
                        color=colors.get(mode, 'black'), marker=markers.get(mode, 'o'),
                        markersize=6, label=mode, linewidth=1.5, alpha=0.8)
            has_snr = True
    if has_snr:
        ax2.set_xlabel('Number of Molecules (Nm)')
        ax2.set_ylabel('SNR (dB)')
        ax2.set_title('(b) SNR vs. Nm')
        ax2.grid(True, which="both", ls="--", alpha=0.3)
        ax2.legend(loc='best')
    else:
        ax2.text(0.5, 0.5, 'SNR data not available', ha='center', va='center')
        ax2.set_title('(b) SNR vs. Nm')
    
    # Plot 3: LoD vs Distance
    ax3 = axes[1, 0]
    for mode, df in results_dict.items():
        if 'distance_um' in df.columns and 'lod_nm' in df.columns:
            df_valid = df.dropna(subset=['lod_nm'])
            df_sorted = df_valid.sort_values('distance_um')
            if len(df_sorted) > 0:
                ax3.semilogy(df_sorted['distance_um'], df_sorted['lod_nm'],
                            color=colors.get(mode, 'black'), marker=markers.get(mode, 'o'),
                            markersize=6, label=mode, linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Distance (μm)')
    ax3.set_ylabel('LoD (molecules)')
    ax3.set_title('(c) LoD vs. Distance')
    ax3.grid(True, which="both", ls="--", alpha=0.3)
    ax3.legend(loc='best')
    
    # Plot 4: Data Rate vs Distance (if available)
    ax4 = axes[1, 1]
    has_datarate = False
    for mode, df in results_dict.items():
        if 'distance_um' in df.columns and 'data_rate_bps' in df.columns:
            df_valid = df.dropna(subset=['data_rate_bps'])
            df_sorted = df_valid.sort_values('distance_um')
            if len(df_sorted) > 0:
                ax4.plot(df_sorted['distance_um'], df_sorted['data_rate_bps'],
                        color=colors.get(mode, 'black'), marker=markers.get(mode, 'o'),
                        markersize=6, label=mode, linewidth=1.5, alpha=0.8)
                has_datarate = True
    if has_datarate:
        ax4.set_xlabel('Distance (μm)')
        ax4.set_ylabel('Data Rate (bits/s)')
        ax4.set_title('(d) Data Rate vs. Distance')
        ax4.grid(True, which="both", ls="--", alpha=0.3)
        ax4.legend(loc='best')
    else:
        ax4.text(0.5, 0.5, 'Data rate not available', ha='center', va='center')
        ax4.set_title('(d) Data Rate vs. Distance')
    
    plt.suptitle('Molecular Communication System Performance Metrics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined metrics plot to {save_path}")
    plt.show()

def main():
    """Main function to generate plots from CSV files."""
    parser = argparse.ArgumentParser(description='Generate plots from CSV files')
    parser.add_argument('--data-dir', type=str, default='results/data',
                       help='Directory containing CSV files (default: results/data)')
    parser.add_argument('--figures-dir', type=str, default='results/figures',
                       help='Directory to save plots (default: results/figures)')
    parser.add_argument('--modes', nargs='+', default=['MoSK', 'CSK', 'Hybrid'],
                       help='Modulation modes to plot')
    parser.add_argument('--combined', action='store_true',
                       help='Generate combined 2x2 subplot figure')
    parser.add_argument('--ser-file', type=str, default=None,
                       help='Direct path to SER vs Nm CSV file')
    parser.add_argument('--lod-file', type=str, default=None,
                       help='Direct path to LoD vs Distance CSV file')
    parser.add_argument('--mode-label', type=str, default='MoSK',
                       help='Mode label for direct file plotting')
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    
    # Create figures directory if it doesn't exist
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle direct file specification
    if args.ser_file or args.lod_file:
        print("\n📂 Using directly specified files:")
        ser_data = {}
        lod_data = {}
        
        if args.ser_file:
            ser_path = Path(args.ser_file)
            if ser_path.exists():
                df = pd.read_csv(ser_path)
                ser_data[args.mode_label] = df
                print(f"✅ Loaded SER data from {ser_path.name}")
                print(f"   Columns: {', '.join(df.columns)}")
            else:
                print(f"⚠️  File not found: {ser_path}")
        
        if args.lod_file:
            lod_path = Path(args.lod_file)
            if lod_path.exists():
                df = pd.read_csv(lod_path)
                lod_data[args.mode_label] = df
                print(f"✅ Loaded LoD data from {lod_path.name}")
                print(f"   Columns: {', '.join(df.columns)}")
            else:
                print(f"⚠️  File not found: {lod_path}")
    else:
        # Automatic file detection (existing code with improvements)
        print(f"\n{'='*60}")
        print("📊 PLOT GENERATION FROM CSV FILES")
        print(f"{'='*60}")
        print(f"Data directory: {data_dir.absolute()}")
        print(f"Figures directory: {figures_dir.absolute()}")
        print(f"Looking for modes: {args.modes}\n")
        
        # First, list all CSV files in the directory
        print("📁 Files found in data directory:")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                for f in csv_files:
                    print(f"   - {f.name}")
            else:
                print("   No CSV files found!")
        else:
            print(f"   ⚠️ Directory does not exist: {data_dir}")
        print()
        
        # Function to find file case-insensitively
        def find_csv_file(directory: Path, pattern: str) -> Optional[Path]:
            """Find CSV file with case-insensitive matching."""
            for file in directory.glob("*.csv"):
                if pattern.lower() in file.name.lower():
                    return file
            # Also try with different separators
            alt_pattern = pattern.replace('_', '-')
            for file in directory.glob("*.csv"):
                if alt_pattern.lower() in file.name.lower():
                    return file
            return None
        
        # Load CSV files for SER vs Nm
        ser_data = {}
        for mode in args.modes:
            # Try multiple naming patterns
            patterns = [
                f"ser_vs_nm_{mode}",
                f"ser_vs_Nm_{mode}",
                f"SER_vs_Nm_{mode}",
                f"ser-vs-nm-{mode}"
            ]
            
            csv_path = None
            for pattern in patterns:
                found_file = find_csv_file(data_dir, pattern)
                if found_file:
                    csv_path = found_file
                    break
            
            if csv_path and csv_path.exists():
                df = pd.read_csv(csv_path)
                ser_data[mode] = df
                print(f"✅ Loaded {csv_path.name} ({len(df)} data points)")
                print(f"   Columns: {', '.join(df.columns[:5])}")  # Show first 5 columns
            else:
                print(f"⚠️  Not found: ser_vs_nm_{mode.lower()}.csv (or similar)")
        
        # Load CSV files for LoD vs Distance  
        lod_data = {}
        for mode in args.modes:
            # Try multiple naming patterns
            patterns = [
                f"lod_vs_distance_{mode}",
                f"LoD_vs_distance_{mode}",
                f"LOD_vs_Distance_{mode}",
                f"lod-vs-distance-{mode}"
            ]
            
            csv_path = None
            for pattern in patterns:
                found_file = find_csv_file(data_dir, pattern)
                if found_file:
                    csv_path = found_file
                    break
            
            if csv_path and csv_path.exists():
                df = pd.read_csv(csv_path)
                lod_data[mode] = df
                print(f"✅ Loaded {csv_path.name} ({len(df)} data points)")
                print(f"   Columns: {', '.join(df.columns[:5])}")  # Show first 5 columns
            else:
                print(f"⚠️  Not found: lod_vs_distance_{mode.lower()}.csv (or similar)")
    
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}\n")
    
    # Generate SER vs Nm plot
    if ser_data:
        plot_path = figures_dir / "ser_vs_nm_all.png"
        plot_ser_vs_nm(ser_data, plot_path)
    else:
        print("⚠️  No SER vs Nm data found")
    
    # Generate LoD vs Distance plot
    if lod_data:
        plot_path = figures_dir / "lod_vs_distance_all.png"
        plot_lod_vs_distance(lod_data, plot_path)
    else:
        print("⚠️  No LoD vs Distance data found")
    
    # Generate combined plot if requested
    if args.combined and (ser_data or lod_data):
        # Merge all data for combined plot
        all_data = {}
        for mode in args.modes:
            if mode in ser_data or mode in lod_data:
                # Merge SER and LoD data
                if mode in ser_data and mode in lod_data:
                    all_data[mode] = ser_data[mode]
                    # Add LoD columns to the dataframe
                    for col in lod_data[mode].columns:
                        if col not in all_data[mode].columns:
                            all_data[mode] = all_data[mode].copy()
                elif mode in ser_data:
                    all_data[mode] = ser_data[mode]
                elif mode in lod_data:
                    all_data[mode] = lod_data[mode]
        
        if all_data:
            plot_path = figures_dir / "combined_metrics.png"
            plot_combined_metrics(all_data, plot_path)
    
    print(f"\n{'='*60}")
    print("✅ PLOT GENERATION COMPLETE!")
    print(f"{'='*60}")
    
    # Summary statistics
    print("\n📈 Data Summary:")
    for mode in args.modes:
        print(f"\n{mode} Mode:")
        if mode in ser_data:
            df = ser_data[mode]
            if 'ser' in df.columns:
                print(f"  SER range: {df['ser'].min():.2e} to {df['ser'].max():.2e}")
        if mode in lod_data:
            df = lod_data[mode]
            if 'lod_nm' in df.columns:
                valid_lod = df['lod_nm'].dropna()
                if len(valid_lod) > 0:
                    print(f"  LoD range: {valid_lod.min():.0f} to {valid_lod.max():.0f} molecules")
                    print(f"  Distances with LoD: {len(valid_lod)}/{len(df)}")

if __name__ == "__main__":
    main()