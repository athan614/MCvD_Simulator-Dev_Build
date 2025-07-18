# analysis/run_all_analyses.py
"""
Convenience script to run analyses for all modulation schemes
and generate comparative plots.
"""

import subprocess
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_analysis_for_mode(mode: str, num_seeds: int = 20, sequence_length: int = 1000):
    """Run the analysis script for a specific mode."""
    print(f"\n{'='*60}")
    print(f"Running analysis for {mode} mode")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        str(project_root / "analysis" / "run_final_analysis.py"),
        "--mode", mode,
        "--num-seeds", str(num_seeds),
        "--sequence-length", str(sequence_length)
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print(f"✓ {mode} analysis completed in {elapsed:.1f} seconds")
        
        # Print any important output
        if "Summary Statistics" in result.stdout:
            lines = result.stdout.split('\n')
            summary_start = next(i for i, line in enumerate(lines) if "Summary Statistics" in line)
            print("\n".join(lines[summary_start:summary_start+10]))
            
    except subprocess.CalledProcessError as e:
        print(f"✗ {mode} analysis failed!")
        print(f"Error: {e.stderr}")
        return False
    
    return True


def main():
    """Run all analyses and generate comparative plots."""
    print("\n" + "="*80)
    print("RUNNING COMPLETE ANALYSIS FOR ALL MODULATION SCHEMES")
    print("="*80)
    
    # Parse any command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run all modulation scheme analyses")
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of Monte Carlo seeds")
    parser.add_argument("--sequence-length", type=int, default=1000, help="Symbols per run")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis with fewer seeds")
    args = parser.parse_args()
    
    if args.quick:
        print("\nRunning in QUICK mode (reduced seeds for faster execution)")
        num_seeds = min(5, args.num_seeds)
        sequence_length = min(200, args.sequence_length)
    else:
        num_seeds = args.num_seeds
        sequence_length = args.sequence_length
    
    print(f"\nParameters:")
    print(f"  - Seeds per point: {num_seeds}")
    print(f"  - Sequence length: {sequence_length}")
    
    # Run analyses for each mode
    modes = ["MoSK", "CSK", "Hybrid"]
    success_count = 0
    
    total_start = time.time()
    
    for mode in modes:
        if run_analysis_for_mode(mode, num_seeds, sequence_length):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Individual analyses complete: {success_count}/{len(modes)} successful")
    print(f"{'='*60}")
    
    # Generate comparative plots if at least 2 modes succeeded
    if success_count >= 2:
        print("\nGenerating comparative plots...")
        try:
            result = subprocess.run([
                sys.executable,
                str(project_root / "analysis" / "generate_comparative_plots.py")
            ], check=True, capture_output=True, text=True)
            print("✓ Comparative plots generated successfully!")
        except subprocess.CalledProcessError as e:
            print("✗ Failed to generate comparative plots")
            print(f"Error: {e.stderr}")
        
        # Generate supplementary figures
        print("\nGenerating supplementary figures...")
        try:
            result = subprocess.run([
                sys.executable,
                str(project_root / "analysis" / "generate_supplementary_figures.py")
            ], check=True, capture_output=True, text=True)
            print("✓ Supplementary figures generated successfully!")
        except subprocess.CalledProcessError as e:
            print("✗ Failed to generate supplementary figures")
            print(f"Error: {e.stderr}")
        
        # Generate Table II
        print("\nGenerating Table II...")
        try:
            result = subprocess.run([
                sys.executable,
                str(project_root / "analysis" / "table_maker.py")
            ], check=True, capture_output=True, text=True)
            print("✓ Table II generated successfully!")
        except subprocess.CalledProcessError as e:
            print("✗ Failed to generate Table II")
            print(f"Error: {e.stderr}")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"Total execution time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved in: {project_root / 'results'}")
    print(f"{'='*80}")
    
    # Print final summary
    results_dir = project_root / "results"
    if (results_dir / "data" / "performance_summary.csv").exists():
        print("\nPerformance Summary:")
        import pandas as pd
        summary = pd.read_csv(results_dir / "data" / "performance_summary.csv")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()