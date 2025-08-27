"""
Generate Appendix figure: ONSI vs cross-channel correlation ρcc.

Shows sensitivity of noise performance to residual DA-SERO correlation
after CTRL subtraction. Useful for reviewer questions about correlated noise.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.ieee_plot_style import apply_ieee_style
from analysis.noise_correlation import onsi_curve

# ADD THIS FUNCTION HERE (after imports, before main())
def get_realistic_noise_values():
    """Extract realistic noise values from config or simulation data."""
    try:
        # Try loading from default config
        import yaml
        config_path = project_root / "config" / "default.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        # Use pipeline calculation with default parameters
        from src.pipeline import calculate_proper_noise_sigma
        from src.config_utils import preprocess_config
        
        cfg = preprocess_config(cfg)
        detection_window_s = cfg['pipeline']['symbol_period_s']
        sigma_da, sigma_sero = calculate_proper_noise_sigma(cfg, detection_window_s)
        
        print(f"Using realistic noise values: σ_DA = {sigma_da:.2e}, σ_SERO = {sigma_sero:.2e}")
        return float(sigma_da), float(sigma_sero)
        
    except Exception as e:
        print(f"Could not load realistic noise values ({e}), using unit values")
        return 1.0, 1.0

def main():
    """Generate ONSI vs ρcc figure for appendix."""
    apply_ieee_style()  # Match IEEE paper formatting

    # Get realistic noise values
    sigma_da, sigma_sero = get_realistic_noise_values()
    
    # Compute sensitivity curve with realistic values
    rho, onsi = onsi_curve(sigma_da, sigma_sero, rho_min=-0.2, rho_max=0.2, n=201)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.1, 2.2))  # Single-column IEEE size
    
    # Main curve
    ax.plot(rho, onsi, 'b-', linewidth=2, 
            label=r'ONSI = $\sigma_{I_{\mathrm{diff}}}(\rho)/\sigma_{I_{\mathrm{diff}}}(0)$')
    
    # Reference line at unity
    ax.axhline(1.0, linestyle=':', color='gray', linewidth=1.0, alpha=0.7)
    
    # Formatting
    ax.set_xlabel(r'$\rho_{\mathrm{cc}}$ (DA–SERO residual correlation)')
    ax.set_ylabel(r'Normalized noise ($\downarrow$ better)')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0.85, 1.10)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc='upper center')
    
    # ADD THIS: Show noise values used in the figure
    ax.text(0.02, 0.02, f'σ_DA = {sigma_da:.2e}, σ_SERO = {sigma_sero:.2e}', 
            transform=ax.transAxes, fontsize=6, alpha=0.7)
    
    # FIX: Dynamic annotation computed from actual curve
    pct_lo = (onsi[0] - 1.0) * 100
    pct_hi = (onsi[-1] - 1.0) * 100
    ax.annotate(f'{rho[0]:.1f}→{pct_lo:+.1f}%, {rho[-1]:.1f}→{pct_hi:+.1f}%',
                xy=(0.0, 1.0), xytext=(0.12, 0.95),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    # Save
    outdir = project_root / "results" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"appendix_onsi_vs_rho_cc.{ext}", 
                   dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved to {outdir}/appendix_onsi_vs_rho_cc.*")
    plt.close()