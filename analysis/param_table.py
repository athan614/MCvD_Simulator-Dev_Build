# analysis/param_table.py
import sys
from pathlib import Path

# Add the src directory to the Python path to allow for package imports
# This is a common pattern for analysis scripts living outside the main package
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.mc_receiver import oect, binding

# --- ADD THIS HELPER FUNCTION ---
def format_value(value, unit=""):
    """
    Formats a number into a clean string for the LaTeX table.
    Handles scientific notation and adds units.
    """
    if isinstance(value, str):  # If it's already a string, just add unit
        return f"{value} {unit}".strip()

    if 0.01 < abs(value) < 1000:
        # For "normal" numbers, use regular formatting
        return f"{value:.1f} {unit}".strip()
    else:
        # For very large or small numbers, use scientific notation
        s = f"{value:.1e}"
        mantissa, exponent = s.split('e')
        # Format as: 3.0 \times 10^{-3}
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$ {unit}".strip()

# --- The generate_latex_table() function starts below ---

def generate_latex_table():
    """Pulls default parameters and generates LaTeX for Table I."""

    # --- Fetch parameters from modules ---
    oect_params = oect.default_params()
    binding_params = binding.default_params()

    # --- Define table data using fetched parameters ---
    # We structure this as a list of dictionaries for clarity
    table_sections = [
        {
            "title": "Electrical Parameters",
            "data": [
                ("Transconductance ($g_m$)", format_value(oect_params['gm_S'] * 1000, "mS")),
                ("Total Capacitance ($C_{tot}$)", format_value(oect_params['C_tot_F'] * 1e9, "nF")),
                ("Hooge Parameter ($\\alpha_H$)", format_value(oect_params['alpha_H'])), # Unit is dimensionless
                ("Drift Coefficient ($K_d$)", format_value(oect_params['K_d_Hz'], "Hz")),
            ]
        },
        {
            "title": "Binding Parameters",
            "data": [
                ("Aptamer Sites ($N_{apt}$)", format_value(binding_params['N_apt'])),
                ("$k_{on,GLU}$", format_value(binding_params['GLU']['k_on_M_s'], "M$^{-1}$s$^{-1}$")),
                ("$k_{off,GLU}$", format_value(binding_params['GLU']['k_off_s'], "s$^{-1}$")),
                ("$k_{on,GABA}$", format_value(binding_params['GABA']['k_on_M_s'], "M$^{-1}$s$^{-1}$")),
                ("$k_{off,GABA}$", format_value(binding_params['GABA']['k_off_s'], "s$^{-1}$")),
            ]
        },
    ]

    # --- Generate LaTeX String ---
    latex_string = "\\begin{tabular}{ll}\n"
    latex_string += "\\toprule\n"
    latex_string += "Parameter & Value \\\\\n"
    latex_string += "\\midrule\n"

    for section in table_sections:
        latex_string += f"\\multicolumn{{2}}{{l}}{{\\textbf{{{section['title']}}}}} \\\\\n"
        for param, value in section['data']:
            # Conditionally escape underscores only if not in math mode
            if '$' not in param:
                param = param.replace('_', '\\_')
            latex_string += f"{param} & {value} \\\\\n"
        latex_string += "\\midrule\n"
    
    # Remove the last midrule and add a bottomrule
    latex_string = latex_string.rsplit('\\midrule\n', 1)[0]
    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}\n"

    return latex_string

if __name__ == "__main__":
    latex_output = generate_latex_table()
    print("--- LaTeX Code for Table I ---")
    print(latex_output)

    # Optional: Save to a file
    output_path = project_root / "results" / "tables"
    output_path.mkdir(exist_ok=True)
    (output_path / "table1.tex").write_text(latex_output)
    print(f"\\nTable saved to: {output_path / 'table1.tex'}")