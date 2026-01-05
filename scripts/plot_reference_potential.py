
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inflation_models import NonMinimalQuarticModel

def plot_potential_with_cmb_mark():
    # Parameters
    XI = 1000.0
    LAMBDA = 0.01
    
    # Initialize Model
    model = NonMinimalQuarticModel(xi=XI, lam=LAMBDA)
    model.S = 1.0
    
    # CMB Horizon Exit (calculated previously)
    phi_cmb = 5.4248
    V_cmb = model.v0 * model.f(phi_cmb)
    
    # Plot Range
    # Show the plateau and the drop to vacuum
    phi_grid = np.linspace(0, 10, 500)
    V_vals = [model.v0 * model.f(p) for p in phi_grid]
    
    plt.figure(figsize=(10, 7))
    plt.plot(phi_grid, V_vals, label=r'Potential $V(\phi)$', linewidth=2)
    
    # Mark CMB scale
    plt.plot(phi_cmb, V_cmb, 'ro', markersize=8, label='CMB Horizon Exit ($N \\approx 60$)')
    plt.axvline(phi_cmb, color='r', linestyle=':', alpha=0.5)
    plt.text(phi_cmb + 0.2, V_cmb, f'$\\phi_{{60}} \\approx {phi_cmb:.2f}\\ M_P$', color='r', va='center')
    
    # Annotations
    plt.xlabel(r'$\phi [M_P]$', fontsize=12)
    plt.ylabel(r'$V(\phi) [M_P^4]$', fontsize=12)
    plt.title(f'Higgs Inflation Potential (Einstein Frame, $\\xi={XI}$)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in slightly to show the plateau structure relevant to inflation
    plt.xlim(0, 10)
    plt.ylim(0, max(V_vals)*1.1)
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots/potential_cmb_reference.png'))
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    plot_potential_with_cmb_mark()
