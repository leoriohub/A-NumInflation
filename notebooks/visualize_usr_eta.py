
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add root path to allow imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from inflation_models import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def plot_usr_eta_standard(xi, yi, phi_i=5.5):
    """
    Plots the evolution of eta defined as d(ln epsilon) / dN.
    This definition yields eta ~ -6 during Ultra Slow Roll (USR).
    """
    print(f"Simulating: xi={xi}, y_i={yi}, phi_i={phi_i}")
    
    # 1. Setup Model
    model = HiggsModel(lam=0.1, xi=xi)
    model.phi0 = phi_i
    model.yi = yi
    
    # 2. Run Simulation
    # Use sufficient resolution for numerical derivatives
    T_max = max(100, xi/10) 
    t_span = np.linspace(0, T_max, 5000)
    
    try:
        sol = run_background_simulation(model, t_span)
    except Exception as e:
        print(f"Simulation failed: {e}")
        return

    # 3. Calculate Derived Quantities
    data = get_derived_quantities(sol, model)
    N = data['N']
    eps = data['epsH']
    
    # 4. Calculate Eta (Standard Definition: d ln(eps) / dN)
    # Using numpy gradient for numerical differentiation with respect to N
    # This handles non-uniform N spacing correctly
    ln_eps = np.log(eps)
    eta_std = np.gradient(ln_eps, N)
    
    # 5. Smart Truncation at End of Inflation
    if np.any(eps >= 1.0):
        end_idx = np.argmax(eps >= 1.0)
        cutoff = min(len(N), end_idx + 20)
    else:
        cutoff = len(N)
        
    N_plot = N[:cutoff]
    eta_plot = eta_std[:cutoff]
    
    # 6. Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(N_plot, eta_plot, label=r'$\eta = \frac{d \ln \epsilon}{dN}$', linewidth=2, color='blue')
    
    # Reference Lines
    plt.axhline(0, color='k', linestyle='-', alpha=0.3, label='Slow Roll (0)')
    plt.axhline(-6, color='r', linestyle='--', alpha=0.8, label='USR Limit (-6)')
    
    # Highlight Region near -6
    is_usr = (eta_plot < -4) & (eta_plot > -8)
    plt.fill_between(N_plot, eta_plot, -6, where=is_usr, color='red', alpha=0.2, label='USR Phase')
    
    plt.xlabel('e-folds ($N$)')
    plt.ylabel(r'$\eta$Parameter')
    plt.title(f'Evolution of $\eta = d\ln\epsilon/dN$\n($\\xi={xi}, y_i={yi}, \phi_i={phi_i}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Focus Y-axis to relevant range
    plt.ylim(-10, 5)
    
    output_path = os.path.join(root_dir, 'images', 'usr_eta_evolution.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # Test with a high velocity to see the USR phase
    plot_usr_eta_standard(xi=1000.0, yi=-250.0, phi_i=5.5)
