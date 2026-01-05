
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import traceback

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inflation_models import NonMinimalQuarticModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def run_background_simulation_safe(model, T_span):
    try:
        sol = run_background_simulation(model, T_span)
        if sol is None or len(sol) == 0:
            return None
        return sol
    except Exception:
        # traceback.print_exc()
        return None

def run_scan_script():
    print("Initializing scan...")
    # Parameters
    XI = 1000.0
    LAMBDA = 0.01
    model = NonMinimalQuarticModel(xi=XI, lam=LAMBDA)
    model.S = 1.0

    # DEBUG: Smaller grid
    resolution = 25
    phi_vals = np.linspace(5.0, 6.0, resolution)
    pi_vals = np.linspace(-10.0, 0.0, resolution)
    
    results_N = np.zeros((resolution, resolution))
    results_eta = np.zeros((resolution, resolution))
    
    T_span = np.linspace(0, 1000000, 5000)
    
    print(f"Scanning {resolution}x{resolution} grid...")
    
    success_count = 0
    
    for i, phi0 in enumerate(phi_vals):
        for j, pi0 in enumerate(pi_vals):
            # Setup Model IC
            model.xi = phi0
            model.Ai = 1e-5
            
            # Velocity check
            if 3 - 0.5 * pi0**2 <= 0:
                results_N[i, j] = np.nan
                results_eta[i, j] = np.nan
                continue
                
            V_val = model.v0 * model.f(phi0)
            H = np.sqrt(V_val / (3 - 0.5 * pi0**2))
            model.yi = pi0 * H
            
            sol_data = run_background_simulation_safe(model, T_span)
            
            if sol_data is not None:
                try:
                    derived = get_derived_quantities(sol_data, model)
                    
                    # N_total
                    eps = derived['epsH']
                    end_mask = eps > 1.0
                    if np.any(end_mask):
                        idx_end = np.argmax(end_mask)
                        N_tot = derived['N'][idx_end]
                    else:
                        N_tot = derived['N'][-1]
                    
                    results_N[i, j] = N_tot
                    
                    # eta_min
                    N_evo = derived['N']
                    eta_evo = derived['etaH']
                    mask = (N_evo < 5) & (N_evo > 0)
                    if np.any(mask):
                        min_eta = np.min(eta_evo[mask])
                    else:
                        min_eta = 0.0
                        
                    results_eta[i, j] = min_eta
                    success_count += 1
                except Exception as e:
                    print(f"Error processing derived quantities: {e}")
                    results_N[i, j] = np.nan
                    results_eta[i, j] = np.nan
            else:
                results_N[i, j] = np.nan
                results_eta[i, j] = np.nan

    print(f"Scan complete. Successes: {success_count}/{resolution*resolution}")
    
    # Check if we have enough data to plot
    valid_mask = ~np.isnan(results_N)
    if not np.any(valid_mask):
        print("Error: No valid simulations found!")
        return

    # Plotting
    try:
        X, Y = np.meshgrid(pi_vals, phi_vals)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # N_tot
        # Handle all NaNs case?
        cp1 = ax1.contourf(X, Y, results_N, 20, cmap='viridis')
        plt.colorbar(cp1, ax=ax1, label='Total e-folds $N_{tot}$')
        
        # Contour levels might fail if range is 0 or NaN
        if np.nanmax(results_N) > 55:
            cnt1 = ax1.contour(X, Y, results_N, levels=[55, 60, 65], colors='r', linewidths=2)
            ax1.clabel(cnt1, fmt='%d')
            
        ax1.set_xlabel(r'Initial Velocity $\Pi_0$')
        ax1.set_ylabel(r'Initial Field $\phi_0$')
        ax1.set_title('Inflation Duration')

        # Eta_min
        cp2 = ax2.contourf(X, Y, results_eta, 20, cmap='coolwarm', vmin=-7, vmax=1)
        plt.colorbar(cp2, ax=ax2, label='Min $\eta_H$')
        
        if np.nanmin(results_eta) < -3:
            cnt2 = ax2.contour(X, Y, results_eta, levels=[-6, -3], colors='white', linestyles='--')
            ax2.clabel(cnt2, fmt='%.1f')
        
        # Overlay N=60
        if np.nanmax(results_N) > 60 and np.nanmin(results_N) < 60:
            cnt3 = ax2.contour(X, Y, results_N, levels=[60], colors='black', linewidths=2)
            ax2.clabel(cnt3, fmt='N=%d')
        
        ax2.set_xlabel(r'Initial Velocity $\Pi_0$')
        ax2.set_ylabel(r'Initial Field $\phi_0$')
        ax2.set_title('USR Intensity (Minimum $\eta$)')
        
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots/usr_scan_heatmap.png'))
        plt.savefig(output_path)
        print(f"Saved heatmap to {output_path}")
        
    except Exception as e:
        print(f"Plotting Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_scan_script()
