
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import root_scalar

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inflation_models import NonMinimalQuarticModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def get_N_total(phi0, pi0, model, T_span, return_details=False):
    # Setup IC
    model.xi = phi0
    model.Ai = 1e-5
    
    # Check kinetic dominance
    if 3 - 0.5 * pi0**2 <= 0:
        return (np.nan, 0) if return_details else np.nan
        
    V_val = model.v0 * model.f(phi0)
    H = np.sqrt(V_val / (3 - 0.5 * pi0**2))
    model.yi = pi0 * H
    
    try:
        sol_data = run_background_simulation(model, T_span)
        derived = get_derived_quantities(sol_data, model)
        
        eps = derived['epsH']
        # Robust end detection
        end_mask = eps > 1.0
        if np.any(end_mask):
            idx_end = np.argmax(end_mask)
            N_tot = derived['N'][idx_end]
        else:
             N_tot = derived['N'][-1]
             
        if return_details:
            # Calculate min eta
            eta = derived['etaH']
            N_arr = derived['N']
            # Look for dip in first 5 e-folds
            mask = (N_arr < 5) & (N_arr > 0)
            if np.any(mask):
                min_eta = np.min(eta[mask])
            else:
                min_eta = 0.0
            return N_tot, min_eta
            
        return N_tot
    except:
        return (np.nan, 0) if return_details else np.nan

def search_trajectory():
    print("=== Automated USR Trajectory Search ===")
    
    # Model Setup
    XI = 1000.0
    LAMBDA = 0.01
    model = NonMinimalQuarticModel(xi=XI, lam=LAMBDA)
    model.S = 1.0
    
    # Increased time span for high pi cases
    T_span = np.linspace(0, 100000000, 10000)

    # 1. Outer Loop: Scan Velocities
    # We scan from 0 down to near the limit -sqrt(6) ~ -2.449
    # Use log spacing near the limit? Or just fine linear.
    pi_scan = np.linspace(-0.1, -2.4, 30)
    
    results = []
    
    print(f"Scanning {len(pi_scan)} velocity values...")
    
    for pi_target in pi_scan:
        
        # 2. Inner Loop: Find phi0 for N=60
        def objective(phi):
            N = get_N_total(phi, pi_target, model, T_span)
            if np.isnan(N): return 1000.0 # Penalty
            return N - 60.0
            
        try:
            # We assume phi is in [5.0, 30.0]. 
            # High Pi requires larger phi to compensate and get same N.
            res = root_scalar(objective, bracket=[5.0, 30.0], method='brentq')
            
            if res.converged:
                phi_sol = res.root
                
                # Measure properties
                N_final, eta_min = get_N_total(phi_sol, pi_target, model, T_span, return_details=True)
                
                print(f"  [FOUND] Pi={pi_target:.3f} -> Phi={phi_sol:.5f} | N={N_final:.2f}, EtaMin={eta_min:.4f}")
                results.append({
                    'pi': pi_target,
                    'phi': phi_sol,
                    'N': N_final,
                    'eta_min': eta_min
                })
            else:
                print(f"  [FAIL] Pi={pi_target:.3f} (Convergence failed)")
                
        except Exception as e:
            # Likely bracket error if solution moves out of range
            print(f"  [FAIL] Pi={pi_target:.3f} (Bracket error: {str(e)[:50]}...)")

    if not results:
        print("No valid trajectories found.")
        return

    # 3. Analyze Results
    pis = [r['pi'] for r in results]
    etas = [r['eta_min'] for r in results]
    
    # Find best USR candidate (minimum eta)
    best_idx = np.argmin(etas)
    best = results[best_idx]
    
    print("\n=== SEARCH RESULTS ===")
    print(f"Best USR Candidate:")
    print(f"  Initial Pi  = {best['pi']:.6f}")
    print(f"  Initial Phi = {best['phi']:.6f}")
    print(f"  Min Eta     = {best['eta_min']:.4f}")
    
    # Save best to file
    with open('outputs/best_usr_candidate.txt', 'w') as f:
        f.write(f"{best['phi']}\n{best['pi']}")

    # Plot Eta vs Pi
    plt.figure(figsize=(10, 6))
    plt.plot(pis, etas, 'b-o')
    plt.axhline(-6.0, color='r', linestyle='--', label='USR Target')
    plt.xlabel(r'Initial Velocity $\Pi_0$')
    plt.ylabel(r'Minimum $\eta_H$')
    plt.title('USR Depth vs Initial Velocity (for N=60 trajectories)')
    plt.grid(True)
    plt.legend()
    
    output_plot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots/usr_search_results.png'))
    plt.savefig(output_plot)
    print(f"Saved results plot to {output_plot}")

if __name__ == "__main__":
    search_trajectory()
