
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import minimize, root_scalar

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inflation_models import NonMinimalQuarticModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def get_N_total(phi0, pi0, model, T_span, return_all=False):
    # Setup IC
    model.xi = phi0
    model.Ai = 1e-5
    
    if 3 - 0.5 * pi0**2 <= 0:
        return np.nan
        
    V_val = model.v0 * model.f(phi0)
    H = np.sqrt(V_val / (3 - 0.5 * pi0**2))
    model.yi = pi0 * H
    
    try:
        sol_data = run_background_simulation(model, T_span)
        derived = get_derived_quantities(sol_data, model)
        
        eps = derived['epsH']
        idx_end = np.argmax(eps > 1.0)
        
        if idx_end > 0:
            N_tot = derived['N'][idx_end]
        else:
            if eps[-1] < 1.0:
                 N_tot = derived['N'][-1]
            else:
                 N_tot = 0
        
        # Check for USR
        eta = derived['etaH']
        mask = derived['N'] < 5
        min_eta = np.min(eta[mask]) if np.any(mask) else 0
        
        if return_all:
            return N_tot, min_eta, sol_data, derived
        return N_tot
    except:
        return np.nan

def find_golden_point():
    print("Searching for Golden Point (N=60, USR)...")
    
    XI = 1000.0
    LAMBDA = 0.01
    model = NonMinimalQuarticModel(xi=XI, lam=LAMBDA)
    model.S = 1.0
    T_span = np.linspace(0, 100000000, 10000)
    
    # We want to find phi0 such that N_tot(phi0, pi0) = 60
    # for a fix pi0.
    # We want deep USR, so let's pick a few Pi values and solve for phi.
    # Constraint: Pi < sqrt(6) approx 2.45
    test_pis = [-2.43, -2.4, -2.0, -1.0, -0.5]
    
    best_candidate = None
    best_eta = 0
    
    for pi_target in test_pis:
        print(f"Scanning Pi = {pi_target}...")
        
        def objective(phi):
            N = get_N_total(phi, pi_target, model, T_span)
            if np.isnan(N): return 1000.0
            return N - 60.0 # Solve for 0
            
        # Bracket search?
        # From heatmap we know phi is in [5.0, 6.0]
        # Debug Brackets
        val_lower = objective(5.0)
        val_upper = objective(10.0)
        print(f"  Debug: N(5.0)-60 = {val_lower:.2f}, N(10.0)-60 = {val_upper:.2f}")

        try:
            res = root_scalar(objective, bracket=[5.0, 10.0], method='brentq')
            if res.converged:
                phi_sol = res.root
                # Check quality
                N, min_eta, _, _ = get_N_total(phi_sol, pi_target, model, T_span, return_all=True)
                print(f"  Found root: phi={phi_sol:.5f}, Pi={pi_target} => N={N:.2f}, min_eta={min_eta:.2f}")
                
                if min_eta < best_eta:
                    best_eta = min_eta
                    best_candidate = (phi_sol, pi_target, N, min_eta)
        except Exception as e:
            print(f"  Failed for Pi={pi_target}: {e}")
            
    if best_candidate:
        print("\nBEST CANDIDATE FOUND:")
        print(f"Phi0 = {best_candidate[0]:.6f}")
        print(f"Pi0  = {best_candidate[1]:.6f}")
        print(f"Ntot = {best_candidate[2]:.4f}")
        print(f"EtaMin = {best_candidate[3]:.4f}")
        
        # Save to file for next script to read? Or just print
        with open('outputs/best_candidate.txt', 'w') as f:
            f.write(f"{best_candidate[0]}\n{best_candidate[1]}")
    else:
        print("No solution found matching criteria.")

if __name__ == "__main__":
    find_golden_point()
