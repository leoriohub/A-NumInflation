
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inflation_models import NonMinimalQuarticModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def find_cmb_horizon_exit_using_existing_solver(xi=1000.0):
    print(f"Finding CMB Horizon Exit (N=60) for xi = {xi}")
    print("Using existing solvers from 'inflation_models.py' and 'inf_dyn_background.py'...")

    model = NonMinimalQuarticModel(xi=xi, lam=0.01)

    model.S = 1.0 
    
    # 2. Define Initial Conditions Grid
    # We want to find which phi_start gives N_total > 60.
    # The model._get_dpsi_dphi() is available.
    
    # User requested 15-20, but that yields >10^7 e-folds.
    # We adjust to 5.0-6.0 to find the N=60 point (approx 5.4).
    phi_starts = np.linspace(5.0, 10.0, 50)
    
    results = []
    
    for phi0 in phi_starts:
        # Set IC manually in the model object
        model.xi = phi0
        model.yi = 0.0 # start at rest
        model.Ai = 1e-5 # This is used by get_initial_conditions to compute Ni

        
        # Run simulation
        # T_span needs to be long enough for inflation to end
        # T in these units is ~ N / H? 
        # Actually inf_dyn_background uses 't' but integrates H equations?
        # Let's check equations:
        # dxdT = y  => dx/dt = y
        # dydT = ... => dy/dt = ...
        # dzdT = -0.5*y**2 => dz/dt = -epsilon*z
        # dAdT = A*z => dA/dt = A*z => dA/A = z dt => dN = z dt => z = H ?
        # Yes, z seems to be H.
        # dzdT = dH/dt = -0.5 * y^2 = -0.5 * dot_phi^2
        # (Using Mp=1).
        # Standard: dot_H = -0.5 dot_phi^2 / Mp^2. Matches if (S=1).
        # So T is indeed cosmic time t.
        # N = int H dt = int z dt.
        
        T_span = np.linspace(0, 100000000, 100000) # Increased duration for small H
        
        try:
            sol_data = run_background_simulation(model, T_span)
            derived = get_derived_quantities(sol_data, model)
            
            N_arr = derived['N'][-1]
            phi_arr = sol_data[0]
            
            # Check if inflation ended (epsilon reached 1)
            eps = derived['epsH']
            
            # Find index where eps crosses 1
            idx_end = np.argmax(eps > 1.0)
            if idx_end == 0 and eps[0] < 1.0:
                # Never crossed 1 within T_span or array limit
                N_total = N_arr
            elif idx_end > 0:
                N_total = derived['N'][idx_end]
            else:
                N_total = 0 # Started with eps > 1?
            
            print(f"  IC phi={phi0:.2f} => N_total={N_total:.2f}")
            
            if N_total > 60.5:
                # Find N = N_total - 60
                N_target = N_total - 60.0
                
                # Interpolate to find phi at N_target
                phi_60 = np.interp(N_target, derived['N'], phi_arr)
                results.append((phi0, N_total, phi_60))
                
        except Exception as e:
            print(f"  Simulation failed for phi={phi0}: {e}")
            
    # Summary
    if results:
        print("\n--- Results (Existing Solver) ---")
        for res in results:
            print(f"Start={res[0]:.2f}, N_tot={res[1]:.1f} => Phi(N=60) = {res[2]:.5f}")
            
        # Plot
        phi_60_vals = [r[2] for r in results]
        avg_phi = np.mean(phi_60_vals)
        print(f"\nAverage Phi at N=60: {avg_phi:.5f} Mp")
    else:
        print("No successful trajectories > 60 e-folds found.")

if __name__ == "__main__":
    find_cmb_horizon_exit_using_existing_solver(xi=1000)
