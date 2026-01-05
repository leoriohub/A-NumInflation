
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inflation_models import NonMinimalQuarticModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def debug_run():
    XI = 1000.0
    LAMBDA = 0.01
    model = NonMinimalQuarticModel(xi=XI, lam=LAMBDA)
    model.S = 1.0

    phi0 = 25.0
    pi0 = -0.5

    # Setup IC
    model.xi = phi0
    model.Ai = 1e-5
    
    psi_val = model._get_psi(phi0)
    f_val = model.f(phi0)
    V_val = model.v0 * f_val
    
    print(f"DEBUG: phi0={phi0}")
    print(f"DEBUG: psi={psi_val}")
    print(f"DEBUG: f(phi)={f_val} (Expect ~ 10^-6)")
    print(f"DEBUG: v0={model.v0}")
    print(f"DEBUG: V_val={V_val} (Expect ~ 10^-9)")
    
    H = np.sqrt(V_val / (3 - 0.5 * pi0**2))
    model.yi = pi0 * H
    
    print(f"Running debug simulation for phi={phi0}, Pi={pi0}")
    print(f"Initial H = {H}")
    print(f"Initial yi = {model.yi}")
    
    T_span = np.linspace(0, 2000000, 10000)
    
    try:
        sol_data = run_background_simulation(model, T_span)
        derived = get_derived_quantities(sol_data, model)
        
        N = derived['N']
        eps = derived['epsH']
        phi = sol_data[0]
        
        print(f"Simulation finished with {len(N)} steps.")
        print(f"Final N = {N[-1]}")
        print(f"Final phi = {phi[-1]}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(N, phi)
        ax1.set_ylabel('phi')
        ax1.set_title(f'Trajectory phi={phi0}, Pi={pi0}')
        ax1.grid(True)
        
        ax2.plot(N, eps)
        ax2.set_ylabel('epsilon')
        ax2.set_xlabel('N')
        ax2.set_yscale('log')
        ax2.axhline(1.0, color='r', linestyle='--')
        ax2.grid(True)
        
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots/debug_run.png'))
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
        
    except Exception as e:
        print(f"Simulation failed: {e}")

if __name__ == "__main__":
    debug_run()
