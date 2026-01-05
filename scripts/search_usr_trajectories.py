import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.getcwd())
from inflation_models import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

phi_target = 5.42
y_vals = [-0.1, -0.15, -0.2, -0.3]

plt.figure(figsize=(12, 8))

for y in y_vals:
    model = HiggsModel()
    model.xi = phi_target
    model.yi = y
    T_span = np.linspace(0, 500, 3000)
    sol = run_background_simulation(model, T_span)
    data = get_derived_quantities(sol, model)
    
    # Filter for active inflation
    mask = (data['epsH'] < 0.9)
    if not np.any(mask): continue
    
    N_inf = data['N'][mask]
    etaH_inf = data['etaH'][mask]
    
    # We are interested in the early part where USR might be triggered
    if len(etaH_inf) > 10:
        max_val = np.max(etaH_inf[:200]) # Look at the first ~200 steps
        print(f"yi: {y:.1e} | Early Max etaH: {max_val:.4f}")
    
    plt.plot(N_inf, etaH_inf, label=f'yi = {y:.1e}')

plt.axhline(3, color='k', ls='--', label='USR Limit')
plt.ylim(-1, 5)
plt.xlim(0, 10)
plt.xlabel('N')
plt.ylabel('etaH (Friction)')
plt.title(f"Searching for USR Plateau at phi_i = {phi_target}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("search_usr_trajectories.png")
print("Plot saved to search_usr_trajectories.png")
