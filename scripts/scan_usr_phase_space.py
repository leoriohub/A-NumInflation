import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from inflation_models import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def detect_usr_duration(N, eta, threshold=-5.5):
    usr_mask = eta < threshold
    if not np.any(usr_mask):
        return 0.0
    indices = np.where(usr_mask)[0]
    return N[indices[-1]] - N[indices[0]]

# Grid setup
phi_vals = np.linspace(5.0, 6.0, 40)
y_vals = -np.logspace(-6, -3, 40)

duration_grid = np.zeros((len(y_vals), len(phi_vals)))

print(f"Starting scan over {len(phi_vals) * len(y_vals)} points...")

for i, y in enumerate(y_vals):
    for j, phi in enumerate(phi_vals):
        model = HiggsModel()
        model.xi = phi
        model.yi = y
        
        # Run short simulation
        T_span = np.linspace(0, 1000, 2000)
        sol = run_background_simulation(model, T_span)
        data = get_derived_quantities(sol, model)
        
        eta_tot = 2 * (data['epsH'] - data['etaH'])
        duration_grid[i, j] = detect_usr_duration(data['N'], eta_tot)

# Plotting
plt.figure(figsize=(10, 8))
X, Y = np.meshgrid(phi_vals, np.abs(y_vals))
cp = plt.pcolormesh(X, Y, duration_grid, cmap='viridis', shading='auto')
plt.colorbar(cp, label='USR Duration (e-folds)')
plt.yscale('log')
plt.xlabel('Initial Field Value $\\phi_i$')
plt.ylabel('Initial Velocity $|y_i|$')
plt.title('USR Duration Phase Space (Fine-Tuning Scan)')

plot_path = os.path.join(os.getcwd(), 'usr_phase_space_scan.png')
plt.savefig(plot_path)
print(f"Scan complete. Plot saved to {plot_path}")

# Identify best points
max_dur = np.max(duration_grid)
if max_dur > 0:
    idx = np.unravel_index(np.argmax(duration_grid), duration_grid.shape)
    print(f"Best Fine-Tuning: phi_i = {phi_vals[idx[1]]:.3f}, yi = {y_vals[idx[0]]:.2e}, Duration = {max_dur:.3f}")
