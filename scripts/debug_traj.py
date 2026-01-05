import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.getcwd())
from inflation_models import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

phi_target = 5.42
y_val = -1e-4

model = HiggsModel()
model.xi = phi_target
model.yi = y_val
T_span = np.linspace(0, 500, 5000)
sol = run_background_simulation(model, T_span)
data = get_derived_quantities(sol, model)

N = data['N']
eps = data['epsH']
etaH = data['etaH']
eta_tot = 2 * (eps - etaH)

plt.figure(figsize=(10, 6))
plt.plot(N, etaH, label='etaH (Friction)')
plt.plot(N, eta_tot, label='eta_tot (SR)')
plt.axhline(3, color='k', ls='--')
plt.axhline(-6, color='k', ls='--')
plt.ylim(-10, 10)
plt.title(f"Trajectory for yi = {y_val}")
plt.legend()
plt.savefig("debug_traj.png")
print("Max etaH:", np.max(etaH))
print("Min eta_tot:", np.min(eta_tot))
