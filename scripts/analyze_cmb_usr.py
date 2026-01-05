import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from inflation_models import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def detect_usr_duration(N, eta, threshold=-5.5):
    usr_mask = eta < threshold
    if not np.any(usr_mask):
        return 0.0
    indices = np.where(usr_mask)[0]
    return N[indices[-1]] - N[indices[0]]

phi_target = 5.42
y_vals = -np.logspace(-7, -4, 50)
durations = []

for y in y_vals:
    model = HiggsModel()
    model.xi = phi_target
    model.yi = y
    T_span = np.linspace(0, 1000, 3000)
    sol = run_background_simulation(model, T_span)
    data = get_derived_quantities(sol, model)
    eta_tot = 2 * (data['epsH'] - data['etaH'])
    durations.append(detect_usr_duration(data['N'], eta_tot))

max_dur = max(durations)
best_y = y_vals[durations.index(max_dur)]
print(f"Target phi = {phi_target}")
print(f"Max USR Duration = {max_dur:.4f}")
print(f"Critical velocity y_i = {best_y:.2e}")
