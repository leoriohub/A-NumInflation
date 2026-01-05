import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from inflation_models import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

def get_max_etaH(N, etaH):
    # We want to see if etaH reaches 3
    return np.max(etaH)

phi_target = 5.42
y_vals = -np.logspace(-6, -3, 50)
results = []

for y in y_vals:
    model = HiggsModel()
    model.xi = phi_target
    model.yi = y
    T_span = np.linspace(0, 1000, 3000)
    sol = run_background_simulation(model, T_span)
    data = get_derived_quantities(sol, model)
    
    # Analyze only the first part of the simulation to avoid oscillations after inflation
    # Find where epsilon < 0.1 (good inflation)
    mask = data['epsH'] < 0.1
    if np.any(mask):
        max_eH = np.max(data['etaH'][mask])
        min_eta = np.min(2 * (data['epsH'][mask] - data['etaH'][mask]))
        results.append((y, max_eH, min_eta))

print(f"{'yi':<10} | {'max(etaH)':<10} | {'min(eta_tot)':<12}")
print("-" * 40)
for y, max_h, min_tot in results:
    if max_h > 1.5:
        print(f"{y:.2e} | {max_h:10.4f} | {min_tot:12.4f}")
