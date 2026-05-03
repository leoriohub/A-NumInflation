#!/usr/bin/env python3
"""Compare HiggsModel (high-field approx) vs FullHiggsModel (exact) across field range.

Quantifies where the approximation breaks for f(x), dfdx(x), d2fdx2(x),
and overlays typical USR trajectories to see if the integration window
ever reaches the breaking region.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.higgs import HiggsModel, FullHiggsModel

xi = 15000
lam = 0.13

approx = HiggsModel(lam=lam, xi=xi)
exact = FullHiggsModel(lam=lam, xi=xi)

# Field range to compare
x_range = np.linspace(0.1, 7.0, 1000)

f_approx = approx.f(x_range)
f_exact = exact.f(x_range)
df_approx = approx.dfdx(x_range)
df_exact = exact.dfdx(x_range)
d2f_approx = approx.d2fdx2(x_range)
d2f_exact = exact.d2fdx2(x_range)

def rel_err(approx_val, exact_val):
    denom = np.abs(exact_val)
    mask = denom > 1e-30
    err = np.zeros_like(approx_val)
    err[mask] = np.abs(approx_val[mask] - exact_val[mask]) / denom[mask]
    err[~mask] = np.abs(approx_val[~mask] - exact_val[~mask])
    return err

err_f = rel_err(f_approx, f_exact)
err_df = rel_err(df_approx, df_exact)
err_d2f = rel_err(d2f_approx, d2f_exact)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Row 1: f(x)
axes[0, 0].plot(x_range, f_approx, 'b--', lw=1.5, label='HiggsModel (approx)')
axes[0, 0].plot(x_range, f_exact, 'r-', lw=1, label='FullHiggsModel (exact)')
axes[0, 0].set_xlabel('x (Einstein frame)')
axes[0, 0].set_ylabel('f(x)')
axes[0, 0].legend()
axes[0, 0].set_title('Potential f(x)')
axes[0, 0].set_xlim(0, 7)

axes[0, 1].semilogy(x_range, err_f, 'k-', lw=1.5)
axes[0, 1].axhline(1e-6, color='gray', ls=':', label='1e-6')
axes[0, 1].axhline(1e-3, color='gray', ls=':', label='1e-3')
axes[0, 1].axhline(1e-2, color='gray', ls=':', label='1e-2')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Relative error')
axes[0, 1].set_title('f(x) relative error')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 7)

# Row 2: dfdx
axes[1, 0].plot(x_range, df_approx, 'b--', lw=1.5, label='Approx')
axes[1, 0].plot(x_range, df_exact, 'r-', lw=1, label='Exact')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel("f'(x)")
axes[1, 0].set_title('First derivative')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 7)

axes[1, 1].semilogy(x_range, err_df, 'k-', lw=1.5)
axes[1, 1].axhline(1e-6, color='gray', ls=':')
axes[1, 1].axhline(1e-3, color='gray', ls=':')
axes[1, 1].axhline(1e-2, color='gray', ls=':')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Relative error')
axes[1, 1].set_title("f'(x) relative error")
axes[1, 1].set_xlim(0, 7)

# Row 3: d2fdx2
axes[2, 0].plot(x_range, d2f_approx, 'b--', lw=1.5, label='Approx')
axes[2, 0].plot(x_range, d2f_exact, 'r-', lw=1, label='Exact')
axes[2, 0].axhline(0, color='gray', ls='-', lw=0.5)
axes[2, 0].set_xlabel('x')
axes[2, 0].set_ylabel("f''(x)")
axes[2, 0].set_title('Second derivative (enters MS effective mass)')
axes[2, 0].legend()
axes[2, 0].set_xlim(0, 7)

axes[2, 1].semilogy(x_range, err_d2f, 'k-', lw=1.5)
axes[2, 1].axhline(1e-6, color='gray', ls=':')
axes[2, 1].axhline(1e-3, color='gray', ls=':')
axes[2, 1].axhline(1e-2, color='gray', ls=':')
axes[2, 1].set_xlabel('x')
axes[2, 1].set_ylabel('Relative error')
axes[2, 1].set_title("f''(x) relative error")
axes[2, 1].set_xlim(0, 7)

plt.tight_layout()
plt.savefig('images/approx_vs_exact_comparison.png', dpi=200)
print("Saved images/approx_vs_exact_comparison.png")

# Print key thresholds
for threshold in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
    mask = err_d2f > threshold
    if mask.any():
        x_break = x_range[mask][0]
        print(f"f''(x) relative error > {threshold:6g}  for x < {x_break:.3f}")

# Now check: where does epsH=1 occur for typical configs?
print("\n--- End of inflation (epsH=1) field values for typical configs ---")
from inf_dyn_background import run_background_simulation, get_derived_quantities

configs = [
    (5.8, -0.01, "standard SR"),
    (5.6, -0.05, "mild USR"),
    (5.5, -0.1, "moderate USR"),
    (5.4, -0.2, "strong USR"),
]

T_span = np.linspace(0, 5000, 10000)

for phi0, yi, label in configs:
    model = HiggsModel(lam=lam, xi=xi)
    model.phi0 = phi0
    model.yi = yi
    
    bg = run_background_simulation(model, T_span)
    derived = get_derived_quantities(bg, model)
    
    epsH = derived['epsH']
    x_traj = bg[0]
    
    # Find epsH=1 crossing
    in_inflation = False
    end_idx = -1
    for i in range(len(epsH)):
        if not in_inflation:
            if epsH[i] < 1.0:
                in_inflation = True
        else:
            if epsH[i] >= 1.0:
                end_idx = i
                break
    
    if end_idx > 0:
        x_end = x_traj[end_idx]
        x_start = x_traj[0]
        x_min = x_traj[:end_idx].min()
        print(f"  {label:20s}  phi0={phi0:.2f}  yi={yi:6.3f}  "
              f"x_start={x_start:.3f}  x_end(epsH=1)={x_end:.3f}  "
              f"x_min={x_min:.3f}")
    else:
        print(f"  {label:20s}  phi0={phi0:.2f}  yi={yi:6.3f}  inflation did not end")
