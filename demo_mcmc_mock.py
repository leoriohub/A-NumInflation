"""
Mock MCMC corner plot: what a successful result would look like.

Simulates a Bayesian posterior over (phi0, yi, xi) showing that Planck data
favours ICs producing a USR-driven power suppression at low ell.

Run:  python3 demo_mcmc_mock.py
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ——— 1. Simulate mock posterior samples ———
# True "best-fit" parameters (the USR golden trajectory)
true_phi0, true_yi, true_xi = 5.75, -0.08, 15000.0

# Correlation structure (physically motivated)
#   phi0 and yi are anti-correlated: to get USR at same N_back,
#   larger |yi| needs smaller phi0 (steeper start to compensate)
#   xi is mostly independent but mildly correlated with phi0
mean = [true_phi0, true_yi, np.log10(true_xi)]
# Enforce positive semidefiniteness
s1, s2, s3 = 0.015, 0.012, 0.08
r12, r13, r23 = -0.5, 0.3, 0.0
cov = np.array([
    [s1**2,     r12*s1*s2,  r13*s1*s3],
    [r12*s1*s2, s2**2,      r23*s2*s3],
    [r13*s1*s3, r23*s2*s3,  s3**2],
])

# Generate samples
samples = np.random.multivariate_normal(mean, cov, size=5000)
phi0_s = samples[:, 0]
yi_s = samples[:, 1]
log_xi_s = samples[:, 2]
xi_s = 10**log_xi_s

# ——— 2. Map each sample → derived observables ———
# Mock the MS solve: use a simple analytic PS(k) that depends on parameters

def compute_ps(k, phi0, yi, xi):
    """Mock P_S(k) for a given set of ICs.

    The USR dip is modelled as a Gaussian suppression in log-log space,
    whose position and depth depend on phi0, yi.
    """
    # Baseline power-law
    ns = 0.965
    As = 2.1e-9
    ps_sr = As * (k / 0.05)**(ns - 1)

    # USR dip parameters — controlled by ICs
    dip_k0 = 10**(-4.5 + 30 * (phi0 - 5.7) + 20 * (yi + 0.08))  # position in k
    dip_depth = 0.7 * np.exp(-20 * (yi + 0.08)**2)             # depth (fractional)
    dip_width = 0.5

    dip = 1 - dip_depth * np.exp(-0.5 * (np.log10(k / dip_k0) / dip_width)**2)
    # Clamp to positive
    dip = np.maximum(dip, 0.05)
    return ps_sr * dip

def compute_chisq(phi0, yi, xi, N_modes=5):
    """Mock chi^2 comparing our PS(k) against 'Planck data'.

    The 'data' is generated from the true model with some noise.
    """
    k_obs = np.logspace(-4.5, 0.0, N_modes)

    # "Observed" data = true model
    ps_true = compute_ps(k_obs, true_phi0, true_yi, true_xi)

    # Model prediction
    ps_model = compute_ps(k_obs, phi0, yi, xi)

    # Mock error bars (increasing with k)
    sigma = 0.05 * ps_true * (1 + k_obs / 0.1)

    chisq = np.sum(((ps_model - ps_true) / sigma)**2)
    return chisq, ps_model, k_obs

# ——— 3. Posterior predictive: show PS(k) for sampled models ———
k_grid = np.logspace(-5, 0, 300)
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Plot 200 random posterior draws
ps_draws = np.array([compute_ps(k_grid, phi0_s[i], yi_s[i], xi_s[i])
                     for i in np.random.choice(len(phi0_s), 200)])
ps_median = np.median(ps_draws, axis=0)
ps_lo = np.percentile(ps_draws, 16, axis=0)
ps_hi = np.percentile(ps_draws, 84, axis=0)

# Baseline power-law (Planck best-fit LCDM)
ps_lcdm = 2.1e-9 * (k_grid / 0.05)**(0.965 - 1)

ax1.fill_between(k_grid, ps_lo, ps_hi, alpha=0.3, color='teal',
                  label=r'Posterior $P_S(k)$ (68% CI)')
ax1.plot(k_grid, ps_median, 'teal', lw=2, label='Median')
ax1.plot(k_grid, ps_lcdm, 'gray', lw=2, ls='--',
         label=r'Power-law $\Lambda$CDM')

# Highlight low-ell anomaly window
k_30 = 30 / 14000.0
ax1.axvspan(1e-5, k_30, color='purple', alpha=0.1,
            label=r'Low-$\ell$ anomaly ($\ell<30$)')
ax1.axvline(0.05, color='red', lw=1, ls=':', alpha=0.5,
            label=r'Pivot $k_*=0.05$ Mpc$^{-1}$')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=13)
ax1.set_ylabel(r'$\mathcal{P}_S(k)$', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, ls=':', alpha=0.4)
ax1.set_title('Posterior Predictive: Primordial Power Spectrum', fontsize=14)
import os
os.makedirs('images', exist_ok=True)
plt.tight_layout()
plt.savefig('images/mock_posterior_ps.png', dpi=150)
print("Saved images/mock_posterior_ps.png")

# ——— 4. Corner plot of parameter posteriors ———
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec

fig2 = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 3, hspace=0.08, wspace=0.08)
labels = [r'$\phi_0$', r'$y_i$', r'$\xi$']
params = [phi0_s, yi_s, xi_s]
ranges = [(5.70, 5.80), (-0.11, -0.05), (10000, 30000)]

# True values
true_vals = [true_phi0, true_yi, true_xi]

for i in range(3):
    for j in range(3):
        if i > j:
            # Scatter plot
            ax = fig2.add_subplot(gs[i, j])
            ax.scatter(params[j], params[i], s=2, alpha=0.3, c='teal')
            ax.scatter(true_vals[j], true_vals[i], s=80, marker='*',
                       c='red', zorder=5, label='True (injected)')
            ax.set_xlim(ranges[j])
            ax.set_ylim(ranges[i])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=13)
            if i == 2:
                ax.set_xlabel(labels[j], fontsize=13)
            # Hide tick labels on inner panels
            if i < 2:
                ax.tick_params(labelbottom=False)
            if j > 0:
                ax.tick_params(labelleft=False)
            ax.grid(True, alpha=0.3)

        elif i == j:
            # 1D marginal histogram
            ax = fig2.add_subplot(gs[i, j])
            ax.hist(params[i], bins=40, density=True, color='teal',
                    alpha=0.6, edgecolor='white')
            ax.axvline(true_vals[i], color='red', lw=2, ls='--')
            # 68% and 95% credible intervals
            lo68, hi68 = np.percentile(params[i], [16, 84])
            lo95, hi95 = np.percentile(params[i], [2.5, 97.5])
            ax.axvspan(lo68, hi68, alpha=0.15, color='teal')
            ax.set_xlim(ranges[i])
            if i == 2:
                ax.set_xlabel(labels[i], fontsize=13)
            if j > 0:
                ax.tick_params(labelleft=False)
            if i < 2:
                ax.tick_params(labelbottom=False)
            # Annotate CI on the diagonal
            ax.annotate(f'{lo68:.2f} - {hi68:.2f}',
                        xy=(0.95, 0.95), xycoords='axes fraction',
                        ha='right', va='top', fontsize=9, color='teal')
        else:
            # Hide empty panel
            ax = fig2.add_subplot(gs[i, j])
            ax.axis('off')

fig2.suptitle('Mock Posterior: USR Parameters from Planck Low-l Data',
              fontsize=14, y=0.98)
plt.savefig('images/mock_corner_plot.png', dpi=150)
print("Saved images/mock_corner_plot.png")

# ——— 5. Key results summary ———
print("\n" + "="*65)
print("MOCK RESULT: What a successful analysis would conclude")
print("="*65)
print(f"""
Posterior constraints (68% CI):
  phi0 = {np.mean(phi0_s):.3f}  +{np.std(phi0_s):.3f}  -{np.std(phi0_s):.3f}
  yi   = {np.mean(yi_s):.3f}  +{np.std(yi_s):.3f}  -{np.std(yi_s):.3f}
  xi   = {np.mean(xi_s):.0f}  +{np.std(xi_s):.0f}  -{np.std(xi_s):.0f}

Improvement over LCDM at low-ell (mock):
  Delta chi^2_{{(ell<30)}} = -8.3 ± 2.1   (lower is better)
  Delta chi^2_{{full}}      = -4.1 ± 3.0

Bayes factor (USR vs LCDM):
  ln B_{{USR/LCDM}} = 1.8   (positive evidence for USR model)

Interpretation:
  Planck low-ell data favours phi0 ~ 5.75, yi ~ -0.08, xi ~ 15000,
  which produce a ~30% suppression in P_S(k) at k ~ 2e-4 Mpc^-1,
  corresponding to ell ~ 3. This aligns with the observed power deficit
  at the lowest CMB multipoles.
""")

# ——— 6. C_ell plot (using SW approximation) ———
from scipy.special import spherical_jn

def sachs_wolfe_cl(ell, ps_interp, k_grid, r_ls=14000.0):
    """C_ell ~ (2pi^2/9) int (dk/k) P_S(k) j_ell^2(k r_ls)"""
    integrand = ps_interp(k_grid) * spherical_jn(ell, k_grid * r_ls)**2
    return (2 * np.pi**2 / 9) * np.trapezoid(integrand / k_grid, k_grid)

ells = np.arange(2, 35)
cl_usr = []
cl_lcdm = []

for ell in ells:
    from scipy.interpolate import interp1d
    ps_med_interp = interp1d(k_grid, ps_median, kind='cubic',
                              bounds_error=False, fill_value=0)
    ps_lcdm_interp = interp1d(k_grid, ps_lcdm, kind='cubic',
                               bounds_error=False, fill_value=0)
    cl_usr.append(sachs_wolfe_cl(int(ell), ps_med_interp, k_grid))
    cl_lcdm.append(sachs_wolfe_cl(int(ell), ps_lcdm_interp, k_grid))

cl_usr = np.array(cl_usr)
cl_lcdm = np.array(cl_lcdm)

# Normalize to D_ell = ell(ell+1) C_ell / (2pi)
d_usr = ells * (ells + 1) * cl_usr / (2 * np.pi)
d_lcdm = ells * (ells + 1) * cl_lcdm / (2 * np.pi)

fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.plot(ells, d_usr, 'teal', lw=2.5, label='USR model (posterior median)')
ax3.plot(ells, d_lcdm, 'gray', lw=2, ls='--',
         label=r'Power-law $\Lambda$CDM')
# Mock "Planck data" with errors
np.random.seed(123)
mock_cl_data = d_lcdm * (1 + 0.08 * np.random.randn(len(ells)))
mock_cl_data[:5] = mock_cl_data[:5] * 0.7  # suppress low ell to mimic anomaly
mock_err = 0.08 * d_lcdm
ax3.errorbar(ells, mock_cl_data, yerr=mock_err, fmt='o', color='k',
             capsize=3, markersize=4, label='Mock Planck data')
ax3.set_xlabel(r'Multipole $\ell$', fontsize=13)
ax3.set_ylabel(r'$D_\ell^{TT} = \ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]',
               fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_title('CMB Angular Power Spectrum (Sachs-Wolfe approx)', fontsize=13)
plt.tight_layout()
plt.savefig('images/mock_cl_plot.png', dpi=150)
print("Saved images/mock_cl_plot.png")
print("\nAll figures saved to images/")
