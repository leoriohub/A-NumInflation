import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def dphi_dpsi(psi, xi):
    """
    Returns d(phi)/d(psi) given psi and xi.
    phi is in units of Mp, psi in units of Mp.
    """
    num = np.sqrt(1 + xi * psi**2 * (1 + 6 * xi))
    den = 1 + xi * psi**2
    return num / den

def get_phi_from_psi(psi_val, xi):
    """
    Integrates dphi/dpsi from 0 to psi_val to get phi(psi_val).
    """
    if psi_val == 0:
        return 0.0
    
    # Check for potentially very large xi causing numerical issues?
    # For very large xi, psi might be small if we keeping physical scale constant?
    # No, psi is just a field value.
    
    res, err = quad(dphi_dpsi, 0, psi_val, args=(xi,))
    return res

def scan_minima_vs_xi():
    print("Scanning minima localization vs xi...")
    
    # Range of xi: 0 to 10^30
    # Log space, but include 0 explicitly? 
    # Log scale for xi > 0.
    
    # Let's do logspace from 1e-2 to 1e30
    xi_log = np.logspace(-2, 30, 100)
    
    # Also include 0 and some small linear values if interesting?
    # The prompt asks for "Large regions... 0 to 10^30"
    # 0 is singular in log plot, handle separately or just start from small number.
    # We will stick to the log range for the main plot.
    
    xi_values = xi_log
    
    # Vacuum expectation value in Jordan frame
    # v_EW ~ 246 GeV. Mp ~ 2.435e18 GeV.
    # v ~ 1e-16 Mp.
    v_vev = 1.0e-16 # approximate EW scale in Mp
    
    phi_minima = []
    
    for xi in xi_values:
        phi_val = get_phi_from_psi(v_vev, xi)
        phi_minima.append(phi_val)
        
    phi_minima = np.array(phi_minima)
    
    # Metric limit check: for xi -> 0, phi -> psi. So phi_min -> v_vev.
    # For xi -> infinity:
    # dphi/dpsi ~ sqrt(6xi^2 psi^2) / (xi psi^2) = sqrt(6) xi psi / (xi psi^2) = sqrt(6)/psi ?
    # Wait. 
    # If xi*psi^2 >> 1:
    # num ~ sqrt(6 xi^2 psi^2) ~ sqrt(6) * xi * psi
    # den ~ xi * psi^2
    # ratio ~ sqrt(6) / psi
    # Integ(sqrt(6)/psi) dpsi -> sqrt(6) ln(psi).
    #
    # If xi*psi^2 << 1 (which is true for v=1e-16 and xi up to 10^30 is NOT always true):
    # If xi=10^30, xi*v^2 = 10^30 * 10^-32 = 0.01. Still small!
    # So actually for the EW scale, we might be in the small field regime even for huge xi?
    # Wait, 10^30 is HUGE.
    # v = 1e-16. v^2 = 1e-32.
    # xi * v^2 = 10^30 * 10^-32 = 0.01.
    # So even at xi=10^30, we are just barely approaching xi*psi^2 ~ 1.
    
    # Let's verify this numerically.
    
    plt.figure(figsize=(10, 6))
    plt.loglog(xi_values, phi_minima, 'b-', linewidth=2, label=r'$\phi_{min}$')
    
    # Reference line for v_vev
    plt.axhline(y=v_vev, color='k', linestyle='--', alpha=0.5, label=r'$v_{vev}$ (Jordan)')
    
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\phi_{min} [M_P]$')
    plt.title(r'Minima Localization $\phi_{min}$ vs $\xi$ (for $\psi_{min} = v_{EW}$)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_file = "minima_localization.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    scan_minima_vs_xi()
