
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def dphi_dpsi(psi, xi):
    """
    Returns d(phi)/d(psi) given psi and xi.
    phi is in units of Mp, psi in units of Mp.
    """
    # dphi/dpsi = sqrt(1 + xi*psi^2 + 6*xi^2*psi^2) / (1 + xi*psi^2) ?
    # Let's double check formula.
    # Metric: (1 + xi psi^2)
    # Omega^2 = 1 + xi psi^2
    # dphi/dpsi = sqrt( (Omega^2 + 1.5 alpha^2 (dOmega^2/dpsi)^2) / Omega^4 )
    # alpha^2 = M_P^2 = 1 (in units)
    # But usually just:
    # dphi/dpsi = sqrt(1 + xi psi^2 (1 + 6xi)) / (1 + xi psi^2)
    
    num = np.sqrt(1 + xi * psi**2 * (1 + 6 * xi))
    den = 1 + xi * psi**2
    return num / den

def get_phi_from_psi(psi_val, xi):
    """
    Integrates dphi/dpsi from 0 to psi_val to get phi(psi_val).
    """
    if psi_val == 0:
        return 0.0
    res, err = quad(dphi_dpsi, 0, psi_val, args=(xi,))
    return res

def potential_jordan(psi, lam, v):
    """
    SSB Potential in Jordan frame: V(psi) = lambda/4 * (psi^2 - v^2)^2
    """
    return (lam / 4.0) * (psi**2 - v**2)**2

def potential_einstein(psi, xi, lam, v):
    """
    Potential in Einstein frame: V_E(phi) = V_J(psi) / Omega^4
    Omega^2 = 1 + xi * psi^2
    """
    v_j = potential_jordan(psi, lam, v)
    omega2 = 1 + xi * psi**2
    return v_j / (omega2**2)

def plot_minima_zoomed():
    print("Plotting potential minima for xi in 10^3 - 10^4 range...")
    
    # Parameters
    xi_values = [1000, 5000, 10000, 100000]
    lam = 0.01 # lambda
    # Calculate scalar field VEV in Planck units
    # v_EW approx 246 GeV
    # M_P (reduced) approx 2.435e18 GeV
    v_ew_gev = 246.0
    mp_gev = 2.435e18
    v_vev = v_ew_gev / mp_gev # approx 1.0102e-16
    
    # We want to plot V(phi) vs phi around phi ~ v_vev
    # Range of psi: 0 to 2*v_vev
    # Note: For xi up to 10^5 and psi ~ 1e-16, xi*psi^2 is extremely small (~10^-27).
    # This means the Einstein frame potential is almost identical to the Jordan frame potential,
    # and all curves will overlap visually. We use different line styles to distinguish them.
    psi_mesh = np.linspace(0, 2.0 * v_vev, 500)
    
    plt.figure(figsize=(10, 6))
    
    colors = ['r', 'g', 'b', 'y']
    
    for idx, xi in enumerate(xi_values):
        phi_mesh = []
        v_e_mesh = []
        
        # Calculate phi and V_E for each psi
        # Since xi*psi^2 is TINY for psi ~ 1e-16 and xi ~ 1e4 (1e4 * 1e-32 = 1e-28), 
        # phi approx psi, and Omega approx 1.
        # So we expect the potential to look exactly like the Jordan frame potential.
        
        for psi in psi_mesh:
            phi = get_phi_from_psi(psi, xi)
            v_e = potential_einstein(psi, xi, lam, v_vev)
            
            phi_mesh.append(phi)
            v_e_mesh.append(v_e)
            
        # Scientific notation label
        exponent = int(np.log10(xi))
        coeff = xi / 10**exponent
        if abs(coeff - 1.0) < 0.1:
            label_str = f'$\\xi = 10^{{{exponent}}}$'
        else:
            label_str = f'$\\xi = {coeff:.0f} \\times 10^{{{exponent}}}$'

        # Use different line styles/widths to make overlapping lines visible
        ls = ['-', '--', '-.', ':'][idx % 4]
        lw = [4, 3, 2, 1][idx % 4] + 1  # 5, 4, 3, 2
        
        plt.plot(phi_mesh, v_e_mesh, color=colors[idx], label=label_str, linestyle=ls, linewidth=lw, alpha=0.8)
        
        # Mark minimum
        # Min is at psi=v_vev
        phi_min = get_phi_from_psi(v_vev, xi)
        v_min = potential_einstein(v_vev, xi, lam, v_vev)
        plt.plot(phi_min, v_min, 'o', color=colors[idx])
        print(f"xi={xi:.1e}: phi_min ~ {phi_min:.4e}, V_min = {v_min:.4e}")

    plt.xlabel(r'$\phi [M_P]$')
    plt.ylabel(r'$V(\phi) [M_P^4]$')
    plt.title(r'Higgs Potential Minima in Einstein Frame ($\xi \sim 10^3-10^4$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scientific notation for axes
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    output_file = "plots/nmc_potential_minima_zoomed.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_minima_zoomed()
