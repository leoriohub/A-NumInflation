
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def dphi_dpsi(psi, xi):
    """
    Returns d(phi)/d(psi) given psi and xi.
    """
    # dphi/dpsi = sqrt(1 + xi*psi^2 + 6*xi^2*psi^2) / (1 + xi*psi^2)
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

def plot_inflationary_regime():
    print("Plotting potential in inflationary regime (large field)...")
    
    # Parameters
    # We choose a range of xi. The plateau height is ~ lambda / (4 xi^2).
    # If we keep lambda fixed, the potentials will have vastly different heights.
    # To make them comparable or just to show the shape, we can plot V / V_plateau or just raw V.
    # Let's plot raw V to show the suppression by xi.
    
    xi_values = [10, 100, 1000] 
    lam = 0.01
    v_vev = 1.0e-16 # negligible here
    
    # We want to cover the transition region where xi*psi^2 ~ 1.
    # For xi=1000, psi ~ 0.03. For xi=10, psi ~ 0.3.
    # So psi up to 0.5 or 1.0 should cover the transition for all these.
    
    psi_mesh = np.linspace(0, 0.5, 200)
    
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'k']
    
    for idx, xi in enumerate(xi_values):
        phi_mesh = []
        v_e_mesh = []
        
        print(f"Calculating for xi = {xi}...")
        for psi in psi_mesh:
            phi = get_phi_from_psi(psi, xi)
            v_e = potential_einstein(psi, xi, lam, v_vev)
            
            phi_mesh.append(phi)
            v_e_mesh.append(v_e)
            
        label_str = f'$\\xi = {xi}$'
        plt.plot(phi_mesh, v_e_mesh, color=colors[idx % len(colors)], label=label_str, linewidth=2)
        
        # Calculate asymptotic plateau value for this xi
        # V_inf = lambda / (4 * xi^2)
        v_plateau = lam / (4 * xi**2)
        plt.axhline(y=v_plateau, color=colors[idx % len(colors)], linestyle='--', alpha=0.5)

    plt.xlabel(r'$\phi [M_P]$')
    plt.ylabel(r'$V(\phi) [M_P^4]$')
    plt.title(r'Higgs Inflation Potential in Einstein Frame')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # The y-values will be quite different (10^-6 vs 10^-10 etc). 
    # Log scale y?
    plt.yscale('log')
    plt.ylim(bottom=1e-12) # Cutoff to avoid -inf from log(0) at origin if we had exactly 0, but we start at 0
    # Actually at phi=0, V=0. Linear might be better if we want to see the shape starting from 0.
    # But the plateaus differ by orders of magnitude. 
    # Let's try log scale for Y to see all plateaus, but need to handle V=0.
    # Actually, for psi=0, V=constant (small). V_E ~ lambda/4 * v^4 ~ 10^-66.
    # Log scale will be dominated by the 10^-66 to 10^-8 range.
    # Let's stick to log scale but setting bottom limit reasonable, like 1e-15 or just auto.
    # Wait, the plateau for xi=1000 is 0.01 / (4 * 10^6) = 2.5e-9.
    # The plateau for xi=10 is 0.01 / (4 * 100) = 2.5e-5.
    # So log scale is necessary.
    
    output_file = "plots/nmc_potential_inflationary.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_inflationary_regime()
