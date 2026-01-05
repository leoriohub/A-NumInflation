#########################################################################################################
#########################################################################################################
#
# Please refer to <arXiv link> for explaination of variables and instructions for using the code
#
#########################################################################################################
#########################################################################################################

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



#########################################################################################################
# The model of inflation is defined in this section
#########################################################################################################

from inflation_models import QuadraticModel

def run_background_simulation(model, T_span):
    """
    Solves the background inflationary equations for a given model.
    """
    phi0, yi, zi, Ni = model.get_initial_conditions()
    v0 = model.v0
    S = model.S

    def sys(var, T):
        [x, y, z, n] = var
        dxdT = y
        dydT = -3*z*y - v0*model.dfdx(x)/S**2 
        dzdT = -0.5*y**2
        dndT = z # d(ln A)/dT = H = z
        return [dxdT, dydT, dzdT, dndT]

    # Using tighter tolerances for general stability
    sol = odeint(sys, [phi0, yi, zi, Ni], T_span, rtol=1e-10, atol=1e-12, mxstep=1000000)
    return np.transpose(sol)

def get_derived_quantities(sol_data, model):
    """
    Calculates physical quantities from simulation results.
    """
    x, y, z, n = sol_data
    v0 = model.v0
    S = model.S
    Ni = model.get_initial_conditions()[3] # Get Ni

    N = n - Ni
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Slow-roll parameters
        # Exact dynamical parameters
        epsH = -(-z**2 + ((v0*model.f(x)/S**2 - y**2))/3)/z**2
        etaH = -(-3*z*y - v0*model.dfdx(x)/S**2)/(y*z)
        
        #Slow roll approximations
        # Observables
        ns = 1 + 2*etaH - 4*epsH
        r = 16*epsH
        Ps = (S*z)**2 / (8 * np.pi**2 * epsH)
        Pt = 2*(S*z)**2 / (np.pi**2)
    
    # Scale Mapping
    # aH = A*z = exp(n)*z.  Be careful with exp(n) if n is large.
    # Usually we don't need aH explicitly as a float if it's huge.
    # But if returned, it might overflow.
    # Let's return log_aH = n + log(z)
    
    return {
        'N': N,
        'epsH': epsH,
        'etaH': etaH,
        'ns': ns,
        'r': r,
        'Ps': Ps,
        'Pt': Pt,
        'n': n # Return log scale factor
    }
    
    return {
        'N': N,
        'epsH': epsH,
        'etaH': etaH,
        'ns': ns,
        'r': r,
        'Ps': Ps,
        'Pt': Pt,
        'aH': aH
    }

# execution block
if __name__ == "__main__":
    from inf_dyn_plot import set_style, plot_background_n_vs_t, plot_eps_h
    set_style()

    T_span = np.linspace(0, 1000, 100000)
    
    # Default to Quadratic Model for legacy support
    model = QuadraticModel()
    sol_data = run_background_simulation(model, T_span)
    derived = get_derived_quantities(sol_data, model)
    
    plot_background_n_vs_t(T_span, derived['N'])
    plot_eps_h(derived['N'], derived['epsH'])

#########################################################################################################
#########################################################################################################
