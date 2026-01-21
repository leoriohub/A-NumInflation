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

# execution block
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # It is required to execute the script 'inf_dyn_background.py' and save the data in a text file before this script can be executed
    # We input the initial conditions and horizon exit for various scales from the background data
    # Please change the filename in this line if you have saved the data with a different name or at a different location
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/inf_bg_data.txt'))
    data = np.loadtxt(data_path) # row: T,N,Ne,x,y,z,aH,epsH,etaH,meff,Ps,Pt. 12 columns

    # This function returns the index of the row in the data file where the number of e-folds before the 
    #   end of inflation (Ne) attains a certain value specified by the argument 
    def i(Ne):
        return np.max(np.where(data[:,2]>=Ne))



#########################################################################################################
# The model of inflation is defined in this section
#########################################################################################################

# This term defines one unit of time 
S = 5e-5 


# parameters used in the potential function
M = 5.9e-6 
v0 = 0.5*M**2


# dimensionless potential function and its derivatives
def f(x):
    return x**2

def dfdx(x):
    return 2*x

def d2fdx2(x):
    return 2




#########################################################################################################
# In this section we set the initial conditions for both background as well as fluctuations
# The background data file is used to input initial conditions for background quantities
# We solve the dynamical equations, including the dimensionless Mukhanov-Sasaki equation using the
#   function scipy.integrate.odeint 
#########################################################################################################


### The dynamical variables are defined as follows:
#
# background:
#
# x : dimensionless field value [ \phi / m_p ]
# y : dimensionless field velocity [ dx/dT or \dot\phi / (m_p ^2 * S) ]
# A : dimensionless scale factor [ a * m_p * S ]
# z : dimensionless hubble parameter [ H / (S * m_p) ]
#
#
# fluctuations:
#
# v : real part of the Mukhanov-Sasaki variable [ v_k ] for scalar fluctuations [ \zeta_k ]
# u : imaginary part of the Mukhanov-Sasaki variable for scalar fluctuations
# h : real part of the Mukhanov-Sasaki variable for tensor fluctuations [ h_k ]
# g : imaginary part of the Mukhanov-Sasaki variable for tensor fluctuations




# def run_ms_simulation(model, T_span, k):  <-- Original Signature
def run_ms_simulation(xi, yi, zi, Ai, T_span, k, model):
    """
    Solves the Mukhanov-Sasaki equations for scalar and tensor fluctuations.
    """
    # xi, yi, zi, Ai = model.get_initial_conditions()  <-- Original IC loading
    v0 = model.v0
    S = model.S
    
    # Calculate initial conditions at correct scale if needed (omitted for brevity, assuming bunch-davies)
    # Re-using previous logic but generalized
    vi = (1/np.sqrt(2*k))
    ui = 0
    v_Ti = 0
    u_Ti = -k*(1/np.sqrt(2*k))/Ai
    hi = (1/np.sqrt(2*k))
    gi = 0
    h_Ti = 0
    g_Ti = -k*(1/np.sqrt(2*k))/Ai

    def sys(var, T):
        [x, y, z, A, v, v_T, u, u_T, h, h_T, g, g_T] = var
        # background
        dxdT = y
        dydT = -3*z*y - v0*model.dfdx(x)/S**2 
        dzdT = -0.5*y**2
        dAdT = A*z

        # scalar fluctuations
        dvdT = v_T
        dv_TdT = -z*v_T + v*(2.5*y**2 + 2*y*(-3*z*y - v0*model.dfdx(x)/S**2 )/z + 2*z**2 + 0.5*y**4/z**2 - v0*model.d2fdx2(x)/S**2 - k**2/A**2)
        dudT = u_T
        du_TdT = -z*u_T + u*(2.5*y**2 + 2*y*(-3*z*y - v0*model.dfdx(x)/S**2 )/z + 2*z**2 + 0.5*y**4/z**2 - v0*model.d2fdx2(x)/S**2 - k**2/A**2)
        
        # tensor fluctuations
        dhdT = h_T
        dh_TdT = -z*h_T - h*(k**2/A**2 - 2*z**2 + 0.5*y**2)
        dgdT = g_T
        dg_TdT = -z*g_T - g*(k**2/A**2 - 2*z**2 + 0.5*y**2)

        return [dxdT, dydT, dzdT, dAdT, dvdT, dv_TdT, dudT, du_TdT, dhdT, dh_TdT, dgdT, dg_TdT]

    sol = odeint(sys, [xi,yi,zi,Ai,vi,v_Ti,ui,u_Ti,hi,h_Ti,gi,g_Ti], T_span, rtol=1e-10, atol=1e-12, mxstep=1000000)
    return np.transpose(sol)

def get_ms_derived_quantities(sol_data, model, k):
    """
    Calculates power spectra for the simulated mode.
    """
    x, y, z, A, v, v_T, u, u_T, h, h_T, g, g_T = sol_data
    v0 = model.v0
    S = model.S
    
    with np.errstate(divide='ignore', invalid='ignore'):
        epsH = -(-z**2 + ((v0*model.f(x)/S**2 - y**2))/3)/z**2
        
        # Power spectra
        zeta2 = (v**2 + u**2)/(2*epsH*(A/S)**2)
        P_S = (k**3 * zeta2)/(2*np.pi**2)
        h2 = (h**2 + g**2)/((A/S)**2)
        P_T = 4*(k**3 * h2)/(np.pi**2)
    
    return {
        'P_S': P_S,
        'P_T': P_T,
        'aHk': (A*z)/k
    }

# execution block
if __name__ == "__main__":
    import os
    if not os.path.exists('data/inf_bg_data.txt'):
        print("Required background data file missing. Run inf_dyn_background.py first.")
    else:
        data = np.loadtxt('data/inf_bg_data.txt')
        def i_idx(Ne_val):
            return np.max(np.where(data[:,2]>=Ne_val))

        Nk = 60 
        k_val = data[i_idx(Nk), 6]
        
        # Initial conditions for this mode
        xi_ms = data[i_idx(Nk+5), 3]
        yi_ms = data[i_idx(Nk+5), 4]
        zi_ms = np.sqrt(yi_ms**2/6 + (v0*f(xi_ms)/(3*S**2)))
        Ai_ms = 1e-3 * np.exp(77.4859 - (Nk+5))
        
        T_span = np.linspace(0, 200, 10000)
        sol_data = run_ms_simulation(xi_ms, yi_ms, zi_ms, Ai_ms, T_span, k_val, v0, S)
        derived = get_ms_derived_quantities(sol_data, k_val, S)
        
        plt.plot(derived['aHk'], derived['P_S'], 'r')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

#########################################################################################################
#########################################################################################################
