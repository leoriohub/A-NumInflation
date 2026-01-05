#########################################################################################################
#########################################################################################################
#
# Inflation Models Module
#
# This module defines the base Model class and specific inflationary potentials.
# It encapsulates the potential V(phi) and its derivatives, as well as the initial conditions.
#
#########################################################################################################
#########################################################################################################

import numpy as np

class InflationModel:
    """Base class for inflationary models."""
    def __init__(self, name, S=5e-5):
        self.name = name
        self.S = S
        self.v0 = None # Potential scale, to be defined in subclasses
        self.xi = None
        self.yi = 0.0
        self.Ai = 1e-5

    def f(self, x):
        """Dimensionless potential f(x)"""
        raise NotImplementedError

    def dfdx(self, x):
        """First derivative of potential  x"""
        raise NotImplementedError

    def d2fdx2(self, x):
        """Second derivative of potential  x"""
        raise NotImplementedError

    def get_initial_conditions(self):
        """Returns [xi, yi, zi, Ai]"""
        # zi depends on potential, calculating here ensures consistency
        zi = np.sqrt(self.yi**2/6 + (self.v0 * self.f(self.xi) / (3 * self.S**2)))
        # Return Ni (log scale factor) instead of Ai
        # Default Ai was 1e-5. Ni = ln(1e-5) approx -11.51
        Ni = np.log(self.Ai)
        return [self.xi, self.yi, zi, Ni]


class QuadraticModel(InflationModel):
    def __init__(self):
        super().__init__("Quadratic Inflation")
        M = 5.9e-6
        self.v0 = 0.5 * M**2
        self.xi = 17.5 # Approx 60 e-folds

    def f(self, x):
        return x**2

    def dfdx(self, x):
        return 2 * x

    def d2fdx2(self, x):
        return 2

class HiggsModel(InflationModel):

    def __init__(self, lam=0.1, xi=1000.0):
        super().__init__("Higgs Inflation")
        self.alpha = np.sqrt(2/3)
        self.lam = lam
        self.xi_val = xi
   
        self.v0 = self.lam / (4 * self.xi_val**2)

    def f(self, x):
        return (1 - np.exp(-self.alpha * x))**2

    def dfdx(self, x):
        return 2 * self.alpha * np.exp(-self.alpha * x) * (1 - np.exp(-self.alpha * x))

    def d2fdx2(self, x):
        return 2 * self.alpha**2 * np.exp(-self.alpha * x) * (2 * np.exp(-self.alpha * x) - 1)


class NonMinimalQuarticModel(InflationModel):
    """
    Quartic potential with non-minimal coupling: V_J = lambda/4 * phi^4,  xi * phi^2 * R
    Handles the field redefinition numerically.
    """
    def __init__(self, xi=10.0, lam=0.01):
        super().__init__(f"NMC Quartic (xi={xi})")
        
        # Physics Parameters
        self.xi_val = xi
        self.lam = lam
        self.v0 = self.lam / 4.0
        

        self.phi_max = 100.0
        self.phi_grid = np.linspace(0, self.phi_max, 1000)
        
        # d(psi)/d(phi) function
        def dpsi_dphi_deriv(psi, phi):
            # Formula: dphi/dpsi = sqrt(1 + xi*psi^2 (1 + 6*xi)) / (1 + xi*psi^2)
            # So dpsi/dphi is the inverse.
            # Handle psi=0 case safely
            psi = psi[0] # odeint passes list
            
            # Avoid division by zero or errors if psi is negative (though we solve for positive)
            if psi < 0: psi = 0 
            
            num = 1 + self.xi_val * psi**2
            den = np.sqrt(1 + self.xi_val * psi**2 * (1 + 6*self.xi_val))
            return [num / den]

        if self.xi_val == 0:
            # Trivial case, psi = phi
            self.psi_of_phi = lambda x: x
        else:
            from scipy.integrate import odeint
            from scipy.interpolate import CubicSpline
            
            # Solve ODE
            # Initial condition: psi(0) = 0
            psi_sol = odeint(dpsi_dphi_deriv, [0.0], self.phi_grid)
            psi_vals = psi_sol[:, 0]
            
            # Create Spline
            self.psi_spline = CubicSpline(self.phi_grid, psi_vals)
            self.psi_of_phi = self.psi_spline

        # Set initial condition guess (approx 60 e-folds)
        # For large xi, phi ~ sqrt(6) ln(...) ~ 5.5 like Higgs
        # For small xi, phi ~ 15 like Quadratic
        if self.xi_val > 1:
            self.xi = 5.5
        else:
            self.xi = 15.0

    def _get_psi(self, x):
        # Handle scalar or array
        if np.isscalar(x):
            if x > self.phi_max:
                # Fallback extrapolation or just warn? 
                # For now linear extrapolation using last derivative approx?
                # Or just let spline extrapolate (CubicSpline does this)
                pass 
            return self.psi_of_phi(x)
        else:
            return self.psi_of_phi(x)

    def _get_dpsi_dphi(self, psi):
        # Analytic form of dpsi/dphi(psi)
        num = 1 + self.xi_val * psi**2
        den = np.sqrt(1 + self.xi_val * psi**2 * (1 + 6*self.xi_val))
        return num / den

    def _get_d2psi_dphi2(self, psi, dpsi_dphi):
        # d/dphi (dpsi/dphi) = d/dpsi(dpsi/dphi) * dpsi/dphi
        # Let G(psi) = dpsi/dphi = (1 + xi*p^2) / sqrt(1 + xi*p^2(1+6xi))
        # dG/dpsi:
        # u = 1 + xi*p^2, v = sqrt(...)
        # G' = (u'v - uv')/v^2
        # u' = 2*xi*p
        # v' = 0.5/v * (2*xi*p*(1+6xi))
        
        xi = self.xi_val
        p = psi
        
        u = 1 + xi*p**2
        v2 = 1 + xi*p**2 * (1 + 6*xi)
        v = np.sqrt(v2)
        
        du = 2*xi*p
        dv = (xi*p*(1+6*xi)) / v
        
        dG_dpsi = (du*v - u*dv) / v2
        
        return dG_dpsi * dpsi_dphi

    def f(self, x):
        psi = self._get_psi(x)
        # V_E = (lam/4 * psi^4) / (1 + xi*psi^2)^2
        # f = V_E / v0 = V_E / (lam/4) = psi^4 / (1 + xi*psi^2)^2
        num = psi**4
        den = (1 + self.xi_val * psi**2)**2
        return num / den

    def dfdx(self, x):
        psi = self._get_psi(x)
        dpsi = self._get_dpsi_dphi(psi)
        
        # f(psi) = p^4 / (1 + xi*p^2)^2
        # df/dpsi = ( 4p^3 * (1+xi*p^2)^2 - p^4 * 2(1+xi*p^2)*2xi*p ) / (1+xi*p^2)^4
        #         = ( 4p^3(1+xi*p^2) - 4xi*p^5 ) / (1+xi*p^2)^3
        #         = ( 4p^3 + 4xi*p^5 - 4xi*p^5 ) / ...
        #         = 4p^3 / (1+xi*p^2)^3
        
        df_dpsi = 4 * psi**3 / (1 + self.xi_val * psi**2)**3
        
        return df_dpsi * dpsi

    def d2fdx2(self, x):
        psi = self._get_psi(x)
        dpsi = self._get_dpsi_dphi(psi)
        d2psi = self._get_d2psi_dphi2(psi, dpsi)
        
        # d2f/dx2 = d/dx ( df_dpsi * dpsi )
        #         = (d/dpsi(df_dpsi) * dpsi) * dpsi  + df_dpsi * d2psi
        
        # f'(p) = 4p^3 (1+xi*p^2)^-3
        # f''(p) = 12p^2 (..)^-3 + 4p^3 * (-3)(..)^-4 * 2xi*p
        #        = 12p^2 / (..)^3 - 24xi*p^4 / (..)^4
        #        = ( 12p^2(1+xi*p^2) - 24xi*p^4 ) / (1+xi*p^2)^4
        #        = ( 12p^2 + 12xi*p^4 - 24xi*p^4 ) / ...
        #        = ( 12p^2 - 12xi*p^4 ) / (1+xi*p^2)^4
        
        term1 = 1 + self.xi_val * psi**2
        d2f_dpsi2 = (12 * psi**2 - 12 * self.xi_val * psi**4) / term1**4
        
        # Re-calc df_dpsi
        df_dpsi = 4 * psi**3 / term1**3
        
        return d2f_dpsi2 * dpsi**2 + df_dpsi * d2psi
