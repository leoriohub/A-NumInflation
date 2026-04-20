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
        self.phi0 = None
        self.yi = None
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
        """Returns [phi0, yi, zi, Ai]"""
        # zi depends on potential, calculating here ensures consistency
        zi = np.sqrt(self.yi**2/6 + (self.v0 * self.f(self.phi0) / (3 * self.S**2)))
        # Return Ni (log scale factor) instead of Ai
        # Default Ai was 1e-5. Ni = ln(1e-5) approx -11.51
        Ni = np.log(self.Ai)
        return [self.phi0, self.yi, zi, Ni]


class QuadraticModel(InflationModel):
    def __init__(self):
        super().__init__("Quadratic Inflation")
        M = 5.9e-6
        self.v0 = 0.5 * M**2
        self.phi0 = 17.5 # Approx 60 e-folds

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
        
        # Default Initial Conditions (USR Exploration Defaults)
        self.phi0 = 5.8
        self.yi = -0.01

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
            self.phi0 = 5.5
        else:
            self.phi0 = 15.0

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

class FullHiggsModel(InflationModel):
    """
    Exact Higgs Inflation potential without the high-field approximation.
    Integrates the exact conformal inversion numerically to retain absolute precision
    throughout reheating down to the true h^4 minimum.
    """
    def __init__(self, lam=0.1, xi=1000.0, v_vev=0.0):
        super().__init__("Full Higgs Inflation")
        
        self.lam = lam
        self.xi_val = xi
        self.v_vev = v_vev # Default to 0, since v/M_P is tiny (~10^-15)
        
        # We scale v0 exactly like the approximated model so the plateau height is ~1.0
        self.v0 = self.lam / (4 * self.xi_val**2) 
        
        # Standard Initial Conditions
        self.phi0 = 5.5
        self.yi = -1

        # Precompute the inverse transformation grid: psi(x) where psi = h/M_P, x = chi/M_P
        self.psi_max = 100.0 / np.sqrt(self.xi_val) # Enough to reach way past the plateau
        self.psi_grid = np.linspace(0, self.psi_max, 5000)
        
        def dx_dpsi_deriv(x, psi): # ODE system for x(psi)
            if psi < 0: psi = 0 
            num = np.sqrt(1 + self.xi_val * psi**2 * (1 + 6 * self.xi_val))
            den = 1 + self.xi_val * psi**2
            return [num / den]

        from scipy.integrate import odeint
        from scipy.interpolate import CubicSpline
        
        # Solve x(psi)
        x_sol = odeint(dx_dpsi_deriv, [0.0], self.psi_grid)
        self.x_grid = x_sol[:, 0]
        
        # We need the inverse: psi(x)
        self.psi_spline = CubicSpline(self.x_grid, self.psi_grid)

    def _get_psi(self, x):
        # Prevent accessing outside the grid by clipping x to 0 at the low end
        if np.isscalar(x):
            return self.psi_spline(max(0, x))
        return self.psi_spline(np.maximum(0, x))

    def _get_dpsi_dx(self, psi):
        # Exact mathematical inverse: dpsi/dx = 1 / (dx/dpsi)
        num = 1 + self.xi_val * psi**2
        den = np.sqrt(1 + self.xi_val * psi**2 * (1 + 6 * self.xi_val))
        return num / den

    def _get_d2psi_dx2(self, psi, dpsi_dx):
        # d/dx (dpsi/dx) = d/dpsi (dpsi/dx) * dpsi/dx
        u = 1 + self.xi_val * psi**2
        v2 = 1 + self.xi_val * psi**2 * (1 + 6 * self.xi_val)
        v = np.sqrt(v2)
        
        du = 2 * self.xi_val * psi
        dv = (self.xi_val * psi * (1 + 6 * self.xi_val)) / v
        
        dG_dpsi = (du * v - u * dv) / v2
        return dG_dpsi * dpsi_dx

    def f(self, x):
        psi = self._get_psi(x)
        # f(psi) = [ xi * (psi^2 - v^2) / (1 + xi*psi^2) ]^2
        num = self.xi_val * (psi**2 - self.v_vev**2)
        den = 1 + self.xi_val * psi**2
        return (num / den)**2

    def dfdx(self, x):
        psi = self._get_psi(x)
        dpsi = self._get_dpsi_dx(psi)
        
        num = self.xi_val * (psi**2 - self.v_vev**2)
        den = 1 + self.xi_val * psi**2
        g = num / den
        
        num_dg = 2 * self.xi_val * psi * (1 + self.xi_val * self.v_vev**2)
        den_dg = (1 + self.xi_val * psi**2)**2
        dg_dpsi = num_dg / den_dg
        
        df_dpsi = 2 * g * dg_dpsi
        return df_dpsi * dpsi

    def d2fdx2(self, x):
        psi = self._get_psi(x)
        dpsi = self._get_dpsi_dx(psi)
        d2psi = self._get_d2psi_dx2(psi, dpsi)
        
        num = self.xi_val * (psi**2 - self.v_vev**2)
        den = 1 + self.xi_val * psi**2
        g = num / den
        
        num_dg = 2 * self.xi_val * psi * (1 + self.xi_val * self.v_vev**2)
        den_dg = (1 + self.xi_val * psi**2)**2
        dg_dpsi = num_dg / den_dg
        
        term1 = 2 * self.xi_val * (1 + self.xi_val * self.v_vev**2)
        term2 = 1 + self.xi_val * psi**2
        term3 = 2 * self.xi_val * psi
        
        dnum_dg = term1
        dden_dg = 2 * term2 * term3
        
        d2g_dpsi2 = (dnum_dg * den_dg - num_dg * dden_dg) / (den_dg**2)
        d2f_dpsi2 = 2 * (dg_dpsi**2 + g * d2g_dpsi2)
        
        return d2f_dpsi2 * (dpsi**2) + (2 * g * dg_dpsi) * d2psi


class SmoothUSRTransitionModel(InflationModel):
    """
    Numerically reconstructed potential from the analytical smooth SR-USR-SR model.
    Based on arXiv:2603.17465v1
    """
    def __init__(self, alpha=22.63, mu=2.0294, eps_sr1=1e-6, H0=1.0):
        super().__init__(f"Smooth USR (alpha={alpha}, mu={mu})")
        
        from scipy.special import hyperu, hyp1f1
        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import CubicSpline
        
        self.v0 = H0**2
        self.S = 5e-5 # Time unit scale
        
        N_vals_sr = np.linspace(-4, 0, 500, endpoint=False)
        N_vals_usr = np.linspace(0, 15, 1500)
        
        q_sq = 9/4 + alpha - mu**2
        q = np.sqrt(q_sq)
        
        def W(kappa, mu_val, z):
            return np.exp(-z/2) * (z**(mu_val + 0.5)) * hyperu(0.5 + mu_val - kappa, 1 + 2*mu_val, z)
            
        def M(kappa, mu_val, z):
            return np.exp(-z/2) * (z**(mu_val + 0.5)) * hyp1f1(0.5 + mu_val - kappa, 1 + 2*mu_val, z)
            
        k0 = alpha / (2*q)
        k1 = k0 + 1
        
        W0 = W(k0, mu, 2*q)
        M0 = M(k0, mu, 2*q)
        W1 = W(k1, mu, 2*q)
        M1 = M(k1, mu, 2*q)
        
        denom = (alpha + q + 2*mu*q)*M1*W0 + 2*q*M0*W1
        B1_prime = ((alpha - 2*q - 2*q**2)*M0 - (alpha + q + 2*mu*q)*M1) / denom
        B2_prime = -((alpha - 2*q - 2*q**2)*W0 + 2*q*W1) / denom
        
        z_arg = 2 * q * np.exp(-N_vals_usr)
        W_val = W(k0, mu, z_arg)
        M_val = M(k0, mu, z_arg)
        
        Z_scaled = B1_prime * W_val + B2_prime * M_val
        eps1_usr = eps_sr1 * np.exp(-2*N_vals_usr) * (Z_scaled)**2
        
        # Stitch arrays
        N_vals = np.concatenate((N_vals_sr, N_vals_usr))
        eps1_vals = np.concatenate((np.full_like(N_vals_sr, eps_sr1), eps1_usr))
        
        # dphi/dN = sqrt(2 * eps1)
        dphi_dN = np.sqrt(2 * eps1_vals)
        int_dphi = cumulative_trapezoid(dphi_dN, x=N_vals, initial=0.0)
        # We want phi to increase with N and be zero at N=0
        idx_N0 = np.argmin(np.abs(N_vals - 0.0))
        phi_vals = int_dphi - int_dphi[idx_N0]
        
        # Calculate exact H(N) instead of assuming H = H0
        int_eps1 = cumulative_trapezoid(eps1_vals, x=N_vals, initial=0.0)
        H_vals = H0 * np.exp(-int_eps1)

        # Now scale the potential with the exact H(N)
        V_vals = H_vals**2 * (3 - eps1_vals) 

        # Interp functions need strictly increasing x.
        # Remove any non-strictly increasing points (eps1 = 0)
        phi_uniq, uniq_idx = np.unique(phi_vals, return_index=True)
        V_uniq = V_vals[uniq_idx]
        
        self.phi_grid = phi_uniq
        self.V_grid = V_uniq
        
        self.v_spline = CubicSpline(self.phi_grid, self.V_grid)
        self.dv_spline = self.v_spline.derivative(nu=1)
        self.d2v_spline = self.v_spline.derivative(nu=2)
        
        # Set initial conditions for integration appropriately
        # Suppose we want to start 2 efolds before transition (N=-2).
        idx_i = np.argmin(np.abs(N_vals - (-2)))
        self.phi0 = phi_vals[idx_i]
        
        # Initial velocity proxy: yi = dx/dT = dphi/dN * z (since dN/dT = z approx H/S)
        H0_val = np.sqrt(self.v0)
        self.yi = dphi_dN[idx_i] * (H0_val / self.S)
    
    def f(self, x):
        return self.v_spline(x) # Let CubicSpline handle any out-of-bounds extrapolation

    def dfdx(self, x):
        x_safe = np.maximum(0, x)
        return self.dv_spline(x_safe)

    def d2fdx2(self, x):
        x_safe = np.maximum(0, x)
        return self.d2v_spline(x_safe)
