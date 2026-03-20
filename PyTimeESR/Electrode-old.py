import numpy as np  
from scipy.integrate import simpson, quad_vec

from .units import Hartree

class Electrode():
    integral_methods_dict = {
        'trapezoid': np.trapezoid,
        'simpson' : simpson
    }
    dos_dict = {
        'flat': lambda _: 1
    }

    def __init__(self, Vdc: float, Vrf: float, Polarization: float, g0: float, gammaC: float, Temperature: float, 
                 Adrive: complex = 0.0, Cutoff: float = 1000., Nint: int = int(5e5), 
                 integral_method: str = 'trapezoid', dos: str = 'flat', dos_kwargs = {}):
        self.Vdc = Vdc/Hartree
        self.Vrf = Vrf/Hartree
        self.Spin_polarization = Polarization
        self.g0 = g0/Hartree
        self.gammaC = gammaC/Hartree
        self.Temperature = Temperature * 25.852 / (Hartree*300.)
        self.Cutoff = Cutoff/Hartree
        self.Adrive = Adrive
        self.Nint = int(Nint)

        self.int_meth_label = integral_method
        self.int_meth = self.integral_methods_dict[integral_method]
        self.dos = lambda x: self.dos_dict[dos](x, **dos_kwargs)

    def fermiIntegralquad(self, E: float, floque_enegrgies):
        # Prepare constants for integration
        
        fermiD = self.FermiFunction((x-self.Vdc)/self.Temperature)* self.dos(x-self.Vdc)
        denom = -E + floquet_energies[:, None] + 1j * self.gammaC
        udenom = E + floquet_energies[:, None] - 1j * self.gammaC

        
        integrand_p = lambda x: fermiD(x) / (x + denom)
        integrand_m =  lambda x: (1 - fermiD(x)) / (x + udenom)

        Ifp, _ = 1j* quad_vec(integrand_p, -self.Cutoff, self.Cutoff)/ np.pi
        Ifm, _ = -1j* quad_vec(integrand_m, -self.Cutoff, self.Cutoff)/ np.pi
    
    def fermiIntegral(self, E: float, floquet_energies: np.ndarray):
        """ Calculate the Fermi integrals Ifplus and Ifminus for a given energy E, Floquet energies, and broadening gammaC.

        The Fermi integrals are defined as:
        Ifplus(E) = \int_{-Cutoff}^{Cutoff} dx f(x) / (x - E + fe + i*gammaC) # electron tunneling into the quantum dot
        Ifminus(E) = \int_{-Cutoff}^{Cutoff} dx (1-f(x)) / (x + E + fe - i*gammaC) # hole tunneling into the quantum dot

        """
        # linear sampling can lead to large number of integration points for
        # instead we can use 
        x = np.linspace(-self.Cutoff, self.Cutoff, self.Nint) # energy grid 
        fermiD = self.FermiFunction((x-self.Vdc)/self.Temperature) # Fermi function
        
        fermiD *= self.dos(x-self.Vdc) # density of states. also shifts the energy grid by Vdc.
        # not really a fermi distribuiton after multiplying by the density of states,
        # but we will still call it fermiD for simplicity.

        denom = -E + floquet_energies[:, None] + 1j * self.gammaC
        udenom = E + floquet_energies[:, None] - 1j * self.gammaC

        integrand_p = fermiD / (x[None, :] + denom)
        integrand_m = (1 - fermiD) / (x[None, :] + udenom)
        
        Ifp =  1j*self.int_meth(integrand_p, x, axis=1)/np.pi
        Ifm = -1j*self.int_meth(integrand_m, x, axis=1)/np.pi
        
        return Ifp, Ifm

    def FermiFunction(self, x):
        F = np.empty_like(x)
        idx = x>0
        F[idx]  = np.exp(-x[idx])/(1+np.exp(-x[idx]))
        F[~idx] = 1/(1+np.exp(x[~idx]))
        return F