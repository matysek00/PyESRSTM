import numpy as np  
from scipy.integrate import simpson, romb, quad_vec, quad

from ..misc.units import Hartree

class Electrode():

    dos_dict = {
        'flat': lambda _: 1
    }

    def __init__(self, Vdc: float, Vrf: float, Polarization: float, g0: float, gammaC: float, Temperature: float, 
                 Adrive: complex = 0.0, Cutoff: float = 1000., Nint: int = int(5e5), 
                 integral_method: str = 'trapezoid', dos: str = 'flat', dos_kwargs = {}, segments={}):
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
        self.dos = lambda x: self.dos_dict[dos](x, **dos_kwargs)
        
        if integral_method == 'trapezoid':
            self.setup_grid(segments)
            self.fermiIntegral = lambda E, floquet: self.FermiIntegralGrid(E, floquet, np.trapezoid)
        elif integral_method == 'simpson':
            self.setup_grid(segments)
            self.fermiIntegral = lambda E, floquet: self.FermiIntegralGrid(E, floquet, simpson)
        #elif integral_method == 'romb':
        #    self.grid = np.linspace(-self.Cutoff, self.Cutoff, self.Nint)
        #    self.fermi = Electrode.FermiFunctionArray((self.grid-self.Vdc)/self.Temperature)
        #    self.fermiIntegral = lambda E, floquet: self.FermiIntegralGrid(E, floquet, romb)
        elif integral_method == 'quad':
            self.inter_fp = lambda x,z: self.FermiFunctionfloat((x-self.Vdc)/self.Temperature)/ (x + z)
            self.inter_fm = lambda x,z: (1-self.FermiFunctionfloat((x-self.Vdc))/self.Temperature)/ (x + z)
            self.fermiIntegral = lambda E, floquet: self.FermiIntegralAdapt(E, floquet)

    def setup_grid(self, segments):
        grids = [np.linspace(-self.Cutoff, self.Cutoff, self.Nint)]
        for seg in segments:
            grids += [np.linspace(self.Vdc+(seg[0]-seg[1])/Hartree, self.Vdc+(seg[0]+seg[1])/Hartree, int(seg[2]))]
        self.grid = np.sort(np.concatenate(grids))
        self.fermi = Electrode.FermiFunctionArray((self.grid-self.Vdc)/self.Temperature)

    def FermiIntegralGrid(self, E: float, floquet_energies, method):
        """ Calculate the Fermi integrals Ifplus and Ifminus for a given energy E, Floquet energies, and broadening gammaC.

        The Fermi integrals are defined as:
        Ifplus(E) = \int_{-Cutoff}^{Cutoff} dx f(x) / (x - E + fe + i*gammaC) # electron tunneling into the quantum dot
        Ifminus(E) = \int_{-Cutoff}^{Cutoff} dx (1-f(x)) / (x + E + fe - i*gammaC) # hole tunneling into the quantum dot

        """
        
        denom = -E + floquet_energies + 1j * self.gammaC
        udenom = E + floquet_energies - 1j * self.gammaC
        
        integrand_p = self.fermi[None,:] / (self.grid[None,:] + denom[:, None])
        integrand_m =  (1 - self.fermi[None,:]) / (self.grid[None,:] + udenom[:, None])

        Ifp = 1j* method(integrand_p, self.grid, axis=1)/ np.pi
        Ifm = -1j* method(integrand_m, self.grid, axis=1)/ np.pi
        
        return Ifp, Ifm
    

    def FermiIntegralAdapt(self, E: float, floquet_energies):
        """ Calculate the Fermi integrals Ifplus and Ifminus for a given energy E, Floquet energies, and broadening gammaC.

        The Fermi integrals are defined as:
        Ifplus(E) = \int_{-Cutoff}^{Cutoff} dx f(x) / (x - E + fe + i*gammaC) # electron tunneling into the quantum dot
        Ifminus(E) = \int_{-Cutoff}^{Cutoff} dx (1-f(x)) / (x + E + fe - i*gammaC) # hole tunneling into the quantum dot

        """
        
        denom = -E + floquet_energies + 1j * self.gammaC
        udenom = E + floquet_energies - 1j * self.gammaC
        
        Ifp = np.array([quad(self.inter_fp, -self.Cutoff, self.Cutoff, z, complex_func=True)[0] for z in denom])
        Ifm = np.array([quad(self.inter_fm, -self.Cutoff, self.Cutoff, z, complex_func=True)[0] for z in udenom])
        
        Ifp *= 1j/ np.pi
        Ifm *= -1j/ np.pi
        
        return Ifp, Ifm

    def quad_integrate(self, integrand):
        return np.array([quad(x, -self.Cutoff, self.Cutoff, complex_func=True)[0] for x in integrand])
    
    @staticmethod
    def FermiFunctionfloat(x):
        return np.exp(-x)/(1+np.exp(-x)) if x > 0 else 1/(1+np.exp(x)) 
    
    @staticmethod
    def FermiFunctionArray(x):
        F = np.empty_like(x)
        idx = x>0
        F[idx]  = np.exp(-x[idx])/(1+np.exp(-x[idx]))
        F[~idx] = 1/(1+np.exp(x[~idx]))
        return F