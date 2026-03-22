import numpy as np

from ..Trasport.rates import rates, sum_rates
from .QME import QME, QME_matrix
from ..Trasport.Curent import current

from ..Electrode.Electrode import Electrode
from ..QD import QD

def ESR(EL_left: Electrode, EL_right: Electrode, 
        dot: QD, frequencies: np.ndarray, NFL: int=0, NFR: int=0, return_Gs: bool = False):
    """Calculatutes ESR signal along a range of frequencies

    Parameters:
    EL_left, EL_right: (Electrode) 
        Electrodes the 

    """
    freqL = 0 if EL_left.Vrf ==0 else np.max(np.abs(frequencies))
    freqR = 0 if EL_right.Vrf ==0 else np.max(np.abs(frequencies))
    
    GLminus, GLplus = rates(dot, EL_left, freqL, NFL)
    GRminus, GRplus = rates(dot, EL_right, freqR, NFR)

    # make sure we add to the correct Floquet component
    GL = GLminus + GLplus
    GR = GRminus + GRplus

    G = sum_rates(GL, GR)
    
    GC = GLplus-GLminus

    I = np.zeros((len(frequencies), 2*max(NFR, NFL)+1), dtype=complex)
    M = QME_matrix(G, dot.Delta, )
    for i, f in enumerate(frequencies):
        rho = QME(M.copy(), f, qme_matrix=True)
        I[i] = current(rho,GC)
    
    if return_Gs:
        return I, GL, GR
    
    return I


def ESR_fit_fun(x, res, gamma, Isym, Iasym, p0, p1, p2, pinv):
    """
    ESR fitting function.
    
    Ips = Isym/(1 + (x - res)**2 / gamma**2)
    Ipa = 2 * Iasym * (x - res) / (2 * (x - res)**2 + gamma**2)
    I0 = pinv/(x+1e-10) + p0 + p1 * x + p2 * x**2
    I = I0 + Ips + Ipa

    Parameters:
    x : array_like
        The independent variable (frequency or field).
    res : float
        Resonance position.
    gamma : float
        Lorentzian width (damping).
    Isym : float
        Symmetric peak current amplitude.
    Iasym : float
        Asymmetric peak current amplitude (Fano).
    p0, p1, p2 : float
        Coefficients for the polynomial background current.
    pinv : float
        Inverse background current coefficient.
    Returns:
    y : array_like
        The fitted values for the given x.
    """

    Background = background_current(x, p0, p1, p2, pinv) 
    peak = peak_curent(x, res, gamma, Isym, Iasym)
              
    return peak + Background

def peak_curent(x, res, gamma, Isym, Iasym):
    """
    Peak current function for ESR fitting.
    Coppied from Jose's Matlab code.

    peak_sym = peak0/(1 + (x - res)**2 / gamma**2)
    peak_asym = 2 * peak1 * (x - res) / (2 * (x - res)**2 + gamma**2)
    return peak_sym + peak_asym
    """
    
    y = 2*(x - res)/gamma 
    # don't know why the factor of 2 is needed, but it is in the original code

    nominator = Isym + 2*Iasym * y
    denominator = y**2 + 1
    peak = nominator / denominator
    return peak

def background_current(x, p0, p1, p2, pinv,):
    """
    Polynomial background current function for ESR fitting.
    Coppied from Jose's Matlab code.

    I0 = pinv/(x+1e-10) + p0 + p1 * x + p2 * x**2
    """
    I0 = pinv/(x+1e-10) + p0 + p1 * x + p2 * x**2
    return I0

def guess_p0(x, y, x0):
    
    b = (y[-1]-y[0])/(x[-1]-x[0])
    a = y[0] - b*x[0]

    y -= b*x + a

    Is = y[np.argmin(np.abs(x-x0))]

    argmin = np.argmin(y)
    argmax = np.argmax(y)
    
    gamma = x[argmax] - x[argmin] # assuming Is = 0, but should be close
    Ia = y[argmax] - y[argmin]
    
    Ia = gamma/np.abs(gamma)*Ia # fix sign
    gamma = np.abs(gamma)

    p0 = np.array([x0, gamma, Is, Ia, a, b , 0., 0.])
    #upper_bound = [x0 + size/2, 1.,  1.1*np.abs(Ia),  1.1*np.abs(Ia), np.inf, np.inf, np.inf, np.inf]
    #lower_bound = [x0 - size/2, 0., -1.1*np.abs(Ia), -1.1*np.abs(Ia), np.inf, np.inf, np.inf, np.inf]
    
    return p0