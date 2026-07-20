
import numpy as np
from scipy.integrate import solve_ivp

from ..misc.units import *
from PyESRSTM.Trasport.rates import sum_rates

allowed_methods = ['RK45', 'RK23', 'DOP853', 'BDF', ]
real_only_mehods = ['Radau', 'LSODA']

def propagate_density(GL: np.ndarray, GR: np.ndarray, Delta: np.ndarray, rho0: np.ndarray, tf: float, 
                      frequency: float = None, adrive: np.ndarray = None, drho_function = None,
                    return_all = False,  kwargs = None):
    """
    Propagate the density matrix in time
    d rho_lj /dt  = i Delta_lj rho_lj + ....
    Parameters
    ----------
    GL, GR : array
        The left and right rate matrices with shape (Nstate, Nstate, Nstate, Nstate, 2*Nfour+1), where Nstate is the dimension of the quantum dot Hilbert space [Hartree].
    Delta : array
        The energy difference matrix with shape (Nstate, Nstate) [Hartree].
    frequency : float, or array
        The frequency of the driving field [GHz]. if an arrya is given, the amplitudes of the driving field are given by adrive.
    rho0 : array
        The initial density matrix with shape (Nstate, Nstate).
    tf : float
        The final time for propagation [ns].
    adrive : array, optional
        The amplitudes of the driving field, with shape (2*Nfour+1,), where Nfour is the number of Fourier components.
    return_all : bool, optional
        Whether to return the density matrix at all time points or just the final state. Default is False.

    Returns
    -------
    time : array
        The time points at which the density matrix is evaluated [ns].
    rho : array
        The propagated density matrix at each time point, with shape (Ntime, Nstate, Nstate).
    
    if return_all is True, returns the full solution object from solve_ivp [A.u.].
    """
    
    Ndim = rho0.shape[0]
    assert rho0.shape == (Ndim, Ndim), "rho0 must be a square matrix."
    assert Delta.shape == (Ndim, Ndim), "Delta must be a square matrix, with the same dimension as rho0."
    assert isinstance(tf, (float, np.float64, int, np.int64)), "tf must be a float."

    # Convert units to atomic units for the integration
    tf /= time_unit # Hart
    if frequency is not None:
        frequency = frequency * 2*np.pi*time_unit # Hart (use multiplication, not in-place)

    if kwargs is not None:
        # Create a copy to avoid modifying the original kwargs dict
        kwargs = kwargs.copy()
        
        if 'method' in kwargs:
            method = kwargs['method']
            if method not in allowed_methods + real_only_mehods:
                raise ValueError(f'Invalid method {method}. Allowed methods are {allowed_methods}.')
            if method in real_only_mehods:
                raise ValueError(f'Method {method} only supports real-valued functions, can\'t use it here :(.')
        
        if 'max_step' in kwargs:
            kwargs['max_step'] /= time_unit # Hart

    # setup the propagation tensor
    #ML = QME_matrix_propagator(GL, Delta)
    #MR = QME_matrix_propagator(GR, Delta)

    # setup the integrant function for solve_ivp
    if drho_function is not None:
        assert GL.shape == (Ndim,Ndim,Ndim,Ndim,1), "GL must have shape (Nstate, Nstate, Nstate, Nstate, 1) when using adrive."
        assert GR.shape == (Ndim,Ndim,Ndim,Ndim,1), "GR must have shape (Nstate, Nstate, Nstate, Nstate, 1) when using adrive."
        assert drho_function.__code__.co_argcount == 4, "drho_function must have 4 arguments: t, rho, ML, MR."

        ML = QME_matrix_propagator(GL, np.zeros_like(Delta)) # delta can only be in included once
        MR = QME_matrix_propagator(GR, Delta)
        integrant = lambda t, rho: drho_function(t, rho.reshape(rho0.shape), ML[:,:,:,:,0], MR[:,:,:,:,0]).flatten()

    elif adrive is not None:
        assert frequency is not None, "frequency must be provided when using adrive."
        assert isinstance(frequency, np.ndarray), "frequency must be a numpy array when using adrive."
        assert len(adrive) == len(frequency), "adrive and frequency must have the same length."
        assert GL.shape == (Ndim,Ndim,Ndim,Ndim,1), "GL must have shape (Nstate, Nstate, Nstate, Nstate, 1) when using adrive."
        assert GR.shape == (Ndim,Ndim,Ndim,Ndim,1), "GR must have shape (Nstate, Nstate, Nstate, Nstate, 1) when using adrive."

        ML = QME_matrix_propagator(GL, np.zeros_like(Delta)) # delta can only be in included once
        MR = QME_matrix_propagator(GR, Delta)
        integrant = lambda t, rho: single_step_adrive(t, rho.reshape(rho0.shape), ML[:,:,:,:,0], MR[:,:,:,:,0], frequency, adrive).flatten()

    else:
        
        assert frequency is not None, "frequency must be provided when using full Fourier."
        assert isinstance(frequency, float), "frequency must be a float when using full Fourier."
        #assert len(M.shape) == 5, "M must have shape (Nstate, Nstate, Nstate, Nstate, 2*Nfour+1) when using full Fourier."
        
        M = QME_matrix_propagator(sum_rates(GL, GR), Delta)
        #ML = QME_matrix_propagator(GL, np.zeros_like(Delta)) # delta can only be in included once
        #MR = QME_matrix_propagator(GR, Delta)
        #M = sum_rates(ML, MR) # sum the left and right rates

        Nfour = (M.shape[-1] - 1) // 2
        Ns = np.arange(-Nfour, Nfour + 1)
        integrant = lambda t, rho: single_step_full_fourier(t, rho.reshape(rho0.shape), M, frequency, Ns).flatten()

    # solve the ODE
    solution = solve_ivp(integrant, (0, tf), rho0.flatten(), **kwargs)

    print(f'Integration successful: {solution.success}, message: {solution.message}')
    
    if return_all:
        return solution

    return solution.t*time_unit, solution.y.T.reshape(-1, *rho0.shape)


def QME_matrix_propagator(G, Delta,):
    """From rates G and Delta, preparse the QME matrix such that 
    returns M_{lj,vu,n} such that    
    M_{lj,vu}(t) = \sum_n M_{lj,vu,n} exp(i n omega t)
    and d rho_{lj}(t)/dt = M_{lj,vu}(t) rho_{vu}(t)

    """
    
    Ndim = G.shape[0]
    NFtotal = G.shape[4]
    NF = int((NFtotal - 1) / 2)

    M = np.zeros((Ndim, Ndim, Ndim, Ndim, NFtotal), dtype=complex)
    
    for j in range(Ndim):
        for l in range(Ndim):
            # rho_{lj} = i Delta_{lj} rho_{lj}
            M[j,l,j,l,NF] += 1j*Delta[l,j] 
    
    for n in range(-NF, NF+1):      
        
        # transport 
        # Gamma_{v,l,j,u}(t) rho_{v,u}(t)
        M[:, :, :, :, n+NF] += np.einsum('vlju->jluv', G[:, :, :, :, n+NF])
        
        # Gamma^*_{u,j,l,v}(t) rho_{v,u}(t)
        M[:, :, :, :, n+NF] += np.einsum('ujlv->jluv', np.conj(G[:, :, :, :, -n+NF]))

        for l in range(Ndim):
            # Gamma_{j,v,v,u}(t) rho_{l,u}(t)
            M[:, l, :, l, n+NF] -= np.einsum('jvvu->ju', G[:, :, :, :, n+NF])

            # Gamma_{l,v,v,u}(t) rho_{u,j}(t) 
            # in the code I spwaped l and j since I don't want to write a seperate loop
            M[l, :, l, :, n+NF] -= np.einsum('jvvu->ju', np.conj(G[:, :, :, :,  -n+NF]))

    return M

def single_step_adrive(t, rho, ML, MR, frequencis, As):
    """"
    Propagate the density matrix by a single time step using the QME matrix.

    t: float
        The time.
    rho: array
        The density matrix at the current time step with shape (Nstate, Nstate).
    M0: array
        The time-propagation tensor with shape (Nstate, Nstate, Nstate, Nstate).
    frequencis: array
        The frequencies of the driving field.
    As: array
        The amplitudes of the driving field.

    Returns
    -------
    drho : array
        The derivative of the density matrix with shape (Nstate, Nstate).
    """
    phase = 1 + np.sum(As*np.cos(frequencis * t))
    M_t = MR + ML * phase
    drho = np.einsum('ljvu,vu->lj', M_t, rho)

    return drho

def single_step_full_fourier(t, rho, M, freq, Ns):
    """
    Propagate the density matrix by a single time step using the QME matrix.

    Parameters
    ----------
    t : float
        The time.
    rho : array
        The density matrix at the current time step with shape (Nstate, Nstate).
    M : array
        The time-propagation tensor with shape (Nstate, Nstate, Nstate, Nstate, 2*Nfour+1), where Nstate is the dimension of the quantum dot Hilbert space.
    freq : float
        The frequency of the driving field.
    Ns : array
        The Floquet sideband indices, with shape (2*Nfour+1,).

    Returns
    -------
    drho : array
        The derivative of the density matrix with shape (Nstate, Nstate).
    """

    phase = np.exp(1j * freq * Ns * t)
    M_t = np.tensordot(M, phase, axes=([-1], [0]))
    drho = np.einsum('ljvu,vu->lj', M_t, rho)
    
    return drho



def rates_in_time(M: np.ndarray, freq: float, t: np.ndarray, Ns: np.ndarray):
    """
    Evaluate the rate matrix at given time points

    Parameters
    ----------
    M : array
        The time-propagation tensor with shape (Nstate, Nstate, Nstate, Nstate, 2*Nfour+1), where Nstate is the dimension of the quantum dot Hilbert space.
    freq : float
        The frequency of the driving field.
    t : array
        The time points at which to evaluate the rate matrix.
    Ns : array
        The Floquet sideband indices, with shape (2*Nfour+1,).

    Returns
    -------
    M_t : array
        The time-propagation tensor evaluated at each time point, with shape (Ntime, Nstate, Nstate, Nstate, Nstate).
    """

    t = np.atleast_1d(t)
    ent = np.exp(1j * freq * np.outer(t, Ns))

    return np.tensordot(ent, M, axes=([-1], [-1]))


def current_in_time(time: np.ndarray, rho: np.ndarray, G: np.ndarray, frequency: float):
    """
    Evaluate the current at given time points

    Parameters
    ----------
    time : array
        The time points at which to evaluate the current.
    rho : array 
        The density matrix at the current time step with shape (Nstate, Nstate).
    G : array
        The rate matrix with shape (Nstate, Nstate, Nstate, Nstate, 2*Nfour+1), where Nstate is the dimension of the quantum dot Hilbert space
    frequency : float
        The frequency of the driving field.

    Returns
    -------
    I_t : array
        The current evaluated at each time point, with shape (Ntime,).
    """
    
    frequency *= 2*np.pi*time_unit # Hart

    NF = (G.shape[-1] - 1) // 2
    Ns = np.arange(-NF, NF + 1)
    Gt = rates_in_time(G, frequency, time, Ns) # Hart

    # a stands for time index
    I_t = np.einsum('aul,aljju -> a', rho, Gt)

    return I_t * pA



