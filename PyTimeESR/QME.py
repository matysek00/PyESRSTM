import numpy as np
from scipy.linalg import solve

from .units import time_unit

def QME_freq_cont(frequency, Ndim, NF):
    """The frequency contribution to QME 
    """
    NFtotal = 2*NF+1
    M = np.zeros((Ndim, Ndim, NFtotal, Ndim, Ndim, NFtotal), dtype=float)
    
    for n in range(NFtotal):
        for l in range(Ndim):
            for j in range(Ndim):
                M[j,l,n, j,l,n] += frequency*(n-NF)
    return M


def QME_matrix(G, Delta,):
    """From rates G and Delta, preparse the QME matrix 
    EXCLUDES FREQUENCY CONTIBUTION!!!!
    """
    
    Ndim = G.shape[0]
    NFtotal = G.shape[4]
    NF = int((NFtotal - 1) / 2)

    M = np.zeros((Ndim, Ndim, NFtotal, Ndim, Ndim, NFtotal), dtype=complex)

    for n in range(NFtotal):
        # fourier transfomr
        #M[:, :, n, :, :,n] += 
        # magnetic field
        # rho_{lj} = 
        for l in range(Ndim):
            for j in range(Ndim):
                M[j,l,n, j,l,n] += Delta[l,j] #+ frequency*(n-NF)
          
        # transport 
        for m in range(max(0, n - NF), min(NFtotal, n + NF+1)):
            # rho_ljn =  M_{ljn,vum} rho_{vum}

            # Gamma_{v,l,j,u;n-m} rho_{v,u,m}
            #M[:, :, n, :, :, m] -= 1j*np.einsum('vlju->ljvu', G[:, :, :, :, n- m + NF])
            M[:, :, n, :, :, m] -= 1j*np.einsum('vlju->jluv', G[:, :, :, :, n- m + NF])
            
            # Gamma_{u,j,l,v;m-n} rho_{v,u,m}
            #M[:, :, n, :, :, m] -= 1j*np.einsum('ujlv->ljvu', np.conj(G[:, :, :, :, m - n + NF]))
            M[:, :, n, :, :, m] -= 1j*np.einsum('ujlv->jluv', np.conj(G[:, :, :, :, m - n + NF]))

            for l in range(Ndim):
                # Gamma_{j,v,v,u:n-m} rho_{l,u,m}
                #M[l, :, n, l, :, m] += 1j*np.einsum('jvvu->ju', G[:, :, :, :, n - m + NF])
                M[:, l, n, :, l, m] += 1j*np.einsum('jvvu->ju', G[:, :, :, :, n - m + NF])

                # Gamma_{l,v,v,u:m-n} rho_{u,j,m} 
                # in the code I spwaped l and j since I don't want to write a seperate loop
                #M [l, :, n, l, :, m]+= 1j*np.einsum('jvvu->ju', np.conj(G[:, :, :, :, m - n + NF]))
                M[l, :, n, l, :, m] += 1j*np.einsum('jvvu->ju', np.conj(G[:, :, :, :, m - n + NF]))
        
        # TODO: why do we do this
    #M[0,0] = 0
    
    #for n in range(NFtotal):
    #    for l in range(Ndim):
    #        M[0,0,n,l,l,n] = 1.

    return M

def QME_constrstants(Ndim, NFtotal, M):
    M[0,0] = 0
    
    for n in range(NFtotal):
        for l in range(Ndim):
            M[0,0,n,l,l,n] = 1.
    
    return M

def QME(G: np.ndarray, frequency: float , qme_matrix: bool = False, Delta: np.ndarray = None,
        )-> np.ndarray:
    """solve the QME. 
    In case qme_matrix=False (default) G represents the transition rates Gamma. 
    
    ```
    rho = QME(G, f, Delta=dot.Delta)
    ```
    In case qme_matrxi=True, the qme_matrix M=G, this is useful when scanning over 
    different frequencies but keep gamma constant
    
    ```
    M = QME_matrix(G, dot.Delta, )
    for i, f in enumerate(frequencies):
        rho[i] = QME(M, f, qme_matrix=True)
    ``` 
    
    Parameters
    ----------
    G: (np.ndarray)
        rates in case qme_matrxi is false, or the matrix M used in the equation rho = M rho (Hartree)
    frequency (float):
        frequency (GHz)
    qme_matrix: (bool):
        whether G=Gamma (False, default) or G=M (True).
    Delta (np.ndarray, optional):
        Energy diferences between states (Hartree). Needed for qme_matrix = False (default).
    """
    
    Ndim = G.shape[0]
    NFtotal = G.shape[-1]
    NF = int((NFtotal - 1) / 2)
    frequency *= 2*np.pi*time_unit # Hart

    if not qme_matrix:
        assert G.ndim == 5, f"G should be a rank 5 tensor with dimensions (Ndim, Ndim, Ndim, Ndim, 2*Nfour+1). Got G with shape {G.shape}."
        assert G.shape == (Ndim, Ndim, Ndim, Ndim, 2*NF+1), f"G should have shape (Ndim, Ndim, Ndim, Ndim, 2*Nfour+1), but got {G.shape}."
        assert Delta is not None, f'If you provide G=Gamma (qme_matrxi= False) you also need to provide Delta'
        assert Delta.shape == (Ndim, Ndim), f'Delta should have shape (Ndim, Ndim), but got {Delta.shape}'
        M = QME_matrix(G, Delta)

    else:
        assert G.ndim == 6, f"M should be a rank 5 tensor with dimensions (Ndim, Ndim, 2*Nfour+1 Ndim, Ndim, 2*Nfour+1). Got M with shape {G.shape}."
        assert G.shape == (Ndim, Ndim, 2*NF+1, Ndim, Ndim, 2*NF+1), f"M should have shape (Ndim, Ndim, 2*Nfour+1 Ndim, Ndim, 2*Nfour+1). Got M with shape {G.shape}."
        M = G.copy()

    M += QME_freq_cont(frequency, Ndim, NF)
    M = QME_constrstants(Ndim, NFtotal, M)
    Nmat = Ndim**2 * NFtotal
    M = M.reshape((Nmat, Nmat))

    # Solve it 
    B = np.zeros(( Ndim, Ndim, NFtotal,), dtype=complex)
    B[0,0, NF] = 1. # Set the trace condition: the matrix element corresponding to the identity operator is set to 1.
    B=B.reshape(Nmat)

    rho = np.linalg.solve(M, B,)
    #rho = solve(M, B, overwrite_a=True, overwrite_b=True) #scipy is slower for some reason

    rho = rho.reshape((Ndim, Ndim, NFtotal))
    return rho