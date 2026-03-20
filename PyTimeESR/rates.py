import numpy as np
from scipy.special import jv

from .units import *
from .Electrode import Electrode

def Bessel_drive(z: float, Adrive: float, Jtol: float = 1e-4, vmax: int = int(1e6), verbose: str = False):
    """Find the Bessel functions J_v(z) for v = 0,1,...,vmax, and find the cutoff vbes such that the cumulative sum of |J_v(z)| converges, meaning no further value of J_v contributes to the sume singifincanltly. Then compute the convolution 
    
    Kbess_n = J_n + Adrive/2*(J_n+1 + J_n-1) for n = -vbes,...,vbes.
    
    Parameters
    ----------
    z : float
        The argument of the Bessel functions. z = eV_{RF}/ħω.
    Adrive : float
        The amplitude of the drive, 
    Jtol : float, optional
        The tolerance for J_v/J_vmax to consider it converged.
    vmax : int, optional
        The maximum order of the Bessel functions to compute. The default is 1e6.
    verbose : bool, optional
        Whether to print the details of the Bessel functions. The default is False.

    Returns
    -------
    vs : array
        The orders of the Bessel functions used in the convolution, from -vbes to vbes.
    Kbess : array
        The convolution of the Bessel functions with the drive, Kbess_n = J_n + Adrive/2*(J_n+1 + J_n-1) for n = -vbes,...,vbes.
    """
    vs = np.arange(0, vmax, 1, dtype=int)
    J = jv(vs, z)
    Jabs = np.abs(J)
    cumsum_Jabs = np.cumsum(Jabs)
    cumsum_Jabs = cumsum_Jabs/cumsum_Jabs[-1]
    
    above = cumsum_Jabs  > 1-Jtol 
    # the combined relative contribution of all J_{v>v_above} is less than Jtol, so we can consider it converged.
    if not np.any(above[:-1]):
        print(f"Warning: Bessel function cutoff not found for z = {z:.2f} and Jtol = {Jtol:.2e}. Consider increasing vmax.")
        print(f"The cummilative sum is {cumsum_Jabs[-1]:.2f}. Should be 1.0.")
        vbes = vmax+1
    
    for p in vs[::]:
        if above[p]:
            vbes = p+1
            break 

    vs = vs[:vbes+2]
    J = J[:vbes+2]
    
    if verbose:
        print(f"Bessel funcitons:")
        print(f"\tz = {z:.2f}")
        print(f"\tmax_v = {vbes}")
        print(f"\tJ_maxv = {J[-1]:.2f}")
        
    # add negative v's. J(-v) = (-1)^v*J(v)
    negativeJ = np.power(-1, vs)*J

    # zero padding for the convolution with cosine. 
    J = np.concatenate(([0],negativeJ[::-1][:-1], J,[0])) 
    vs = np.concatenate((-vs[::-1][:-1],vs)) 
    
    # convolution with cosine. Kbess_n = J_n + Adrive/2*(J_n+1 + J_n-1)
    Kbess = J[1:-1] + Adrive/2*(J[2:] + J[:-2])
    #print(Kbess[73:78], vbes, Kbess.shape)
    return vs, Kbess


def orbital_overlap(j,u, lamb, g):
    """ Calculate the projection of the orbital overlap onto the spin dependent coupling strength, which is the prefactor of the rate equations.
    Lvluj = \sum_s g * lambd_{vls} * conj(lamb{ujs})
    Ljuvl = \sum_s g * conj(lambd_{lvs}) * lamb{jus}

    where g is the spin dependent coupling strength, lamb is the orbital overlap, and the sum over s is over the electrode spin degree of freedom.

    Parameters
    ----------
    j,u : int
        The indicies of the final QD state.
    lamb : array
        The orbital overlap, with shape (Nstate, Nstate, 2*Norb), where Nstate is the dimension of the quantum dot Hilbert space and Norb is the number of orbitals.
    g : array
        The spin dependent coupling strength, with shape (2,), where g[0] is the coupling strength for spin up and g[1] is the coupling strength for spin down.

    Returns
    -------
    Lvluj, Ljuvl : array
        The projection of the orbital overlap onto the spin dependent coupling strength, which is the prefactor of the rate equations. Both have shape (Nstate, Nstate).
    """
    
    Nstate = lamb.shape[0]
    Lvluj = np.zeros((Nstate, Nstate), dtype=complex)
    Ljulv = np.zeros((Nstate, Nstate), dtype=complex)
    Norb = lamb.shape[2]

    # TODO: maybe a dot product can be used here to speed up the calculation.
    # but that is minor
    for orb in range(0,Norb,2):
        Lvluj += (lamb[:,:,orb]*np.conj(lamb[u,j,orb])*g[0] # spin up
                + lamb[:,:,orb+1]*np.conj(lamb[u,j,orb+1])*g[1]# spin down
                  ) 
        Ljulv += (lamb[j,u,orb]*np.conj(lamb[:,:,orb].T)*g[0] # spin up
                + lamb[j,u,orb+1]*np.conj(lamb[:,:,orb+1].T)*g[1]# spin down
                  )
    return Lvluj, Ljulv
      

def rates(QD, Electrode, frequency: float, Nfour: int, verbose: bool = False):
    """
    Gplus_{vlju} = \sum_s g * lambd_{vls} * conj(lamb{ujs}) * sum_p Ifplus(Ej - Eu - p * omega) * Kbess(p-n) * Kbess(p)
    Gminus_{vlju} = \sum_s g * conj(lambd_{lvs}) * lamb{jus} * sum_p Ifminus(Ej - Eu + p * omega) * Kbess(p+n) * Kbess(p)

    where g is the spin dependent coupling strength, lamb is the orbital overlap, Ifplus and Ifminus are the Fermi integrals, and Kbess is the convolution of the Bessel functions with the A driving. The sum over p is from -Nfour to Nfour, where Nfour is the cutoff for the Floquet sidebands. The sum over s is over the electrode spin degree of freedom.

    Parameters
    ----------
    QD : QD
        Object containing the quantum dot parameters, including the orbital overlap lamb, the energy differences Delta, and the coupling strength gammaC.
    Electrode : Electrode
        Object containing the electrode parameters, including the spin polarization, the drive amplitude Adrive, and the Fermi integral function.
    frequency : float
        The frequency of the drive, in GHz.
    Nfour : int
        The cutoff for the Floquet sidebands, meaning the number of sidebands to consider in the calculation. The sum over p is from -Nfour to Nfour.

    Returns
    -------
    Gplus, Gminus : array
        The rate matricies for hole (plus) and electron (minus) transport, with shape (Nstate, Nstate, Nstate, Nstate, 2*Nfour+1), where Nstate is the dimension of the quantum dot Hilbert space.
    """
    frequency *= 2*np.pi*time_unit # Hart
    
    g = 0.5*Electrode.g0*(1+Electrode.Spin_polarization*np.array([1,-1])) 
    
    # Time dependent drive.
    # check for 0/0 error
    z = 0 if Electrode.Vrf ==0 else Electrode.Vrf/frequency
    vs, Kbess = Bessel_drive(z, Electrode.Adrive, verbose=verbose)

    floq_energies = frequency*vs

    Nstate = QD.Nstate
    Gplus = np.zeros((Nstate, Nstate, Nstate, Nstate, 2*Nfour+1), dtype=complex)
    Gminus = np.zeros((Nstate, Nstate, Nstate, Nstate, 2*Nfour+1), dtype=complex)
    lamb = QD.lamb
    
    # ignore higher fourier components if the bessel functions vanish there
    active_four = int(min(Nfour, (vs.shape[0]-1)/2))

    for j in range(Nstate):
        joccup = QD.Occupancy == QD.Occupancy[j]
        for u in range(Nstate):
            uoccup = QD.Occupancy == QD.Occupancy[u]


            if np.all(lamb[u,j,:] == 0) and np.all(lamb[j,u,:] == 0):
                continue
            # the 

            Ifplus, Ifminus = Electrode.fermiIntegral(QD.Delta[u,j], floq_energies)
            Lvluj, Ljulv = orbital_overlap(j, u,lamb, g)
            # removing rates that do not contribute 
            #Ljulv[~uoccup] = 0. 
            #Lvluj[ :, ~uoccup] = 0. 
            #Lvluj[~joccup] = 0. 
            #Ljulv[ :, ~joccup] = 0. 
            
            #print(j, u, QD.Delta[u,j], Ifplus)
            #print(j, u, QD.Delta[u,j], Ifminus)
            #print()
            
    #        print(j+1,  u+1, Ifplus, Ifminus)

            #idx1 = np.array(np.where(Lvluj)).T
            #idx2 = np.array(np.where(Ljuvl)).T

            #for a in idx1: 
            #    print('vluj ', *(a+1), u+1, j+1, np.round(Lvluj[tuple(a)],12))
            #for a in idx2: 
            #    print('juvl ', j+1, u+1, *(a[::-1]+1), np.round(Ljuvl[tuple(a)], 12))
            
            for n in range(-active_four, active_four+1):
                # Ensure bounds are respected for Kbess and Ifplus/Ifminus
                if n >= 0:
                    FouirC_plus = np.sum(np.conj(Kbess[:len(Kbess)-n]) * Kbess[n:] * Ifplus[:len(Kbess)-n])
                    FouirC_minus = np.sum(np.conj(Kbess[:len(Kbess)-n]) * Kbess[n:] * Ifminus[:len(Kbess)-n])
                else:
                    n_abs = abs(n)
                    FouirC_plus = np.sum(np.conj(Kbess[n_abs:]) * Kbess[:len(Kbess)-n_abs] * Ifplus[n_abs:])
                    FouirC_minus = np.sum(np.conj(Kbess[n_abs:]) * Kbess[:len(Kbess)-n_abs] * Ifminus[n_abs:])
                
                #print(n, nidx,Kbess[::-1][:n-1],Kbess[n:])
                #if n == 0:  
                #    # annoying if n=0 [:-n] becomes empty
                #    FouirC_plus  = np.sum(np.conj(Kbess[::-1]) * Kbess * Ifplus)
                #    FouirC_minus = np.sum(np.conj(Kbess[::-1]) * Kbess * Ifminus)
                #else:
                #    FouirC_minus = np.sum(np.conj(Kbess[:n:-1]) * Kbess[n:] * Ifminus[:n:-1])
                #    FouirC_plus  = np.sum(np.conj(Kbess[:n:-1]) * Kbess[n:] *  Ifplus[n:])

                nidx = n + Nfour
                Gplus[:,:,j,u,nidx] = .5*Lvluj*FouirC_plus
                Gminus[:,:,j,u,nidx] = .5*Ljulv*FouirC_minus
    
    return Gplus, Gminus
                

def sum_rates(GL, GR):
    NFL = int((GL.shape[-1]-1)/2)
    NFR = int((GR.shape[-1]-1)/2)
    
    if NFL>=NFR: 
        G = GL.copy()
        G[:,:,:,:,NFL-NFR:NFR-NFL] += GR
    else: 
        G = GR.copy()
        G[:,:,:,:,NFL-NFR:NFL-NFR] += GL
    return G
