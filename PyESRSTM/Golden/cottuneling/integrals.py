import numpy as np 
from ...misc.units import *

def g1(x, D, gammaC):
    return 1/(x-D+1j*gammaC)

def time_int(f, Vrf, n_int_four=500, n=0):
    t = np.linspace(-np.pi, np.pi, n_int_four)

    amplitude = f(Vrf*np.cos(t))
    phase = np.exp(1j * n * t)
    # align phase shape with amplitude for broadcasting
    phase = phase.reshape((t.size,) + (1,)*(amplitude.ndim-1))

    return np.trapezoid(amplitude*phase, t, axis=0)/(2*np.pi)

def ener_int(f, E_cut, n_int_ener=1000,):
    e = np.linspace(-E_cut, E_cut, n_int_ener)
    return np.trapezoid(f(e), e, axis=0)

def all_integrals(Left, Right, frequency, Delta, gammaC, n_four=0, n_int_four=500, n_int_ener=1000, verbose=False):

    Ndim = Delta.shape[0]
    assert Ndim == 4, "Only implemented for 2-level systems"

    frequency *= 2*np.pi*time_unit # Hart
    
    # [L/R,L/R, ud/du, Ndim, Ndim,] ]
    fermiR = lambda x: Right.FermiFunctionArray((x-Right.Vdc)/Right.Temperature)
    fermiL = lambda x: Left.FermiFunctionArray((x-Left.Vdc)/Left.Temperature)
    fermiRh = lambda x: 1-fermiR(x)
    fermiLh = lambda x: 1-fermiL(x)

    dosR = Right.dos(0)
    dosL = Left.dos(0)
    Vrf = Left.Vrf

    i = 0
    j = (i+1)  % 2
    l_idx = np.array([2,3])
    
    def F1RR(D, s):
        # rho_R**2 sum_l fermiR(D_lj)*fermiR(D_lj -/+ D_ij)
        ##print(1)
        return (np.pi/gammaC)**2 * dosR**2 * np.sum(fermiR(D)*fermiRh(D - s*Delta[i,j]))

    def F1RL(D, s):
        ##print(2)
        # rho_R * rho_L sum_l fermiR(D_lj)*int dt fermiL(D_lj -/+ D_ij -/+ Vrf cos(t))
        fL = lambda y: fermiLh(D[None,:] - s*Delta[i,j] - s*y[:,None])
        return (np.pi/gammaC)**2 * dosR*dosL * np.sum(fermiR(D) * time_int(fL, Vrf))

    def F1LRp(D):
        ##print(3)
        # rho_R * rho_L sum_l fermiR(D_lj + D_ij)*int dt fermiL(D_lj - Vrf cos(t))
        fL = lambda y: fermiL(D[None, :] - y[:,None])
        return (np.pi/gammaC)**2 * dosR*dosL * np.sum(fermiRh(D+Delta[i,j]) * time_int(fL, Vrf))

    def F1LRm(D):
        ##print(4)
        # rho_R * rho_L sum_l int dt fermiR(D_lj - D_ij - 2 Vrf cos(t))*fermiL(D_lj - Vrf cos(t))
        fLfR = lambda y: fermiL(D[None, :] - y[:,None])*fermiRh(D[None, :] - Delta[i,j] - 2*y[:,None])
        return (np.pi/gammaC)**2 * dosR*dosL * np.sum(time_int(fLfR, Vrf))

    def FLL(D1, D2, s):

        res = 0
        for n in range(-n_four, n_four+1):
            glj  = lambda x: time_int(lambda y: g1(x[None,:,None] + s*Delta[i,j] + s*n*frequency + y[:,None,None], D1[None,None,:], gammaC), Vrf, n= n, n_int_four=n_int_four)

            gli = lambda x: time_int(lambda y: g1(x[None,:,None] - s*Delta[i,j] - s*n*frequency + y[:,None,None], D2[None,None,:], gammaC), Vrf, n= n, n_int_four=n_int_four)

            fL = lambda x: fermiL(x+s*Delta[i,j]+s*n*frequency)

            ftot = lambda x: np.sum(np.conj(gli(x)*glj(x)),axis=1) * fL(x) * fermiLh(x)
            res +=  np.sum(ener_int(ftot, Left.Cutoff, n_int_ener=n_int_ener))

        return np.pi/gammaC * dosL**2 * res

    F1p = np.array([
        F1RR(Delta[l_idx,j], 1) + F1RR(Delta[i,l_idx], 1),
        F1RL(Delta[l_idx,j], 1) + F1RL(Delta[i,l_idx], 1),
        F1LRp(Delta[l_idx,j]) + F1LRp(Delta[i,l_idx]),
        FLL(Delta[l_idx,j],Delta[l_idx,j], 1) + FLL(Delta[i,l_idx],Delta[l_idx,j], 1)
    ])
    F1m = np.array([
        F1RR(Delta[l_idx,j], -1) + F1RR(Delta[i,l_idx], -1),
        F1RL(Delta[l_idx,j], -1) + F1RL(Delta[i,l_idx], -1),
        F1LRm(Delta[l_idx,j]) + F1LRm(Delta[i,l_idx]),
        FLL(Delta[l_idx,j],Delta[l_idx,j], -1) + FLL(Delta[i,l_idx],Delta[l_idx,j], -1)
    ])

    #def F2RR(s):
    #    f = lambda x: g1(x, s*Delta[None, l_idx, i]) * g1(x, s*Delta[None, l_idx, i])*fermiR(x) 
    #    return dosR**2 * np.sum(ener_int)
    def F2RR(s):
        # rho_R**2 sum_l fermiR(D_lj)*fermiR(D_lj -/+ D_ij)
        ##print(1)
        return (np.pi/gammaC)**2 * dosR**2 * np.sum(fermiR(s*Delta[l_idx,j] + s*Delta[i,j])*fermiRh(s*Delta[l_idx,j]))

    def F2RL(D1, D2, s):        

        ftot = lambda x: fermiLh(x)[:,None] * time_int(lambda y:
            g1(x[None,:,None] + s*Delta[i,j] + s*y[:,None,None], D1[None, None, :], gammaC)
            * g1(x[None,:,None] + s*y[:,None,None], D2[None, None, :], gammaC)
            * fermiR(x[None, :, None] + s*Delta[i,j]+s*y[:,None,None]),
            Vrf, n_int_four=n_int_four)

        res = np.sum(ener_int(ftot, Left.Cutoff, n_int_ener=n_int_ener))
        return np.pi/gammaC * dosR*dosL * res

    def F2LR(D1, D2, s):
        #glj = lambda x: g1(x[:, :, None], D1[None, None, :], gammaC)
        #gli = lambda x: g1(x[:, :, None] - s*Delta[i,j], D2[None, None, :], gammaC)
        #fR = lambda x: fermiR(x )
        #fL = lambda x: fermiLh(x)
        

        ftot = lambda x: fermiLh(x)[:,None] * time_int(lambda y: 
            g1(x[None, :, None] + y[:,None,None], D1[None, None, :], gammaC)
            * g1(x[None, :, None] - s*Delta[i,j]- s*y[:,None,None], D2[None, None, :], gammaC)
            * fermiR(x[None, :, None] - s*y[:,None,None]- s*Delta[i,j]),
            Vrf, n_int_four=n_int_four)

        res = np.sum(ener_int(ftot, Left.Cutoff, n_int_ener=n_int_ener))
        return np.pi/gammaC * dosR*dosL * res

    F2 = np.array([
        np.sum(F2RR(1) + F2RR(-1)),
        np.sum(F2RL(Delta[l_idx,j], Delta[l_idx, i], 1)
               + F2RL(Delta[j, l_idx], Delta[i, l_idx], -1)),
        np.sum(F2LR(Delta[l_idx,j], Delta[l_idx, i], 1)
               + F2LR(Delta[j, l_idx], Delta[i, l_idx], -1)), 
        np.sum(FLL(Delta[l_idx,j], Delta[l_idx, i], 1)
               + FLL(Delta[j, l_idx], Delta[i, l_idx], -1))])
       
    return F1p, F1m, F2
 

            
                    

    