import numpy as np

from .rates import sum_rates

def Bexch(GL, GR, theta, n=0):
    G = sum_rates(GL, GR)
    n += int((G.shape[-1]-1)/2)
    Bexch = np.imag(G[1,2,2,0,n] + G[1,3,3,0,n] +G[0,2,2,1,n] + G[0,3,3,1,n])/np.sin(theta)
    return Bexch

def Trel(GL, GR, n=0):
    G = sum_rates(GL, GR)
    n += int((G.shape[-1]-1)/2)
    trel = np.real( G[0,2,2,0,n] + G[0,3,3,0,n] 
        + G[1,2,2,1,n] + G[1,3,3,1,n])
    return trel


def Szacc(GL, GR, rho, Pl, theta, n=0):
    
    G = sum_rates(GL, GR)
    nmax = int((G.shape[-1]-1)/2)
    szacc = 0

    for m in range(max(n-nmax,-nmax), min(n+nmax,nmax)+1):
        G0 = 2 * GL[2,0,0,2, n-m+nmax] * Pl / (1 - Pl*np.cos(theta))
        G2 = 2 * GL[3,0,0,3, n-m+nmax] * Pl / (1 + Pl*np.cos(theta))
    
        G1 = GL[0,3,3,0, n-m+nmax] - GL[1,3,3,1, n-m+nmax] + GL[0,2,2,0, n-m+nmax] - GL[1,2,2,1,n-m+nmax]

        szacc +=  G0*rho[2,2,m+nmax]- G2*rho[3,3,m+nmax] - G1*(rho[2,2,m+nmax] +rho[3,3,m+nmax])/2
        if m == 0:
            szacc += G1/2

    return np.real(szacc)