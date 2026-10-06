import numpy as np

def Bexch(G, theta, n=0):
    """
    Calculate the exchange field Bexch for a spin-1/2 system from the rates G and angle theta.
    B_{xc} = \frac{2 \sum_{v=0,2} \Im{\Gamma_{evvg}}}{g\mu_B \sin \theta}
    """
    assert G.shape[:-1] == (4,4,4,4), "G must be a (4,4,4,4,Nfour) tensor corresponding to a spin-1/2 system got shape {}".format(G.shape)

    n += int((G.shape[-1]-1)/2)
    Bexch = np.imag(G[1,2,2,0,n] + G[1,3,3,0,n] +G[0,2,2,1,n] + G[0,3,3,1,n])/np.sin(theta)
    return Bexch

def Trel(G, n=0):
    """
    Calculate the relaxation rate Trel for a spin-1/2 system from the rates G.
    frac{1}{\tau_{rel}} = \frac{1}{\hbar} \sum_{v=0,2} \Re\{\Gamma^{0}_{gvvg} + \Gamma^{0}_{evve}\}
    """
    assert G.shape[:-1] == (4,4,4,4), "G must be a (4,4,4,4,Nfour) tensor corresponding to a spin-1/2 system got shape {}".format(G.shape)

    n += int((G.shape[-1]-1)/2)
    trel = np.real( G[0,2,2,0,n] + G[0,3,3,0,n] 
        + G[1,2,2,1,n] + G[1,3,3,1,n])
    return trel


def Szacc(G, rho, theta, n=0):
    """
    Calculate the accumulated spin polarization Sz for a spin-1/2 system from the rates G, density matrix rho, and angle theta.

    Sz_n = 2 \sum_{\alpha,m} P_\alpha [\Re \{\Gamma^{+0}_{0gg0,n-m}\}\rho_{00,m} - \Re\{\Gamma^{-0}_{2gg2,n-m}\}\rho_{22,m} + \Re\{\Gamma^{-0}_{g22g,n-m} - \Gamma^{+0}_{g00g,n-m}\}(1-\rho_{00,m}-\rho_{22,m})/2 ]
    """
    
    assert G.shape[:-1] == (4,4,4,4), "G must be a (4,4,4,4,Nfour) tensor corresponding to a spin-1/2 system got shape {}".format(G.shape)
    assert rho.shape[:-1] == (4,4), "rho must be a (4,4,Nfour) tensor corresponding to a spin-1/2 system got shape {}".format(rho.shape)
    assert G.shape[-1] == rho.shape[-1], "G and rho must have the same number of Fourier components got shapes {} and {}".format(G.shape, rho.shape)
    
    nmax = int((G.shape[-1]-1)/2)
    szacc = 0

    for m in range(max(n-nmax,-nmax), min(n+nmax,nmax)+1):
        # G{0ee0} - G{0gg0} = G0{0ee0}(1+P cos) - G{0gg0}(1-P cos) =  2 P G{0gg0} cos
        # G{2ee2} - G{2gg2} = G0{2ee2}(1-P cos) - G{2gg2}(1+P cos) = -2 P G{2gg2} cos
        G0 = (G[2,1,1,2,n-m+nmax] - G[2,0,0,2,n-m+nmax])/np.cos(theta)
        G2 = (G[3,1,1,3,n-m+nmax] - G[3,0,0,3,n-m+nmax])/np.cos(theta)
        
        # G{g22g} - G{e22e} = G0{g22g}(1+P cos) - G{e22e}(1-P cos) =  2 P G0{g22g} cos
        # G{g00g} - G{e00e} = G0{g00g}(1-P cos) - G{e00e}(1+P cos) = -2 P G0{g00g} cos
        G1 = ((G[0,3,3,0, n-m+nmax] - G[1,3,3,1, n-m+nmax]) + (G[0,2,2,0, n-m+nmax] - G[1,2,2,1,n-m+nmax]))/np.cos(theta)

        szacc +=  G0*rho[2,2,m+nmax] + G2*rho[3,3,m+nmax] - G1*(rho[2,2,m+nmax] +rho[3,3,m+nmax])/2

        # from the term (1-rho00-rho22)/2 the term 1 is added to the sum, but only for m=0, since 1 doesn't oscillate
        if m == 0:
            szacc += G1/2

    return np.real(szacc)