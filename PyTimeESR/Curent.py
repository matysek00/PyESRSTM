import numpy as np 
from .units import pA 
def current(rho, G):
    NFT  = rho.shape[-1]
    Ndim = rho.shape[0]
    NF = int((NFT -1)/2)
    I = np.zeros(NFT, dtype=complex)
    for n in range(-NF, NF+1): 
        #I[n] = rho(u,l,-m-n) * G(l,j,j,u,m) + rho*(u,l,n-m) * G*(l,j,j,u,m)
        
        #    #
        #for l in range(Ndim): 
        #    for u in range(Ndim): 
        #        for j in range(Ndim): 
        for m in range(max(-NF-n,-NF), min(NF+1, NF-n+1)):
            #I[n+NF] += rho[l,u,NF-n-m]*G[l,j,j,u, NF+m]
            I[n+NF] += np.einsum('ul,ljju', rho[:,:,NF-n-m], G[:,:,:,:, NF+m])
        for m in range(max(-NF+n,-NF), min(NF+1, NF+n+1)):
            I[n+NF] += np.conj(np.einsum('ul,ljju', rho[:,:,NF+n-m], G[:,:,:,:, NF+m]))
                        #I[n+NF] += np.conj(rho[l,u,NF+n-m]*G[l,j,j,u, NF+m])
    return I*pA         