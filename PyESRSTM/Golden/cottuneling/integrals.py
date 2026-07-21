import numpy as np 


def g1(x, D, gammaC):
    return 1/(x-D+1j*gammaC)


def all_integrals(Left, Right, Delta, n_int_four=1000, n_int_ener=1000, integral_width=10, verbose=False):

    Ndim = Delta.shape[0]
    assert Ndim == 4, "Only implemented for 2-level systems"
    
    # [L/R,L/R, ud/du, Ndim, Ndim,] ]
    T1 = np.zeros((2,2,2, Ndim, Ndim), dtype=complex)
    T2p = np.zeros((2,2,2, Ndim, Ndim), dtype=complex)
    T2m = np.zeros((2,2,2, Ndim, Ndim), dtype=complex)

    for i in range(0,1):
        j = i // 2
        for l in range(2,3):
            # grid for the integrals
            x_il = np.linspace(-integral_width, integral_width, n_int_ener) + Delta[i,l]
            x_li = np.linspace(-integral_width, integral_width, n_int_ener) + Delta[l,i]
            
            g1(x_kl)*g1()


            frp = 
            I10 = 

            
                    

    