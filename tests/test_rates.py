import unittest
import numpy as np

import sys
sys.path.append('/home/matyas/Programs/PyTimeESR-full')

from PyTimeESR import QD, rates, Electrode
from PyTimeESR.units import Hartree

from test_QD import TestQD


class TestRate(TestQD):

    def setUp(self):
        super().setUp()
        self.Electrode_cases = [
            {'label': 'Simple Electrode', 
             'id': 1,
            'args': [0., 0., 0., 2e-3, 2e-3, .5],
            'kwargs': {},
            'nmax' : 0
            },
            {'label': 'Static Electrode', 
             'id': 2,
            'args': [-30., 0., 0.5, 1e-4, 2e-3, .5],
            'kwargs': {},
            'nmax' : 0
            },
            {'label': 'A-driven Electrode',
             'id': 3,
             'args': [-30., 0., 0.5, 1e-4, 2e-3, .5],
            'kwargs': {'Adrive': .5},
            'nmax' : 2
            },
            {'label': 'Vrf-driven Electrode',
             'id': 4,
             'args': [-30., 5., 0.5, 1e-4, 2e-3, .5],
            'kwargs': {},
            'nmax' : 30
        }]
    
    def test_rate(self): 
        frequency = 20        

        for dot_case in self.QD_cases:
            Dot = QD(*dot_case['args'], **dot_case['kwargs'])
            qd_label = dot_case['label']
            qd_id = dot_case['id']
            for el_case in self.Electrode_cases:
                continue
                nmax = el_case['nmax']
                el = Electrode(*el_case['args'],**el_case['kwargs'])
                el_label = el_case['label']
                el_id = el_case['id']
                fn_ref = f'tests/data/rates-{el_id:d}-{qd_id:d}.dat'

                Gplus, Gminus = rates(Dot, el, frequency, nmax)
                G = (Gplus+Gminus)*Hartree
                Gref = load_ref_rates(fn_ref)
                tol = np.max(np.abs(Gref[:,-1],))*1e-2
                
                Gref = expand_rates(Gref, Dot.Nstate, nmax)
                np.testing.assert_allclose(np.abs(G), np.abs(Gref), atol=tol, rtol=1e-2,
                    err_msg=f'{qd_label} QD with {el_label} RATES do not match')


def expand_rates(G_red, N, NF):
    """Turn a sparse version of Gamma into a full tensor. 
    In the sparse version only nonzero components are pressent,
    each is of the form [u,v,i,j,n, Re(GL), Im(GL), Re(GR), Im(GR)]. 
    NOTE: it is assumed that only one Fourier component is being passed. 
    In the full tensor, has N*N*N*N componens of the form [Re(GL), Im(GL), Re(GR), Im(GR)] 

    Parameters: 
        G_red (np.array, (:, 9)): 
            Gamma in reduced form
        N (int):
            Number of levels in the system
    Returns:
        G (np.array, (N, N, N, N,4)):
            Gamma as a tensor 
    """

    G = np.zeros((N, N, N, N, 2*NF+1), dtype=np.complex128)
    G_red[:,:4] -= 1
    G_red[:,4] += NF
    for comp in G_red:
        G[tuple((comp[:5]).astype(int))] = comp[5] 
    return G 

def load_ref_rates(fn):
    a = np.loadtxt(fn, skiprows=1)
    idx= a[:,:5]
    El = a[:,5] + 1j*a[:,6]
    use = El !=0

    ref = np.column_stack((idx[use], El[use]))
    ref = ref[np.lexsort((ref[:,4], ref[:,3], ref[:,2],ref[:,1], ref[:,0]))]
    return ref


if __name__ == '__main__':
    unittest.main()
            

    
        