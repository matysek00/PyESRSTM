import unittest
import numpy as np

import sys
sys.path.append('/home/matyas/Programs/PyTimeESR-full')

from PyTimeESR import QD, rates, Electrode, QME
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
        el_simple = self.Electrode_cases[0]
        el_simple = Electrode(*el_simple['args'],**el_simple['kwargs'])

        for dot_case in self.QD_cases:
            qd_label = dot_case['label']
            qd_id = dot_case['id']

            Dot = QD(*dot_case['args'], **dot_case['kwargs'])
            GLplus, GLminus = rates(Dot, el_simple, frequency, 0)
            GL = (GLplus+GLminus)

            for el_case in self.Electrode_cases[1:]:
                nmax = el_case['nmax']
                el_label = el_case['label']
                el_id = el_case['id']
                fn_ref = f'tests/data/coherence-{el_id:d}-{qd_id:d}.dat'
                
                el = Electrode(*el_case['args'],**el_case['kwargs'])
                GRplus, GRminus = rates(Dot, el, frequency, nmax)
                GR = (GRplus+GRminus)
                rho = QME(GR+GL, frequency, Delta=Dot.Delta, )

                rho_ref = load_ref_coh(fn_ref)
                
                rho_ref = expand_coh(rho_ref, Dot.Nstate, nmax)
                np.testing.assert_allclose(np.abs(rho), np.abs(rho_ref), atol=1e-2, rtol=1e-2,
                    err_msg=f'{qd_label} QD with {el_label} COHERENCES do not match')


def expand_coh(rho_red, N, NF):
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

    rho = np.zeros((N, N, 2*NF+1), dtype=np.complex128)
    rho_red = rho_red[np.abs(rho_red[:,2])<NF+.5]
    rho_red[:,:2] -= 1
    rho_red[:,2] += NF
    
    for comp in rho_red:
        rho[tuple((comp[:3]).astype(int))] = comp[3] 
    return rho 


def load_ref_coh(fn):
    a = np.loadtxt(fn, skiprows=1)
    idx= a[:,:3]
    El = a[:,3] + 1j*a[:,4]
    use = El !=0

    ref = np.column_stack((idx[use], El[use]))
    ref = ref[np.lexsort((ref[:,2],ref[:,1], ref[:,0]))]
    return ref    



if __name__ == '__main__':
    unittest.main()