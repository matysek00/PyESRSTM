import unittest
import numpy as np

import sys
sys.path.append('/home/matyas/Programs/PyTimeESR-full')

from PyTimeESR import QD
from PyTimeESR.units import *

class TestQD(unittest.TestCase):

    def setUp(self):
        # Prepare different test cases as a list of dictionaries
        self.QD_cases = [{   
            'label': 'Single Spin',
            'id': 1,
            'args': [-10, 100, np.array([0.5]), np.array([[0.60827625, 0.0, 0.0]]), np.array([[2.0, 2.0, 2.0]])],
            'kwargs': {},
            'results': {'Nspin': 1,'Nstate': 4, 'Occupancy': np.array([1, 1, 0, 2]), 'Energies': np.array([0.0, 17.03, 2427.0, 0.2177E+05]), 'Spin': np.array([  [[1, 1, -0.5000, 0.0, 0.0]], [[2, 1, 0.5000, 0.0, 0.0]], [[3, 1, 0.0, 0.0, 0.0]], [[4, 1, 0.0, 0.0, 0.0]],]), 'Spin_Sq': None, 'lambda': np.array([[0, 3, 0,-0.707107],[0, 3, 1, 0.707107],[1, 3, 0, 0.707107],[1, 3, 1, 0.707107],[2, 0, 0, 0.707107],[2, 0, 1,-0.707107],[2, 1, 0, 0.707107],[2, 1, 1, 0.707107]], dtype=complex)}
            },
            {'label': 'Two Spins Exchange only', # a tiny magnetic field is included to break state degeneracies
            'id': 2,
            'args': [-5, 100, np.array([.5, .5]), np.array([[1e-5, 0.0, 0.0],[1e-5, 0.0, 0.0]]), np.array([[2.0,2.0,2.0], [2.0,2.0,2.0]])],
            'kwargs': {'Jexch': [[[-5.0, -5.0, -5.0 ], 0,1]]},
            'results': {'Nspin': 2,'Nstate': 8, 'Occupancy': np.array([1, 1, 1, 1, 0, 0, 2, 2]), 'Energies': np.array([0.0000000000000000, 2.7992489927394451E-004, 5.5984979765617566E-004, 5.0002799248987309, 1210.2447610452093, 1210.2450409701078, 22972.147940534884, 22972.148220459785]), 'Spin': np.array([[[ 1,  1, -5.000e-01,  0, 0], [ 1,  2, -5.000e-01,  0, 0]],[[ 2,  1,  0,  0, 0], [ 2,  2, 0,  0, 0]],[[ 3,  1,  5.000e-01,  0, 0], [ 3,  2,  5.000e-01,  0, 0]],[[ 4,  1, 0,  0, 0], [ 4,  2,  0,  0, 0]],[[ 5,  1,  0.000e+00,  0, 0], [ 5,  2, -5.000e-01,  0, 0]],[[ 6,  1,  0.000e+00,  0, 0], [ 6,  2,  5.000e-01,  0, 0]],[[ 7,  1,  0.000e+00,  0, 0], [ 7,  2, -5.000e-01,  0, 0]],[[ 8,  1,  0.000e+00,  0, 0], [ 8,  2,  5.000e-01,  0, 0]]]), 'Spin_Sq': None, 'lambda': np.array([[0, 6, 0,  0.70710678137014527], [0, 6, 1, -0.70710678100294944], [1, 6, 0, -0.50001399579851713], [1, 6, 1, -0.50001399631778853], [1, 7, 0, -0.49998600302615626], [1, 7, 1,  0.49998600407395694], [2, 7, 0, -0.70710678155699302], [2, 7, 1, -0.70710678081610157], [3, 6, 0,  0.49998600355005302], [3, 6, 1,  0.49998600355006029], [3, 7, 0, -0.50001399605814922], [3, 7, 1,  0.50001399605815655], [4, 0, 0, -0.70710678100294944], [4, 0, 1,  0.70710678137014527], [4, 1, 0, -0.50001399631778853], [4, 1, 1, -0.50001399579851713], [4, 3, 0,  0.49998600355006029], [4, 3, 1,  0.49998600355005302], [5, 1, 0,  0.49998600407395694], [5, 1, 1, -0.49998600302615626], [5, 2, 0, -0.70710678081610157], [5, 2, 1, -0.70710678155699302], [5, 3, 0,  0.50001399605815655], [5, 3, 1, -0.50001399605814922]], dtype=complex) }
            },
            {'label': 'Two Spins Exachange and Bfield',
             'id': 3,
            'args': [-5, 100, np.array([.5, .5]), np.array([[.5, 0.0, 0.0],[0.5, 0.0, 0.0]]), np.array([[2.0,2.0,2.0], [2.0,2.0,2.0]])],
            'kwargs': {'Jexch': [[[-5.0, -5.0, -5.0 ], 0,1]]},
            'results': {'Nspin': 2,'Nstate': 8, 'Occupancy': np.array([1, 1, 1, 1, 0, 0, 2, 2]), 'Energies': np.array([0., 14.00, 19.00, 27.99, 1217., 1231., 0.2298E+05, 0.2299E+05,]), 'Spin': np.array([[[ 1,  1, -.5,  0, 0],  [ 1,  2, -.5,  0, 0]], [[ 2,  1,  0.,  0, 0],  [ 2,  2,  0.,  0, 0]], [[ 3,  1,  0.,  0, 0],  [ 3,  2,  0.,  0, 0]], [[ 4,  1,  .5,  0, 0],  [ 4,  2,  .5,  0, 0]], [[ 5,  1,  .0,  0, 0],  [ 5,  2, -.5,  0, 0]], [[ 6,  1,  .0,  0, 0],  [ 6,  2,  .5,  0, 0]], [[ 7,  1,  .0,  0, 0],  [ 7,  2, -.5,  0, 0]], [[ 8,  1,  .0,  0, 0],  [ 8,  2, .5,  0, 0]]]), 'Spin_Sq': None, 'lambda': np.array([[0, 6, 0, 0.70710678118655124],[0, 6, 1, -0.70710678118654346],[1, 6, 0, 0.49999999999996775],[1, 6, 1, 0.49999999999995359],[1, 7, 0, 0.50000000000003564],[1, 7, 1, -0.50000000000004297],[2, 6, 0, -0.50000000000002665],[2, 6, 1, -0.50000000000005185],[2, 7, 0, 0.49999999999994355],[2, 7, 1, -0.49999999999997791],[3, 7, 0, 0.70710678118656212],[3, 7, 1, 0.70710678118653281],[4, 0, 0, -0.70710678118654346],[4, 0, 1, 0.70710678118655124],[4, 1, 0, 0.49999999999995359],[4, 1, 1, 0.49999999999996775],[4, 2, 0, -0.50000000000005185],[4, 2, 1, -0.50000000000002665],[5, 1, 0, -0.50000000000004297],[5, 1, 1, 0.50000000000003564],[5, 2, 0, -0.49999999999997791],[5, 2, 1, 0.49999999999994355],[5, 3, 0, 0.70710678118653281],[5, 3, 1, 0.70710678118656212]], dtype=complex)}
            },
            # Add more test cases as needed
        ]

    def test_QD(self):
        for case in self.QD_cases:
            with self.subTest(case=case):
                continue
                Dot = QD(*case['args'], **case['kwargs'])
                label = case['label']
                

                self.assertEqual(Dot.Nspin, case['results']['Nspin'], f'QD case of {label} number of spins does not match')
                self.assertEqual(Dot.Nstate, case['results']['Nstate'], f'QD case of {label} number of states does not match')

                np.testing.assert_array_equal(Dot.Occupancy, case['results']['Occupancy'], f'QD case of {label} OCCUPANCIES does not match')
                
                Energies = Dot.Energies*Hartree/GHz
                Energies -= Energies[0]
                np.testing.assert_allclose(Energies, case['results']['Energies'], rtol=1e-3, err_msg=f'QD case of {label} ENERGIES not match')
                np.testing.assert_allclose(Dot.CalcAllSpin(), case['results']['Spin'], atol= 1e-6, err_msg=f'QD case of {label} SPINS do not match')

                lamb = Dot.reduce_lamb()
                np.testing.assert_allclose(np.abs(lamb), np.abs(case['results']['lambda']), atol=1e-4, err_msg=f'QD case of {label} LAMBDA does not match')

if __name__ == '__main__':
    unittest.main()