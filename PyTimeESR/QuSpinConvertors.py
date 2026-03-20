import numpy as np
from quspin.operators import hamiltonian

class QuSpinConvertors():
    coords = ['x', 'y', 'z']
    
    ### The following methods convert the old format of arrays to a Qspin notation. 
    # since we are using tensor basis each spin is spin site is indexed by 0, and the different
    # sites are indefied by their hilbert states, thus in 4 site hamiltonian xx interaction between 
    # sites 1 and 3 (indexing from 1) is 'x||x|', 
    # and a single spin intecation of 2nd spin in y direction '|y||' and so on@staticmethod

    def __init__(self):
        pass
    
    @staticmethod
    def calc_spin(basis, eigenstate, theta: float=0, phi: float=0, idx = None):
        Nstate = eigenstate.shape[1]
        Nspin = basis.N
        Nspin = sum(Nspin) if type(Nspin)  == tuple else Nspin
        if Nspin == 0: 
            # this can happen if we only consider the centeral spin 
            return 0.
        
        Si = np.zeros((Nspin, 3))       

        idx = np.arange(0,Nspin,1, dtype=int) if idx is None else idx
        Si[idx] = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        
        # generate operator
        operator = QuSpinConvertors.collect_single_spin_ineractions(Si)
        operator, _ = basis.expanded_form(operator,[])
        # TODO: figure how to use an operator not Hamiltonian
        operator = hamiltonian(operator, [], basis=basis, check_herm=False, check_symm=False, check_pcon=False)

        # calculate 
        S_expected = np.zeros(Nstate)
        for j in range(Nstate):
            S_expected[j] = np.real(operator.expt_value(eigenstate[:,j]))
        return S_expected

    
    @staticmethod
    def calc_spin_square(basis, eigenstate, idx = None):
        Nstate = eigenstate.shape[1]
        Nspin = basis.N
        Nspin = sum(Nspin) if type(Nspin)  == tuple else Nspin

        idx = np.arange(0,Nspin,1, dtype=int) if idx is None else idx
        
        SS = [[np.ones(3), i,i] for i in idx]
        
        # generate operator
        operator = QuSpinConvertors.collect_exchange_interactions(Nspin, SS)
        
        operator, _ = basis.expanded_form(operator,[])
        # TODO: figure how to use an operator not Hamiltonian
        operator = hamiltonian(operator, [], basis=basis, check_herm=False, check_symm=False, check_pcon=False)
        
        # calculate 
        SS_expected = np.zeros(Nstate)
        for j in range(Nstate):
            SS_expected[j] = np.real(operator.expt_value(eigenstate[:,j]))

        return SS_expected
    
    @staticmethod
    def collect_single_spin_ineractions(Hlocal):
        """Given Hlocal gives a list of strings of an coresponding single operator 
        Can be used for magnetic field in a hamiltonian or a spin operator. 

        Parameters
        ----------
        Hlocal (np.ndarray (Nspin, 3)):
            Filed applied to the each spin site in cartesian coordinates. 
            This can be a magnetic field or an operator to measure spin 
        
        Returns
        -------
        interactions (list):
            to be passed to quspin. format ['|x|||...', [[Jx,i]],....]
        """
        Nspin = Hlocal.shape[0]
        string_base = ['|' for _ in range(Nspin-1)] # this we use to identify hilbert spaces and directions
        interactions = [] # here we colloect interactions in cartesian basis

        for idx, H in enumerate(Hlocal):
            for coord, Hi in zip(QuSpinConvertors.coords, H):
                string = string_base.copy()
                string.insert(idx, coord)
                interactions.append([''.join(string), [[.5*Hi, 0]]])
        
        return interactions

    @staticmethod
    def spin2string(s):
        nom = np.round(2*s, 0).astype(int)
        return f'{nom:d}/2'

    @staticmethod
    def collect_exchange_interactions(Nspin, Jexch):
        """Given 
        """
        string_base = ['|' for _ in range(Nspin-1)] # this we use to identify hilbert spaces and directions
        interactions = []
        for J in Jexch: 
            # TODO: add anisotropic interactions
            indx = J[1:]
            indx.sort() # sorting to not fuck up the indicies
            # indx now identifies the to coordinates of our interactions. 
            for coord, Ji in zip(QuSpinConvertors.coords, J[0]): 
                string = string_base.copy()
                # looping over directions
                string.insert(indx[1], coord)
                string.insert(indx[0], coord)
                interactions.append([''.join(string), [[.25*Ji,0,0]]])
        return interactions 
    