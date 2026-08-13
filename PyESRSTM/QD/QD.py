import numpy as np
from quspin.basis import spin_basis_1d, tensor_basis
from quspin.operators import hamiltonian
from ..misc.units import * 


from .QuSpinConvertors import QuSpinConvertors

class QD(QuSpinConvertors):
    """Class for a quantum dot with a central spin and N-1 other spins.
    The Hamiltonian is given by:
    H = \sum_i \mu_B B \cdot S_i + \sum_{i<j} J_{ij} S_i \cdot S_j, 
    B is the magnetic field, S_i is the effective spin of the i-th atom, and J_{ij} is the exchange interaction between the i-th and j-th atom.
    The effective spin of the i-th atom is given by S_i = g_i \cdot S_i, where g_i is the g-tensor of the i-th atom.

    Parameters
    ----------
    eps : float
        The energy of the quantum dot [meV].
    U : float
        The charging energy of the quantum dot [meV].
    Spin : array_like(Nspin,)
        The  spin of the atoms in the quantum dot. The first element is the central spin, and the rest are the other spins.
        First element must be 1/2. The rest can be any half integer. 
    Hlocal : array_like(Nspin, 3)
        The local magnetic field on each spin in the quantum dot [T]. 
    Gyro : array_like(Nspin, 3) or (Nspin, 3, 3)
        The g-tensor of the atoms in the quantum dot. Should be shape (Nspin, 3) for diagonal g-tensors, or (Nspin, 3, 3) for full g-tensors.
    Jexch : list of tuples (array_like(3,) or array_like(3,3), int, int)
        The exchange interaction between the atoms in the quantum dot [GHz]. Each tuple contains the exchange interaction tensor, and the indices of the two atoms. 
        The exchange interaction tensor should be shape (3,) for diagonal exchange interactions, or (3, 3) for full exchange interactions.
    Stephen : array_like(Nspin, 4)
        The Stephen tensor of the atoms in the quantum dot. Should be shape (Nspin, 4). If None, it will be set to zero.
    StephenAx : array_like(Nspin, 3)
        The principal axes of the Stephen tensor of the atoms in the quantum dot. Should be shape (Nspin, 3). If None, it will be set to the z-axis.
    cuttof_energy : float
        The cutoff energy for the quantum dot. States with energy above this value will be ignored. [meV]
    gyro_to_J : bool
        If true, the exchange interaction will be transformed with the g-tensor. If False, it will be used as is. This is useful if the exchange interaction is already in the g-tensor basis.
        Default is True.
    """

    def __init__(self, eps: float, U: float, Spin: np.ndarray,  Hlocal: np.ndarray, Gyro: np.ndarray, Jexch: list = [], Stephen: np.ndarray=None, StephenAx: np.ndarray = None, cuttof_energy: float = np.inf, gyro_to_J: bool = True):
        
        Spin = np.array(Spin, dtype=float)
        Hlocal = np.array(Hlocal, dtype=float)
        Gyro = np.array(Gyro, dtype=float)

        assert Spin.ndim == 1, f'Spin shold be shape (Nspin,) got {Spin.shape}'
        Nspin = len(Spin)
        assert np.isclose(Spin[0],.5) , f'The central spin must be 1/2 got{Spin[0]}'

        if Stephen is not None or StephenAx is not None: 
            print('WARNING: Seems like you tried specified Stephen operatros. Since the author of this code is too stupid to implement them they will be IGNORED.')
            print('On the plus site they will stive lead to an error if you dared to use the wrong foramt.')

        Stephen = np.zeros((Nspin,4)) if Stephen is None else np.array(Stephen, dtype=float)
        if StephenAx is None:
            StephenAx = np.zeros((Nspin, 3))
            StephenAx[:,2] = 1

        assert Hlocal.shape == (Nspin, 3), f'Hlocal should be shape (Nspin, 3), where Nspin={Nspin}, got {Hlocal.shape}'
        assert Gyro.shape == (Nspin, 3) or Gyro.shape == (Nspin, 3, 3), f'Gyro should be shape (Nspin, 3), where Nspin={Nspin}, got {Gyro.shape}'
        assert StephenAx.shape == (Nspin, 3), f'nAx should be shape (Nspin, 3), where Nspin={Nspin}, got {StephenAx.shape}'
        assert Stephen.shape == (Nspin, 4), f'Stephen should be shape (Nspin, 4), where Nspin={Nspin}, got {Stephen.shape}'
        
        # expand Gyro to 3x3 if it is only a diagonal matrix
        Gyro_exp = np.empty((Nspin, 3, 3))
        if Gyro.ndim == 2:
            for i in range(Nspin):
                Gyro_exp[i] = np.diag(Gyro[i])
        else:
            Gyro_exp = Gyro
        Gyro = Gyro_exp

        #Jexch_mat = np.zeros((Nspin, Nspin, 3)) if Jexch_mat == None else Jexch_mat
        #assert Jexch_mat.shape == (Nspin, Nspin, 3), f'Jexch_mat should be shape (Nspin,  Nspin, 3), where Nspin={Nspin}, got {Jexch_mat.shape}'
        for J in Jexch: 
            # as it is now inputing positive J leads to atractive interaction
            Jexhc_err_mesage = f'Jexch must be of the followin form [[np.array([Jx,Jy,Jz],i,j], ...] or [[np.array([[Jxx,Jxy,Jxz],[..,],...],i,j], ...]  got {Jexch}'
            assert isinstance(J[0], np.ndarray) or isinstance(J[0], list), Jexhc_err_mesage
            if isinstance(J[0], list):
                J[0] = np.array(J[0], dtype=float)
            
            assert J[0].shape == (3,) or J[0].shape == (3,3), Jexhc_err_mesage
            
            J[1], J[2] = int(J[1]), int(J[2])

            if J[0].shape == (3,):
                J[0] = np.diag(J[0])
            
            if gyro_to_J:
                J[0] = np.einsum('ki, lj, kl -> ij', Gyro[J[1]], Gyro[J[2]], J[0])
            
            J[0] = J[0]*GHz/Hartree
            

        # Unit converion 
        for i in range(Nspin):
            Hlocal[i] = np.dot(Gyro[i], Hlocal[i])*BohrMagneton*1000/Hartree
        
        Stephen /= Hartree
        eps /= Hartree
        U /= Hartree
        
        self.eps = eps
        self.U = U
        self.theta = np.arctan(np.linalg.norm(Hlocal[0,:2])/Hlocal[0,2])
        self.Nspin = Nspin
        self.cutoff_energy = cuttof_energy

        # Diagonalize the Hamiltonian
        self.setup_ham(Spin, Jexch, Hlocal)   

    def remove_states(self, states: np.array):
        useful = np.ones(self.Nstate, dtype=bool)
        useful[states] = False

        self.Nstate = int(sum(useful))
        self.Energies = self.Energies[useful]
        self.Occupancy = self.Occupancy[useful]
        
        self.Spin_central = self.Spin_central[useful]
        self.Spin_total = self.Spin_total[useful]

        self.lamb = self.lamb[np.ix_(useful, useful, [0, 1])]
        self.Delta = self.Delta[np.ix_(useful, useful)]

    def setup_ham(self, Spin, Jexch, Hlocal):
        # first solve including all spins (asume the central atoms has a finite spin)
        basis_neutral, E_neutral, V_neutral = QD.solve_spin_ham(self.Nspin, Spin, Jexch, Hlocal)
        Nm = int(basis_neutral.Ns/2)
        # WARNINNG HARDCODING
        #V_neutral[:,0] = -V_neutral[:,0]
        
        # now assume a charged stat (0,2 occupancy) meaning the first spin will not be included
        if self.Nspin > 1:
            Jexch_charged = QD.remove_central_exch(Jexch) # remove all exchange interaction with the transport atom
            basis_charged, E_charged, V_charged = QD.solve_spin_ham(self.Nspin-1, Spin[1:], Jexch_charged, Hlocal[1:])
            assert basis_charged.Ns == Nm, 'Basis sizes don\'t match something is off.'
        else:
            # only 1 spin exists
            basis_charged = None
            E_charged = np.array([0])
            V_charged = np.array([[1]])
            assert Nm == 1,  'Basis sizes don\'t match something is off.'

        Energies = np.concatenate((E_neutral + self.eps, E_charged, E_charged + self.U + 2*self.eps))
        Nstate = len(Energies)
        assert Nstate == 4*Nm, 'Basis sizes don\'t match the number of states.'

        # now we calculate lambda
        lamb = np.zeros((Nstate, Nstate,2), dtype=complex)        
        for i in range(Nm): # charged
            for j in range(2*Nm): # singly occupied
                # states
                for m in range(Nm):
                    # the basis is the same for charged and neutral but for 
                    # neutral we have extra spin degree of freeedon
                    # 0 to 1
                    lamb[i+2*Nm,j,0] += np.conj(V_charged[m,i])*V_neutral[m,    j] # spin down
                    lamb[i+2*Nm,j,1] += np.conj(V_charged[m,i])*V_neutral[m+Nm, j] # spin_up
                    #lamb[i+2*Nm,j,0] += np.conj(V_charged[m,i])*V_neutral[m,    j] # spin down
                    #lamb[i+2*Nm,j,1] += np.conj(V_charged[m,i])*V_neutral[m,    j] # spin_up
                    # +2*Nm moves to unocupied sub space, +3 nm to doubly occupied
                    # +Nm flips the central spin 
                    
                    # 2 to 1
                    lamb[j, i+3*Nm,0] += np.conj(V_neutral[m+Nm, j])*V_charged[m,i] # spin down
                    lamb[j, i+3*Nm,1] += np.conj(V_neutral[m,    j])*V_charged[m,i] # spin up
        
        # calculate expected spin on the central site and total spin
        S_central = np.zeros((2*Nm, 3))
        S_total_neutral = np.zeros((2*Nm, 3))
        S_total_charged = np.zeros((Nm, 3))

        for i, theta, phi in zip([0,1,2], [np.pi/2, np.pi/2 ,0], [0, np.pi/2, 0]):  
            # looping x y z
            S_central[:,i] = QD.calc_spin(basis_neutral, V_neutral, theta=theta, phi=phi, idx=0)
            S_total_neutral[:,i] = QD.calc_spin(basis_neutral, V_neutral, theta=theta, phi=phi)
            
            if not basis_charged is None: # if there is only the central spin
                S_total_charged[:,i] = QD.calc_spin(basis_charged, V_charged, theta=theta, phi=phi)

        # Collect all results
        useful = Energies < self.cutoff_energy
        self.Nstate = int(sum(useful))
        self.Energies = Energies[useful]
        
        self.Occupancy = np.concatenate((
            np.ones(2*Nm), np.zeros(Nm), 2*np.ones(Nm)
            )).astype(int)[useful]
        
        self.Spin_central = np.concatenate((S_central,np.zeros((2*Nm,3))))[useful]
        self.Spin_total = np.concatenate((S_total_neutral, S_total_charged, S_total_charged))[useful]
        self.lamb = lamb[np.ix_(useful, useful, [0, 1])]
        
        self.basis_charged = basis_charged
        self.basis_neutral = basis_neutral
        self.eigen_neutral = V_neutral
        self.eigen_charged = V_charged

        self.Delta = self.Energies[None,:] - self.Energies[:,None]
    
    def Spin(self, state_idx, theta=0, phi=0, spin_idx=None):
        """Calculates spin for a given state. 
        The direction of the operator is given by theta and phi. 
        The default direction is along the z axis. 
        
        Parameters:
        ----------
        state_idx: int
            index of the state
        thetha: float
            angle with the z axis
        phi: float
            angle with the x axis
        spin_idx: (int, or np.ndarray)
            which spin sites contribute, if None all contribute 
        
        Returns:
        -------
        S: float
            The expected value
        """

        occupancy = self.Occupancy[state_idx]
        # choose between a charged and uncharged states
        basis = self.basis_neutral if occupancy == 1 else self.basis_charged
        spin_idx = np.arange(0, self.Nspin, 1, dtype=int) if spin_idx==None else spin_idx
        spin_idx = np.array([spin_idx], dtype=int) if type(spin_idx) == int else spin_idx
        
        spin_idx = spin_idx-1 if occupancy !=1 else spin_idx
        new_spin_idx = spin_idx[spin_idx >= 0]
        
        eigen = self.eigen_neutral if occupancy == 1 else self.eigen_charged
        new_state_idx = np.array([np.sum(self.Occupancy[:state_idx] == occupancy)])
        # number of states with the same occupancy but lower energy
        S = np.zeros_like(spin_idx, dtype=float)

        if len(new_spin_idx) > 0:
            S[spin_idx>=0] = QD.calc_spin(basis, eigen[:,new_state_idx], theta=theta, phi=phi, idx=new_spin_idx)

        return S
    
    def SpinSquare(self, state_idx, spin_idx=None):
        """Calculates spin for a given state. 
        The default direction is along the z axis. 
        
        Parameters:
        ----------
        state_idx: int
            index of the state
        thetha: float
            angle with the z axis
        phi: float
            angle with the x axis
        spin_idx: (int, or np.ndarray)
            which spin sites contribute, if None all contribute 
        
        Returns:
        -------
        S: float
            The expected value
        """
        occupancy = self.Occupancy[state_idx]
        # choose between a charged and unchargated states
        basis = self.basis_neutral if occupancy == 1 else self.basis_charged
        spin_idx = np.arange(0, self.Nspin, 1, dtype=int) if spin_idx==None else spin_idx
        spin_idx = np.array([spin_idx], dtype=int) if type(spin_idx) == int else spin_idx
        
        spin_idx = spin_idx-1 if occupancy !=1 else spin_idx
        new_spin_idx = spin_idx[spin_idx >= 0]
        
        eigen = self.eigen_neutral if occupancy == 1 else self.eigen_charged
        new_state_idx = np.array([np.sum(self.Occupancy[:state_idx] == occupancy)])
        # number of states with the same occupancy but lower energy
        S = np.zeros_like(spin_idx, dtype=float)
        
        S = np.zeros_like(spin_idx)
        if len(new_spin_idx) > 0:
            S[spin_idx>=0] = QD.calc_spin_square(basis, eigen[:,new_state_idx], idx=new_spin_idx)
        return S
    
    def CalcAllSpin(self):
        """Calculate spin for all sites and all states in xyz
        """
        
        Spins = np.zeros((self.Nstate, self.Nspin, 5))

        for j in range(self.Nstate): 
            Spins[j,:,0] = j+1
            for k in range(self.Nspin):
                Spins[j,k,1] = k+1
                for i, theta, phi in zip([0,1,2], [np.pi/2, np.pi/2 ,0], [0, np.pi/2, 0]):  
                    # xyz
                    Spins[j,k,i+2] = self.Spin(j, theta=theta, phi=phi, spin_idx=k)[0]
        
        return Spins
    
    def solve_spin_ham(self, Spin, Jexch, Hlocal):
        Nspin = len(Spin)
        single_spins_basis = [spin_basis_1d(1, S=QD.spin2string(s)) for s in Spin]
        
        if Nspin == 1: 
            basis = single_spins_basis[0]
        else:
            basis = tensor_basis(*single_spins_basis) 

        # need to use tensor basis in case we are looking at combination of different magnitudes of spin 

        # generate interactions
        # single spin
        interactions_xyz = QD.collect_single_spin_ineractions(Hlocal)
        # exchange interaction
        interactions_xyz += QD.collect_exchange_interactions(Nspin, Jexch)

        # convert to creation and anhilation basis
        static, _ = basis.expanded_form(interactions_xyz,[])
        # generate a Hamiltonian.
        spinHam = hamiltonian(static,[], basis=basis, check_pcon=False, check_symm=False)
        # solve it 
        E, V = spinHam.eigh()
        
        return basis, E, V
    
    def reduce_lamb(self, tol=1e-5):
        non_zero = np.where(np.abs(self.lamb) > tol)
        red_lamb = np.zeros((len(non_zero[0]), 4), dtype=complex)
        for i, (v, l, s) in enumerate(zip(*non_zero)):
            red_lamb[i] = np.array([v,l,s, self.lamb[v,l,s]])
        return red_lamb

    def print_lamb(self, tol=1e-5):
        red_lamb = self.reduce_lamb(tol=tol)
        string = ''
        for state in red_lamb:
            string += '{:.0f}\t{:.0f}\t{:.0f}\t{:.5f} + i{:.5f}\n'.format(*state[:3].astype(int), np.real(state[3]), np.imag(state[3]))
        return string

    def __str__(self):
        string = 'idx \tOccup \tEnergy [meV] \tSpin central \tSpin total\n'
        string += '\t\t\t\t x,  y,  z \t x,  y,  z\n'
        states = []
        for i in range(self.Nstate):
            states.append(
                f'{i:d}\t{self.Occupancy[i]:d} \t{self.Energies[i]*Hartree:.9f} \t{self.Spin_central[i,0]:.1f} {self.Spin_central[i,1]:.1f} {self.Spin_central[i,2]:.1f} \t{self.Spin_total[i,0]:.1f} {self.Spin_total[i,1]:.1f} {self.Spin_total[i,2]:.1f}'
            )
        string += '\n'.join(states)
        return string

    @staticmethod
    def remove_central_exch(Jexch):
        new_Jexch = []
        for J in Jexch:
            if 0 in J[1:]:
                continue
            # reduces all indicies by one
            J[1] -= 1
            J[2] -= 1
            new_Jexch.append(J)
        return new_Jexch





