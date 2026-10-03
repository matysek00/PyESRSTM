# PyESRSTM Documentation

PyESRSTM is a Python package for modeling spin dynamics and electron spin resonance (ESR) in quantum dots coupled to electrodes. The package provides a workflow for building spin Hamiltonians, computing tunneling rates, solving the master equation, and analyzing current and ESR spectra under static and time-dependent driving.

This documentation is intended to serve as a practical guide to the package API and the typical computational workflow used in the project.

## Overview

The library is organized around a few core objects:

- `QD`: quantum-dot Hamiltonian and many-body spin states
- `Electrode`: electrode model with Fermi statistics and drive parameters
- `rates`: computation of tunneling rates between lead and dot
- `QME` / `QME_matrix`: steady-state master-equation solver
- `current`: current calculation from the density matrix
- `ESR`: ESR spectrum calculation over a range of frequencies
- `propagate_density`: time evolution of the density matrix

These building blocks can be combined to simulate:

- spin resonance in a single quantum dot
- multi-spin exchange physics
- driven tunneling and Floquet sidebands
- pulsed or time-dependent excitation
- transport and current response under bias and microwave drive

## Installation

From the project root:

```bash
cd /path/to/PyESRSTM
export PYTHONPATH=$PWD
```

Then import the package in Python:

```python
import PyESRSTM
from PyESRSTM import QD, Electrode
```

Required dependencies include:

- NumPy
- SciPy
- Matplotlib
- QuSpin

## Quick Start

```python
import numpy as np
import PyESRSTM

# Single-spin quantum dot
# eps [meV], U [meV], spins, magnetic field [T], g-factors
dot = PyESRSTM.QD(
    -30,
    100,
    np.array([0.5]),
    np.array([[0.20804273, 0.0, 0.57159271]]),
    np.array([[2.0, 2.0, 2.0]]),
)

# Electrodes
Right = PyESRSTM.Electrode(0, 0, 0.0, 5e-3, 1e-1, 4.5, Nint=1e4, Cutoff=1000)
Left = PyESRSTM.Electrode(-25, 4, 0.5, 1.25e-3, 1e-1, 4.5, Nint=1e4, Cutoff=1000)

# Frequency sweep for ESR
freq = np.linspace(16.5, 17.5, 1000)
I = PyESRSTM.ESR.ESR(Left, Right, dot, freq, NFL=1)
```

## Core Concepts

### Quantum Dot Model

The `QD` class constructs the many-body spin Hamiltonian for the quantum dot. It includes:

- local magnetic fields
- g-tensor information
- exchange interactions between spins
- single-particle and charging energies
- occupation and state-level information

The resulting object exposes properties such as:

- `Nspin`: number of spins
- `Nstate`: number of many-body states
- `Occupancy`: occupancy of each state
- `Energies`: eigenenergies
- `Delta`: energy difference matrix
- `lamb`: orbital overlap factors used in the tunneling model

### Electrodes

The `Electrode` class stores the lead parameters required to compute tunneling and transport. These include:

- DC bias voltage
- RF drive amplitude
- spin polarization
- tunnel coupling strength
- temperature
- integration grid and broadening parameters

The Fermi function and the Fermi integrals are precomputed on a numerical grid, which is then used when evaluating transport coefficients.

### Rate Calculations

The rate model builds transition amplitudes between dot states and electrodes. It uses a Floquet expansion and Bessel-function treatment of the RF drive. The key functions are:

- `rates(QD, Electrode, frequency, Nfour)`
- `sum_rates(GL, GR)`

The result is a set of rate tensors that can be used in the master equation or current calculation.

## Steady-State Transport and ESR

The steady-state solver is provided by `QME` and `QME_matrix`. These functions solve the density-matrix equation for the quantum dot under the influence of the rates.

Typical workflow:

```python
from PyESRSTM import rates, sum_rates, QME, QME_matrix, current

GLplus, GLminus = rates(dot, Left, 20, 5)
GRplus, GRminus = rates(dot, Right, 0, 0)
G = sum_rates(GLplus + GLminus, GRplus + GRminus)

rho = QME(G, 20.0, Delta=dot.Delta)
I = current(rho, GLplus - GLminus)
```

The `ESR.ESR` function wraps this logic for a scan over many driving frequencies.

## Time Evolution

The package also supports time propagation of the density matrix with the `propagate_density` function.

This is useful for:

- transient spin dynamics
- Rabi oscillations
- pulsed excitation
- arbitrary driving protocols
- non-steady-state response

Example:

```python
rho0 = np.zeros((4, 4), dtype=complex)
rho0[0, 0] = 1

time, rhot = PyESRSTM.propagate_density(
    rho0,
    700,
    GL,
    GR,
    dot.Delta,
    frequency=17.1,
    kwargs={'max_step': 0.005, 'method': 'RK45'},
)
```

## API Reference

### `PyESRSTM.QD`

The `QD` class constructs the many-body Hamiltonian for the quantum dot.

Key methods and attributes:

- `remove_states(states)`
- `CalcAllSpin()`
- `print_lamb()`
- `Nspin`, `Nstate`, `Energies`, `Occupancy`, `Delta`, `lamb`

### `PyESRSTM.Electrode`

The `Electrode` class models the lead's potential, tunneling, and Fermi distribution.

Typical parameters:

- `Vdc`
- `Vrf`
- `Spin_polarization`
- `g0`
- `gammaC`
- `Temperature`
- `Adrive`
- `Cutoff`
- `Nint`

### `rates`

Computes the Floquet-resolved tunneling rates between electrode and quantum dot.

```python
GLplus, GLminus = rates(dot, Left, 20, 5)
```

### `sum_rates`

Adds two sets of rates with different Fourier truncations in a consistent way.

### `QME` and `QME_matrix`

Solve the quantum master equation at fixed frequency, either directly or using a precomputed matrix.

### `current`

Computes the current from the density matrix and current-generating rate tensor.

### `ESR.ESR`

Computes ESR spectra over a frequency range:

```python
I = PyESRSTM.ESR.ESR(Left, Right, dot, freq, NFL=1)
```

For `return_Gs=True`, the function can also return the rate tensors used in the calculation.

## Notes on Units

The code mixes user-facing frequencies and times with internal Hartree-based atomic units. In practice:

- frequency is often supplied in GHz
- time is often supplied in ns
- internal calculations use Hartree-scaled units and conversion factors from `PyESRSTM.misc.units`

This is handled internally by the library, but it is useful to keep in mind when interpreting numerical results.

## Tutorials and Examples

The best starting points are the notebooks in the `Tutorial/` directory:

- `01-single-spin.ipynb`
- `02-Time-propagation.ipynb`
- `two-spin.ipynb`
- other notebooks in the same folder for parameter scans and advanced examples

These examples cover the main simulation routines and help explain the physical assumptions behind the code.

## Limitations and Caveats

- Some methods are specialized to the spin-1/2 transport problem and may require care when extending to more general systems.
- Time-dependent calculations can be computationally heavy.
- Convergence of the integral grid and Fourier truncation must be checked for accurate results.
- The code relies on QuSpin for the spin-basis Hamiltonian representation.

## Summary

PyESRSTM is a simulation package for quantum-dot ESR and transport problems. It combines:

- many-body spin Hamiltonians
- lead and tunneling models
- Floquet-driven rate calculations
- master-equation solutions
- current and ESR analysis
- time propagation under driving

The package is most useful for studying the non-equilibrium spin physics of nanoscale quantum dots, especially under microwave drive and bias conditions.

## Citation

If you use this software, please cite the original methodology paper with A-driving:

Reina-Gálvez, J., Lorente, N., Delgado, F., & Arrachea, L. (2021). All-electric electron spin resonance studied by means of Floquet quantum master equations. Physical Review B, 104(24), 245435. https://doi.org/10.1103/PHYSREVB.104.245435

If you are using the RF driving also include this paper:

M. Nachtigall, J. Reina-Galvez, C. Wolf, N. Lorente "The effects of an alternating bias on a single orbital spin impurity" 
I preparation (2026)

If you are using spin dynamics descriptors please cite 

Reina-Gálvez, J., Nachtigall, M., Lorente, N., Martinek, J., & Wolf, C. (2025). Contrasting exchange-field and spin-transfer torque driving mechanisms in all-electric electron spin resonance. Physical Review B, 112(24), 245408. https://doi.org/10.1103/nzhr-syhs

BibTeX:

```bibtex
@article{delavarga2021,
  title = {All-electric electron spin resonance studied by means of Floquet quantum master equations},
  author = {Reina-Galvez J., Lorente N., Delgado F., Arrachea L.},
  journal = {Physical Review B},
  volume = {104},
  pages = {245435},
  year = {2021},
  doi = {10.1103/PhysRevB.104.245435}
}
@article{Reina-Glvez2025,
   author = {Jose Reina-Gálvez and Matyas Nachtigall and Nicolás Lorente and Jan Martinek and Christoph Wolf},
   doi = {10.1103/nzhr-syhs},
   journal = {Physical Review B},
   pages = {245408},
   title = {Contrasting exchange-field and spin-transfer torque driving mechanisms in all-electric electron spin resonance},
   volume = {112},
   year = {2025}
}

