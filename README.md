# PyESRSTM

PyESRSTM is a Python package for simulating electron spin resonance (ESR) and spin dynamics in quantum dots coupled to leads. It provides tools to build Anderson impurity Hamiltonians, compute tunneling rates, solve the quantum master equation, and model time-dependent driven dynamics and ESR spectra.

The project is designed for exploring:

- single-spin and multi-spin quantum dots
- magnetic-field and exchange-coupling effects
- electrode-driven tunneling and RF modulation
- steady-state ESR response
- time propagation and pulsed driving

## Features

- `QD` model for quantum-dot spin Hamiltonians
- `Electrode` object for modeling leads, broadening, and Fermi functions
- rate calculations for tunneling between the dot and electrodes
- Floquet and master-equation solvers for steady-state transport
- ESR spectrum calculation across a frequency sweep
- time propagation of the density matrix under arbitrary or harmonic driving
- tutorial notebooks demonstrating common workflows

## Requirements

This project relies on scientific Python libraries, including:

- NumPy
- SciPy
- Matplotlib
- QuSpin

## Installation

From the repository root, make sure the project directory is on your Python path:

```bash
cd /path/to/PyESRSTM
export PYTHONPATH=$PWD
```

You can then import the package directly:

```python
import PyESRSTM
from PyESRSTM import QD, Electrode
```

If you are using a virtual environment, install the required dependencies first:

```bash
python -m pip install numpy scipy matplotlib quspin jupyter
```

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
import PyESRSTM

# Create a single-spin quantum dot
# eps [meV], U [meV], spins, magnetic field [T], g-factors

dot = PyESRSTM.QD(
    -30,
    100,
    np.array([0.5]),
    np.array([[0.20804273, 0.0, 0.57159271]]),
    np.array([[2.0, 2.0, 2.0]]),
)

# Define electrodes
Right = PyESRSTM.Electrode(0, 0, 0.0, 5e-3, 1e-1, 4.5, Nint=1e4, Cutoff=1000)
Left = PyESRSTM.Electrode(-25, 4, 0.5, 1.25e-3, 1e-1, 4.5, Nint=1e4, Cutoff=1000)

# ESR sweep
freq = np.linspace(16.5, 17.5, 1000)
I = PyESRSTM.ESR.ESR(Left, Right, dot, freq, NFL=1)

plt.plot(freq, I)
plt.xlabel(r'$\omega/2\pi$')
plt.ylabel('I [pA]')
plt.show()
```

## Tutorials

The repository includes Jupyter notebooks in the `Tutorial/` directory that walk through the main workflows:

- `Tutorial/01-single-spin.ipynb` — single-spin quantum dot and ESR basics
- `Tutorial/02-Time-propagation.ipynb` — time propagation and transient dynamics
- `Tutorial/two-spin.ipynb` — two-spin systems and multi-resonance behavior

These notebooks are the best place to start if you want to understand how the simulation tools are used in practice.

## Project Structure

```text
PyESRSTM/
├── PyESRSTM/
│   ├── __init__.py
│   ├── QD/
│   ├── Electrode/
│   ├── Floquet/
│   ├── Time/
│   ├── Trasport/
│   ├── misc/
│   └── postproces/
├── Tutorial/
├── tests/
├── README.md
└── ignore/
```

## Typical Workflow

1. Define a `QD` object with the dot energies, spin configuration, and magnetic field.
2. Define `Electrode` objects with DC and RF parameters.
3. Compute tunneling rates between the dot and electrodes.
4. Solve the quantum master equation or propagate the density matrix.
5. Compute the current or ESR signal.
6. Fit or analyze resonances, bias dependence, and time-dependent dynamics.

## Notes

This package is intended for research in quantum transport and ESR of nanoscale spin systems. The notebooks in `Tutorial/` are especially useful for understanding the conventions and units used throughout the code.

## Citing
If you use this software, please cite the original methodology paper with A-driving, and QuSpin paper that is used to solve the eigenstates of the isolated quantum dot:

Reina-Gálvez, J., Lorente, N., Delgado, F., & Arrachea, L. (2021). All-electric electron spin resonance studied by means of Floquet quantum master equations. Physical Review B, 104(24), 245435. https://doi.org/10.1103/PHYSREVB.104.245435

Weinberg, P., & Bukov, M. (2017). QuSpin: A Python package for dynamics and exact diagonalisation of quantum many body systems Part I: Spin chains. SciPost Physics, 2(1), 003. https://doi.org/10.21468/SCIPOSTPHYS.2.1.003

If you are using the RF driving also include this paper:

M. Nachtigall, J. Reina-Galvez, C. Wolf, N. Lorente "The effects of an alternating bias on a single orbital spin impurity", 
In preparation (2026)

If you are using spin dynamics descriptors please cite: 

Reina-Gálvez, J., Nachtigall, M., Lorente, N., Martinek, J., & Wolf, C. (2025). Contrasting exchange-field and spin-transfer torque driving mechanisms in all-electric electron spin resonance. Physical Review B, 112(24), 245408. https://doi.org/10.1103/nzhr-syhs

BibTeX:

```bibtex
@article{Reina-Galvez2021,
  title = {All-electric electron spin resonance studied by means of Floquet quantum master equations},
  author = {Reina-Galvez J., Lorente N., Delgado F., Arrachea L.},
  journal = {Physical Review B},
  volume = {104},
  pages = {245435},
  year = {2021},
  doi = {10.1103/PhysRevB.104.245435}
}
@article{Weinberg2017,
   title = {QuSpin: A Python package for dynamics and exact diagonalisation of quantum many body systems Part I: Spin chains},
   author = {Phillip Weinberg and Marin Bukov},
   doi = {10.21468/SCIPOSTPHYS.2.1.003},
   journal = {SciPost Physics},
   pages = {003},
   publisher = {SciPost Foundation},
   volume = {2},
   year = {2017}
}

@article{Reina-Galvez2025,
   title = {Contrasting exchange-field and spin-transfer torque driving mechanisms in all-electric electron spin resonance},
   author = {Jose Reina-Gálvez and Matyas Nachtigall and Nicolás Lorente and Jan Martinek and Christoph Wolf},
   doi = {10.1103/nzhr-syhs},
   journal = {Physical Review B},
   pages = {245408},
   volume = {112},
   year = {2025}
}
```