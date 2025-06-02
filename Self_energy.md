---
noIndex: true
---

# Self\_energy

## Purpose

The `Self_energy.cpp` file (and its corresponding `SelfEnergy.h` or `Self_energy.h` header) implements the `SelfEnergy` class. This class is responsible for calculating the self-energy of particles within the simulated quantum system. Self-energy accounts for the interactions a particle experiences with its surrounding environment or with other particles, effectively modifying its propagation characteristics (as described by the Green's function).

Self-energies are crucial for accurately modeling many-body effects and transport phenomena.

## Key Functionalities/Classes

The primary class in this file is `SelfEnergy`.

Key methods and functionalities likely include:

* `SelfEnergy(...)`: Constructor, which may take parameters related to the interaction type, system properties, and potentially references to `Schroedinger` or `Propagation` objects if the self-energy depends on the system's eigenstates or momentum mesh.
* `calculate_self_energy(double energy, ...)`: The main method to compute the self-energy matrix for a given energy and other relevant parameters (e.g., momentum, spin). This is the core calculation performed by the class.
* `get_retarded_self_energy(...)`: Specifically returns the retarded component of the self-energy.
* `get_lesser_self_energy(...)`: Specifically returns the lesser component of the self-energy (often related to occupation and scattering-in rates).
* `get_greater_self_energy(...)`: Specifically returns the greater component of the self-energy (related to scattering-out rates).
* `set_interaction_parameters(...)`: Allows for setting or modifying parameters related to the interaction model being used (e.g., electron-phonon coupling strength, impurity concentration).
* `initialize_lead_self_energy(...)`: If the system is connected to leads (contacts), this method would calculate the self-energies due to these leads, which act as boundary conditions for open systems. This often involves using surface Green's functions of the leads.
* `update_self_energy_approximation(...)`: Some self-energies might be calculated iteratively (e.g., in a self-consistent Born approximation). This method would update the approximation based on previous results.

## Data Structures

* `arma::cx_mat`: Armadillo complex matrices are extensively used to represent the self-energy for a given energy and momentum. Self-energy is generally a matrix in the same basis as the Hamiltonian.
* `std::map<KeyType, arma::cx_mat>`: A map could be used to store pre-computed or frequently accessed self-energies, keyed by energy or other parameters.
* `InteractionModelParameters` (custom struct/class): To store parameters defining the type and strength of interactions (e.g., coupling constants, disorder parameters).
* `MomentumMesh`: If the self-energy is k-dependent, it would use a `MomentumMesh` from `Propagation`.
* `PetscMatrixParallelComplex`: For large systems where the self-energy matrix might be distributed.

## Dependencies

The `SelfEnergy` class and its implementation in `Self_energy.cpp` would likely depend on:

* `Schroedinger.h`: May require eigenvalues and eigenvectors from `Schroedinger` if the self-energy is calculated in the eigenbasis or depends on wavefunctions (e.g., for electron-phonon coupling).
* `Propagation.h`: To access momentum mesh information if the self-energy is k-dependent or involves sums over momentum.
* `Greensolver.h`: In self-consistent calculations (like the self-consistent Born approximation), the self-energy calculation would depend on the Green's functions calculated by `Greensolver`.
* `Input.h` (or a specific `SelfEnergyInput.h`): To read parameters for the interaction model, lead properties, etc.
* Armadillo library (`armadillo`): For matrix operations.
* Standard C++ libraries (e.g., `<vector>`, `<map>`, `<complex>`).
* Specialized utility functions for specific interaction types (e.g., phonon spectrum calculators for electron-phonon interaction).
* Its results are primarily used by `Greensolver.cpp` to incorporate interaction effects into the Green's function calculations.
