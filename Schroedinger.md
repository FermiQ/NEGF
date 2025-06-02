---
noIndex: true
---

# Schroedinger

## Purpose

The `Schroedinger.cpp` file implements the `Schroedinger` class, which is responsible for constructing the Hamiltonian of the quantum system and solving the time-independent Schroedinger equation. This typically involves finding the eigenvalues and eigenvectors of the Hamiltonian, which represent the energy levels and corresponding wavefunctions of the system. These are crucial inputs for other components like `Greensolver`.

## Key Functionalities/Classes

The primary class in this file is `Schroedinger`.

Key methods and functionalities likely include:

* `Schroedinger(...)`: Constructor, which takes parameters defining the system, such as its geometry, material properties, and any external fields. It might also take a `MomentumMesh` from `Propagation`.
* `setup_hamiltonian()`: Constructs the Hamiltonian matrix for the system. This could involve defining tight-binding parameters, kinetic energy terms, potential energy terms, and spin-orbit coupling, depending on the physical model.
* `solve()`: Solves the eigenvalue problem H |psi> = E |psi> for the constructed Hamiltonian H. This method would populate internal data structures with eigenvalues and eigenvectors.
* `get_eigenvalues()`: Returns the calculated eigenvalues (energy levels).
* `get_eigenvectors()`: Returns the calculated eigenvectors (wavefunctions).
* `get_hamiltonian_matrix()`: Returns the constructed Hamiltonian matrix.
* `apply_boundary_conditions(...)`: Applies appropriate boundary conditions (e.g., periodic, open) to the Hamiltonian.
* `set_potential(...)`: Allows for setting or modifying the potential energy landscape of the system.

## Data Structures

* `arma::mat` or `arma::cx_mat`: Armadillo matrices (real or complex) are used to represent the Hamiltonian of the system.
* `arma::vec` (real): Armadillo vector used to store the eigenvalues.
* `arma::mat` or `arma::cx_mat`: Armadillo matrix used to store the eigenvectors, where each column typically represents an eigenvector.
* `MomentumMesh`: If the Hamiltonian is defined in momentum space, it would use a `MomentumMesh` object (from `Propagation`) to define the k-points.
* `SystemParameters` (or similar custom struct/class): A data structure to hold various physical parameters of the system being modeled (e.g., lattice constants, effective masses, potential parameters).
* `PetscMatrixParallelComplex` or `PetscMatrixParallelReal`: If the Hamiltonian is very large, PETSc parallel matrices might be used for its storage and for eigensolving (using SLEPc, PETSc's eigenvalue solver library).

## Dependencies

The `Schroedinger` class and its implementation in `Schroedinger.cpp` would likely depend on:

* `Propagation.h`: To get access to momentum mesh information if the Hamiltonian is k-dependent.
* `Input.h` (or a specific `SchroedingerInput.h`): To read system parameters, model choices, and solver settings.
* Armadillo library (`armadillo`): For matrix and vector operations, and potentially for its built-in eigensolvers for smaller systems.
* PETSc/SLEPc libraries: For large-scale, parallel eigensolving.
* Standard C++ libraries (e.g., `<vector>`, `<complex>`).
* Utility files for defining physical constants or specific model Hamiltonians (e.g., `HamiltonianBuilder.h`).
* Its results (eigenvalues, eigenvectors, Hamiltonian) are used by `Greensolver.cpp` and potentially `SelfEnergy.cpp`.
