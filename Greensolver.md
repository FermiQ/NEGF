# Greensolver

## Purpose

The `Greensolver.cpp` file implements the `Greensolver` class, which is responsible for calculating Green's functions for the quantum system. It serves as a central component in the simulation framework, orchestrating calculations by interacting with other modules like `Schroedinger` and `SelfEnergy`.

## Key Functionalities/Classes

The primary class in this file is `Greensolver`.

Key methods likely include:

*   `Greensolver(const Schroedinger& schroedinger, const SelfEnergy& self_energy, ...)`: Constructor, likely taking references to `Schroedinger` and `SelfEnergy` objects to access Hamiltonian and self-energy information.
*   `do_solve_retarded(...)`: Computes the retarded Green's function. This function is fundamental for determining the system's response and propagation characteristics.
*   `do_solve_lesser(...)`: Computes the lesser Green's function, which is essential for calculating quantities like particle density and current.
*   `solve_retarded_Green_back_RGF(...)`: A specific solver, possibly using a recursive Green's function (RGF) technique, for the retarded Green's function, potentially calculating it by iterating "backwards" through system slices or layers.
*   `solve_advanced_Green_back_RGF(...)`: Similar to the above, but for the advanced Green's function.
*   `solve_lesser_Green_back_RGF(...)`: Similar to the above, but for the lesser Green's function using the RGF method.
*   `get_DOS_total_energy(...)`: Calculates the total Density of States (DOS) over energy.
*   `get_transmission_names()`: Returns the names of available transmission channels.
*   `get_transmission(std::string name)`: Retrieves a specific transmission function by name.

## Data Structures

*   `Propagator::PropagatorMap`: This data structure, likely a map, is used to store and manage `Propagator` objects. Propagators are essential for describing the motion of particles in the system. The map might be keyed by energy, momentum, or other relevant parameters.
*   `arma::cx_mat`: Armadillo library's complex matrix type, likely used extensively for representing Green's functions, Hamiltonians, self-energies, and other physical quantities.
*   `PetscMatrixParallelComplex`: A PETSc-based parallel complex matrix, used for large-scale numerical linear algebra operations required in solving Green's functions, especially in distributed memory environments.

## Dependencies

The `Greensolver` class and its implementation in `Greensolver.cpp` are expected to interact significantly with:

*   `Propagation.h`: For base class functionalities related to propagation and momentum meshes.
*   `Schroedinger.h`: To access the system's Hamiltonian, eigenvalues, and eigenvectors.
*   `SelfEnergy.h` (or `Self_energy.h`): To obtain self-energy corrections.
*   `PetscMatrixParallelComplex.h`: For utilizing PETSc's parallel matrix capabilities.
*   `Propagator.h`: For definitions of `Propagator` objects and `PropagatorMap`.
*   `GreensolverInput.h` (potentially): If there's a separate input class for `Greensolver` parameters.
*   Standard C++ libraries (e.g., `<vector>`, `<map>`, `<string>`).
*   Armadillo library (`armadillo`): For matrix operations.
*   MPI (Message Passing Interface): For parallel computations, often used in conjunction with PETSc.
