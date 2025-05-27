# ScatteringBackwardRGFSolver

## Purpose

The `ScatteringBackwardRGFSolver.cpp` file likely implements a specialized solver class, `ScatteringBackwardRGFSolver`, which calculates scattering properties (like Green's functions or transmission) using a recursive Green's function (RGF) method. The "Backward" in its name suggests that the recursion proceeds from one end of the device (e.g., the right lead) towards the other (e.g., the left lead), accumulating information about the scattering states.

This solver is typically used for systems that can be divided into layers or slices, allowing for an efficient recursive calculation.

## Key Functionalities/Classes

The primary class in this file is `ScatteringBackwardRGFSolver`.

Key methods and functionalities likely include:

*   `ScatteringBackwardRGFSolver(...)`: Constructor, which may take parameters defining the system's geometry, Hamiltonian, and connections to leads.
*   `solve()`: The main method that executes the recursive Green's function algorithm to compute the desired scattering quantities (e.g., retarded Green's function of the device).
*   `initialize_recursion(...)`: Sets up the initial conditions for the recursion, often starting with the self-energy of the right lead.
*   `recursive_step(int slice_index)`: Performs a single step of the RGF algorithm, calculating the Green's function for `slice_index` based on the information from `slice_index + 1`.
*   `get_surface_green_function(...)`: Returns the surface Green's function at the interface, which is a key result of the RGF method.
*   `get_transmission(...)`: If applicable, this method might calculate the transmission probability through the device based on the computed Green's functions.
*   `add_self_energy(...)`: A method to incorporate self-energies from leads or other interactions into the calculation for each slice.

## Data Structures

*   `arma::cx_mat`: Armadillo complex matrices are certainly used to represent Green's functions for each slice, Hamiltonian blocks, self-energies, and coupling matrices between slices.
*   `std::vector<arma::cx_mat>`: A vector of Armadillo complex matrices might be used to store the Green's functions or self-energies for all slices of the device.
*   `DeviceRegion` (or similar custom struct/class): A data structure to define the properties of the central scattering region (e.g., number of slices, Hamiltonians of slices).
*   `Lead` (or similar custom struct/class): Data structures to define the properties of the left and right leads connected to the device, including their self-energies.

## Dependencies

The `ScatteringBackwardRGFSolver` class would likely depend on:

*   `Greensolver.h` (or a base `RGFSolver.h`): It might inherit from a more general Green's function solver or share interfaces.
*   `Schroedinger.h`: To obtain Hamiltonian blocks for different slices of the device.
*   `SelfEnergy.h`: To get lead self-energies or other interaction self-energies.
*   `PetscMatrixParallelComplex.h` or similar: If the problem size is large, it might leverage PETSc for distributed linear algebra, though RGF methods are often designed to work with smaller, dense matrices per slice.
*   Armadillo library (`armadillo`): For all dense matrix operations.
*   Standard C++ libraries (e.g., `<vector>`, `<complex>`).
*   Input classes: To read device geometry, material parameters, and solver settings.
*   It might be a component used *by* a higher-level `Greensolver` class, or it could be a standalone solver for specific scattering problems.
