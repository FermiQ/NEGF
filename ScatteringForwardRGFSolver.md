---
noIndex: true
---

# ScatteringForwardRGFSolver

## Purpose

The `ScatteringForwardRGFSolver.cpp` file likely implements a specialized solver class, `ScatteringForwardRGFSolver`, which calculates scattering properties using a recursive Green's function (RGF) method. The "Forward" in its name suggests that the recursion proceeds from one end of the device (e.g., the left lead) towards the other (e.g., the right lead).

This solver is complementary to the `ScatteringBackwardRGFSolver` and is used for systems that can be divided into layers or slices, allowing for an efficient recursive calculation. Sometimes both forward and backward RGF passes are needed for certain quantities.

## Key Functionalities/Classes

The primary class in this file is `ScatteringForwardRGFSolver`.

Key methods and functionalities likely include:

* `ScatteringForwardRGFSolver(...)`: Constructor, which may take parameters defining the system's geometry, Hamiltonian, and connections to leads.
* `solve()`: The main method that executes the recursive Green's function algorithm.
* `initialize_recursion(...)`: Sets up the initial conditions for the recursion, often starting with the self-energy of the left lead.
* `recursive_step(int slice_index)`: Performs a single step of the RGF algorithm, calculating the Green's function for `slice_index` based on the information from `slice_index - 1`.
* `get_surface_green_function(...)`: Returns the surface Green's function at the "forward" interface.
* `add_self_energy(...)`: A method to incorporate self-energies from leads or other interactions into the calculation for each slice.
* Potentially methods to combine results with a `ScatteringBackwardRGFSolver` to get the full device Green's function.

## Data Structures

* `arma::cx_mat`: Armadillo complex matrices are used to represent Green's functions for each slice, Hamiltonian blocks, self-energies, and coupling matrices between slices.
* `std::vector<arma::cx_mat>`: A vector of Armadillo complex matrices might be used to store the Green's functions or self-energies for all slices of the device as the recursion progresses.
* `DeviceRegion` (or similar custom struct/class): A data structure to define the properties of the central scattering region.
* `Lead` (or similar custom struct/class): Data structures to define the properties of the leads.

## Dependencies

The `ScatteringForwardRGFSolver` class would likely depend on:

* `Greensolver.h` (or a base `RGFSolver.h`): It might inherit from a more general Green's function solver.
* `Schroedinger.h`: To obtain Hamiltonian blocks for different slices.
* `SelfEnergy.h`: To get lead self-energies.
* Armadillo library (`armadillo`): For dense matrix operations.
* Standard C++ libraries (e.g., `<vector>`, `<complex>`).
* Input classes: To read device geometry, material parameters, and solver settings.
* It might interact with `ScatteringBackwardRGFSolver.h` if results from both are combined.
