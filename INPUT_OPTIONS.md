---
noIndex: true
---

# Inferred Input Options

**Disclaimer:** This document lists potential input options inferred from the likely functionalities of the simulation framework's components. Without direct access to the source code's `options.get_option(...)` calls or example input files, this list is speculative and may not be exhaustive or perfectly accurate. It is intended as a general guide to the types of parameters that might be configurable.

## General Options / Global Settings

These options might control overall simulation behavior or settings that apply to multiple components.

* `simulation_mode`: (string) Defines the overall goal of the simulation (e.g., "band\_structure", "dos\_calculation", "transport\_calculation", "self\_consistent\_run").
* `output_directory`: (string) Path to the directory where output files should be saved.
* `verbosity_level`: (integer) Controls the amount of logging and output messages (e.g., 0 for silent, 5 for debug).
* `energy_min`: (double) Minimum energy for energy-dependent calculations.
* `energy_max`: (double) Maximum energy for energy-dependent calculations.
* `energy_points`: (integer) Number of energy points in energy meshes.
* `temperature`: (double) System temperature in Kelvin, affecting Fermi distributions and some interactions.

## `Propagation` Component Options

These options likely control the setup of momentum meshes and propagators.

* `Propagation::momentum_mesh_names`: (list of strings) Defines the names or types of momentum meshes to create (e.g., \["k\_mesh", "q\_mesh"]).
* `Propagation::k_mesh_type`: (string) Type of k-mesh (e.g., "uniform", "cartesian", "spherical").
* `Propagation::k_mesh_dimensions`: (list of integers) Number of points in each dimension of the k-mesh (e.g., \[10, 10, 5]).
* `Propagation::k_mesh_center`: (list of doubles) Center point of the k-mesh in momentum space.
* `Propagation::k_mesh_range`: (list of doubles or list of lists of doubles) Range of momentum values for each dimension.
* `Propagation::q_mesh_type`: (string) Type of q-mesh (similar to k\_mesh\_type).
* `Propagation::q_mesh_dimensions`: (list of integers) Dimensions for the q-mesh.
* `Propagation::enable_symmetries`: (boolean) Whether to use symmetries to reduce the momentum mesh size.

## `Schroedinger` Component Options

These options relate to defining the Hamiltonian and solving the Schroedinger equation.

* `Schroedinger::hamiltonian_model`: (string) Specifies the model for the Hamiltonian (e.g., "tight\_binding", "effective\_mass", "kp\_8\_band").
* `Schroedinger::material_parameters_file`: (string) Path to a file containing material-specific parameters (e.g., effective masses, band gaps, tight-binding hopping elements).
* `Schroedinger::geometry_file`: (string) Path to a file defining the system's geometry or structure.
* `Schroedinger::number_of_bands`: (integer) Number of bands to calculate (alternative to `number_of_eigenvalues`).
* `Schroedinger::number_of_eigenvalues`: (integer) Number of eigenvalues/eigenvectors to compute if an eigensolver is used directly.
* `Schroedinger::eigenvalue_solver_type`: (string) Specifies the eigensolver algorithm (e.g., "arpack", "lapack", "slepc\_krylovschur").
* `Schroedinger::boundary_conditions_x`: (string) Boundary condition along x-axis (e.g., "periodic", "dirichlet"). (Similar options for y and z axes).
* `Schroedinger::potential_profile_file`: (string) Path to a file defining an external potential profile.
* `Schroedinger::spin_orbit_coupling_strength`: (double) Strength of spin-orbit interaction.

## `SelfEnergy` Component Options

These options configure the calculation of self-energies due to interactions or leads.

* `SelfEnergy::interaction_types`: (list of strings) Specifies which types of interactions to include (e.g., \["electron\_phonon", "impurity\_scattering", "lead\_coupling\_left", "lead\_coupling\_right"]).
* `SelfEnergy::electron_phonon_coupling_strength`: (double) Coupling constant for electron-phonon interaction.
* `SelfEnergy::phonon_model_file`: (string) Path to a file describing the phonon spectrum.
* `SelfEnergy::impurity_concentration`: (double) Concentration of impurities for impurity scattering.
* `SelfEnergy::impurity_potential_strength`: (double) Strength of the impurity scattering potential.
* `SelfEnergy::lead_left_model_file`: (string) Path to a file describing the left lead for its self-energy calculation.
* `SelfEnergy::lead_right_model_file`: (string) Path to a file describing the right lead.
* `SelfEnergy::self_consistency_max_iterations`: (integer) Maximum iterations for self-consistent self-energy calculations (e.g., SCBA).
* `SelfEnergy::self_consistency_tolerance`: (double) Convergence tolerance for self-consistent calculations.
* `SelfEnergy::broadening_eta`: (double) Small imaginary part added to energy for Green's functions and self-energies (related to lifetime/scattering).

## `Greensolver` Component Options

These options control how Green's functions are calculated.

* `Greensolver::solver_type`: (string) Main method for solving Green's functions (e.g., "matrix\_inversion", "rgf").
* `Greensolver::inversion_method`: (string) Specific algorithm if matrix inversion is chosen (e.g., "lu\_decomposition", "petsc\_ksp").
* `Greensolver::rgf_direction`: (string) If RGF is used, specifies direction (e.g., "forward", "backward", "forward\_backward\_combined"). This might be handled by choosing `ScatteringForwardRGFSolver` or `ScatteringBackwardRGFSolver` more directly.
* `Greensolver::device_slices`: (integer) Number of slices in the device region for RGF methods.
* `Greensolver::calculate_retarded_G`: (boolean) Flag to calculate the retarded Green's function.
* `Greensolver::calculate_lesser_G`: (boolean) Flag to calculate the lesser Green's function.
* `Greensolver::calculate_dos`: (boolean) Flag to calculate the Density of States.
* `Greensolver::calculate_transmission`: (boolean) Flag to calculate transmission coefficients.
* `Greensolver::transmission_lead_pairs`: (list of string pairs) Defines pairs of leads for which to calculate transmission (e.g., \[\["left\_lead", "right\_lead"]]).

## `ScatteringBackwardRGFSolver` / `ScatteringForwardRGFSolver` Options

These might have specific options if invoked directly, though many would overlap with `Greensolver` or be implicitly set.

* `RGF::slice_hamiltonian_prefix`: (string) Prefix for files containing slice Hamiltonians if not constructed internally.
* `RGF::coupling_matrix_prefix`: (string) Prefix for files containing coupling matrices between slices.
* `RGF::initial_surface_green_function_file` (string): File to load an initial surface Green's function (e.g. for the lead).

This list provides a starting point for understanding the configurability of the simulation framework. The actual options and their exact names/formats would be found in the project's input parsing logic.
