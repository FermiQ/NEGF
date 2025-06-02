---
noIndex: true
---

# Typical Simulation Workflow

This document describes a typical workflow for setting up, executing, and analyzing simulations using this framework. The workflow relies on the interaction of several core components, including `Propagation`, `Schroedinger`, `SelfEnergy`, `Greensolver`, and specialized RGF solvers.

## 1. Setup: Defining the System

The initial phase involves defining the quantum system to be simulated. This includes specifying its physical geometry, material properties, and the theoretical models to be applied.

* **Defining Geometry and Materials:**
  * The user defines the physical dimensions of the system, which might be a bulk material, a heterostructure, a quantum wire, or a device with leads.
  * Material parameters are specified, such as effective masses, dielectric constants, band offsets, and lattice parameters. These parameters are crucial for constructing the system's Hamiltonian.
  * Input files or dedicated setup routines are likely used to provide this information to the simulation framework.
* **Discretization and Meshes (`Propagation`):**
  * The `Propagation` component is used to define and initialize momentum space meshes (k-space, q-space). This involves choosing the type of mesh (e.g., uniform, logarithmic), its boundaries, and the density of points.
  * For example, `Propagation::initialize_meshes(...)` would be called based on user inputs defining the desired resolution in momentum space.
* **Hamiltonian Definition (`Schroedinger`):**
  * The `Schroedinger` component is responsible for constructing the system's Hamiltonian.
  * The user specifies the physical model for the Hamiltonian (e.g., tight-binding, effective mass approximation, k.p theory).
  * `Schroedinger::setup_hamiltonian()` would be invoked, incorporating the material parameters and geometry.
  * Boundary conditions (periodic, Dirichlet, Neumann) are applied via methods like `Schroedinger::apply_boundary_conditions(...)`.
  * If required, external potentials (e.g., gate voltages) can be set using `Schroedinger::set_potential(...)`.
* **Defining Interactions and Leads (`SelfEnergy`):**
  * If interactions (e.g., electron-phonon, electron-electron, disorder) are to be included, the `SelfEnergy` component is configured.
  * This involves selecting the type of interaction and providing relevant parameters (e.g., coupling strengths, impurity concentrations) via `SelfEnergy::set_interaction_parameters(...)`.
  * For open systems connected to leads (e.g., in transport calculations), lead properties are defined, and `SelfEnergy::initialize_lead_self_energy(...)` is used to compute the self-energies due to these leads. These act as boundary conditions for the Green's function calculations.

## 2. Execution: Running the Simulation

Once the system is defined, the simulation is executed by invoking the appropriate solvers.

* **Solving for Eigenstates (Optional, `Schroedinger`):**
  * For some analyses or as a preliminary step, the user might solve for the system's eigenvalues and eigenvectors directly using `Schroedinger::solve()`. This provides information about the band structure and wavefunctions of the isolated system.
* **Calculating Self-Energies (`SelfEnergy`):**
  * The `SelfEnergy` component calculates the self-energy matrices for the specified interactions and/or leads. This is typically done by calling methods like `SelfEnergy::calculate_self_energy(energy, ...)`.
  * In self-consistent loops (e.g., Self-Consistent Born Approximation), this step might be iterated with Green's function calculations.
* **Solving for Green's Functions (`Greensolver` / RGF Solvers):**
  * The core of many simulations involves calculating Green's functions.
  * The `Greensolver` component is used for this purpose. Methods like `Greensolver::do_solve_retarded(...)` and `Greensolver::do_solve_lesser(...)` are called, taking the Hamiltonian (from `Schroedinger`) and self-energies (from `SelfEnergy`) as inputs.
  * For specific device geometries, specialized solvers like `ScatteringBackwardRGFSolver` and `ScatteringForwardRGFSolver` might be invoked. These typically have a `solve()` method that implements the recursive algorithm. For example, `ScatteringBackwardRGFSolver::solve()` would compute the Green's functions slice by slice.
  * These solvers often require a range of energies for which the Green's functions are to be computed.
* **Iterative Procedures:**
  * Some simulations might involve self-consistent calculations where, for example, the Green's function depends on the self-energy, and the self-energy, in turn, depends on the Green's function. The framework would iterate between `Greensolver` and `SelfEnergy` calculations until convergence is reached (e.g., using `SelfEnergy::update_self_energy_approximation(...)`).

## 3. Output: Generating and Interpreting Results

After the simulation completes, various physical quantities are extracted and analyzed. The specific outputs depend on the goals of the simulation.

* **Green's Functions:**
  * The raw Green's functions (retarded, advanced, lesser, greater) are often primary outputs. These can be stored in `Propagator::PropagatorMap` structures and might be written to file using utility functions (e.g., a hypothetical `print_Propagator` or similar).
  * They are complex functions of energy, momentum, and position (or site index).
* **Density of States (DOS):**
  * The DOS can be calculated from the retarded Green's function. `Greensolver` might have methods like `get_DOS_total_energy(...)` or functions to compute local or projected DOS.
  * Interpreting the DOS helps understand the available energy levels in the system.
* **Particle Density and Current:**
  * The lesser Green's function is used to calculate particle densities (e.g., electron density) and currents.
  * Functions like a hypothetical `calculate_density_from_G_lesser` or `calculate_current_from_G_lesser` would process the output from `Greensolver::do_solve_lesser(...)`.
* **Transmission and Conductance:**
  * For transport simulations, the transmission spectrum T(E) is a key output. This is often calculated using the Green's functions and lead self-energies (e.g., via the Fisher-Lee relation).
  * `Greensolver` might provide `Greensolver::get_transmission(name)` after the relevant calculations.
  * Conductance can then be derived from the transmission spectrum.
* **Self-Energies:**
  * The calculated self-energies themselves can be outputted for analysis, providing insight into the strength and nature of interactions or the coupling to leads.
* **Eigenvalues and Eigenvectors:**
  * If `Schroedinger::solve()` was called, the eigenvalues (band structure) and eigenvectors (wavefunctions) can be outputted and visualized.
* **Output Format:**
  * Results are typically written to text files (e.g., CSV, plain text tables) or binary files (e.g., HDF5, for large datasets like full Green's functions or volumetric data). These files can then be processed by external visualization and analysis tools (e.g., Python scripts with Matplotlib/Scipy, Gnuplot, Origin).

This workflow provides a flexible framework for simulating a wide range of quantum mechanical systems and phenomena. The modular design allows users to combine different components and models to address specific research questions.
