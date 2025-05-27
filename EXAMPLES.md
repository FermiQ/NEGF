# Conceptual Simulation Examples

## Note on Runnable Examples

Providing concrete, runnable examples for this simulation framework is challenging without:

1.  **A Build System:** The C++ source code (`Greensolver.cpp`, `Schroedinger.cpp`, etc.) needs to be compiled and linked against its dependencies (e.g., Armadillo, PETSc, MPI). A `CMakeLists.txt` or other build script would be required to automate this.
2.  **Example Input Files:** The framework likely relies on input files to define system parameters, geometry, and simulation settings. Without knowing the exact format of these input files or having sample data, creating a working example is not feasible.
3.  **Test Data/Expected Output:** To verify that an example is working correctly, one would need reference data or expected output values.

Therefore, the example below is **conceptual**. It illustrates how a user *might* interact with the framework and its components based on the documented functionalities and inferred input options. It is not runnable code but rather a descriptive walkthrough.

## Conceptual Example: 1D System - Retarded Green's Function

Let's imagine we want to simulate a simple 1D tight-binding chain and calculate its retarded Green's function at a single energy point.

**Goal:** Calculate the retarded Green's function G<sup>R</sup>(E) for a 1D chain.

**System:**
*   A 1D chain of 10 sites.
*   Tight-binding model with on-site energy (alpha) and nearest-neighbor hopping (beta).
*   No interactions or self-energies beyond a small broadening factor.
*   Calculation at a single energy point E = 0.0.

---

### Step 1: Input Parameter Configuration (Conceptual Input File)

A user would typically provide an input file (e.g., `input.ini` or `params.json`) that the framework parses. Based on `INPUT_OPTIONS.md`, this might look like:

```ini
# General Options
simulation_mode = "greens_function_calculation"
output_directory = "./simulation_output_1D_chain"
energy_min = 0.0
energy_max = 0.0
energy_points = 1
temperature = 0.0 # Not strictly needed for simple GR

# Propagation Options (minimal for a simple chain, k-space not explicitly used here)
Propagation::momentum_mesh_names = ["site_basis"] # Conceptual, as it's real space

# Schroedinger Component Options
Schroedinger::hamiltonian_model = "tight_binding_1D"
Schroedinger::material_parameters_file = "materials_1D.txt" # Contains alpha, beta
# materials_1D.txt might conceptually contain:
# onsite_energy_alpha = 0.0
# hopping_energy_beta = -1.0
Schroedinger::geometry_file = "geometry_1D.txt" # Defines chain length
# geometry_1D.txt might conceptually contain:
# num_sites = 10
Schroedinger::boundary_conditions_x = "open" # Or specific lead self-energies

# SelfEnergy Component Options
SelfEnergy::interaction_types = ["broadening"] # Just a simple broadening
SelfEnergy::broadening_eta = 0.01 # Small imaginary part for GR

# Greensolver Component Options
Greensolver::solver_type = "matrix_inversion" # For a small system
Greensolver::calculate_retarded_G = true
Greensolver::calculate_lesser_G = false
Greensolver::calculate_dos = false
Greensolver::calculate_transmission = false
```

---

### Step 2: Initialization and Setup

1.  **Main Program / Input Parsing:**
    *   The simulation executable is run with the input file.
    *   An internal `Options` class (or similar) parses `input.ini` and stores the parameters.

2.  **`Propagation` Class:**
    *   Although this is a real-space 1D chain, `Propagation` might still be initialized, perhaps to set up a simple basis or site indexing scheme if the framework is designed primarily for k-space. For this example, its role is minimal.
    *   `Propagation propagation_handler(options);`

3.  **`Schroedinger` Class:**
    *   `Schroedinger schroedinger_solver(options, propagation_handler);`
    *   `schroedinger_solver.setup_hamiltonian();`
        *   This reads `Schroedinger::hamiltonian_model` ("tight_binding_1D").
        *   It reads `materials_1D.txt` for alpha (on-site energy, e.g., 0.0) and beta (hopping energy, e.g., -1.0).
        *   It reads `geometry_1D.txt` for the number of sites (e.g., 10).
        *   It constructs the 10x10 Hamiltonian matrix (H) for the 1D chain. For example, diagonal elements are alpha, off-diagonal (i, i+1) and (i+1, i) are beta.
        *   Boundary conditions ("open") are applied.

4.  **`SelfEnergy` Class:**
    *   `SelfEnergy self_energy_handler(options, propagation_handler, schroedinger_solver);`
    *   `self_energy_handler.calculate_self_energy(energy_point);` (or initialized to provide a constant broadening)
        *   Reads `SelfEnergy::interaction_types` (sees "broadening").
        *   Creates a diagonal self-energy matrix (&Sigma;) where each diagonal element is `i * SelfEnergy::broadening_eta`.
        *   For more complex scenarios with leads, `initialize_lead_self_energy(...)` would be called here.

---

### Step 3: Solving for the Green's Function

1.  **`Greensolver` Class:**
    *   `Greensolver green_solver(options, schroedinger_solver, self_energy_handler, propagation_handler);`
    *   The simulation loop iterates through the energy points (here, only E=0.0).
    *   `arma::cx_mat retarded_G = green_solver.do_solve_retarded(energy_point);`
        *   Inside `do_solve_retarded`:
            *   It retrieves the Hamiltonian (H) from `schroedinger_solver`.
            *   It retrieves the self-energy matrix (&Sigma;) from `self_energy_handler` for the current `energy_point`.
            *   It computes the retarded Green's function using the formula:
                G<sup>R</sup>(E) = [ (E + i&eta;)I - H - &Sigma;<sup>R</sup>(E) ]<sup>-1</sup>
            *   Given `Greensolver::solver_type = "matrix_inversion"`, this involves constructing the matrix `M = (E + i*eta)I - H - Sigma` and then inverting it.
            *   The small `eta` comes from `SelfEnergy::broadening_eta` which is part of &Sigma;.

---

### Step 4: Output

1.  **Output Generation:**
    *   The calculated `retarded_G` (an Armadillo `arma::cx_mat`) would be saved to a file in the `output_directory`. The format could be plain text, binary, HDF5, etc.
    *   A utility function, possibly part of `Greensolver` or a dedicated `Output` class, would handle this.

---

This conceptual example outlines the flow and interaction of components for a basic simulation. More complex simulations (e.g., 2D/3D systems, self-consistent interactions, transport calculations with RGF methods) would involve more detailed configurations and more sophisticated use of the component functionalities.
