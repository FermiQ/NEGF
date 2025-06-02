---
noIndex: true
---

# PropagationUtilities

## Purpose

The `PropagationUtilities.cpp` file likely provides a collection of utility functions and helper classes that support the functionalities of `Propagation.cpp` and other components dealing with quantum mechanical propagation, momentum meshes, and related calculations. It encapsulates common operations and mathematical routines to avoid code duplication and improve modularity.

## Key Functionalities/Classes

This file might not contain a single primary class but rather a namespace or a static class with utility functions. Potential functionalities include:

* **Mesh Generation Helpers:**
  * `generate_uniform_mesh(...)`: Functions to create uniform momentum grids.
  * `generate_logarithmic_mesh(...)`: Functions to create logarithmic momentum grids.
  * `apply_symmetries_to_mesh(...)`: Functions to reduce the mesh size by applying symmetry operations.
* **Interpolation Functions:**
  * `interpolate_on_mesh(...)`: Functions to interpolate data defined on a momentum mesh.
* **Mathematical Operations:**
  * `calculate_distances(...)`: Functions to compute distances between momentum points.
  * `fourier_transform_utility(...)`: Helper functions for performing Fourier transforms between real and momentum space.
* **Coordinate Transformations:**
  * `cartesian_to_spherical(...)`: Functions to convert momentum coordinates between different systems.
  * `transform_mesh_basis(...)`: Functions to change the basis of a momentum mesh.
* **Input/Output Helpers:**
  * `read_mesh_from_file(...)`: Functions to load mesh data from a file.
  * `write_mesh_to_file(...)`: Functions to save mesh data to a file.

If there are classes, they might be small helper classes like:

* `MeshSymmetrizer`: A class to handle symmetry operations on meshes.
* `Interpolator1D/2D/3D`: Classes for performing interpolation of data on meshes.

## Data Structures

The data structures in this file are likely to be standard C++ containers or basic structures used to pass data to and from the utility functions:

* `std::vector<double>` or `arma::vec`: To represent lists of momentum coordinates or values.
* `std::vector<std::vector<double>>` or `arma::mat`: To represent entire meshes or collections of momentum vectors.
* Possibly simple `struct`s or `class`es to hold parameters for mesh generation or other utility operations.

## Dependencies

`PropagationUtilities.cpp` is likely to be relatively self-contained for core utilities but might depend on:

* Armadillo library (`armadillo`): For vector and matrix operations if it performs numerical calculations.
* Standard C++ libraries (e.g., `<vector>`, `<cmath>`, `<fstream>`, `<iostream>`).
* It is primarily _used by_ other files like `Propagation.cpp`, `Greensolver.cpp`, and `SelfEnergy.cpp` rather than depending on them. It provides services to these higher-level components.
* Possibly `Input.h` if some utilities are configured via input files.
