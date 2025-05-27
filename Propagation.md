# Propagation

## Purpose

The `Propagation.cpp` file implements the `Propagation` class, which serves as a foundational component for managing momentum meshes and `Propagator` objects within the simulation framework. It provides the basic structures and functionalities necessary for describing how quantum states evolve or propagate through the system.

## Key Functionalities/Classes

The primary class in this file is `Propagation`.

Key methods and functionalities likely include:

*   `Propagation(...)`: Constructor, responsible for initializing the momentum meshes (e.g., k-mesh, q-mesh) based on input parameters. This might involve setting up uniform or non-uniform grids in momentum space.
*   `get_k_mesh()`: Returns the primary momentum mesh (k-mesh) used for electron states, for instance.
*   `get_q_mesh()`: Returns a secondary momentum mesh (q-mesh), possibly used for interactions or collective modes.
*   `get_propagator(double energy, ...)`: Retrieves or constructs a `Propagator` object for a given energy and other relevant quantum numbers.
*   `add_propagator(...)`: Adds a pre-computed or externally defined `Propagator` to its internal storage.
*   `initialize_meshes(...)`: A method to set up the dimensions, boundaries, and discretization of various momentum meshes.
*   `map_momentum_to_index(...)`: Utility function to map a momentum vector to an index in the discretized mesh, and vice-versa.

The `Propagator` class (which might be defined in `Propagator.h` but heavily used and managed by `Propagation.cpp`) is central here. A `Propagator` object typically encapsulates information about how a particle moves from one state to another, often represented as a matrix in momentum or real space.

## Data Structures

*   `MomentumMesh`: A custom data structure to represent discretized momentum space. This could be a class or struct holding the grid points, weights, and potentially symmetry information. There might be multiple instances (e.g., `k_mesh_`, `q_mesh_`).
*   `Propagator::PropagatorMap` (likely defined in `Propagator.h`): A map, possibly `std::map<KeyType, Propagator>`, used to store and manage `Propagator` objects. The `KeyType` could be energy, spin, or a combination of quantum numbers.
*   `arma::vec` or `std::vector<double>`: Used to store the coordinates of momentum points within the meshes.
*   `arma::mat` or `arma::cx_mat`: Armadillo matrices used within `Propagator` objects to represent the propagator itself.

## Dependencies

The `Propagation` class and its implementation in `Propagation.cpp` are expected to interact with or depend on:

*   `PropagationUtilities.h`: For utility functions related to mesh generation, manipulation, or symmetry operations.
*   `Propagator.h`: For the definition of the `Propagator` class and associated types like `PropagatorMap`.
*   `Input.h` (or a specific input class like `PropagationInput.h`): To read parameters for mesh generation and other settings.
*   Standard C++ libraries (e.g., `<vector>`, `<map>`).
*   Armadillo library (`armadillo`): For numerical vector and matrix operations.
*   Possibly MPI, if mesh initialization or `Propagator` storage is distributed.
