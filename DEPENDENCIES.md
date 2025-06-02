---
noIndex: true
---

# External Dependencies

This document lists potential external libraries and tools that the C++ simulation framework appears to depend on. This list is primarily inferred from common practices in scientific computing and specific mentions in the project's documentation context, rather than a direct scan of `#include` directives in the source code.

* **Armadillo:**
  * **Role:** C++ library for linear algebra and scientific computing. Used extensively for matrix and vector operations (e.g., `arma::mat`, `arma::cx_mat`).
  * **Likely Inferred From:** General use in C++ numerical codes; consistent with data structures mentioned in component documentation (e.g., `Greensolver.md`, `Schroedinger.md`).
* **PETSc (Portable, Extensible Toolkit for Scientific Computation):**
  * **Role:** Suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations. Used for parallel matrix and vector operations, and often as a foundation for linear and nonlinear solvers.
  * **Likely Inferred From:** Common in large-scale simulations; mentioned as a dependency for parallel complex matrices.
* **MPI (Message Passing Interface):**
  * **Role:** Standardized and portable message-passing system designed to function on a wide variety of parallel computing architectures. Used for distributed memory parallelization.
  * **Likely Inferred From:** Essential for running large simulations on clusters; often used with PETSc.
* **SLEPc (Scalable Library for Eigenvalue Problem Computations):**
  * **Role:** Software library for the solution of large-scale sparse eigenvalue problems on parallel computers. It is an extension of PETSc.
  * **Likely Inferred From:** Use in `Schroedinger.cpp` for solving eigenvalue problems (finding energy bands and wavefunctions) or by `Greensolver.cpp` if eigen-decomposition methods are employed.
* **MAGMA (Matrix Algebra on GPU and Multicore Architectures):**
  * **Role:** Provides a dense linear algebra library similar to LAPACK but for heterogeneous/hybrid architectures, including multicore Systems-on-Chip (SoCs) and systems with multiple GPUs.
  * **Likely Inferred From:** Mentioned as a potential dependency, possibly for GPU acceleration of specific computationally intensive parts of `Greensolver.cpp` or other components.
* **MKL (Intel Math Kernel Library):**
  * **Role:** Library of highly optimized math routines for science, engineering, and financial applications. Includes BLAS, LAPACK, ScaLAPACK, sparse solvers, Fast Fourier Transforms (FFT), vector math, and more.
  * **Likely Inferred From:** Common for high performance on Intel CPUs; suggested by potential `#include <mkl.h>`.
* **LibMesh:**
  * **Role:** C++ finite element library for the numerical simulation of partial differential equations using adaptive mesh refinement and coarsening (AMR/C) on serial and parallel platforms.
  * **Likely Inferred From:** Suggested by a potential `#include "libmesh.h"`. If used, it would likely be for defining the simulation geometry, meshing, and handling basis functions if a finite element approach is part of the `Schroedinger` solver.
* **Boost C++ Libraries:**
  * **Role:** Collection of peer-reviewed, high-quality C++ libraries that extend the functionality of C++. Could be used for various utilities.
  * **Likely Inferred From:** Common general-purpose libraries. Examples of usage:
    * `Boost.Filesystem`: For platform-independent path manipulation and file system operations.
    * `Boost.Algorithm`: For string algorithms, searching, sorting.
    * `Boost.Regex`: For regular expression parsing (perhaps in input file handling).
    * `Boost.Program_options`: For parsing command-line arguments and configuration files.
* **HDF5 (Hierarchical Data Format version 5):**
  * **Role:** Data model, library, and file format for storing and managing large amounts of numerical data.
  * **Likely Inferred From:** Common for outputting large datasets from scientific simulations (e.g., full Green's functions, volumetric data).
* **Standard C++ Libraries:**
  * **Role:** Core language features and utilities (e.g., `<vector>`, `<map>`, `<string>`, `<iostream>`, `<fstream>`, `<complex>`, `<cmath>`).
  * **Likely Inferred From:** Fundamental to any C++ project.

Installation and linking of these libraries are crucial for successfully compiling and running the simulation framework. Refer to the `BUILD_AND_RUN.md` document for general compilation guidance.
