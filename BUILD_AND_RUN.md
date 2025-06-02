---
noIndex: true
---

# Building and Running Simulations

This document provides general guidance on how one might compile the C++ source code for this simulation framework and how to run a simulation.

## 1. Building the Code

**Note:** No specific build system (e.g., CMakeLists.txt, Makefile) is provided with this set of source files. The following instructions provide a general example of how one might compile the code manually using a C++ compiler like g++. The exact commands will depend on your system, compiler, and the locations of the required libraries.

### Dependencies

The C++ source files appear to use several external libraries for numerical computation and parallel processing. You would need to have these libraries installed on your system. Based on typical include headers found in such projects, these may include:

* **PETSc:** For solving partial differential equations and linear algebra, especially in parallel.
* **MPI (Message Passing Interface):** For distributed memory parallelization (e.g., MPICH, OpenMPI).
* **SLEPc:** For solving large-scale eigenvalue problems (often used with PETSc).
* **Armadillo:** For linear algebra and scientific computing in C++.
* **MAGMA (Matrix Algebra on GPU and Multicore Architectures):** For GPU-accelerated linear algebra routines.
* **Intel MKL (Math Kernel Library):** For optimized math routines, including BLAS, LAPACK.

### Example Compilation Command (g++)

Here's a conceptual example of how you might compile the source files using `g++`. You will need to list all the `.cpp` files that are part of the project.

```bash
g++ -std=c++11 -O2 \
    Greensolver.cpp \
    Propagation.cpp \
    PropagationUtilities.cpp \
    Schroedinger.cpp \
    Self_energy.cpp \
    ScatteringBackwardRGFSolver.cpp \
    ScatteringForwardRGFSolver.cpp \
    # Add any other .cpp files here (e.g., main.cpp, input_parser.cpp) \
    -o nemo_simulation \
    -I/path/to/armadillo/include \
    -I/path/to/petsc/include \
    -I/path/to/slepc/include \
    # Add other include paths as necessary (e.g., for MAGMA, MKL) \
    -L/path/to/armadillo/lib -larmadillo \
    -L/path/to/petsc/lib -lpetsc \
    -L/path/to/slepc/lib -lslepc \
    # Add other library paths and linker flags for MPI, MAGMA, MKL, BLAS, LAPACK
    # For example, for MKL: -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
    # For MPI (this might be handled by using mpicxx as the compiler wrapper): -lmpich or -lopen-mpi
```

**Important Considerations:**

* **Compiler:** Instead of `g++` directly, if you are using MPI, you would typically use an MPI wrapper compiler like `mpicxx` which handles MPI headers and libraries automatically.
  * Example: `mpicxx -std=c++11 -O2 ... -o nemo_simulation ... -larmadillo -lpetsc -lslepc ...`
* **File List:** Ensure all necessary `.cpp` files are included in the compilation command. A `main.cpp` file containing the `main()` function to drive the simulation would be essential.
* **Paths:** Replace `/path/to/...` with the actual installation paths of the libraries on your system.
* **Library Linking:** The `-l` flags (e.g., `-lpetsc`, `-larmadillo`) link against the libraries. The order of libraries can sometimes matter. You may also need to link against BLAS and LAPACK if not included with Armadillo or MKL.
* **Optimization Flags:** `-O2` or `-O3` are common optimization flags. `-g` can be added for debugging symbols.
* **C++ Standard:** `-std=c++11` (or a newer standard like `c++14`, `c++17`) might be required depending on the C++ features used.

A proper `Makefile` or `CMakeLists.txt` file is highly recommended for managing compilation in a real project, as it automates dependency tracking and simplifies the build process.

## 2. Running a Simulation

Once the code is successfully compiled into an executable (e.g., `nemo_simulation`), you would typically run it from the command line.

### Single Process Execution

If the simulation is intended to run on a single processor or if MPI is not used/needed for a particular run:

```bash
./nemo_simulation input.dat
```

* `./nemo_simulation`: This is the compiled executable.
* `input.dat`: This is a hypothetical input file. The actual name and format of this file would depend on how the program parses its inputs (as described conceptually in `INPUT_OPTIONS.md`). This file would contain the parameters controlling the simulation.

### Parallel Execution (MPI)

If the program has been compiled with MPI support for distributed memory parallelization, you would use `mpirun` or `mpiexec` to launch it:

```bash
mpirun -np 4 ./nemo_simulation input.dat
```

* `mpirun`: The command to run MPI applications.
* `-np 4`: Specifies the number of processes to launch (e.g., 4 processes). This should generally not exceed the number of available CPU cores for CPU-bound tasks.
* `./nemo_simulation`: The executable.
* `input.dat`: The input parameter file, which all MPI processes would typically read.

The simulation framework would need to be designed internally to distribute the workload across the MPI processes (e.g., PETSc and SLEPc handle much of this for their respective tasks).

Output files would be generated in the directory specified in the input file or the current working directory, as designed in the framework. The nature of parallel output (all processes write to one file, or each process writes its own files) also depends on the framework's design.
