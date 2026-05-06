# HOW-MHD-Kokkos

A Kokkos-based development version of **HOW-MHD**, a high-order WENO-based magnetohydrodynamics (MHD) code for astrophysical fluid simulations.

This repository contains a C++/Kokkos implementation of core modules for solving ideal MHD equations using high-order finite-difference WENO reconstruction and strong-stability-preserving Runge--Kutta time integration.

> **Status:** This repository is an active development version. Some physics modules and production-level components are still being ported and tested.

---

## Overview

**HOW-MHD-Kokkos** is intended to provide a portable, performance-oriented implementation of the HOW-MHD algorithm using the Kokkos programming model.

Current code components include:

- Input parameter parsing
- Kokkos-based array allocation
- Problem-dependent initial condition setup
- Boundary condition routines
- CFL timestep calculation
- Primitive variable conversion
- MHD eigenstructure calculation
- Finite-difference WENO flux reconstruction
- SSPRK stage driver
- Directional sweep infrastructure

The long-term goal is to provide a portable high-order MHD solver that can run efficiently on modern CPU and accelerator-based HPC systems.

---

## Numerical Method

The code is based on the following numerical ingredients:

- Ideal adiabatic MHD equations
- Conservative variables
- Fifth-order finite-difference WENO-type reconstruction
- Local Lax--Friedrichs flux splitting
- Characteristic decomposition using MHD eigenvectors
- Multi-stage SSPRK time integration
- CFL timestep based on fast magnetosonic wave speeds
- Ghost-cell based boundary conditions

The conserved variable ordering used in the current implementation is

```text
0 : density
1 : x-momentum
2 : y-momentum
3 : z-momentum
4 : Bx
5 : By
6 : Bz
7 : total energy
```

---

## Repository Structure

The current source layout is organized under `src/`:

```text
HOW-MHD-Kokkos/
├── Makefile
├── README.md
├── .gitignore
└── src/
    ├── main.cpp
    ├── parameters.cpp
    ├── parameters.hpp
    ├── bound.cpp
    ├── bound.hpp
    ├── tstep.cpp
    ├── tstep.hpp
    ├── ssprk.cpp
    ├── ssprk.hpp
    └── ...
    └── problems/
        ├── brio_wu.cpp
        ├── mhd_rotor.hpp
        └── ...
```

Problem-specific initial conditions are placed under

```text
src/problems/
```

For example, the MHD rotor setup is implemented as one of the sample problems.

---

## Main Modules

### `main.cpp`

Main driver of the code.

The driver typically performs the following steps:

1. Initialize Kokkos
2. Read runtime parameters
3. Compute grid spacing
4. Allocate the main conserved-variable array
5. Initialize the selected problem
6. Apply boundary conditions
7. Compute timestep
8. Advance the solution with the SSPRK/WENO solver
9. Finalize Kokkos

---

### `parameters.cpp` / `parameters.hpp`

Reads runtime parameters from standard input.

Example parameters include:

```text
problem
nx
ny
nz
nt
xsize
ysize
zsize
gam
cour
tend
rhomin
pgmin
x1bc
x2bc
x3bc
```

Lines beginning with `#` are treated as comments.

---

### `src/problems/`

Contains problem-dependent initial condition setups.

Example:

```text
problem mhd_rotor
```

The selected problem is specified in the input file using the `problem` keyword.

---

### `bound.cpp` / `bound.hpp`

Applies ghost-cell boundary conditions.

Currently supported boundary options include:

```text
periodic
open
```

The implementation is under active development.

---

### `tstep.cpp` / `tstep.hpp`

Computes the timestep using a CFL condition based on the maximum characteristic speeds in the x, y, and z directions.

---

### `ssprk.cpp` / `ssprk.hpp`

Contains the SSPRK time integration infrastructure and directional sweep routines.

This module includes:

- RK coefficient setup
- Stage copying
- x/y/z directional sweeps
- Primitive-variable conversion
- MHD eigenstructure calculation
- WENO flux update

Some routines are still under active development.

---

## Build

This code requires a C++ compiler and Kokkos.

Example build command:

```bash
make
```

The executable is expected to be generated as

```bash
bin/how-mhd
```

depending on the Makefile configuration.

---

## Run

Run the code with an input file through standard input:

```bash
./bin/how-mhd < input.in
```

For example:

```bash
./bin/how-mhd < mhd_rotor.in
```

If the input files are stored elsewhere, adjust the path accordingly.

---

## Example Input: MHD Rotor

```text
problem mhd_rotor

# ===== grid setup =====
nx 128
ny 128
nz 1
nt 10

xsize 1.0
ysize 1.0
zsize 1.0

# ===== constants =====
gam 1.6666666666667
cour 1.5
tend 0.15

rhomin 1.0e-12
pgmin 1.0e-12

# ===== boundary =====
x1bc open
x2bc open
x3bc open
```

---

## Development Notes

This repository is currently intended for code development, testing, and porting of HOW-MHD components to Kokkos.

Known active-development items include:

- Completing multidimensional boundary conditions
- Adding the full constrained-transport magnetic-field update
- Adding more production initial conditions
- Adding output routines
- Adding regression and verification tests
- Improving GPU portability and performance
- Cleaning up module interfaces

---

## Citation

If you use this code or the HOW-MHD method, please cite:

```text
Seo, J. & Ryu, D. 2023,
"HOW-MHD: A High-order WENO-based Magnetohydrodynamic Code with a
High-order Constrained Transport Algorithm for Astrophysical Applications",
The Astrophysical Journal, 953, 39.
doi:10.3847/1538-4357/acdf4b
```

---

## Reference

The original HOW-MHD method is described in:

Seo, J. & Ryu, D. 2023, The Astrophysical Journal, 953, 39.

---

## Author

Jeongbhin Seo
