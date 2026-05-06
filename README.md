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
output_format
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
./bin/how-mhd < src/problems/mhd_rotor.in
```

---

### Running with Kokkos OpenMP threads

If HOW-MHD-Kokkos is built with the Kokkos OpenMP backend, the number of CPU threads can be controlled at runtime.

For example, to run with 128 OpenMP threads:

```bash
export OMP_NUM_THREADS=128
export KOKKOS_NUM_THREADS=128
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

./bin/how-mhd --kokkos-num-threads=128 < input.in
```

On a Slurm-based HPC system, a typical one-rank OpenMP run is

```bash
srun -N 1 -n 1 -c 128 --cpu-bind=cores \
  ./bin/how-mhd --kokkos-num-threads=128 < input.in
```

Here,

```text
-N 1   : use one node
-n 1   : use one MPI rank / one process
-c 128 : assign 128 CPU cores to the process
```

The code prints the active Kokkos execution space and concurrency at startup. A successful 128-thread OpenMP run should show something like

```text
Kokkos execution space: OpenMP
Kokkos concurrency: 128
```

If the concurrency is smaller than expected, check the runtime environment:

```bash
echo $OMP_NUM_THREADS
echo $KOKKOS_NUM_THREADS
echo $SLURM_CPUS_PER_TASK
echo $SLURM_JOB_CPUS_PER_NODE
env | grep -i kokkos
```

For large 2D/3D runs, using the Slurm `srun` form is recommended so that CPU allocation and binding are handled explicitly.
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

# ===== output =====
output_format vtk
```

---

## Output

HOW-MHD-Kokkos supports multiple output formats selected from the input file.

```text
output_format dat
```

or

```text
output_format vtk
```

### ASCII table output

If

```text
output_format dat
```

is selected, the code writes plain text dump files:

```text
bin/output/dump000000.dat
bin/output/dump000001.dat
bin/output/dump000002.dat
...
```

Each row corresponds to one active cell. The columns are

```text
i j k x y z rho Mx My Mz Bx By Bz E pg vx vy vz divB
```

where `divB` is a cell-centered diagnostic estimate of the magnetic-field divergence.

This format is useful for debugging, quick inspection, and small 2D tests.

### Binary VTK output

If

```text
output_format vtk
```

is selected, the code writes ParaView-readable binary VTK StructuredGrid files:

```text
bin/output/dump000000.vts
bin/output/dump000001.vts
bin/output/dump000002.vts
...
```

The `.vts` files contain appended raw binary data and can be opened directly with ParaView.

Currently written variables include:

```text
rho
pressure
energy
v2
B2
divB
velocity
magnetic_field
momentum
```

where

```text
velocity       = (vx, vy, vz)
magnetic_field = (Bx, By, Bz)
momentum       = (Mx, My, Mz)
v2             = |v|^2
B2             = |B|^2
```

This format is recommended for visualization and larger 2D/3D simulations.

### Example

To write binary VTK output, add the following line to the input file:

```text
# ===== output =====
output_format vtk
```

For debugging with plain text output, use

```text
# ===== output =====
output_format dat
```

### Visualization

The VTK output files can be opened directly in ParaView:

```text
File -> Open -> bin/output/dump000000.vts
```

For a sequence of dumps, open the `dump*.vts` series in ParaView.

A Python plotting script can also read the `.vts` files using the Python VTK package:

```bash
pip install vtk
```

or, on some HPC systems,

```bash
module avail vtk
module load vtk
```

For 3D simulations, the plotting script can extract 2D slices such as

```text
xy
xz
yz
```

from the binary VTK output.

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
