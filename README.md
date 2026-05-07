# HOW-MHD-Kokkos

A C++/Kokkos development version of **HOW-MHD**, a high-order WENO-based magnetohydrodynamics (MHD) code for astrophysical fluid simulations.

This repository contains a portable CPU-oriented implementation of the core HOW-MHD algorithm using **MPI + Kokkos/OpenMP**.  The current version supports 3D MPI domain decomposition, Kokkos/OpenMP threading within each MPI rank, high-order finite-difference WENO reconstruction, SSPRK time integration, and a high-order constrained-transport (CT) magnetic-field update.

> **Status:** Active development version.  The current code is suitable for method development, MPI/Kokkos testing, and 2D/3D MHD experiments.  Additional verification tests, production I/O improvements, and performance tuning are still ongoing.

---

## Overview

**HOW-MHD-Kokkos** is intended to provide a portable, performance-oriented implementation of the HOW-MHD algorithm using Kokkos and MPI.

The current implementation includes:

- Runtime input parameter parsing
- Global/local grid separation for MPI domain decomposition
- 3D MPI Cartesian topology
- MPI halo exchange for conserved variables
- MPI halo exchange for CT magnetic-field arrays and CT electromotive-force/flux arrays
- Kokkos-based array allocation and OpenMP execution
- Problem-dependent initial condition registry
- CFL timestep calculation with MPI global maximum reduction
- Primitive variable conversion
- MHD eigenstructure calculation
- Finite-difference WENO-type flux reconstruction
- SSPRK(5,4) stage driver
- Workspace-optimized x/y/z directional sweeps
- High-order constrained-transport magnetic-field update
- Rank-wise VTS output for parallel runs
- Python plotting scripts for rank-wise `.vts` output

The long-term goal is to provide a portable high-order MHD solver that can run efficiently on modern CPU and accelerator-based HPC systems.

---

## Numerical Method

The code is based on the following numerical ingredients:

- Ideal adiabatic MHD equations
- Conservative finite-difference formulation
- Fifth-order finite-difference WENO-type reconstruction
- Local Lax--Friedrichs flux splitting
- Characteristic decomposition using MHD eigenvectors
- Five-stage fourth-order SSPRK time integration, SSPRK(5,4)
- High-order constrained transport for maintaining magnetic-field consistency
- CFL timestep based on fast magnetosonic wave speeds
- MPI global timestep reduction
- Ghost-cell based boundary conditions and MPI halo exchange

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

The current banner/method summary is

```text
WENO5  |  SSPRK(5,4)  |  4th-order CT  |  MPI + Kokkos
```

---

## Repository Structure

The current source layout is organized as follows:

```text
HOW-MHD-Kokkos/
├── Makefile
├── README.md
├── .gitignore
├── input*.txt
├── scripts/
│   ├── plot_vts_slice.py
│   └── plot_vts_slice_fast_spyder.py
└── src/
    ├── main.cpp
    ├── parameters.cpp
    ├── parameters.hpp
    ├── mpi_domain.cpp
    ├── mpi_domain.hpp
    ├── bound.cpp
    ├── bound.hpp
    ├── tstep.cpp
    ├── tstep.hpp
    ├── ssprk.cpp
    ├── ssprk.hpp
    ├── fluxct.cpp
    ├── fluxct.hpp
    ├── output.cpp
    ├── output.hpp
    ├── problem.cpp
    ├── problem.hpp
    ├── prot.cpp
    ├── prot.hpp
    └── problems/
        ├── blast3d.cpp
        ├── mpi_smooth3d.cpp
        ├── brio_wu.cpp
        ├── mhd_rotor.cpp
        ├── orszag_tang.cpp
        ├── uniform.cpp
        └── ...
```

Problem-specific initial conditions are placed under

```text
src/problems/
```

The selected problem is specified in the input file using the `problem` keyword.

---

## Main Modules

### `main.cpp`

Main driver of the code.

The driver performs the following steps:

1. Initialize MPI when compiled with `USE_MPI`
2. Initialize Kokkos
3. Read runtime parameters from an input file
4. Finalize global grid spacing and runtime parameters
5. Set up MPI Cartesian domain decomposition
6. Allocate the main conserved-variable array
7. Initialize the selected problem
8. Apply initial boundary conditions
9. Allocate and initialize CT magnetic-field arrays
10. Advance the solution with SSPRK/WENO/CT
11. Write output
12. Clean up Kokkos and MPI

The preferred run mode is now

```bash
./bin/how_mhd_mpi input.in
```

rather than stdin redirection.  Passing the input filename explicitly ensures that every MPI rank opens the same input file.

---

### `parameters.cpp` / `parameters.hpp`

Reads runtime parameters from an input stream.

The input grid size

```text
nx
ny
nz
```

is interpreted as the **global active grid size**.  The local grid size is computed after MPI domain setup.

Common parameters include:

```text
problem
nx
ny
nz
px
py
pz
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

### `mpi_domain.cpp` / `mpi_domain.hpp`

Sets up the MPI domain decomposition.

The code uses a 3D Cartesian MPI topology.  The input parameters

```text
px
py
pz
```

specify the MPI rank layout.  The product must match the number of MPI ranks:

```text
px * py * pz = number of MPI ranks
```

For example,

```text
px 2
py 2
pz 1
```

requires 4 MPI ranks.

Each rank stores its local active grid plus three ghost cells on each side.

---

### `bound.cpp` / `bound.hpp`

Applies ghost-cell boundary conditions and MPI halo exchange for conserved variables.

Currently supported physical boundary options include:

```text
periodic
open
```

For MPI runs, rank boundaries are filled using halo exchange.  Physical boundary conditions are applied only at global domain boundaries.

---

### `tstep.cpp` / `tstep.hpp`

Computes the timestep using a CFL condition based on the maximum fast magnetosonic speeds in the x, y, and z directions.

For MPI runs, local maximum wave speeds are reduced with an MPI global maximum so that all ranks use the same timestep.

---

### `ssprk.cpp` / `ssprk.hpp`

Contains the SSPRK time integration infrastructure and directional sweep routines.

This module includes:

- RK coefficient setup
- Stage copying
- Workspace-optimized x/y/z directional sweeps
- Primitive-variable conversion
- MHD eigenstructure calculation
- WENO flux update
- Coupling to CT flux accumulation

The current optimized version keeps the fixed sweep order

```text
x -> y -> z
```

and uses workspace-based directional sweeps to reduce excessive line-by-line Kokkos kernel launch overhead.

---

### `fluxct.cpp` / `fluxct.hpp`

Contains the high-order constrained-transport update.

The current MPI-aware CT implementation includes halo exchange for:

```text
bxb, byb, bzb
fsy, fsz, gsx, gsz, hsx, hsy
Ox2, Oy2, Oz2
Ox, Oy, Oz
```

This is necessary for correct magnetic-field evolution across MPI rank boundaries.

---

### `src/problems/`

Contains problem-dependent initial condition setups.

Current examples include:

```text
problem uniform
problem mpi_smooth3d
problem blast3d
problem brio_wu
problem mhd_rotor
problem orszag_tang
```

`mpi_smooth3d` is a smooth periodic 3D test problem useful for checking MPI decomposition, halo exchange, and rank-wise output.

`blast3d` is a 3D MHD blast-wave setup useful for testing nonlinear MHD evolution and CT behavior.

---

## Build

This code requires:

- C++20-capable compiler
- MPI compiler wrapper, such as `mpicxx`, for MPI builds
- Kokkos installed with the OpenMP backend

Example Kokkos installation:

```bash
cd ~/codes

git clone https://github.com/kokkos/kokkos.git
cd kokkos

cmake -B build -S . \
  -DCMAKE_INSTALL_PREFIX=$HOME/codes/kokkos-install \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DCMAKE_CXX_STANDARD=20

cmake --build build -j 16
cmake --install build
```

Set the Kokkos installation path:

```bash
export KOKKOS_ROOT=$HOME/codes/kokkos-install
```

To make this persistent:

```bash
echo 'export KOKKOS_ROOT=$HOME/codes/kokkos-install' >> ~/.bashrc
source ~/.bashrc
```

Build the MPI + Kokkos/OpenMP executable:

```bash
make clean
make mpi KOKKOS_ROOT=$KOKKOS_ROOT
```

The executable is generated as

```text
bin/how_mhd_mpi
```

A serial/Kokkos-only build can be generated with

```bash
make clean
make serial KOKKOS_ROOT=$KOKKOS_ROOT
```

which produces

```text
bin/how_mhd
```

---

## Run

### MPI input style

For MPI runs, pass the input filename explicitly:

```bash
./bin/how_mhd_mpi input.in
```

This is preferred over

```bash
./bin/how_mhd_mpi < input.in
```

because stdin redirection may not be delivered reliably to every MPI rank under `srun`.

---

### One-rank smoke test

```bash
cd bin

export OMP_NUM_THREADS=8
export KOKKOS_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

./how_mhd_mpi input_mpi_smooth3d_1rank.txt
```

---

### Four-rank MPI test on one node

Example input layout:

```text
px 2
py 2
pz 1
```

Run:

```bash
cd bin

export OMP_NUM_THREADS=8
export KOKKOS_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

srun -N 1 -n 4 --cpus-per-task=8 --cpu-bind=cores \
  ./how_mhd_mpi input_mpi_smooth3d_4rank.txt
```

---

### Eight-rank 3D MPI test

Example input layout:

```text
px 2
py 2
pz 2
```

Run:

```bash
cd bin

export OMP_NUM_THREADS=4
export KOKKOS_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

srun -N 1 -n 8 --cpus-per-task=4 --cpu-bind=cores \
  ./how_mhd_mpi input_mpi_smooth3d_8rank.txt
```

---

### Four-node hybrid MPI + Kokkos/OpenMP run

A typical 4-node hybrid run using one MPI rank per node and 128 OpenMP threads per rank is

```bash
cd bin

export OMP_NUM_THREADS=128
export KOKKOS_NUM_THREADS=128
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

srun -N 4 -n 4 --ntasks-per-node=1 \
  --cpus-per-task=128 --cpu-bind=cores \
  ./how_mhd_mpi input_blast3d_4rank.txt
```

For some systems, stdout may be block-buffered under `srun`.  The code now flushes key logs, but interactive testing can also use

```bash
stdbuf -oL -eL srun -N 4 -n 4 --ntasks-per-node=1 \
  --cpus-per-task=128 --cpu-bind=cores \
  ./how_mhd_mpi input_blast3d_4rank.txt
```

---

### Alternative rank/thread layouts

Depending on the problem size and hardware, using more MPI ranks with fewer OpenMP threads per rank may be faster than using one MPI rank per node.

Examples:

```bash
# 4 nodes x 2 ranks/node x 64 threads/rank = 512 cores
export OMP_NUM_THREADS=64
export KOKKOS_NUM_THREADS=64

srun -N 4 -n 8 --ntasks-per-node=2 \
  --cpus-per-task=64 --cpu-bind=cores \
  ./how_mhd_mpi input_blast3d_8rank.txt
```

```bash
# 4 nodes x 4 ranks/node x 32 threads/rank = 512 cores
export OMP_NUM_THREADS=32
export KOKKOS_NUM_THREADS=32

srun -N 4 -n 16 --ntasks-per-node=4 \
  --cpus-per-task=32 --cpu-bind=cores \
  ./how_mhd_mpi input_blast3d_16rank.txt
```

The input file must satisfy

```text
px * py * pz = total MPI ranks
```

---

## Example Input: 3D Smooth MPI Test

```text
problem mpi_smooth3d
output_format vts

nx 32
ny 32
nz 32

px 2
py 2
pz 1

xsize 1.0
ysize 1.0
zsize 1.0

gam 1.6666666666666667
cour 0.3
tend 0.01
nt 1

rhomin 1.0e-12
pgmin 1.0e-12

x1bc periodic
x2bc periodic
x3bc periodic
```

---

## Example Input: 3D MHD Blast Wave

```text
problem blast3d
output_format vts

nx 128
ny 128
nz 128

px 2
py 2
pz 1

xsize 1.0
ysize 1.0
zsize 1.0

gam 1.6666666666666667
cour 0.25
tend 0.05
nt 5

rhomin 1.0e-10
pgmin 1.0e-10

x1bc periodic
x2bc periodic
x3bc periodic
```

---

## Output

HOW-MHD-Kokkos supports output formats selected from the input file:

```text
output_format dat
```

or

```text
output_format vts
```

The current MPI version writes one output file per MPI rank.

---

### ASCII table output

If

```text
output_format dat
```

is selected, the code writes plain text dump files.

For MPI runs, files are written as

```text
bin/output/dump000000_rank000000.dat
bin/output/dump000000_rank000001.dat
...
```

Each row corresponds to one active cell.  The columns include local/global indices, coordinates, conserved variables, primitive variables, and diagnostic fields.

This format is useful for debugging, quick inspection, and small tests.

---

### Binary VTS output

If

```text
output_format vts
```

is selected, the code writes ParaView-readable binary VTK StructuredGrid files.

For MPI runs, files are written as

```text
bin/output/dump000000_rank000000.vts
bin/output/dump000000_rank000001.vts
bin/output/dump000000_rank000002.vts
...
```

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

The rank-wise VTS output can be loaded in ParaView file-by-file.  A `.pvts` master-file writer is a planned convenience improvement.

---

## Visualization

Python plotting scripts are provided under

```text
scripts/
```

For rank-wise VTS files written under

```text
bin/output/
```

the fast Spyder-friendly script can be used to plot 2D slices without constructing a full merged 3D mesh:

```bash
python scripts/plot_vts_slice_fast_spyder.py
```

The script reads files such as

```text
bin/output/dump000001_rank000000.vts
bin/output/dump000001_rank000001.vts
...
```

and writes PNG figures to

```text
figures/
```

Common scalar fields include:

```text
rho
pressure
energy
divB
v_abs
B_abs
Bx
By
Bz
vx
vy
vz
```

---

## Development Notes

This repository is currently intended for code development, testing, and porting of HOW-MHD components to Kokkos and MPI.

Recent development items include:

- 3D MPI Cartesian domain decomposition
- MPI halo exchange for conserved variables
- MPI-aware constrained-transport magnetic-field update
- MPI halo exchange for CT flux and EMF arrays
- Rank-wise VTS output
- 3D smooth periodic MPI test problem
- 3D MHD blast-wave problem
- Workspace-optimized directional sweeps in `ssprk.cpp`
- Fast Python plotting utilities for rank-wise VTS output

Known active-development items include:

- Adding `.pvts` master-file output for ParaView
- Adding `output_format none` for performance testing
- Adding regression tests for 1-rank vs multi-rank consistency
- Adding global diagnostics such as min/max density, pressure, magnetic field, and `max|divB|`
- Improving performance through further sweep and memory-layout optimization
- Investigating GPU portability and performance
- Cleaning up module interfaces

---

## Citation

If you use this code or the HOW-MHD method, please cite:

```text
Seo, J. & Ryu, D. 2023,
"HOW-MHD: A High-order WENO-based Magnetohydrodynamic Code with a
High-order Constrained Transport Algorithm for Astrophysical Applications",
The Astrophysical Journal, 953, 39.
doi:10.3847/1538-4357/acdfc7
```

---

## Reference

The original HOW-MHD method is described in:

Seo, J. & Ryu, D. 2023, The Astrophysical Journal, 953, 39.

---

## Author

Jeongbhin Seo

