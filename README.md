# HOW-MHD-Kokkos

A Kokkos-based development version of **HOW-MHD**, a high-order WENO-based magnetohydrodynamics (MHD) code for astrophysical fluid simulations.

This repository contains a C++/Kokkos implementation of core modules for solving ideal MHD equations using high-order finite-difference WENO reconstruction and strong-stability-preserving Runge--Kutta time integration.

> **Status:** This repository is an active development version. Some physics modules and production-level constrained-transport components are still being ported and tested.

---

## Overview

**HOW-MHD-Kokkos** is intended to provide a portable, performance-oriented implementation of the HOW-MHD algorithm using the Kokkos programming model.

Current code components include:

- Input parameter parsing
- Kokkos-based array allocation
- Initial condition setup
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
