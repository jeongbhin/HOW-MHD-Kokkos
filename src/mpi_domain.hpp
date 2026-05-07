#pragma once

#include "parameters.hpp"

// Set up the MPI Cartesian topology and overwrite par.nx/par.ny/par.nz
// with this rank's local active-zone size.
//
// Serial build behavior:
//   - No MPI calls are made.
//   - par.nx/par.ny/par.nz remain equal to par.gnx/par.gny/par.gnz.
//
// MPI build behavior (-DUSE_MPI):
//   - Creates a 3D Cartesian communicator.
//   - Uses input px/py/pz when provided; otherwise MPI_Dims_create fills them.
//   - Requires gnx,gny,gnz to be divisible by dims[0],dims[1],dims[2].
//   - Sets rank coordinates, neighbor ranks, and zero-based global offsets.
void setup_mpi_domain(Parameters& par);

// Release the Cartesian communicator when MPI is enabled.
void cleanup_mpi_domain(Parameters& par);

// True on rank 0 in MPI mode; always true in serial mode.
bool is_root_rank(const Parameters& par);


