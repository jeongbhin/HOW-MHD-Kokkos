#pragma once

#include <iostream>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

struct Parameters {
    // ============================================================
    // Grid size convention
    // ------------------------------------------------------------
    // gnx/gny/gnz : global number of active cells from the input file
    // nx/ny/nz    : local number of active cells on this MPI rank
    //
    // In serial mode, nx=gnx, ny=gny, nz=gnz.
    // In MPI mode, setup_mpi_domain(par) will overwrite nx/ny/nz.
    // ============================================================
    int gnx = 0, gny = 0, gnz = 0;
    int nx  = 0, ny  = 0, nz  = 0;

    int nstep = 0;
    int ntime = 0;
    int nt    = 0;

    double xsize = 1.0, ysize = 1.0, zsize = 1.0;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    double dt = 0.0;
    double dtdx = 0.0, dtdy = 0.0, dtdz = 0.0;

    double gam  = 5.0 / 3.0;
    double cour = 0.4;
    double pi   = 3.141592653589793;
    double t    = 0.0;
    double tend = 0.0;

    double rhomin = 1.0e-12;
    double pgmin  = 1.0e-12;

    std::string problem;
    std::string x1bc = "periodic";
    std::string x2bc = "periodic";
    std::string x3bc = "periodic";
    std::string output_format = "dat";

    // ============================================================
    // MPI / domain-decomposition parameters
    // ------------------------------------------------------------
    // px/py/pz are the requested MPI Cartesian layout.
    // If they are set to 0, MPI_Dims_create can fill them later.
    // For 4 nodes with 1 MPI rank per node, total ranks = 4, so
    // examples are px py pz = 2 2 1, or 4 1 1.
    // ============================================================
    int px = 0, py = 0, pz = 0;

    int rank      = 0;
    int nranks    = 1;
    int cart_rank = 0;

    int dims[3]    = {1, 1, 1};
    int periods[3] = {0, 0, 0};
    int coords[3]  = {0, 0, 0};

    int nbr_xm = -1, nbr_xp = -1;
    int nbr_ym = -1, nbr_yp = -1;
    int nbr_zm = -1, nbr_zp = -1;

    // Global starting index of this local block.
    // These are zero-based active-cell offsets.
    int istart = 0, jstart = 0, kstart = 0;

#ifdef USE_MPI
    MPI_Comm comm_cart = MPI_COMM_NULL;
#endif
};

void read_parameters(std::istream& in, Parameters& par);

// Finalize grid spacing and serial-domain defaults after read_parameters().
// In MPI mode, call this first, then call setup_mpi_domain(par), which will
// overwrite nx/ny/nz and offsets while preserving dx/dy/dz based on global sizes.
void finalize_parameters_after_read(Parameters& par);

