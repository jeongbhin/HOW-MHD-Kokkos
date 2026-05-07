#include "mpi_domain.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void die_serial(const std::string& msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(EXIT_FAILURE);
}

#ifdef USE_MPI
void die_mpi(Parameters& par, const std::string& msg) {
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (initialized) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "Error: " << msg << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } else {
        die_serial(msg);
    }
}
#endif

bool is_periodic_bc(const std::string& bc) {
    return bc == "periodic" || bc == "Periodic" || bc == "PERIODIC";
}

void validate_positive_global_grid(const Parameters& par) {
    if (par.gnx <= 0 || par.gny <= 0 || par.gnz <= 0) {
        die_serial("global grid sizes gnx/gny/gnz must be positive. Check nx, ny, nz in the input file.");
    }
}

} // namespace

bool is_root_rank(const Parameters& par) {
#ifdef USE_MPI
    return par.rank == 0;
#else
    (void)par;
    return true;
#endif
}

void setup_mpi_domain(Parameters& par) {
    validate_positive_global_grid(par);

#ifndef USE_MPI
    // Serial fallback. Keep local grid equal to global grid.
    par.rank = 0;
    par.nranks = 1;
    par.cart_rank = 0;

    par.dims[0] = 1;
    par.dims[1] = 1;
    par.dims[2] = 1;

    par.periods[0] = is_periodic_bc(par.x1bc) ? 1 : 0;
    par.periods[1] = is_periodic_bc(par.x2bc) ? 1 : 0;
    par.periods[2] = is_periodic_bc(par.x3bc) ? 1 : 0;

    par.coords[0] = 0;
    par.coords[1] = 0;
    par.coords[2] = 0;

    par.nbr_xm = par.nbr_xp = -1;
    par.nbr_ym = par.nbr_yp = -1;
    par.nbr_zm = par.nbr_zp = -1;

    par.istart = 0;
    par.jstart = 0;
    par.kstart = 0;

    par.nx = par.gnx;
    par.ny = par.gny;
    par.nz = par.gnz;

    if (par.px > 1 || par.py > 1 || par.pz > 1) {
        std::cerr << "Warning: px/py/pz were provided, but this is a serial build. "
                  << "Ignoring MPI layout and using one rank.\n";
    }
    return;
#else
    MPI_Comm_rank(MPI_COMM_WORLD, &par.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par.nranks);

    int dims[3] = {par.px, par.py, par.pz};

    if (dims[0] < 0 || dims[1] < 0 || dims[2] < 0) {
        die_mpi(par, "px, py, pz must be non-negative. Use 0 to let MPI_Dims_create choose a dimension.");
    }

    const int fixed_product =
        (dims[0] > 0 ? dims[0] : 1) *
        (dims[1] > 0 ? dims[1] : 1) *
        (dims[2] > 0 ? dims[2] : 1);

    if (fixed_product > par.nranks || par.nranks % fixed_product != 0) {
        die_mpi(par, "requested px*py*pz is incompatible with the number of MPI ranks.");
    }

    MPI_Dims_create(par.nranks, 3, dims);

    par.dims[0] = dims[0];
    par.dims[1] = dims[1];
    par.dims[2] = dims[2];

    par.periods[0] = is_periodic_bc(par.x1bc) ? 1 : 0;
    par.periods[1] = is_periodic_bc(par.x2bc) ? 1 : 0;
    par.periods[2] = is_periodic_bc(par.x3bc) ? 1 : 0;

    if (par.gnx % par.dims[0] != 0 ||
        par.gny % par.dims[1] != 0 ||
        par.gnz % par.dims[2] != 0) {
        die_mpi(par,
                "global grid must be divisible by MPI layout: "
                "gnx%px, gny%py, and gnz%pz must all be zero.");
    }

    const int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD,
                    3,
                    par.dims,
                    par.periods,
                    reorder,
                    &par.comm_cart);

    if (par.comm_cart == MPI_COMM_NULL) {
        die_mpi(par, "MPI_Cart_create returned MPI_COMM_NULL.");
    }

    MPI_Comm_rank(par.comm_cart, &par.cart_rank);
    MPI_Cart_coords(par.comm_cart, par.cart_rank, 3, par.coords);

    MPI_Cart_shift(par.comm_cart, 0, 1, &par.nbr_xm, &par.nbr_xp);
    MPI_Cart_shift(par.comm_cart, 1, 1, &par.nbr_ym, &par.nbr_yp);
    MPI_Cart_shift(par.comm_cart, 2, 1, &par.nbr_zm, &par.nbr_zp);

    par.nx = par.gnx / par.dims[0];
    par.ny = par.gny / par.dims[1];
    par.nz = par.gnz / par.dims[2];

    par.istart = par.coords[0] * par.nx;
    par.jstart = par.coords[1] * par.ny;
    par.kstart = par.coords[2] * par.nz;
#endif
}

void cleanup_mpi_domain(Parameters& par) {
#ifdef USE_MPI
    if (par.comm_cart != MPI_COMM_NULL) {
        MPI_Comm_free(&par.comm_cart);
        par.comm_cart = MPI_COMM_NULL;
    }
#else
    (void)par;
#endif
}

