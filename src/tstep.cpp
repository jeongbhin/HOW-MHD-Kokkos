#include "tstep.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

void tstep(Kokkos::View<double*****> q, Parameters& par) {

    double vxmax = 0.0;
    double vymax = 0.0;
    double vzmax = 0.0;

    using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    // ------------------------------------------------------------
    // Local CFL scan on this MPI rank.
    //
    // The active cell-centered range is
    //   i = 3 ... nx+2
    //   j = 3 ... ny+2
    //   k = 3 ... nz+2
    // because each side has three ghost cells.
    // ------------------------------------------------------------
    Kokkos::parallel_reduce(
        "tstep_local",
        policy_type({3, 3, 3}, {par.nx + 3, par.ny + 3, par.nz + 3}),
        KOKKOS_LAMBDA(const int i, const int j, const int k,
                      double& vxmax_loc,
                      double& vymax_loc,
                      double& vzmax_loc) {
            const double DD = q(0,0,i,j,k);
            const double Mx = q(0,1,i,j,k);
            const double My = q(0,2,i,j,k);
            const double Mz = q(0,3,i,j,k);
            const double Bx = q(0,4,i,j,k);
            const double By = q(0,5,i,j,k);
            const double Bz = q(0,6,i,j,k);
            const double EE = q(0,7,i,j,k);

            const double rho = fmax(DD, par.rhomin);

            const double vx = Mx / rho;
            const double vy = My / rho;
            const double vz = Mz / rho;

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;

            const double pg = fmax(
                par.pgmin,
                (par.gam - 1.0) * (EE - 0.5 * (rho * vv2 + BB2))
            );

            const double bbn2 = BB2 / rho;
            const double cs2  = fmax(0.0, par.gam * pg / rho);

            const double bnx2 = Bx*Bx / rho;
            const double bny2 = By*By / rho;
            const double bnz2 = Bz*Bz / rho;

            const double rootx = sqrt(fmax(0.0, (bbn2 + cs2)*(bbn2 + cs2)
                                           - 4.0 * bnx2 * cs2));
            const double rooty = sqrt(fmax(0.0, (bbn2 + cs2)*(bbn2 + cs2)
                                           - 4.0 * bny2 * cs2));
            const double rootz = sqrt(fmax(0.0, (bbn2 + cs2)*(bbn2 + cs2)
                                           - 4.0 * bnz2 * cs2));

            const double lfx = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 + rootx)));
            const double lfy = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 + rooty)));
            const double lfz = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 + rootz)));

            vxmax_loc = fmax(vxmax_loc, fabs(vx) + lfx);
            vymax_loc = fmax(vymax_loc, fabs(vy) + lfy);
            vzmax_loc = fmax(vzmax_loc, fabs(vz) + lfz);
        },
        Kokkos::Max<double>(vxmax),
        Kokkos::Max<double>(vymax),
        Kokkos::Max<double>(vzmax)
    );

    Kokkos::fence();

#ifdef USE_MPI
    // ------------------------------------------------------------
    // Global CFL reduction.
    //
    // Each MPI rank only scans its local subdomain above.  All ranks
    // must use the same timestep, so reduce the three directional
    // maximum signal speeds across the Cartesian communicator.
    // ------------------------------------------------------------
    double local_max[3]  = {vxmax, vymax, vzmax};
    double global_max[3] = {0.0, 0.0, 0.0};

    MPI_Comm comm = (par.comm_cart == MPI_COMM_NULL)
                  ? MPI_COMM_WORLD
                  : par.comm_cart;

    const int ierr = MPI_Allreduce(
        local_max,
        global_max,
        3,
        MPI_DOUBLE,
        MPI_MAX,
        comm
    );

    if (ierr != MPI_SUCCESS) {
        if (par.rank == 0) {
            std::cerr << "Error: MPI_Allreduce failed in tstep().\n";
        }
        MPI_Abort(comm, ierr);
    }

    vxmax = global_max[0];
    vymax = global_max[1];
    vzmax = global_max[2];
#endif

    const double tiny = 1.0e-300;
    double invdt = 0.0;

    if (par.gnx > 1) invdt += fmax(vxmax, tiny) / par.dx;
    if (par.gny > 1) invdt += fmax(vymax, tiny) / par.dy;
    if (par.gnz > 1) invdt += fmax(vzmax, tiny) / par.dz;

    // Purely degenerate safety fallback.
    if (invdt <= 0.0) {
        invdt = fmax(vxmax, tiny) / par.dx;
    }

    par.dt = par.cour / invdt;
}

