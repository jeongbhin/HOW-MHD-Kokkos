#include "tstep.hpp"
#include <cmath>
#include <algorithm>

void tstep(Kokkos::View<double*****> q, Parameters& par) {

    double vxmax = 0.0;
    double vymax = 0.0;
    double vzmax = 0.0;

    using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    Kokkos::parallel_reduce(
        "tstep",
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

            const double pg = fmax(par.pgmin,
                (par.gam - 1.0) * (EE - 0.5 * (rho * vv2 + BB2)));

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

    const double tiny = 1.0e-300;
    double invdt = 0.0;

    if (par.nx > 1) invdt += fmax(vxmax, tiny) / par.dx;
    if (par.ny > 1) invdt += fmax(vymax, tiny) / par.dy;
    if (par.nz > 1) invdt += fmax(vzmax, tiny) / par.dz;

    // Purely degenerate safety fallback.
    if (invdt <= 0.0) {
        invdt = fmax(vxmax, tiny) / par.dx;
    }

    par.dt = par.cour / invdt;
}

