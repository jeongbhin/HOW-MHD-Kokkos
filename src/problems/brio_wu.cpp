#include "problem.hpp"

#include <Kokkos_Core.hpp>

void init_brio_wu(Kokkos::View<double*****> q,
                  const Parameters& par) {

    const double xmid = 0.5 * par.xsize;

    Kokkos::parallel_for(
        "init_brio_wu",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {3, 3, 3},
            {par.nx + 3, par.ny + 3, par.nz + 3}
        ),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

            const double x = (static_cast<double>(i - 3) + 0.5) * par.dx;

            double rho, vx, vy, vz, pg, Bx, By, Bz;

            if (x < xmid) {
                rho = 1.0;
                vx  = 0.0;
                vy  = 0.0;
                vz  = 0.0;
                pg  = 1.0;
                Bx  = 0.75;
                By  = 1.0;
                Bz  = 0.0;
            } else {
                rho = 0.125;
                vx  = 0.0;
                vy  = 0.0;
                vz  = 0.0;
                pg  = 0.1;
                Bx  = 0.75;
                By  = -1.0;
                Bz  = 0.0;
            }

            const double Mx = rho * vx;
            const double My = rho * vy;
            const double Mz = rho * vz;

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;

            const double E = pg / (par.gam - 1.0)
                           + 0.5 * rho * vv2
                           + 0.5 * BB2;

            q(0,0,i,j,k) = rho;
            q(0,1,i,j,k) = Mx;
            q(0,2,i,j,k) = My;
            q(0,3,i,j,k) = Mz;
            q(0,4,i,j,k) = Bx;
            q(0,5,i,j,k) = By;
            q(0,6,i,j,k) = Bz;
            q(0,7,i,j,k) = E;
        }
    );

    Kokkos::fence();
}

static RegisterProblem register_brio_wu("brio_wu", init_brio_wu);
