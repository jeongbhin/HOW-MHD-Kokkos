#include "prot.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

void prot(Kokkos::View<double*****> q,
          int stage,
          const Parameters& par) {

    Kokkos::parallel_for(
        "prot",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {3, 3, 3},
            {par.nx + 3, par.ny + 3, par.nz + 3}
        ),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

            double rho = q(stage,0,i,j,k);

            const double Mx = q(stage,1,i,j,k);
            const double My = q(stage,2,i,j,k);
            const double Mz = q(stage,3,i,j,k);

            const double Bx = q(stage,4,i,j,k);
            const double By = q(stage,5,i,j,k);
            const double Bz = q(stage,6,i,j,k);

            double E = q(stage,7,i,j,k);

            // If rho is too small, protect it before velocity calculation.
            rho = fmax(rho, par.rhomin);

            const double vx = Mx / rho;
            const double vy = My / rho;
            const double vz = Mz / rho;

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;

            double pg = (par.gam - 1.0)
                      * (E - 0.5 * (rho*vv2 + BB2));

            pg = fmax(pg, par.pgmin);

            E = pg / (par.gam - 1.0)
              + 0.5 * (rho*vv2 + BB2);

            q(stage,0,i,j,k) = rho;
            q(stage,7,i,j,k) = E;
        }
    );

    Kokkos::fence();
}
