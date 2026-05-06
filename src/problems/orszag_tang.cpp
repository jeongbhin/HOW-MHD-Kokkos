#include "problem.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

void init_orszag_tang(Kokkos::View<double*****> q,
                      const Parameters& par) {

    const double pi = 3.141592653589793238462643383279502884;

    Kokkos::parallel_for(
        "init_orszag_tang",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {3, 3, 3},
            {par.nx + 3, par.ny + 3, par.nz + 3}
        ),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

            const double x = (static_cast<double>(i - 3) + 0.5) * par.dx;
            const double y = (static_cast<double>(j - 3) + 0.5) * par.dy;

            // ------------------------------------------------------------
            // Orszag-Tang vortex initial condition
            //
            // Domain: [0,1] x [0,1]
            // Periodic in x and y.
            //
            // Conserved variables:
            // q(0,0) = rho
            // q(0,1) = Mx = rho * vx
            // q(0,2) = My = rho * vy
            // q(0,3) = Mz = rho * vz
            // q(0,4) = Bx
            // q(0,5) = By
            // q(0,6) = Bz
            // q(0,7) = E
            // ------------------------------------------------------------

            const double rho = 25.0 / (36.0 * pi);
            const double pg  = 5.0  / (12.0 * pi);

            const double vx = -std::sin(2.0 * pi * y);
            const double vy =  std::sin(2.0 * pi * x);
            const double vz =  0.0;

            // Many Orszag-Tang tests use B / sqrt(4*pi).
            // This is also reasonable for codes where magnetic pressure is B^2/2.
            const double normB = std::sqrt(4.0 * pi);

            const double Bx = -std::sin(2.0 * pi * y) / normB;
            const double By =  std::sin(4.0 * pi * x) / normB;
            const double Bz =  0.0;

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

static RegisterProblem register_orszag_tang("orszag_tang", init_orszag_tang);
