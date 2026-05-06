#include "problem.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

void init_mhd_rotor(Kokkos::View<double*****> q,
                    const Parameters& par) {

    const double pi = 3.141592653589793238462643383279502884;

    Kokkos::parallel_for(
        "init_mhd_rotor",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {3, 3, 3},
            {par.nx + 3, par.ny + 3, par.nz + 3}
        ),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

            const double x = (static_cast<double>(i - 3) + 0.5) * par.dx;
            const double y = (static_cast<double>(j - 3) + 0.5) * par.dy;

            // ------------------------------------------------------------
            // 2D MHD Rotor initial condition
            //
            // Domain: [0,1] x [0,1]
            //
            // Standard rotor setup:
            //   dense rotating disk at center
            //   ambient medium outside
            //   linear transition layer between r0 and r1
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

            const double xc = 0.5;
            const double yc = 0.5;

            const double dx = x - xc;
            const double dy = y - yc;
            const double r  = std::sqrt(dx*dx + dy*dy);

            const double r0 = 0.10;
            const double r1 = 0.115;

            const double rho_in  = 10.0;
            const double rho_out = 1.0;

            const double omega = 20.0;

            double rho;
            double vx;
            double vy;

            if (r <= r0) {
                rho = rho_in;
                vx  = -omega * dy;
                vy  =  omega * dx;
            } else if (r <= r1) {
                const double f = (r1 - r) / (r1 - r0);

                rho = rho_out + f * (rho_in - rho_out);
                vx  = f * (-omega * dy);
                vy  = f * ( omega * dx);
            } else {
                rho = rho_out;
                vx  = 0.0;
                vy  = 0.0;
            }

            const double vz = 0.0;

            const double pg = 1.0;

            // Standard rotor often uses Bx = 5 / sqrt(4*pi).
            // This normalization is consistent with magnetic pressure B^2/2.
            const double normB = std::sqrt(4.0 * pi);

            const double Bx = 5.0 / normB;
            const double By = 0.0;
            const double Bz = 0.0;

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

static RegisterProblem register_mhd_rotor("mhd_rotor", init_mhd_rotor);
