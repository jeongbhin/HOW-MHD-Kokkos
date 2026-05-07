#include "problem.hpp"
#include "parameters.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

namespace {

void init_mpi_smooth3d(Kokkos::View<double*****> q,
                       const Parameters& par) {
    const double pi = 3.141592653589793238462643383279502884;

    const double gam = par.gam;

    const double dx = par.dx;
    const double dy = par.dy;
    const double dz = par.dz;

    const double Lx = par.xsize;
    const double Ly = par.ysize;
    const double Lz = par.zsize;

    // Local active-cell range:
    // i,j,k = 3 ... nx+2, ny+2, nz+2
    Kokkos::parallel_for(
        "init_mpi_smooth3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {3, 3, 3},
            {par.nx + 3, par.ny + 3, par.nz + 3}
        ),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int il = i - 3;
            const int jl = j - 3;
            const int kl = k - 3;

            // Global active-cell index.
            const int ig = par.istart + il;
            const int jg = par.jstart + jl;
            const int kg = par.kstart + kl;

            const double x = (static_cast<double>(ig) + 0.5) * dx;
            const double y = (static_cast<double>(jg) + 0.5) * dy;
            const double z = (static_cast<double>(kg) + 0.5) * dz;

            const double sx = sin(2.0 * pi * x / Lx);
            const double sy = sin(2.0 * pi * y / Ly);
            const double sz = sin(2.0 * pi * z / Lz);

            const double cx = cos(2.0 * pi * x / Lx);
            const double cy = cos(2.0 * pi * y / Ly);
            const double cz = cos(2.0 * pi * z / Lz);

            // Smooth, periodic, weakly perturbed state.
            const double rho = 1.0 + 0.05 * sx * sy * sz;

            const double vx = 0.10 + 0.01 * cy * cz;
            const double vy = 0.05 + 0.01 * cz * cx;
            const double vz = 0.02 + 0.01 * cx * cy;

            // Nearly uniform magnetic field with small smooth perturbations.
            // This is meant as an MPI/Kokkos smoke test, not a strict analytic solution.
            const double Bx = 0.10 + 0.005 * sy * sz;
            const double By = 0.08 + 0.005 * sz * sx;
            const double Bz = 0.06 + 0.005 * sx * sy;

            const double pg = 1.0 + 0.02 * cx * cy * cz;

            const double Mx = rho * vx;
            const double My = rho * vy;
            const double Mz = rho * vz;

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;

            const double E = pg / (gam - 1.0)
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

struct RegisterMPISmooth3D {
    RegisterMPISmooth3D() {
        register_problem("mpi_smooth3d", init_mpi_smooth3d);
    }
};

RegisterMPISmooth3D register_mpi_smooth3d;

} // namespace
