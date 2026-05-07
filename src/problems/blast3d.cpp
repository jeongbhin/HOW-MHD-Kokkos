#include "problem.hpp"
#include "parameters.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

namespace {

void init_blast3d(Kokkos::View<double*****> q,
                  const Parameters& par) {
    const double gam = par.gam;

    const double dx = par.dx;
    const double dy = par.dy;
    const double dz = par.dz;

    const double Lx = par.xsize;
    const double Ly = par.ysize;
    const double Lz = par.zsize;

    // ------------------------------------------------------------
    // 3D MHD blast-wave setup.
    //
    // The center and radius are defined in global physical coordinates.
    // Therefore the initial condition is independent of MPI decomposition.
    // ------------------------------------------------------------
    const double xc = 0.5 * Lx;
    const double yc = 0.5 * Ly;
    const double zc = 0.5 * Lz;

    const double r0 = 0.10 * fmin(Lx, fmin(Ly, Lz));

    const double rho0 = 1.0;
    const double pg_out = 1.0;
    const double pg_in  = 100.0;

    // Uniform weak magnetic field.
    // Change these if you want a stronger anisotropic blast.
    const double Bx0 = 0.1;
    const double By0 = 0.0;
    const double Bz0 = 0.0;

    Kokkos::parallel_for(
        "init_blast3d",
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

            const double rx = x - xc;
            const double ry = y - yc;
            const double rz = z - zc;

            const double rr = sqrt(rx*rx + ry*ry + rz*rz);

            const double rho = rho0;

            const double vx = 0.0;
            const double vy = 0.0;
            const double vz = 0.0;

            const double Bx = Bx0;
            const double By = By0;
            const double Bz = Bz0;

            const double pg = (rr <= r0) ? pg_in : pg_out;

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

struct RegisterBlast3D {
    RegisterBlast3D() {
        register_problem("blast3d", init_blast3d);
    }
};

RegisterBlast3D register_blast3d;

} // namespace
