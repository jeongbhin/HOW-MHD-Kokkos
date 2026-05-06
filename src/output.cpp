#include "output.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <iostream>

// ================================================================
// Boundary-condition-aware index helper for host-side diagnostics
// ================================================================
int bc_index_host(const int idx,
                  const int lo,
                  const int hi,
                  const bool periodic) {

    if (periodic) {
        const int n = hi - lo + 1;
        int r = (idx - lo) % n;
        if (r < 0) r += n;
        return lo + r;
    } else {
        // open / outflow boundary: clamp
        if (idx < lo) return lo;
        if (idx > hi) return hi;
        return idx;
    }
}


// ================================================================
// Output dump file
// ================================================================
void output(Kokkos::View<double*****> q,
            const Parameters& par,
            int step,
            double time) {

    std::filesystem::create_directories("output");

    std::ostringstream filename;
    filename << "output/dump"
             << std::setw(6) << std::setfill('0') << step
             << ".dat";

    auto qh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);

    std::ofstream fout(filename.str());

    if (!fout) {
        std::cerr << "Error: cannot open output file "
                  << filename.str() << "\n";
        return;
    }

    fout << std::scientific << std::setprecision(15);

    fout << "# step = " << step << "\n";
    fout << "# time = " << time << "\n";
    fout << "# columns: i j k x y z rho Mx My Mz Bx By Bz E pg vx vy vz divB\n";

    // ------------------------------------------------------------
    // Active cell-centered range
    // ------------------------------------------------------------
    const int ic_lo = 3;
    const int jc_lo = 3;
    const int kc_lo = 3;

    const int ic_hi = par.nx + 2;
    const int jc_hi = par.ny + 2;
    const int kc_hi = par.nz + 2;

    const bool x_periodic = (par.x1bc == "periodic");
    const bool y_periodic = (par.x2bc == "periodic");
    const bool z_periodic = (par.x3bc == "periodic");

    for (int k = kc_lo; k <= kc_hi; ++k) {
        for (int j = jc_lo; j <= jc_hi; ++j) {
            for (int i = ic_lo; i <= ic_hi; ++i) {

                const double x = (static_cast<double>(i - 3) + 0.5) * par.dx;
                const double y = (static_cast<double>(j - 3) + 0.5) * par.dy;
                const double z = (static_cast<double>(k - 3) + 0.5) * par.dz;

                const double rho = qh(0,0,i,j,k);
                const double Mx  = qh(0,1,i,j,k);
                const double My  = qh(0,2,i,j,k);
                const double Mz  = qh(0,3,i,j,k);
                const double Bx  = qh(0,4,i,j,k);
                const double By  = qh(0,5,i,j,k);
                const double Bz  = qh(0,6,i,j,k);
                const double E   = qh(0,7,i,j,k);

                const double vx = Mx / rho;
                const double vy = My / rho;
                const double vz = Mz / rho;

                const double vv2 = vx*vx + vy*vy + vz*vz;
                const double BB2 = Bx*Bx + By*By + Bz*Bz;

                const double pg =
                    (par.gam - 1.0) * (E - 0.5 * rho * vv2 - 0.5 * BB2);

                // --------------------------------------------------------
                // Cell-centered diagnostic divB.
                //
                // This is not the exact face-centered CT divergence.
                // It is a useful diagnostic to see whether divB grows badly.
                //
                // For periodic boundaries: wrap.
                // For open boundaries: clamp.
                // For collapsed dimensions, skip derivative.
                // --------------------------------------------------------
                double divB = 0.0;

                if (par.nx > 1) {
                    const int im1 = bc_index_host(i - 1, ic_lo, ic_hi, x_periodic);
                    const int ip1 = bc_index_host(i + 1, ic_lo, ic_hi, x_periodic);

                    if (im1 == i || ip1 == i) {
                        // one-sided/clamped effective derivative near open edge
                        divB += (qh(0,4,ip1,j,k) - qh(0,4,im1,j,k)) / par.dx;
                    } else {
                        divB += (qh(0,4,ip1,j,k) - qh(0,4,im1,j,k)) / (2.0 * par.dx);
                    }
                }

                if (par.ny > 1) {
                    const int jm1 = bc_index_host(j - 1, jc_lo, jc_hi, y_periodic);
                    const int jp1 = bc_index_host(j + 1, jc_lo, jc_hi, y_periodic);

                    if (jm1 == j || jp1 == j) {
                        divB += (qh(0,5,i,jp1,k) - qh(0,5,i,jm1,k)) / par.dy;
                    } else {
                        divB += (qh(0,5,i,jp1,k) - qh(0,5,i,jm1,k)) / (2.0 * par.dy);
                    }
                }

                if (par.nz > 1) {
                    const int km1 = bc_index_host(k - 1, kc_lo, kc_hi, z_periodic);
                    const int kp1 = bc_index_host(k + 1, kc_lo, kc_hi, z_periodic);

                    if (km1 == k || kp1 == k) {
                        divB += (qh(0,6,i,j,kp1) - qh(0,6,i,j,km1)) / par.dz;
                    } else {
                        divB += (qh(0,6,i,j,kp1) - qh(0,6,i,j,km1)) / (2.0 * par.dz);
                    }
                }

                fout << i - 3 << " "
                     << j - 3 << " "
                     << k - 3 << " "
                     << x << " "
                     << y << " "
                     << z << " "
                     << rho << " "
                     << Mx << " "
                     << My << " "
                     << Mz << " "
                     << Bx << " "
                     << By << " "
                     << Bz << " "
                     << E << " "
                     << pg << " "
                     << vx << " "
                     << vy << " "
                     << vz << " "
                     << divB << "\n";
            }
        }
    }

    fout.close();

    std::cout << "Wrote " << filename.str() << "\n";
}
