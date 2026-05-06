#include <iostream>
#include <iomanip>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "parameters.hpp"
#include "problem.hpp"
#include "tstep.hpp"
#include "ssprk.hpp"
#include "output.hpp"
#include "fluxct.hpp"
#include "bound.hpp"

void print_banner();
void print_log(const Parameters& par);

int main(int argc, char* argv[]) {

    Kokkos::initialize(argc, argv);
    {

        Parameters par;
        read_parameters(std::cin, par);

        par.dx = par.xsize / par.nx;
        par.dy = par.ysize / par.ny;
        par.dz = par.zsize / par.nz;

	print_banner();
	print_log(par);

        Kokkos::View<double*****> q(
            "q",
            6, 8,
            par.nx + 6,
            par.ny + 6,
            par.nz + 6
        );

	init_problem(q, par);

	bound(q, 0, par);

	allocate_bfield_ct(par);
	initialize_bfield_from_q(q, par);

	double time = 0.0;

        const double dtout = par.tend / static_cast<double>(par.nt);

        std::cout << "dtout = " << dtout << "\n";

        // ------------------------------------------------------------
        // Output initial condition
        // dump000000.dat at t = 0
        // ------------------------------------------------------------
        output(q, par, 0, time);
        std::cout << "output step = 0"
                  << " time = " << time
                  << "\n";

        // ------------------------------------------------------------
        // Main loop:
        // par.nt means number of output intervals.
        // The code internally takes CFL-limited hydro time steps.
        // Output is written at uniformly spaced times.
        // ------------------------------------------------------------
        for (int out_step = 1; out_step <= par.nt; ++out_step) {

            const double target_time = dtout * static_cast<double>(out_step);

            int hydro_step = 0;

            while (time < target_time) {

                tstep(q, par);

                // Do not step beyond the next output time.
                if (time + par.dt > target_time) {
                    par.dt = target_time - time;
                }

                // Safety check against zero or negative dt.
                if (par.dt <= 0.0) {
                    std::cerr << "Error: non-positive dt detected.\n";
                    std::cerr << "time        = " << time << "\n";
                    std::cerr << "target_time = " << target_time << "\n";
                    std::cerr << "dt          = " << par.dt << "\n";
                    break;
                }

                ssprk(q, par);

                time += par.dt;
                hydro_step++;

                if (!std::isfinite(time)) {
                    std::cerr << "Error: time became non-finite.\n";
                    break;
                }
            }

            output(q, par, out_step, time);

            std::cout << "output step = " << out_step
                      << " time = " << time
                      << " last_dt = " << par.dt
                      << "\n";
        }
	deallocate_bfield_ct();
    }
    Kokkos::finalize();

    return 0;
}

void print_banner() {
    std::cout << R"(

╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██╗  ██╗ ██████╗ ██╗    ██╗      ███╗   ███╗██╗  ██╗██████╗        ║
║   ██║  ██║██╔═══██╗██║    ██║      ████╗ ████║██║  ██║██╔══██╗       ║
║   ███████║██║   ██║██║ █╗ ██║█████╗██╔████╔██║███████║██║  ██║       ║
║   ██╔══██║██║   ██║██║███╗██║╚════╝██║╚██╔╝██║██╔══██║██║  ██║       ║
║   ██║  ██║╚██████╔╝╚███╔███╔╝      ██║ ╚═╝ ██║██║  ██║██████╔╝       ║
║   ╚═╝  ╚═╝ ╚═════╝  ╚══╝╚══╝       ╚═╝     ╚═╝╚═╝  ╚═╝╚═════╝        ║
║                                                                      ║
║        High-Order WENO Magnetohydrodynamics Solver                   ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║   WENO5  |  SSPRK(5,4)  |  High-order CT  |  Kokkos Backend          ║
╠══════════════════════════════════════════════════════════════════════╣
║   Reference:                                                         ║
║   Seo & Ryu (2023), The Astrophysical Journal, 953, 39               ║
║   doi:10.3847/1538-4357/acdf4b                                       ║
╚══════════════════════════════════════════════════════════════════════╝

)";
}


void print_log(const Parameters& par) {
    std::cout << "===== LOG =====\n";
    std::cout << "nx    = " << par.nx << "\n";
    std::cout << "ny    = " << par.ny << "\n";
    std::cout << "nz    = " << par.nz << "\n";
    std::cout << "nt    = " << par.nt << "  # number of output intervals\n";
    std::cout << "xsize = " << par.xsize << "\n";
    std::cout << "ysize = " << par.ysize << "\n";
    std::cout << "zsize = " << par.zsize << "\n";
    std::cout << "tend  = " << par.tend << "\n";
    std::cout << "dx    = " << par.dx << "\n";
    std::cout << "dy    = " << par.dy << "\n";
    std::cout << "dz    = " << par.dz << "\n";
    std::cout << "x boundary  = " << par.x1bc << "\n";
    std::cout << "y boundary  = " << par.x2bc << "\n";
    std::cout << "z boundary  = " << par.x3bc << "\n";
    std::cout << "==============\n";
}
