#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "parameters.hpp"
#include "mpi_domain.hpp"
#include "problem.hpp"
#include "tstep.hpp"
#include "ssprk.hpp"
#include "output.hpp"
#include "fluxct.hpp"
#include "bound.hpp"

void print_banner();
void print_log(const Parameters& par);

int main(int argc, char* argv[]) {

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    // ------------------------------------------------------------
    // Force immediate stdout/stderr flushing.
    //
    // Under srun/MPI, stdout can be block-buffered, so banner/log
    // messages may not appear until much later. This makes interactive
    // testing clearer. For production runs with very frequent logging,
    // this can be disabled later.
    // ------------------------------------------------------------
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    Kokkos::initialize(argc, argv);
    {
        Parameters par;

        // ------------------------------------------------------------
        // Read input.
        //
        // Preferred:
        //   ./how_mhd_mpi input.txt
        //
        // Fallback:
        //   ./how_mhd_mpi < input.txt
        //
        // The filename mode is strongly preferred for MPI runs because
        // stdin redirection is not always delivered to every MPI rank.
        // ------------------------------------------------------------
        std::ifstream input_file;
        std::istream* input_stream = &std::cin;

        if (argc > 1) {
            input_file.open(argv[1]);

            if (!input_file) {
#ifdef USE_MPI
                int rank_for_error = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank_for_error);

                std::cerr << "Rank " << rank_for_error
                          << ": Error: cannot open input file: "
                          << argv[1] << "\n";

                MPI_Abort(MPI_COMM_WORLD, 1);
#else
                std::cerr << "Error: cannot open input file: "
                          << argv[1] << "\n";
                std::abort();
#endif
            }

            input_stream = &input_file;
        }

        read_parameters(*input_stream, par);
        finalize_parameters_after_read(par);
        setup_mpi_domain(par);

        // ------------------------------------------------------------
        // Banner and log.
        // Only rank 0 prints normal run information.
        // ------------------------------------------------------------
        if (par.cart_rank == 0) {
            print_banner();
            print_log(par);
            std::cout << std::flush;
        }

        // ------------------------------------------------------------
        // Allocate state array.
        //
        // q(stage, var, i, j, k)
        //
        // stage = 0 ... 5
        // var   = 0 ... 7
        //
        // Active cells:
        //   i = 3 ... nx+2
        //   j = 3 ... ny+2
        //   k = 3 ... nz+2
        //
        // Ghost cells:
        //   3 cells on each side
        // ------------------------------------------------------------
        Kokkos::View<double*****> q(
            "q",
            6, 8,
            par.nx + 6,
            par.ny + 6,
            par.nz + 6
        );

        // ------------------------------------------------------------
        // Initial condition and ghost zones.
        // ------------------------------------------------------------
        init_problem(q, par);

        bound(q, 0, par);

        allocate_bfield_ct(par);
        initialize_bfield_from_q(q, par);

        double time = 0.0;

        const double dtout =
            par.tend / static_cast<double>(par.nt);

        if (par.cart_rank == 0) {
            std::cout << "dtout = " << dtout << std::endl;
        }

        // ------------------------------------------------------------
        // Output initial condition.
        // ------------------------------------------------------------
        output(q, par, 0, time);

        if (par.cart_rank == 0) {
            std::cout << "output step = 0"
                      << " time = " << time
                      << std::endl;
        }

        // ------------------------------------------------------------
        // Main loop.
        //
        // par.nt is the number of output intervals.
        // The code internally takes CFL-limited hydro time steps.
        // ------------------------------------------------------------
        for (int out_step = 1; out_step <= par.nt; ++out_step) {

            const double target_time =
                dtout * static_cast<double>(out_step);

            int hydro_step = 0;

            while (time < target_time) {

                tstep(q, par);

                // Do not step beyond the next output time.
                if (time + par.dt > target_time) {
                    par.dt = target_time - time;
                }

                // Safety check against zero or negative dt.
                if (par.dt <= 0.0) {
                    if (par.cart_rank == 0) {
                        std::cerr << "Error: non-positive dt detected.\n";
                        std::cerr << "time        = " << time << "\n";
                        std::cerr << "target_time = " << target_time << "\n";
                        std::cerr << "dt          = " << par.dt << "\n";
                    }
                    break;
                }

		if (par.cart_rank == 0) {
    std::cout << "[advance begin] "
              << "out_step = " << out_step
              << " hydro_step = " << hydro_step + 1
              << " time = " << time
              << " dt = " << par.dt
              << " target_time = " << target_time
              << std::endl;
}

                ssprk(q, par);

                time += par.dt;
                hydro_step++;
if (par.cart_rank == 0) {
    std::cout << "[advance done ] "
              << "out_step = " << out_step
              << " hydro_step = " << hydro_step
              << " time = " << time
              << " dt = " << par.dt
              << std::endl;
}
                if (!std::isfinite(time)) {
                    if (par.cart_rank == 0) {
                        std::cerr << "Error: time became non-finite.\n";
                    }
                    break;
                }

                // Optional progress print.
                // Useful for long interactive tests. Not too frequent.
                if (par.cart_rank == 0 && hydro_step % 50 == 0) {
                    std::cout << "hydro_step = " << hydro_step
                              << " time = " << time
                              << " dt = " << par.dt
                              << " target_time = " << target_time
                              << std::endl;
                }
            }

            output(q, par, out_step, time);

            if (par.cart_rank == 0) {
                std::cout << "output step = " << out_step
                          << " time = " << time
                          << " last_dt = " << par.dt
                          << " hydro_steps = " << hydro_step
                          << std::endl;
            }
        }

        deallocate_bfield_ct();

        cleanup_mpi_domain(par);
    }

    Kokkos::finalize();

#ifdef USE_MPI
    MPI_Finalize();
#endif

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
║   WENO5  |  SSPRK(5,4)  |  High-order CT  |  MPI + Kokkos            ║
╠══════════════════════════════════════════════════════════════════════╣
║   Reference:                                                         ║
║   Seo & Ryu (2023), The Astrophysical Journal, 953, 39               ║
║   doi:10.3847/1538-4357/acdfc7                                       ║
╚══════════════════════════════════════════════════════════════════════╝

)";
}

void print_log(const Parameters& par) {
    std::cout << "\n";
    std::cout << "===== LOG rank " << par.cart_rank << " =====\n";

#ifdef USE_MPI
    std::cout << "MPI ranks          = " << par.nranks << "\n";
    std::cout << "MPI layout         = "
              << par.dims[0] << " x "
              << par.dims[1] << " x "
              << par.dims[2] << "\n";
    std::cout << "rank coords        = ("
              << par.coords[0] << ", "
              << par.coords[1] << ", "
              << par.coords[2] << ")\n";
    std::cout << "global offset      = "
              << par.istart << " "
              << par.jstart << " "
              << par.kstart << "\n";
    std::cout << "neighbors x        = "
              << par.nbr_xm << " "
              << par.nbr_xp << "\n";
    std::cout << "neighbors y        = "
              << par.nbr_ym << " "
              << par.nbr_yp << "\n";
    std::cout << "neighbors z        = "
              << par.nbr_zm << " "
              << par.nbr_zp << "\n";
#else
    std::cout << "MPI ranks          = 1\n";
    std::cout << "MPI layout         = 1 x 1 x 1\n";
    std::cout << "rank coords        = (0, 0, 0)\n";
    std::cout << "global offset      = 0 0 0\n";
#endif

    std::cout << "global nx ny nz    = "
              << par.gnx << " "
              << par.gny << " "
              << par.gnz << "\n";

    std::cout << "local  nx ny nz    = "
              << par.nx << " "
              << par.ny << " "
              << par.nz << "\n";

    std::cout << "nt                 = " << par.nt << "\n";
    std::cout << "xsize ysize zsize  = "
              << par.xsize << " "
              << par.ysize << " "
              << par.zsize << "\n";

    std::cout << "tend               = " << par.tend << "\n";

    std::cout << "dx dy dz           = "
              << par.dx << " "
              << par.dy << " "
              << par.dz << "\n";

    std::cout << "boundary           = "
              << par.x1bc << " "
              << par.x2bc << " "
              << par.x3bc << "\n";

    std::cout << "output_format      = "
              << par.output_format << "\n";

    std::cout << "============================\n";
}
