#include "problem.hpp"

#include <Kokkos_Core.hpp>

void init_uniform(Kokkos::View<double*****> q,
                  const Parameters& par) {

    Kokkos::parallel_for(
        "init_uniform",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {3, 3, 3},
            {par.nx + 3, par.ny + 3, par.nz + 3}
        ),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

            q(0,0,i,j,k) = 1.0;
            q(0,1,i,j,k) = 0.0;
            q(0,2,i,j,k) = 0.0;
            q(0,3,i,j,k) = 0.0;
            q(0,4,i,j,k) = 1.0;
            q(0,5,i,j,k) = 0.0;
            q(0,6,i,j,k) = 0.0;
            q(0,7,i,j,k) = 1.0;
        }
    );

    Kokkos::fence();
}


static RegisterProblem register_uniform("uniform", init_uniform);
