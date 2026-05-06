#pragma once

#include <Kokkos_Core.hpp>
#include "parameters.hpp"

void output_dat(Kokkos::View<double*****> q,
                const Parameters& par,
                int step,
                double time);

void output_vts_binary(Kokkos::View<double*****> q,
                       const Parameters& par,
                       int step,
                       double time);

void output(Kokkos::View<double*****> q,
            const Parameters& par,
            int step,
            double time);
