#pragma once

#include <Kokkos_Core.hpp>
#include "parameters.hpp"

void prot(Kokkos::View<double*****> q,
          int stage,
          const Parameters& par);
