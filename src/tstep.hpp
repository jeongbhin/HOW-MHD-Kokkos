#pragma once

#include <Kokkos_Core.hpp>
#include "parameters.hpp"

void tstep(Kokkos::View<double*****> q, Parameters& par);
