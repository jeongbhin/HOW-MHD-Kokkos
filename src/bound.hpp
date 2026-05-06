#pragma once

#include <Kokkos_Core.hpp>
#include "parameters.hpp"

void bound(Kokkos::View<double*****> q, int stage, const Parameters& par);