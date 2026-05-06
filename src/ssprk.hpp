#pragma once
#include <Kokkos_Core.hpp>
#include "parameters.hpp"

void ssprk(Kokkos::View<double*****> q, Parameters& par);
