#pragma once

#include <Kokkos_Core.hpp>
#include "parameters.hpp"

void output(Kokkos::View<double*****> q,
		const Parameters& par,
		int step,
		double time);


