#pragma once

#include <Kokkos_Core.hpp>
#include <string>

#include "parameters.hpp"

using ProblemInitFunc =
    void (*)(Kokkos::View<double*****>, const Parameters&);

void init_problem(Kokkos::View<double*****> q,
                  const Parameters& par);

void register_problem(const std::string& name,
                      ProblemInitFunc func);

struct RegisterProblem {
    RegisterProblem(const std::string& name,
                    ProblemInitFunc func) {
        register_problem(name, func);
    }
};
