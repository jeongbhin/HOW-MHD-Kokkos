#include "problem.hpp"

#include <unordered_map>
#include <iostream>
#include <cstdlib>

static std::unordered_map<std::string, ProblemInitFunc>& problem_registry() {
    static std::unordered_map<std::string, ProblemInitFunc> registry;
    return registry;
}

void register_problem(const std::string& name,
                      ProblemInitFunc func) {
    problem_registry()[name] = func;
}

void init_problem(Kokkos::View<double*****> q,
                  const Parameters& par) {

    auto& registry = problem_registry();

    auto it = registry.find(par.problem);

    if (it == registry.end()) {
        std::cerr << "Error: unknown problem = "
                  << par.problem << "\n";

        std::cerr << "Available problems:\n";
        for (const auto& kv : registry) {
            std::cerr << "  " << kv.first << "\n";
        }

        std::abort();
    }

    it->second(q, par);
}
