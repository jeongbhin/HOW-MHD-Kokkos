#include "parameters.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

void read_parameters(std::istream& in, Parameters& par) {
    // Safe defaults; the input file can override these values.
    par.pi = 3.141592653589793;

    par.gnx = par.gny = par.gnz = 0;
    par.nx  = par.ny  = par.nz  = 0;

    par.nt = par.nstep = par.ntime = 0;

    par.xsize = par.ysize = par.zsize = 1.0;
    par.dx = par.dy = par.dz = 0.0;
    par.dt = par.dtdx = par.dtdy = par.dtdz = 0.0;

    par.gam  = 5.0 / 3.0;
    par.cour = 0.4;
    par.t    = 0.0;
    par.tend = 0.0;

    par.rhomin = 1.0e-12;
    par.pgmin  = 1.0e-12;

    par.x1bc = "periodic";
    par.x2bc = "periodic";
    par.x3bc = "periodic";
    par.output_format = "dat";

    par.px = par.py = par.pz = 0;

    std::string key;
    while (in >> key) {
        if (key.empty()) {
            continue;
        }

        if (key[0] == '#') {
            std::string dummy;
            std::getline(in, dummy);
        }
        // Input nx/ny/nz are GLOBAL active-cell counts.
        else if (key == "nx" || key == "gnx") {
            in >> par.gnx;
        }
        else if (key == "ny" || key == "gny") {
            in >> par.gny;
        }
        else if (key == "nz" || key == "gnz") {
            in >> par.gnz;
        }
        else if (key == "nt") {
            in >> par.nt;
        }
        else if (key == "nstep") {
            in >> par.nstep;
        }
        else if (key == "ntime") {
            in >> par.ntime;
        }
        else if (key == "xsize") {
            in >> par.xsize;
        }
        else if (key == "ysize") {
            in >> par.ysize;
        }
        else if (key == "zsize") {
            in >> par.zsize;
        }
        else if (key == "gam") {
            in >> par.gam;
        }
        else if (key == "cour") {
            in >> par.cour;
        }
        else if (key == "tend") {
            in >> par.tend;
        }
        else if (key == "rhomin") {
            in >> par.rhomin;
        }
        else if (key == "pgmin") {
            in >> par.pgmin;
        }
        else if (key == "x1bc") {
            in >> par.x1bc;
        }
        else if (key == "x2bc") {
            in >> par.x2bc;
        }
        else if (key == "x3bc") {
            in >> par.x3bc;
        }
        else if (key == "problem") {
            in >> par.problem;
        }
        else if (key == "output_format") {
            in >> par.output_format;
        }
        else if (key == "px") {
            in >> par.px;
        }
        else if (key == "py") {
            in >> par.py;
        }
        else if (key == "pz") {
            in >> par.pz;
        }
        else {
            std::cerr << "Warning: unknown parameter " << key << "\n";
            std::string dummy;
            std::getline(in, dummy);
        }
    }
}

void finalize_parameters_after_read(Parameters& par) {
    if (par.gnx <= 0 || par.gny <= 0 || par.gnz <= 0) {
        std::cerr << "Error: global grid size must be positive.\n";
        std::cerr << "gnx = " << par.gnx
                  << " gny = " << par.gny
                  << " gnz = " << par.gnz << "\n";
        std::abort();
    }

    if (par.nt <= 0) {
        std::cerr << "Error: nt must be positive. nt = " << par.nt << "\n";
        std::abort();
    }

    if (par.tend <= 0.0) {
        std::cerr << "Error: tend must be positive. tend = " << par.tend << "\n";
        std::abort();
    }

    // Serial-domain default. MPI setup will overwrite nx/ny/nz later.
    par.nx = par.gnx;
    par.ny = par.gny;
    par.nz = par.gnz;

    par.istart = 0;
    par.jstart = 0;
    par.kstart = 0;

    // Cell size is always based on the global domain.
    par.dx = par.xsize / static_cast<double>(par.gnx);
    par.dy = par.ysize / static_cast<double>(par.gny);
    par.dz = par.zsize / static_cast<double>(par.gnz);

    par.dims[0] = (par.px > 0) ? par.px : 1;
    par.dims[1] = (par.py > 0) ? par.py : 1;
    par.dims[2] = (par.pz > 0) ? par.pz : 1;

    par.periods[0] = (par.x1bc == "periodic") ? 1 : 0;
    par.periods[1] = (par.x2bc == "periodic") ? 1 : 0;
    par.periods[2] = (par.x3bc == "periodic") ? 1 : 0;
}

