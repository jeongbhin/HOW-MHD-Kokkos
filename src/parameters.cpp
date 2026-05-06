#include "parameters.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

void read_parameters(std::istream& in, Parameters& par) {
    // Safe defaults; input file can override these values.
    par.pi = 3.141592653589793;
    par.nx = par.ny = par.nz = 0;
    par.nt = par.nstep = par.ntime = 0;
    par.xsize = par.ysize = par.zsize = 1.0;
    par.dx = par.dy = par.dz = 0.0;
    par.dt = par.dtdx = par.dtdy = par.dtdz = 0.0;
    par.gam = 5.0 / 3.0;
    par.cour = 0.4;
    par.t = 0.0;
    par.tend = 0.0;
    par.rhomin = 1.0e-12;
    par.pgmin = 1.0e-12;
    par.x1bc = "periodic";
    par.x2bc = "periodic";
    par.x3bc = "periodic";

	std::string key;
	while (in >> key) {
		if      (key == "nx")    in >> par.nx;
  	        else if (key == "ny")    in >> par.ny;
    	        else if (key == "nz")    in >> par.nz;
		else if (key == "nt")    in >> par.nt;
                else if (key == "xsize") in >> par.xsize;
                else if (key == "ysize") in >> par.ysize;
                else if (key == "zsize") in >> par.zsize;
                else if (key == "gam")   in >> par.gam;
                else if (key == "cour")  in >> par.cour;
                else if (key == "tend")  in >> par.tend;
                else if (key == "rhomin") in >> par.rhomin;
                else if (key == "pgmin")  in >> par.pgmin;
                else if (key == "x1bc")   in >> par.x1bc;
                else if (key == "x2bc")   in >> par.x2bc;
                else if (key == "x3bc")   in >> par.x3bc;
		else if (key == "problem") in >> par.problem;
		else if (key[0] == '#') {
		    std::string dummy;
		    std::getline(in, dummy);
	}else {
            std::cerr << "Warning: unknown parameter " << key << "\n";
        }
    }
}

