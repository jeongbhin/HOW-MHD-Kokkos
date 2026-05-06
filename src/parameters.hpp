#pragma once
#include <string>
#include <iostream>

struct Parameters {
	int nx, ny, nz, nstep, ntime, nt;

	double xsize, ysize, zsize;
	double dx, dy, dz, dt;
	double dtdx, dtdy, dtdz;

	double gam, cour, pi;
	double t, tend;

	double rhomin, pgmin;
	
	std::string x1bc;
	std::string x2bc;
	std::string x3bc;
    	std::string problem; 
};

void read_parameters(std::istream& in, Parameters& par);
