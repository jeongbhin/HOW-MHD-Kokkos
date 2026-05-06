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
	
    	std::string problem; 
	std::string x1bc = "open";
	std::string x2bc = "open";
	std::string x3bc = "open";
	std::string output_format = "dat";

};

void read_parameters(std::istream& in, Parameters& par);
