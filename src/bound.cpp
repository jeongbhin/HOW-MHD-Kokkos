#include "bound.hpp"

#include <iostream>
#include <cstdlib>

static constexpr int NVAR = 8;

void bound_periodic_x(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int nx = par.nx;
    Kokkos::parallel_for("bound_periodic_x",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NVAR, par.ny+6, par.nz+6}),
        KOKKOS_LAMBDA(const int m, const int j, const int k) {
            q(stage,m,0,   j,k) = q(stage,m,nx,   j,k);
            q(stage,m,1,   j,k) = q(stage,m,nx+1, j,k);
            q(stage,m,2,   j,k) = q(stage,m,nx+2, j,k);
            q(stage,m,nx+3,j,k) = q(stage,m,3,    j,k);
            q(stage,m,nx+4,j,k) = q(stage,m,4,    j,k);
            q(stage,m,nx+5,j,k) = q(stage,m,5,    j,k);
        });
    Kokkos::fence();
}

void bound_open_x(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int nx = par.nx;
    Kokkos::parallel_for("bound_open_x",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NVAR, par.ny+6, par.nz+6}),
        KOKKOS_LAMBDA(const int m, const int j, const int k) {
            q(stage,m,0,   j,k) = q(stage,m,3,    j,k);
            q(stage,m,1,   j,k) = q(stage,m,3,    j,k);
            q(stage,m,2,   j,k) = q(stage,m,3,    j,k);
            q(stage,m,nx+3,j,k) = q(stage,m,nx+2,j,k);
            q(stage,m,nx+4,j,k) = q(stage,m,nx+2,j,k);
            q(stage,m,nx+5,j,k) = q(stage,m,nx+2,j,k);
        });
    Kokkos::fence();
}

void bound_periodic_y(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int ny = par.ny;
    Kokkos::parallel_for("bound_periodic_y",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NVAR, par.nx+6, par.nz+6}),
        KOKKOS_LAMBDA(const int m, const int i, const int k) {
            q(stage,m,i,0,   k) = q(stage,m,i,ny,   k);
            q(stage,m,i,1,   k) = q(stage,m,i,ny+1, k);
            q(stage,m,i,2,   k) = q(stage,m,i,ny+2, k);
            q(stage,m,i,ny+3,k) = q(stage,m,i,3,    k);
            q(stage,m,i,ny+4,k) = q(stage,m,i,4,    k);
            q(stage,m,i,ny+5,k) = q(stage,m,i,5,    k);
        });
    Kokkos::fence();
}

void bound_open_y(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int ny = par.ny;
    Kokkos::parallel_for("bound_open_y",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NVAR, par.nx+6, par.nz+6}),
        KOKKOS_LAMBDA(const int m, const int i, const int k) {
            q(stage,m,i,0,   k) = q(stage,m,i,3,    k);
            q(stage,m,i,1,   k) = q(stage,m,i,3,    k);
            q(stage,m,i,2,   k) = q(stage,m,i,3,    k);
            q(stage,m,i,ny+3,k) = q(stage,m,i,ny+2,k);
            q(stage,m,i,ny+4,k) = q(stage,m,i,ny+2,k);
            q(stage,m,i,ny+5,k) = q(stage,m,i,ny+2,k);
        });
    Kokkos::fence();
}

void bound_periodic_z(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int nz = par.nz;
    Kokkos::parallel_for("bound_periodic_z",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NVAR, par.nx+6, par.ny+6}),
        KOKKOS_LAMBDA(const int m, const int i, const int j) {
            q(stage,m,i,j,0)    = q(stage,m,i,j,nz);
            q(stage,m,i,j,1)    = q(stage,m,i,j,nz+1);
            q(stage,m,i,j,2)    = q(stage,m,i,j,nz+2);
            q(stage,m,i,j,nz+3) = q(stage,m,i,j,3);
            q(stage,m,i,j,nz+4) = q(stage,m,i,j,4);
            q(stage,m,i,j,nz+5) = q(stage,m,i,j,5);
        });
    Kokkos::fence();
}

void bound_open_z(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int nz = par.nz;
    Kokkos::parallel_for("bound_open_z",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {NVAR, par.nx+6, par.ny+6}),
        KOKKOS_LAMBDA(const int m, const int i, const int j) {
            q(stage,m,i,j,0)    = q(stage,m,i,j,3);
            q(stage,m,i,j,1)    = q(stage,m,i,j,3);
            q(stage,m,i,j,2)    = q(stage,m,i,j,3);
            q(stage,m,i,j,nz+3) = q(stage,m,i,j,nz+2);
            q(stage,m,i,j,nz+4) = q(stage,m,i,j,nz+2);
            q(stage,m,i,j,nz+5) = q(stage,m,i,j,nz+2);
        });
    Kokkos::fence();
}

void bound(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    if      (par.x1bc == "periodic") bound_periodic_x(q, stage, par);
    else if (par.x1bc == "open")     bound_open_x(q, stage, par);
    else { std::cerr << "Error: unknown x1bc = " << par.x1bc << "\n"; std::abort(); }

    if      (par.x2bc == "periodic") bound_periodic_y(q, stage, par);
    else if (par.x2bc == "open")     bound_open_y(q, stage, par);
    else { std::cerr << "Error: unknown x2bc = " << par.x2bc << "\n"; std::abort(); }

    if      (par.x3bc == "periodic") bound_periodic_z(q, stage, par);
    else if (par.x3bc == "open")     bound_open_z(q, stage, par);
    else { std::cerr << "Error: unknown x3bc = " << par.x3bc << "\n"; std::abort(); }
}

