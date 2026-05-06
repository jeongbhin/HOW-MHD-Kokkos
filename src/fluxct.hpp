#pragma once

#include <Kokkos_Core.hpp>
#include "parameters.hpp"

struct CTFluxes {
    Kokkos::View<double***> fsy, fsz;
    Kokkos::View<double***> gsx, gsz;
    Kokkos::View<double***> hsx, hsy;
    Kokkos::View<double***> Ox2, Oy2, Oz2;
    Kokkos::View<double***> Ox, Oy, Oz;

    CTFluxes(const Parameters& par)
        : fsy("fsy", par.nx+6, par.ny+6, par.nz+6),
          fsz("fsz", par.nx+6, par.ny+6, par.nz+6),
          gsx("gsx", par.nx+6, par.ny+6, par.nz+6),
          gsz("gsz", par.nx+6, par.ny+6, par.nz+6),
          hsx("hsx", par.nx+6, par.ny+6, par.nz+6),
          hsy("hsy", par.nx+6, par.ny+6, par.nz+6),
          Ox2("Ox2", par.nx+6, par.ny+6, par.nz+6),
          Oy2("Oy2", par.nx+6, par.ny+6, par.nz+6),
          Oz2("Oz2", par.nx+6, par.ny+6, par.nz+6),
          Ox("Ox", par.nx+6, par.ny+6, par.nz+6),
          Oy("Oy", par.nx+6, par.ny+6, par.nz+6),
          Oz("Oz", par.nx+6, par.ny+6, par.nz+6) {}
};

extern Kokkos::View<double****> bxb;
extern Kokkos::View<double****> byb;
extern Kokkos::View<double****> bzb;

void allocate_bfield_ct(const Parameters& par);
void deallocate_bfield_ct();
void initialize_bfield_from_q(Kokkos::View<double*****> q, const Parameters& par);
void copy_bfield_stage(int dst, int src, const Parameters& par);
void prepare_bfield_stage5(double k1, double k4, double k5, const Parameters& par);
void copy_bfield_to_stage0(const Parameters& par);
void bound_bfield(int stage, const Parameters& par);
void zero_ct_fluxes(CTFluxes& ct);
void bound_ct_fluxes(CTFluxes& ct, const Parameters& par);
void fluxct(Kokkos::View<double*****> q, int iw, double k1, double k3,
            const Parameters& par, CTFluxes& ct);

