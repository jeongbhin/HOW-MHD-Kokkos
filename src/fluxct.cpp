#include "fluxct.hpp"

#include <Kokkos_Core.hpp>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

Kokkos::View<double****> bxb;
Kokkos::View<double****> byb;
Kokkos::View<double****> bzb;

KOKKOS_INLINE_FUNCTION
int bc_index_ct(const int idx, const int lo, const int hi, const bool periodic) {
    if (periodic) {
        const int n = hi - lo + 1;
        int r = (idx - lo) % n;
        if (r < 0) r += n;
        return lo + r;
    }
    if (idx < lo) return lo;
    if (idx > hi) return hi;
    return idx;
}

void allocate_bfield_ct(const Parameters& par) {
    bxb = Kokkos::View<double****>("bxb", 6, par.nx+6, par.ny+6, par.nz+6);
    byb = Kokkos::View<double****>("byb", 6, par.nx+6, par.ny+6, par.nz+6);
    bzb = Kokkos::View<double****>("bzb", 6, par.nx+6, par.ny+6, par.nz+6);
    Kokkos::deep_copy(bxb, 0.0);
    Kokkos::deep_copy(byb, 0.0);
    Kokkos::deep_copy(bzb, 0.0);
    Kokkos::fence();
}

void deallocate_bfield_ct() {
    bxb = Kokkos::View<double****>();
    byb = Kokkos::View<double****>();
    bzb = Kokkos::View<double****>();
    Kokkos::fence();
}



// ================================================================
// Initialize bxb/byb/bzb from cell-centered q.
//
// This follows the original Fortran initialization:
//
// bxb(ix) = (-Bx(ix-1) + 9 Bx(ix) + 9 Bx(ix+1) - Bx(ix+2)) / 16
// byb(iy) = (-By(iy-1) + 9 By(iy) + 9 By(iy+1) - By(iy+2)) / 16
// bzb(iz) = (-Bz(iz-1) + 9 Bz(iz) + 9 Bz(iz+1) - Bz(iz+2)) / 16
//
// Fortran loop:
//   ix = -1 ... nx+1
//
// C++ shifted by +2:
//   i = 1 ... nx+3
//
// IMPORTANT:
//   q ghost zones must already be filled before calling this.
//   Therefore call bound(q, 0, par) before initialize_bfield_from_q().
// ================================================================
void initialize_bfield_from_q(Kokkos::View<double*****> q,
                              const Parameters& par) {

    const int ilo = 1;
    const int jlo = 1;
    const int klo = 1;

    const int ihi = par.nx + 3;
    const int jhi = par.ny + 3;
    const int khi = par.nz + 3;

    Kokkos::parallel_for(
        "initialize_bfield_from_q_fortran_style",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {ilo, jlo, klo},
            {ihi + 1, jhi + 1, khi + 1}
        ),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

            bxb(0,i,j,k) =
                ( - q(0,4,i-1,j,k)
                  + 9.0*q(0,4,i  ,j,k)
                  + 9.0*q(0,4,i+1,j,k)
                  - q(0,4,i+2,j,k) ) / 16.0;

            byb(0,i,j,k) =
                ( - q(0,5,i,j-1,k)
                  + 9.0*q(0,5,i,j  ,k)
                  + 9.0*q(0,5,i,j+1,k)
                  - q(0,5,i,j+2,k) ) / 16.0;

            bzb(0,i,j,k) =
                ( - q(0,6,i,j,k-1)
                  + 9.0*q(0,6,i,j,k  )
                  + 9.0*q(0,6,i,j,k+1)
                  - q(0,6,i,j,k+2) ) / 16.0;
        }
    );

    Kokkos::fence();

    bound_bfield(0, par);
}



void copy_bfield_stage(int dst, int src, const Parameters& par) {
    Kokkos::parallel_for(
        "copy_bfield_stage",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {par.nx+6, par.ny+6, par.nz+6}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            bxb(dst,i,j,k) = bxb(src,i,j,k);
            byb(dst,i,j,k) = byb(src,i,j,k);
            bzb(dst,i,j,k) = bzb(src,i,j,k);
        }
    );
    Kokkos::fence();
}

void prepare_bfield_stage5(double k1, double k4, double k5, const Parameters& par) {
    Kokkos::parallel_for(
        "prepare_bfield_stage5",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {par.nx+6, par.ny+6, par.nz+6}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            bxb(5,i,j,k) = bxb(0,i,j,k) + (k4/k1)*bxb(2,i,j,k) + (k5/k1)*bxb(3,i,j,k);
            byb(5,i,j,k) = byb(0,i,j,k) + (k4/k1)*byb(2,i,j,k) + (k5/k1)*byb(3,i,j,k);
            bzb(5,i,j,k) = bzb(0,i,j,k) + (k4/k1)*bzb(2,i,j,k) + (k5/k1)*bzb(3,i,j,k);
        }
    );
    Kokkos::fence();
}

void copy_bfield_to_stage0(const Parameters& par) {
    copy_bfield_stage(0, 5, par);
}

static void bound_bfield_local_all(int stage, const Parameters& par) {
    const bool xp = (par.x1bc == "periodic");
    const bool yp = (par.x2bc == "periodic");
    const bool zp = (par.x3bc == "periodic");
    const int nx = par.nx, ny = par.ny, nz = par.nz;

    Kokkos::parallel_for("bound_bfield_x", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny+6,nz+6}),
        KOKKOS_LAMBDA(const int j, const int k) {
            if (xp) {
                bxb(stage,0,j,k)=bxb(stage,nx,j,k);     bxb(stage,1,j,k)=bxb(stage,nx+1,j,k); bxb(stage,2,j,k)=bxb(stage,nx+2,j,k);
                byb(stage,0,j,k)=byb(stage,nx,j,k);     byb(stage,1,j,k)=byb(stage,nx+1,j,k); byb(stage,2,j,k)=byb(stage,nx+2,j,k);
                bzb(stage,0,j,k)=bzb(stage,nx,j,k);     bzb(stage,1,j,k)=bzb(stage,nx+1,j,k); bzb(stage,2,j,k)=bzb(stage,nx+2,j,k);
                bxb(stage,nx+3,j,k)=bxb(stage,3,j,k);  bxb(stage,nx+4,j,k)=bxb(stage,4,j,k); bxb(stage,nx+5,j,k)=bxb(stage,5,j,k);
                byb(stage,nx+3,j,k)=byb(stage,3,j,k);  byb(stage,nx+4,j,k)=byb(stage,4,j,k); byb(stage,nx+5,j,k)=byb(stage,5,j,k);
                bzb(stage,nx+3,j,k)=bzb(stage,3,j,k);  bzb(stage,nx+4,j,k)=bzb(stage,4,j,k); bzb(stage,nx+5,j,k)=bzb(stage,5,j,k);
            } else {
                bxb(stage,0,j,k)=bxb(stage,3,j,k);     bxb(stage,1,j,k)=bxb(stage,3,j,k);    bxb(stage,2,j,k)=bxb(stage,3,j,k);
                byb(stage,0,j,k)=byb(stage,3,j,k);     byb(stage,1,j,k)=byb(stage,3,j,k);    byb(stage,2,j,k)=byb(stage,3,j,k);
                bzb(stage,0,j,k)=bzb(stage,3,j,k);     bzb(stage,1,j,k)=bzb(stage,3,j,k);    bzb(stage,2,j,k)=bzb(stage,3,j,k);
                bxb(stage,nx+3,j,k)=bxb(stage,nx+2,j,k); bxb(stage,nx+4,j,k)=bxb(stage,nx+2,j,k); bxb(stage,nx+5,j,k)=bxb(stage,nx+2,j,k);
                byb(stage,nx+3,j,k)=byb(stage,nx+2,j,k); byb(stage,nx+4,j,k)=byb(stage,nx+2,j,k); byb(stage,nx+5,j,k)=byb(stage,nx+2,j,k);
                bzb(stage,nx+3,j,k)=bzb(stage,nx+2,j,k); bzb(stage,nx+4,j,k)=bzb(stage,nx+2,j,k); bzb(stage,nx+5,j,k)=bzb(stage,nx+2,j,k);
            }
        });

    Kokkos::parallel_for("bound_bfield_y", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,nz+6}),
        KOKKOS_LAMBDA(const int i, const int k) {
            if (yp) {
                bxb(stage,i,0,k)=bxb(stage,i,ny,k);     bxb(stage,i,1,k)=bxb(stage,i,ny+1,k); bxb(stage,i,2,k)=bxb(stage,i,ny+2,k);
                byb(stage,i,0,k)=byb(stage,i,ny,k);     byb(stage,i,1,k)=byb(stage,i,ny+1,k); byb(stage,i,2,k)=byb(stage,i,ny+2,k);
                bzb(stage,i,0,k)=bzb(stage,i,ny,k);     bzb(stage,i,1,k)=bzb(stage,i,ny+1,k); bzb(stage,i,2,k)=bzb(stage,i,ny+2,k);
                bxb(stage,i,ny+3,k)=bxb(stage,i,3,k);  bxb(stage,i,ny+4,k)=bxb(stage,i,4,k); bxb(stage,i,ny+5,k)=bxb(stage,i,5,k);
                byb(stage,i,ny+3,k)=byb(stage,i,3,k);  byb(stage,i,ny+4,k)=byb(stage,i,4,k); byb(stage,i,ny+5,k)=byb(stage,i,5,k);
                bzb(stage,i,ny+3,k)=bzb(stage,i,3,k);  bzb(stage,i,ny+4,k)=bzb(stage,i,4,k); bzb(stage,i,ny+5,k)=bzb(stage,i,5,k);
            } else {
                bxb(stage,i,0,k)=bxb(stage,i,3,k);     bxb(stage,i,1,k)=bxb(stage,i,3,k);    bxb(stage,i,2,k)=bxb(stage,i,3,k);
                byb(stage,i,0,k)=byb(stage,i,3,k);     byb(stage,i,1,k)=byb(stage,i,3,k);    byb(stage,i,2,k)=byb(stage,i,3,k);
                bzb(stage,i,0,k)=bzb(stage,i,3,k);     bzb(stage,i,1,k)=bzb(stage,i,3,k);    bzb(stage,i,2,k)=bzb(stage,i,3,k);
                bxb(stage,i,ny+3,k)=bxb(stage,i,ny+2,k); bxb(stage,i,ny+4,k)=bxb(stage,i,ny+2,k); bxb(stage,i,ny+5,k)=bxb(stage,i,ny+2,k);
                byb(stage,i,ny+3,k)=byb(stage,i,ny+2,k); byb(stage,i,ny+4,k)=byb(stage,i,ny+2,k); byb(stage,i,ny+5,k)=byb(stage,i,ny+2,k);
                bzb(stage,i,ny+3,k)=bzb(stage,i,ny+2,k); bzb(stage,i,ny+4,k)=bzb(stage,i,ny+2,k); bzb(stage,i,ny+5,k)=bzb(stage,i,ny+2,k);
            }
        });

    Kokkos::parallel_for("bound_bfield_z", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,ny+6}),
        KOKKOS_LAMBDA(const int i, const int j) {
            if (zp) {
                bxb(stage,i,j,0)=bxb(stage,i,j,nz);     bxb(stage,i,j,1)=bxb(stage,i,j,nz+1); bxb(stage,i,j,2)=bxb(stage,i,j,nz+2);
                byb(stage,i,j,0)=byb(stage,i,j,nz);     byb(stage,i,j,1)=byb(stage,i,j,nz+1); byb(stage,i,j,2)=byb(stage,i,j,nz+2);
                bzb(stage,i,j,0)=bzb(stage,i,j,nz);     bzb(stage,i,j,1)=bzb(stage,i,j,nz+1); bzb(stage,i,j,2)=bzb(stage,i,j,nz+2);
                bxb(stage,i,j,nz+3)=bxb(stage,i,j,3);  bxb(stage,i,j,nz+4)=bxb(stage,i,j,4); bxb(stage,i,j,nz+5)=bxb(stage,i,j,5);
                byb(stage,i,j,nz+3)=byb(stage,i,j,3);  byb(stage,i,j,nz+4)=byb(stage,i,j,4); byb(stage,i,j,nz+5)=byb(stage,i,j,5);
                bzb(stage,i,j,nz+3)=bzb(stage,i,j,3);  bzb(stage,i,j,nz+4)=bzb(stage,i,j,4); bzb(stage,i,j,nz+5)=bzb(stage,i,j,5);
            } else {
                bxb(stage,i,j,0)=bxb(stage,i,j,3);     bxb(stage,i,j,1)=bxb(stage,i,j,3);    bxb(stage,i,j,2)=bxb(stage,i,j,3);
                byb(stage,i,j,0)=byb(stage,i,j,3);     byb(stage,i,j,1)=byb(stage,i,j,3);    byb(stage,i,j,2)=byb(stage,i,j,3);
                bzb(stage,i,j,0)=bzb(stage,i,j,3);     bzb(stage,i,j,1)=bzb(stage,i,j,3);    bzb(stage,i,j,2)=bzb(stage,i,j,3);
                bxb(stage,i,j,nz+3)=bxb(stage,i,j,nz+2); bxb(stage,i,j,nz+4)=bxb(stage,i,j,nz+2); bxb(stage,i,j,nz+5)=bxb(stage,i,j,nz+2);
                byb(stage,i,j,nz+3)=byb(stage,i,j,nz+2); byb(stage,i,j,nz+4)=byb(stage,i,j,nz+2); byb(stage,i,j,nz+5)=byb(stage,i,j,nz+2);
                bzb(stage,i,j,nz+3)=bzb(stage,i,j,nz+2); bzb(stage,i,j,nz+4)=bzb(stage,i,j,nz+2); bzb(stage,i,j,nz+5)=bzb(stage,i,j,nz+2);
            }
        });

    Kokkos::fence();
}


// ================================================================
// Local physical boundary fill for bxb/byb/bzb.
// In MPI mode this is only used on true physical domain boundaries
// where the Cartesian neighbor is MPI_PROC_NULL. Internal boundaries
// are filled by halo exchange.
// ================================================================
static void apply_bfield_physical_boundaries(int stage, const Parameters& par) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;

#ifdef USE_MPI
    const bool xminus_physical = (par.nbr_xm == MPI_PROC_NULL);
    const bool xplus_physical  = (par.nbr_xp == MPI_PROC_NULL);
    const bool yminus_physical = (par.nbr_ym == MPI_PROC_NULL);
    const bool yplus_physical  = (par.nbr_yp == MPI_PROC_NULL);
    const bool zminus_physical = (par.nbr_zm == MPI_PROC_NULL);
    const bool zplus_physical  = (par.nbr_zp == MPI_PROC_NULL);
#else
    const bool xminus_physical = true;
    const bool xplus_physical  = true;
    const bool yminus_physical = true;
    const bool yplus_physical  = true;
    const bool zminus_physical = true;
    const bool zplus_physical  = true;
#endif

    if (xminus_physical) {
        Kokkos::parallel_for("bfield_phys_xm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny+6,nz+6}),
            KOKKOS_LAMBDA(const int j, const int k) {
                bxb(stage,0,j,k)=bxb(stage,3,j,k); byb(stage,0,j,k)=byb(stage,3,j,k); bzb(stage,0,j,k)=bzb(stage,3,j,k);
                bxb(stage,1,j,k)=bxb(stage,3,j,k); byb(stage,1,j,k)=byb(stage,3,j,k); bzb(stage,1,j,k)=bzb(stage,3,j,k);
                bxb(stage,2,j,k)=bxb(stage,3,j,k); byb(stage,2,j,k)=byb(stage,3,j,k); bzb(stage,2,j,k)=bzb(stage,3,j,k);
            });
    }
    if (xplus_physical) {
        Kokkos::parallel_for("bfield_phys_xp", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny+6,nz+6}),
            KOKKOS_LAMBDA(const int j, const int k) {
                bxb(stage,nx+3,j,k)=bxb(stage,nx+2,j,k); byb(stage,nx+3,j,k)=byb(stage,nx+2,j,k); bzb(stage,nx+3,j,k)=bzb(stage,nx+2,j,k);
                bxb(stage,nx+4,j,k)=bxb(stage,nx+2,j,k); byb(stage,nx+4,j,k)=byb(stage,nx+2,j,k); bzb(stage,nx+4,j,k)=bzb(stage,nx+2,j,k);
                bxb(stage,nx+5,j,k)=bxb(stage,nx+2,j,k); byb(stage,nx+5,j,k)=byb(stage,nx+2,j,k); bzb(stage,nx+5,j,k)=bzb(stage,nx+2,j,k);
            });
    }

    if (yminus_physical) {
        Kokkos::parallel_for("bfield_phys_ym", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,nz+6}),
            KOKKOS_LAMBDA(const int i, const int k) {
                bxb(stage,i,0,k)=bxb(stage,i,3,k); byb(stage,i,0,k)=byb(stage,i,3,k); bzb(stage,i,0,k)=bzb(stage,i,3,k);
                bxb(stage,i,1,k)=bxb(stage,i,3,k); byb(stage,i,1,k)=byb(stage,i,3,k); bzb(stage,i,1,k)=bzb(stage,i,3,k);
                bxb(stage,i,2,k)=bxb(stage,i,3,k); byb(stage,i,2,k)=byb(stage,i,3,k); bzb(stage,i,2,k)=bzb(stage,i,3,k);
            });
    }
    if (yplus_physical) {
        Kokkos::parallel_for("bfield_phys_yp", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,nz+6}),
            KOKKOS_LAMBDA(const int i, const int k) {
                bxb(stage,i,ny+3,k)=bxb(stage,i,ny+2,k); byb(stage,i,ny+3,k)=byb(stage,i,ny+2,k); bzb(stage,i,ny+3,k)=bzb(stage,i,ny+2,k);
                bxb(stage,i,ny+4,k)=bxb(stage,i,ny+2,k); byb(stage,i,ny+4,k)=byb(stage,i,ny+2,k); bzb(stage,i,ny+4,k)=bzb(stage,i,ny+2,k);
                bxb(stage,i,ny+5,k)=bxb(stage,i,ny+2,k); byb(stage,i,ny+5,k)=byb(stage,i,ny+2,k); bzb(stage,i,ny+5,k)=bzb(stage,i,ny+2,k);
            });
    }

    if (zminus_physical) {
        Kokkos::parallel_for("bfield_phys_zm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,ny+6}),
            KOKKOS_LAMBDA(const int i, const int j) {
                bxb(stage,i,j,0)=bxb(stage,i,j,3); byb(stage,i,j,0)=byb(stage,i,j,3); bzb(stage,i,j,0)=bzb(stage,i,j,3);
                bxb(stage,i,j,1)=bxb(stage,i,j,3); byb(stage,i,j,1)=byb(stage,i,j,3); bzb(stage,i,j,1)=bzb(stage,i,j,3);
                bxb(stage,i,j,2)=bxb(stage,i,j,3); byb(stage,i,j,2)=byb(stage,i,j,3); bzb(stage,i,j,2)=bzb(stage,i,j,3);
            });
    }
    if (zplus_physical) {
        Kokkos::parallel_for("bfield_phys_zp", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,ny+6}),
            KOKKOS_LAMBDA(const int i, const int j) {
                bxb(stage,i,j,nz+3)=bxb(stage,i,j,nz+2); byb(stage,i,j,nz+3)=byb(stage,i,j,nz+2); bzb(stage,i,j,nz+3)=bzb(stage,i,j,nz+2);
                bxb(stage,i,j,nz+4)=bxb(stage,i,j,nz+2); byb(stage,i,j,nz+4)=byb(stage,i,j,nz+2); bzb(stage,i,j,nz+4)=bzb(stage,i,j,nz+2);
                bxb(stage,i,j,nz+5)=bxb(stage,i,j,nz+2); byb(stage,i,j,nz+5)=byb(stage,i,j,nz+2); bzb(stage,i,j,nz+5)=bzb(stage,i,j,nz+2);
            });
    }

    Kokkos::fence();
}

#ifdef USE_MPI
static constexpr int BFIELD_NG = 3;
static constexpr int BFIELD_NCOMP = 3;

static void exchange_bfield_x(int stage, const Parameters& par) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;
    const int nbuf = BFIELD_NCOMP * BFIELD_NG * (ny + 6) * (nz + 6);

    Kokkos::View<double*> send_xm("send_bfield_xm", nbuf), send_xp("send_bfield_xp", nbuf);
    Kokkos::View<double*> recv_xm("recv_bfield_xm", nbuf), recv_xp("recv_bfield_xp", nbuf);

    Kokkos::parallel_for("pack_bfield_x", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (ny + 6) * (nz + 6);
            const int comp = p / (BFIELD_NG * plane);
            int r = p - comp * BFIELD_NG * plane;
            const int g = r / plane;
            r -= g * plane;
            const int j = r / (nz + 6);
            const int k = r - j * (nz + 6);

            const int i_xm = 3 + g;
            const int i_xp = nx + g;

            double vxm = 0.0, vxp = 0.0;
            if (comp == 0) { vxm = bxb(stage,i_xm,j,k); vxp = bxb(stage,i_xp,j,k); }
            if (comp == 1) { vxm = byb(stage,i_xm,j,k); vxp = byb(stage,i_xp,j,k); }
            if (comp == 2) { vxm = bzb(stage,i_xm,j,k); vxp = bzb(stage,i_xp,j,k); }
            send_xm(p) = vxm;
            send_xp(p) = vxp;
        });
    Kokkos::fence();

    auto h_send_xm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_xm);
    auto h_send_xp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_xp);
    auto h_recv_xm = Kokkos::create_mirror_view(recv_xm);
    auto h_recv_xp = Kokkos::create_mirror_view(recv_xp);

    MPI_Sendrecv(h_send_xm.data(), nbuf, MPI_DOUBLE, par.nbr_xm, 910,
                 h_recv_xp.data(), nbuf, MPI_DOUBLE, par.nbr_xp, 910,
                 par.comm_cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_xp.data(), nbuf, MPI_DOUBLE, par.nbr_xp, 911,
                 h_recv_xm.data(), nbuf, MPI_DOUBLE, par.nbr_xm, 911,
                 par.comm_cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_xm, h_recv_xm);
    Kokkos::deep_copy(recv_xp, h_recv_xp);

    Kokkos::parallel_for("unpack_bfield_x", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (ny + 6) * (nz + 6);
            const int comp = p / (BFIELD_NG * plane);
            int r = p - comp * BFIELD_NG * plane;
            const int g = r / plane;
            r -= g * plane;
            const int j = r / (nz + 6);
            const int k = r - j * (nz + 6);

            const int i_xm = g;
            const int i_xp = nx + 3 + g;
            if (comp == 0) { bxb(stage,i_xm,j,k) = recv_xm(p); bxb(stage,i_xp,j,k) = recv_xp(p); }
            if (comp == 1) { byb(stage,i_xm,j,k) = recv_xm(p); byb(stage,i_xp,j,k) = recv_xp(p); }
            if (comp == 2) { bzb(stage,i_xm,j,k) = recv_xm(p); bzb(stage,i_xp,j,k) = recv_xp(p); }
        });
    Kokkos::fence();
}

static void exchange_bfield_y(int stage, const Parameters& par) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;
    const int nbuf = BFIELD_NCOMP * BFIELD_NG * (nx + 6) * (nz + 6);

    Kokkos::View<double*> send_ym("send_bfield_ym", nbuf), send_yp("send_bfield_yp", nbuf);
    Kokkos::View<double*> recv_ym("recv_bfield_ym", nbuf), recv_yp("recv_bfield_yp", nbuf);

    Kokkos::parallel_for("pack_bfield_y", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (nz + 6);
            const int comp = p / (BFIELD_NG * plane);
            int r = p - comp * BFIELD_NG * plane;
            const int g = r / plane;
            r -= g * plane;
            const int i = r / (nz + 6);
            const int k = r - i * (nz + 6);

            const int j_ym = 3 + g;
            const int j_yp = ny + g;

            double vym = 0.0, vyp = 0.0;
            if (comp == 0) { vym = bxb(stage,i,j_ym,k); vyp = bxb(stage,i,j_yp,k); }
            if (comp == 1) { vym = byb(stage,i,j_ym,k); vyp = byb(stage,i,j_yp,k); }
            if (comp == 2) { vym = bzb(stage,i,j_ym,k); vyp = bzb(stage,i,j_yp,k); }
            send_ym(p) = vym;
            send_yp(p) = vyp;
        });
    Kokkos::fence();

    auto h_send_ym = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_ym);
    auto h_send_yp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_yp);
    auto h_recv_ym = Kokkos::create_mirror_view(recv_ym);
    auto h_recv_yp = Kokkos::create_mirror_view(recv_yp);

    MPI_Sendrecv(h_send_ym.data(), nbuf, MPI_DOUBLE, par.nbr_ym, 920,
                 h_recv_yp.data(), nbuf, MPI_DOUBLE, par.nbr_yp, 920,
                 par.comm_cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_yp.data(), nbuf, MPI_DOUBLE, par.nbr_yp, 921,
                 h_recv_ym.data(), nbuf, MPI_DOUBLE, par.nbr_ym, 921,
                 par.comm_cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_ym, h_recv_ym);
    Kokkos::deep_copy(recv_yp, h_recv_yp);

    Kokkos::parallel_for("unpack_bfield_y", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (nz + 6);
            const int comp = p / (BFIELD_NG * plane);
            int r = p - comp * BFIELD_NG * plane;
            const int g = r / plane;
            r -= g * plane;
            const int i = r / (nz + 6);
            const int k = r - i * (nz + 6);

            const int j_ym = g;
            const int j_yp = ny + 3 + g;
            if (comp == 0) { bxb(stage,i,j_ym,k) = recv_ym(p); bxb(stage,i,j_yp,k) = recv_yp(p); }
            if (comp == 1) { byb(stage,i,j_ym,k) = recv_ym(p); byb(stage,i,j_yp,k) = recv_yp(p); }
            if (comp == 2) { bzb(stage,i,j_ym,k) = recv_ym(p); bzb(stage,i,j_yp,k) = recv_yp(p); }
        });
    Kokkos::fence();
}

static void exchange_bfield_z(int stage, const Parameters& par) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;
    const int nbuf = BFIELD_NCOMP * BFIELD_NG * (nx + 6) * (ny + 6);

    Kokkos::View<double*> send_zm("send_bfield_zm", nbuf), send_zp("send_bfield_zp", nbuf);
    Kokkos::View<double*> recv_zm("recv_bfield_zm", nbuf), recv_zp("recv_bfield_zp", nbuf);

    Kokkos::parallel_for("pack_bfield_z", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (ny + 6);
            const int comp = p / (BFIELD_NG * plane);
            int r = p - comp * BFIELD_NG * plane;
            const int g = r / plane;
            r -= g * plane;
            const int i = r / (ny + 6);
            const int j = r - i * (ny + 6);

            const int k_zm = 3 + g;
            const int k_zp = nz + g;

            double vzm = 0.0, vzp = 0.0;
            if (comp == 0) { vzm = bxb(stage,i,j,k_zm); vzp = bxb(stage,i,j,k_zp); }
            if (comp == 1) { vzm = byb(stage,i,j,k_zm); vzp = byb(stage,i,j,k_zp); }
            if (comp == 2) { vzm = bzb(stage,i,j,k_zm); vzp = bzb(stage,i,j,k_zp); }
            send_zm(p) = vzm;
            send_zp(p) = vzp;
        });
    Kokkos::fence();

    auto h_send_zm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_zm);
    auto h_send_zp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_zp);
    auto h_recv_zm = Kokkos::create_mirror_view(recv_zm);
    auto h_recv_zp = Kokkos::create_mirror_view(recv_zp);

    MPI_Sendrecv(h_send_zm.data(), nbuf, MPI_DOUBLE, par.nbr_zm, 930,
                 h_recv_zp.data(), nbuf, MPI_DOUBLE, par.nbr_zp, 930,
                 par.comm_cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_zp.data(), nbuf, MPI_DOUBLE, par.nbr_zp, 931,
                 h_recv_zm.data(), nbuf, MPI_DOUBLE, par.nbr_zm, 931,
                 par.comm_cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_zm, h_recv_zm);
    Kokkos::deep_copy(recv_zp, h_recv_zp);

    Kokkos::parallel_for("unpack_bfield_z", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (ny + 6);
            const int comp = p / (BFIELD_NG * plane);
            int r = p - comp * BFIELD_NG * plane;
            const int g = r / plane;
            r -= g * plane;
            const int i = r / (ny + 6);
            const int j = r - i * (ny + 6);

            const int k_zm = g;
            const int k_zp = nz + 3 + g;
            if (comp == 0) { bxb(stage,i,j,k_zm) = recv_zm(p); bxb(stage,i,j,k_zp) = recv_zp(p); }
            if (comp == 1) { byb(stage,i,j,k_zm) = recv_zm(p); byb(stage,i,j,k_zp) = recv_zp(p); }
            if (comp == 2) { bzb(stage,i,j,k_zm) = recv_zm(p); bzb(stage,i,j,k_zp) = recv_zp(p); }
        });
    Kokkos::fence();
}
#endif

void bound_bfield(int stage, const Parameters& par) {
#ifdef USE_MPI
    exchange_bfield_x(stage, par);
    exchange_bfield_y(stage, par);
    exchange_bfield_z(stage, par);
    apply_bfield_physical_boundaries(stage, par);
#else
    bound_bfield_local_all(stage, par);
#endif
}

void zero_ct_fluxes(CTFluxes& ct) {
    Kokkos::deep_copy(ct.fsy, 0.0); Kokkos::deep_copy(ct.fsz, 0.0);
    Kokkos::deep_copy(ct.gsx, 0.0); Kokkos::deep_copy(ct.gsz, 0.0);
    Kokkos::deep_copy(ct.hsx, 0.0); Kokkos::deep_copy(ct.hsy, 0.0);
    Kokkos::deep_copy(ct.Ox2, 0.0); Kokkos::deep_copy(ct.Oy2, 0.0); Kokkos::deep_copy(ct.Oz2, 0.0);
    Kokkos::deep_copy(ct.Ox, 0.0);  Kokkos::deep_copy(ct.Oy, 0.0);  Kokkos::deep_copy(ct.Oz, 0.0);
    Kokkos::fence();
}

// ================================================================
// CT flux / corner-EMF boundary handling.
//
// In serial mode, this is the old local periodic/open copy.
// In MPI mode, all CT arrays must be exchanged across rank boundaries,
// exactly like q and bxb/byb/bzb.  This is required because build_O2,
// smooth_O, and update_bfield use stencils that cross subdomain faces.
// ================================================================

template <class View3D>
static void bound_one_ct_array_local(View3D a,
                                     const Parameters& par,
                                     const std::string& label) {
    const bool xp = (par.x1bc == "periodic");
    const bool yp = (par.x2bc == "periodic");
    const bool zp = (par.x3bc == "periodic");
    const int nx = par.nx, ny = par.ny, nz = par.nz;

    Kokkos::parallel_for(label + "_x_local", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny+6,nz+6}),
        KOKKOS_LAMBDA(const int j, const int k) {
            if (xp) {
                a(0,j,k)=a(nx,j,k); a(1,j,k)=a(nx+1,j,k); a(2,j,k)=a(nx+2,j,k);
                a(nx+3,j,k)=a(3,j,k); a(nx+4,j,k)=a(4,j,k); a(nx+5,j,k)=a(5,j,k);
            } else {
                a(0,j,k)=a(3,j,k); a(1,j,k)=a(3,j,k); a(2,j,k)=a(3,j,k);
                a(nx+3,j,k)=a(nx+2,j,k); a(nx+4,j,k)=a(nx+2,j,k); a(nx+5,j,k)=a(nx+2,j,k);
            }
        });

    Kokkos::parallel_for(label + "_y_local", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,nz+6}),
        KOKKOS_LAMBDA(const int i, const int k) {
            if (yp) {
                a(i,0,k)=a(i,ny,k); a(i,1,k)=a(i,ny+1,k); a(i,2,k)=a(i,ny+2,k);
                a(i,ny+3,k)=a(i,3,k); a(i,ny+4,k)=a(i,4,k); a(i,ny+5,k)=a(i,5,k);
            } else {
                a(i,0,k)=a(i,3,k); a(i,1,k)=a(i,3,k); a(i,2,k)=a(i,3,k);
                a(i,ny+3,k)=a(i,ny+2,k); a(i,ny+4,k)=a(i,ny+2,k); a(i,ny+5,k)=a(i,ny+2,k);
            }
        });

    Kokkos::parallel_for(label + "_z_local", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,ny+6}),
        KOKKOS_LAMBDA(const int i, const int j) {
            if (zp) {
                a(i,j,0)=a(i,j,nz); a(i,j,1)=a(i,j,nz+1); a(i,j,2)=a(i,j,nz+2);
                a(i,j,nz+3)=a(i,j,3); a(i,j,nz+4)=a(i,j,4); a(i,j,nz+5)=a(i,j,5);
            } else {
                a(i,j,0)=a(i,j,3); a(i,j,1)=a(i,j,3); a(i,j,2)=a(i,j,3);
                a(i,j,nz+3)=a(i,j,nz+2); a(i,j,nz+4)=a(i,j,nz+2); a(i,j,nz+5)=a(i,j,nz+2);
            }
        });
    Kokkos::fence();
}

#ifdef USE_MPI
static constexpr int CT_NG = 3;

template <class View3D>
static void exchange_ct_array_x(View3D a, const Parameters& par, const std::string& label) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;
    const int nbuf = CT_NG * (ny + 6) * (nz + 6);

    Kokkos::View<double*> send_xm(label + "_send_xm", nbuf), send_xp(label + "_send_xp", nbuf);
    Kokkos::View<double*> recv_xm(label + "_recv_xm", nbuf), recv_xp(label + "_recv_xp", nbuf);

    Kokkos::parallel_for(label + "_pack_x", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (ny + 6) * (nz + 6);
            const int g = p / plane;
            int r = p - g * plane;
            const int j = r / (nz + 6);
            const int k = r - j * (nz + 6);

            send_xm(p) = a(3 + g, j, k);   // send left active cells to x- neighbor
            send_xp(p) = a(nx + g, j, k);  // send right active cells to x+ neighbor
        });
    Kokkos::fence();

    auto h_send_xm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_xm);
    auto h_send_xp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_xp);
    auto h_recv_xm = Kokkos::create_mirror_view(recv_xm);
    auto h_recv_xp = Kokkos::create_mirror_view(recv_xp);

    MPI_Sendrecv(h_send_xm.data(), nbuf, MPI_DOUBLE, par.nbr_xm, 1010,
                 h_recv_xp.data(), nbuf, MPI_DOUBLE, par.nbr_xp, 1010,
                 par.comm_cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_xp.data(), nbuf, MPI_DOUBLE, par.nbr_xp, 1011,
                 h_recv_xm.data(), nbuf, MPI_DOUBLE, par.nbr_xm, 1011,
                 par.comm_cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_xm, h_recv_xm);
    Kokkos::deep_copy(recv_xp, h_recv_xp);

    Kokkos::parallel_for(label + "_unpack_x", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (ny + 6) * (nz + 6);
            const int g = p / plane;
            int r = p - g * plane;
            const int j = r / (nz + 6);
            const int k = r - j * (nz + 6);

            a(g, j, k)        = recv_xm(p);      // left ghost  0,1,2
            a(nx + 3 + g,j,k) = recv_xp(p);      // right ghost nx+3,nx+4,nx+5
        });
    Kokkos::fence();
}

template <class View3D>
static void exchange_ct_array_y(View3D a, const Parameters& par, const std::string& label) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;
    const int nbuf = CT_NG * (nx + 6) * (nz + 6);

    Kokkos::View<double*> send_ym(label + "_send_ym", nbuf), send_yp(label + "_send_yp", nbuf);
    Kokkos::View<double*> recv_ym(label + "_recv_ym", nbuf), recv_yp(label + "_recv_yp", nbuf);

    Kokkos::parallel_for(label + "_pack_y", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (nz + 6);
            const int g = p / plane;
            int r = p - g * plane;
            const int i = r / (nz + 6);
            const int k = r - i * (nz + 6);

            send_ym(p) = a(i, 3 + g, k);
            send_yp(p) = a(i, ny + g, k);
        });
    Kokkos::fence();

    auto h_send_ym = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_ym);
    auto h_send_yp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_yp);
    auto h_recv_ym = Kokkos::create_mirror_view(recv_ym);
    auto h_recv_yp = Kokkos::create_mirror_view(recv_yp);

    MPI_Sendrecv(h_send_ym.data(), nbuf, MPI_DOUBLE, par.nbr_ym, 1020,
                 h_recv_yp.data(), nbuf, MPI_DOUBLE, par.nbr_yp, 1020,
                 par.comm_cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_yp.data(), nbuf, MPI_DOUBLE, par.nbr_yp, 1021,
                 h_recv_ym.data(), nbuf, MPI_DOUBLE, par.nbr_ym, 1021,
                 par.comm_cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_ym, h_recv_ym);
    Kokkos::deep_copy(recv_yp, h_recv_yp);

    Kokkos::parallel_for(label + "_unpack_y", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (nz + 6);
            const int g = p / plane;
            int r = p - g * plane;
            const int i = r / (nz + 6);
            const int k = r - i * (nz + 6);

            a(i, g, k)        = recv_ym(p);
            a(i, ny + 3 + g,k)= recv_yp(p);
        });
    Kokkos::fence();
}

template <class View3D>
static void exchange_ct_array_z(View3D a, const Parameters& par, const std::string& label) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;
    const int nbuf = CT_NG * (nx + 6) * (ny + 6);

    Kokkos::View<double*> send_zm(label + "_send_zm", nbuf), send_zp(label + "_send_zp", nbuf);
    Kokkos::View<double*> recv_zm(label + "_recv_zm", nbuf), recv_zp(label + "_recv_zp", nbuf);

    Kokkos::parallel_for(label + "_pack_z", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (ny + 6);
            const int g = p / plane;
            int r = p - g * plane;
            const int i = r / (ny + 6);
            const int j = r - i * (ny + 6);

            send_zm(p) = a(i, j, 3 + g);
            send_zp(p) = a(i, j, nz + g);
        });
    Kokkos::fence();

    auto h_send_zm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_zm);
    auto h_send_zp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_zp);
    auto h_recv_zm = Kokkos::create_mirror_view(recv_zm);
    auto h_recv_zp = Kokkos::create_mirror_view(recv_zp);

    MPI_Sendrecv(h_send_zm.data(), nbuf, MPI_DOUBLE, par.nbr_zm, 1030,
                 h_recv_zp.data(), nbuf, MPI_DOUBLE, par.nbr_zp, 1030,
                 par.comm_cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_zp.data(), nbuf, MPI_DOUBLE, par.nbr_zp, 1031,
                 h_recv_zm.data(), nbuf, MPI_DOUBLE, par.nbr_zm, 1031,
                 par.comm_cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_zm, h_recv_zm);
    Kokkos::deep_copy(recv_zp, h_recv_zp);

    Kokkos::parallel_for(label + "_unpack_z", Kokkos::RangePolicy<>(0, nbuf),
        KOKKOS_LAMBDA(const int p) {
            const int plane = (nx + 6) * (ny + 6);
            const int g = p / plane;
            int r = p - g * plane;
            const int i = r / (ny + 6);
            const int j = r - i * (ny + 6);

            a(i, j, g)        = recv_zm(p);
            a(i, j, nz+3+g)   = recv_zp(p);
        });
    Kokkos::fence();
}

template <class View3D>
static void apply_ct_physical_boundaries(View3D a,
                                         const Parameters& par,
                                         const std::string& label) {
    const int nx = par.nx, ny = par.ny, nz = par.nz;

    if (par.nbr_xm == MPI_PROC_NULL) {
        Kokkos::parallel_for(label + "_phys_xm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny+6,nz+6}),
            KOKKOS_LAMBDA(const int j, const int k) {
                a(0,j,k)=a(3,j,k); a(1,j,k)=a(3,j,k); a(2,j,k)=a(3,j,k);
            });
    }
    if (par.nbr_xp == MPI_PROC_NULL) {
        Kokkos::parallel_for(label + "_phys_xp", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny+6,nz+6}),
            KOKKOS_LAMBDA(const int j, const int k) {
                a(nx+3,j,k)=a(nx+2,j,k); a(nx+4,j,k)=a(nx+2,j,k); a(nx+5,j,k)=a(nx+2,j,k);
            });
    }
    if (par.nbr_ym == MPI_PROC_NULL) {
        Kokkos::parallel_for(label + "_phys_ym", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,nz+6}),
            KOKKOS_LAMBDA(const int i, const int k) {
                a(i,0,k)=a(i,3,k); a(i,1,k)=a(i,3,k); a(i,2,k)=a(i,3,k);
            });
    }
    if (par.nbr_yp == MPI_PROC_NULL) {
        Kokkos::parallel_for(label + "_phys_yp", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,nz+6}),
            KOKKOS_LAMBDA(const int i, const int k) {
                a(i,ny+3,k)=a(i,ny+2,k); a(i,ny+4,k)=a(i,ny+2,k); a(i,ny+5,k)=a(i,ny+2,k);
            });
    }
    if (par.nbr_zm == MPI_PROC_NULL) {
        Kokkos::parallel_for(label + "_phys_zm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,ny+6}),
            KOKKOS_LAMBDA(const int i, const int j) {
                a(i,j,0)=a(i,j,3); a(i,j,1)=a(i,j,3); a(i,j,2)=a(i,j,3);
            });
    }
    if (par.nbr_zp == MPI_PROC_NULL) {
        Kokkos::parallel_for(label + "_phys_zp", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,ny+6}),
            KOKKOS_LAMBDA(const int i, const int j) {
                a(i,j,nz+3)=a(i,j,nz+2); a(i,j,nz+4)=a(i,j,nz+2); a(i,j,nz+5)=a(i,j,nz+2);
            });
    }
    Kokkos::fence();
}
#endif

template <class View3D>
void bound_one_ct_array(View3D a, const Parameters& par, const std::string& label) {
#ifdef USE_MPI
    exchange_ct_array_x(a, par, label);
    exchange_ct_array_y(a, par, label);
    exchange_ct_array_z(a, par, label);
    apply_ct_physical_boundaries(a, par, label);
#else
    bound_one_ct_array_local(a, par, label);
#endif
}

void bound_ct_fluxes(CTFluxes& ct, const Parameters& par) {
    bound_one_ct_array(ct.fsy, par, "bound_fsy");
    bound_one_ct_array(ct.fsz, par, "bound_fsz");
    bound_one_ct_array(ct.gsx, par, "bound_gsx");
    bound_one_ct_array(ct.gsz, par, "bound_gsz");
    bound_one_ct_array(ct.hsx, par, "bound_hsx");
    bound_one_ct_array(ct.hsy, par, "bound_hsy");
}

void bound_ct_emf2(CTFluxes& ct, const Parameters& par) {
    bound_one_ct_array(ct.Ox2, par, "bound_Ox2");
    bound_one_ct_array(ct.Oy2, par, "bound_Oy2");
    bound_one_ct_array(ct.Oz2, par, "bound_Oz2");
}

void bound_ct_emf(CTFluxes& ct, const Parameters& par) {
    bound_one_ct_array(ct.Ox, par, "bound_Ox");
    bound_one_ct_array(ct.Oy, par, "bound_Oy");
    bound_one_ct_array(ct.Oz, par, "bound_Oz");
}

void fluxct(Kokkos::View<double*****> q, int iw, double k1, double k3,
            const Parameters& par, CTFluxes& ct) {

    const bool use_x = (par.nx > 1), use_y = (par.ny > 1), use_z = (par.nz > 1);
    const bool use_Ox = use_x && use_y;
    const bool use_Oy = use_y && use_z;
    const bool use_Oz = use_x && use_z;
    if (!use_Ox && !use_Oy && !use_Oz) return;

    const bool xp = (par.x1bc == "periodic");
    const bool yp = (par.x2bc == "periodic");
    const bool zp = (par.x3bc == "periodic");

    const int ic_lo=3, jc_lo=3, kc_lo=3;
    const int ic_hi=par.nx+2, jc_hi=par.ny+2, kc_hi=par.nz+2;
    const int full_lo=0, full_hi_x=par.nx+5, full_hi_y=par.ny+5, full_hi_z=par.nz+5;

    const int build_lo=0, build_hi_x=par.nx+3, build_hi_y=par.ny+3, build_hi_z=par.nz+3;
    const int smooth_lo=1, smooth_hi_x=par.nx+4, smooth_hi_y=par.ny+4, smooth_hi_z=par.nz+4;
    const int update_lo=1, update_hi_x=par.nx+3, update_hi_y=par.ny+3, update_hi_z=par.nz+3;

    bound_ct_fluxes(ct, par);

    Kokkos::parallel_for("fluxct_build_O2", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {build_lo,build_lo,build_lo}, {build_hi_x+1,build_hi_y+1,build_hi_z+1}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int im1=bc_index_ct(i-1,full_lo,full_hi_x,xp), ip1=bc_index_ct(i+1,full_lo,full_hi_x,xp), ip2=bc_index_ct(i+2,full_lo,full_hi_x,xp);
            const int jm1=bc_index_ct(j-1,full_lo,full_hi_y,yp), jp1=bc_index_ct(j+1,full_lo,full_hi_y,yp), jp2=bc_index_ct(j+2,full_lo,full_hi_y,yp);
            const int km1=bc_index_ct(k-1,full_lo,full_hi_z,zp), kp1=bc_index_ct(k+1,full_lo,full_hi_z,zp), kp2=bc_index_ct(k+2,full_lo,full_hi_z,zp);
            if (use_Ox) ct.Ox2(i,j,k)=(-ct.gsx(im1,j,k)+9.0*ct.gsx(i,j,k)+9.0*ct.gsx(ip1,j,k)-ct.gsx(ip2,j,k))/16.0
                                      -(-ct.fsy(i,jm1,k)+9.0*ct.fsy(i,j,k)+9.0*ct.fsy(i,jp1,k)-ct.fsy(i,jp2,k))/16.0;
            if (use_Oy) ct.Oy2(i,j,k)=(-ct.hsy(i,jm1,k)+9.0*ct.hsy(i,j,k)+9.0*ct.hsy(i,jp1,k)-ct.hsy(i,jp2,k))/16.0
                                      -(-ct.gsz(i,j,km1)+9.0*ct.gsz(i,j,k)+9.0*ct.gsz(i,j,kp1)-ct.gsz(i,j,kp2))/16.0;
            if (use_Oz) ct.Oz2(i,j,k)=(-ct.fsz(i,j,km1)+9.0*ct.fsz(i,j,k)+9.0*ct.fsz(i,j,kp1)-ct.fsz(i,j,kp2))/16.0
                                      -(-ct.hsx(im1,j,k)+9.0*ct.hsx(i,j,k)+9.0*ct.hsx(ip1,j,k)-ct.hsx(ip2,j,k))/16.0;
        });
    Kokkos::fence();

    // The corner EMF candidate must also be halo-filled before smoothing.
    // This matches the Fortran sequence: bound_priod(Ox2) before constructing Ox.
    bound_ct_emf2(ct, par);

    Kokkos::parallel_for("fluxct_smooth_O", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {smooth_lo,smooth_lo,smooth_lo}, {smooth_hi_x+1,smooth_hi_y+1,smooth_hi_z+1}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int im1=bc_index_ct(i-1,full_lo,full_hi_x,xp), ip1=bc_index_ct(i+1,full_lo,full_hi_x,xp);
            const int jm1=bc_index_ct(j-1,full_lo,full_hi_y,yp), jp1=bc_index_ct(j+1,full_lo,full_hi_y,yp);
            const int km1=bc_index_ct(k-1,full_lo,full_hi_z,zp), kp1=bc_index_ct(k+1,full_lo,full_hi_z,zp);
            if (use_Ox) ct.Ox(i,j,k)=ct.Ox2(i,j,k)+(ct.Ox2(im1,j,k)-2.0*ct.Ox2(i,j,k)+ct.Ox2(ip1,j,k))/24.0
                                                 +(ct.Ox2(i,jm1,k)-2.0*ct.Ox2(i,j,k)+ct.Ox2(i,jp1,k))/24.0;
            if (use_Oy) ct.Oy(i,j,k)=ct.Oy2(i,j,k)+(ct.Oy2(i,jm1,k)-2.0*ct.Oy2(i,j,k)+ct.Oy2(i,jp1,k))/24.0
                                                 +(ct.Oy2(i,j,km1)-2.0*ct.Oy2(i,j,k)+ct.Oy2(i,j,kp1))/24.0;
            if (use_Oz) ct.Oz(i,j,k)=ct.Oz2(i,j,k)+(ct.Oz2(im1,j,k)-2.0*ct.Oz2(i,j,k)+ct.Oz2(ip1,j,k))/24.0
                                                 +(ct.Oz2(i,j,km1)-2.0*ct.Oz2(i,j,k)+ct.Oz2(i,j,kp1))/24.0;
        });
    Kokkos::fence();

    // The smoothed EMF is used with difference stencils in update_bfield,
    // so it also needs rank-boundary halo exchange.
    // This matches the Fortran sequence: bound_priod(Ox) before updating b.
    bound_ct_emf(ct, par);

    const double aa=9.0/8.0, bb=-1.0/24.0, cc=0.0;

    Kokkos::parallel_for("fluxct_update_bfield", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {update_lo,update_lo,update_lo}, {update_hi_x+1,update_hi_y+1,update_hi_z+1}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int im1=bc_index_ct(i-1,full_lo,full_hi_x,xp), im2=bc_index_ct(i-2,full_lo,full_hi_x,xp), im3=bc_index_ct(i-3,full_lo,full_hi_x,xp);
            const int ip1=bc_index_ct(i+1,full_lo,full_hi_x,xp), ip2=bc_index_ct(i+2,full_lo,full_hi_x,xp);
            const int jm1=bc_index_ct(j-1,full_lo,full_hi_y,yp), jm2=bc_index_ct(j-2,full_lo,full_hi_y,yp), jm3=bc_index_ct(j-3,full_lo,full_hi_y,yp);
            const int jp1=bc_index_ct(j+1,full_lo,full_hi_y,yp), jp2=bc_index_ct(j+2,full_lo,full_hi_y,yp);
            const int km1=bc_index_ct(k-1,full_lo,full_hi_z,zp), km2=bc_index_ct(k-2,full_lo,full_hi_z,zp), km3=bc_index_ct(k-3,full_lo,full_hi_z,zp);
            const int kp1=bc_index_ct(k+1,full_lo,full_hi_z,zp), kp2=bc_index_ct(k+2,full_lo,full_hi_z,zp);
            if (use_Ox || use_Oz) {
                double dOx=0.0, dOz=0.0;
                if (use_Ox) dOx=aa*(ct.Ox(i,j,k)-ct.Ox(i,jm1,k))+bb*(ct.Ox(i,jp1,k)-ct.Ox(i,jm2,k))+cc*(ct.Ox(i,jp2,k)-ct.Ox(i,jm3,k));
                if (use_Oz) dOz=aa*(ct.Oz(i,j,k)-ct.Oz(i,j,km1))+bb*(ct.Oz(i,j,kp1)-ct.Oz(i,j,km2))+cc*(ct.Oz(i,j,kp2)-ct.Oz(i,j,km3));
                bxb(iw,i,j,k)=k1*bxb(iw,i,j,k)+k3*bxb(iw-1,i,j,k)-par.dtdy*dOx+par.dtdz*dOz;
            }
            if (use_Oy || use_Ox) {
                double dOy=0.0, dOx=0.0;
                if (use_Oy) dOy=aa*(ct.Oy(i,j,k)-ct.Oy(i,j,km1))+bb*(ct.Oy(i,j,kp1)-ct.Oy(i,j,km2))+cc*(ct.Oy(i,j,kp2)-ct.Oy(i,j,km3));
                if (use_Ox) dOx=aa*(ct.Ox(i,j,k)-ct.Ox(im1,j,k))+bb*(ct.Ox(ip1,j,k)-ct.Ox(im2,j,k))+cc*(ct.Ox(ip2,j,k)-ct.Ox(im3,j,k));
                byb(iw,i,j,k)=k1*byb(iw,i,j,k)+k3*byb(iw-1,i,j,k)-par.dtdz*dOy+par.dtdx*dOx;
            }
            if (use_Oz || use_Oy) {
                double dOz=0.0, dOy=0.0;
                if (use_Oz) dOz=aa*(ct.Oz(i,j,k)-ct.Oz(im1,j,k))+bb*(ct.Oz(ip1,j,k)-ct.Oz(im2,j,k))+cc*(ct.Oz(ip2,j,k)-ct.Oz(im3,j,k));
                if (use_Oy) dOy=aa*(ct.Oy(i,j,k)-ct.Oy(i,jm1,k))+bb*(ct.Oy(i,jp1,k)-ct.Oy(i,jm2,k))+cc*(ct.Oy(i,jp2,k)-ct.Oy(i,jm3,k));
                bzb(iw,i,j,k)=k1*bzb(iw,i,j,k)+k3*bzb(iw-1,i,j,k)-par.dtdx*dOz+par.dtdy*dOy;
            }
        });
    Kokkos::fence();

    bound_bfield(iw, par);

    Kokkos::parallel_for("fluxct_interp_b_to_q", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {ic_lo,jc_lo,kc_lo}, {ic_hi+1,jc_hi+1,kc_hi+1}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int im1=bc_index_ct(i-1,full_lo,full_hi_x,xp), im2=bc_index_ct(i-2,full_lo,full_hi_x,xp), im3=bc_index_ct(i-3,full_lo,full_hi_x,xp);
            const int ip1=bc_index_ct(i+1,full_lo,full_hi_x,xp), ip2=bc_index_ct(i+2,full_lo,full_hi_x,xp);
            const int jm1=bc_index_ct(j-1,full_lo,full_hi_y,yp), jm2=bc_index_ct(j-2,full_lo,full_hi_y,yp), jm3=bc_index_ct(j-3,full_lo,full_hi_y,yp);
            const int jp1=bc_index_ct(j+1,full_lo,full_hi_y,yp), jp2=bc_index_ct(j+2,full_lo,full_hi_y,yp);
            const int km1=bc_index_ct(k-1,full_lo,full_hi_z,zp), km2=bc_index_ct(k-2,full_lo,full_hi_z,zp), km3=bc_index_ct(k-3,full_lo,full_hi_z,zp);
            const int kp1=bc_index_ct(k+1,full_lo,full_hi_z,zp), kp2=bc_index_ct(k+2,full_lo,full_hi_z,zp);
            if (use_x) q(iw,4,i,j,k)=(3.0*bxb(iw,im3,j,k)-25.0*bxb(iw,im2,j,k)+150.0*bxb(iw,im1,j,k)+150.0*bxb(iw,i,j,k)-25.0*bxb(iw,ip1,j,k)+3.0*bxb(iw,ip2,j,k))/256.0;
            else       q(iw,4,i,j,k)=bxb(iw,i,j,k);
            if (use_y) q(iw,5,i,j,k)=(3.0*byb(iw,i,jm3,k)-25.0*byb(iw,i,jm2,k)+150.0*byb(iw,i,jm1,k)+150.0*byb(iw,i,j,k)-25.0*byb(iw,i,jp1,k)+3.0*byb(iw,i,jp2,k))/256.0;
            else       q(iw,5,i,j,k)=byb(iw,i,j,k);
            if (use_z) q(iw,6,i,j,k)=(3.0*bzb(iw,i,j,km3)-25.0*bzb(iw,i,j,km2)+150.0*bzb(iw,i,j,km1)+150.0*bzb(iw,i,j,k)-25.0*bzb(iw,i,j,kp1)+3.0*bzb(iw,i,j,kp2))/256.0;
            else       q(iw,6,i,j,k)=bzb(iw,i,j,k);
        });
    Kokkos::fence();
}


