#include "fluxct.hpp"

#include <Kokkos_Core.hpp>
#include <string>

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

void bound_bfield(int stage, const Parameters& par) {
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

void zero_ct_fluxes(CTFluxes& ct) {
    Kokkos::deep_copy(ct.fsy, 0.0); Kokkos::deep_copy(ct.fsz, 0.0);
    Kokkos::deep_copy(ct.gsx, 0.0); Kokkos::deep_copy(ct.gsz, 0.0);
    Kokkos::deep_copy(ct.hsx, 0.0); Kokkos::deep_copy(ct.hsy, 0.0);
    Kokkos::deep_copy(ct.Ox2, 0.0); Kokkos::deep_copy(ct.Oy2, 0.0); Kokkos::deep_copy(ct.Oz2, 0.0);
    Kokkos::deep_copy(ct.Ox, 0.0);  Kokkos::deep_copy(ct.Oy, 0.0);  Kokkos::deep_copy(ct.Oz, 0.0);
    Kokkos::fence();
}

template <class View3D>
void bound_one_ct_array(View3D a, const Parameters& par, const std::string& label) {
    const bool xp = (par.x1bc == "periodic");
    const bool yp = (par.x2bc == "periodic");
    const bool zp = (par.x3bc == "periodic");
    const int nx = par.nx, ny = par.ny, nz = par.nz;

    Kokkos::parallel_for(label + "_x", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny+6,nz+6}),
        KOKKOS_LAMBDA(const int j, const int k) {
            if (xp) {
                a(0,j,k)=a(nx,j,k); a(1,j,k)=a(nx+1,j,k); a(2,j,k)=a(nx+2,j,k);
                a(nx+3,j,k)=a(3,j,k); a(nx+4,j,k)=a(4,j,k); a(nx+5,j,k)=a(5,j,k);
            } else {
                a(0,j,k)=a(3,j,k); a(1,j,k)=a(3,j,k); a(2,j,k)=a(3,j,k);
                a(nx+3,j,k)=a(nx+2,j,k); a(nx+4,j,k)=a(nx+2,j,k); a(nx+5,j,k)=a(nx+2,j,k);
            }
        });

    Kokkos::parallel_for(label + "_y", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,nz+6}),
        KOKKOS_LAMBDA(const int i, const int k) {
            if (yp) {
                a(i,0,k)=a(i,ny,k); a(i,1,k)=a(i,ny+1,k); a(i,2,k)=a(i,ny+2,k);
                a(i,ny+3,k)=a(i,3,k); a(i,ny+4,k)=a(i,4,k); a(i,ny+5,k)=a(i,5,k);
            } else {
                a(i,0,k)=a(i,3,k); a(i,1,k)=a(i,3,k); a(i,2,k)=a(i,3,k);
                a(i,ny+3,k)=a(i,ny+2,k); a(i,ny+4,k)=a(i,ny+2,k); a(i,ny+5,k)=a(i,ny+2,k);
            }
        });

    Kokkos::parallel_for(label + "_z", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nx+6,ny+6}),
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

void bound_ct_fluxes(CTFluxes& ct, const Parameters& par) {
    bound_one_ct_array(ct.fsy, par, "bound_fsy");
    bound_one_ct_array(ct.fsz, par, "bound_fsz");
    bound_one_ct_array(ct.gsx, par, "bound_gsx");
    bound_one_ct_array(ct.gsz, par, "bound_gsz");
    bound_one_ct_array(ct.hsx, par, "bound_hsx");
    bound_one_ct_array(ct.hsy, par, "bound_hsy");
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

