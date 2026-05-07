#include "ssprk.hpp"
#include "parameters.hpp"
#include "bound.hpp"
#include "fluxct.hpp"
#include "prot.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

#ifdef USE_MPI
#include <mpi.h>
#endif


static constexpr int NVAR  = 8;
static constexpr int NSTAGE = 6;


// ================================================================
// Forward declarations
// C++ must see these prototypes before x_sweep/y_sweep/z_sweep call them.
// ================================================================
void primit(Kokkos::View<double**> qone,
            Kokkos::View<double**> uone,
            int nn,
            const Parameters& par);

void eigenst(Kokkos::View<double**> qone,
             Kokkos::View<double**> uone,
             Kokkos::View<double**> a,
             Kokkos::View<double**> F,
             Kokkos::View<double***> R,
             Kokkos::View<double***> L,
             int nn,
             const Parameters& par);

void weno(Kokkos::View<double**> qone,
          Kokkos::View<double**> uone,
          Kokkos::View<double**> a,
          Kokkos::View<double**> F,
          Kokkos::View<double***> R,
          Kokkos::View<double***> L,
          Kokkos::View<double*> bsy,
          Kokkos::View<double*> bsz,
          int nn,
          double dtdl,
          const Parameters& par);


// ================================================================
// Sign helper
// Must be declared before the x-workspace kernels that call it.
// ================================================================
KOKKOS_INLINE_FUNCTION
double sign1(const double x) {
    return (x >= 0.0) ? 1.0 : -1.0;
}

// ================================================================
// RK coefficients
// ================================================================
struct RKCoeff {
    double k1, k2, k3, k4, k5;
};

RKCoeff get_rk_coeff(const int iw) {
    RKCoeff c{0.0, 0.0, 0.0, 0.0, 0.0};

    if (iw == 1) {
        c.k1 = 1.0;
        c.k2 = 0.39175222700392;
        c.k3 = 0.0;
    } else if (iw == 2) {
        c.k1 = 0.44437049406734;
        c.k2 = 0.36841059262959;
        c.k3 = 0.55562950593266;
    } else if (iw == 3) {
        c.k1 = 0.62010185138540;
        c.k2 = 0.25189177424738;
        c.k3 = 0.37989814861460;
    } else if (iw == 4) {
        c.k1 = 0.17807995410773;
        c.k2 = 0.54497475021237;
        c.k3 = 0.82192004589227;
    } else if (iw == 5) {
        c.k1 = -2.081261929715610e-02;
        c.k2 = 0.22600748319395;
        c.k3 = 5.03580947213895e-1;
        c.k4 = 0.51723167208978;
        c.k5 = -6.518979800418380e-12;
    }

    return c;
}

// ================================================================
// Stage copy
// ================================================================
void copy_stage(Kokkos::View<double*****> q,
                const int dst,
                const int src,
                const Parameters& par) {

    Kokkos::parallel_for(
        "copy_stage",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, 0, 0, 0},
            {NVAR, par.nx + 6, par.ny + 6, par.nz + 6}
        ),
        KOKKOS_LAMBDA(const int m, const int i, const int j, const int k) {
            q(dst,m,i,j,k) = q(src,m,i,j,k);
        }
    );

    Kokkos::fence();
}

// ================================================================
// Stage 5 special preparation
// ================================================================
void prepare_stage5(Kokkos::View<double*****> q,
                    const RKCoeff c,
                    const Parameters& par) {

    Kokkos::parallel_for(
        "prepare_stage5",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, 0, 0, 0},
            {NVAR, par.nx + 6, par.ny + 6, par.nz + 6}
        ),
        KOKKOS_LAMBDA(const int m, const int i, const int j, const int k) {
            q(5,m,i,j,k) =
                q(0,m,i,j,k)
              + (c.k4 / c.k1) * q(2,m,i,j,k)
              + (c.k5 / c.k1) * q(3,m,i,j,k);
        }
    );

    Kokkos::fence();
}

// ================================================================
// X sweep load/store
// ================================================================
void load_x_line(Kokkos::View<double*****> q,
                 Kokkos::View<double**> qone,
                 int stage,
                 int jj,
                 int kk,
                 const Parameters& par) {

    Kokkos::parallel_for(
        "load_x_line",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {NVAR, par.nx + 6}
        ),
        KOKKOS_LAMBDA(const int m, const int ii) {
            qone(m,ii) = q(stage,m,ii,jj,kk);
        }
    );

    Kokkos::fence();
}

void store_x_line(Kokkos::View<double*****> q,
                  Kokkos::View<double**> qone,
                  int iw,
                  double k1,
                  double k3,
                  int jj,
                  int kk,
                  const Parameters& par) {

    Kokkos::parallel_for(
        "store_x_line",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 3},
            {NVAR, par.nx + 3}
        ),
        KOKKOS_LAMBDA(const int m, const int ii) {
            q(iw,m,ii,jj,kk) =
                k1 * q(iw,m,ii,jj,kk)
              + k3 * q(iw-1,m,ii,jj,kk)
              + qone(m,ii);
        }
    );

    Kokkos::fence();
}

// ================================================================
// Y sweep load/store with variable permutation
// y: rho, My, Mz, Mx, By, Bz, Bx, E
// ================================================================
void load_y_line(Kokkos::View<double*****> q,
                 Kokkos::View<double**> qone,
                 int stage,
                 int ii,
                 int kk,
                 const Parameters& par) {

    Kokkos::parallel_for(
        "load_y_line",
        Kokkos::RangePolicy<>(0, par.ny + 6),
        KOKKOS_LAMBDA(const int jj) {
            qone(0,jj) = q(stage,0,ii,jj,kk);
            qone(1,jj) = q(stage,2,ii,jj,kk);
            qone(2,jj) = q(stage,3,ii,jj,kk);
            qone(3,jj) = q(stage,1,ii,jj,kk);
            qone(4,jj) = q(stage,5,ii,jj,kk);
            qone(5,jj) = q(stage,6,ii,jj,kk);
            qone(6,jj) = q(stage,4,ii,jj,kk);
            qone(7,jj) = q(stage,7,ii,jj,kk);
        }
    );

    Kokkos::fence();
}

void store_y_line(Kokkos::View<double*****> q,
                  Kokkos::View<double**> qone,
                  int iw,
                  int ii,
                  int kk,
                  const Parameters& par) {

    Kokkos::parallel_for(
        "store_y_line",
        Kokkos::RangePolicy<>(3, par.ny + 3),
        KOKKOS_LAMBDA(const int jj) {
            q(iw,0,ii,jj,kk) += qone(0,jj);
            q(iw,2,ii,jj,kk) += qone(1,jj);
            q(iw,3,ii,jj,kk) += qone(2,jj);
            q(iw,1,ii,jj,kk) += qone(3,jj);
            q(iw,5,ii,jj,kk) += qone(4,jj);
            q(iw,6,ii,jj,kk) += qone(5,jj);
            q(iw,4,ii,jj,kk) += qone(6,jj);
            q(iw,7,ii,jj,kk) += qone(7,jj);
        }
    );

    Kokkos::fence();
}

// ================================================================
// Z sweep load/store with variable permutation
// z: rho, Mz, Mx, My, Bz, Bx, By, E
// ================================================================
void load_z_line(Kokkos::View<double*****> q,
                 Kokkos::View<double**> qone,
                 int stage,
                 int ii,
                 int jj,
                 const Parameters& par) {

    Kokkos::parallel_for(
        "load_z_line",
        Kokkos::RangePolicy<>(0, par.nz + 6),
        KOKKOS_LAMBDA(const int kk) {
            qone(0,kk) = q(stage,0,ii,jj,kk);
            qone(1,kk) = q(stage,3,ii,jj,kk);
            qone(2,kk) = q(stage,1,ii,jj,kk);
            qone(3,kk) = q(stage,2,ii,jj,kk);
            qone(4,kk) = q(stage,6,ii,jj,kk);
            qone(5,kk) = q(stage,4,ii,jj,kk);
            qone(6,kk) = q(stage,5,ii,jj,kk);
            qone(7,kk) = q(stage,7,ii,jj,kk);
        }
    );

    Kokkos::fence();
}

void store_z_line(Kokkos::View<double*****> q,
                  Kokkos::View<double**> qone,
                  int iw,
                  int ii,
                  int jj,
                  const Parameters& par) {

    Kokkos::parallel_for(
        "store_z_line",
        Kokkos::RangePolicy<>(3, par.nz + 3),
        KOKKOS_LAMBDA(const int kk) {
            q(iw,0,ii,jj,kk) += qone(0,kk);
            q(iw,3,ii,jj,kk) += qone(1,kk);
            q(iw,1,ii,jj,kk) += qone(2,kk);
            q(iw,2,ii,jj,kk) += qone(3,kk);
            q(iw,6,ii,jj,kk) += qone(4,kk);
            q(iw,4,ii,jj,kk) += qone(5,kk);
            q(iw,5,ii,jj,kk) += qone(6,kk);
            q(iw,7,ii,jj,kk) += qone(7,kk);
        }
    );

    Kokkos::fence();
}

// ================================================================
// Store CT fluxes from directional sweeps
// ================================================================

void store_ct_x(CTFluxes& ct,
                Kokkos::View<double*> bsy,
                Kokkos::View<double*> bsz,
                int jj,
                int kk,
                const Parameters& par) {

    // x-sweep produces transverse magnetic fluxes for By and Bz.
    Kokkos::parallel_for(
        "store_ct_x",
        Kokkos::RangePolicy<>(0, par.nx + 1),
        KOKKOS_LAMBDA(const int i) {
            const int ii = i + 2;
            ct.fsy(ii,jj,kk) = bsy(i);
            ct.fsz(ii,jj,kk) = bsz(i);
        }
    );

    Kokkos::fence();
}

void store_ct_y(CTFluxes& ct,
                Kokkos::View<double*> bsy,
                Kokkos::View<double*> bsz,
                int ii,
                int kk,
                const Parameters& par) {

    // y-sweep variable permutation:
    // qone transverse magnetic fields are Bz and Bx.
    Kokkos::parallel_for(
        "store_ct_y",
        Kokkos::RangePolicy<>(0, par.ny + 1),
        KOKKOS_LAMBDA(const int j) {
            const int jj = j + 2;
            ct.gsz(ii,jj,kk) = bsy(j);
            ct.gsx(ii,jj,kk) = bsz(j);
        }
    );

    Kokkos::fence();
}

void store_ct_z(CTFluxes& ct,
                Kokkos::View<double*> bsy,
                Kokkos::View<double*> bsz,
                int ii,
                int jj,
                const Parameters& par) {

    // z-sweep variable permutation:
    // qone transverse magnetic fields are Bx and By.
    Kokkos::parallel_for(
        "store_ct_z",
        Kokkos::RangePolicy<>(0, par.nz + 1),
        KOKKOS_LAMBDA(const int k) {
            const int kk = k + 2;
            ct.hsx(ii,jj,kk) = bsy(k);
            ct.hsy(ii,jj,kk) = bsz(k);
        }
    );

    Kokkos::fence();
}

// ================================================================
// Sweep drivers
// ================================================================


// ================================================================
// Experimental x-sweep all-line workspace version.
//
// This removes the host loop over x-lines and reduces Kokkos kernel
// launches. Only x_sweep is replaced; y/z sweeps are kept unchanged
// for safer step-by-step validation.
// ================================================================
void load_x_all_lines(Kokkos::View<double*****> q,
                      Kokkos::View<double***> qx,
                      int stage,
                      const Parameters& par) {
    const int nxp = par.nx + 6;
    const int nyp = par.ny + 6;
    const int nzp = par.nz + 6;
    const int nlines = nyp * nzp;

    Kokkos::parallel_for(
        "load_x_all_lines",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {nlines, NVAR, nxp}
        ),
        KOKKOS_LAMBDA(const int line, const int m, const int ii) {
            const int jj = line % nyp;
            const int kk = line / nyp;
            qx(line,m,ii) = q(stage,m,ii,jj,kk);
        }
    );
    Kokkos::fence();
}

void store_x_all_lines(Kokkos::View<double*****> q,
                       Kokkos::View<double***> qx,
                       int iw,
                       double k1,
                       double k3,
                       const Parameters& par) {
    const int nyp = par.ny + 6;
    const int nzp = par.nz + 6;
    const int nlines = nyp * nzp;

    Kokkos::parallel_for(
        "store_x_all_lines",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 3},
            {nlines, NVAR, par.nx + 3}
        ),
        KOKKOS_LAMBDA(const int line, const int m, const int ii) {
            const int jj = line % nyp;
            const int kk = line / nyp;
            q(iw,m,ii,jj,kk) =
                k1 * q(iw,m,ii,jj,kk)
              + k3 * q(iw-1,m,ii,jj,kk)
              + qx(line,m,ii);
        }
    );
    Kokkos::fence();
}

void store_ct_x_all(CTFluxes& ct,
                    Kokkos::View<double**> bsyx,
                    Kokkos::View<double**> bszx,
                    const Parameters& par) {
    const int nyp = par.ny + 6;
    const int nzp = par.nz + 6;
    const int nlines = nyp * nzp;

    Kokkos::parallel_for(
        "store_ct_x_all",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {nlines, par.nx + 1}
        ),
        KOKKOS_LAMBDA(const int line, const int i) {
            const int jj = line % nyp;
            const int kk = line / nyp;
            const int ii = i + 2;
            ct.fsy(ii,jj,kk) = bsyx(line,i);
            ct.fsz(ii,jj,kk) = bszx(line,i);
        }
    );
    Kokkos::fence();
}

void primit_x_all(Kokkos::View<double***> qx,
                  Kokkos::View<double***> ux,
                  int nlines,
                  int nn,
                  const Parameters& par) {
    Kokkos::parallel_for(
        "primit_x_all",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {nlines, nn + 6}
        ),
        KOKKOS_LAMBDA(const int line, const int ii) {
            const double DD = qx(line,0,ii);
            const double Mx = qx(line,1,ii);
            const double My = qx(line,2,ii);
            const double Mz = qx(line,3,ii);
            const double Bx = qx(line,4,ii);
            const double By = qx(line,5,ii);
            const double Bz = qx(line,6,ii);
            double EE       = qx(line,7,ii);

            double rho = DD;

            const double vx = Mx / rho;
            const double vy = My / rho;
            const double vz = Mz / rho;

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;

            double pg = (par.gam - 1.0)
                      * (EE - 0.5 * (rho*vv2 + BB2));

            if (rho < par.rhomin || pg < par.pgmin) {
                rho = fmax(par.rhomin, rho);
                pg  = fmax(par.pgmin, pg);

                EE = pg / (par.gam - 1.0)
                   + 0.5 * (rho*vv2 + BB2);

                qx(line,0,ii) = rho;
                qx(line,7,ii) = EE;
            }

            const double HH = (EE + pg) / rho;

            ux(line,0,ii) = rho;
            ux(line,1,ii) = vx;
            ux(line,2,ii) = vy;
            ux(line,3,ii) = vz;
            ux(line,4,ii) = Bx;
            ux(line,5,ii) = By;
            ux(line,6,ii) = Bz;
            ux(line,7,ii) = pg;
            ux(line,8,ii) = HH;
        }
    );
    Kokkos::fence();
}


void eigenst_x_all(Kokkos::View<double***> qx,
                  Kokkos::View<double***> ux,
                  Kokkos::View<double***> ax,
                  Kokkos::View<double***> Fx,
                  Kokkos::View<double****> Rx,
                  Kokkos::View<double****> Lx,
                  int nlines,
                  int nn,
                  const Parameters& par) {

    const double gam0 = 1.0 - par.gam;
    const double gam1 = 0.5 * (par.gam - 1.0);
    const double gam2 = (par.gam - 2.0) / (par.gam - 1.0);

    // ------------------------------------------------------------
    // cell-center values: eigenvalues a and flux F
    // Fortran i = -2 ... nn+3
    // C++ ii = i + 2 = 0 ... nn+5
    // ------------------------------------------------------------
    Kokkos::parallel_for(
        "eigenst_x_all_center",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {nlines, nn + 6}
        ),
        KOKKOS_LAMBDA(const int line, const int ii) {

            const double DD = qx(line,0,ii);
            const double Mx = qx(line,1,ii);
            const double My = qx(line,2,ii);
            const double Mz = qx(line,3,ii);
            const double EE = qx(line,7,ii);

            const double rho = ux(line,0,ii);
            const double vx  = ux(line,1,ii);
            const double vy  = ux(line,2,ii);
            const double vz  = ux(line,3,ii);
            const double Bx  = ux(line,4,ii);
            const double By  = ux(line,5,ii);
            const double Bz  = ux(line,6,ii);
            const double pg  = ux(line,7,ii);

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;
            const double pt  = pg + 0.5 * BB2;
            const double vdB = vx*Bx + vy*By + vz*Bz;

            const double bbn2 = BB2 / rho;
            const double bnx2 = Bx*Bx / rho;

            const double cs2 = fmax(0.0, par.gam * fabs(pg / rho));

            const double root =
                sqrt(fmax(0.0, (bbn2 + cs2)*(bbn2 + cs2)
                         - 4.0 * bnx2 * cs2));

            const double lf = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 + root)));
            const double la = sqrt(fmax(0.0, bnx2));
            const double ls = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 - root)));

            ax(line,0,ii) = vx - lf;
            ax(line,1,ii) = vx - la;
            ax(line,2,ii) = vx - ls;
            ax(line,3,ii) = vx;
            ax(line,4,ii) = vx + ls;
            ax(line,5,ii) = vx + la;
            ax(line,6,ii) = vx + lf;

            Fx(line,0,ii) = DD * vx;
            Fx(line,1,ii) = Mx * vx + pt - Bx*Bx;
            Fx(line,2,ii) = My * vx - Bx*By;
            Fx(line,3,ii) = Mz * vx - Bx*Bz;
            Fx(line,4,ii) = By * vx - Bx*vy;
            Fx(line,5,ii) = Bz * vx - Bx*vz;
            Fx(line,6,ii) = (EE + pt) * vx - Bx * vdB;
        }
    );

    // ------------------------------------------------------------
    // cell-boundary values: left/right eigenvectors
    // Fortran i = -1 ... nn+1
    // C++ ii = i + 2 = 1 ... nn+3
    // ------------------------------------------------------------
    Kokkos::parallel_for(
        "eigenst_x_all_boundary",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 1},
            {nlines, nn + 4}
        ),
        KOKKOS_LAMBDA(const int line, const int ii) {

            const double rho = 0.5 * (ux(line,0,ii) + ux(line,0,ii+1));
            const double vx  = 0.5 * (ux(line,1,ii) + ux(line,1,ii+1));
            const double vy  = 0.5 * (ux(line,2,ii) + ux(line,2,ii+1));
            const double vz  = 0.5 * (ux(line,3,ii) + ux(line,3,ii+1));
            const double Bx  = 0.5 * (ux(line,4,ii) + ux(line,4,ii+1));
            const double By  = 0.5 * (ux(line,5,ii) + ux(line,5,ii+1));
            const double Bz  = 0.5 * (ux(line,6,ii) + ux(line,6,ii+1));
            const double pg  = 0.5 * (ux(line,7,ii) + ux(line,7,ii+1));
            const double HH  = 0.5 * (ux(line,8,ii) + ux(line,8,ii+1));

            const double vv2  = vx*vx + vy*vy + vz*vz;
            const double BB2  = Bx*Bx + By*By + Bz*Bz;
            const double bbn2 = BB2 / rho;
            const double bnx2 = Bx*Bx / rho;

            const double cs2 = (par.gam - 1.0)
                             * (HH - 0.5 * (vv2 + bbn2));
            const double cs  = sqrt(fmax(0.0, cs2));

            const double root =
                sqrt(fmax(0.0, (bbn2 + cs2)*(bbn2 + cs2)
                         - 4.0 * bnx2 * cs2));

            const double lf = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 + root)));
            const double la = sqrt(fmax(0.0, bnx2));
            const double ls = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 - root)));

            const double Bt2 = By*By + Bz*Bz;
            const double sgnBx = sign1(Bx);

            double bty, btz;
            if (Bt2 >= 1.0e-30) {
                bty = By / sqrt(Bt2);
                btz = Bz / sqrt(Bt2);
            } else {
                bty = 1.0 / sqrt(2.0);
                btz = 1.0 / sqrt(2.0);
            }

            double af, as;
            if ((lf*lf - ls*ls) >= 1.0e-30) {
                af = sqrt(fmax(0.0, cs2 - ls*ls)) / sqrt(lf*lf - ls*ls);
                as = sqrt(fmax(0.0, lf*lf - cs2)) / sqrt(lf*lf - ls*ls);
            } else {
                af = 1.0;
                as = 1.0;
            }

            const double sqrtrho = sqrt(rho);
            const double invsqrtrho = 1.0 / sqrtrho;

            // ---------------- left eigenvectors Lx(line,row,col,ii)
            Lx(line,0,0,ii) =  af*(gam1*vv2 + lf*vx) - as*ls*(bty*vy + btz*vz)*sgnBx;
            Lx(line,0,1,ii) =  af*(gam0*vx - lf);
            Lx(line,0,2,ii) =  gam0*af*vy + as*ls*bty*sgnBx;
            Lx(line,0,3,ii) =  gam0*af*vz + as*ls*btz*sgnBx;
            Lx(line,0,4,ii) =  gam0*af*By + cs*as*bty*sqrtrho;
            Lx(line,0,5,ii) =  gam0*af*Bz + cs*as*btz*sqrtrho;
            Lx(line,0,6,ii) = -gam0*af;

            Lx(line,1,0,ii) =  btz*vy - bty*vz;
            Lx(line,1,1,ii) =  0.0;
            Lx(line,1,2,ii) = -btz;
            Lx(line,1,3,ii) =  bty;
            Lx(line,1,4,ii) = -btz*sgnBx*sqrtrho;
            Lx(line,1,5,ii) =  bty*sgnBx*sqrtrho;
            Lx(line,1,6,ii) =  0.0;

            Lx(line,2,0,ii) =  as*(gam1*vv2 + ls*vx) + af*lf*(bty*vy + btz*vz)*sgnBx;
            Lx(line,2,1,ii) =  gam0*as*vx - as*ls;
            Lx(line,2,2,ii) =  gam0*as*vy - af*lf*bty*sgnBx;
            Lx(line,2,3,ii) =  gam0*as*vz - af*lf*btz*sgnBx;
            Lx(line,2,4,ii) =  gam0*as*By - cs*af*bty*sqrtrho;
            Lx(line,2,5,ii) =  gam0*as*Bz - cs*af*btz*sqrtrho;
            Lx(line,2,6,ii) = -gam0*as;

            Lx(line,3,0,ii) = -cs2/gam0 - 0.5*vv2;
            Lx(line,3,1,ii) =  vx;
            Lx(line,3,2,ii) =  vy;
            Lx(line,3,3,ii) =  vz;
            Lx(line,3,4,ii) =  By;
            Lx(line,3,5,ii) =  Bz;
            Lx(line,3,6,ii) = -1.0;

            Lx(line,4,0,ii) =  as*(gam1*vv2 - ls*vx) - af*lf*(bty*vy + btz*vz)*sgnBx;
            Lx(line,4,1,ii) =  as*(gam0*vx + ls);
            Lx(line,4,2,ii) =  gam0*as*vy + af*lf*bty*sgnBx;
            Lx(line,4,3,ii) =  gam0*as*vz + af*lf*btz*sgnBx;
            Lx(line,4,4,ii) =  gam0*as*By - cs*af*bty*sqrtrho;
            Lx(line,4,5,ii) =  gam0*as*Bz - cs*af*btz*sqrtrho;
            Lx(line,4,6,ii) = -gam0*as;

            Lx(line,5,0,ii) =  btz*vy - bty*vz;
            Lx(line,5,1,ii) =  0.0;
            Lx(line,5,2,ii) = -btz;
            Lx(line,5,3,ii) =  bty;
            Lx(line,5,4,ii) =  btz*sgnBx*sqrtrho;
            Lx(line,5,5,ii) = -bty*sgnBx*sqrtrho;
            Lx(line,5,6,ii) =  0.0;

            Lx(line,6,0,ii) =  af*(gam1*vv2 - lf*vx) + as*ls*(bty*vy + btz*vz)*sgnBx;
            Lx(line,6,1,ii) =  af*(gam0*vx + lf);
            Lx(line,6,2,ii) =  gam0*af*vy - as*ls*bty*sgnBx;
            Lx(line,6,3,ii) =  gam0*af*vz - as*ls*btz*sgnBx;
            Lx(line,6,4,ii) =  gam0*af*By + cs*as*bty*sqrtrho;
            Lx(line,6,5,ii) =  gam0*af*Bz + cs*as*btz*sqrtrho;
            Lx(line,6,6,ii) = -gam0*af;

            for (int m = 0; m < 7; ++m) {
                Lx(line,0,m,ii) = 0.5 * Lx(line,0,m,ii) / cs2;
                Lx(line,1,m,ii) = 0.5 * Lx(line,1,m,ii);
                Lx(line,2,m,ii) = 0.5 * Lx(line,2,m,ii) / cs2;
                Lx(line,3,m,ii) = -gam0 * Lx(line,3,m,ii) / cs2;
                Lx(line,4,m,ii) = 0.5 * Lx(line,4,m,ii) / cs2;
                Lx(line,5,m,ii) = 0.5 * Lx(line,5,m,ii);
                Lx(line,6,m,ii) = 0.5 * Lx(line,6,m,ii) / cs2;
            }

            // ---------------- right eigenvectors Rx(line,row,col,ii)
            Rx(line,0,0,ii) = af;
            Rx(line,1,0,ii) = af*(vx-lf);
            Rx(line,2,0,ii) = af*vy + as*ls*bty*sgnBx;
            Rx(line,3,0,ii) = af*vz + as*ls*btz*sgnBx;
            Rx(line,4,0,ii) = cs*as*bty*invsqrtrho;
            Rx(line,5,0,ii) = cs*as*btz*invsqrtrho;
            Rx(line,6,0,ii) = af*(lf*lf - lf*vx + 0.5*vv2 - gam2*cs2)
                       + as*ls*(bty*vy + btz*vz)*sgnBx;

            Rx(line,0,1,ii) = 0.0;
            Rx(line,1,1,ii) = 0.0;
            Rx(line,2,1,ii) = -btz;
            Rx(line,3,1,ii) = bty;
            Rx(line,4,1,ii) = -btz*sgnBx*invsqrtrho;
            Rx(line,5,1,ii) = bty*sgnBx*invsqrtrho;
            Rx(line,6,1,ii) = bty*vz - btz*vy;

            Rx(line,0,2,ii) = as;
            Rx(line,1,2,ii) = as*(vx-ls);
            Rx(line,2,2,ii) = as*vy - af*lf*bty*sgnBx;
            Rx(line,3,2,ii) = as*vz - af*lf*btz*sgnBx;
            Rx(line,4,2,ii) = -cs*af*bty*invsqrtrho;
            Rx(line,5,2,ii) = -cs*af*btz*invsqrtrho;
            Rx(line,6,2,ii) = as*(ls*ls - ls*vx + 0.5*vv2 - gam2*cs2)
                       - af*lf*(bty*vy + btz*vz)*sgnBx;

            Rx(line,0,3,ii) = 1.0;
            Rx(line,1,3,ii) = vx;
            Rx(line,2,3,ii) = vy;
            Rx(line,3,3,ii) = vz;
            Rx(line,4,3,ii) = 0.0;
            Rx(line,5,3,ii) = 0.0;
            Rx(line,6,3,ii) = 0.5*vv2;

            Rx(line,0,4,ii) = as;
            Rx(line,1,4,ii) = as*(vx+ls);
            Rx(line,2,4,ii) = as*vy + af*lf*bty*sgnBx;
            Rx(line,3,4,ii) = as*vz + af*lf*btz*sgnBx;
            Rx(line,4,4,ii) = -cs*af*bty*invsqrtrho;
            Rx(line,5,4,ii) = -cs*af*btz*invsqrtrho;
            Rx(line,6,4,ii) = as*(ls*ls + ls*vx + 0.5*vv2 - gam2*cs2)
                       + af*lf*(bty*vy + btz*vz)*sgnBx;

            Rx(line,0,5,ii) = 0.0;
            Rx(line,1,5,ii) = 0.0;
            Rx(line,2,5,ii) = -btz;
            Rx(line,3,5,ii) = bty;
            Rx(line,4,5,ii) = btz*sgnBx*invsqrtrho;
            Rx(line,5,5,ii) = -bty*sgnBx*invsqrtrho;
            Rx(line,6,5,ii) = bty*vz - btz*vy;

            Rx(line,0,6,ii) = af;
            Rx(line,1,6,ii) = af*(vx+lf);
            Rx(line,2,6,ii) = af*vy - as*ls*bty*sgnBx;
            Rx(line,3,6,ii) = af*vz - as*ls*btz*sgnBx;
            Rx(line,4,6,ii) = cs*as*bty*invsqrtrho;
            Rx(line,5,6,ii) = cs*as*btz*invsqrtrho;
            Rx(line,6,6,ii) = af*(lf*lf + lf*vx + 0.5*vv2 - gam2*cs2)
                       - as*ls*(bty*vy + btz*vz)*sgnBx;

            double sgnBt;
            if (By != 0.0) {
                sgnBt = sign1(By);
            } else {
                sgnBt = sign1(Bz);
            }

            if (cs >= la) {
                for (int m = 0; m < 7; ++m) {
                    Lx(line,2,m,ii) *= sgnBt;
                    Lx(line,4,m,ii) *= sgnBt;
                }
            } else {
                for (int m = 0; m < 7; ++m) {
                    Lx(line,0,m,ii) *= sgnBt;
                    Lx(line,6,m,ii) *= sgnBt;
                }
            }

            if (cs >= la) {
                for (int m = 0; m < 7; ++m) {
                    Rx(line,m,2,ii) *= sgnBt;
                    Rx(line,m,4,ii) *= sgnBt;
                }
            } else {
                for (int m = 0; m < 7; ++m) {
                    Rx(line,m,0,ii) *= sgnBt;
                    Rx(line,m,6,ii) *= sgnBt;
                }
            }
        }
    );

    Kokkos::fence();
}

void weno_x_all(Kokkos::View<double***> qx,
                Kokkos::View<double***> ux,
                Kokkos::View<double***> ax,
                Kokkos::View<double***> Fx,
                Kokkos::View<double****> Rx,
                Kokkos::View<double****> Lx,
                Kokkos::View<double**> bsyx,
                Kokkos::View<double**> bszx,
                int nlines,
                int nn,
                double dtdl,
                const Parameters& par) {

    Kokkos::View<double***> dF("dF_x_all", nlines, 7, nn + 1);
    Kokkos::deep_copy(dF, 0.0);

    Kokkos::parallel_for(
        "weno_x_all_flux",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {nlines, nn + 1, 7}
        ),
        KOKKOS_LAMBDA(const int line, const int i, const int m) {

            const double eps = 1.0e-10;

            double amx = 0.0;
            for (int s = -2; s <= 3; ++s) {
                amx = fmax(amx, fabs(ax(line,m, i + s + 2)));
            }

            double Fsk[6];
            double qsk[6];
            double dFsk[5];
            double dqsk[5];

            for (int ks = 0; ks < 6; ++ks) {
                const int ii = i - 2 + ks; 

                Fsk[ks] =
                    Lx(line,m,0,i+2) * Fx(line,0,ii+2)
                  + Lx(line,m,1,i+2) * Fx(line,1,ii+2)
                  + Lx(line,m,2,i+2) * Fx(line,2,ii+2)
                  + Lx(line,m,3,i+2) * Fx(line,3,ii+2)
                  + Lx(line,m,4,i+2) * Fx(line,4,ii+2)
                  + Lx(line,m,5,i+2) * Fx(line,5,ii+2)
                  + Lx(line,m,6,i+2) * Fx(line,6,ii+2);

                qsk[ks] =
                    Lx(line,m,0,i+2) * qx(line,0,ii+2)
                  + Lx(line,m,1,i+2) * qx(line,1,ii+2)
                  + Lx(line,m,2,i+2) * qx(line,2,ii+2)
                  + Lx(line,m,3,i+2) * qx(line,3,ii+2)
                  + Lx(line,m,4,i+2) * qx(line,5,ii+2)
                  + Lx(line,m,5,i+2) * qx(line,6,ii+2)
                  + Lx(line,m,6,i+2) * qx(line,7,ii+2);
            }

            for (int ks = 0; ks < 5; ++ks) {
                dFsk[ks] = Fsk[ks+1] - Fsk[ks];
                dqsk[ks] = qsk[ks+1] - qsk[ks];
            }

            double first =
                (-Fsk[1] + 7.0*Fsk[2] + 7.0*Fsk[3] - Fsk[4]) / 12.0;

            double aterm = 0.5 * (dFsk[0] + amx*dqsk[0]);
            double bterm = 0.5 * (dFsk[1] + amx*dqsk[1]);
            double cterm = 0.5 * (dFsk[2] + amx*dqsk[2]);
            double dterm = 0.5 * (dFsk[3] + amx*dqsk[3]);

            double IS0 = 13.0*(aterm-bterm)*(aterm-bterm)
                       + 3.0*(aterm-3.0*bterm)*(aterm-3.0*bterm);
            double IS1 = 13.0*(bterm-cterm)*(bterm-cterm)
                       + 3.0*(bterm+cterm)*(bterm+cterm);
            double IS2 = 13.0*(cterm-dterm)*(cterm-dterm)
                       + 3.0*(3.0*cterm-dterm)*(3.0*cterm-dterm);

            double alpha0 = 1.0 / ((eps + IS0)*(eps + IS0));
            double alpha1 = 6.0 / ((eps + IS1)*(eps + IS1));
            double alpha2 = 3.0 / ((eps + IS2)*(eps + IS2));

            double omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
            double omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

            double second =
                omega0 * (aterm - 2.0*bterm + cterm) / 3.0
              + (omega2 - 0.5) * (bterm - 2.0*cterm + dterm) / 6.0;

            aterm = 0.5 * (dFsk[4] - amx*dqsk[4]);
            bterm = 0.5 * (dFsk[3] - amx*dqsk[3]);
            cterm = 0.5 * (dFsk[2] - amx*dqsk[2]);
            dterm = 0.5 * (dFsk[1] - amx*dqsk[1]);

            IS0 = 13.0*(aterm-bterm)*(aterm-bterm)
                + 3.0*(aterm-3.0*bterm)*(aterm-3.0*bterm);
            IS1 = 13.0*(bterm-cterm)*(bterm-cterm)
                + 3.0*(bterm+cterm)*(bterm+cterm);
            IS2 = 13.0*(cterm-dterm)*(cterm-dterm)
                + 3.0*(3.0*cterm-dterm)*(3.0*cterm-dterm);

            alpha0 = 1.0 / ((eps + IS0)*(eps + IS0));
            alpha1 = 6.0 / ((eps + IS1)*(eps + IS1));
            alpha2 = 3.0 / ((eps + IS2)*(eps + IS2));

            omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
            omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

            double third =
                omega0 * (aterm - 2.0*bterm + cterm) / 3.0
              + (omega2 - 0.5) * (bterm - 2.0*cterm + dterm) / 6.0;

            double Fs = first - second + third;

            // Transform characteristic flux back to physical variables.
            // Rx(line,row=physical variable, col=characteristic wave, interface).
            for (int r = 0; r < 7; ++r) {
                Kokkos::atomic_add(&dF(line,r,i), Rx(line,r,m,i+2) * Fs);
            }
        }
    );

    Kokkos::fence();

    Kokkos::parallel_for(
        "weno_x_all_update",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 1},
            {nlines, nn + 1}
        ),
        KOKKOS_LAMBDA(const int line, const int i) {
            qx(line,0,i+2) = -dtdl * (dF(line,0,i) - dF(line,0,i-1));
            qx(line,1,i+2) = -dtdl * (dF(line,1,i) - dF(line,1,i-1));
            qx(line,2,i+2) = -dtdl * (dF(line,2,i) - dF(line,2,i-1));
            qx(line,3,i+2) = -dtdl * (dF(line,3,i) - dF(line,3,i-1));
            qx(line,4,i+2) = 0.0;
            qx(line,5,i+2) = -dtdl * (dF(line,4,i) - dF(line,4,i-1));
            qx(line,6,i+2) = -dtdl * (dF(line,5,i) - dF(line,5,i-1));
            qx(line,7,i+2) = -dtdl * (dF(line,6,i) - dF(line,6,i-1));
        }
    );

    Kokkos::parallel_for(
        "weno_x_all_fluxct_terms",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {nlines, nn + 1}
        ),
        KOKKOS_LAMBDA(const int line, const int i) {
            const int im1 = i - 1 + 2;
            const int i0  = i     + 2;
            const int ip1 = i + 1 + 2;
            const int ip2 = i + 2 + 2;

            const double ftmp1 =
                (-0.0 * (ux(line,4,im1) * ux(line,2,im1))
                + 9.0 * (ux(line,4,i0 ) * ux(line,2,i0 ))
                + 9.0 * (ux(line,4,ip1) * ux(line,2,ip1))
                - 0.0 * (ux(line,4,ip2) * ux(line,2,ip2))) / 18.0;

            const double ftmp2 =
                (-0.0 * (ux(line,4,im1) * ux(line,3,im1))
                + 9.0 * (ux(line,4,i0 ) * ux(line,3,i0 ))
                + 9.0 * (ux(line,4,ip1) * ux(line,3,ip1))
                - 0.0 * (ux(line,4,ip2) * ux(line,3,ip2))) / 18.0;

            bsyx(line,i) = dF(line,4,i) + ftmp1;
            bszx(line,i) = dF(line,5,i) + ftmp2;
        }
    );

    Kokkos::fence();
}


void x_sweep(Kokkos::View<double*****> q,
             int iw,
             const RKCoeff c,
             const Parameters& par,
             CTFluxes& ct) {

    const int nn = par.nx;
    const int nlines = (par.ny + 6) * (par.nz + 6);

    // Workspaces are line-major: (line, variable, i)
    Kokkos::View<double***>  qx("qx_all", nlines, 8, nn + 6);
    Kokkos::View<double***>  ux("ux_all", nlines, 9, nn + 6);
    Kokkos::View<double***>  ax("ax_all", nlines, 7, nn + 6);
    Kokkos::View<double***>  Fx("Fx_all", nlines, 7, nn + 6);
    Kokkos::View<double****> Rx("Rx_all", nlines, 7, 7, nn + 6);
    Kokkos::View<double****> Lx("Lx_all", nlines, 7, 7, nn + 6);
    Kokkos::View<double**>   bsyx("bsy_x_all", nlines, nn + 1);
    Kokkos::View<double**>   bszx("bsz_x_all", nlines, nn + 1);

    load_x_all_lines(q, qx, iw - 1, par);
    primit_x_all(qx, ux, nlines, nn, par);
    eigenst_x_all(qx, ux, ax, Fx, Rx, Lx, nlines, nn, par);
    weno_x_all(qx, ux, ax, Fx, Rx, Lx, bsyx, bszx, nlines, nn, par.dtdx, par);
    store_ct_x_all(ct, bsyx, bszx, par);
    store_x_all_lines(q, qx, iw, c.k1, c.k3, par);
}


// ================================================================
// Y/Z sweep all-line workspace versions.
//
// These reuse the same line-oriented primitive/eigen/WENO kernels as
// x_sweep after applying the same variable permutations used by the
// original line-by-line y_sweep and z_sweep.
// ================================================================
void load_y_all_lines(Kokkos::View<double*****> q,
                      Kokkos::View<double***> qy,
                      int stage,
                      const Parameters& par) {
    const int nxp = par.nx + 6;
    const int nyp = par.ny + 6;
    const int nzp = par.nz + 6;
    const int nlines = nxp * nzp;

    Kokkos::parallel_for(
        "load_y_all_lines",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {nlines, NVAR, nyp}
        ),
        KOKKOS_LAMBDA(const int line, const int m, const int jj) {
            const int ii = line % nxp;
            const int kk = line / nxp;

            if      (m == 0) qy(line,m,jj) = q(stage,0,ii,jj,kk);
            else if (m == 1) qy(line,m,jj) = q(stage,2,ii,jj,kk);
            else if (m == 2) qy(line,m,jj) = q(stage,3,ii,jj,kk);
            else if (m == 3) qy(line,m,jj) = q(stage,1,ii,jj,kk);
            else if (m == 4) qy(line,m,jj) = q(stage,5,ii,jj,kk);
            else if (m == 5) qy(line,m,jj) = q(stage,6,ii,jj,kk);
            else if (m == 6) qy(line,m,jj) = q(stage,4,ii,jj,kk);
            else if (m == 7) qy(line,m,jj) = q(stage,7,ii,jj,kk);
        }
    );
    Kokkos::fence();
}

void store_y_all_lines(Kokkos::View<double*****> q,
                       Kokkos::View<double***> qy,
                       int iw,
                       const Parameters& par) {
    const int nxp = par.nx + 6;
    const int nzp = par.nz + 6;
    const int nlines = nxp * nzp;

    Kokkos::parallel_for(
        "store_y_all_lines",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 3},
            {nlines, NVAR, par.ny + 3}
        ),
        KOKKOS_LAMBDA(const int line, const int m, const int jj) {
            const int ii = line % nxp;
            const int kk = line / nxp;

            if      (m == 0) q(iw,0,ii,jj,kk) += qy(line,m,jj);
            else if (m == 1) q(iw,2,ii,jj,kk) += qy(line,m,jj);
            else if (m == 2) q(iw,3,ii,jj,kk) += qy(line,m,jj);
            else if (m == 3) q(iw,1,ii,jj,kk) += qy(line,m,jj);
            else if (m == 4) q(iw,5,ii,jj,kk) += qy(line,m,jj);
            else if (m == 5) q(iw,6,ii,jj,kk) += qy(line,m,jj);
            else if (m == 6) q(iw,4,ii,jj,kk) += qy(line,m,jj);
            else if (m == 7) q(iw,7,ii,jj,kk) += qy(line,m,jj);
        }
    );
    Kokkos::fence();
}

void store_ct_y_all(CTFluxes& ct,
                    Kokkos::View<double**> bsyy,
                    Kokkos::View<double**> bszy,
                    const Parameters& par) {
    const int nxp = par.nx + 6;
    const int nzp = par.nz + 6;
    const int nlines = nxp * nzp;

    Kokkos::parallel_for(
        "store_ct_y_all",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {nlines, par.ny + 1}
        ),
        KOKKOS_LAMBDA(const int line, const int j) {
            const int ii = line % nxp;
            const int kk = line / nxp;
            const int jj = j + 2;

            // y-sweep variable permutation:
            // qline transverse magnetic fields are Bz and Bx.
            ct.gsz(ii,jj,kk) = bsyy(line,j);
            ct.gsx(ii,jj,kk) = bszy(line,j);
        }
    );
    Kokkos::fence();
}

void load_z_all_lines(Kokkos::View<double*****> q,
                      Kokkos::View<double***> qz,
                      int stage,
                      const Parameters& par) {
    const int nxp = par.nx + 6;
    const int nyp = par.ny + 6;
    const int nzp = par.nz + 6;
    const int nlines = nxp * nyp;

    Kokkos::parallel_for(
        "load_z_all_lines",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {nlines, NVAR, nzp}
        ),
        KOKKOS_LAMBDA(const int line, const int m, const int kk) {
            const int ii = line % nxp;
            const int jj = line / nxp;

            if      (m == 0) qz(line,m,kk) = q(stage,0,ii,jj,kk);
            else if (m == 1) qz(line,m,kk) = q(stage,3,ii,jj,kk);
            else if (m == 2) qz(line,m,kk) = q(stage,1,ii,jj,kk);
            else if (m == 3) qz(line,m,kk) = q(stage,2,ii,jj,kk);
            else if (m == 4) qz(line,m,kk) = q(stage,6,ii,jj,kk);
            else if (m == 5) qz(line,m,kk) = q(stage,4,ii,jj,kk);
            else if (m == 6) qz(line,m,kk) = q(stage,5,ii,jj,kk);
            else if (m == 7) qz(line,m,kk) = q(stage,7,ii,jj,kk);
        }
    );
    Kokkos::fence();
}

void store_z_all_lines(Kokkos::View<double*****> q,
                       Kokkos::View<double***> qz,
                       int iw,
                       const Parameters& par) {
    const int nxp = par.nx + 6;
    const int nyp = par.ny + 6;
    const int nlines = nxp * nyp;

    Kokkos::parallel_for(
        "store_z_all_lines",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 3},
            {nlines, NVAR, par.nz + 3}
        ),
        KOKKOS_LAMBDA(const int line, const int m, const int kk) {
            const int ii = line % nxp;
            const int jj = line / nxp;

            if      (m == 0) q(iw,0,ii,jj,kk) += qz(line,m,kk);
            else if (m == 1) q(iw,3,ii,jj,kk) += qz(line,m,kk);
            else if (m == 2) q(iw,1,ii,jj,kk) += qz(line,m,kk);
            else if (m == 3) q(iw,2,ii,jj,kk) += qz(line,m,kk);
            else if (m == 4) q(iw,6,ii,jj,kk) += qz(line,m,kk);
            else if (m == 5) q(iw,4,ii,jj,kk) += qz(line,m,kk);
            else if (m == 6) q(iw,5,ii,jj,kk) += qz(line,m,kk);
            else if (m == 7) q(iw,7,ii,jj,kk) += qz(line,m,kk);
        }
    );
    Kokkos::fence();
}

void store_ct_z_all(CTFluxes& ct,
                    Kokkos::View<double**> bsyz,
                    Kokkos::View<double**> bszz,
                    const Parameters& par) {
    const int nxp = par.nx + 6;
    const int nyp = par.ny + 6;
    const int nlines = nxp * nyp;

    Kokkos::parallel_for(
        "store_ct_z_all",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {nlines, par.nz + 1}
        ),
        KOKKOS_LAMBDA(const int line, const int k) {
            const int ii = line % nxp;
            const int jj = line / nxp;
            const int kk = k + 2;

            // z-sweep variable permutation:
            // qline transverse magnetic fields are Bx and By.
            ct.hsx(ii,jj,kk) = bsyz(line,k);
            ct.hsy(ii,jj,kk) = bszz(line,k);
        }
    );
    Kokkos::fence();
}


void y_sweep(Kokkos::View<double*****> q,
             int iw,
             const Parameters& par,
             CTFluxes& ct) {

    const int nn = par.ny;
    const int nlines = (par.nx + 6) * (par.nz + 6);

    // Workspaces are line-major: (line, variable, j)
    // The variable order follows the original y-sweep permutation:
    // rho, My, Mz, Mx, By, Bz, Bx, E.
    Kokkos::View<double***>  qy("qy_all", nlines, 8, nn + 6);
    Kokkos::View<double***>  uy("uy_all", nlines, 9, nn + 6);
    Kokkos::View<double***>  ay("ay_all", nlines, 7, nn + 6);
    Kokkos::View<double***>  Fy("Fy_all", nlines, 7, nn + 6);
    Kokkos::View<double****> Ry("Ry_all", nlines, 7, 7, nn + 6);
    Kokkos::View<double****> Ly("Ly_all", nlines, 7, 7, nn + 6);
    Kokkos::View<double**>   bsyy("bsy_y_all", nlines, nn + 1);
    Kokkos::View<double**>   bszy("bsz_y_all", nlines, nn + 1);

    load_y_all_lines(q, qy, iw - 1, par);
    primit_x_all(qy, uy, nlines, nn, par);
    eigenst_x_all(qy, uy, ay, Fy, Ry, Ly, nlines, nn, par);
    weno_x_all(qy, uy, ay, Fy, Ry, Ly, bsyy, bszy, nlines, nn, par.dtdy, par);
    store_ct_y_all(ct, bsyy, bszy, par);
    store_y_all_lines(q, qy, iw, par);
}

void z_sweep(Kokkos::View<double*****> q,
             int iw,
             const Parameters& par,
             CTFluxes& ct) {

    const int nn = par.nz;
    const int nlines = (par.nx + 6) * (par.ny + 6);

    // Workspaces are line-major: (line, variable, k)
    // The variable order follows the original z-sweep permutation:
    // rho, Mz, Mx, My, Bz, Bx, By, E.
    Kokkos::View<double***>  qz("qz_all", nlines, 8, nn + 6);
    Kokkos::View<double***>  uz("uz_all", nlines, 9, nn + 6);
    Kokkos::View<double***>  az("az_all", nlines, 7, nn + 6);
    Kokkos::View<double***>  Fz("Fz_all", nlines, 7, nn + 6);
    Kokkos::View<double****> Rz("Rz_all", nlines, 7, 7, nn + 6);
    Kokkos::View<double****> Lz("Lz_all", nlines, 7, 7, nn + 6);
    Kokkos::View<double**>   bsyz("bsy_z_all", nlines, nn + 1);
    Kokkos::View<double**>   bszz("bsz_z_all", nlines, nn + 1);

    load_z_all_lines(q, qz, iw - 1, par);
    primit_x_all(qz, uz, nlines, nn, par);
    eigenst_x_all(qz, uz, az, Fz, Rz, Lz, nlines, nn, par);
    weno_x_all(qz, uz, az, Fz, Rz, Lz, bsyz, bszz, nlines, nn, par.dtdz, par);
    store_ct_z_all(ct, bsyz, bszz, par);
    store_z_all_lines(q, qz, iw, par);
}

// ================================================================
// Main SSPRK driver
// ================================================================
void ssprk(Kokkos::View<double*****> q, Parameters& par) {

    CTFluxes ct(par);

    for (int iw = 1; iw <= 5; ++iw) {

        zero_ct_fluxes(ct);

        RKCoeff c = get_rk_coeff(iw);

        if (iw >= 1 && iw <= 4) {
            copy_stage(q, iw, 0, par);
            copy_bfield_stage(iw, 0, par);
        } else if (iw == 5) {
            prepare_stage5(q, c, par);
            prepare_bfield_stage5(c.k1, c.k4, c.k5, par);
        }

        // Fill ghost zones for the source stage before loading 1D lines.
        // WENO uses ghost-cell stencils.
        bound(q, iw - 1, par);
        if (iw == 5) {
            bound(q, iw, par);
        }

        par.dtdx = c.k2 * par.dt / par.dx;
        par.dtdy = c.k2 * par.dt / par.dy;
        par.dtdz = c.k2 * par.dt / par.dz;

        if (par.nx > 1) {
            x_sweep(q, iw, c, par, ct);
        }

        // Do not run WENO along degenerate one-cell directions.
        if (par.ny > 1) {
            y_sweep(q, iw, par, ct);
        }

        if (par.nz > 1) {
            z_sweep(q, iw, par, ct);
        }

        bound_bfield(iw - 1, par);
        fluxct(q, iw, c.k1, c.k3, par, ct);

        prot(q, iw, par);
        bound(q, iw, par);
        bound_bfield(iw, par);
    }

    copy_stage(q, 0, 5, par);
    copy_bfield_to_stage0(par);
}


static constexpr int NG = 2;

void primit(Kokkos::View<double**> qone,
            Kokkos::View<double**> uone,
            int nn,
            const Parameters& par) {

    Kokkos::parallel_for(
        "primit",
        Kokkos::RangePolicy<>(0, nn + 6),
        KOKKOS_LAMBDA(const int ii) {

            const double DD = qone(0,ii);
            const double Mx = qone(1,ii);
            const double My = qone(2,ii);
            const double Mz = qone(3,ii);
            const double Bx = qone(4,ii);
            const double By = qone(5,ii);
            const double Bz = qone(6,ii);
            double EE       = qone(7,ii);

            double rho = DD;

            const double vx = Mx / rho;
            const double vy = My / rho;
            const double vz = Mz / rho;

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;

            double pg = (par.gam - 1.0)
                      * (EE - 0.5 * (rho*vv2 + BB2));

            if (rho < par.rhomin || pg < par.pgmin) {
                rho = fmax(par.rhomin, rho);
                pg  = fmax(par.pgmin, pg);

                EE = pg / (par.gam - 1.0)
                   + 0.5 * (rho*vv2 + BB2);

                qone(0,ii) = rho;
                qone(7,ii) = EE;
            }

            const double HH = (EE + pg) / rho;

            uone(0,ii) = rho;
            uone(1,ii) = vx;
            uone(2,ii) = vy;
            uone(3,ii) = vz;
            uone(4,ii) = Bx;
            uone(5,ii) = By;
            uone(6,ii) = Bz;
            uone(7,ii) = pg;
            uone(8,ii) = HH;
        }
    );

    Kokkos::fence();
}




void eigenst(Kokkos::View<double**> qone,
             Kokkos::View<double**> uone,
             Kokkos::View<double**> a,
             Kokkos::View<double**> F,
             Kokkos::View<double***> R,
             Kokkos::View<double***> L,
             int nn,
             const Parameters& par) {

    const double gam0 = 1.0 - par.gam;
    const double gam1 = 0.5 * (par.gam - 1.0);
    const double gam2 = (par.gam - 2.0) / (par.gam - 1.0);

    // ------------------------------------------------------------
    // cell-center values: eigenvalues a and flux F
    // Fortran i = -2 ... nn+3
    // C++ ii = i + 2 = 0 ... nn+5
    // ------------------------------------------------------------
    Kokkos::parallel_for(
        "eigenst_center",
        Kokkos::RangePolicy<>(0, nn + 6),
        KOKKOS_LAMBDA(const int ii) {

            const double DD = qone(0,ii);
            const double Mx = qone(1,ii);
            const double My = qone(2,ii);
            const double Mz = qone(3,ii);
            const double EE = qone(7,ii);

            const double rho = uone(0,ii);
            const double vx  = uone(1,ii);
            const double vy  = uone(2,ii);
            const double vz  = uone(3,ii);
            const double Bx  = uone(4,ii);
            const double By  = uone(5,ii);
            const double Bz  = uone(6,ii);
            const double pg  = uone(7,ii);

            const double vv2 = vx*vx + vy*vy + vz*vz;
            const double BB2 = Bx*Bx + By*By + Bz*Bz;
            const double pt  = pg + 0.5 * BB2;
            const double vdB = vx*Bx + vy*By + vz*Bz;

            const double bbn2 = BB2 / rho;
            const double bnx2 = Bx*Bx / rho;

            const double cs2 = fmax(0.0, par.gam * fabs(pg / rho));

            const double root =
                sqrt(fmax(0.0, (bbn2 + cs2)*(bbn2 + cs2)
                         - 4.0 * bnx2 * cs2));

            const double lf = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 + root)));
            const double la = sqrt(fmax(0.0, bnx2));
            const double ls = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 - root)));

            a(0,ii) = vx - lf;
            a(1,ii) = vx - la;
            a(2,ii) = vx - ls;
            a(3,ii) = vx;
            a(4,ii) = vx + ls;
            a(5,ii) = vx + la;
            a(6,ii) = vx + lf;

            F(0,ii) = DD * vx;
            F(1,ii) = Mx * vx + pt - Bx*Bx;
            F(2,ii) = My * vx - Bx*By;
            F(3,ii) = Mz * vx - Bx*Bz;
            F(4,ii) = By * vx - Bx*vy;
            F(5,ii) = Bz * vx - Bx*vz;
            F(6,ii) = (EE + pt) * vx - Bx * vdB;
        }
    );

    // ------------------------------------------------------------
    // cell-boundary values: left/right eigenvectors
    // Fortran i = -1 ... nn+1
    // C++ ii = i + 2 = 1 ... nn+3
    // ------------------------------------------------------------
    Kokkos::parallel_for(
        "eigenst_boundary",
        Kokkos::RangePolicy<>(1, nn + 4),
        KOKKOS_LAMBDA(const int ii) {

            const double rho = 0.5 * (uone(0,ii) + uone(0,ii+1));
            const double vx  = 0.5 * (uone(1,ii) + uone(1,ii+1));
            const double vy  = 0.5 * (uone(2,ii) + uone(2,ii+1));
            const double vz  = 0.5 * (uone(3,ii) + uone(3,ii+1));
            const double Bx  = 0.5 * (uone(4,ii) + uone(4,ii+1));
            const double By  = 0.5 * (uone(5,ii) + uone(5,ii+1));
            const double Bz  = 0.5 * (uone(6,ii) + uone(6,ii+1));
            const double pg  = 0.5 * (uone(7,ii) + uone(7,ii+1));
            const double HH  = 0.5 * (uone(8,ii) + uone(8,ii+1));

            const double vv2  = vx*vx + vy*vy + vz*vz;
            const double BB2  = Bx*Bx + By*By + Bz*Bz;
            const double bbn2 = BB2 / rho;
            const double bnx2 = Bx*Bx / rho;

            const double cs2 = (par.gam - 1.0)
                             * (HH - 0.5 * (vv2 + bbn2));
            const double cs  = sqrt(fmax(0.0, cs2));

            const double root =
                sqrt(fmax(0.0, (bbn2 + cs2)*(bbn2 + cs2)
                         - 4.0 * bnx2 * cs2));

            const double lf = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 + root)));
            const double la = sqrt(fmax(0.0, bnx2));
            const double ls = sqrt(fmax(0.0, 0.5 * (bbn2 + cs2 - root)));

            const double Bt2 = By*By + Bz*Bz;
            const double sgnBx = sign1(Bx);

            double bty, btz;
            if (Bt2 >= 1.0e-30) {
                bty = By / sqrt(Bt2);
                btz = Bz / sqrt(Bt2);
            } else {
                bty = 1.0 / sqrt(2.0);
                btz = 1.0 / sqrt(2.0);
            }

            double af, as;
            if ((lf*lf - ls*ls) >= 1.0e-30) {
                af = sqrt(fmax(0.0, cs2 - ls*ls)) / sqrt(lf*lf - ls*ls);
                as = sqrt(fmax(0.0, lf*lf - cs2)) / sqrt(lf*lf - ls*ls);
            } else {
                af = 1.0;
                as = 1.0;
            }

            const double sqrtrho = sqrt(rho);
            const double invsqrtrho = 1.0 / sqrtrho;

            // ---------------- left eigenvectors L(row,col,ii)
            L(0,0,ii) =  af*(gam1*vv2 + lf*vx) - as*ls*(bty*vy + btz*vz)*sgnBx;
            L(0,1,ii) =  af*(gam0*vx - lf);
            L(0,2,ii) =  gam0*af*vy + as*ls*bty*sgnBx;
            L(0,3,ii) =  gam0*af*vz + as*ls*btz*sgnBx;
            L(0,4,ii) =  gam0*af*By + cs*as*bty*sqrtrho;
            L(0,5,ii) =  gam0*af*Bz + cs*as*btz*sqrtrho;
            L(0,6,ii) = -gam0*af;

            L(1,0,ii) =  btz*vy - bty*vz;
            L(1,1,ii) =  0.0;
            L(1,2,ii) = -btz;
            L(1,3,ii) =  bty;
            L(1,4,ii) = -btz*sgnBx*sqrtrho;
            L(1,5,ii) =  bty*sgnBx*sqrtrho;
            L(1,6,ii) =  0.0;

            L(2,0,ii) =  as*(gam1*vv2 + ls*vx) + af*lf*(bty*vy + btz*vz)*sgnBx;
            L(2,1,ii) =  gam0*as*vx - as*ls;
            L(2,2,ii) =  gam0*as*vy - af*lf*bty*sgnBx;
            L(2,3,ii) =  gam0*as*vz - af*lf*btz*sgnBx;
            L(2,4,ii) =  gam0*as*By - cs*af*bty*sqrtrho;
            L(2,5,ii) =  gam0*as*Bz - cs*af*btz*sqrtrho;
            L(2,6,ii) = -gam0*as;

            L(3,0,ii) = -cs2/gam0 - 0.5*vv2;
            L(3,1,ii) =  vx;
            L(3,2,ii) =  vy;
            L(3,3,ii) =  vz;
            L(3,4,ii) =  By;
            L(3,5,ii) =  Bz;
            L(3,6,ii) = -1.0;

            L(4,0,ii) =  as*(gam1*vv2 - ls*vx) - af*lf*(bty*vy + btz*vz)*sgnBx;
            L(4,1,ii) =  as*(gam0*vx + ls);
            L(4,2,ii) =  gam0*as*vy + af*lf*bty*sgnBx;
            L(4,3,ii) =  gam0*as*vz + af*lf*btz*sgnBx;
            L(4,4,ii) =  gam0*as*By - cs*af*bty*sqrtrho;
            L(4,5,ii) =  gam0*as*Bz - cs*af*btz*sqrtrho;
            L(4,6,ii) = -gam0*as;

            L(5,0,ii) =  btz*vy - bty*vz;
            L(5,1,ii) =  0.0;
            L(5,2,ii) = -btz;
            L(5,3,ii) =  bty;
            L(5,4,ii) =  btz*sgnBx*sqrtrho;
            L(5,5,ii) = -bty*sgnBx*sqrtrho;
            L(5,6,ii) =  0.0;

            L(6,0,ii) =  af*(gam1*vv2 - lf*vx) + as*ls*(bty*vy + btz*vz)*sgnBx;
            L(6,1,ii) =  af*(gam0*vx + lf);
            L(6,2,ii) =  gam0*af*vy - as*ls*bty*sgnBx;
            L(6,3,ii) =  gam0*af*vz - as*ls*btz*sgnBx;
            L(6,4,ii) =  gam0*af*By + cs*as*bty*sqrtrho;
            L(6,5,ii) =  gam0*af*Bz + cs*as*btz*sqrtrho;
            L(6,6,ii) = -gam0*af;

            for (int m = 0; m < 7; ++m) {
                L(0,m,ii) = 0.5 * L(0,m,ii) / cs2;
                L(1,m,ii) = 0.5 * L(1,m,ii);
                L(2,m,ii) = 0.5 * L(2,m,ii) / cs2;
                L(3,m,ii) = -gam0 * L(3,m,ii) / cs2;
                L(4,m,ii) = 0.5 * L(4,m,ii) / cs2;
                L(5,m,ii) = 0.5 * L(5,m,ii);
                L(6,m,ii) = 0.5 * L(6,m,ii) / cs2;
            }

            // ---------------- right eigenvectors R(row,col,ii)
            R(0,0,ii) = af;
            R(1,0,ii) = af*(vx-lf);
            R(2,0,ii) = af*vy + as*ls*bty*sgnBx;
            R(3,0,ii) = af*vz + as*ls*btz*sgnBx;
            R(4,0,ii) = cs*as*bty*invsqrtrho;
            R(5,0,ii) = cs*as*btz*invsqrtrho;
            R(6,0,ii) = af*(lf*lf - lf*vx + 0.5*vv2 - gam2*cs2)
                       + as*ls*(bty*vy + btz*vz)*sgnBx;

            R(0,1,ii) = 0.0;
            R(1,1,ii) = 0.0;
            R(2,1,ii) = -btz;
            R(3,1,ii) = bty;
            R(4,1,ii) = -btz*sgnBx*invsqrtrho;
            R(5,1,ii) = bty*sgnBx*invsqrtrho;
            R(6,1,ii) = bty*vz - btz*vy;

            R(0,2,ii) = as;
            R(1,2,ii) = as*(vx-ls);
            R(2,2,ii) = as*vy - af*lf*bty*sgnBx;
            R(3,2,ii) = as*vz - af*lf*btz*sgnBx;
            R(4,2,ii) = -cs*af*bty*invsqrtrho;
            R(5,2,ii) = -cs*af*btz*invsqrtrho;
            R(6,2,ii) = as*(ls*ls - ls*vx + 0.5*vv2 - gam2*cs2)
                       - af*lf*(bty*vy + btz*vz)*sgnBx;

            R(0,3,ii) = 1.0;
            R(1,3,ii) = vx;
            R(2,3,ii) = vy;
            R(3,3,ii) = vz;
            R(4,3,ii) = 0.0;
            R(5,3,ii) = 0.0;
            R(6,3,ii) = 0.5*vv2;

            R(0,4,ii) = as;
            R(1,4,ii) = as*(vx+ls);
            R(2,4,ii) = as*vy + af*lf*bty*sgnBx;
            R(3,4,ii) = as*vz + af*lf*btz*sgnBx;
            R(4,4,ii) = -cs*af*bty*invsqrtrho;
            R(5,4,ii) = -cs*af*btz*invsqrtrho;
            R(6,4,ii) = as*(ls*ls + ls*vx + 0.5*vv2 - gam2*cs2)
                       + af*lf*(bty*vy + btz*vz)*sgnBx;

            R(0,5,ii) = 0.0;
            R(1,5,ii) = 0.0;
            R(2,5,ii) = -btz;
            R(3,5,ii) = bty;
            R(4,5,ii) = btz*sgnBx*invsqrtrho;
            R(5,5,ii) = -bty*sgnBx*invsqrtrho;
            R(6,5,ii) = bty*vz - btz*vy;

            R(0,6,ii) = af;
            R(1,6,ii) = af*(vx+lf);
            R(2,6,ii) = af*vy - as*ls*bty*sgnBx;
            R(3,6,ii) = af*vz - as*ls*btz*sgnBx;
            R(4,6,ii) = cs*as*bty*invsqrtrho;
            R(5,6,ii) = cs*as*btz*invsqrtrho;
            R(6,6,ii) = af*(lf*lf + lf*vx + 0.5*vv2 - gam2*cs2)
                       - as*ls*(bty*vy + btz*vz)*sgnBx;

            double sgnBt;
            if (By != 0.0) {
                sgnBt = sign1(By);
            } else {
                sgnBt = sign1(Bz);
            }

            if (cs >= la) {
                for (int m = 0; m < 7; ++m) {
                    L(2,m,ii) *= sgnBt;
                    L(4,m,ii) *= sgnBt;
                }
            } else {
                for (int m = 0; m < 7; ++m) {
                    L(0,m,ii) *= sgnBt;
                    L(6,m,ii) *= sgnBt;
                }
            }

            if (cs >= la) {
                for (int m = 0; m < 7; ++m) {
                    R(m,2,ii) *= sgnBt;
                    R(m,4,ii) *= sgnBt;
                }
            } else {
                for (int m = 0; m < 7; ++m) {
                    R(m,0,ii) *= sgnBt;
                    R(m,6,ii) *= sgnBt;
                }
            }
        }
    );

    Kokkos::fence();
}



void weno(Kokkos::View<double**> qone,
          Kokkos::View<double**> uone,
          Kokkos::View<double**> a,
          Kokkos::View<double**> F,
          Kokkos::View<double***> R,
          Kokkos::View<double***> L,
          Kokkos::View<double*> bsy,
          Kokkos::View<double*> bsz,
          int nn,
          double dtdl,
          const Parameters& par) {

    Kokkos::View<double**> dF("dF", 7, nn + 1);
    Kokkos::deep_copy(dF, 0.0);

    Kokkos::parallel_for(
        "weno_flux",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {nn + 1, 7}
        ),
        KOKKOS_LAMBDA(const int i, const int m) {

            const double eps = 1.0e-10;

            double amx = 0.0;
            for (int s = -2; s <= 3; ++s) {
                amx = fmax(amx, fabs(a(m, i + s + 2)));
            }

            double Fsk[6];
            double qsk[6];
            double dFsk[5];
            double dqsk[5];

            for (int ks = 0; ks < 6; ++ks) {
                const int ii = i - 2 + ks; 

                Fsk[ks] =
                    L(m,0,i+2) * F(0,ii+2)
                  + L(m,1,i+2) * F(1,ii+2)
                  + L(m,2,i+2) * F(2,ii+2)
                  + L(m,3,i+2) * F(3,ii+2)
                  + L(m,4,i+2) * F(4,ii+2)
                  + L(m,5,i+2) * F(5,ii+2)
                  + L(m,6,i+2) * F(6,ii+2);

                qsk[ks] =
                    L(m,0,i+2) * qone(0,ii+2)
                  + L(m,1,i+2) * qone(1,ii+2)
                  + L(m,2,i+2) * qone(2,ii+2)
                  + L(m,3,i+2) * qone(3,ii+2)
                  + L(m,4,i+2) * qone(5,ii+2)
                  + L(m,5,i+2) * qone(6,ii+2)
                  + L(m,6,i+2) * qone(7,ii+2);
            }

            for (int ks = 0; ks < 5; ++ks) {
                dFsk[ks] = Fsk[ks+1] - Fsk[ks];
                dqsk[ks] = qsk[ks+1] - qsk[ks];
            }

            double first =
                (-Fsk[1] + 7.0*Fsk[2] + 7.0*Fsk[3] - Fsk[4]) / 12.0;

            double aterm = 0.5 * (dFsk[0] + amx*dqsk[0]);
            double bterm = 0.5 * (dFsk[1] + amx*dqsk[1]);
            double cterm = 0.5 * (dFsk[2] + amx*dqsk[2]);
            double dterm = 0.5 * (dFsk[3] + amx*dqsk[3]);

            double IS0 = 13.0*(aterm-bterm)*(aterm-bterm)
                       + 3.0*(aterm-3.0*bterm)*(aterm-3.0*bterm);
            double IS1 = 13.0*(bterm-cterm)*(bterm-cterm)
                       + 3.0*(bterm+cterm)*(bterm+cterm);
            double IS2 = 13.0*(cterm-dterm)*(cterm-dterm)
                       + 3.0*(3.0*cterm-dterm)*(3.0*cterm-dterm);

            double alpha0 = 1.0 / ((eps + IS0)*(eps + IS0));
            double alpha1 = 6.0 / ((eps + IS1)*(eps + IS1));
            double alpha2 = 3.0 / ((eps + IS2)*(eps + IS2));

            double omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
            double omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

            double second =
                omega0 * (aterm - 2.0*bterm + cterm) / 3.0
              + (omega2 - 0.5) * (bterm - 2.0*cterm + dterm) / 6.0;

            aterm = 0.5 * (dFsk[4] - amx*dqsk[4]);
            bterm = 0.5 * (dFsk[3] - amx*dqsk[3]);
            cterm = 0.5 * (dFsk[2] - amx*dqsk[2]);
            dterm = 0.5 * (dFsk[1] - amx*dqsk[1]);

            IS0 = 13.0*(aterm-bterm)*(aterm-bterm)
                + 3.0*(aterm-3.0*bterm)*(aterm-3.0*bterm);
            IS1 = 13.0*(bterm-cterm)*(bterm-cterm)
                + 3.0*(bterm+cterm)*(bterm+cterm);
            IS2 = 13.0*(cterm-dterm)*(cterm-dterm)
                + 3.0*(3.0*cterm-dterm)*(3.0*cterm-dterm);

            alpha0 = 1.0 / ((eps + IS0)*(eps + IS0));
            alpha1 = 6.0 / ((eps + IS1)*(eps + IS1));
            alpha2 = 3.0 / ((eps + IS2)*(eps + IS2));

            omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
            omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

            double third =
                omega0 * (aterm - 2.0*bterm + cterm) / 3.0
              + (omega2 - 0.5) * (bterm - 2.0*cterm + dterm) / 6.0;

            double Fs = first - second + third;

            // Transform characteristic flux back to physical variables.
            // R(row=physical variable, col=characteristic wave, interface).
            for (int r = 0; r < 7; ++r) {
                Kokkos::atomic_add(&dF(r,i), R(r,m,i+2) * Fs);
            }
        }
    );

    Kokkos::fence();

    Kokkos::parallel_for(
        "weno_update",
        Kokkos::RangePolicy<>(1, nn + 1),
        KOKKOS_LAMBDA(const int i) {
            qone(0,i+2) = -dtdl * (dF(0,i) - dF(0,i-1));
            qone(1,i+2) = -dtdl * (dF(1,i) - dF(1,i-1));
            qone(2,i+2) = -dtdl * (dF(2,i) - dF(2,i-1));
            qone(3,i+2) = -dtdl * (dF(3,i) - dF(3,i-1));
            qone(4,i+2) = 0.0;
            qone(5,i+2) = -dtdl * (dF(4,i) - dF(4,i-1));
            qone(6,i+2) = -dtdl * (dF(5,i) - dF(5,i-1));
            qone(7,i+2) = -dtdl * (dF(6,i) - dF(6,i-1));
        }
    );

    Kokkos::parallel_for(
        "weno_fluxct_terms",
        Kokkos::RangePolicy<>(0, nn + 1),
        KOKKOS_LAMBDA(const int i) {
            const int im1 = i - 1 + 2;
            const int i0  = i     + 2;
            const int ip1 = i + 1 + 2;
            const int ip2 = i + 2 + 2;

            const double ftmp1 =
                (-0.0 * (uone(4,im1) * uone(2,im1))
                + 9.0 * (uone(4,i0 ) * uone(2,i0 ))
                + 9.0 * (uone(4,ip1) * uone(2,ip1))
                - 0.0 * (uone(4,ip2) * uone(2,ip2))) / 18.0;

            const double ftmp2 =
                (-0.0 * (uone(4,im1) * uone(3,im1))
                + 9.0 * (uone(4,i0 ) * uone(3,i0 ))
                + 9.0 * (uone(4,ip1) * uone(3,ip1))
                - 0.0 * (uone(4,ip2) * uone(3,ip2))) / 18.0;

            bsy(i) = dF(4,i) + ftmp1;
            bsz(i) = dF(5,i) + ftmp2;
        }
    );

    Kokkos::fence();
}


