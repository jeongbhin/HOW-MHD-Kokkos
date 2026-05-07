#include "bound.hpp"

#include <iostream>
#include <cstdlib>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

static constexpr int NVAR = 8;
static constexpr int NG   = 3;

// ================================================================
// Local physical/serial boundary fills.
// These are also used in MPI mode only on true physical boundaries
// where MPI_Cart_shift returns MPI_PROC_NULL.
// ================================================================
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

#ifdef USE_MPI

// ----------------------------------------------------------------
// Indexing convention for flattened halo buffers.
// ----------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int idx_xbuf(const int m, const int g, const int j, const int k,
             const int ny6, const int nz6) {
    return (((m*NG + g)*ny6 + j)*nz6 + k);
}

KOKKOS_INLINE_FUNCTION
int idx_ybuf(const int m, const int g, const int i, const int k,
             const int nx6, const int nz6) {
    return (((m*NG + g)*nx6 + i)*nz6 + k);
}

KOKKOS_INLINE_FUNCTION
int idx_zbuf(const int m, const int g, const int i, const int j,
             const int nx6, const int ny6) {
    return (((m*NG + g)*nx6 + i)*ny6 + j);
}

void exchange_q_x(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int nx  = par.nx;
    const int ny6 = par.ny + 6;
    const int nz6 = par.nz + 6;
    const int count = NVAR * NG * ny6 * nz6;

    Kokkos::View<double*> send_xm_d("send_q_xm", count);
    Kokkos::View<double*> send_xp_d("send_q_xp", count);
    Kokkos::View<double*> recv_xm_d("recv_q_xm", count);
    Kokkos::View<double*> recv_xp_d("recv_q_xp", count);

    Kokkos::parallel_for("pack_q_x",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {NVAR,NG,ny6,nz6}),
        KOKKOS_LAMBDA(const int m, const int g, const int j, const int k) {
            const int p = idx_xbuf(m,g,j,k,ny6,nz6);
            // minus-side send: left active cells, from boundary inward
            send_xm_d(p) = q(stage,m,3 + g,j,k);
            // plus-side send: right active cells, far ghost ordering compatible
            // with q(0:2) <- q(nx:nx+2) used by the local periodic fill.
            send_xp_d(p) = q(stage,m,nx + g,j,k);
        });
    Kokkos::fence();

    auto send_xm_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_xm_d);
    auto send_xp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_xp_d);
    auto recv_xm_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), recv_xm_d);
    auto recv_xp_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), recv_xp_d);

    MPI_Sendrecv(send_xm_h.data(), count, MPI_DOUBLE, par.nbr_xm, 100,
                 recv_xp_h.data(), count, MPI_DOUBLE, par.nbr_xp, 100,
                 par.comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_xp_h.data(), count, MPI_DOUBLE, par.nbr_xp, 101,
                 recv_xm_h.data(), count, MPI_DOUBLE, par.nbr_xm, 101,
                 par.comm_cart, MPI_STATUS_IGNORE);

    if (par.nbr_xm != MPI_PROC_NULL) Kokkos::deep_copy(recv_xm_d, recv_xm_h);
    if (par.nbr_xp != MPI_PROC_NULL) Kokkos::deep_copy(recv_xp_d, recv_xp_h);

    Kokkos::parallel_for("unpack_q_x",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {NVAR,NG,ny6,nz6}),
        KOKKOS_LAMBDA(const int m, const int g, const int j, const int k) {
            const int p = idx_xbuf(m,g,j,k,ny6,nz6);
            if (par.nbr_xm != MPI_PROC_NULL) q(stage,m,g,     j,k) = recv_xm_d(p);
            if (par.nbr_xp != MPI_PROC_NULL) q(stage,m,nx+3+g,j,k) = recv_xp_d(p);
        });
    Kokkos::fence();
}

void exchange_q_y(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int ny  = par.ny;
    const int nx6 = par.nx + 6;
    const int nz6 = par.nz + 6;
    const int count = NVAR * NG * nx6 * nz6;

    Kokkos::View<double*> send_ym_d("send_q_ym", count);
    Kokkos::View<double*> send_yp_d("send_q_yp", count);
    Kokkos::View<double*> recv_ym_d("recv_q_ym", count);
    Kokkos::View<double*> recv_yp_d("recv_q_yp", count);

    Kokkos::parallel_for("pack_q_y",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {NVAR,NG,nx6,nz6}),
        KOKKOS_LAMBDA(const int m, const int g, const int i, const int k) {
            const int p = idx_ybuf(m,g,i,k,nx6,nz6);
            send_ym_d(p) = q(stage,m,i,3 + g,k);
            send_yp_d(p) = q(stage,m,i,ny + g,k);
        });
    Kokkos::fence();

    auto send_ym_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_ym_d);
    auto send_yp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_yp_d);
    auto recv_ym_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), recv_ym_d);
    auto recv_yp_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), recv_yp_d);

    MPI_Sendrecv(send_ym_h.data(), count, MPI_DOUBLE, par.nbr_ym, 200,
                 recv_yp_h.data(), count, MPI_DOUBLE, par.nbr_yp, 200,
                 par.comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_yp_h.data(), count, MPI_DOUBLE, par.nbr_yp, 201,
                 recv_ym_h.data(), count, MPI_DOUBLE, par.nbr_ym, 201,
                 par.comm_cart, MPI_STATUS_IGNORE);

    if (par.nbr_ym != MPI_PROC_NULL) Kokkos::deep_copy(recv_ym_d, recv_ym_h);
    if (par.nbr_yp != MPI_PROC_NULL) Kokkos::deep_copy(recv_yp_d, recv_yp_h);

    Kokkos::parallel_for("unpack_q_y",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {NVAR,NG,nx6,nz6}),
        KOKKOS_LAMBDA(const int m, const int g, const int i, const int k) {
            const int p = idx_ybuf(m,g,i,k,nx6,nz6);
            if (par.nbr_ym != MPI_PROC_NULL) q(stage,m,i,g,     k) = recv_ym_d(p);
            if (par.nbr_yp != MPI_PROC_NULL) q(stage,m,i,ny+3+g,k) = recv_yp_d(p);
        });
    Kokkos::fence();
}

void exchange_q_z(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    const int nz  = par.nz;
    const int nx6 = par.nx + 6;
    const int ny6 = par.ny + 6;
    const int count = NVAR * NG * nx6 * ny6;

    Kokkos::View<double*> send_zm_d("send_q_zm", count);
    Kokkos::View<double*> send_zp_d("send_q_zp", count);
    Kokkos::View<double*> recv_zm_d("recv_q_zm", count);
    Kokkos::View<double*> recv_zp_d("recv_q_zp", count);

    Kokkos::parallel_for("pack_q_z",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {NVAR,NG,nx6,ny6}),
        KOKKOS_LAMBDA(const int m, const int g, const int i, const int j) {
            const int p = idx_zbuf(m,g,i,j,nx6,ny6);
            send_zm_d(p) = q(stage,m,i,j,3 + g);
            send_zp_d(p) = q(stage,m,i,j,nz + g);
        });
    Kokkos::fence();

    auto send_zm_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_zm_d);
    auto send_zp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_zp_d);
    auto recv_zm_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), recv_zm_d);
    auto recv_zp_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), recv_zp_d);

    MPI_Sendrecv(send_zm_h.data(), count, MPI_DOUBLE, par.nbr_zm, 300,
                 recv_zp_h.data(), count, MPI_DOUBLE, par.nbr_zp, 300,
                 par.comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_zp_h.data(), count, MPI_DOUBLE, par.nbr_zp, 301,
                 recv_zm_h.data(), count, MPI_DOUBLE, par.nbr_zm, 301,
                 par.comm_cart, MPI_STATUS_IGNORE);

    if (par.nbr_zm != MPI_PROC_NULL) Kokkos::deep_copy(recv_zm_d, recv_zm_h);
    if (par.nbr_zp != MPI_PROC_NULL) Kokkos::deep_copy(recv_zp_d, recv_zp_h);

    Kokkos::parallel_for("unpack_q_z",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {NVAR,NG,nx6,ny6}),
        KOKKOS_LAMBDA(const int m, const int g, const int i, const int j) {
            const int p = idx_zbuf(m,g,i,j,nx6,ny6);
            if (par.nbr_zm != MPI_PROC_NULL) q(stage,m,i,j,g)      = recv_zm_d(p);
            if (par.nbr_zp != MPI_PROC_NULL) q(stage,m,i,j,nz+3+g) = recv_zp_d(p);
        });
    Kokkos::fence();
}

void apply_physical_boundaries_q(Kokkos::View<double*****> q, int stage, const Parameters& par) {
    if (par.nbr_xm == MPI_PROC_NULL || par.nbr_xp == MPI_PROC_NULL) {
        if      (par.x1bc == "open")     bound_open_x(q, stage, par);
        else if (par.x1bc == "periodic") bound_periodic_x(q, stage, par);
        else { std::cerr << "Error: unknown x1bc = " << par.x1bc << "\n"; std::abort(); }
    }

    if (par.nbr_ym == MPI_PROC_NULL || par.nbr_yp == MPI_PROC_NULL) {
        if      (par.x2bc == "open")     bound_open_y(q, stage, par);
        else if (par.x2bc == "periodic") bound_periodic_y(q, stage, par);
        else { std::cerr << "Error: unknown x2bc = " << par.x2bc << "\n"; std::abort(); }
    }

    if (par.nbr_zm == MPI_PROC_NULL || par.nbr_zp == MPI_PROC_NULL) {
        if      (par.x3bc == "open")     bound_open_z(q, stage, par);
        else if (par.x3bc == "periodic") bound_periodic_z(q, stage, par);
        else { std::cerr << "Error: unknown x3bc = " << par.x3bc << "\n"; std::abort(); }
    }
}

#endif // USE_MPI

void bound(Kokkos::View<double*****> q, int stage, const Parameters& par) {
#ifdef USE_MPI
    // Sequential x -> y -> z exchanges use full transverse extents including
    // existing ghost zones. This propagates edge/corner ghost data in the same
    // spirit as the original Fortran MPI boundary exchange.
    exchange_q_x(q, stage, par);
    exchange_q_y(q, stage, par);
    exchange_q_z(q, stage, par);
    apply_physical_boundaries_q(q, stage, par);
#else
    if      (par.x1bc == "periodic") bound_periodic_x(q, stage, par);
    else if (par.x1bc == "open")     bound_open_x(q, stage, par);
    else { std::cerr << "Error: unknown x1bc = " << par.x1bc << "\n"; std::abort(); }

    if      (par.x2bc == "periodic") bound_periodic_y(q, stage, par);
    else if (par.x2bc == "open")     bound_open_y(q, stage, par);
    else { std::cerr << "Error: unknown x2bc = " << par.x2bc << "\n"; std::abort(); }

    if      (par.x3bc == "periodic") bound_periodic_z(q, stage, par);
    else if (par.x3bc == "open")     bound_open_z(q, stage, par);
    else { std::cerr << "Error: unknown x3bc = " << par.x3bc << "\n"; std::abort(); }
#endif
}

