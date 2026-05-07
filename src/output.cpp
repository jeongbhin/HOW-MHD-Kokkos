#include "output.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <cstdint>

// ================================================================
// MPI-safe output file naming
// ------------------------------------------------------------
// In serial mode this preserves the old file names:
//   dump000010.dat / dump000010.vts
// In MPI mode each rank writes its own piece:
//   dump000010_rank000003.dat / dump000010_rank000003.vts
// ================================================================
std::string make_output_filename(const Parameters& par,
                                 int step,
                                 const std::string& ext) {
    std::ostringstream filename;
    filename << "output/dump"
             << std::setw(6) << std::setfill('0') << step;

#ifdef USE_MPI
    filename << "_rank"
             << std::setw(6) << std::setfill('0') << par.cart_rank;
#else
    (void)par;
#endif

    filename << ext;
    return filename.str();
}


// ================================================================
// Boundary-condition-aware index helper for host-side diagnostics
// ================================================================
int bc_index_host(const int idx,
                  const int lo,
                  const int hi,
                  const bool periodic) {

    if (periodic) {
        const int n = hi - lo + 1;
        int r = (idx - lo) % n;
        if (r < 0) r += n;
        return lo + r;
    } else {
        // open / outflow boundary: clamp
        if (idx < lo) return lo;
        if (idx > hi) return hi;
        return idx;
    }
}


// ================================================================
// Output dump file
// ================================================================
void output_dat(Kokkos::View<double*****> q,
            const Parameters& par,
            int step,
            double time) {

    std::filesystem::create_directories("output");

    const std::string filename = make_output_filename(par, step, ".dat");

    auto qh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);

    std::ofstream fout(filename);

    if (!fout) {
        std::cerr << "Error: cannot open output file "
                  << filename << "\n";
        return;
    }

    fout << std::scientific << std::setprecision(15);

    fout << "# step = " << step << "\n";
    fout << "# time = " << time << "\n";
    fout << "# rank = " << par.cart_rank << "\n";
    fout << "# local_active_size = " << par.nx << " " << par.ny << " " << par.nz << "\n";
    fout << "# global_active_size = " << par.gnx << " " << par.gny << " " << par.gnz << "\n";
    fout << "# global_start_index = " << par.istart << " " << par.jstart << " " << par.kstart << "\n";
    fout << "# columns: ig jg kg il jl kl x y z rho Mx My Mz Bx By Bz E pg vx vy vz divB\n";

    // ------------------------------------------------------------
    // Active cell-centered range
    // ------------------------------------------------------------
    const int ic_lo = 3;
    const int jc_lo = 3;
    const int kc_lo = 3;

    const int ic_hi = par.nx + 2;
    const int jc_hi = par.ny + 2;
    const int kc_hi = par.nz + 2;

    const bool x_periodic = (par.x1bc == "periodic");
    const bool y_periodic = (par.x2bc == "periodic");
    const bool z_periodic = (par.x3bc == "periodic");

    for (int k = kc_lo; k <= kc_hi; ++k) {
        for (int j = jc_lo; j <= jc_hi; ++j) {
            for (int i = ic_lo; i <= ic_hi; ++i) {

                const int il = i - 3;
                const int jl = j - 3;
                const int kl = k - 3;

                const int ig = par.istart + il;
                const int jg = par.jstart + jl;
                const int kg = par.kstart + kl;

                const double x = (static_cast<double>(ig) + 0.5) * par.dx;
                const double y = (static_cast<double>(jg) + 0.5) * par.dy;
                const double z = (static_cast<double>(kg) + 0.5) * par.dz;

                const double rho = qh(0,0,i,j,k);
                const double Mx  = qh(0,1,i,j,k);
                const double My  = qh(0,2,i,j,k);
                const double Mz  = qh(0,3,i,j,k);
                const double Bx  = qh(0,4,i,j,k);
                const double By  = qh(0,5,i,j,k);
                const double Bz  = qh(0,6,i,j,k);
                const double E   = qh(0,7,i,j,k);

                const double vx = Mx / rho;
                const double vy = My / rho;
                const double vz = Mz / rho;

                const double vv2 = vx*vx + vy*vy + vz*vz;
                const double BB2 = Bx*Bx + By*By + Bz*Bz;

                const double pg =
                    (par.gam - 1.0) * (E - 0.5 * rho * vv2 - 0.5 * BB2);

                // --------------------------------------------------------
                // Cell-centered diagnostic divB.
                //
                // This is not the exact face-centered CT divergence.
                // It is a useful diagnostic to see whether divB grows badly.
                //
                // For periodic boundaries: wrap.
                // For open boundaries: clamp.
                // For collapsed dimensions, skip derivative.
                // --------------------------------------------------------
                double divB = 0.0;

                if (par.nx > 1) {
                    const int im1 = bc_index_host(i - 1, ic_lo, ic_hi, x_periodic);
                    const int ip1 = bc_index_host(i + 1, ic_lo, ic_hi, x_periodic);

                    if (im1 == i || ip1 == i) {
                        // one-sided/clamped effective derivative near open edge
                        divB += (qh(0,4,ip1,j,k) - qh(0,4,im1,j,k)) / par.dx;
                    } else {
                        divB += (qh(0,4,ip1,j,k) - qh(0,4,im1,j,k)) / (2.0 * par.dx);
                    }
                }

                if (par.ny > 1) {
                    const int jm1 = bc_index_host(j - 1, jc_lo, jc_hi, y_periodic);
                    const int jp1 = bc_index_host(j + 1, jc_lo, jc_hi, y_periodic);

                    if (jm1 == j || jp1 == j) {
                        divB += (qh(0,5,i,jp1,k) - qh(0,5,i,jm1,k)) / par.dy;
                    } else {
                        divB += (qh(0,5,i,jp1,k) - qh(0,5,i,jm1,k)) / (2.0 * par.dy);
                    }
                }

                if (par.nz > 1) {
                    const int km1 = bc_index_host(k - 1, kc_lo, kc_hi, z_periodic);
                    const int kp1 = bc_index_host(k + 1, kc_lo, kc_hi, z_periodic);

                    if (km1 == k || kp1 == k) {
                        divB += (qh(0,6,i,j,kp1) - qh(0,6,i,j,km1)) / par.dz;
                    } else {
                        divB += (qh(0,6,i,j,kp1) - qh(0,6,i,j,km1)) / (2.0 * par.dz);
                    }
                }

                fout << ig << " "
                     << jg << " "
                     << kg << " "
                     << il << " "
                     << jl << " "
                     << kl << " "
                     << x << " "
                     << y << " "
                     << z << " "
                     << rho << " "
                     << Mx << " "
                     << My << " "
                     << Mz << " "
                     << Bx << " "
                     << By << " "
                     << Bz << " "
                     << E << " "
                     << pg << " "
                     << vx << " "
                     << vy << " "
                     << vz << " "
                     << divB << "\n";
            }
        }
    }

    fout.close();

    std::cout << "Wrote " << filename << "\n";
}


// ================================================================
// Binary VTS output file
// ParaView-readable XML StructuredGrid with appended raw binary data
// ================================================================
void output_vts_binary(Kokkos::View<double*****> q,
                       const Parameters& par,
                       int step,
                       double time) {

    std::filesystem::create_directories("output");

    const std::string filename = make_output_filename(par, step, ".vts");

    auto qh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);

    const int nx = par.nx;
    const int ny = par.ny;
    const int nz = par.nz;

    const int npts = nx * ny * nz;

    const int ic_lo = 3;
    const int jc_lo = 3;
    const int kc_lo = 3;

    const int ic_hi = par.nx + 2;
    const int jc_hi = par.ny + 2;
    const int kc_hi = par.nz + 2;

    const bool x_periodic = (par.x1bc == "periodic");
    const bool y_periodic = (par.x2bc == "periodic");
    const bool z_periodic = (par.x3bc == "periodic");

    // ------------------------------------------------------------
    // Storage arrays
    // ------------------------------------------------------------
    std::vector<double> rho(npts);
    std::vector<double> pressure(npts);
    std::vector<double> energy(npts);
    std::vector<double> v2(npts);
    std::vector<double> B2(npts);
    std::vector<double> divB(npts);

    std::vector<double> velocity(3 * npts);
    std::vector<double> magnetic_field(3 * npts);
    std::vector<double> momentum(3 * npts);
    std::vector<double> points(3 * npts);

    auto idx = [nx, ny](int i, int j, int k) {
        return i + nx * (j + ny * k);
    };

    // ------------------------------------------------------------
    // Fill host-side output arrays
    // ------------------------------------------------------------
    for (int k = kc_lo; k <= kc_hi; ++k) {
        for (int j = jc_lo; j <= jc_hi; ++j) {
            for (int i = ic_lo; i <= ic_hi; ++i) {

                const int ii = i - 3;
                const int jj = j - 3;
                const int kk = k - 3;

                const int ig = par.istart + ii;
                const int jg = par.jstart + jj;
                const int kg = par.kstart + kk;

                const int p = idx(ii, jj, kk);

                const double x = (static_cast<double>(ig) + 0.5) * par.dx;
                const double y = (static_cast<double>(jg) + 0.5) * par.dy;
                const double z = (static_cast<double>(kg) + 0.5) * par.dz;

                const double DD = qh(0,0,i,j,k);
                const double Mx = qh(0,1,i,j,k);
                const double My = qh(0,2,i,j,k);
                const double Mz = qh(0,3,i,j,k);
                const double Bx = qh(0,4,i,j,k);
                const double By = qh(0,5,i,j,k);
                const double Bz = qh(0,6,i,j,k);
                const double EE = qh(0,7,i,j,k);

                const double vx = Mx / DD;
                const double vy = My / DD;
                const double vz = Mz / DD;

                const double vv2 = vx*vx + vy*vy + vz*vz;
                const double BB2 = Bx*Bx + By*By + Bz*Bz;

                const double pg =
                    (par.gam - 1.0) * (EE - 0.5 * DD * vv2 - 0.5 * BB2);

                double divb = 0.0;

                if (par.nx > 1) {
                    const int im1 = bc_index_host(i - 1, ic_lo, ic_hi, x_periodic);
                    const int ip1 = bc_index_host(i + 1, ic_lo, ic_hi, x_periodic);

                    if (im1 == i || ip1 == i) {
                        divb += (qh(0,4,ip1,j,k) - qh(0,4,im1,j,k)) / par.dx;
                    } else {
                        divb += (qh(0,4,ip1,j,k) - qh(0,4,im1,j,k)) / (2.0 * par.dx);
                    }
                }

                if (par.ny > 1) {
                    const int jm1 = bc_index_host(j - 1, jc_lo, jc_hi, y_periodic);
                    const int jp1 = bc_index_host(j + 1, jc_lo, jc_hi, y_periodic);

                    if (jm1 == j || jp1 == j) {
                        divb += (qh(0,5,i,jp1,k) - qh(0,5,i,jm1,k)) / par.dy;
                    } else {
                        divb += (qh(0,5,i,jp1,k) - qh(0,5,i,jm1,k)) / (2.0 * par.dy);
                    }
                }

                if (par.nz > 1) {
                    const int km1 = bc_index_host(k - 1, kc_lo, kc_hi, z_periodic);
                    const int kp1 = bc_index_host(k + 1, kc_lo, kc_hi, z_periodic);

                    if (km1 == k || kp1 == k) {
                        divb += (qh(0,6,i,j,kp1) - qh(0,6,i,j,km1)) / par.dz;
                    } else {
                        divb += (qh(0,6,i,j,kp1) - qh(0,6,i,j,km1)) / (2.0 * par.dz);
                    }
                }

                rho[p]      = DD;
                pressure[p] = pg;
                energy[p]   = EE;
                v2[p]       = vv2;
                B2[p]       = BB2;
                divB[p]     = divb;

                velocity[3*p + 0] = vx;
                velocity[3*p + 1] = vy;
                velocity[3*p + 2] = vz;

                magnetic_field[3*p + 0] = Bx;
                magnetic_field[3*p + 1] = By;
                magnetic_field[3*p + 2] = Bz;

                momentum[3*p + 0] = Mx;
                momentum[3*p + 1] = My;
                momentum[3*p + 2] = Mz;

                points[3*p + 0] = x;
                points[3*p + 1] = y;
                points[3*p + 2] = z;
            }
        }
    }

    // ------------------------------------------------------------
    // Appended binary offset bookkeeping
    //
    // Each binary block is written as:
    // [uint64_t number_of_bytes][raw binary payload]
    //
    // offset is measured from the first byte after the "_" marker.
    // ------------------------------------------------------------
    std::uint64_t offset = 0;

    auto add_block = [&](std::uint64_t nbytes) {
        const std::uint64_t old = offset;
        offset += sizeof(std::uint64_t) + nbytes;
        return old;
    };

    const std::uint64_t off_rho      = add_block(sizeof(double) * rho.size());
    const std::uint64_t off_pressure = add_block(sizeof(double) * pressure.size());
    const std::uint64_t off_energy   = add_block(sizeof(double) * energy.size());
    const std::uint64_t off_v2       = add_block(sizeof(double) * v2.size());
    const std::uint64_t off_B2       = add_block(sizeof(double) * B2.size());
    const std::uint64_t off_divB     = add_block(sizeof(double) * divB.size());

    const std::uint64_t off_velocity =
        add_block(sizeof(double) * velocity.size());

    const std::uint64_t off_magnetic_field =
        add_block(sizeof(double) * magnetic_field.size());

    const std::uint64_t off_momentum =
        add_block(sizeof(double) * momentum.size());

    const std::uint64_t off_points =
        add_block(sizeof(double) * points.size());

    // ------------------------------------------------------------
    // Open file in binary mode
    // ------------------------------------------------------------
    std::ofstream fout(filename, std::ios::binary);

    if (!fout) {
        std::cerr << "Error: cannot open output file "
                  << filename << "\n";
        return;
    }

    // ------------------------------------------------------------
    // XML header
    // ------------------------------------------------------------
    fout << "<?xml version=\"1.0\"?>\n";
    fout << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" "
         << "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";

    const int i0 = par.istart;
    const int i1 = par.istart + nx - 1;
    const int j0 = par.jstart;
    const int j1 = par.jstart + ny - 1;
    const int k0 = par.kstart;
    const int k1 = par.kstart + nz - 1;

    fout << "  <StructuredGrid WholeExtent=\"0 " << par.gnx - 1
         << " 0 " << par.gny - 1
         << " 0 " << par.gnz - 1 << "\">\n";

    fout << "    <Piece Extent=\"" << i0 << " " << i1
         << " " << j0 << " " << j1
         << " " << k0 << " " << k1 << "\">\n";

    fout << "      <FieldData>\n";
    fout << "        <DataArray type=\"Float64\" Name=\"time\" "
         << "NumberOfTuples=\"1\" format=\"ascii\">\n";
    fout << "          " << std::scientific << std::setprecision(15)
         << time << "\n";
    fout << "        </DataArray>\n";
    fout << "        <DataArray type=\"Int32\" Name=\"step\" "
         << "NumberOfTuples=\"1\" format=\"ascii\">\n";
    fout << "          " << step << "\n";
    fout << "        </DataArray>\n";
    fout << "        <DataArray type=\"Int32\" Name=\"rank\" "
         << "NumberOfTuples=\"1\" format=\"ascii\">\n";
    fout << "          " << par.cart_rank << "\n";
    fout << "        </DataArray>\n";
    fout << "        <DataArray type=\"Int32\" Name=\"global_start_index\" "
         << "NumberOfComponents=\"3\" NumberOfTuples=\"1\" format=\"ascii\">\n";
    fout << "          " << par.istart << " " << par.jstart << " " << par.kstart << "\n";
    fout << "        </DataArray>\n";
    fout << "      </FieldData>\n";

    fout << "      <PointData>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"rho\" "
         << "NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << off_rho << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"pressure\" "
         << "NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << off_pressure << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"energy\" "
         << "NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << off_energy << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"v2\" "
         << "NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << off_v2 << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"B2\" "
         << "NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << off_B2 << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"divB\" "
         << "NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << off_divB << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"velocity\" "
         << "NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << off_velocity << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"magnetic_field\" "
         << "NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << off_magnetic_field << "\"/>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"momentum\" "
         << "NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << off_momentum << "\"/>\n";

    fout << "      </PointData>\n";

    fout << "      <CellData>\n";
    fout << "      </CellData>\n";

    fout << "      <Points>\n";
    fout << "        <DataArray type=\"Float64\" "
         << "NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << off_points << "\"/>\n";
    fout << "      </Points>\n";

    fout << "    </Piece>\n";
    fout << "  </StructuredGrid>\n";

    fout << "  <AppendedData encoding=\"raw\">\n";
    fout << "_";

    // ------------------------------------------------------------
    // Write appended binary blocks
    // ------------------------------------------------------------
    auto write_block = [&](const std::vector<double>& data) {
        const std::uint64_t nbytes =
            static_cast<std::uint64_t>(sizeof(double) * data.size());

        fout.write(reinterpret_cast<const char*>(&nbytes),
                   sizeof(std::uint64_t));

        fout.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(nbytes));
    };

    write_block(rho);
    write_block(pressure);
    write_block(energy);
    write_block(v2);
    write_block(B2);
    write_block(divB);
    write_block(velocity);
    write_block(magnetic_field);
    write_block(momentum);
    write_block(points);

    fout << "\n";
    fout << "  </AppendedData>\n";
    fout << "</VTKFile>\n";

    fout.close();

    std::cout << "Wrote " << filename
              << "  [binary VTS]\n";

}


void output(Kokkos::View<double*****> q,
            const Parameters& par,
            int step,
            double time) {

    if (par.output_format == "dat") {
        output_dat(q, par, step, time);
    } else if (par.output_format == "vtk" ||
               par.output_format == "vts") {
        output_vts_binary(q, par, step, time);
    } else {
        std::cerr << "Error: unknown output_format = "
                  << par.output_format << "\n";
        std::cerr << "Available options: dat, vtk\n";
        std::abort();
    }
}

