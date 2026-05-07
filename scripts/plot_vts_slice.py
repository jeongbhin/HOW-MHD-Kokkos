#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast Spyder-friendly HOW-MHD VTS slice plotter.

This version avoids:
  - PyVista Plotter
  - VTK/OpenGL rendering
  - MultiBlock.combine()

It reads rank-wise VTS files one by one, extracts only the central slice,
and plots with Matplotlib.

Expected directory structure:

how-mhd/
  scripts/
    plot_vts_slice_fast_spyder.py
  bin/
    output/
      dump000000_rank000000.vts
      dump000000_rank000001.vts
      ...
  figures/
"""

import gc
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


# ============================================================
# User settings
# ============================================================

STEP = 10            # dump number
NORMAL = "z"             # "x", "y", or "z"

FIELD = "rho"       # rho, pressure, energy, divB, v_abs, B_abs, Bx, By, Bz, vx, vy, vz

DO_SINGLE_PANEL = True
DO_FOUR_PANEL = True

LOGSCALE_SINGLE = True

SHOW_FIGURE = True
SAVE_FIGURE = True

CMAP = "inferno"

# If True, overwrite duplicate points/cells by averaging.
# This is useful if VTS pieces share boundary points.
AVERAGE_DUPLICATES = True


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

OUTDIR = ROOT_DIR / "bin" / "output"
FIGDIR = ROOT_DIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Basic helpers
# ============================================================

def get_vts_files(step, outdir):
    pattern = outdir / f"dump{step:06d}_rank*.vts"
    files = sorted(glob.glob(str(pattern)))

    if not files:
        raise FileNotFoundError(f"No VTS files found with pattern:\n{pattern}")

    print(f"Found {len(files)} VTS files for step {step}")
    for f in files:
        print("  ", f)

    return files


def get_scalar_from_mesh(mesh, field):
    """
    Return scalar array from mesh, including derived fields.
    This function does not modify mesh.
    """
    names = mesh.array_names

    # Direct scalar fields
    if field in names:
        arr = np.asarray(mesh[field])
        if arr.ndim == 1:
            return arr
        raise ValueError(f"Field '{field}' is not scalar. Shape = {arr.shape}")

    # Derived from velocity
    if "velocity" in names:
        v = np.asarray(mesh["velocity"])

        if field == "vx":
            return v[:, 0]
        if field == "vy":
            return v[:, 1]
        if field == "vz":
            return v[:, 2]
        if field == "v_abs":
            return np.sqrt(np.sum(v * v, axis=1))

    # Derived from magnetic_field
    if "magnetic_field" in names:
        b = np.asarray(mesh["magnetic_field"])

        if field == "Bx":
            return b[:, 0]
        if field == "By":
            return b[:, 1]
        if field == "Bz":
            return b[:, 2]
        if field == "B_abs":
            return np.sqrt(np.sum(b * b, axis=1))

    # Derived from v2 / B2
    if field == "v_abs" and "v2" in names:
        return np.sqrt(np.maximum(np.asarray(mesh["v2"]), 0.0))

    if field == "B_abs" and "B2" in names:
        return np.sqrt(np.maximum(np.asarray(mesh["B2"]), 0.0))

    raise KeyError(f"Field '{field}' not found. Available arrays:\n{names}")


def get_geometry_for_field(mesh, values):
    """
    Get coordinates corresponding to field values.

    If values are point data, use mesh.points.
    If values are cell data, use mesh.cell_centers().points.
    """
    nval = values.size

    if nval == mesh.n_points:
        pts = mesh.points
    elif nval == mesh.n_cells:
        pts = mesh.cell_centers().points
    else:
        raise RuntimeError(
            f"Cannot match field size to mesh geometry.\n"
            f"field size = {nval}\n"
            f"n_points   = {mesh.n_points}\n"
            f"n_cells    = {mesh.n_cells}"
        )

    return pts[:, 0], pts[:, 1], pts[:, 2]


def axis_info(normal):
    """
    Return coordinate mapping for a slice.

    normal = "z":
        plot x-y at fixed z
    normal = "y":
        plot x-z at fixed y
    normal = "x":
        plot y-z at fixed x
    """
    if normal == "z":
        return 2, 0, 1, "z", "x", "y"
    if normal == "y":
        return 1, 0, 2, "y", "x", "z"
    if normal == "x":
        return 0, 1, 2, "x", "y", "z"

    raise ValueError("NORMAL must be 'x', 'y', or 'z'")


def collect_global_bounds(files, field):
    """
    First pass:
    determine global coordinate bounds and available arrays.
    """
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    zmin, zmax = np.inf, -np.inf

    available_printed = False

    for fname in files:
        mesh = pv.read(fname)

        if not available_printed:
            print("\nAvailable arrays from first file:")
            for name in mesh.array_names:
                print("  ", name)
            available_printed = True

        values = get_scalar_from_mesh(mesh, field)
        x, y, z = get_geometry_for_field(mesh, values)

        xmin = min(xmin, np.nanmin(x))
        xmax = max(xmax, np.nanmax(x))
        ymin = min(ymin, np.nanmin(y))
        ymax = max(ymax, np.nanmax(y))
        zmin = min(zmin, np.nanmin(z))
        zmax = max(zmax, np.nanmax(z))

        del mesh, values, x, y, z

    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    print("\nGlobal data bounds from pieces:")
    print(f"  x = {xmin} ... {xmax}")
    print(f"  y = {ymin} ... {ymax}")
    print(f"  z = {zmin} ... {zmax}")

    return bounds


def find_slice_coordinate(files, field, normal, bounds):
    """
    Find the actual available coordinate closest to the geometric center.
    """
    fixed_axis, _, _, fixed_name, _, _ = axis_info(normal)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    if normal == "z":
        target = 0.5 * (zmin + zmax)
    elif normal == "y":
        target = 0.5 * (ymin + ymax)
    elif normal == "x":
        target = 0.5 * (xmin + xmax)

    coords_all = []

    for fname in files:
        mesh = pv.read(fname)
        values = get_scalar_from_mesh(mesh, field)
        x, y, z = get_geometry_for_field(mesh, values)

        coords = [x, y, z][fixed_axis]
        coords_all.append(np.round(coords, 12))

        del mesh, values, x, y, z, coords

    coords_all = np.concatenate(coords_all)
    unique_coords = np.unique(coords_all)

    idx = np.argmin(np.abs(unique_coords - target))
    slice_coord = unique_coords[idx]

    print(f"\nCentral slice:")
    print(f"  normal       = {normal}")
    print(f"  target {fixed_name} = {target}")
    print(f"  using  {fixed_name} = {slice_coord}")

    return slice_coord


def build_slice_grid(files, field, normal, slice_coord):
    """
    Second pass:
    read each rank file, extract points/cells on the selected slice,
    and assemble one global 2D array.
    """
    fixed_axis, ax1, ax2, fixed_name, xlabel, ylabel = axis_info(normal)

    all_x2 = []
    all_y2 = []
    all_val = []

    for fname in files:
        mesh = pv.read(fname)
        values = get_scalar_from_mesh(mesh, field)
        x, y, z = get_geometry_for_field(mesh, values)

        coords = [x, y, z]
        fixed = coords[fixed_axis]
        c1 = coords[ax1]
        c2 = coords[ax2]

        mask = np.isclose(np.round(fixed, 12), slice_coord, rtol=0.0, atol=1.0e-12)

        if np.any(mask):
            all_x2.append(np.round(c1[mask], 12))
            all_y2.append(np.round(c2[mask], 12))
            all_val.append(values[mask])

        del mesh, values, x, y, z, coords, fixed, c1, c2, mask

    if not all_x2:
        raise RuntimeError("No data found on selected slice.")

    xs = np.concatenate(all_x2)
    ys = np.concatenate(all_y2)
    vs = np.concatenate(all_val)

    xu = np.unique(xs)
    yu = np.unique(ys)

    nx = len(xu)
    ny = len(yu)

    print(f"\n2D slice grid:")
    print(f"  {xlabel} cells/points = {nx}")
    print(f"  {ylabel} cells/points = {ny}")
    print(f"  raw slice samples     = {vs.size}")

    x_index = {v: i for i, v in enumerate(xu)}
    y_index = {v: i for i, v in enumerate(yu)}

    Z_sum = np.zeros((ny, nx), dtype=float)
    Z_cnt = np.zeros((ny, nx), dtype=float)

    for xx, yy, vv in zip(xs, ys, vs):
        i = x_index[xx]
        j = y_index[yy]

        if AVERAGE_DUPLICATES:
            Z_sum[j, i] += vv
            Z_cnt[j, i] += 1.0
        else:
            Z_sum[j, i] = vv
            Z_cnt[j, i] = 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        Z = Z_sum / Z_cnt

    Z[Z_cnt == 0.0] = np.nan

    X, Y = np.meshgrid(xu, yu)

    return X, Y, Z, xlabel, ylabel


# ============================================================
# Plot helpers
# ============================================================

def plot_single_slice(step, field, normal, outdir, figdir,
                      cmap="inferno", logscale=False,
                      save=True, show=True):
    files = get_vts_files(step, outdir)

    bounds = collect_global_bounds(files, field)
    slice_coord = find_slice_coordinate(files, field, normal, bounds)
    X, Y, Z, xlabel, ylabel = build_slice_grid(files, field, normal, slice_coord)

    plot_data = Z.copy()
    label = field

    if logscale:
        plot_data = np.log10(np.maximum(plot_data, 1.0e-300))
        label = f"log10_{field}"

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    im = ax.pcolormesh(X, Y, plot_data, shading="auto", cmap=cmap)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{label}, {normal}-slice, step={step:06d}")

    if save:
        png = figdir / f"{label}_slice_{normal}_step{step:06d}.png"
        fig.savefig(png, dpi=200)
        print(f"Saved {png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    gc.collect()


def plot_four_panel_slice(step, normal, outdir, figdir, cmap="inferno",
                          save=True, show=True):
    fields = [
        ("rho", r"$\rho$"),
        ("pressure", r"$p_g$"),
        ("v_abs", r"$|v|$"),
        ("B_abs", r"$|B|$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Use rho to determine slice coordinate consistently.
    files = get_vts_files(step, outdir)
    bounds = collect_global_bounds(files, "rho")
    slice_coord = find_slice_coordinate(files, "rho", normal, bounds)

    for ax, (field, title) in zip(axes.ravel(), fields):
        try:
            X, Y, Z, xlabel, ylabel = build_slice_grid(files, field, normal, slice_coord)

            im = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=0.9)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)

        except Exception as e:
            ax.text(
                0.5, 0.5,
                f"{field} failed\n{e}",
                ha="center", va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()

    fig.suptitle(f"HOW-MHD {normal}-slice, step={step:06d}", fontsize=16)

    if save:
        png = figdir / f"four_panel_slice_{normal}_step{step:06d}.png"
        fig.savefig(png, dpi=200)
        print(f"Saved {png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    gc.collect()


# ============================================================
# Main
# ============================================================

def main():
    print("========================================")
    print("HOW-MHD fast slice-only plotter")
    print("========================================")
    print(f"ROOT_DIR = {ROOT_DIR}")
    print(f"OUTDIR   = {OUTDIR}")
    print(f"FIGDIR   = {FIGDIR}")
    print(f"STEP     = {STEP}")
    print(f"NORMAL   = {NORMAL}")
    print(f"FIELD    = {FIELD}")
    print("========================================\n")

    if DO_SINGLE_PANEL:
        plot_single_slice(
            step=STEP,
            field=FIELD,
            normal=NORMAL,
            outdir=OUTDIR,
            figdir=FIGDIR,
            cmap=CMAP,
            logscale=LOGSCALE_SINGLE,
            save=SAVE_FIGURE,
            show=SHOW_FIGURE,
        )

    if DO_FOUR_PANEL:
        plot_four_panel_slice(
            step=STEP,
            normal=NORMAL,
            outdir=OUTDIR,
            figdir=FIGDIR,
            cmap=CMAP,
            save=SAVE_FIGURE,
            show=SHOW_FIGURE,
        )

    plt.close("all")
    gc.collect()

    print("\nDone.")


main()