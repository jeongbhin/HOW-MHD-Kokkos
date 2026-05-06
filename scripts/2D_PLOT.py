import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# USER SETTINGS
# ============================================================

try:
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    PROJECT_DIR = os.getcwd()

# Default HOW-MHD dump location.
# This assumes the executable is run from bin/ and writes to bin/output/.
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bin", "output")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")

# Select which dump to plot.
# If DUMP_STEP is an integer, the script reads dump000000.dat, dump000001.dat, ...
# If DUMP_STEP is None, the script reads the latest dump automatically.
DUMP_STEP = 9

SAVE_FIGURE = True
SHOW_FIGURE = True

# Supported options:
# "rho", "pg", "vabs", "Babs", "vx", "vy", "vz", "Bx", "By", "Bz", "divB", "logdivB"
PLOT_VARS = ["rho", "pg", "vabs", "Babs"]

FIGSIZE = (13, 10)
DPI = 220
COLORMAP = "turbo"

# Percentile clipping makes the colormap cleaner for shocks/outliers.
# Set to None to use the full min/max range.
PERCENTILE_CLIP = (1.0, 99.0)

# Used only when divB is computed in Python from old 15-column outputs.
X_PERIODIC = True
Y_PERIODIC = True


# ============================================================
# Global plot style
# ============================================================

mpl.rcParams.update({
    "font.size": 13,
    "axes.linewidth": 1.2,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": DPI,
})


# ============================================================
# Helper functions
# ============================================================

def find_latest_dump(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "dump*.dat")))
    if len(files) == 0:
        raise FileNotFoundError(f"No dump files found in {output_dir}")
    return files[-1]


def make_dump_filename(output_dir, dump_step):
    if dump_step is None:
        return find_latest_dump(output_dir)
    return os.path.join(output_dir, f"dump{dump_step:06d}.dat")


def read_header_info(filename):
    step = None
    time = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("# step"):
                step = int(line.split("=")[1])
            elif line.startswith("# time"):
                time = float(line.split("=")[1])
    return step, time


def compute_divB_cell_centered(Bx, By, x, y, x_periodic=True, y_periodic=True):
    """
    Compute a cell-centered divergence diagnostic:
        divB = dBx/dx + dBy/dy

    This is not the exact face-centered CT divergence.
    It is only intended as a plotting diagnostic.
    """
    ny, nx = Bx.shape
    divB = np.zeros_like(Bx)

    dx = x[0, 1] - x[0, 0] if nx > 1 else 1.0
    dy = y[1, 0] - y[0, 0] if ny > 1 else 1.0

    for j in range(ny):
        for i in range(nx):
            if nx > 1:
                if x_periodic:
                    im1 = (i - 1) % nx
                    ip1 = (i + 1) % nx
                    divB[j, i] += (Bx[j, ip1] - Bx[j, im1]) / (2.0 * dx)
                else:
                    im1 = max(i - 1, 0)
                    ip1 = min(i + 1, nx - 1)
                    denom = dx if (im1 == i or ip1 == i) else 2.0 * dx
                    divB[j, i] += (Bx[j, ip1] - Bx[j, im1]) / denom

            if ny > 1:
                if y_periodic:
                    jm1 = (j - 1) % ny
                    jp1 = (j + 1) % ny
                    divB[j, i] += (By[jp1, i] - By[jm1, i]) / (2.0 * dy)
                else:
                    jm1 = max(j - 1, 0)
                    jp1 = min(j + 1, ny - 1)
                    denom = dy if (jm1 == j or jp1 == j) else 2.0 * dy
                    divB[j, i] += (By[jp1, i] - By[jm1, i]) / denom

    return divB


def read_2d_dump(filename):
    """
    Read a HOW-MHD 2D dump.

    Supported formats:
      15 columns: i j k x y z rho Mx My Mz Bx By Bz E pg
      19 columns: i j k x y z rho Mx My Mz Bx By Bz E pg vx vy vz divB
    """
    step, time = read_header_info(filename)
    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncol = data.shape[1]
    print(f"Detected {ncol} columns")

    col = {
        "i": data[:, 0].astype(int),
        "j": data[:, 1].astype(int),
        "k": data[:, 2].astype(int),
        "x": data[:, 3],
        "y": data[:, 4],
        "z": data[:, 5],
        "rho": data[:, 6],
        "Mx": data[:, 7],
        "My": data[:, 8],
        "Mz": data[:, 9],
        "Bx": data[:, 10],
        "By": data[:, 11],
        "Bz": data[:, 12],
        "E": data[:, 13],
        "pg": data[:, 14],
    }

    if ncol >= 19:
        col["vx"] = data[:, 15]
        col["vy"] = data[:, 16]
        col["vz"] = data[:, 17]
        col["divB"] = data[:, 18]
    else:
        col["vx"] = col["Mx"] / col["rho"]
        col["vy"] = col["My"] / col["rho"]
        col["vz"] = col["Mz"] / col["rho"]

    col["vabs"] = np.sqrt(col["vx"]**2 + col["vy"]**2 + col["vz"]**2)
    col["Babs"] = np.sqrt(col["Bx"]**2 + col["By"]**2 + col["Bz"]**2)

    nx = col["i"].max() + 1
    ny = col["j"].max() + 1
    nz = col["k"].max() + 1

    if nz != 1:
        print(f"Warning: nz = {nz}. This script plots the k=0 slice only.")

    mask = col["k"] == 0
    grid = {}

    for key, val in col.items():
        if key in ["i", "j", "k"]:
            continue
        grid[key] = val[mask].reshape((ny, nx))

    if "divB" not in grid:
        grid["divB"] = compute_divB_cell_centered(
            grid["Bx"], grid["By"], grid["x"], grid["y"],
            x_periodic=X_PERIODIC,
            y_periodic=Y_PERIODIC,
        )

    grid["logdivB"] = np.log10(np.abs(grid["divB"]) + 1.0e-30)

    return step, time, nx, ny, grid


def get_color_limits(arr):
    if PERCENTILE_CLIP is None:
        return None, None
    pmin, pmax = PERCENTILE_CLIP
    vmin, vmax = np.nanpercentile(arr, [pmin, pmax])
    if np.isclose(vmin, vmax):
        return None, None
    return vmin, vmax


def pretty_label(var):
    labels = {
        "rho": r"$\rho$",
        "pg": r"$p_g$",
        "vabs": r"$|\mathbf{v}|$",
        "Babs": r"$|\mathbf{B}|$",
        "vx": r"$v_x$",
        "vy": r"$v_y$",
        "vz": r"$v_z$",
        "Bx": r"$B_x$",
        "By": r"$B_y$",
        "Bz": r"$B_z$",
        "divB": r"$\nabla\cdot\mathbf{B}$",
        "logdivB": r"$\log_{10}(|\nabla\cdot\mathbf{B}|)$",
    }
    return labels.get(var, var)


def plot_2d(grid, vars_to_plot, step, time, filename):
    os.makedirs(FIGURE_DIR, exist_ok=True)

    nvar = len(vars_to_plot)
    ncols = 2
    nrows = int(np.ceil(nvar / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE, constrained_layout=True)

    if nrows == 1:
        axes = np.array([axes])

    axes = axes.ravel()
    x = grid["x"]
    y = grid["y"]

    for ax, var in zip(axes, vars_to_plot):
        if var not in grid:
            raise KeyError(f"Variable '{var}' not found. Available variables: {list(grid.keys())}")

        arr = grid[var]
        vmin, vmax = get_color_limits(arr)

        im = ax.pcolormesh(
            x, y, arr,
            shading="auto",
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(pretty_label(var), pad=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label(pretty_label(var))

    for ax in axes[nvar:]:
        ax.axis("off")

    basename = os.path.basename(filename)
    fig.suptitle(f"HOW-MHD 2D dump: {basename}   step={step}   time={time:.6e}", fontsize=15)

    if SAVE_FIGURE:
        outname = os.path.join(FIGURE_DIR, f"dump2d_{step:06d}.png")
        fig.savefig(outname, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure: {outname}")

    return fig, axes


# ============================================================
# Main
# ============================================================

filename = make_dump_filename(OUTPUT_DIR, DUMP_STEP)

print(f"PROJECT_DIR = {PROJECT_DIR}")
print(f"OUTPUT_DIR  = {OUTPUT_DIR}")
print(f"Reading     = {filename}")

step, time, nx, ny, grid = read_2d_dump(filename)

print(f"step = {step}")
print(f"time = {time:.15e}")
print(f"nx = {nx}, ny = {ny}")
print(f"rho min/max  = {grid['rho'].min():.6e}, {grid['rho'].max():.6e}")
print(f"pg  min/max  = {grid['pg'].min():.6e}, {grid['pg'].max():.6e}")
print(f"|v| min/max  = {grid['vabs'].min():.6e}, {grid['vabs'].max():.6e}")
print(f"|B| min/max  = {grid['Babs'].min():.6e}, {grid['Babs'].max():.6e}")
print(f"divB min/max = {grid['divB'].min():.6e}, {grid['divB'].max():.6e}")
print(f"max |divB|   = {np.max(np.abs(grid['divB'])):.6e}")

fig, axes = plot_2d(grid, PLOT_VARS, step, time, filename)

if SHOW_FIGURE:
    plt.show()
