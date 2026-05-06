import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

try:
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    PROJECT_DIR = os.getcwd()

# 현재 dump가 bin/output 안에 있다고 했으므로 기본값을 이렇게 둠
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bin", "output")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")

# None이면 가장 마지막 dump 자동 선택
# 예: DUMP_FILE = "dump000010.dat"
DUMP_FILE = "dump000009.dat"

SAVE_FIGURE = True
SHOW_FIGURE = True

# plot variable choices:
# "rho", "pg", "vx", "vy", "vz", "Bx", "By", "Bz", "divB", "logdivB"
PLOT_VARS = ["rho", "pg", "Bz", "logdivB"]

FIGSIZE = (12, 10)
DPI = 200

# 예전 15-column output에서 divB를 Python에서 계산할 때 사용할 BC.
# Orszag-Tang은 보통 x,y periodic.
X_PERIODIC = True
Y_PERIODIC = True


# ============================================================
# Helper functions
# ============================================================

def find_latest_dump(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "dump*.dat")))
    if len(files) == 0:
        raise FileNotFoundError(f"No dump files found in {output_dir}")
    return files[-1]


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


def read_2d_dump(filename):
    """
    Read HOW-MHD 2D dump.

    Supports both formats:

    Old 15-column format:
    i j k x y z rho Mx My Mz Bx By Bz E pg

    New 19-column format:
    i j k x y z rho Mx My Mz Bx By Bz E pg vx vy vz divB
    """

    step, time = read_header_info(filename)

    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncol = data.shape[1]
    print(f"Detected {ncol} columns")

    col = {}
    col["i"]   = data[:, 0].astype(int)
    col["j"]   = data[:, 1].astype(int)
    col["k"]   = data[:, 2].astype(int)
    col["x"]   = data[:, 3]
    col["y"]   = data[:, 4]
    col["z"]   = data[:, 5]
    col["rho"] = data[:, 6]
    col["Mx"]  = data[:, 7]
    col["My"]  = data[:, 8]
    col["Mz"]  = data[:, 9]
    col["Bx"]  = data[:, 10]
    col["By"]  = data[:, 11]
    col["Bz"]  = data[:, 12]
    col["E"]   = data[:, 13]
    col["pg"]  = data[:, 14]

    # New output이면 vx,vy,vz,divB 직접 읽기
    # Old output이면 Python에서 계산
    if ncol >= 19:
        col["vx"]   = data[:, 15]
        col["vy"]   = data[:, 16]
        col["vz"]   = data[:, 17]
        col["divB"] = data[:, 18]
    else:
        col["vx"] = col["Mx"] / col["rho"]
        col["vy"] = col["My"] / col["rho"]
        col["vz"] = col["Mz"] / col["rho"]

    nx = col["i"].max() + 1
    ny = col["j"].max() + 1
    nz = col["k"].max() + 1

    if nz != 1:
        print(f"Warning: nz = {nz}. This script plots the k=0 slice only.")

    # k=0 slice only
    mask = col["k"] == 0

    grid = {}
    for key, val in col.items():
        arr = val[mask]
        if key in ["i", "j", "k"]:
            continue
        grid[key] = arr.reshape((ny, nx))

    grid["x"] = col["x"][mask].reshape((ny, nx))
    grid["y"] = col["y"][mask].reshape((ny, nx))
    grid["z"] = col["z"][mask].reshape((ny, nx))

    # Old output이면 divB 계산
    if "divB" not in grid:
        grid["divB"] = compute_divB_cell_centered(
            grid["Bx"],
            grid["By"],
            grid["x"],
            grid["y"],
            x_periodic=X_PERIODIC,
            y_periodic=Y_PERIODIC,
        )

    grid["logdivB"] = np.log10(np.abs(grid["divB"]) + 1.0e-30)

    return step, time, nx, ny, grid


def compute_divB_cell_centered(Bx, By, x, y,
                               x_periodic=True,
                               y_periodic=True):
    """
    Cell-centered diagnostic:
        divB = dBx/dx + dBy/dy

    This is not exact face-centered CT divergence.
    It is only a useful diagnostic for plotting.
    """

    ny, nx = Bx.shape

    divB = np.zeros_like(Bx)

    if nx > 1:
        dx = x[0, 1] - x[0, 0]
    else:
        dx = 1.0

    if ny > 1:
        dy = y[1, 0] - y[0, 0]
    else:
        dy = 1.0

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

                    if im1 == i or ip1 == i:
                        divB[j, i] += (Bx[j, ip1] - Bx[j, im1]) / dx
                    else:
                        divB[j, i] += (Bx[j, ip1] - Bx[j, im1]) / (2.0 * dx)

            if ny > 1:
                if y_periodic:
                    jm1 = (j - 1) % ny
                    jp1 = (j + 1) % ny
                    divB[j, i] += (By[jp1, i] - By[jm1, i]) / (2.0 * dy)
                else:
                    jm1 = max(j - 1, 0)
                    jp1 = min(j + 1, ny - 1)

                    if jm1 == j or jp1 == j:
                        divB[j, i] += (By[jp1, i] - By[jm1, i]) / dy
                    else:
                        divB[j, i] += (By[jp1, i] - By[jm1, i]) / (2.0 * dy)

    return divB


def plot_2d(grid, vars_to_plot, step, time, filename):
    os.makedirs(FIGURE_DIR, exist_ok=True)

    nvar = len(vars_to_plot)
    ncols = 2
    nrows = int(np.ceil(nvar / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    if nrows == 1:
        axes = np.array([axes])

    axes = axes.ravel()

    x = grid["x"]
    y = grid["y"]

    for ax, var in zip(axes, vars_to_plot):

        if var not in grid:
            raise KeyError(
                f"Variable '{var}' not found. "
                f"Available variables: {list(grid.keys())}"
            )

        arr = grid[var]

        im = ax.pcolormesh(
            x,
            y,
            arr,
            shading="auto",
        )

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(var)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(var)

    for ax in axes[nvar:]:
        ax.axis("off")

    basename = os.path.basename(filename)
    fig.suptitle(
        f"2D dump: {basename}, step={step}, time={time:.6e}",
        fontsize=14,
    )

    if SAVE_FIGURE:
        outname = os.path.join(FIGURE_DIR, f"dump2d_{step:06d}.png")
        fig.savefig(outname, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure: {outname}")

    return fig, axes


# ============================================================
# Main
# ============================================================

if DUMP_FILE is None:
    filename = find_latest_dump(OUTPUT_DIR)
else:
    filename = os.path.join(OUTPUT_DIR, DUMP_FILE)

print(f"PROJECT_DIR = {PROJECT_DIR}")
print(f"OUTPUT_DIR  = {OUTPUT_DIR}")
print(f"Reading     = {filename}")

step, time, nx, ny, grid = read_2d_dump(filename)

print(f"step = {step}")
print(f"time = {time:.15e}")
print(f"nx = {nx}, ny = {ny}")
print(f"rho min/max  = {grid['rho'].min():.6e}, {grid['rho'].max():.6e}")
print(f"pg  min/max  = {grid['pg'].min():.6e}, {grid['pg'].max():.6e}")
print(f"Bx  min/max  = {grid['Bx'].min():.6e}, {grid['Bx'].max():.6e}")
print(f"By  min/max  = {grid['By'].min():.6e}, {grid['By'].max():.6e}")
print(f"Bz  min/max  = {grid['Bz'].min():.6e}, {grid['Bz'].max():.6e}")
print(f"divB min/max = {grid['divB'].min():.6e}, {grid['divB'].max():.6e}")
print(f"max |divB|   = {np.max(np.abs(grid['divB'])):.6e}")

fig, axes = plot_2d(
    grid,
    PLOT_VARS,
    step,
    time,
    filename,
)

if SHOW_FIGURE:
    plt.show()