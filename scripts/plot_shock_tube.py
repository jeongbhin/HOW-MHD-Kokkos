import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================
# Project root 기준으로 실행한다고 가정
# 예: how-mhd/scripts/plot_shock_tube.py 를 Spyder에서 실행
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(PROJECT_DIR, "bin/output")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")

# None이면 가장 마지막 dump를 자동 선택
# 예: "dump000020.dat"로 지정 가능
DUMP_FILE = "dump000007.dat"

SAVE_FIGURE = True
SHOW_FIGURE = True

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------
FIGSIZE = (10, 10)
DPI = 200


# ============================================================
# Helper functions
# ============================================================
def find_latest_dump(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "dump*.dat")))
    if len(files) == 0:
        raise FileNotFoundError(f"No dump files found in {output_dir}")
    return files[-1]


def read_dump(filename):
    """
    Read output/dumpXXXXXX.dat.

    Columns:
    i j k x y z rho Mx My Mz Bx By Bz E pg
    """

    time = None
    step = None

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("# step"):
                step = int(line.split("=")[1])
            elif line.startswith("# time"):
                time = float(line.split("=")[1])

    data = np.loadtxt(filename, comments="#")

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

    # derived quantities
    col["vx"] = col["Mx"] / col["rho"]
    col["vy"] = col["My"] / col["rho"]
    col["vz"] = col["Mz"] / col["rho"]

    col["B2"] = col["Bx"]**2 + col["By"]**2 + col["Bz"]**2
    col["v2"] = col["vx"]**2 + col["vy"]**2 + col["vz"]**2

    return step, time, col


def plot_shock_tube(col, step, time, filename, save_figure=True):
    os.makedirs(FIGURE_DIR, exist_ok=True)

    x = col["x"]

    fig, axes = plt.subplots(4, 1, figsize=FIGSIZE, sharex=True)

    axes[0].plot(x, col["rho"], lw=2)
    axes[0].set_ylabel(r"$\rho$")

    axes[1].plot(x, col["vx"], lw=2)
    axes[1].set_ylabel(r"$v_x$")

    axes[2].plot(x, col["pg"], lw=2)
    axes[2].set_ylabel(r"$p_g$")

    axes[3].plot(x, col["By"], lw=2)
    axes[3].set_ylabel(r"$B_y$")
    axes[3].set_xlabel(r"$x$")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    basename = os.path.basename(filename)
    fig.suptitle(
        f"Brio-Wu MHD Shock Tube: {basename}, step={step}, time={time:.6e}",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_figure:
        outname = os.path.join(FIGURE_DIR, f"shock_tube_{step:06d}.png")
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

print(f"Reading: {filename}")

step, time, col = read_dump(filename)

print(f"step = {step}")
print(f"time = {time}")
print(f"rho min/max = {col['rho'].min():.6e}, {col['rho'].max():.6e}")
print(f"pg  min/max = {col['pg'].min():.6e}, {col['pg'].max():.6e}")
print(f"By  min/max = {col['By'].min():.6e}, {col['By'].max():.6e}")

fig, axes = plot_shock_tube(
    col,
    step,
    time,
    filename,
    save_figure=SAVE_FIGURE,
)

if SHOW_FIGURE:
    plt.show()
