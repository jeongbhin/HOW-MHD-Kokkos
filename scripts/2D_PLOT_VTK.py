import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError as exc:
    raise ImportError(
        "This script requires the Python VTK package.\n"
        "Install with: pip install vtk\n"
        "or load a Python environment that provides vtk."
    ) from exc


# ============================================================
# USER SETTINGS
# ============================================================

try:
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    PROJECT_DIR = os.getcwd()

# If executable is run from bin/ and writes to bin/output/
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bin", "output")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")

# If DUMP_STEP is an integer, read dump000009.vts, etc.
# If DUMP_STEP is None, read the latest dump*.vts automatically.
DUMP_STEP = 6

SAVE_FIGURE = True
SHOW_FIGURE = True

# Supported variables:
# "rho", "pg", "pressure", "E", "energy",
# "vabs", "Babs", "v2", "B2",
# "vx", "vy", "vz", "Bx", "By", "Bz",
# "Mx", "My", "Mz", "divB", "logdivB"
PLOT_VARS = ["rho", "pg", "vabs", "Babs"]

# For 3D data, this script plots a 2D slice.
# Options: "xy", "xz", "yz"
SLICE_PLANE = "xy"

# If None, use the middle slice.
# For 2D runs with nz = 1, this automatically gives k = 0.
SLICE_INDEX = None

FIGSIZE = (13, 10)
DPI = 220
COLORMAP = "turbo"

# Percentile clipping makes the colormap cleaner for shocks/outliers.
# Set to None to use the full min/max range.
PERCENTILE_CLIP = (1.0, 99.0)


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
# File helpers
# ============================================================

def find_latest_dump(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "dump*.vts")))
    if len(files) == 0:
        raise FileNotFoundError(f"No VTS dump files found in {output_dir}")
    return files[-1]


def make_dump_filename(output_dir, dump_step):
    if dump_step is None:
        return find_latest_dump(output_dir)
    return os.path.join(output_dir, f"dump{dump_step:06d}.vts")


# ============================================================
# VTK loader
# ============================================================

def read_vts_dump(filename):
    """
    Read a HOW-MHD binary .vts dump.

    Expected arrays from output_vts_binary:

    Scalars:
        rho
        pressure
        energy
        v2
        B2
        divB

    Vectors:
        velocity
        magnetic_field
        momentum

    FieldData:
        time
        step
    """

    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    grid_vtk = reader.GetOutput()

    if grid_vtk is None or grid_vtk.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read VTS file or empty grid: {filename}")

    # VTK version-compatible GetDimensions call.
    dims = [0, 0, 0]
    grid_vtk.GetDimensions(dims)
    nx, ny, nz = dims

    npts = grid_vtk.GetNumberOfPoints()

    if npts != nx * ny * nz:
        raise RuntimeError(
            f"Point count mismatch: npts={npts}, nx*ny*nz={nx*ny*nz}"
        )

    point_data = grid_vtk.GetPointData()
    field_data = grid_vtk.GetFieldData()

    def get_field_scalar(name, default=None):
        arr = field_data.GetArray(name)
        if arr is None:
            return default
        vals = vtk_to_numpy(arr)
        if vals.size == 0:
            return default
        return vals[0]

    step = get_field_scalar("step", default=None)
    time = get_field_scalar("time", default=None)

    if step is not None:
        step = int(step)

    if time is not None:
        time = float(time)

    points_vtk = grid_vtk.GetPoints()

    if points_vtk is None:
        raise RuntimeError(f"No point coordinates found in {filename}")

    points_np = vtk_to_numpy(points_vtk.GetData())

    if points_np.shape[0] != npts:
        raise RuntimeError(
            f"Point coordinate size mismatch: {points_np.shape[0]} vs {npts}"
        )

    # VTK StructuredGrid point ordering is consistent with the write order:
    # p = i + nx * (j + ny * k)
    x = points_np[:, 0].reshape((nz, ny, nx))
    y = points_np[:, 1].reshape((nz, ny, nx))
    z = points_np[:, 2].reshape((nz, ny, nx))

    data = {
        "x": x,
        "y": y,
        "z": z,
    }

    def read_scalar(name, alias=None):
        arr = point_data.GetArray(name)
        if arr is None:
            return

        vals = vtk_to_numpy(arr)

        if vals.size != npts:
            raise RuntimeError(
                f"Scalar array '{name}' has size {vals.size}, expected {npts}"
            )

        vals = vals.reshape((nz, ny, nx))
        data[name] = vals

        if alias is not None:
            data[alias] = vals

    def read_vector(name, component_names):
        arr = point_data.GetArray(name)
        if arr is None:
            return

        vals = vtk_to_numpy(arr)

        if vals.shape[0] != npts:
            raise RuntimeError(
                f"Vector array '{name}' has length {vals.shape[0]}, expected {npts}"
            )

        vals = vals.reshape((nz, ny, nx, 3))
        data[name] = vals

        for c, cname in enumerate(component_names):
            data[cname] = vals[..., c]

    read_scalar("rho")
    read_scalar("pressure", alias="pg")
    read_scalar("energy", alias="E")
    read_scalar("v2")
    read_scalar("B2")
    read_scalar("divB")

    read_vector("velocity", ["vx", "vy", "vz"])
    read_vector("magnetic_field", ["Bx", "By", "Bz"])
    read_vector("momentum", ["Mx", "My", "Mz"])

    # Convenience derived variables.
    if "v2" in data:
        data["vabs"] = np.sqrt(np.maximum(data["v2"], 0.0))
    elif all(k in data for k in ["vx", "vy", "vz"]):
        data["vabs"] = np.sqrt(data["vx"]**2 + data["vy"]**2 + data["vz"]**2)

    if "B2" in data:
        data["Babs"] = np.sqrt(np.maximum(data["B2"], 0.0))
    elif all(k in data for k in ["Bx", "By", "Bz"]):
        data["Babs"] = np.sqrt(data["Bx"]**2 + data["By"]**2 + data["Bz"]**2)

    if "divB" in data:
        data["logdivB"] = np.log10(np.abs(data["divB"]) + 1.0e-30)

    return step, time, nx, ny, nz, data


# ============================================================
# Slice extraction
# ============================================================

def extract_slice(data3d, slice_plane="xy", slice_index=None):
    """
    Convert 3D arrays with shape (nz, ny, nx) into a 2D plotting grid.

    slice_plane:
        "xy": fixed k / z index
        "xz": fixed j / y index
        "yz": fixed i / x index
    """

    x3 = data3d["x"]
    y3 = data3d["y"]
    z3 = data3d["z"]

    nz, ny, nx = x3.shape

    if slice_plane == "xy":
        if slice_index is None:
            slice_index = nz // 2

        if not (0 <= slice_index < nz):
            raise ValueError(
                f"Invalid xy slice_index={slice_index}; valid range: 0 ... {nz-1}"
            )

        grid2d = {}

        for key, arr in data3d.items():
            if not isinstance(arr, np.ndarray):
                continue

            if arr.ndim == 3:
                grid2d[key] = arr[slice_index, :, :]
            elif arr.ndim == 4:
                grid2d[key] = arr[slice_index, :, :, :]

        coord_x = grid2d["x"]
        coord_y = grid2d["y"]
        xlabel = "x"
        ylabel = "y"

    elif slice_plane == "xz":
        if slice_index is None:
            slice_index = ny // 2

        if not (0 <= slice_index < ny):
            raise ValueError(
                f"Invalid xz slice_index={slice_index}; valid range: 0 ... {ny-1}"
            )

        grid2d = {}

        for key, arr in data3d.items():
            if not isinstance(arr, np.ndarray):
                continue

            if arr.ndim == 3:
                grid2d[key] = arr[:, slice_index, :]
            elif arr.ndim == 4:
                grid2d[key] = arr[:, slice_index, :, :]

        coord_x = grid2d["x"]
        coord_y = grid2d["z"]
        xlabel = "x"
        ylabel = "z"

    elif slice_plane == "yz":
        if slice_index is None:
            slice_index = nx // 2

        if not (0 <= slice_index < nx):
            raise ValueError(
                f"Invalid yz slice_index={slice_index}; valid range: 0 ... {nx-1}"
            )

        grid2d = {}

        for key, arr in data3d.items():
            if not isinstance(arr, np.ndarray):
                continue

            if arr.ndim == 3:
                grid2d[key] = arr[:, :, slice_index]
            elif arr.ndim == 4:
                grid2d[key] = arr[:, :, slice_index, :]

        coord_x = grid2d["y"]
        coord_y = grid2d["z"]
        xlabel = "y"
        ylabel = "z"

    else:
        raise ValueError("SLICE_PLANE must be one of: 'xy', 'xz', 'yz'")

    grid2d["_plot_x"] = coord_x
    grid2d["_plot_y"] = coord_y
    grid2d["_xlabel"] = xlabel
    grid2d["_ylabel"] = ylabel
    grid2d["_slice_plane"] = slice_plane
    grid2d["_slice_index"] = slice_index

    return grid2d


# ============================================================
# Plot helpers
# ============================================================

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
        "pressure": r"$p_g$",
        "E": r"$E$",
        "energy": r"$E$",
        "vabs": r"$|\mathbf{v}|$",
        "Babs": r"$|\mathbf{B}|$",
        "v2": r"$|\mathbf{v}|^2$",
        "B2": r"$|\mathbf{B}|^2$",
        "vx": r"$v_x$",
        "vy": r"$v_y$",
        "vz": r"$v_z$",
        "Bx": r"$B_x$",
        "By": r"$B_y$",
        "Bz": r"$B_z$",
        "Mx": r"$M_x$",
        "My": r"$M_y$",
        "Mz": r"$M_z$",
        "divB": r"$\nabla\cdot\mathbf{B}$",
        "logdivB": r"$\log_{10}(|\nabla\cdot\mathbf{B}|)$",
    }

    return labels.get(var, var)


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

    x = grid["_plot_x"]
    y = grid["_plot_y"]

    for ax, var in zip(axes, vars_to_plot):
        if var not in grid:
            available = [k for k in grid.keys() if not k.startswith("_")]
            raise KeyError(
                f"Variable '{var}' not found.\n"
                f"Available variables: {available}"
            )

        arr = grid[var]
        vmin, vmax = get_color_limits(arr)

        im = ax.pcolormesh(
            x,
            y,
            arr,
            shading="auto",
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        ax.set_aspect("equal")
        ax.set_xlabel(grid["_xlabel"])
        ax.set_ylabel(grid["_ylabel"])
        ax.set_title(pretty_label(var), pad=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label(pretty_label(var))

    for ax in axes[nvar:]:
        ax.axis("off")

    basename = os.path.basename(filename)

    step_text = "unknown" if step is None else f"{step:06d}"
    time_text = "unknown" if time is None else f"{time:.6e}"

    fig.suptitle(
        f"HOW-MHD VTS dump: {basename}   "
        f"step={step_text}   time={time_text}   "
        f"{grid['_slice_plane']} slice index={grid['_slice_index']}",
        fontsize=15,
    )

    if SAVE_FIGURE:
        step_for_name = 0 if step is None else step
        outname = os.path.join(FIGURE_DIR, f"dump2d_{step_for_name:06d}.png")
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

step, time, nx, ny, nz, data3d = read_vts_dump(filename)

print(f"step = {step}")

if time is not None:
    print(f"time = {time:.15e}")
else:
    print("time = None")

print(f"nx = {nx}, ny = {ny}, nz = {nz}")

available_vars = [k for k in data3d.keys() if not k.startswith("_")]
print("Available variables:")
print("  " + ", ".join(available_vars))

if "rho" in data3d:
    print(f"rho min/max  = {data3d['rho'].min():.6e}, {data3d['rho'].max():.6e}")

if "pg" in data3d:
    print(f"pg  min/max  = {data3d['pg'].min():.6e}, {data3d['pg'].max():.6e}")

if "vabs" in data3d:
    print(f"|v| min/max  = {data3d['vabs'].min():.6e}, {data3d['vabs'].max():.6e}")

if "Babs" in data3d:
    print(f"|B| min/max  = {data3d['Babs'].min():.6e}, {data3d['Babs'].max():.6e}")

if "divB" in data3d:
    print(f"divB min/max = {data3d['divB'].min():.6e}, {data3d['divB'].max():.6e}")
    print(f"max |divB|   = {np.max(np.abs(data3d['divB'])):.6e}")

grid = extract_slice(
    data3d,
    slice_plane=SLICE_PLANE,
    slice_index=SLICE_INDEX,
)

fig, axes = plot_2d(grid, PLOT_VARS, step, time, filename)

if SHOW_FIGURE:
    plt.show()