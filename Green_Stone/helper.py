#!/usr/bin/env python3
# helper.py
import os
import numpy as np
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree

# -------- knobs (match your working script) --------
MC_STEP       = 2          # marching_cubes stride
N_LATTICE     = 40         # UV control-grid resolution (n x n)
MIN_MC_POINTS = 50         # skip tiny/failed extractions
SAVE_NPZ      = True       # also dump NPZ for debugging (optional)
# ---------------------------------------------------

def _make_uv_grid_knn(V: np.ndarray, n: int, k: int = 8, eps: float = 1e-12):
    """Same PCA + kNN lattice you used downstream."""
    V = np.asarray(V, float)
    C = V.mean(axis=0); X = V - C
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    e0, e1 = VT[0], VT[1]
    uv = np.c_[X @ e0, X @ e1]

    umin, vmin = uv.min(axis=0); umax, vmax = uv.max(axis=0)
    u = np.linspace(umin, umax, n); v = np.linspace(vmin, vmax, n)
    UU, VV = np.meshgrid(u, v, indexing="ij")
    P = np.c_[UU.ravel(), VV.ravel()]
    tree = cKDTree(uv)
    kk = min(int(k), len(uv))
    if kk <= 1:
        _, idx = tree.query(P, k=1)
        Vgrid = V[idx]
    else:
        d, idx = tree.query(P, k=kk)
        if kk == 1:
            d = d[:, None]; idx = idx[:, None]
        w = 1.0 / (d + eps)
        w = (w.T / np.sum(w, axis=1)).T
        Vgrid = np.sum(w[:, :, None] * V[idx], axis=1)
    return Vgrid.reshape(n, n, 3)

def _try_get_active_mask(geo_model, total_nodes: int, n_cols_sfm: int):
    """
    Try common places GemPy stores an active mask. Return a boolean mask
    with length == total_nodes whose True-count equals n_cols_sfm, or None.
    """
    candidates = []
    for attr in ("active_mask", "mask", "_mask"):
        if hasattr(geo_model.grid, attr):
            candidates.append(getattr(geo_model.grid, attr))
    # sometimes on the regular grid object
    if hasattr(geo_model.grid, "regular_grid"):
        for attr in ("active", "mask", "active_mask"):
            if hasattr(geo_model.grid.regular_grid, attr):
                candidates.append(getattr(geo_model.grid.regular_grid, attr))

    for m in candidates:
        try:
            arr = np.asarray(m).ravel()
            if arr.dtype != bool:  # some stores as 0/1
                arr = arr.astype(bool)
            if arr.size == total_nodes and int(arr.sum()) == int(n_cols_sfm):
                return arr
        except Exception:
            continue
    return None

def _build_full_axes_from_regular_grid(geo_model):
    """
    Prefer regular_grid extent+resolution to build strict axes.
    """
    rg = geo_model.grid.regular_grid
    nx, ny, nz = map(int, rg.resolution)
    xmin, xmax, ymin, ymax, zmin, zmax = rg.extent
    xs = np.linspace(xmin, xmax, nx) if nx > 1 else np.array([xmin], float)
    ys = np.linspace(ymin, ymax, ny) if ny > 1 else np.array([ymin], float)
    zs = np.linspace(zmin, zmax, nz) if nz > 1 else np.array([zmin], float)
    return xs, ys, zs

def export_fault_grids_and_ids_txt(
    geo_model,
    out_dir: str = "gmsh_io",
    num_faults: int = 8,
    n_lattice: int = N_LATTICE,
    mc_step: int = MC_STEP,
    also_write_npz: bool = SAVE_NPZ,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) Pull grid & solutions ----
    grid_vals = np.asarray(geo_model.grid.values, float)  # (N_all, 3) possibly full grid
    X_all = grid_vals[:, 0]; Y_all = grid_vals[:, 1]; Z_all = grid_vals[:, 2]

    # scalar fields
    try:
        sfm = np.asarray(geo_model.solutions.raw_arrays.scalar_field_matrix)
    except Exception:
        sfm = np.asarray(geo_model.solutions._raw_arrays.scalar_field_matrix)
    nFields, N_cols = sfm.shape

    # level sets (for faults)
    try:
        ls = np.asarray(geo_model.solutions.scalar_field_at_surface_points).ravel()
    except Exception:
        try:
            ls = np.asarray(geo_model.solutions.level_set_s1).ravel()
        except Exception:
            ls = np.asarray(geo_model.solutions._raw_arrays.level_set_s1).ravel()

    # regular-grid axes (authoritative for spacing/origin & box)
    xs, ys, zs = _build_full_axes_from_regular_grid(geo_model)
    Nx, Ny, Nz = xs.size, ys.size, zs.size
    total_nodes = Nx * Ny * Nz

    dx = float(xs[1] - xs[0]) if Nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if Ny > 1 else 1.0
    dz = float(zs[1] - zs[0]) if Nz > 1 else 1.0
    x0, y0, z0 = float(xs[0]), float(ys[0]), float(zs[0])

    Xmin, Xmax = float(xs.min()), float(xs.max())
    Ymin, Ymax = float(ys.min()), float(ys.max())
    Zmin, Zmax = float(zs.min()), float(zs.max())

    # ---- 2) Build full-grid IDs in strict C-order (robust & consistent) ----
    # Use lith_block for IDs; it should correspond to the full grid.
    try:
        lith = np.asarray(geo_model.solutions.raw_arrays.lith_block)
    except Exception:
        lith = np.asarray(geo_model.solutions._raw_arrays.lith_block)

    ids_3d = np.empty((Nx, Ny, Nz), dtype=np.int64)
    ids_3d.fill(-1)

    # If lith_block matches total_nodes, reshape directly (fast path).
    if lith.size == total_nodes:
        # GemPy stores lith_block flattened. Empirically, Fortran-column major reshape matches
        # GemPy → use order="F" into (Nx,Ny,Nz), then ravel in C-order when writing.
        ids_3d[:, :, :] = np.round(lith).astype(np.int64).reshape((Nx, Ny, Nz), order="C")
    else:
        # Fallback: map each geo_model.grid.values row to its voxel via nearest axis.
        # (covers cases where grid.values contains only active nodes)
        # Make a KDTree over full-grid points for robust matching.
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
        full_pts = np.column_stack([Xg.ravel(order="C"),
                                    Yg.ravel(order="C"),
                                    Zg.ravel(order="C")])
        tree_full = cKDTree(full_pts)
        # grid_vals may be only active points; match them and fill IDs
        _, nearest = tree_full.query(np.column_stack([X_all, Y_all, Z_all]), k=1)
        ids_flat_tmp = np.round(lith).astype(np.int64)
        if ids_flat_tmp.size != nearest.size:
            raise ValueError(f"lith_block size {ids_flat_tmp.size} != grid.values size {nearest.size}; "
                             f"cannot align IDs.")
        ids_full = np.full(total_nodes, -1, dtype=np.int64)
        ids_full[nearest] = ids_flat_tmp
        ids_3d[:, :, :] = ids_full.reshape((Nx, Ny, Nz), order="C")

    if (ids_3d < 0).any():
        missing = int((ids_3d < 0).sum())
        raise RuntimeError(f"{missing} grid nodes were not assigned an ID — check grid/mask consistency.")

    # Write IDs table (X Y Z ID) in strict C-order (no header)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
    table = np.column_stack([
        Xg.ravel(order="C"),
        Yg.ravel(order="C"),
        Zg.ravel(order="C"),
        ids_3d.ravel(order="C"),
    ])
    np.savetxt(os.path.join(out_dir, "ids_points.txt"), table, fmt="%.9g\t%.9g\t%.9g\t%d")
    np.savetxt(os.path.join(out_dir, "bbox.txt"),
               np.array([[Xmin, Xmax, Ymin, Ymax, Zmin, Zmax]]),
               fmt="%.9g", delimiter="\t")
    np.savetxt(os.path.join(out_dir, "resolution.txt"),
               np.array([[Nx, Ny, Nz]], dtype=np.int64),
               fmt="%d", delimiter="\t")

    # ---- 3) Prepare coordinates for NPZ / fault extraction when shapes mismatch ----
    # If scalar_field_matrix columns != total_nodes, try to extract the ACTIVE mask
    # so that X/Y/Z we save alongside sfm have matching length/order.
    if N_cols == total_nodes:
        # full grid case
        X_for_sfm = Xg.ravel(order="F")  # to match Fortran-like flatten often used by GemPy
        Y_for_sfm = Yg.ravel(order="F")
        Z_for_sfm = Zg.ravel(order="F")
    else:
        mask = _try_get_active_mask(geo_model, total_nodes, N_cols)
        if mask is None:
            # best-effort: fall back to current geo_model.grid.values order
            # (length must equal N_cols, otherwise raise)
            if grid_vals.shape[0] != N_cols:
                raise ValueError(
                    f"scalar_field_matrix has {N_cols} columns but cannot find a matching "
                    f"active-mask; grid.values has {grid_vals.shape[0]} rows."
                )
            X_for_sfm, Y_for_sfm, Z_for_sfm = X_all, Y_all, Z_all
        else:
            # map mask (full grid, C-order) → active coords in the same order
            X_for_sfm = Xg.ravel(order="C")[mask.ravel(order="C")]
            Y_for_sfm = Yg.ravel(order="C")[mask.ravel(order="C")]
            Z_for_sfm = Zg.ravel(order="C")[mask.ravel(order="C")]

    if also_write_npz:
        np.savez(
            os.path.join(out_dir, "gempy_scalar_field_data.npz"),
            X=X_for_sfm, Y=Y_for_sfm, Z=Z_for_sfm,
            scalar_field_matrix=sfm,
            level_set_s1=ls
        )

    # ---- 4) Fault surfaces (exact pipeline you validated) ----
    used = []
    # first num_faults entries of scalar fields and level-sets are faults;
    # the last field is sediments (ignored here).
    n_faults_eff = min(int(num_faults), nFields - 1, len(ls))

    for f_idx in range(n_faults_eff):
        level_val = float(ls[f_idx])

        # reshape row -> (Nx,Ny,Nz), then transpose to (Nz,Ny,Nx) for marching_cubes
        try:
            vol_xyz = np.asarray(sfm[f_idx, :], float).reshape((Nx, Ny, Nz))
        except Exception as e:
            raise ValueError(
                f"Cannot reshape scalar_field_matrix row {f_idx} into "
                f"(Nx,Ny,Nz)=({Nx},{Ny},{Nz}). "
                f"Got N_cols={N_cols}. Original error: {e}"
            )
        vol_zyx = np.transpose(vol_xyz, (2, 1, 0))

        try:
            Vz, _, _, _ = marching_cubes(
                vol_zyx, level=level_val, spacing=(dz, dy, dx), step_size=mc_step
            )
        except Exception as e:
            print(f"[Skip] fault {f_idx}: marching_cubes failed ({e})")
            continue

        # (z,y,x) -> world (x,y,z)
        V = np.c_[Vz[:, 2] + x0, Vz[:, 1] + y0, Vz[:, 0] + z0]
        if V.shape[0] < MIN_MC_POINTS:
            print(f"[Skip] fault {f_idx}: too few verts ({V.shape[0]})")
            continue

        Vgrid = _make_uv_grid_knn(V, n_lattice, k=8)

        # TXT: headerless XYZ, written row-wise: for j in 0..nV-1, for i in 0..nU-1
        nU, nV, _ = Vgrid.shape
        fout = os.path.join(out_dir, f"fault_{f_idx+1:03d}.txt")
        with open(fout, "w") as f:
            for j in range(nV):
                for i in range(nU):
                    x, y, z = Vgrid[i, j, :]
                    f.write(f"{x:.9g}\t{y:.9g}\t{z:.9g}\n")
        print(f"[OK] {fout}  ({nU}x{nV})")
        used.append(os.path.basename(fout))

    # manifest (no header)
    with open(os.path.join(out_dir, "faults_manifest.txt"), "w") as f:
        for name in used:
            f.write(f"{name}\n")

    return {
        "out_dir": out_dir,
        "fault_files": used,
        "ids_points": "ids_points.txt",
        "bbox": "bbox.txt",
        "resolution": "resolution.txt",
        "n_lattice": n_lattice,
        "count_faults": len(used),
    }
