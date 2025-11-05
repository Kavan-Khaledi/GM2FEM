# bspline_surface_builder_fixed_scaling.py
from __future__ import annotations
import os
import numpy as np
from typing import Tuple, Optional

from scipy.spatial import Delaunay, cKDTree
try:
    from scipy.interpolate import RBFInterpolator
    _HAVE_RBFI = True
except Exception:
    from scipy.interpolate import Rbf
    _HAVE_RBFI = False

import gmsh



class GmshBSplineSurfaceBuilder:
    """
    Build a Gmsh OCC BSplineSurface from scattered 3D points of a single surface.
    """

    def __init__(
        self,
        model_name: str = "bspline_single_surface",
        n_lattice: int = 60,
        deg_u: int = 3,
        deg_v: int = 3,
        smoothing: float = 1e1,
        pad_frac: float = 0.0,
        max_ctrl: int = 60,
        verbose: bool = True
    ):
        self.model_name = model_name
        self.n_lattice = int(n_lattice)
        self.deg_u = int(deg_u)
        self.deg_v = int(deg_v)
        self.smoothing = float(smoothing)
        self.pad_frac = float(pad_frac)
        self.max_ctrl = int(max_ctrl)
        self.verbose = verbose

        # Will store scaling parameters for rescaling back later
        self._Vmin = None
        self._Vscale = None

        self._ensure_gmsh_model()

    # ---------------------- public API ----------------------

    def build_from_txt(self, path: str) -> int:
        V = np.loadtxt(path)
        return self.build_from_points(V)

    def build_from_points(self, V: np.ndarray) -> int:
        V = self._validate_points(V)

        # 1) UV lattice on normalized [0,1]^2 via PCA + Delaunay barycentric
        Vgrid, _ = self._make_uv_grid_normalized(V, self.n_lattice)

        # 2) Optional RBF smoothing + expanded evaluation on [-pad, 1+pad]^2
        try:
            Vg = self._rbf_smooth_and_expand_normalized(Vgrid, self.smoothing, self.pad_frac)
        except Exception as e:
            if self.verbose:
                print(f"[Warn] RBF smoothing failed ({e}); using lattice grid directly.")
            Vg = Vgrid

        # 3) Downsample control net if too dense
        nU, nV, _ = Vg.shape
        if max(nU, nV) > self.max_ctrl:
            sU = max(1, int(np.ceil(nU / self.max_ctrl)))
            sV = max(1, int(np.ceil(nV / self.max_ctrl)))
            Vg = Vg[::sU, ::sV, :]
            nU, nV, _ = Vg.shape

        # --- NEW: rescale back to original coordinate space ---
        Vg = Vg * self._Vscale + self._Vmin

        # 4) Degrees/knots (open uniform)
        knotsU, multU, degU = self._open_uniform_knots(nU, self.deg_u)
        knotsV, multV, degV = self._open_uniform_knots(nV, self.deg_v)

        # 5) Push control points to OCC, create BSpline surface
        pts = []
        for j in range(nV):
            for i in range(nU):
                x, y, z = map(float, Vg[i, j, :])
                pts.append(gmsh.model.occ.addPoint(x, y, z, 0.0))

        try:
            sTag = gmsh.model.occ.addBSplineSurface(
                pts, nU, -1, degU, degV,
                [], knotsU, knotsV, multU, multV, [], False
            )
        except Exception as e:
            gmsh.model.occ.remove([(0, t) for t in pts], recursive=False)
            gmsh.model.occ.synchronize()
            raise RuntimeError(f"addBSplineSurface failed: {e}")

        gmsh.model.occ.synchronize()
        gmsh.model.occ.remove([(0, t) for t in pts], recursive=False)
        gmsh.model.occ.synchronize()

        if self.verbose:
            print(f"[OK] BSpline tag={sTag}  deg {degU}x{degV}, ctrl {nU}x{nV}")

        return sTag

    def write_step(self, path: str) -> None:
        gmsh.write(os.path.abspath(path))
        if self.verbose:
            print(f"[Write] STEP: {os.path.abspath(path)}")

    # --------------------- helpers/internal ---------------------

    def _ensure_gmsh_model(self) -> None:
        if not gmsh.isInitialized():
            gmsh.initialize()
        current = gmsh.model.getCurrent()
        if current != self.model_name:
            gmsh.model.add(self.model_name)

    def _validate_points(self, V: np.ndarray) -> np.ndarray:
        """
        Normalize to [0,1]^3 for stable fitting, store scaling for later rescale.
        """
        V = np.asarray(V, dtype=float)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("Input points must be an (N,3) array.")

        Vmin = V.min(axis=0)
        Vmax = V.max(axis=0)
        scale = Vmax - Vmin
        scale[scale == 0] = 1.0  # avoid div by zero

        self._Vmin = Vmin
        self._Vscale = scale

        Vn = (V - Vmin) / scale
        return Vn

    @staticmethod
    def _open_uniform_knots(nc: int, deg: int) -> Tuple[list, list, int]:
        if nc < deg + 1:
            deg = max(1, min(deg, nc - 1))
        n_int = nc - deg - 1
        if n_int <= 0:
            return [0.0, 1.0], [deg + 1, deg + 1], deg
        interior = [(i / (n_int + 1)) for i in range(1, n_int + 1)]
        return [0.0] + interior + [1.0], [deg + 1] + [1] * n_int + [deg + 1], deg

    @staticmethod
    def _make_uv_grid_normalized(V: np.ndarray, n: int) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        C = V.mean(axis=0)
        X = V - C
        _, _, VT = np.linalg.svd(X, full_matrices=False)
        e0, e1 = VT[0], VT[1]

        uv = np.column_stack([X @ e0, X @ e1])
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        du = max(umax - umin, 1e-12)
        dv = max(vmax - vmin, 1e-12)
        uvn = np.column_stack([(uv[:, 0] - umin) / du, (uv[:, 1] - vmin) / dv])

        tri = Delaunay(uvn)

        u_lin = np.linspace(0.0, 1.0, n)
        v_lin = np.linspace(0.0, 1.0, n)
        UU, VV = np.meshgrid(u_lin, v_lin, indexing="ij")
        P = np.column_stack([UU.ravel(), VV.ravel()])
        simplex = tri.find_simplex(P)

        Vgrid = np.empty((P.shape[0], 3), float)

        outside = np.where(simplex < 0)[0]
        if outside.size:
            nn = cKDTree(uvn).query(P[outside], k=1)[1]
            Vgrid[outside] = V[nn]

        inside = np.where(simplex >= 0)[0]
        if inside.size:
            T = tri.simplices[simplex[inside]]
            A = uvn[T[:, 0]]
            B = uvn[T[:, 1]]
            D = uvn[T[:, 2]]
            PA = P[inside] - A
            BA = B - A
            DA = D - A
            den = BA[:, 0] * DA[:, 1] - BA[:, 1] * DA[:, 0]
            m = np.abs(den) > 1e-14
            w1 = np.zeros(len(inside))
            w2 = np.zeros(len(inside))
            w1[m] = (PA[m, 0] * DA[m, 1] - PA[m, 1] * DA[m, 0]) / den[m]
            w2[m] = (-PA[m, 0] * BA[m, 1] + PA[m, 1] * BA[m, 0]) / den[m]
            w0 = 1.0 - w1 - w2

            V0 = V[T[:, 0]]
            V1 = V[T[:, 1]]
            V2 = V[T[:, 2]]
            Vgrid[inside] = (w0[:, None] * V0 + w1[:, None] * V1 + w2[:, None] * V2)

        return Vgrid.reshape(n, n, 3), (0.0, 1.0, 0.0, 1.0)

    @staticmethod
    def _rbf_smooth_and_expand_normalized(
        Vgrid: np.ndarray,
        smoothing: float,
        pad_frac: float,
        eval_n: Optional[int] = None
    ) -> np.ndarray:
        n = Vgrid.shape[0]
        if eval_n is None:
            eval_n = n

        u_lin = np.linspace(0.0, 1.0, n)
        v_lin = np.linspace(0.0, 1.0, n)
        UU, VV = np.meshgrid(u_lin, v_lin, indexing="ij")
        P = np.column_stack([UU.ravel(), VV.ravel()])
        vals = Vgrid.reshape(-1, 3)

        u_eval = np.linspace(-pad_frac, 1.0 + pad_frac, eval_n)
        v_eval = np.linspace(-pad_frac, 1.0 + pad_frac, eval_n)
        Ue, Ve = np.meshgrid(u_eval, v_eval, indexing="ij")
        Pe = np.column_stack([Ue.ravel(), Ve.ravel()])

        if _HAVE_RBFI:
            rbf = RBFInterpolator(P, vals, kernel="thin_plate_spline", smoothing=smoothing)
            Veval = rbf(Pe)
        else:
            x = vals[:, 0]; y = vals[:, 1]; z = vals[:, 2]
            rbf_x = Rbf(P[:, 0], P[:, 1], x, function="thin_plate", smooth=smoothing)
            rbf_y = Rbf(P[:, 0], P[:, 1], y, function="thin_plate", smooth=smoothing)
            rbf_z = Rbf(P[:, 0], P[:, 1], z, function="thin_plate", smooth=smoothing)
            Veval = np.column_stack([
                rbf_x(Pe[:, 0], Pe[:, 1]),
                rbf_y(Pe[:, 0], Pe[:, 1]),
                rbf_z(Pe[:, 0], Pe[:, 1])
            ])

        return Veval.reshape(eval_n, eval_n, 3)



