#!/usr/bin/env python3
# File: build_model_from_txt_direct_grid.py
import os
import glob
import numpy as np
import multiprocessing as mp
from scipy.spatial import cKDTree
import gmsh
from GmshBSplineSurfaceBuilder import GmshBSplineSurfaceBuilder

# ---------------- user knobs ----------------
FAULTS_DIR        = "gmsh_io"
FAULTS_MANIFEST   = "faults_manifest.txt"
IDS_POINTS_TXT    = "ids_points.txt"

#  Set to 0 to force "no faults" mode (skips all fault logic)
MAX_FAULTS        = None

THREADS           = max(1, mp.cpu_count() - 1)

# far-field size
UNIFORM_LC        = None
UNIFORM_LC_FRAC   = 0.02

# refinement near all fault faces
FAULT_LC_FRAC           = 0.01
FAULT_REF_DIST_FRAC     = 0.05
FAULT_REF_BLEND_FACTOR  = 2.0

# builder
BUILDER_N_LATTICE = 40
BUILDER_DEG_U     = 2
BUILDER_DEG_V     = 2
BUILDER_SMOOTHING = 1e1
BUILDER_PAD_FRAC  = 0.2
BUILDER_MAX_CTRL  = 25
BUILDER_VERBOSE   = True

OUT_MSH           = "Perth_Basin.msh"
OPEN_GUI          = True
# -------------------------------------------

def list_fault_files(base_dir, manifest="faults_manifest.txt", max_faults=None):
    """
    Return absolute paths to fault_*.txt files, in the order of the manifest
    (or glob order if the manifest is missing).
    If max_faults == 0, return [] immediately.
    """
    if max_faults is not None and int(max_faults) == 0:
        return []

    man_path = os.path.join(base_dir, manifest)
    if os.path.isfile(man_path):
        with open(man_path, "r") as f:
            files = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        fault_files = [os.path.join(base_dir, fn) for fn in files]
    else:
        fault_files = sorted(glob.glob(os.path.join(base_dir, "fault_*.txt")))

    if max_faults is not None and int(max_faults) > 0:
        fault_files = fault_files[:int(max_faults)]
    return fault_files

def load_ids_points(ids_txt_path):
    if not os.path.isfile(ids_txt_path):
        raise FileNotFoundError(f"IDs text file not found: {ids_txt_path}")
    data = np.loadtxt(ids_txt_path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f"{ids_txt_path} must have 4 columns: X Y Z ID")
    XYZ = np.asarray(data[:, 0:3], float)
    ids = np.asarray(np.round(data[:, 3]).astype(np.int64))
    return XYZ, ids

# ===================== Main =====================
if __name__ == "__main__":
    # IDs & bbox
    ids_XYZ, ids_flat = load_ids_points(os.path.join(FAULTS_DIR, IDS_POINTS_TXT))
    Xmin, Ymin, Zmin = ids_XYZ.min(axis=0)
    Xmax, Ymax, Zmax = ids_XYZ.max(axis=0)
    bbox_diag = float(np.linalg.norm([Xmax - Xmin, Ymax - Ymin, Zmax - Zmin]))

    # Fault files (may be empty if MAX_FAULTS == 0 or manifest/glob empty)
    fault_files = list_fault_files(FAULTS_DIR, FAULTS_MANIFEST, max_faults=MAX_FAULTS)

    # Gmsh init
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", THREADS)
    gmsh.model.add("faults_and_layers_from_txt_builder")

    fault_surfs_original = {}   # name -> surface tag

    # ---------- Build B-spline fault surfaces (only if we have any) ----------
    if len(fault_files) > 0:
        builder = GmshBSplineSurfaceBuilder(
            model_name="faults_and_layers_from_txt_builder",
            n_lattice=BUILDER_N_LATTICE,
            deg_u=BUILDER_DEG_U,
            deg_v=BUILDER_DEG_V,
            smoothing=BUILDER_SMOOTHING,
            pad_frac=BUILDER_PAD_FRAC,
            max_ctrl=BUILDER_MAX_CTRL,
            verbose=BUILDER_VERBOSE
        )

        for fpath in fault_files:
            name = os.path.splitext(os.path.basename(fpath))[0]
            try:
                V = np.loadtxt(fpath)
                if V.ndim == 1:
                    V = V[None, :]
                if V.shape[1] < 3:
                    print(f"[Skip] {name}: need 3 columns (X Y Z)")
                    continue
            except Exception as e:
                print(f"[Skip] {name}: failed to read points ({e})")
                continue

            try:
                sTag = builder.build_from_points(np.asarray(V[:, :3], float))
            except Exception as e:
                print(f"[Skip] {name}: builder failed ({e})")
                continue

            fault_surfs_original[name] = sTag
            print(f"[OK] {name} -> BSpline tag={sTag}")

    gmsh.model.occ.synchronize()

    # ---------- Box (always) ----------
    Xlen, Ylen, Zlen = Xmax - Xmin, Ymax - Ymin, Zmax - Zmin
    box_tag = gmsh.model.occ.addBox(float(Xmin), float(Ymin), float(Zmin),
                                    float(Xlen), float(Ylen), float(Zlen))
    gmsh.model.occ.synchronize()

    # ---------- Fragment (only if we actually have fault surfaces) ----------
    box_faces = gmsh.model.getBoundary([(3, box_tag)], oriented=False)  # list of (2, tag)

    have_faults = len(fault_surfs_original) > 0
    if have_faults:
        tool_surfs = [(2, t) for t in fault_surfs_original.values()]
        objectDimTags = [(3, box_tag)] + box_faces

        outDimTags, outDimTagsMap = gmsh.model.occ.fragment(objectDimTags, tool_surfs)
        gmsh.model.occ.synchronize()

        # Keep only faces bounding volumes
        all_surfaces = {tag for dim, tag in gmsh.model.getEntities(2)}
        keep = set()
        for _, vtag in gmsh.model.getEntities(3):
            for dB, tB in gmsh.model.getBoundary([(3, vtag)], oriented=False):
                if dB == 2:
                    keep.add(tB)
        for tag in (all_surfaces - keep):
            gmsh.model.occ.remove([(2, tag)], recursive=True)
        gmsh.model.occ.synchronize()

        # Map fault tools -> all their child faces that survived (for naming & refinement)
        faces_now = {tag for dim, tag in gmsh.model.getEntities(2)}
        offset_tools = len(objectDimTags)
        fault_child_faces = []
        for k, _ in enumerate(tool_surfs):
            i_map = offset_tools + k
            if i_map >= len(outDimTagsMap):
                continue
            children = outDimTagsMap[i_map]
            for (d, t) in children:
                if d == 2 and t in faces_now:
                    fault_child_faces.append(t)

        # ---------- Name planes (fault planes + box boundaries) ----------
        assigned = set()

        # (A) Fault planes
        plane_idx = 1
        for k, _ in enumerate(tool_surfs):
            i_map = offset_tools + k
            if i_map >= len(outDimTagsMap):
                continue
            children = outDimTagsMap[i_map]
            group = [t for (d, t) in children if d == 2 and t in faces_now and t not in assigned]
            if not group:
                continue
            pg = gmsh.model.addPhysicalGroup(2, group)
            gmsh.model.setPhysicalName(2, pg, f"plane {plane_idx}")
            assigned.update(group)
            plane_idx += 1

        # (B) Box boundary planes
        bc_idx = 1
        for j in range(1, 1 + len(box_faces)):
            if j >= len(outDimTagsMap):
                continue
            children = outDimTagsMap[j]
            group = [t for (d, t) in children if d == 2 and t in faces_now and t not in assigned]
            if not group:
                continue
            pg = gmsh.model.addPhysicalGroup(2, group)
            gmsh.model.setPhysicalName(2, pg, f"bc{bc_idx}")
            assigned.update(group)
            bc_idx += 1

        # (C) any remaining faces -> plane N
        remaining = [t for t in faces_now if t not in assigned]
        for t in remaining:
            pg = gmsh.model.addPhysicalGroup(2, [t])
            gmsh.model.setPhysicalName(2, pg, f"plane {plane_idx}")
            plane_idx += 1

        gmsh.model.occ.synchronize()

    else:
        # No faults: just name the 6 box faces as bc1..bc6
        # (no fragment, no fault planes, nothing else to clean)
        faces_now = [t for d, t in box_faces if d == 2]
        for i, t in enumerate(faces_now, start=1):
            pg = gmsh.model.addPhysicalGroup(2, [t])
            gmsh.model.setPhysicalName(2, pg, f"bc{i}")
        gmsh.model.occ.synchronize()
        fault_child_faces = []  # empty for later logic

    # ----------- Mesh (refine near faults only if they exist) -----------
    lc_far  = UNIFORM_LC if UNIFORM_LC is not None else float(bbox_diag * UNIFORM_LC_FRAC)
    lc_near = float(bbox_diag * FAULT_LC_FRAC)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(lc_near, lc_far))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_far)

    if have_faults and len(fault_child_faces) > 0:
        # Distance/Threshold field for all child faces from all faults
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", fault_child_faces)

        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMin", lc_near)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMax", lc_far)

        dist_min = float(bbox_diag * FAULT_REF_DIST_FRAC)
        dist_max = float(FAULT_REF_BLEND_FACTOR) * dist_min
        gmsh.model.mesh.field.setNumber(f_th, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(f_th, "DistMax", dist_max)

        gmsh.model.mesh.field.setAsBackgroundMesh(f_th)
    else:
        # No faults: uniform mesh
        pass

    gmsh.model.mesh.generate(3)

    # ----------- Sedimentary tagging via IDs (direct C-order) -----------
    tree_ids = cKDTree(ids_XYZ)

    etags, ntags = gmsh.model.mesh.getElementsByType(4)  # TET4
    etags = np.asarray(etags, dtype=np.int64)
    if etags.size == 0:
        print("[Info] No TET4 elements — skipping sedimentary tagging")
    else:
        ntags = np.asarray(ntags, dtype=np.int64).reshape(-1, 4)

        node_ids, node_coords, _ = gmsh.model.mesh.getNodes()
        node_ids = np.asarray(node_ids, dtype=np.int64)
        node_coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        order = np.argsort(node_ids)
        node_ids_sorted = node_ids[order]
        node_coords_sorted = node_coords[order]

        idxs = np.searchsorted(node_ids_sorted, ntags)
        coords_e = node_coords_sorted[idxs]
        centroids = coords_e.mean(axis=1)

        _, nearest = tree_ids.query(centroids, k=1)
        elem_ids = ids_flat[nearest]

        uniq = np.unique(elem_ids)
        print("[INFO] integer sediment IDs found:", uniq)

        for uid in uniq:
            mask = (elem_ids == uid)
            if not np.any(mask):
                continue
            etags_lab = etags[mask]
            ntags_lab = ntags[mask]
            flat_nodes = ntags_lab.reshape(-1).tolist()

            vol_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.mesh.addElements(3, vol_tag, [4], [etags_lab.tolist()], [flat_nodes])
            pg = gmsh.model.addPhysicalGroup(3, [vol_tag])
            gmsh.model.setPhysicalName(3, pg, f"Layer_{int(uid)}")
            print(f"[INFO] Layer_{int(uid)}: {etags_lab.size} elems -> volume {vol_tag}")

    # write
    gmsh.write(OUT_MSH)


    gmsh.finalize()
