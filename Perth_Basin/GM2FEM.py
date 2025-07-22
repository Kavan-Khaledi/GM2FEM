import gmsh
from scipy.interpolate import RBFInterpolator
import numpy as np
from collections import defaultdict

# -----------------------------
# Load data
# -----------------------------
input_path = "gempy_gmesh_input/gempy_gmesh_surface_vertices.npz"
data = np.load(input_path, allow_pickle=True)
back_transformed_vertices_list = data["back_transformed_vertices"].tolist()

gmsh.initialize()
gmsh.model.add("gempy_gmsh")

# -----------------------------
# Model bounds and mesh size
# -----------------------------
xmin, ymin, zmin = 337000, 6640000, -12000
xmax, ymax, zmax = 400000, 6710000, 1000
model_bounds = {'min': [xmin, ymin, zmin], 'max': [xmax, ymax, zmax]}

dx = xmax - xmin
dy = ymax - ymin
dz = zmax - zmin
meshSizeMax = ((dx**2 + dy**2 + dz**2)**0.5) / 60
meshSizeMin = ((dx**2 + dy**2 + dz**2)**0.5) / 120

# -----------------------------
# Subsampling utilities
# -----------------------------
def normalize_points(points, bounds):
    min_bounds = np.array(bounds['min'])
    max_bounds = np.array(bounds['max'])
    return (points - min_bounds) / (max_bounds - min_bounds)

def grid_subsample(points, max_points, dims):
    min_bounds = points[:, dims].min(axis=0)
    max_bounds = points[:, dims].max(axis=0)
    volume = np.prod(max_bounds - min_bounds)
    cell_volume = volume / max_points
    cell_size = np.cbrt(cell_volume)
    grid_indices = np.floor((points[:, dims] - min_bounds) / cell_size).astype(int)
    cell_points = defaultdict(list)
    for idx, point in zip(map(tuple, grid_indices), points):
        cell_points[idx].append(point)
    return np.array([np.mean(pts, axis=0) for pts in cell_points.values()])

def select_best_grid(points, max_points, bounds):
    xy = grid_subsample(points, max_points, [0, 1])
    yz = grid_subsample(points, max_points, [1, 2])
    xz = grid_subsample(points, max_points, [0, 2])
    var_xy = np.var(normalize_points(xy, bounds)[:, 2])
    var_yz = np.var(normalize_points(yz, bounds)[:, 0])
    var_xz = np.var(normalize_points(xz, bounds)[:, 1])
    best = min({'xy': var_xy, 'yz': var_yz, 'xz': var_xz}, key=lambda k: {'xy': var_xy, 'yz': var_yz, 'xz': var_xz}[k])
    return {'xy': (xy, 'xy'), 'yz': (yz, 'yz'), 'xz': (xz, 'xz')}[best]

# -----------------------------
# Surface creation
# -----------------------------
grid_res = 200
rbf_models = []
surface_info = []  # will store tuples (surf_tag, surface_name, orientation)

for idx, points in enumerate(back_transformed_vertices_list):
    max_points = 10000
    points_subsampled, best_plane = select_best_grid(points, max_points, model_bounds)

    if best_plane == 'xy':
        inputs = points_subsampled[:, :2]
        targets = points_subsampled[:, 2]
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid_res),
                             np.linspace(ymin, ymax, grid_res))
        grid_points = np.column_stack((gx.ravel(), gy.ravel()))
    elif best_plane == 'yz':
        inputs = points_subsampled[:, 1:]
        targets = points_subsampled[:, 0]
        gy, gz = np.meshgrid(np.linspace(ymin, ymax, grid_res),
                             np.linspace(zmin, zmax, grid_res))
        grid_points = np.column_stack((gy.ravel(), gz.ravel()))
    else:
        inputs = points_subsampled[:, [0, 2]]
        targets = points_subsampled[:, 1]
        gx, gz = np.meshgrid(np.linspace(xmin, xmax, grid_res),
                             np.linspace(zmin, zmax, grid_res))
        grid_points = np.column_stack((gx.ravel(), gz.ravel()))

    rbf = RBFInterpolator(inputs, targets, kernel='multiquadric', epsilon=1, smoothing=1e6)
    rbf_models.append(rbf)
    grid_values = rbf(grid_points)

    if best_plane == 'xy':
        pts_surface = np.column_stack((grid_points[:, 0], grid_points[:, 1], grid_values))
    elif best_plane == 'yz':
        pts_surface = np.column_stack((grid_values, grid_points[:, 0], grid_points[:, 1]))
    else:
        pts_surface = np.column_stack((grid_points[:, 0], grid_values, grid_points[:, 1]))

    point_tags = [gmsh.model.occ.addPoint(*pt) for pt in pts_surface]
    gmsh.model.occ.synchronize()

    try:
        surf_tag = gmsh.model.occ.addBSplineSurface(point_tags, numPointsU=grid_res)
        surface_name = f"Surface_{idx:02d}_{best_plane}"  # store orientation here
        surface_info.append((surf_tag, surface_name, best_plane))
    except Exception as e:
        print(f"Failed to add surface {idx}: {e}")

    gmsh.model.occ.remove([(0, tag) for tag in point_tags], recursive=True)
    gmsh.model.occ.synchronize()

# -----------------------------
# Add box, fragment, clean
# -----------------------------
box_tag = 1
gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz, box_tag)
gmsh.model.occ.synchronize()

gmsh.model.occ.fragment([(3, 1)], gmsh.model.getEntities(2))
gmsh.model.occ.synchronize()

# Clean up unused surfaces
all_surfaces = {tag for dim, tag in gmsh.model.getEntities(2)}
volumes = gmsh.model.getEntities(3)
used_surfaces = set()

for vol in volumes:
    for s_dim, s_tag in gmsh.model.getBoundary([vol], oriented=False):
        if s_dim == 2:
            used_surfaces.add(s_tag)

unused_surfaces = all_surfaces - used_surfaces
gmsh.model.occ.remove([(2, tag) for tag in unused_surfaces], recursive=True)
gmsh.model.occ.synchronize()

# -----------------------------
# Helper: project 3D points to RBF input based on orientation
# -----------------------------
def project_points_to_rbf_input(points, orientation):
    if orientation == 'xy':
        # input to RBF is (x, y)
        return points[:, :2]
    elif orientation == 'yz':
        # input to RBF is (y, z)
        return points[:, 1:]
    else:  # 'xz'
        # input to RBF is (x, z)
        return points[:, [0, 2]]

# -----------------------------
# Mesh settings and generate
# -----------------------------
gmsh.option.setNumber("Mesh.MeshSizeMax", meshSizeMax)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshSizeMin)
gmsh.option.setNumber("Mesh.Algorithm", 0)
gmsh.option.setNumber("Mesh.SmoothRatio", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)

# -----------------------------
# Helper: get mesh nodes on a gmsh surface tag
# -----------------------------
def get_surface_mesh_nodes(surface_tag):
    """
    Get the 3D coordinates of mesh nodes on the given surface.
    Returns a numpy array of shape (N, 3).
    """
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2, tag=surface_tag)
    if len(node_coords) == 0:
        return np.empty((0, 3))
    coords = np.array(node_coords).reshape(-1, 3)
    return coords

# -----------------------------
# Assign physical groups to volumes using RBF matching on boundary surfaces
# -----------------------------
# Assign volume names by lowest-z matching XY surface
# -----------------------------

grouped_volumes = defaultdict(list)

for dim, vol_tag in volumes:
    boundary = gmsh.model.getBoundary([(dim, vol_tag)], oriented=False)
    matched_xy_surfaces = []

    for s_dim, s_tag in boundary:
        if s_dim != 2:
            continue

        pts = get_surface_mesh_nodes(s_tag)
        if pts.shape[0] == 0:
            continue

        best_match = None
        lowest_error = float('inf')

        for (surf_tag, surface_name, orientation), rbf_model in zip(surface_info, rbf_models):
            if orientation != 'xy':
                continue

            try:
                inputs = project_points_to_rbf_input(pts, orientation)
                predicted = rbf_model(inputs)
                true_vals = pts[:, 2]  # z for 'xy'
                error = np.mean((predicted - true_vals)**2)
            except Exception:
                continue

            if error < lowest_error:
                lowest_error = error
                best_match = (surface_name, np.mean(pts[:, 2]))  # surface name + avg Z

        if best_match:
            matched_xy_surfaces.append(best_match)

    if matched_xy_surfaces:
        chosen_surface = min(matched_xy_surfaces, key=lambda x: x[1])  # min avg Z
        surface_name = chosen_surface[0]
        grouped_volumes[surface_name].append(vol_tag)
        print(f"Volume {vol_tag} → {surface_name}")
    else:
        print(f"Volume {vol_tag} → no matched XY surface")

# Create one physical group per surface name
for i, (name, vol_tags) in enumerate(grouped_volumes.items(), start=1):
    gmsh.model.addPhysicalGroup(3, vol_tags, i)
    gmsh.model.setPhysicalName(3, i, name)

# -----------------------------
# Mesh settings and generate
# -----------------------------
gmsh.option.setNumber("Mesh.MeshSizeMax", meshSizeMax)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshSizeMin)
gmsh.option.setNumber("Mesh.Algorithm", 0)
gmsh.option.setNumber("Mesh.SmoothRatio", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write("gempy_gmsh.msh")
gmsh.finalize()
