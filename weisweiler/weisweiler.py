"""
1.1 -Basics of geological modeling with GemPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import numpy as np

# %%
# Importing Necessary Libraries
# """"""""""""""""""""""""""""""
import gempy as gp
import gempy_viewer as gpv
import gmsh
import math
import random
import sys
from scipy.interpolate import Rbf
from scipy.interpolate import SmoothBivariateSpline
import glob
from scipy.interpolate import RBFInterpolator
from collections import defaultdict



back_transformed_vertices_list = []
# Use glob to find all txt files in your directory, adjust path and pattern if needed
file_list = sorted(glob.glob("*.txt"))

for file_path in file_list:
    # Load x,y,z columns from the txt file
    points = np.loadtxt(file_path)  # assumes whitespace delimiter by default
    # Check shape, must be Nx3
    if points.shape[1] != 3:
        raise ValueError(f"File {file_path} does not have 3 columns")
    back_transformed_vertices_list.append(points)

all_points = np.vstack(back_transformed_vertices_list)  # shape (total_points, 3)

xmin, ymin, zmin = np.min(all_points, axis=0)
xmax, ymax, zmax = np.max(all_points, axis=0)

for i in range(len(back_transformed_vertices_list)):
    pts = back_transformed_vertices_list[i]
    pts[:, 0] = pts[:, 0] - xmin  # shift x so xmin=0
    pts[:, 1] = pts[:, 1] - ymin  # shift y so ymin=0
    pts[:, 2] = pts[:, 2] - zmax  # shift z so zmax=0
    back_transformed_vertices_list[i] = pts

all_points = np.vstack(back_transformed_vertices_list)  # shape (total_points, 3)
xmin, ymin, zmin = np.min(all_points, axis=0)
xmax, ymax, zmax = np.max(all_points, axis=0)

# Function to check if a point is on the boundary of the box
def is_on_box_boundary(coords):
    if abs(coords[0]-xmin)<1 or abs(coords[0]-xmax)<1 or abs(coords[1]-ymin)<1 or abs(coords[1]-ymax)<1 or abs(coords[2]-zmin)<1 or abs(coords[2]-zmax)<1:
        return True
    return False

def is_out_box(coords):

    if coords[0] > xmin and coords[0]< xmax and coords[1] > ymin and coords[1]< ymax and coords[2] > zmin and coords[2]< zmax:
        return True
    return False

def get_middle_three_elements(arr):
    n = len(arr)

    if n % 2 == 1:  # Odd length
        middle_index = n // 2
        middle_elements = arr[middle_index - 1:middle_index + 2]
    else:  # Even length
        middle_index = n // 2
        middle_elements = arr[middle_index - 1:middle_index + 2]

    return middle_elements

gmsh.initialize()

gmsh.model.add("gempy_gmsh")


model_bounds = {
    'min': [xmin, ymin, zmin],
    'max': [xmax, ymax, zmax]
}

# Compute dimensions
dx = xmax - xmin
dy = ymax - ymin
dz = zmax - zmin

meshSizeMax=((dx**2+dy**2+dz**2)**0.5)/60
meshSizeMin=((dx**2+dy**2+dz**2)**0.5)/120

box_tag = 1  # optional tag

# Add box using corners
gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz, box_tag)

gmsh.model.occ.synchronize()

def normalize_points(points, bounds):
    min_bounds = np.array(bounds['min'])
    max_bounds = np.array(bounds['max'])
    return (points - min_bounds) / (max_bounds - min_bounds)
# -----------------------------
# Step 2: Grid subsampling functions
# -----------------------------
def grid_subsample_xy(points, max_points):
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    volume = np.prod(max_bounds - min_bounds)
    cell_volume = volume / max_points
    cell_size = np.cbrt(cell_volume)

    grid_indices = np.floor((points[:, :2] - min_bounds[:2]) / cell_size).astype(int)

    cell_points = defaultdict(list)
    for idx, point in zip(map(tuple, grid_indices), points):
        cell_points[idx].append(point)

    subsampled_points = [np.mean(cell_points[idx], axis=0) for idx in cell_points]
    return np.array(subsampled_points)

def grid_subsample_yz(points, max_points):
    min_bounds = points[:, 1:].min(axis=0)  # y,z
    max_bounds = points[:, 1:].max(axis=0)
    volume = np.prod(max_bounds - min_bounds)
    cell_volume = volume / max_points
    cell_size = np.cbrt(cell_volume)

    grid_indices = np.floor((points[:, 1:] - min_bounds) / cell_size).astype(int)

    cell_points = defaultdict(list)
    for idx, point in zip(map(tuple, grid_indices), points):
        cell_points[idx].append(point)

    subsampled_points = [np.mean(cell_points[idx], axis=0) for idx in cell_points]
    return np.array(subsampled_points)

def grid_subsample_xz(points, max_points):
    min_bounds = points[:, [0, 2]].min(axis=0)  # x,z
    max_bounds = points[:, [0, 2]].max(axis=0)
    volume = np.prod(max_bounds - min_bounds)
    cell_volume = volume / max_points
    cell_size = np.cbrt(cell_volume)

    grid_indices = np.floor((points[:, [0, 2]] - min_bounds) / cell_size).astype(int)

    cell_points = defaultdict(list)
    for idx, point in zip(map(tuple, grid_indices), points):
        cell_points[idx].append(point)

    subsampled_points = [np.mean(cell_points[idx], axis=0) for idx in cell_points]
    return np.array(subsampled_points)

def select_best_grid(points, max_points, model_bounds):
    # Subsample each plane
    subsampled_xy = grid_subsample_xy(points, max_points)
    subsampled_yz = grid_subsample_yz(points, max_points)
    subsampled_xz = grid_subsample_xz(points, max_points)

    # Normalize points to [0, 1] space using model bounds
    normalized_xy = normalize_points(subsampled_xy, model_bounds)
    normalized_yz = normalize_points(subsampled_yz, model_bounds)
    normalized_xz = normalize_points(subsampled_xz, model_bounds)

    # Compute variance of the "dependent" axis in each projection
    var_z_xy = np.var(normalized_xy[:, 2])  # z is dependent in xy plane
    var_x_yz = np.var(normalized_yz[:, 0])  # x is dependent in yz plane
    var_y_xz = np.var(normalized_xz[:, 1])  # y is dependent in xz plane

    variances = {
        'xy': var_z_xy,
        'yz': var_x_yz,
        'xz': var_y_xz
    }

    best_plane = min(variances, key=variances.get)

    if best_plane == 'xy':
        return subsampled_xy, 'xy'
    elif best_plane == 'yz':
        return subsampled_yz, 'yz'
    else:
        return subsampled_xz, 'xz'

from scipy.interpolate import RBFInterpolator

grid_res = 300
rbf_models = []
eps_idx=[0.001,0.001,0.001,100,0.001,0.001,0.001,100,100]

for idx, extracted_points in enumerate(back_transformed_vertices_list):
    max_points = 30000
    points = extracted_points


    points_subsampled, best_plane = select_best_grid(points, max_points,model_bounds)


    # Decide input and value dimensions based on best_plane
    if best_plane == 'xy':
        points_sub = points_subsampled[:, :2]  # x, y
        values_sub = points_subsampled[:, 2]   # z
    
        # Use bounding box x and y ranges instead of points range
        grid_x, grid_y = np.meshgrid(
            np.linspace(xmin, xmax, grid_res),
            np.linspace(ymin, ymax, grid_res)
        )
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    elif best_plane == 'yz':
        points_sub = points_subsampled[:, 1:]  # y, z
        values_sub = points_subsampled[:, 0]   # x
    
        # Use bounding box y and z ranges
        grid_y, grid_z = np.meshgrid(
            np.linspace(ymin, ymax, grid_res),
            np.linspace(zmin, zmax, grid_res)
        )
        grid_points = np.column_stack((grid_y.ravel(), grid_z.ravel()))
    
    else:  # 'xz'
        points_sub = points_subsampled[:, [0, 2]]  # x, z
        values_sub = points_subsampled[:, 1]       # y
    
        # Use bounding box x and z ranges
        grid_x, grid_z = np.meshgrid(
            np.linspace(xmin, xmax, grid_res),
            np.linspace(zmin, zmax, grid_res)
        )
        grid_points = np.column_stack((grid_x.ravel(), grid_z.ravel()))

    # Fit RBF model
    rbf_model = RBFInterpolator(points_sub, values_sub, kernel='multiquadric',epsilon=eps_idx[idx],smoothing=10000)
    rbf_models.append(rbf_model)

    # Evaluate RBF on grid
    grid_values = rbf_model(grid_points)

    # Reshape grid values for surface creation
    if best_plane == 'xy':
        grid_values_2d = grid_values.reshape(grid_x.shape)
        pts_surface = np.column_stack((
            grid_x.ravel(),
            grid_y.ravel(),
            grid_values.ravel()
        ))
    elif best_plane == 'yz':
        grid_values_2d = grid_values.reshape(grid_y.shape)
        pts_surface = np.column_stack((
            grid_values.ravel(),
            grid_y.ravel(),
            grid_z.ravel()
        ))
    else:  # 'xz'
        grid_values_2d = grid_values.reshape(grid_x.shape)
        pts_surface = np.column_stack((
            grid_x.ravel(),
            grid_values.ravel(),
            grid_z.ravel()
        ))

    # Add points to Gmsh for surface
    point_tags = []
    for pt in pts_surface:
        tag = gmsh.model.occ.addPoint(pt[0], pt[1], pt[2])
        point_tags.append(tag)
    gmsh.model.occ.synchronize()

    # Add BSpline surface (U, V = grid_res)
    try:
        gmsh.model.occ.addBSplineSurface(point_tags, numPointsU=grid_res, tag=100*(idx+1))
    except Exception as e:
        print(f"Failed to add surface for idx {idx}: {e}")

    # Remove points to clean up
    gmsh.model.occ.remove([(0, tag) for tag in point_tags], recursive=True)
    gmsh.model.occ.synchronize()




# Final mesh options and mesh generation
gmsh.option.setNumber("Mesh.MeshSizeMax", meshSizeMax)
gmsh.option.setNumber("Mesh.MeshSizeMax", meshSizeMin)
gmsh.option.setNumber("Mesh.Algorithm", 0)
gmsh.option.setNumber("Mesh.SmoothRatio", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # 10 = HXT
gmsh.model.mesh.generate(2)
gmsh.model.occ.synchronize()
#gmsh.model.mesh.generate(3)
gmsh.write("gempy_gmsh.msh")
gmsh.fltk()
gmsh.finalize()