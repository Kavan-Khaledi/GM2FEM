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


def grid_subsample_xy(points, max_points):
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)

    volume = np.prod(max_bounds - min_bounds)
    cell_volume = volume / max_points
    cell_size = np.cbrt(cell_volume)

    # Compute grid indices
    grid_indices = np.floor((points[:, :2] - min_bounds[:2]) / cell_size).astype(int)

    # Collect points in each grid cell
    cell_points = defaultdict(list)
    for idx, point in zip(map(tuple, grid_indices), points):
        cell_points[idx].append(point)

    # Compute the average of z values for each grid cell
    subsampled_points = []
    for idx in cell_points:
        # Average over the z values
        avg_point = np.mean(cell_points[idx], axis=0)
        subsampled_points.append(avg_point)

    return np.array(subsampled_points)





back_transformed_vertices_list = []
# Use glob to find all txt files in your directory, adjust path and pattern if needed
file_list = sorted(glob.glob("Point_Clouds_Full_Version.txt"))

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



grid_res = 300
rbf_models = []

for idx, extracted_points in enumerate(back_transformed_vertices_list):
    # Original points
    x = extracted_points[:, 0]
    y = extracted_points[:, 1]
    z = extracted_points[:, 2]

    max_points = 100000  # limit for memory/speed

    points = extracted_points  # shape (N,3)

    if len(points) > max_points:
        # Use grid subsampling instead of random sampling
        points_subsampled = grid_subsample_xy(all_points, max_points)
        x_sub, y_sub, z_sub = points_subsampled[:, 0], points_subsampled[:, 1], points_subsampled[:, 2]
    else:
        x_sub, y_sub, z_sub = x, y, z

    points_sub = np.column_stack((x_sub, y_sub))
    values_sub = z_sub

    # Create RBFInterpolator with linear kernel, epsilon and smoothing as you had
    rbf_model = RBFInterpolator(points_sub, values_sub, kernel='multiquadric', epsilon=10000)
    rbf_models.append(rbf_model)

    # Evaluate on grid for plotting
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), grid_res),
                                 np.linspace(y.min(), y.max(), grid_res))
    eval_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    grid_z = rbf_model(eval_points).reshape(grid_x.shape)

    # Flatten grid to point list for Gmsh
    points = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))

    point_tags = []
    for pt in points:
        tag = gmsh.model.occ.addPoint(pt[0], pt[1], pt[2])
        point_tags.append(tag)

    gmsh.model.geo.synchronize()

    try:
        gmsh.model.occ.addBSplineSurface(
            point_tags,
            numPointsU=grid_res,
            tag=100*(idx+1)  # unique tag for each surface
        )
    except Exception as e:
        print(f"Failed to add surface for index {idx}: {e}")

    gmsh.model.occ.remove([(0, tag) for tag in point_tags], recursive=True)
    gmsh.model.occ.synchronize()






all_surfaces = gmsh.model.getEntities(2)

gmsh.model.occ.fragment(all_surfaces,all_surfaces, -1, True, True)
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMax", 100)
gmsh.option.setNumber("Mesh.Algorithm", 0)
gmsh.option.setNumber("Mesh.SmoothRatio", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # 10 = HXT
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.occ.synchronize()



gmsh.write("gempy_gmsh.msh")

gmsh.finalize()


