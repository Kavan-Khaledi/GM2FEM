import os, glob, numpy as np, open3d as o3d, pyvista as pv
import matplotlib.cm as cm

import glob

folder_path = "."  # Current directory
file_list = glob.glob(os.path.join(folder_path, "*.txt"))

print(f"{len(file_list)} files found.")
print("Files:", file_list)

plotter = pv.Plotter()
plotter.set_background("black")

for i, file in enumerate(file_list):
    points = np.loadtxt(file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=5)
    pts = np.asarray(pcd_clean.points)

    surf = pv.PolyData(pts).delaunay_2d().smooth(n_iter=4000, relaxation_factor=0.01)
    color = cm.tab10(i % 10)[:3]
    plotter.add_mesh(surf, color=color, opacity=0.4, show_edges=False)

plotter.add_axes()       # ← standard static axes
plotter.show_axes()      # ← small corner widget with orientation
plotter.reset_camera()
plotter.show()

