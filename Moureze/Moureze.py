import gempy as gp
import numpy as np
import gempy_viewer as gpv






# Create instance of geomodel
geo_model = gp.create_geomodel(
    project_name = 'Moureze',
    extent=[-5, 305, -5, 405, -200, -50],
    # resolution=resolution_low,
    refinement=5,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations="Moureze_orientations.csv",
        path_to_surface_points="Moureze_surface_points.csv"
    )
)

# Compute a solution for the model
gp.compute_model(geo_model)

import helper
helper.export_fault_grids_and_ids_txt(
    geo_model,
    out_dir="gmsh_io",
    num_faults=0,         # <- set your number of faults here
    n_lattice=60,
    mc_step=2,
)