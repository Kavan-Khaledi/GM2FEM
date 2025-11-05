import gempy as gp
import numpy as np
import gempy_viewer as gpv






# Create instance of geomodel
geo_model = gp.create_geomodel(
    project_name = 'Perth_Basin',
    extent=[337000, 400000, 6640000, 6710000, -12000, 1000],
    resolution=[100,100,100],
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations="Perth_Basin_orientations.csv",
        path_to_surface_points="Perth_Basin_surface_points.csv"
    )
)
# %%

# %%


# %%


gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object= 
    {
        "fault_Abrolhos_Transfer": ["Abrolhos_Transfer"],
        "fault_Coomallo": ["Coomallo"],
        "fault_Eneabba_South": ["Eneabba_South"],
        "fault_Hypo_fault_W": ["Hypo_fault_W"],
        "fault_Hypo_fault_E": ["Hypo_fault_E"],
        "fault_Urella_North": ["Urella_North"],
        "fault_Urella_South": ["Urella_South"],
        "fault_Darling": ["Darling"],
        "Sedimentary_Series": ['Cretaceous', 'Yarragadee', 'Eneabba',
                               'Lesueur', 'Permian']
    }
)


gp.set_is_fault(
    frame=geo_model.structural_frame,
    fault_groups=["fault_Abrolhos_Transfer",
                        "fault_Coomallo",
                        "fault_Eneabba_South",
                        "fault_Hypo_fault_W",
                        "fault_Hypo_fault_E",
                        "fault_Urella_North",
                        "fault_Urella_South",
                        "fault_Darling"])

# %%

# Compute a solution for the model
gp.compute_model(geo_model)

import helper
helper.export_fault_grids_and_ids_txt(
    geo_model,
    out_dir="gmsh_io",
    num_faults=8,         # <- set your number of faults here
    n_lattice=60,
    mc_step=2,
)