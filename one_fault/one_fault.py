"""
1.1 -Basics of geological modeling with GemPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import numpy as np

# %%
# Importing Necessary Libraries
# """"""""""""""""""""""""""""""
import gempy as gp






# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Tutorial_ch1_1_Basics',
    extent=[0, 2000, 0, 2000, 0, 750],
    resolution=[50,50,50],  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/getting_started/simple_fault_model_orientations.csv",
        path_to_surface_points=data_path + "/data/input_data/getting_started/simple_fault_model_points.csv",

    )
)


# %%
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object=  # TODO: This mapping I do not like it too much. We should be able to do it passing the data objects directly
    {
            "Fault_Series": 'Main_Fault',
            "Strat_Series": ('Sandstone_2', 'Siltstone', 'Shale', 'Sandstone_1')
    }
)


# %%
gp.set_is_fault(
    frame=geo_model.structural_frame,
    fault_groups=['Fault_Series']
)



geo_model.interpolation_options.number_octree_levels_surface=4


sol = gp.compute_model(geo_model)

import helper
helper.export_fault_grids_and_ids_txt(
    geo_model,
    out_dir="gmsh_io",
    num_faults=1,         # <- set your number of faults here
    n_lattice=60,
    mc_step=2,
)
