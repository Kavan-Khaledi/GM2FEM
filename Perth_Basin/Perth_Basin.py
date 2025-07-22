import gempy as gp
import numpy as np





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
                        "fault_Darling",
                        "fault_Urella_South"])

# %%

# Compute a solution for the model
gp.compute_model(geo_model)


back_transformed_vertices_list = []

# Loop through each surface in geo_model.solutions.dc_meshes
for i in range(len(geo_model.solutions.dc_meshes)):
    # Get the vertices of the current mesh
    vertices = geo_model.solutions.dc_meshes[i].vertices
    
    # Apply the inverse transformation to the vertices
    transformed_vertices = geo_model.input_transform.apply_inverse(vertices)
    
    # Store the transformed vertices in the list or dictionary
    back_transformed_vertices_list.append(transformed_vertices)

import os
# Set your output directory
output_dir = "gempy_gmesh_input"
os.makedirs(output_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(output_dir, "gempy_gmesh_surface_vertices.npz"),
    back_transformed_vertices=np.array(back_transformed_vertices_list, dtype=object)
)

