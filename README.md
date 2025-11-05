# GM2FEM
GM2FEM is a workflow to convert a GemPy geological model into a Gmsh 3D mesh containing fault surfaces and sedimentary layers.

The helper.py script extracts scalar fields, level sets, and lithology IDs from GemPy, generating text files with fault surface points and a grid of sedimentary IDs.

The GM2FEM_mesh_generator.py script reads the .txt files, fits smooth B-spline fault surfaces using GmshBSplineSurfaceBuilder.py, creates and fragments a 3D box around the model domain, refines the mesh near faults, and assigns sedimentary layers as physical volumes based on the extracted IDs.

The final output is a fully meshed .msh model suitable for numerical simulations, e.g. coupled THM reservior modeling in MOOSE FRamework

<p align="center">
  <img src="docs/images/perth_basin_gempy.png" alt="Perth_Basin model created in GemPy" width="600">
</p>

<p align="center">
  <img src="docs/images/perth_basin_sedimentary.png" alt="Perth_Basin model, 3D volume mesh" width="600">
</p>

<p align="center">
  <img src="docs/images/perth_basin_faults.png" alt="Perth_Basin model, 3D volume mesh with embedded 2D faults" width="600">
</p>
