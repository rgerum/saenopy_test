import saenopy


# load the relaxed and the contracted stack, {z} is the placeholder for the z stack
# use * as a placeholder to import multiple experiments at once
#results = saenopy.get_stacks([
#    '/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Relaxed/Mark_and_Find_001/Pos003_S001_z*_ch0{z}.tif',
#    '/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos003_S001_z*_ch0{z}.tif',
#], '/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output', voxel_size=[0.7211, 0.7211, 0.988])
results = saenopy.load('/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output/Relaxed/Mark_and_Find_001/Pos003_S001_z*_ch0{z}.tif')

# define the parameters for the piv deformation detection
params = {'win_um': 30.0, 'elementsize': 30.0, 'signoise_filter': 1.3, 'drift_correction': True}

# iterate over all the results objects
for result in results:
    # set the parameters
    result.piv_parameter = params
    # iterate over all stack pairs
    for i in range(len(result.stack) - 1):
        # and calculate the displacement between them
        result.mesh_piv[i] = saenopy.get_displacements_from_stacks(result.stack[i], result.stack[i + 1],
                                                                   params["win_um"],
                                                                   params["elementsize"],
                                                                   params["signoise_filter"],
                                                                   params["drift_correction"])
    # save the displacements
    result.save()

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'reference_stack': 'first', 'element_size': 30.0, 'inner_region': 100.0, 'thinning_factor': 0.0, 'mesh_size_same': True, 'mesh_size_x': 338.4362666666666, 'mesh_size_y': 338.4362666666666, 'mesh_size_z': 340.53066666666666}

# iterate over all the results objects
for result in results:
    # correct for the reference state
    displacement_list = saenopy.subtract_reference_state(result.mesh_piv, params["reference_stack"])
    # set the parameters
    result.interpolate_parameter = params
    # iterate over all stack pairs
    for i in range(len(result.mesh_piv)):
        # and create the interpolated solver mesh
        result.solver[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], params)
    # save the meshes
    result.save()

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'k': 1645.0, 'd0': 0.0008, 'lambda_s': 0.0075, 'ds': 0.033, 'alpha': 10000000000.0, 'stepper': 0.33, 'i_max': 100, 'rel_conv_crit': 0.01}

# iterate over all the results objects
for result in results:
    result.solve_parameter = params
    for M in result.solver:
        # set the material model
        M.setMaterialModel(saenopy.materials.SemiAffineFiberMaterial(
            params["k"],
            params["d0"],
            params["lambda_s"],
            params["ds"],
        ))
        # find the regularized force solution
        M.solve_regularized(stepper=params["stepper"], i_max=params["i_max"], alpha=params["alpha"], rel_conv_crit=params["rel_conv_crit"], verbose=True)
    # save the forces
    result.save()

