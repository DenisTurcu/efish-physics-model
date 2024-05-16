import sys
sys.path.append('helper_functions/')
sys.path.append('objects/')
sys.path.append('uniform_points_generation/')

import copy
import time
import dill
import numpy as np

from Worm import Worm
from Worm_SmallSpherical import SmallSphericalWorm

from Fish import Fish
from FishGeneration import FishGeneration
from Fish_Capsule import CapsuleFish

from Boundaries import *
from Aquarium import Aquarium
from Aquarium_SinglePlane import SinglePlaneAquarium

from receptors_response_data_Generate import generate_receptors_responses

HDF5_save_period = 100000
save_LEODs = False
GPU_id = None
save_folder = 'data/processed'
save_worm_objs = False
sys.argv = sys.argv[1:]
while sys.argv:
    if sys.argv[0] == 'HDF5_save_period':
        HDF5_save_period = int(sys.argv[1])
        print(f'HDF5_save_period was modified to {HDF5_save_period}.')
    elif sys.argv[0] == 'save_LEODs':
        save_LEODs = True if sys.argv[1].lower() == 'true' else False
        print(f'save_LEODs was modified to {save_LEODs}.')
    elif sys.argv[0] == 'GPU':
        GPU_id = int(sys.argv[1])
        print(f'GPU id was modified to {GPU_id}.')
    elif sys.argv[0] == 'save_folder':
        save_folder = sys.argv[1]
        print(f'save_folder was modified to {save_folder}.')
    elif sys.argv[0] == 'save_worm_objs':
        save_worm_objs = True if sys.argv[1].lower() == 'true' else False
    else:
        break
    sys.argv = sys.argv[2:]

for i in range(len(sys.argv)):
    data_file_name = sys.argv[i]
    data_params_dict = dill.load(open(data_file_name, 'rb'))
    print('Processing %s...' % data_file_name)

    generate_receptors_responses(
        save_file_name=data_params_dict['save_file_name'],
        # aquarium properties
        water_conductivities=data_params_dict['water_conductivities'],
        boundary_displacements=data_params_dict['boundary_displacements'],
        boundary_normals=data_params_dict['boundary_normals'],
        # fish properties
        fish_obj=data_params_dict['fish_obj'],
        fish_bend_angle_lateral=data_params_dict['fish_bend_angle_lateral'],
        fish_bend_angle_dorso_ventral=data_params_dict['fish_bend_angle_dorso_ventral'],
        fish_bend_location_percentages=data_params_dict['fish_bend_location_percentages'],
        fish_yaw=data_params_dict['fish_yaw'],
        fish_pitch=data_params_dict['fish_pitch'],
        fish_roll=data_params_dict['fish_roll'],
        # worm properties
        worm_resistances=data_params_dict['worm_resistances'],
        worm_capacitances=data_params_dict['worm_capacitances'],
        worm_radii=data_params_dict['worm_radii'],
        worm_position_xs=data_params_dict['worm_position_xs'],
        worm_position_ys=data_params_dict['worm_position_ys'],
        worm_position_zs=data_params_dict['worm_position_zs'],
        HDF5_save_period=HDF5_save_period,
        save_LEODs=save_LEODs,
        GPU=GPU_id,
        save_folder=save_folder,
        save_worm_objs=save_worm_objs,
    )
    print()




