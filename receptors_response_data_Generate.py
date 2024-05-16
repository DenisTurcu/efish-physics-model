import sys
sys.path.append('helper_functions/')
sys.path.append('objects/')
sys.path.append('uniform_points_generation/')

import copy
import time
import dill
import numpy as np
import h5py
import torch

from Worm import Worm
from Worm_SmallSpherical import SmallSphericalWorm

from Fish import Fish
from FishGeneration import FishGeneration
from Fish_Capsule import CapsuleFish

from Boundaries import *
from Aquarium import Aquarium
from Aquarium_SinglePlane import SinglePlaneAquarium


def running_stream_computation(responses, expectation, expectation_of_squares, num_samples):
    expectation = (responses + num_samples * expectation) / (num_samples+1)
    expectation_of_squares = (np.power(responses,2) + num_samples * expectation_of_squares) / (num_samples+1)
    return expectation, expectation_of_squares, num_samples+1


def generate_receptors_responses(
        save_file_name,
        # aquarium properties
        water_conductivities,
        boundary_displacements,
        boundary_normals,
        # fish properties
        fish_obj,
        fish_bend_angle_lateral,
        fish_bend_angle_dorso_ventral,
        fish_bend_location_percentages,
        fish_yaw,
        fish_pitch,
        fish_roll,
        # worm properties
        worm_resistances,
        worm_capacitances,
        worm_radii,
        worm_position_xs,
        worm_position_ys,
        worm_position_zs,
        HDF5_save_period=100000,
        save_LEODs=False,
        GPU=None,
        save_worm_objs=False,
        save_folder='data/processed',
    ):
    ''' 
    Function to prepare dataset of electric images. Fish nose is at [0,0,0] and the 
    unbend & unrotated fish lies straight along the negative x axis. 
    The boundary height is set with respect to the bottom of the fish. 
    The worm-ys position is also modified by adding the fish's lateral semi-axis 
    in order to approximate the skin-to-object-center distance.
    Make sure that the parameters are self-consistent.
        save_file_name,
        
        water_conductivities            = np.array([35, 100, 300, 600]) * 1e-6 / 1e-2,  # in S/m
        boundary_displacements          = [np.array([-5, -10, -20, -30, -1e9]) * 1e-3],
        boundary_normals                = [[0,0,1]]
        
        fish_obj,
        fish_bend_angle_lateral         = [list(x) for x in np.linspace(-80,80,17).reshape(-1,1)] + [[15,5]] + [[]],  # in degrees
        fish_bend_angle_dorso_ventral   = [[0], [0,0], []],  # in degrees
        fish_bend_location_percentages  = [[50], [40,70], []],  # percentages from 0 to 100
        fish_yaw                        = [0],  # in degrees
        fish_pitch                      = [0],  # in degrees
        fish_roll                       = [0],  # in degrees
        
        worm_resistances                = np.exp(np.linspace(np.log(0.5), np.log(100), 10)) * 1e3,   # in Ohm
        worm_capacitances               = np.exp(np.linspace(np.log(0.5), np.log(100), 10)) * 1e-9,  # in F
        worm_radii                      = np.array([5, 10, 15]) * 1e-3,  # in m
        worm_position_xs                = np.array([10,0,-10,-20,-30,-40]) * 1e-3,  # in m
        worm_position_ys                = np.array([5,7,9,11,13,15]) * 1e-3,        # in m
        worm_position_zs                = np.array([-10,0,10]) * 1e-3               # in m
    '''
    start_time = time.time()
    device = 'cpu' if GPU is None else int(GPU)

    dataset = {}
    dataset['worms'] = {}
    dataset['fish']  = {}
    dataset['aqua']  = {}
    dataset['electric_images'] = {}
        
    # define the aquarium's variable properties
    dataset['aqua']['properties_dict']        = {'conductivity':0, 'boundary_normals':1, 'boundary_displacements':2} 
    dataset['aqua']['conductivity']           = water_conductivities
    dataset['aqua']['boundary_normals']       = boundary_normals
    dataset['aqua']['boundary_displacements'] = boundary_displacements
    assert len(boundary_displacements) == len(boundary_normals), 'There must be the same number of normals as displacements lists in the dataset.'

    # define the fish's variable properties
    dataset['fish']['properties_dict']           = {'bend_angle_lateral':0, 
                                                    'bend_angle_dorso_ventral':1, 
                                                    'bend_location_percentages':2,
                                                    'fish_yaw':3,
                                                    'fish_pitch':4,
                                                    'fish_roll':5}
    dataset['fish']['bend_angle_lateral']        = fish_bend_angle_lateral
    dataset['fish']['bend_angle_dorso_ventral']  = fish_bend_angle_dorso_ventral
    dataset['fish']['bend_location_percentages'] = fish_bend_location_percentages
    dataset['fish']['fish_yaw']                  = fish_yaw
    dataset['fish']['fish_pitch']                = fish_pitch
    dataset['fish']['fish_roll']                 = fish_roll

    # define the worm's variable properties
    dataset['worms']['properties_dict'] = {'resistances':0, 
                                           'capacitances':1, 
                                           'radii':2,
                                           'position_xs':3,
                                           'position_ys':4,
                                           'position_zs':5}
    dataset['worms']['resistances']     = worm_resistances
    dataset['worms']['capacitances']    = worm_capacitances
    dataset['worms']['radii']           = worm_radii
    dataset['worms']['position_xs']     = worm_position_xs
    dataset['worms']['position_ys']     = worm_position_ys
    dataset['worms']['position_zs']     = worm_position_zs
        
    # prepare the electric images structure
    dataset['electric_images']['properties_dict'] = {'aqua':0, 'fish':1, 'worm':2}
    dataset['electric_images']['base'] = dict(objs_ids=[], data=[], LEODs=[])
    dataset['electric_images']['pert'] = dict(objs_ids=[])
    dataset_electric_images_pert_data = []
    if save_LEODs:
        dataset_LEODs = []

    ############################################################################################################
    #### Create the Object Lists Below #########################################################################
    ############################################################################################################
    # create the aquarium obj list
    dataset['aqua']['combined_properties'] = []
    for i0 in range(len(dataset['aqua']['conductivity'])):
        for i1 in range(len(dataset['aqua']['boundary_normals'])):
            for i2 in range(len(dataset['aqua']['boundary_displacements'][i1])):
                dataset['aqua']['combined_properties'].append([i0,i1,i2])
    dataset['aqua']['combined_properties'] = np.array(dataset['aqua']['combined_properties'])
    dataset['aqua']['aqua_objs'] = []
    for cp in dataset['aqua']['combined_properties']:
        id_cond = cp[dataset['aqua']['properties_dict']['conductivity']]
        id_norm = cp[dataset['aqua']['properties_dict']['boundary_normals']]
        id_disp = cp[dataset['aqua']['properties_dict']['boundary_displacements']]
        if np.abs(dataset['aqua']['boundary_displacements'][id_norm][id_disp]) > 1:
            dataset['aqua']['aqua_objs'].append(
                Aquarium(
                    relative_permitivity=80,
                    conductivity=dataset['aqua']['conductivity'][id_cond],
                    _init_tests=False,
                )
            )
        else:
            temp_normal = dataset['aqua']['boundary_normals'][id_norm]
            temp_central_point = np.array(temp_normal) * dataset['aqua']['boundary_displacements'][id_norm][id_disp]
            dataset['aqua']['aqua_objs'].append(
                SinglePlaneAquarium(
                    relative_permitivity=80,
                    conductivity=dataset['aqua']['conductivity'][id_cond],
                    boundaries=[('plane', dict(normal=temp_normal,
                                               central_point=temp_central_point))],
                    _init_tests=False,
                )
            )

    # create the fish obj list
    dataset['fish']['combined_properties'] = []
    for i0, temp0 in enumerate(dataset['fish']['bend_angle_lateral']):
        for i1, temp1 in enumerate(dataset['fish']['bend_angle_dorso_ventral']):
            for i2, temp2 in enumerate(dataset['fish']['bend_location_percentages']):
                assert ((len(temp0) == len(temp1)) and (len(temp1) == len(temp2))), 'Fish bent angles and location must have matching sizes.'
                for i3 in range(len(dataset['fish']['fish_yaw'])):
                    for i4 in range(len(dataset['fish']['fish_pitch'])):
                        for i5 in range(len(dataset['fish']['fish_roll'])):
                            dataset['fish']['combined_properties'].append([i0, i1, i2, i3, i4, i5])
    dataset['fish']['combined_properties'] = np.array(dataset['fish']['combined_properties'])
    # define the base fish
    dataset['fish']['fish_objs'] = []
    for cp in dataset['fish']['combined_properties']:
        temp_fish = copy.deepcopy(fish_obj)
        id_lat = cp[dataset['fish']['properties_dict']['bend_angle_lateral']]
        id_dve = cp[dataset['fish']['properties_dict']['bend_angle_dorso_ventral']]
        id_loc = cp[dataset['fish']['properties_dict']['bend_location_percentages']]
        id_yaw = cp[dataset['fish']['properties_dict']['fish_yaw']]
        id_pit = cp[dataset['fish']['properties_dict']['fish_pitch']]
        id_rol = cp[dataset['fish']['properties_dict']['fish_roll']]
        temp_fish.update_parameters( 
            nose_position=None,
            angle_yaw   = (dataset['fish']['fish_yaw'][id_yaw], 'deg'), 
            angle_pitch = (dataset['fish']['fish_pitch'][id_pit], 'deg'), 
            angle_roll  = (dataset['fish']['fish_roll'][id_rol], 'deg'),
            relative_bend_locations_percentage  = np.array(dataset['fish']['bend_location_percentages'][id_loc]), 
            relative_bend_angle_lateral         =(np.array(dataset['fish']['bend_angle_lateral'][id_lat]), 'deg'), 
            relative_bend_angle_dorso_ventral   =(np.array(dataset['fish']['bend_angle_dorso_ventral'][id_dve]), 'deg'),
        )
        dataset['fish']['fish_objs'].append(temp_fish)


    # create the worm obj list
    dataset['worms']['combined_properties'] = []
    for i0 in range(len(dataset['worms']['resistances'])):
        for i1 in range(len(dataset['worms']['capacitances'])):
            for i2, rr in enumerate(dataset['worms']['radii']):
                for i3 in range(len(dataset['worms']['position_xs'])):
                    for i4, yy in enumerate(dataset['worms']['position_ys']):
                        for i5 in range(len(dataset['worms']['position_zs'])):
                            if rr <= yy:
                                dataset['worms']['combined_properties'].append([i0,i1,i2,i3,i4,i5])
                            else:
                                print('One instance of a worm large enough that it crosses the skin of the fish detected.')
    dataset['worms']['combined_properties'] = np.array(dataset['worms']['combined_properties'])
    dataset['worms']['worm_objs'] = []
    for cp in dataset['worms']['combined_properties']:
        id_res = cp[dataset['worms']['properties_dict']['resistances']]
        id_cap = cp[dataset['worms']['properties_dict']['capacitances']]
        id_rad = cp[dataset['worms']['properties_dict']['radii']]
        id_pxs = cp[dataset['worms']['properties_dict']['position_xs']]
        id_pys = cp[dataset['worms']['properties_dict']['position_ys']]
        id_pzs = cp[dataset['worms']['properties_dict']['position_zs']]
        dataset['worms']['worm_objs'].append(
            SmallSphericalWorm(
                radius=dataset['worms']['radii'][id_rad],
                resistance=dataset['worms']['resistances'][id_res],
                capacitance=dataset['worms']['capacitances'][id_cap],
                center_position=[dataset['worms']['position_xs'][id_pxs],
                                 dataset['worms']['position_ys'][id_pys],
                                 dataset['worms']['position_zs'][id_pzs]
                                ],
                _init_tests=False,
            )
        )
    end_time = time.time()
    print(f'Time to prepare dataset: {end_time - start_time:.3f} s')
    print('Total aquarium-fish pairs: %d' % (len(dataset['fish']['fish_objs']) * len(dataset['aqua']['aqua_objs'])))
    print('Total worms: %d' % len(dataset['worms']['worm_objs']))

    ############################################################################################################
    #### Compute the electric images for every objects pairing #################################################
    ############################################################################################################
    keeper_id = 0
    print('Current aqua-fish pair: ')
    main_start_time = time.time()
    # store the dipole waveforms for each (R,C,sig_water) - given a single EOD type/waveform, this doesn't change w.r.t to fish/aqua
    dict_dipole_wave_forms = {}
    # compute the running mean and standard deviation of each receptor's response
    expectation_receptor_responses = 0
    expecation_of_squared_receptors_responses = 0 
    if save_LEODs:
        expectation_LEODs = 0
        expectation_of_squared_LEODs = 0
    number_of_responses = 0
    # store and save data in hdf5 format
    count_pert_EI = 0
    with h5py.File(f'{save_folder}/{save_file_name}.hdf5', 'w') as f:
        f.create_dataset('pert_EI', 
                         (len(dataset['fish']['fish_objs']) * len(dataset['aqua']['aqua_objs']) * len(dataset['worms']['worm_objs']), 
                          fish_obj.get_N_receptors(), 
                          fish_obj.get_N_filters()), 
                         dtype=np.float64)
    if save_LEODs:
        with h5py.File(f'{save_folder}/{save_file_name}_LEODs.hdf5', 'w') as f:
            f.create_dataset('pert_LEODs', 
                            (len(dataset['fish']['fish_objs']) * len(dataset['aqua']['aqua_objs']) * len(dataset['worms']['worm_objs']), 
                            fish_obj.get_N_receptors(), 
                            fish_obj.eod_length), 
                            dtype=np.float64)
    for i_aq, aqua in enumerate(dataset['aqua']['aqua_objs']):
        # identify the id of water conductivity
        aqua_cp = dataset['aqua']['combined_properties'][i_aq]
        id_sig_water = aqua_cp[dataset['aqua']['properties_dict']['conductivity']]
        for i_fi, fish in enumerate(dataset['fish']['fish_objs']):
            print(f'ID: {keeper_id}.', end=' ')
            start_time = time.time()
            start_time_HDF5_save = start_time
            aqua.insert_fish(fish)
            _, base_E_field_mag, wave_form = aqua.electric_potential_and_field_single_fish(
                points=fish.get_receptors_locations(), 
                fish_id=0, 
                return_potential=False, 
                return_field=True
            )
            base_E_field_full = torch.tensor(base_E_field_mag).to(device) * torch.tensor(wave_form).to(device)
            # compute the base electric image
            base_transdermal_signal = fish.compute_transdermal_signal(
                                    E_field=base_E_field_mag.squeeze().reshape(-1,3), 
                                    water_conductivity=aqua.get_conductivity(), 
                                    temporal_wave_form=wave_form
                          )
            base_receptors_responses = fish.compute_receptors_responses(base_transdermal_signal)
            dataset['electric_images']['base']['objs_ids'].append([i_aq, i_fi])
            dataset['electric_images']['base']['data'].append(base_receptors_responses)
            expectation_receptor_responses, expecation_of_squared_receptors_responses, number_of_responses = running_stream_computation(
                base_receptors_responses, expectation_receptor_responses, expecation_of_squared_receptors_responses, number_of_responses
            )
            if save_LEODs:
                dataset['electric_images']['base']['LEODs'].append(base_transdermal_signal)
                expectation_LEODs, expectation_of_squared_LEODs, number_of_responses = running_stream_computation(
                    base_transdermal_signal, expectation_LEODs, expectation_of_squared_LEODs, number_of_responses
                )
            
            # store the perturbation magnitudes for each 3D position - can change due to fish position/tail/movement
            dict_perturbation_magnitude = {}
            for i_wo, worm in enumerate(dataset['worms']['worm_objs']):
                # identify the relevant ids for the worm
                worm_cp = dataset['worms']['combined_properties'][i_wo]
                id_res  = worm_cp[dataset['worms']['properties_dict']['resistances']]
                id_cap  = worm_cp[dataset['worms']['properties_dict']['capacitances']]
                id_rad  = worm_cp[dataset['worms']['properties_dict']['radii']]
                id_pxs  = worm_cp[dataset['worms']['properties_dict']['position_xs']]
                id_pys  = worm_cp[dataset['worms']['properties_dict']['position_ys']]
                id_pzs  = worm_cp[dataset['worms']['properties_dict']['position_zs']]
                
                i_rRCsig  = (id_rad, id_res, id_cap, id_sig_water)  # indices for rad, R, C and water conductivity
                i_3Dp    = (id_pxs, id_pys, id_pzs)  # indices for the 3D position
                
                # populate the dipole waveforms dictionary, if necessary
                if i_rRCsig not in dict_dipole_wave_forms:
                    dict_dipole_wave_forms[i_rRCsig] = worm.dipole_waveform_single_fish(
                        wave_form=wave_form,
                        sig_water=aqua.get_conductivity(),
                        eps_r_water=aqua.get_relative_permitivity(),
                        sampling_frequency=fish.get_sampling_rate()
                    )
                # populate the perturbations magnitude dictionary, if necessary
                if i_3Dp not in dict_perturbation_magnitude:
                    # compute field at worm location
                    _, E_field, _ = aqua.electric_potential_and_field_single_fish(
                        points=worm.get_position()[np.newaxis], 
                        fish_id=0,
                        return_potential=False, 
                        return_field=True,
                    )
                    dict_perturbation_magnitude[i_3Dp] = worm.perturbation_magnitude_single_fish(
                        receptor_locs=fish.get_receptors_locations(), 
                        E_field=E_field, 
                        return_potential=False, 
                        return_field=True, 
                        include_radius_factor=False
                    )[1]
                
                # put together the E field perturbation
                # manually include the radius factor (i.e. multiply by cubed radius)
                E_field_perturbation = torch.tensor(np.power(worm.get_radius(), 3)).to(device) * \
                                        torch.tensor(dict_perturbation_magnitude[i_3Dp][:,:,np.newaxis]).to(device) * \
                                        torch.tensor(dict_dipole_wave_forms[i_rRCsig]).to(device)
                # compute the perturbed electric image
                total_E_field = base_E_field_full + E_field_perturbation
                # pert_transdermal_signal = fish.compute_transdermal_signal(E_field=total_E_field, water_conductivity=aqua.get_conductivity())
                pert_transdermal_signal = ((torch.tensor(fish.receptors_normals[:,:,np.newaxis]).to(device) * total_E_field).sum(1) * \
                    torch.tensor(fish.get_skin_resistivity() * aqua.get_conductivity()).to(device)).cpu().numpy()
                #####################
                ####### consider "torch"-ing this as well
                pert_receptors_responses = fish.compute_receptors_responses(pert_transdermal_signal)
                #####################
                dataset['electric_images']['pert']['objs_ids'].append([i_aq, i_fi, i_wo])
                dataset_electric_images_pert_data.append(pert_receptors_responses)
                expectation_receptor_responses, expecation_of_squared_receptors_responses, number_of_responses = running_stream_computation(
                    pert_receptors_responses, expectation_receptor_responses, expecation_of_squared_receptors_responses, number_of_responses
                )
                if save_LEODs:
                    dataset_LEODs.append(pert_transdermal_signal)
                    expectation_LEODs, expectation_of_squared_LEODs, number_of_responses = running_stream_computation(
                        pert_transdermal_signal, expectation_LEODs, expectation_of_squared_LEODs, number_of_responses
                    )
                count_pert_EI += 1
                if (count_pert_EI % HDF5_save_period) == 0:
                    with h5py.File(f'{save_folder}/{save_file_name}.hdf5', 'r+') as f:
                        f['pert_EI'][(count_pert_EI-HDF5_save_period):count_pert_EI] = np.array(dataset_electric_images_pert_data)
                    dataset_electric_images_pert_data = []

                    if save_LEODs:
                        with h5py.File(f'{save_folder}/{save_file_name}_LEODs.hdf5', 'r+') as f:
                            f['pert_LEODs'][(count_pert_EI-HDF5_save_period):count_pert_EI] = np.array(dataset_LEODs)
                        dataset_LEODs = []
                    end_time_HDF5_save = time.time()
                    print(f'Saved {count_pert_EI // HDF5_save_period}*{HDF5_save_period} EIs ({end_time_HDF5_save-start_time_HDF5_save:.2f} s).', end=' ')
                    start_time_HDF5_save = end_time_HDF5_save

            aqua.remove_fish(fish)
            end_time = time.time()
            print(f'Time: {(end_time-start_time):.2f} s.', end=' ')
            keeper_id += 1
            estim_time_remaining = (end_time - main_start_time) / keeper_id * (len(dataset['fish']['fish_objs']) * len(dataset['aqua']['aqua_objs']) - keeper_id)
            print(f'Estimated time remaining: {estim_time_remaining/60:.2f} min / {estim_time_remaining/3600:.2f} h.')
    
    dataset['electric_images']['responses_avg'] = expectation_receptor_responses
    dataset['electric_images']['responses_std'] = np.sqrt(expecation_of_squared_receptors_responses - expectation_receptor_responses**2)
    if save_LEODs:
        dataset['electric_images']['LEODs_avg'] = expectation_LEODs
        dataset['electric_images']['LEODs_std'] = np.sqrt(expectation_of_squared_LEODs - expectation_LEODs**2)
    print(f'Saving {len(dataset_electric_images_pert_data)} additional EIs.', end=' ')
    if len(dataset_electric_images_pert_data) > 0:
        with h5py.File(f'{save_folder}/{save_file_name}.hdf5', 'r+') as f:
            f['pert_EI'][-len(dataset_electric_images_pert_data):] = np.array(dataset_electric_images_pert_data)
    if save_LEODs:
        if len(dataset_LEODs) > 0:
            with h5py.File(f'{save_folder}/{save_file_name}_LEODs.hdf5', 'r+') as f:
                f['pert_LEODs'][-len(dataset_LEODs):] = np.array(dataset_LEODs)
    end_time_HDF5_save = time.time()
    print(f'({end_time_HDF5_save-start_time_HDF5_save:.2f} s)')
    print('Saving data... ', end='')
    start_time = time.time()
    if not save_worm_objs:
        dataset['worms']['worm_objs'] = None
    dill.dump(dataset, open(f'{save_folder}/{save_file_name}.pkl', 'wb'), protocol=4)
    end_time=time.time()
    print(f'{((end_time - start_time) / 60):.3f} minutes.')
    print('Done!')
    return dataset


if __name__=='__main__':
    print('Simulate data...')
    save_LEODs = True
    for fname in sys.argv[1:]:
        data_params_dict = dill.load(open(fname, 'rb'))
        print(data_params_dict['save_file_name'])
        generate_receptors_responses(save_LEODs=save_LEODs, **data_params_dict)
        print()
        print()
    