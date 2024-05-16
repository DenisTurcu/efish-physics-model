import numpy as np

# helper functions to aggregate large data
def find_EI_ids(
        properties_ids_base,
        properties_ids_pert,
        properties_dict,
        water_conductivities_id,
        boundary_normals_id,
        boundary_displacements_id,
        tail_lateral_angle_id,
        tail_dor_ven_angle_id,
        tail_location_percent_id,
        fish_yaw_id,
        fish_pitch_id,
        fish_roll_id,
        resistances_id,
        capacitances_id,
        worm_radii_id,
        worm_xs_id,
        worm_ys_id,
        worm_zs_id,
        cancel_boundary=True,
        cancel_tail=True,
        cancel_rotation=True,
        default_boundary_displacement_id=0,
        default_tail_lateral_angle_id=4,
        default_tail_dor_ven_angle_id=0,
        default_yaw_id=0,
        default_pitch_id=0,
        default_roll_id=0,
    ):
    bound_disp_id = boundary_displacements_id if cancel_boundary else default_boundary_displacement_id 

    tail_lat_id   = tail_lateral_angle_id     if cancel_tail else default_tail_lateral_angle_id
    tail_dve_id   = tail_dor_ven_angle_id     if cancel_tail else default_tail_dor_ven_angle_id

    yaw_id = fish_yaw_id if cancel_rotation else default_yaw_id
    pitch_id = fish_pitch_id if cancel_rotation else default_pitch_id
    roll_id = fish_roll_id if cancel_rotation else default_roll_id

    base_EI_id = properties_ids_base[
        (properties_ids_base[:,properties_dict['water_conductivity']] == water_conductivities_id) & 
        (properties_ids_base[:,properties_dict['boundary_normals']] == boundary_normals_id) &
        (properties_ids_base[:,properties_dict['boundary_displacements']] == bound_disp_id) & 
        (properties_ids_base[:,properties_dict['tail_lateral_angle']] == tail_lat_id) & 
        (properties_ids_base[:,properties_dict['tail_dor_ven_angle']] == tail_dve_id) & 
        (properties_ids_base[:,properties_dict['tail_location_percent']] == tail_location_percent_id) & # this one doesn't really change ??
        (properties_ids_base[:,properties_dict['fish_yaw']] == yaw_id) & 
        (properties_ids_base[:,properties_dict['fish_pitch']] == pitch_id) & 
        (properties_ids_base[:,properties_dict['fish_roll']] == roll_id), 
        -1]

    pert_EI_id = properties_ids_pert[
        (properties_ids_pert[:,properties_dict['water_conductivity']] == water_conductivities_id) & 
        (properties_ids_pert[:,properties_dict['boundary_normals']] == boundary_normals_id) &
        (properties_ids_pert[:,properties_dict['boundary_displacements']] == boundary_displacements_id) & 
        (properties_ids_pert[:,properties_dict['tail_lateral_angle']] == tail_lateral_angle_id) & 
        (properties_ids_pert[:,properties_dict['tail_dor_ven_angle']] == tail_dor_ven_angle_id) & 
        (properties_ids_pert[:,properties_dict['tail_location_percent']] == tail_location_percent_id) & 
        (properties_ids_pert[:,properties_dict['fish_yaw']] == fish_yaw_id) & 
        (properties_ids_pert[:,properties_dict['fish_pitch']] == fish_pitch_id) & 
        (properties_ids_pert[:,properties_dict['fish_roll']] == fish_roll_id) & 
        (properties_ids_pert[:,properties_dict['resistances']] == resistances_id) & 
        (properties_ids_pert[:,properties_dict['capacitances']] == capacitances_id) & 
        (properties_ids_pert[:,properties_dict['worm_radii']] == worm_radii_id) & 
        (properties_ids_pert[:,properties_dict['worm_xs']] == worm_xs_id) & 
        (properties_ids_pert[:,properties_dict['worm_ys']] == worm_ys_id) & 
        (properties_ids_pert[:,properties_dict['worm_zs']] == worm_zs_id), 
        properties_dict['pert_EI']]
    
    print('Base: ', base_EI_id, ' | Pert: ', pert_EI_id)

    assert len(base_EI_id) == 1, 'Could not extract a single base EI id based on the provided information.'
    assert len(pert_EI_id) == 1, 'Could not extract a single pert EI id based on the provided information.'

    return base_EI_id[0], pert_EI_id[0]


def find_base_ids(
        properties_ids_base,
        properties_ids_pert,
        properties_dict,
        cancel_boundary=True,
        cancel_tail=True,
        cancel_rotation=True,
        default_boundary_displacement_id=0,
        default_tail_lateral_angle_id=4,
        default_tail_dor_ven_angle_id=0,
        default_yaw_id=0,
        default_pitch_id=0,
        default_roll_id=0,
    ):

    temp_pert_props = properties_ids_pert.copy()
    if not cancel_boundary:
        temp_pert_props[:,properties_dict['boundary_displacements']] = default_boundary_displacement_id 

    if not cancel_tail: 
        temp_pert_props[:,properties_dict['tail_lateral_angle']] = default_tail_lateral_angle_id
        temp_pert_props[:,properties_dict['tail_dor_ven_angle']] = default_tail_dor_ven_angle_id

    if not cancel_rotation:
        temp_pert_props[:,properties_dict['fish_yaw']] = default_yaw_id
        temp_pert_props[:,properties_dict['fish_pitch']] = default_pitch_id
        temp_pert_props[:,properties_dict['fish_roll']] = default_roll_id

    temp_base_ids = (properties_ids_base[np.newaxis,:,:-1] == temp_pert_props[:,np.newaxis,:properties_ids_base.shape[1]-1]).all(-1)
    assert (temp_base_ids.sum(1) == np.ones(properties_ids_pert.shape[0])).all(), 'There should be a single base_EI ID for a given pert_EI.'
    return np.where(temp_base_ids)[1]
