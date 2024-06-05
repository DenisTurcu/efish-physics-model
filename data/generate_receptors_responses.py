import sys
import copy
import time
import datetime
import dill
import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.append("objects/")
sys.path.append("../objects/")

from Aquarium import Aquarium  # noqa E402
from Aquarium_SinglePlane import SinglePlaneAquarium  # noqa E402
from FishGeneration import FishGeneration  # noqa E402
from Worm_SmallSpherical import SmallSphericalWorm  # noqa E402


def running_stream_computation(responses, expectation, expectation_of_squares, num_samples):
    expectation = (responses + num_samples * expectation) / (num_samples + 1)
    expectation_of_squares = (np.power(responses, 2) + num_samples * expectation_of_squares) / (num_samples + 1)
    return expectation, expectation_of_squares


def generate_receptors_responses(
    # save name for storing the data
    save_name: str,
    # aquarium properties
    water_conductivities: list | np.ndarray,
    boundary_normals: list[list] | list[np.ndarray] | np.ndarray,
    boundary_displacements: list[list] | list[np.ndarray] | np.ndarray,
    # fish properties
    fish_obj: FishGeneration,
    fish_bend_angle_lateral_deg: list[list] | list[np.ndarray],
    fish_bend_angle_dorso_ventral_deg: list[list] | list[np.ndarray],
    fish_bend_location_percentages: list[list] | list[np.ndarray],
    fish_yaw_deg: list | np.ndarray,
    fish_pitch_deg: list | np.ndarray,
    fish_roll_deg: list | np.ndarray,
    # worm properties
    worm_resistances: list | np.ndarray,
    worm_capacitances: list | np.ndarray,
    worm_radii: list | np.ndarray,
    worm_position_xs: list | np.ndarray,
    worm_position_ys: list | np.ndarray,
    worm_position_zs: list | np.ndarray,
    # additional parameters
    HDF5_save_period: int = 100000,
    HDF5_save_dtype: type = np.float32,
    save_LEODs: bool = False,
    GPU: int | None = None,
    save_worm_objs: bool = False,
    save_folder: str = "processed",
) -> dict:
    """Function to prepare and generate dataset of electric signals.
    Make sure that the parameters are self-consistent.

    Args:
        save_name (str): Folder name to save the data, within specified path via "save_folder".

        Aquarium properties:
            water_conductivities (list | np.ndarray): Conductivities of the water in the aquarium
            boundary_normals (list[list] | list[np.ndarray] | np.ndarray): Normals to the planar boundary surface.
            boundary_displacements (list[list] | list[np.ndarray] | np.ndarray): Displacements of the planar boundary
                from origin ([0, 0, 0]) in the direction of the provided boundary normal.

        Fish properties:
            fish_obj (FishGeneration): Base fish object to be used for the simulation. This base fish should have
                the base_receptors_locations such that the fish is straight, i.e. no bend or rotation angles,
                along +x axis with tail is at [0, 0, 0] and nose is at [self.length, 0, 0]
            fish_bend_angle_lateral_deg (list[list] | list[np.ndarray]): Lateral bend angles of the fish for simulating
                the dataset.
            fish_bend_angle_dorso_ventral_deg (list[list] | list[np.ndarray]): Vertical (dorso-ventral) bend angles of
                the fish for simulating the dataset.
            fish_bend_location_percentages (list[list] | list[np.ndarray]): Bend location percentages of the fish for
                simulating the dataset.
            fish_yaw_deg (list | np.ndarray): Yaw angles of the fish for simulating the dataset.
            fish_pitch_deg (list | np.ndarray): Pitch angles of the fish for simulating the dataset.
            fish_roll_deg (list | np.ndarray): Roll angles of the fish for simulating the dataset.

        Worm properties:
            worm_resistances (list | np.ndarray): Resistances of simulated nearby objects (worms) in the aquarium.
            worm_capacitances (list | np.ndarray): Capacitances of simulated nearby objects (worms) in the aquarium.
            worm_radii (list | np.ndarray): Radii of simulated nearby objects (worms) in the aquarium.
            worm_position_xs (list | np.ndarray): x-axis positions of simulated nearby objects (worms) in the aquarium.
            worm_position_ys (list | np.ndarray): y-axis positions of simulated nearby objects (worms) in the aquarium.
            worm_position_zs (list | np.ndarray): z-axis positions of simulated nearby objects (worms) in the aquarium.

        Additional parameters:
            HDF5_save_period (int, optional): Number of aquarium-fish-worm groups simulated before saving the data to
                file. This helps generate very large datasets without running out of RAM. Defaults to 100000.
            HDF5_save_dtype (type, optional): Data type for the save electric signals. Defaults to np.float32.
            save_LEODs (bool, optional): Whether to save the local EODs of the simulations. The receptor responses are
                always saved. Defaults to False.
            GPU (int | None, optional): GPU id to help speed up data simulation. Speedup performance depends on hardware
                abilities. Defaults to None.
            save_worm_objs (bool, optional): Whether to save the worm objects as the defined Python classes used in
                this framework. Typically, saving these objects should not be needed since their properties are easily
                summarized with much less memory. Defaults to False.
            save_folder (str, optional): Path to the location where to save the data. Defaults to "processed".

    Returns:
        dict: _description_
    """
    # Create the save folder if it doesn't exist
    Path(f"{save_folder}/{save_name}").mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    device = "cpu" if GPU is None else int(GPU)

    # initialize the dataset structure
    dataset = {}
    dataset["worms"] = {}
    dataset["fish"] = {}
    dataset["aquarium"] = {}
    dataset["electric_images"] = {}

    # initialize the aquarium's properties
    dataset["aquarium"]["conductivities"] = water_conductivities
    dataset["aquarium"]["boundary_normals"] = boundary_normals
    dataset["aquarium"]["boundary_displacements"] = boundary_displacements
    assert len(boundary_displacements) == len(
        boundary_normals
    ), "There must be the same number of normals as displacements lists in the dataset."

    # initialize the fish's properties
    dataset["fish"]["properties"] = {
        "bend_angle_lateral": 0,
        "bend_angle_dorso_ventral": 1,
        "bend_location_percentages": 2,
        "fish_yaw": 3,
        "fish_pitch": 4,
        "fish_roll": 5,
    }
    dataset["fish"]["bend_angle_lateral"] = fish_bend_angle_lateral_deg
    dataset["fish"]["bend_angle_dorso_ventral"] = fish_bend_angle_dorso_ventral_deg
    dataset["fish"]["bend_location_percentages"] = fish_bend_location_percentages
    dataset["fish"]["yaw"] = fish_yaw_deg
    dataset["fish"]["pitch"] = fish_pitch_deg
    dataset["fish"]["roll"] = fish_roll_deg

    # initialize the worms' properties
    dataset["worms"]["properties_dict"] = {
        "resistances": 0,
        "capacitances": 1,
        "radii": 2,
        "position_xs": 3,
        "position_ys": 4,
        "position_zs": 5,
    }
    dataset["worms"]["resistances"] = worm_resistances
    dataset["worms"]["capacitances"] = worm_capacitances
    dataset["worms"]["radii"] = worm_radii
    dataset["worms"]["position_xs"] = worm_position_xs
    dataset["worms"]["position_ys"] = worm_position_ys
    dataset["worms"]["position_zs"] = worm_position_zs

    # prepare the electric images structure
    dataset["electric_images"]["base"] = pd.DataFrame()  # dict(objs_ids=[], data=[], LEODs=[])
    dataset["electric_images"]["pert"] = pd.DataFrame()  # dict(objs_ids=[])
    dataset_electric_images_pert_data = []
    if save_LEODs:
        dataset_LEODs = []

    ############################################################################################################
    # Create the Object Lists ##################################################################################
    ############################################################################################################
    # create the aquarium objects
    list_df = []
    for i0 in range(len(dataset["aquarium"]["conductivities"])):
        for i1 in range(len(dataset["aquarium"]["boundary_normals"])):
            # get at least one displacement for the boundary normal to be able to create the aquarium object
            for i2 in range(max(len(dataset["aquarium"]["boundary_displacements"][i1]), 1)):
                list_df.append([i0, i1, i2])
    dataset["aquarium"]["dataframe"] = pd.DataFrame(
        list_df, columns=["conductivities", "boundary_normals", "boundary_displacements"]
    )
    aqua_objs = []
    for _, row in dataset["aquarium"]["dataframe"].iterrows():
        id_cond = row["conductivities"]
        id_norm = row["boundary_normals"]
        id_disp = row["boundary_displacements"]
        if (not dataset["aquarium"]["boundary_displacements"][id_norm]) or np.abs(
            dataset["aquarium"]["boundary_displacements"][id_norm][id_disp]
        ) > 1:
            aqua_objs.append(
                Aquarium(
                    relative_permittivity=80,
                    conductivity=dataset["aquarium"]["conductivities"][id_cond],
                    _init_tests=False,
                )
            )
        else:
            temp_normal = dataset["aquarium"]["boundary_normals"][id_norm]
            temp_central_point = np.array(temp_normal) * dataset["aquarium"]["boundary_displacements"][id_norm][id_disp]
            aqua_objs.append(
                SinglePlaneAquarium(
                    relative_permittivity=80,
                    conductivity=dataset["aquarium"]["conductivities"][id_cond],
                    boundaries=[("plane", dict(normal=temp_normal, central_point=temp_central_point))],
                    _init_tests=False,
                )
            )
    dataset["aquarium"]["dataframe"]["objs"] = aqua_objs

    # create the fish objects
    list_df = []
    for i0, temp0 in enumerate(dataset["fish"]["bend_angle_lateral"]):
        for i1, temp1 in enumerate(dataset["fish"]["bend_angle_dorso_ventral"]):
            for i2, temp2 in enumerate(dataset["fish"]["bend_location_percentages"]):
                # can only simulated fish for which the bend angles and locations have the same lengths
                if (len(temp0) == len(temp1)) and (len(temp1) == len(temp2)):
                    for i3 in range(len(dataset["fish"]["yaw"])):
                        for i4 in range(len(dataset["fish"]["pitch"])):
                            for i5 in range(len(dataset["fish"]["roll"])):
                                list_df.append([i0, i1, i2, i3, i4, i5])
                else:
                    print(
                        f"Fish bend details {i0}, {i1}, {i2} have different lengths: "
                        f"{len(temp0)}, {len(temp1)}, {len(temp2)}. Skipping..."
                    )
    dataset["fish"]["dataframe"] = pd.DataFrame(
        list_df,
        columns=["bend_angle_lateral", "bend_angle_dorso_ventral", "bend_location_percentages", "yaw", "pitch", "roll"],
    )
    fish_objs = []
    for _, row in dataset["fish"]["dataframe"].iterrows():
        temp_fish = copy.deepcopy(fish_obj)
        id_lat = row["bend_angle_lateral"]
        id_dve = row["bend_angle_dorso_ventral"]
        id_loc = row["bend_location_percentages"]
        id_yaw = row["yaw"]
        id_pit = row["pitch"]
        id_rol = row["roll"]
        temp_fish.update_parameters(
            nose_position=None,
            angle_yaw=(dataset["fish"]["yaw"][id_yaw], "deg"),
            angle_pitch=(dataset["fish"]["pitch"][id_pit], "deg"),
            angle_roll=(dataset["fish"]["roll"][id_rol], "deg"),
            relative_bend_locations_percentage=np.array(dataset["fish"]["bend_location_percentages"][id_loc]),
            relative_bend_angle_lateral=(np.array(dataset["fish"]["bend_angle_lateral"][id_lat]), "deg"),
            relative_bend_angle_dorso_ventral=(np.array(dataset["fish"]["bend_angle_dorso_ventral"][id_dve]), "deg"),
        )
        fish_objs.append(temp_fish)
    dataset["fish"]["dataframe"]["objs"] = fish_objs

    # create the worm obj list
    list_df = []
    for i0 in range(len(dataset["worms"]["resistances"])):
        for i1 in range(len(dataset["worms"]["capacitances"])):
            for i2 in range(len(dataset["worms"]["radii"])):
                for i3 in range(len(dataset["worms"]["position_xs"])):
                    for i4 in range(len(dataset["worms"]["position_ys"])):
                        for i5 in range(len(dataset["worms"]["position_zs"])):
                            list_df.append([i0, i1, i2, i3, i4, i5])
    dataset["worms"]["dataframe"] = pd.DataFrame(
        list_df, columns=["resistances", "capacitances", "radii", "position_xs", "position_ys", "position_zs"]
    )
    worm_objs = []
    for _, row in dataset["worms"]["dataframe"].iterrows():
        id_res = row["resistances"]
        id_cap = row["capacitances"]
        id_rad = row["radii"]
        id_pxs = row["position_xs"]
        id_pys = row["position_ys"]
        id_pzs = row["position_zs"]
        worm_objs.append(
            SmallSphericalWorm(
                radius=dataset["worms"]["radii"][id_rad],  # type: ignore
                resistance=dataset["worms"]["resistances"][id_res],  # type: ignore
                capacitance=dataset["worms"]["capacitances"][id_cap],  # type: ignore
                center_position=[
                    dataset["worms"]["position_xs"][id_pxs],
                    dataset["worms"]["position_ys"][id_pys],
                    dataset["worms"]["position_zs"][id_pzs],
                ],
                _init_tests=False,
            )
        )
    dataset["worms"]["dataframe"]["objs"] = worm_objs
    end_time = time.time()
    print(f"Time to prepare dataset: {end_time - start_time:.3f} s")
    print("Total aquarium-fish pairs: %d" % (len(dataset["fish"]["dataframe"]) * len(dataset["aquarium"]["dataframe"])))
    print("Total worms: %d" % len(dataset["worms"]["dataframe"]))

    ############################################################################################################
    # Compute electric images for every aquarium-fish-worm grouping ############################################
    ############################################################################################################
    print("Current aqua-fish pair: ")
    main_start_time = time.time()
    # store the dipole waveforms for each (R, C, sig_water) using the dict below
    #       - helps to avoid recomputing the waveform for each worm, generating the dataset faster
    #       - assumes there is a single EOD waveform among the fish
    #       - this doesn't change w.r.t to fish/aqua
    dict_dipole_wave_forms = {}
    # compute the running mean and standard deviation of each receptor's response
    expectation_receptor_responses = 0
    expectation_of_squared_receptors_responses = 0
    if save_LEODs:
        expectation_LEODs = 0
        expectation_of_squared_LEODs = 0
    # store and save data in hdf5 format
    keeper_id = 0  # id that keeps track of the current aquarium-fish pair (ignoring the worm)
    count_pert_EI = 0  # id that keeps track of the current aquarium-fish-worm group
    num_responses = 0  # id that keeps track of the number of responses computed (base and pert)
    with h5py.File(f"{save_folder}/{save_name}/responses.hdf5", "w") as f:
        f.create_dataset(
            "responses",
            (
                len(dataset["fish"]["dataframe"])
                * len(dataset["aquarium"]["dataframe"])
                * len(dataset["worms"]["dataframe"]),
                fish_obj.get_N_receptors(),
                fish_obj.get_N_filters(),
            ),
            dtype=HDF5_save_dtype,
        )
    if save_LEODs:
        with h5py.File(f"{save_folder}/{save_name}/leods.hdf5", "w") as f:
            f.create_dataset(
                "localEODs",
                (
                    len(dataset["fish"]["dataframe"])
                    * len(dataset["aquarium"]["dataframe"])
                    * len(dataset["worms"]["dataframe"]),
                    fish_obj.get_N_receptors(),
                    fish_obj.eod_length,
                ),
                dtype=HDF5_save_dtype,
            )
    for aqua_id, aqua_row in dataset["aquarium"]["dataframe"].iterrows():
        # identify the id of water conductivity
        aquarium = aqua_row["objs"]
        id_sig_water = aqua_row["conductivities"]
        for fish_id, fish_row in dataset["fish"]["dataframe"].iterrows():
            keeper_id += 1
            print(f"ID: {str(keeper_id).rjust(3)}.", end=" ")
            fish = fish_row["objs"]
            start_time = time.time()
            start_time_HDF5_save = start_time
            aquarium.insert_fish(fish)
            # compute the basal electric field at fish receptors
            _, base_E_field_mag, wave_form = aquarium.electric_potential_and_field_single_fish(
                points=fish.get_receptors_locations(), fish_id=0, return_potential=False, return_field=True
            )
            base_E_field_full = torch.tensor(base_E_field_mag).to(device) * torch.tensor(wave_form).to(device)
            # compute the basal electric image
            base_transdermal_signal = fish.compute_transdermal_signal(
                E_field=base_E_field_mag.squeeze().reshape(-1, 3),
                water_conductivity=aquarium.get_conductivity(),
                temporal_wave_form=wave_form,
            )
            base_receptors_responses = fish.compute_receptors_responses(base_transdermal_signal)
            expectation_receptor_responses, expectation_of_squared_receptors_responses = running_stream_computation(
                responses=base_receptors_responses,
                expectation=expectation_receptor_responses,
                expectation_of_squares=expectation_of_squared_receptors_responses,
                num_samples=num_responses,
            )
            if save_LEODs:
                expectation_LEODs, expectation_of_squared_LEODs = running_stream_computation(
                    responses=base_transdermal_signal,
                    expectation=expectation_LEODs,
                    expectation_of_squares=expectation_of_squared_LEODs,
                    num_samples=num_responses,
                )
            num_responses += 1
            dataset["electric_images"]["base"] = pd.concat(
                [
                    dataset["electric_images"]["base"],
                    pd.DataFrame(
                        dict(
                            aqua_id=aqua_id,
                            fish_id=fish_id,
                            receptors_responses=(base_receptors_responses,),
                            LEODs=None if not save_LEODs else (base_transdermal_signal,),
                        ),
                        index=[0],
                    ),
                ],
                axis=0,
                ignore_index=True,
            )

            # store the perturbation magnitudes for each 3D position (can change due
            # to fish position/tail/movement) using the dict below
            #       - helps to avoid recomputing the perturbation magnitudes for each worm,
            #           generating the dataset faster
            dict_perturbation_magnitude = {}
            for worm_id, worm_row in dataset["worms"]["dataframe"].iterrows():
                # identify the relevant ids for the worm
                count_pert_EI += 1
                worm = worm_row["objs"]
                id_res = worm_row["resistances"]
                id_cap = worm_row["capacitances"]
                id_rad = worm_row["radii"]
                id_pxs = worm_row["position_xs"]
                id_pys = worm_row["position_ys"]
                id_pzs = worm_row["position_zs"]

                # indices for rad, R, C and water conductivity
                #       these make up the keys of "dict_dipole_wave_forms"
                i_rRCsig = (id_rad, id_res, id_cap, id_sig_water)
                # indices for the 3D position
                #       these make up the keys of "dict_perturbation_magnitude"
                i_3Dp = (id_pxs, id_pys, id_pzs)

                # populate the dipole waveforms dictionary, if necessary
                if i_rRCsig not in dict_dipole_wave_forms:
                    dict_dipole_wave_forms[i_rRCsig] = worm.dipole_waveform_single_fish(
                        wave_form=wave_form,
                        sig_water=aquarium.get_conductivity(),
                        eps_r_water=aquarium.get_relative_permittivity(),
                        sampling_frequency=fish.get_sampling_rate(),
                    )
                # populate the perturbations magnitude dictionary, if necessary
                if i_3Dp not in dict_perturbation_magnitude:
                    # compute field at worm location
                    _, E_field, _ = aquarium.electric_potential_and_field_single_fish(
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
                        include_radius_factor=False,
                    )[1]

                # put together the E field perturbation
                # manually include the radius factor (i.e. multiply by radius^3)
                E_field_perturbation = (
                    torch.tensor(np.power(worm.get_radius(), 3)).to(device)
                    * torch.tensor(dict_perturbation_magnitude[i_3Dp][:, :, np.newaxis]).to(device)
                    * torch.tensor(dict_dipole_wave_forms[i_rRCsig]).to(device)
                )
                # compute the perturbed electric image
                total_E_field = base_E_field_full + E_field_perturbation
                # use ad-hoc torch computation for speedup in case GPUs are used (otherwise, computation commented
                # below is equivalent)
                #       pert_transdermal_signal = fish.compute_transdermal_signal(
                #           E_field=total_E_field, water_conductivity=aqua.get_conductivity()
                #       )
                pert_transdermal_signal = (
                    torch.tensor(fish.get_receptors_normals()[:, :, np.newaxis]).to(device) * total_E_field
                ).sum(1) * torch.tensor(fish.get_skin_resistivity() * aquarium.get_conductivity()).to(device)
                # use ad-hoc torch computation for speedup in case GPUs are used (otherwise, computation commented
                # below is equivalent)
                # pert_receptors_responses = fish.compute_receptors_responses(pert_transdermal_signal)
                pert_receptors_responses = torch.inner(
                    pert_transdermal_signal[:, -fish.get_EOD_length() :],  # noqa E203
                    torch.tensor(fish.get_receptors_filters()).to(device),
                )
                # convert back to numpy for storage purposes
                pert_transdermal_signal = pert_transdermal_signal.cpu().numpy()
                pert_receptors_responses = pert_receptors_responses.cpu().numpy()

                dataset["electric_images"]["pert"] = pd.concat(
                    [
                        dataset["electric_images"]["pert"],
                        pd.DataFrame(
                            dict(
                                aqua_id=aqua_id,
                                fish_id=fish_id,
                                worm_id=worm_id,
                            ),
                            index=[0],
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                )
                dataset_electric_images_pert_data.append(pert_receptors_responses)

                expectation_receptor_responses, expectation_of_squared_receptors_responses = running_stream_computation(
                    responses=pert_receptors_responses,
                    expectation=expectation_receptor_responses,
                    expectation_of_squares=expectation_of_squared_receptors_responses,
                    num_samples=num_responses,
                )
                if save_LEODs:
                    dataset_LEODs.append(pert_transdermal_signal)
                    expectation_LEODs, expectation_of_squared_LEODs = running_stream_computation(
                        responses=pert_transdermal_signal,
                        expectation=expectation_LEODs,
                        expectation_of_squares=expectation_of_squared_LEODs,
                        num_samples=num_responses,
                    )
                num_responses += 1

                if (count_pert_EI % HDF5_save_period) == 0:
                    with h5py.File(f"{save_folder}/{save_name}/responses.hdf5", "r+") as f:
                        f["responses"][(count_pert_EI - HDF5_save_period) : count_pert_EI] = np.array(  # type: ignore # noqa E203
                            dataset_electric_images_pert_data
                        )
                    dataset_electric_images_pert_data = []

                    if save_LEODs:
                        with h5py.File(f"{save_folder}/{save_name}/leods.hdf5", "r+") as f:
                            f["localEODs"][(count_pert_EI - HDF5_save_period) : count_pert_EI] = np.array(  # type: ignore # noqa E203
                                dataset_LEODs
                            )
                        dataset_LEODs = []
                    end_time_HDF5_save = time.time()
                    print(
                        (
                            f"Saved {count_pert_EI // HDF5_save_period}*{HDF5_save_period} EIs "
                            f"({end_time_HDF5_save-start_time_HDF5_save:.2f} s)."
                        ),
                        end=" ",
                    )
                    start_time_HDF5_save = end_time_HDF5_save

            aquarium.remove_fish(fish)
            end_time = time.time()
            print(f"Time: {(end_time-start_time):.2f} s.", end=" ")
            estimated_time_remaining = (
                (end_time - main_start_time)
                / keeper_id
                * (len(dataset["fish"]["dataframe"]) * len(dataset["aquarium"]["dataframe"]) - keeper_id)
            )
            estimated_datetime_finished = datetime.datetime.now() + datetime.timedelta(seconds=estimated_time_remaining)
            print(
                "Estimated time remaining: "
                f"{estimated_time_remaining // 3600} h "
                f"{estimated_time_remaining % 3600 // 60} min "
                f"{estimated_time_remaining % 60:.2f} s. "
                "Estimated datetime finished: "
                f"{estimated_datetime_finished.strftime('%Y_%m_%d-T-%H:%M:%S')}"
            )

    dataset["electric_images"]["responses_avg"] = expectation_receptor_responses
    dataset["electric_images"]["responses_std"] = np.sqrt(
        expectation_of_squared_receptors_responses - expectation_receptor_responses**2
    )
    if save_LEODs:
        dataset["electric_images"]["LEODs_avg"] = expectation_LEODs
        dataset["electric_images"]["LEODs_std"] = np.sqrt(expectation_of_squared_LEODs - expectation_LEODs**2)

    if len(dataset_electric_images_pert_data) > 0:
        print(f"Saving {len(dataset_electric_images_pert_data)} additional EIs.", end=" ")
        with h5py.File(f"{save_folder}/{save_name}/responses.hdf5", "r+") as f:
            f["responses"][-len(dataset_electric_images_pert_data) :] = np.array(dataset_electric_images_pert_data)  # type: ignore # noqa E203
        if save_LEODs:
            with h5py.File(f"{save_folder}/{save_name}/leods.hdf5", "r+") as f:
                f["localEODs"][-len(dataset_LEODs) :] = np.array(dataset_LEODs)  # type: ignore # noqa E203
        end_time_HDF5_save = time.time()
        print(f"({end_time_HDF5_save - start_time_HDF5_save:.2f} s)")

    print("Saving rest of data... ", end="")
    start_time = time.time()
    if not save_worm_objs:
        dataset["worms"]["dataframe"].drop(columns="objs", inplace=True)
    dill.dump(dataset, open(f"{save_folder}/{save_name}/dataset.pkl", "wb"), protocol=4)
    end_time = time.time()
    print(f"({((end_time - start_time) // 60):.2f} min {((end_time - start_time) % 60):.2f} s)")
    print("Done!")
    return dataset


if __name__ == "__main__":
    print("Simulate data...")

    HDF5_save_period = 100000
    HDF5_save_dtype = np.float32
    save_LEODs = False
    GPU = None
    save_folder = "processed"
    save_worm_objs = False

    for fname in sys.argv[1:]:
        data_params_dict = dill.load(open(fname, "rb"))
        print(f"Dataset name: {data_params_dict['save_name']}")
        generate_receptors_responses(
            **data_params_dict,
            save_LEODs=save_LEODs,
            GPU=GPU,
            save_folder=save_folder,
            save_worm_objs=save_worm_objs,
            HDF5_save_period=HDF5_save_period,
            HDF5_save_dtype=HDF5_save_dtype,
        )
        print()
