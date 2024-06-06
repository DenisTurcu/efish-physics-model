import dill
import argparse
import numpy as np

from generate_receptors_responses import generate_receptors_responses


def parser_receptors_response_data_run() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Generate electrolocation data based on dicts of parameters provided via file names."
    )
    parser.add_argument(
        "--params_file_names",
        type=str,
        nargs="+",
        help="Name of files containing the parameters dicts for the simulation.",
    )
    parser.add_argument(
        "--HDF5_save_period",
        type=int,
        default=100000,
        help="The period in number of simulations at which the data is saved in the HDF5 file.",
    )
    parser.add_argument(
        "--HDF5_save_dtype",
        type=type,
        default=np.float32,
        help="The data type for saving arrays in the HDF5 file.",
    )
    parser.add_argument(
        "--save_LEODs",
        type=bool,
        default=False,
        help="Whether to save the local EODs in an HDF5 file.",
    )
    parser.add_argument(
        "--GPU",
        type=int,
        default=None,
        help="The GPU id to use for the simulation.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="processed",
        help="The folder where the data is saved.",
    )
    parser.add_argument(
        "--save_worm_objs",
        type=bool,
        default=False,
        help="Whether to save the worm objects.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_receptors_response_data_run()
    for fname in args.params_file_names:
        params_dict = dill.load(open(fname, "rb"))
        print(f"Processing parameters dict: {fname}...")

        generate_receptors_responses(
            save_name=params_dict["save_name"],
            # aquarium properties
            water_conductivities=params_dict["water_conductivities"],
            boundary_displacements=params_dict["boundary_displacements"],
            boundary_normals=params_dict["boundary_normals"],
            # fish properties
            fish_obj=params_dict["fish_obj"],
            fish_bend_angle_lateral_deg=params_dict["fish_bend_angle_lateral_deg"],
            fish_bend_angle_dorso_ventral_deg=params_dict["fish_bend_angle_dorso_ventral_deg"],
            fish_bend_location_percentages=params_dict["fish_bend_location_percentages"],
            fish_yaw_deg=params_dict["fish_yaw_deg"],
            fish_pitch_deg=params_dict["fish_pitch_deg"],
            fish_roll_deg=params_dict["fish_roll_deg"],
            # worm properties
            worm_resistances=params_dict["worm_resistances"],
            worm_capacitances=params_dict["worm_capacitances"],
            worm_radii=params_dict["worm_radii"],
            worm_position_xs=params_dict["worm_position_xs"],
            worm_position_ys=params_dict["worm_position_ys"],
            worm_position_zs=params_dict["worm_position_zs"],
            HDF5_save_period=args.HDF5_save_period,
            HDF5_save_dtype=args.HDF5_save_dtype,
            save_LEODs=args.save_LEODs,
            GPU=args.GPU,
            save_folder=args.save_folder,
            save_worm_objs=args.save_worm_objs,
        )
        print()
