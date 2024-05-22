import time
import numpy as np
import plotly.graph_objects as go  # type: ignore
import contextlib
import os
import sys

sys.path.append("helper_functions/")
sys.path.append("../helper_functions/")

from prefixes import add_prefix  # noqa E402
from conversions import convert2mainSI  # noqa E402
from ElectricObject import ElectricObject  # noqa E402
from Fish import Fish  # noqa E402
from FishGeneration import FishGeneration  # noqa E402
from Fish_IceCreamCone import IceCreamConeFish  # noqa E402
from Worm import Worm  # noqa E402
from Boundaries import Boundary, Plane  # noqa E402


class Aquarium(ElectricObject):
    """Represents an aquarium for the fish. Can contain multiple fish and worms. Parent class "ElectricObject":"""

    __doc__ += ElectricObject.__doc__  # type: ignore

    def __init__(
        self,
        fish_objs: list = [],
        worm_objs: list = [],
        boundaries: list[tuple[str, dict]] = [],  # [('plane', dict(normal=[0,0,1],central_point=[0,0,0]))],
        _init_tests: bool = True,
        **kwds,
    ):
        """Initialize the aquarium for the fish.

        Args:
            fish_objs (list, optional): List of fish objects present in the aquarium. Defaults to [].
            worm_objs (list, optional): List of worm objects present in the aquarium. Defaults to [].
            boundaries (list[tuple[str, dict]], optional): List of boundaries present in the aquarium. Each list
                element should be a tuple containing the boundary type and properties passed to that boundary type
                for initialization (properties provided as dict). Defaults to [].

        Parent __doc__:\n
        """
        Aquarium.__init__.__doc__ += super().__init__.__doc__  # type: ignore

        # initialize parent class properties
        super().__init__(**kwds, _init_tests=False)

        # initialize aquarium boundaries
        self.initialize_boundaries(boundaries)
        self._initialize_input_argument_names()

        # initialize objects in aquarium by emptying it
        self.empty_aquarium()

        # insert provided objects
        for fish in fish_objs:
            self.insert_fish(fish)
        for worm in worm_objs:
            self.insert_worm(worm)

        # Run assertion tests
        if _init_tests:
            print(self.run_tests())

    def initialize_boundaries(self, boundaries: list[tuple[str, dict]]):
        """Initialize this aquarium's boundaries.

        Args:
            boundaries (list[tuple[str, dict]]): List of boundaries present in the aquarium. Each list
                element should be a tuple containing the boundary type and properties passed to that boundary type
                for initialization (properties provided as dict).

        Raises:
            NotImplementedError: If boundary type not yet implemented.
        """

        self.boundaries = []
        for boundary in boundaries:
            if boundary[0].lower() == "plane":
                # self.boundaries.append(Plane(normal=boundary[1]["normal"],
                # central_point=boundary[1]["central_point"]))
                self.boundaries.append(Plane(**boundary[1]))
            else:
                raise NotImplementedError(f'Boundary type "{boundary[0].lower()}" note yet implemented')
        pass

    def _insert_object(
        self,
        obj_list: list[tuple | Worm | Fish],
        obj: tuple | Worm | Fish,
        obj_instance: type,
        image_currents: bool = False,
    ) -> list[tuple | Worm | Fish]:
        """Insert the given object in the aquarium. If the object is a fish, insert/update its associated
        image currents as well.

        Args:
            obj_list (list[tuple  |  Worm  |  Fish]): The list of objects that needs to be updated.
            obj (tuple | Worm | Fish): The objects that needs to be inserted.
            obj_instance (type): The instance of the object that needs to be inserted.
            image_currents (bool, optional): Whether the objects that need to be updated are related to image currents.
                Defaults to False.

        Returns:
            list[tuple | Worm | Fish]: List with the updated objects.
        """
        assert isinstance(obj, obj_instance), f"Given object must be a {obj_instance}."
        # avoid duplicate objects - compare given object with all objects in the list
        #       (unless image currents are being updated, which have different structure)
        if not image_currents:
            for o in obj_list:
                if obj.is_equal(o):  # type: ignore
                    print(f"Object already exists in the aquarium:\n-----\n{obj}-----\n")
                    return obj_list
        obj_list.append(obj)
        # update image currents if object is a fish
        if isinstance(obj, Fish):
            self.update_image_currents(obj, mode="insert")
        return obj_list

    def _remove_object(
        self,
        obj_list: list[tuple | Worm | Fish],
        obj: tuple | Worm | Fish,
        obj_instance: type,
        image_currents: bool = False,
    ) -> list[tuple | Worm | Fish]:
        """Remove the given object in the aquarium. If the object is a fish, remove/update its associated
        image currents as well.

        Args:
            obj_list (list[tuple  |  Worm  |  Fish]): The list of objects that needs to be updated.
            obj (tuple | Worm | Fish): The objects that needs to be removed.
            obj_instance (type): The instance of the object that needs to be removed.
            image_currents (bool, optional): Whether the objects that need to be updated are related to image currents.
                Defaults to False.

        Returns:
            list[tuple | Worm | Fish]: List with the updated objects.
        """
        assert isinstance(obj, obj_instance), f"Given object must be a {obj_instance}."
        # build up list of the objects that will remain in the aquarium from scratch
        temp_obj_list = []
        for o in obj_list:
            # if removing image currents, compare the fish objects generating the image currents
            if image_currents:
                if not obj.is_equal(o[0]):  # type: ignore
                    temp_obj_list.append(o)
            # if removing worm/fish, compare the objects themselves
            else:
                if not obj.is_equal(o):  # type: ignore
                    temp_obj_list.append(o)
                # if fish is being removed, update its image currents
                else:
                    if isinstance(obj, Fish):
                        self.update_image_currents(obj, mode="remove")
        if len(temp_obj_list) == len(obj_list):
            print(f"Object was not found in the aquarium:\n-----\n{obj}-----\n")
        return temp_obj_list

    def empty_aquarium(self):
        """Empty the aquarium of all fish and worms."""
        self.remove_all_fish()
        self.remove_all_worms()
        pass

    def remove_all_fish(self):
        """Remove all fish from aquarium, as well as their image currents."""
        self.fish_objs = []
        self.image_point_currents_magnitude = []
        self.image_point_currents_location = []
        self.time_stamps = np.array([])
        pass

    def remove_all_worms(self):
        """Remove all worms from aquarium."""
        self.worm_objs = []
        pass

    def update_image_currents(self, obj: Fish, mode: str = "insert"):
        """Update the image currents generated by the boundaries. Needs to be implemented according
        to boundary type for each aquarium boundary. This base implementation can be build upon in
        specific types of aquariums.

        Args:
            obj (Fish): Fish object for which to update image currents.
            mode (str, optional): "insert" or "remove". Defaults to "insert".
        """

        if mode.lower() == "insert":
            self.image_point_currents_magnitude = self._insert_object(
                self.image_point_currents_magnitude, (obj, np.array([])), tuple, image_currents=True
            )
            self.image_point_currents_location = self._insert_object(
                self.image_point_currents_location, (obj, np.array([]).reshape([0, 3])), tuple, image_currents=True
            )
        elif mode.lower() == "remove":
            self.image_point_currents_magnitude = self._remove_object(
                self.image_point_currents_magnitude, obj, Fish, image_currents=True
            )
            self.image_point_currents_location = self._remove_object(
                self.image_point_currents_location, obj, Fish, image_currents=True
            )
        pass

    def insert_fish(self, obj: Fish, _check_points: bool = True):
        """Insert a fish in the aquarium.

        Args:
            obj (Worm): Fish object to insert.
            _check_points (bool, optional): Whether to check that given object is within aquarium boundaries.
                Defaults to True.
        """
        assert isinstance(obj, Fish), "Given object must be a fish"
        if _check_points:
            assert self.verify_points(
                obj.get_receptors_locations(), obj=obj
            ), "Given fish must lie within aquarium boundaries."
        for fish in self.fish_objs:
            assert (
                obj.sampling_rate - fish.sampling_rate  # type: ignore
            ) < self.assert_err, "All fish in the Aquarium must be adjusted to have the same sampling rates of the eod."
        self.fish_objs = self._insert_object(self.fish_objs, obj, Fish)
        self.update_time_stamps()
        pass

    def insert_worm(self, obj: Worm, _check_points: bool = True):
        """Insert a worm in the aquarium.

        Args:
            obj (Worm): Worm object to insert.
            _check_points (bool, optional): Whether to check that given object is within aquarium boundaries.
                Defaults to True.
        """
        assert isinstance(obj, Worm), "Given object must be a worm"
        if _check_points:
            assert self.verify_points(obj.get_points(), obj=obj), "Given worm must lie within aquarium boundaries."
        self.worm_objs = self._insert_object(self.worm_objs, obj, Worm)
        pass

    def remove_fish(self, obj: Fish):
        """Remove a fish from the aquarium."""
        self.fish_objs = self._remove_object(self.fish_objs, obj, Fish)
        self.update_time_stamps()
        pass

    def remove_worm(self, obj: Worm):
        """Remove a worm from the aquarium."""
        self.worm_objs = self._remove_object(self.worm_objs, obj, Worm)
        pass

    def electric_potential_and_field_single_fish(
        self,
        points: np.ndarray,
        fish_id: int,
        return_potential: bool = True,
        return_field: bool = True,
        include_image_point_currents: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        """Returns electric potential/field computed at given points due to a single fish.
        Implemented for image currents child classes - other types of aquariums or boundaries
        that do not use image currents will need to implement their own.

        Args:
            points (np.ndarray): The points at which to compute electric potential/field.
            fish_id (int): The fish ID among the fish in the aquarium.
            return_potential (bool, optional): Whether to return electric POTENTIAL. Defaults to True.
            return_field (bool, optional): Whether to return electric FIELD. Defaults to True.
            include_image_point_currents (bool, optional): Whether to include the image point currents for
                field/potential computations. Defaults to True.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None, np.ndarray]: Electric potential, electric field, waveform.
        """
        # get the point currents from the fish; magnitude and location
        N_time_steps = self.time_stamps.shape[0]
        fish = self.fish_objs[fish_id]
        pc_mag = fish.get_point_currents_magnitude()  # type: ignore
        pc_loc = fish.get_point_currents_location()  # type: ignore
        # get the image point currents, computed according to the boundary
        assert fish.is_equal(  # type: ignore
            self.image_point_currents_magnitude[fish_id][0]  # type: ignore
        ), "Currents and images should correspond to the same fish."
        assert fish.is_equal(  # type: ignore
            self.image_point_currents_location[fish_id][0]  # type: ignore
        ), "Currents and images should correspond to the same fish."
        # get the IMAGE point currents from the fish; magnitude and location
        ipc_mag = (
            np.empty(shape=0)
            if not include_image_point_currents
            else self.image_point_currents_magnitude[fish_id][1]  # type: ignore
        )
        ipc_loc = (
            np.empty(shape=(0, 3))
            if not include_image_point_currents
            else self.image_point_currents_location[fish_id][1]  # type: ignore
        )
        # put together magnitudes and locations
        all_mag = np.hstack([pc_mag, ipc_mag])
        all_loc = np.vstack([pc_loc, ipc_loc])

        # process wave-form
        waveform = fish.get_eod_wave_form()  # type: ignore
        if waveform.shape[0] < N_time_steps:
            waveform = np.hstack([waveform, np.zeros(N_time_steps - waveform.shape[0])])

        # compute the relative points into shape N_points x N_currents x 3
        relative_points = points[:, np.newaxis] - all_loc  # N_points x N_currents x 3
        norm_relative_points = np.linalg.norm(relative_points, axis=2)  # N_points x N_currents
        individual_potential = all_mag / (4 * self.sig * np.pi * norm_relative_points)  # N_points x N_currents
        potential = None if not return_potential else individual_potential.sum(1)[:, np.newaxis]  # N_points x 1
        field = (
            None
            if not return_field
            else ((individual_potential / np.power(norm_relative_points, 2))[:, :, np.newaxis] * relative_points).sum(
                1
            )[:, :, np.newaxis]
        )  # N_points x 3 x 1
        return potential, field, waveform

    def electric_potential_and_field(
        self, points: np.ndarray, return_potential: bool = True, return_field: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Returns electric potential/field computed at given points due to all fish in the aquarium and
        including the wave form in the last dimension.
        Implemented for image currents child classes - other types of aquariums or boundaries
        that do not use image currents will need to implement their own.

        Args:
            points (np.ndarray): The points at which to compute electric potential/field.
            return_potential (bool, optional): Whether to return electric POTENTIAL. Defaults to True.
            return_field (bool, optional): Whether to return electric FIELD. Defaults to True.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: Electric potential, electric field (including wave form
                in last dimension)
        """
        assert (
            len(self.fish_objs) > 0
        ), "There must be at least one fish in the aquarium to compute electric potential/field."
        _, points = self.verify_points(points=points, simple_return=False)  # type: ignore

        N_time_steps = self.time_stamps.shape[0]
        potential = np.zeros([points.shape[0], N_time_steps]) if return_potential else None
        field = np.zeros([points.shape[0], points.shape[1], N_time_steps]) if return_field else None

        # for every fish, compute the electric potential, electric field and wave form and combine them
        for i in range(len(self.fish_objs)):
            temp_potential, temp_field, waveform = self.electric_potential_and_field_single_fish(
                points, i, return_potential, return_field
            )
            if return_potential:
                potential += temp_potential * waveform  # type: ignore
            if return_field:
                field += temp_field * waveform  # type: ignore
        if return_potential:
            assert (
                potential.shape == np.array([points.shape[0], N_time_steps])  # type: ignore
            ).all(), "Returned electric potential should be shape (N x T)"  # type: ignore
        if return_field:
            assert (
                field.shape == np.array([points.shape[0], 3, N_time_steps])  # type: ignore
            ).all(), "Returned electric field should be shape (N x 3 x T)"  # type: ignore
        return potential, field

    def transdermal_signal_for_all_fish(self, include_perturbations: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Compute the electric signals for all fish in the aquarium.

        Args:
            include_perturbations (bool, optional): Whether or not to include the perturbations due to worms.
                Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray]: All points/receptors of all fish and the signal they received.
        """
        # gather all receptors from all fish
        points = np.array([]).reshape(0, 3)
        for fish in self.fish_objs:
            points = np.vstack([points, fish.get_receptors_locations()])  # type: ignore

        # compute the electric field from all active current generating objects in the aquarium, i.e. all fish
        _, E_field = self.electric_potential_and_field(points, return_potential=False)
        assert (
            E_field.shape == np.array([points.shape[0], 3, self.time_stamps.shape[0]])  # type: ignore
        ).all(), "Electric field should be shape (N x 3 x T)"  # type: ignore

        # compute the electric field from all passive objects in the aquarium, i.e. all worms
        if include_perturbations:
            for worm in self.worm_objs:
                _, E_field_from_worm = worm.perturbation_trace(  # type: ignore
                    points, self, return_potential=False, return_field=True
                )
                E_field += E_field_from_worm

        # compute the electric signal for each fish
        electric_signal = np.zeros([E_field.shape[0], E_field.shape[2]])  # type: ignore
        start_id = 0
        for fish in self.fish_objs:
            end_id = start_id + fish.get_N_receptors()  # type: ignore
            electric_signal_current_fish = fish.compute_transdermal_signal(  # type: ignore
                E_field=E_field[start_id:end_id], water_conductivity=self.sig  # type: ignore
            )
            assert (
                electric_signal_current_fish.shape == np.array([end_id - start_id, self.time_stamps.shape[0]])
            ).all(), "Electric image for current fish should be shape (N_points x T). "  # type: ignore
            electric_signal[start_id:end_id] = electric_signal_current_fish
            start_id = end_id
        return points, electric_signal

    def select_bounded_points(
        self, points: np.ndarray, simple_return: bool
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Select only the given points which are within the aquarium's boundaries.

        Args:
            points (np.ndarray): Points to be filtered based on the aquarium's boundaries.
            simple_return (bool): Whether to return the filtered points only, or return the filtered points along
                with the IDs for the points that are within the boundaries.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: Points that are within the boundaries. (including their IDs
                if simple_return=False)
        """
        within_boundaries_ids = np.zeros(points.shape[0]) == np.zeros(points.shape[0])
        for boundary in self.boundaries:
            truth_val_current, points = boundary.verify_points(points, simple_return=False)
            within_boundaries_ids &= truth_val_current
        if simple_return:
            return points[within_boundaries_ids]
        else:
            return points[within_boundaries_ids], within_boundaries_ids

    def verify_points(
        self, points: list | np.ndarray, obj: tuple | Worm | Fish | None = None, simple_return: bool = True
    ) -> bool | tuple[bool, list | np.ndarray]:
        """Verify that provided points are located within the boundary limits of the aquarium. E.g. for planar
        boundary this means on the correct side of the plane, for spherical this means inside the sphere etc.
        Args:
            points (list | np.ndarray): Given points locations to test for positioning with respect to the boundary.
            obj (tuple | Worm | Fish | None, optional): Objects whose points are being tested. Defaults to None.
            simple_return (bool, optional): Whether to return the truth value of the test, or return the processed
                points as well. Defaults to True.

        Returns:
            bool | tuple: Truth value of test, or truth value with the points themselves.
        """

        truth_val = True
        failed_boundaries = []
        for boundary in self.boundaries:
            truth_val_current, points = boundary.verify_points(points, simple_return=False)
            if not truth_val_current.all():
                failed_boundaries.append(boundary)
                truth_val = False
        assert truth_val, (
            f"Some points lie outside boundaries {failed_boundaries}." ""
            if obj is None
            else f"These points belong to object {obj}, with details:\n{obj.details()}"  # type: ignore
        )
        if simple_return:
            return True
        else:
            return True, points

    def update_time_stamps(self):
        """Update the time stamps of the Aquarium. Time 0 is the start of a simulation (start of the EOD trace
        of the first fish to emit an EOD).
            - this method assumes that all fish have the same sampling frequency of the EOD.
            - it does not remove leading timestamps where no fish have started and EOD, but removes trailing timestamps
            where all fish have stopped emitting EODs.
        """
        self.time_stamps = np.array([])
        for fish in self.fish_objs:
            fish_time_stamps = fish.get_time_stamps()  # type: ignore
            if fish_time_stamps.shape[0] > self.time_stamps.shape[0]:
                self.time_stamps = fish_time_stamps
        pass

    def visualize_aquarium(
        self,
        potential: np.ndarray | None = None,
        E_field: np.ndarray | None = None,
        electric_signal: np.ndarray | None = None,
        time_points: int | list | tuple = 10,
        shown_time_prefix: str = "",
        range_points: int = 30,
        lengths_prefix: str = "",
        show_fish: int = 0,
        show_electric_signal: bool = False,
        show_normals: int = 0,
        show_point_currents: int = 0,
        show_image_currents: int = 0,
        show_worms: bool | int = False,
        show_boundaries: bool = False,
        show_potential: bool = False,
        show_field: bool = False,
        potential_double_sided_percentile: float = 10,
        E_field_double_sided_percentile: float = 10,
        include_active_objects: bool = False,
        include_passive_objects: bool = False,
        electric_signal_perturbation_only: bool = False,
        xaxis_title: str = "X AXIS TITLE",
        yaxis_title: str = "Y AXIS TITLE",
        zaxis_title: str = "Z AXIS TITLE",
    ):
        """Visualize and aquarium with fish, worms, boundaries, electric potential and electric field.
        Can visualize
            - the electric signal on the fish
            - the 3D potential (3D equipotential contours) and electric field (3D cones)
            - point currents of fish and image point currents due to boundaries
            - worms

        Args:
            potential (np.ndarray | None, optional): Electric potential for visualization.
                If None but "show_potential=True", it will be computed here. Defaults to None.
            E_field (np.ndarray | None, optional): Electric field for visualization.
                If None but "show_field=True", it will be computed here. Defaults to None.
            electric_signal (np.ndarray | None, optional): Electric signal on fish's skin for visualization.
                If None but "show_electric_signal=True", it will be computed here. Defaults to None.
            time_points (int | list | tuple, optional): Provides the time points during the wave form to
                create individual frames for visualization. Defaults to 10.
                    - int -> number of linearly spaced time points between 1/3 and 2/3 of the wave form duration
                    - list -> indices into self.time_stamps to select frame_times
                    - tuple -> times to select frame_times, with prefix for the time stamps. E.g. ([1,2,3], "ms")
            shown_time_prefix (str, optional): Whether or not to show the time prefix on the plot. Defaults to "".
            range_points (int, optional): Number of mesh-grid points in each dimension to use for generating
                the mesh-grid. Defaults to 30.
            lengths_prefix (str, optional): Prefixes of the spatial variables shown in the visualization,
                e.g. "c" for centimeter. Defaults to "".
            show_fish (int, optional): Whether to show the fish receptors in the visualization. Magnitude dictates
                the size of the scatter marks. Defaults to 0.
            show_electric_signal (bool, optional): Whether to show the electric signal for all receptors in the
                visualization. Defaults to False.
            show_normals (int, optional): Whether to show the normal direction to the skin surface of the fish for
                all receptors. Magnitude dictates the size of the cones. Defaults to 0.
            show_point_currents (int, optional): Whether to show the point currents generating the discharge.
                Magnitude dictates the size of the scatter marks. Defaults to 0.
            show_image_currents (int, optional): Whether to show the image point currents generated by aquarium
                boundaries contributing to electric potential and field. Magnitude dictates the size of the scatter
                marks. Defaults to 0.
            show_worms (bool | int, optional): Whether to show the worms in the aquarium. Defaults to False.
            show_boundaries (bool, optional): Whether to show the boundaries. Defaults to False.
                BOundary visualization not yet implemented.
            show_potential (bool, optional): Whether to show the electric potential as equipotential contours.
                Defaults to False.
            show_field (bool, optional): Whether to show the electric field as 3D cones. Defaults to False.
            potential_double_sided_percentile (float, optional): Truncate electric potentials that are very large in
                absolute value for cleaner visualization. Defaults to 10.
            E_field_double_sided_percentile (float, optional): Truncate electric fields that are very large in
                absolute value for cleaner visualization. Defaults to 10.
            include_active_objects (bool, optional): Whether to include fish (active objects) for the visualization.
                Defaults to False.
            include_passive_objects (bool, optional): Whether to include worms (passive objects) for the visualization.
                Defaults to False.
            electric_signal_perturbation_only (bool, optional): Whether to show the full signal or "perturbed - base".
                Defaults to False.
            xaxis_title (str, optional): Axes label. Defaults to "X AXIS TITLE".
            yaxis_title (str, optional): Axes label. Defaults to "Y AXIS TITLE".
            zaxis_title (str, optional): Axes label. Defaults to "Z AXIS TITLE".

        Raises:
            ValueError: If "time_points" is not of the correct type (int, list or tuple).
            NotImplementedError: If boundary visualization is requested - boundary visualization not yet implemented.
            ValueError: If not enough frames to visualize - "time_points" should be chosen such that at least one
                frame is created.
        """
        start_time = time.time()
        ########################################################################
        # find aquarium (x,y,z) ranges based on the objects inside
        points = np.array([]).reshape(0, 3)
        for fish in self.fish_objs:
            points = np.vstack([points, fish.get_receptors_locations()])  # type: ignore
        for worm in self.worm_objs:
            points = np.vstack([points, worm.get_position()])  # type: ignore
        min_ranges = points.min(0)
        max_ranges = points.max(0)
        # del_ranges = max_ranges - min_ranges
        # min_ranges = min_ranges - del_ranges/2
        # max_ranges = max_ranges + del_ranges/2
        min_ranges = np.array([-15, -10, -10]) / 100
        max_ranges = np.array([5, 10, 10]) / 100

        ########################################################################
        # determine the frame times for the animation
        frame_times = []
        if isinstance(time_points, int):
            frame_times = [
                int(x)
                for x in np.linspace(self.time_stamps.shape[0] / 3, 2 * self.time_stamps.shape[0] / 3, time_points)
            ]
        elif isinstance(time_points, list):
            for tp in time_points:
                assert isinstance(tp, int) or isinstance(
                    tp, np.int64  # type: ignore
                ), '"time_points" should be a list of ints, which index into self.time_stamps to select frame_times.'
                assert tp >= 0, '"time_points" should not contain negative elements.'
                assert (
                    tp < self.time_stamps.shape[0]
                ), '"time_points" should be bounded from above by number of time stamps.'
            time_points.sort()
            frame_times = time_points
        elif isinstance(time_points, tuple):
            time_points = convert2mainSI(time_points)  # type: ignore
            time_points.sort()  # type: ignore
            frame_times = list((time_points[:, np.newaxis] > self.time_stamps).argmin(1))  # type: ignore
        else:
            raise ValueError('"time_points" should be of correct type: int, list of ints or tuple of list and prefix.')
        prefixed_time_stamps = add_prefix(self.time_stamps, shown_time_prefix)

        ########################################################################
        # potential or field generation
        if show_potential or show_field:
            # build the mesh-grids for plotting
            # complex number step means that the range_points is the number of points
            # in each dimension (instead of step size)
            X, Y, Z = np.mgrid[
                min_ranges[0] : max_ranges[0] : range_points * 1j,  # noqa E231
                min_ranges[1] : max_ranges[1] : range_points * 1j,  # noqa E231
                min_ranges[2] : max_ranges[2] : range_points * 1j,  # noqa E231
            ]
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            mesh_grid_points = np.vstack([X, Y, Z]).T

            mesh_grid_points, within_boundary_ids = self.select_bounded_points(mesh_grid_points, simple_return=False)
            print(f"Number of mesh-grid points included in the 3D plot: {within_boundary_ids.sum()}")
            X = X[within_boundary_ids]
            Y = Y[within_boundary_ids]
            Z = Z[within_boundary_ids]

            # compute potentials or fields
            if (potential is None) or (E_field is None):
                potential = 0  # type: ignore
                E_field = 0  # type: ignore

                if include_active_objects:
                    pot, E_f = self.electric_potential_and_field(mesh_grid_points, show_potential, show_field)
                    if show_potential:
                        potential += pot  # type: ignore
                    if show_field:
                        E_field += E_f  # type: ignore

                if include_passive_objects:
                    for worm in self.worm_objs:
                        pot, E_f = worm.perturbation_trace(  # type: ignore
                            mesh_grid_points, self, show_potential, show_field
                        )
                        if show_potential:
                            potential += pot
                        if show_field:
                            E_field += E_f

            if show_potential:
                assert (
                    potential.shape == np.array([X.shape[0], self.time_stamps.shape[0]])  # type: ignore
                ).all(), "Potential should be shape (N x T)."  # type: ignore
                potential = potential[:, frame_times]  # type: ignore
                # eliminate large potentials (in absolute value) for visualization
                potential_abs = np.abs(potential)
                max_potential_abs_in_time = potential_abs[:, potential_abs.max(0).argmax()]
                percentile_potential = np.percentile(max_potential_abs_in_time, 100 - potential_double_sided_percentile)
                potential_abs[potential_abs > percentile_potential] = percentile_potential
                potential_final = potential_abs * np.sign(potential)
                # color scale for plotting
                potential_color_bound = np.max(np.abs([potential.min(), potential.max()]))

            if show_field:
                E_field_norm = np.linalg.norm(E_field, 1)  # type: ignore
                max_over_time_id = E_field_norm.max(0).argmax()
                temp_E_field = E_field_norm[:, max_over_time_id]  # type: ignore
                bounded_E_field_locations = temp_E_field < np.percentile(
                    temp_E_field, 100 - E_field_double_sided_percentile
                )
                X_E_f = X[bounded_E_field_locations]
                Y_E_f = Y[bounded_E_field_locations]
                Z_E_f = Z[bounded_E_field_locations]

                E_field = E_field[bounded_E_field_locations]  # type: ignore
                assert (
                    E_field.shape == np.array([X_E_f.shape[0], 3, self.time_stamps.shape[0]])
                ).all(), "E field should be shape (N x 3 x T)."  # type: ignore
                E_field = E_field[:, :, frame_times]
                # color scale for plotting
                E_field_norm = np.sqrt(np.power(E_field, 2).sum(1))
                E_field_color_bound = np.max(np.abs([E_field_norm.min(), E_field_norm.max()]))

        ########################################################################
        # electric image generation
        if show_fish and show_electric_signal:
            if electric_signal is None:
                _, electric_signal = self.transdermal_signal_for_all_fish(include_perturbations=True)
                if electric_signal_perturbation_only:
                    _, base_signal = self.transdermal_signal_for_all_fish(include_perturbations=False)
                    electric_signal = electric_signal - base_signal
            assert electric_signal.shape[1] == self.time_stamps.shape[0], "Electric signal should have time length T."
            electric_signal = electric_signal[:, frame_times]
        end_time = time.time()
        print("COMPUTING TIME: ", end_time - start_time, " s.")

        ########################################################################
        # generate frames for animation ########################################
        ########################################################################
        # initialize frames
        frames = []
        for i in range(len(frame_times)):
            # initialize data and name for current frame
            frame_data = []
            frame_name = f"{prefixed_time_stamps[frame_times[i]]:.2f} {shown_time_prefix}s"  # type: ignore

            ########################################################################
            # show the potential
            if show_potential:
                graph_obj = go.Volume(
                    x=add_prefix(X, lengths_prefix),
                    y=add_prefix(Y, lengths_prefix),
                    z=add_prefix(Z, lengths_prefix),
                    value=potential_final[:, i],
                    opacity=0.1,
                    surface_count=40,
                    showscale=False,
                    colorscale="Picnic",
                    showlegend=False,
                    cmin=-potential_color_bound,  # type: ignore
                    cmax=potential_color_bound,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                )
                frame_data.append(graph_obj)

            ########################################################################
            # show the field
            if show_field:
                graph_obj = go.Cone(
                    x=add_prefix(X_E_f, lengths_prefix),
                    y=add_prefix(Y_E_f, lengths_prefix),
                    z=add_prefix(Z_E_f, lengths_prefix),
                    u=E_field[:, 0, i],  # type: ignore
                    v=E_field[:, 1, i],  # type: ignore
                    w=E_field[:, 2, i],  # type: ignore
                    colorscale="Blues",
                    sizemode="scaled",
                    sizeref=0,
                    showlegend=False,
                    opacity=0.8,
                    showscale=False,
                    cmin=-E_field_color_bound,  # type: ignore
                    cmax=E_field_color_bound,
                )
                frame_data.append(graph_obj)

            ########################################################################
            # show each worm (including its size)
            if show_worms:
                for worm in self.worm_objs:
                    graph_obj = worm.create_graph_obj(  # type: ignore
                        size_scale=5000 * show_worms, units_prefix=lengths_prefix  # type: ignore
                    )
                    frame_data.append(graph_obj)

            ########################################################################
            # show each boundary
            if show_boundaries:
                raise NotImplementedError("Boundary visualization is not yet implemented.")

            ########################################################################
            # show each fish (including the electric image and normals)
            if show_fish or show_normals or show_point_currents or show_image_currents:
                # set up intensity range, if needed, outside the for-loop
                intensity_range = None
                if show_electric_signal:
                    intensity_range = np.max(np.abs([electric_signal.min(), electric_signal.max()]))  # type: ignore
                    intensity_range = [-intensity_range, intensity_range]
                # go through each fish
                start_id = 0
                for i_fi, fish in enumerate(self.fish_objs):
                    end_id = start_id + fish.get_N_receptors()  # type: ignore

                    # show normals
                    if show_normals:
                        graph_obj = fish.create_normals_graph_obj(  # type: ignore
                            size_scale=show_normals, units_prefix=lengths_prefix
                        )
                        frame_data.append(graph_obj)

                    # show fish point currents
                    if show_point_currents:
                        graph_obj = fish.create_point_currents_graph_obj(  # type: ignore
                            marker_size=show_point_currents, units_prefix=lengths_prefix
                        )
                        frame_data.append(graph_obj)

                    # show IMAGE point currents
                    if show_image_currents:
                        graph_obj = self.create_image_currents_graph_obj(
                            i_fi, marker_size=show_image_currents, units_prefix=lengths_prefix
                        )
                        frame_data.append(graph_obj)

                    # show fish points; gather in intensity map shown on the fish surface
                    if show_fish:
                        intensity = None
                        if show_electric_signal:
                            intensity = electric_signal[start_id:end_id, i]  # type: ignore
                        graph_obj = fish.create_scatter_graph_obj(  # type: ignore
                            intensity=intensity,
                            intensity_range=intensity_range,
                            units_prefix=lengths_prefix,
                            marker_size=show_fish,
                        )
                        frame_data.append(graph_obj)

                    start_id = end_id

            frames.append(go.Frame(data=frame_data, name=frame_name))

        ########################################################################
        # construct figure #####################################################
        ########################################################################
        if len(frames) == 0:
            raise ValueError('Not enough time points to compute - "time_points" should be at least 1.')
        elif len(frames) == 1:
            fig = go.Figure()
        else:
            fig = go.Figure(frames=frames)

        for graph_obj in frames[0].data:
            fig.add_trace(graph_obj)

        if len(frames) > 1:

            def frame_args(duration):
                return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

            sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": f.name,
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

            fig.update_layout(
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(10)],
                                "label": "&#9654;",  # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;",  # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders,
            )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgb(220, 220, 240)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="grey",
                    range=add_prefix(np.array([min_ranges[0], max_ranges[0]]), lengths_prefix),
                ),
                yaxis=dict(
                    backgroundcolor="rgb(240, 220, 240)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="grey",
                    range=add_prefix(np.array([min_ranges[1], max_ranges[1]]), lengths_prefix),
                ),
                zaxis=dict(
                    backgroundcolor="rgb(240, 240, 220)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="grey",
                    range=add_prefix(np.array([min_ranges[2], max_ranges[2]]), lengths_prefix),
                ),
                xaxis_title=xaxis_title + f" ({lengths_prefix}m)",
                yaxis_title=yaxis_title + f" ({lengths_prefix}m)",
                zaxis_title=zaxis_title + f" ({lengths_prefix}m)",
                # aspectmode='data',
                aspectratio=dict(
                    x=1,
                    y=(max_ranges[1] - min_ranges[1]) / (max_ranges[0] - min_ranges[0]),
                    z=(max_ranges[2] - min_ranges[2]) / (max_ranges[0] - min_ranges[0]),
                ),
            ),
            width=None,
            height=None,
            margin=dict(r=0, l=0, b=0, t=0),
        )

        fig.show()
        print(f"PLOTTING TIME: {time.time() - end_time:.1f} s.")
        pass

    def create_image_currents_graph_obj(
        self, id_fish: int, marker_size: int = 15, units_prefix: str = ""
    ) -> go.Scatter3d:
        """Create a plotly graphical object to insert into a plot at a later time.
        This GO shows *the image* point currents of the selected fish in the aquarium.

        Args:
            id_fish (int): ID of the fish for which to create the graph object.
            marker_size (int, optional): Scatter point size of the *image* point currents to show. Defaults to 15.
            units_prefix (str, optional): Length prefixes to use for the (xyz)-axis labels. Defaults to "".

        Returns:
            go.Scatter3d: Plotly object to be inserted into a plot at a later time.
        """
        mags = self.image_point_currents_magnitude[id_fish][1]  # type: ignore
        locs = self.image_point_currents_location[id_fish][1]  # type: ignore
        graph_obj = go.Scatter3d(
            x=add_prefix(locs[:, 0], units_prefix),
            y=add_prefix(locs[:, 1], units_prefix),
            z=add_prefix(locs[:, 2], units_prefix),
            mode="markers",
            showlegend=False,
            marker=dict(
                size=marker_size,
                color=mags,
                colorscale="Armyrose",
                opacity=1,
            ),
        )
        return graph_obj

    def details(self):
        """Provides the details of the aquarium. Parent __doc__:\n"""
        Aquarium.details.__doc__ += super().details.__doc__  # type: ignore

        details_string = super().details()
        details_string += f"This aquarium has boundaries {self.boundaries}.\n"
        return details_string

    def get_boundaries(self) -> list[Boundary]:
        """Extract the named property of the fish object (immutable).

        Returns:
            list[Boundary]"""
        return self.boundaries

    def get_fish(self) -> list[Fish]:
        """Extract the named property of the fish object (immutable).

        Returns:
            list[Fish]"""
        return self.fish_objs  # type: ignore

    def get_worms(self) -> list[Worm]:
        """Extract the named property of the fish object (immutable).

        Returns:
            list[Worm]"""
        return self.worm_objs  # type: ignore

    def get_image_point_currents(self) -> tuple[list, list]:
        """Extract the named property of the fish object (immutable).

        Returns:
            tuple[list, list]"""
        return self.image_point_currents_magnitude, self.image_point_currents_location

    def get_time_stamps(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray"""
        return self.time_stamps

    @classmethod
    def _initialize_input_argument_names(cls):
        Aquarium._initialize_input_argument_names.__func__.__doc__ = super()._initialize_input_argument_names.__doc__

        inp_args = super()._initialize_input_argument_names()
        inp_args += [
            "fish_objs=[]",
            "worm_objs=[]",
            "boundaries=[], {{e.g. [('plane', dict(normal=[0,0,1],central_point=[0,0,0]))]}}",
        ]
        return inp_args

    def run_tests(self) -> str:
        Aquarium.run_tests.__doc__ = super().run_tests.__doc__

        super().run_tests()

        print(
            "Testing Aquarium class... Testing fish and worm insertion and removal. "
            "Expect to see text that 'object already exists' and 'object does not exist'."
        )

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            fish1 = FishGeneration(
                nose_position=[0, 1, 0], fish_length=0.5, eod_wave_form=np.zeros(100), skin_resistivity=1
            )
            fish2 = IceCreamConeFish(
                nose_position=[0, 0, 1],
                fish_length=0.5,
                eod_wave_form=np.ones(100),
                skin_resistivity=1,
                receptors_init=dict(method="random", head=100, body=200),
            )
            fish3 = FishGeneration(
                nose_position=[1, 0, 0], fish_length=0.5, eod_wave_form=np.zeros(100), skin_resistivity=2
            )
            worm1 = Worm(center_position=[1, 2, 3])
            worm2 = Worm(center_position=[2, 1, 3])
            worm3 = Worm(center_position=[1, 3, 2])

        N_fish = len(self.fish_objs)
        N_worms = len(self.worm_objs)
        assert N_fish == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert N_fish == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."
        self.insert_fish(fish1, _check_points=False)
        assert len(self.fish_objs) == N_fish + 1, "Inserting a fish should increment the list by 1."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.insert_fish(fish2, _check_points=False)
        assert len(self.fish_objs) == N_fish + 2, "Inserting two fish should increment the list by 2."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.insert_fish(fish1, _check_points=False)
        assert len(self.fish_objs) == N_fish + 2, "Inserting the same fish again should leave the list unchanged."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.remove_fish(fish3)
        assert len(self.fish_objs) == N_fish + 2, "Removing non-existent fish should leave the list unchanged."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.insert_fish(fish3, _check_points=False)
        assert len(self.fish_objs) == N_fish + 3, "Inserting three fish should increment the list by 3."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.remove_fish(fish2)
        assert (
            len(self.fish_objs) == N_fish + 2
        ), "Removing one of the three fish just added should decrement the list by 1."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.remove_fish(fish1)
        assert (
            len(self.fish_objs) == N_fish + 1
        ), "Removing one of the two remaining fish just added should decrement the list by 1."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.remove_fish(fish3)
        assert (
            len(self.fish_objs) == N_fish
        ), "Removing the three fish just added should leave the original list unchanged."
        assert len(self.fish_objs) == len(
            self.image_point_currents_magnitude
        ), "Number of fish and number of sets of image point currents should match."
        assert len(self.fish_objs) == len(
            self.image_point_currents_location
        ), "Number of fish and number of sets of image point currents should match."

        self.insert_worm(worm1, _check_points=False)
        assert len(self.worm_objs) == N_worms + 1, "Inserting a worm should increment the list by 1."
        self.insert_worm(worm2, _check_points=False)
        assert len(self.worm_objs) == N_worms + 2, "Inserting two worms should increment the list by 2."
        self.insert_worm(worm1, _check_points=False)
        assert len(self.worm_objs) == N_worms + 2, "Inserting the same worm again should leave the list unchanged."
        self.insert_worm(worm3, _check_points=False)
        assert len(self.worm_objs) == N_worms + 3, "Inserting three worms should increment the list by 3."
        self.remove_worm(worm2)
        self.remove_worm(worm1)
        self.remove_worm(worm3)
        assert (
            len(self.worm_objs) == N_worms
        ), "Removing the three worms just added should leave the original list unchanged."

        # back up aquarium
        temp_fish_objs = self.fish_objs
        temp_worm_objs = self.worm_objs
        temp_image_mag = self.image_point_currents_magnitude
        temp_image_loc = self.image_point_currents_location
        temp_time_stamps = self.time_stamps

        self.empty_aquarium()
        assert self.fish_objs == [], "Emptying aquarium should leave fish list empty."
        assert self.worm_objs == [], "Emptying aquarium should leave worm list empty."
        assert self.image_point_currents_magnitude == [], "Emptying aquarium should leave no image currents."
        assert self.image_point_currents_location == [], "Emptying aquarium should leave no image currents."

        # restore the aquarium to initial state
        self.fish_objs = temp_fish_objs
        self.worm_objs = temp_worm_objs
        self.image_point_currents_magnitude = temp_image_mag
        self.image_point_currents_location = temp_image_loc
        self.time_stamps = temp_time_stamps
        return "Success!"
