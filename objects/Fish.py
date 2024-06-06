import numpy as np
import plotly.graph_objects as go  # type: ignore
import time
from typing import Self

import sys

sys.path.append("helper_functions/")
sys.path.append("../helper_functions/")

from prefixes import add_prefix  # noqa: E402
from conversions import convert2mainSI  # noqa: E402
from ElectricObject import ElectricObject  # noqa: E402


class Fish(ElectricObject):
    """Represents a fish. The eod waveform is always normalized to have maximum at 1 - the magnitude of the EOD can be
    adjusted via the strength of the point currents magnitudes. Parent class "ElectricObject":"""

    __doc__ += ElectricObject.__doc__  # type: ignore

    def __init__(
        self,
        receptors_locations: np.ndarray = np.array([]).reshape(0, 3),
        receptors_normals: np.ndarray = np.array([]).reshape(0, 3),
        point_currents_magnitudes: np.ndarray = np.array([]).reshape(0),
        point_currents_locations: np.ndarray = np.array([]).reshape(0, 3),
        receptor_filters: np.ndarray = np.array([]).reshape(0, 1000),
        eod_wave_form: np.ndarray = np.arange(1000),
        skin_resistivity: float = 1,
        sampling_rate: float | tuple = (2.5, "M"),  # MHz
        eod_delay: float | tuple = (0, ""),  # s
        _init_tests: bool = True,
        **kwds,
    ):
        """Initialize a basic Fish object.

        Args:
            receptors_locations (np.ndarray, optional): Locations of the receptors on the virtual "skin" of the fish.
                Defaults to np.array([]).reshape(0, 3).
            receptors_normals (np.ndarray, optional): Normals to the skin surface at the receptor locations.
                Defaults to np.array([]).reshape(0, 3).
            point_currents_magnitudes (np.ndarray, optional): Magnitudes of the point currents that generate the
                currents in water. Defaults to np.array([]).reshape(0).
            point_currents_locations (np.ndarray, optional): Locations of the point currents.
                Defaults to np.array([]).reshape(0, 3).
            receptor_filters (np.ndarray, optional): Temporal convolutional filters for the receptors.
                Defaults to np.array([]).reshape(0, 1000).
            eod_wave_form (np.ndarray, optional): Wave form of the EOD of the fish. Defaults to np.arange(1000).
            skin_resistivity (float, optional): Resistivity of the skin of the fish. Defaults to 1.
            sampling_rate (float | tuple, optional): Sampling rate of the EOD wave form. Defaults to (2.5, "M").
            eod_delay (float | tuple, optional): Delay of the EOD wave form that helps distribute EODs in time when
                multiple fish are present in the aquarium. Defaults to (0, "").
            _init_tests (bool, optional): Run init tests or not. Defaults to True.

        See parent class "ElectricObject" for more details.
        """

        super().__init__(**kwds, _init_tests=False)

        # set an ID for this fish
        self.ID = int(time.time() * 1000)

        # initialize the Fish receptors and point currents
        self.receptors_locations = receptors_locations
        self.receptors_normals = receptors_normals
        self.point_currents_magnitudes = point_currents_magnitudes
        self.point_currents_locations = point_currents_locations

        # initialize EOD related attributes
        self.sampling_rate = convert2mainSI(sampling_rate)
        self.eod_wave_form = eod_wave_form
        self.eod_length = eod_wave_form.shape[0]
        self.update_eod_wave_form_and_delay(
            eod_wave_form=eod_wave_form, sampling_rate=sampling_rate, eod_delay=eod_delay  # type: ignore
        )

        # electric related attributes
        self.skin_rho = convert2mainSI(skin_resistivity)

        # initialize the convolutional filters
        self.receptor_filters = receptor_filters

        # expand the input argument names
        self._initialize_input_argument_names()

        # run assertion tests
        if _init_tests:
            print(self.run_tests())

    def compute_transdermal_signal(
        self, E_field: np.ndarray, water_conductivity: float, temporal_wave_form: np.ndarray | None = None
    ) -> np.ndarray:
        """Computes the transdermal signal processed by receptors on the fish's skin. This is given by
        the normal E field going through the fish's skin and adjusted by the skin resistivity and water conductivity.

        Args:
            E_field (np.ndarray): Electric field at the fish's receptors locations.
            water_conductivity (float): Electric conductivity of the water.
            temporal_wave_form (np.ndarray | None, optional): The wave form of the signal. Defaults to None.

        Raises:
            ValueError: If the provided E fields do not contain the temporal wave form and
                the wave form is not provided separately.
            ValueError: If the electric field input is not of shape (N x 3) or (N x 3 x T).

        Returns:
            np.ndarray: Transdermal signal at the fish's receptors, processed by receptors via the temporal filters.
        """
        assert (E_field.shape[0] == self.receptors_locations.shape[0]) and (
            E_field.shape[1] == 3
        ), "Electric field input should be shape (N x 3) or (N x 3 x T)."

        if len(E_field.shape) == 2:
            if temporal_wave_form is None:
                raise ValueError(
                    "The temporal wave form of the eods needs to be reflected in the 3rd dimension of the field or "
                    "be provided separately."
                )
            # compute image perturbation, proportional to the electric field normal to the fish's skin
            transdermal_signal = (
                (self.receptors_normals * E_field).sum(1, keepdims=True) * self.skin_rho * water_conductivity
            )
            assert (transdermal_signal.shape[0] == self.receptors_locations.shape[0]) and (
                transdermal_signal.shape[1] == 1
            ), "Transdermal signal shape should be the number of fish receptors by 1 (Nx1)."
            # include temporal component in the image perturbation
            transdermal_signal = transdermal_signal * temporal_wave_form
            assert (transdermal_signal.shape[0] == self.receptors_locations.shape[0]) and (
                transdermal_signal.shape[1] == temporal_wave_form.shape[0]
            ), "Transdermal signal shape should be the number of fish receptors by number of time points (NxT)."
        elif len(E_field.shape) == 3:
            transdermal_signal = (
                (self.receptors_normals[:, :, np.newaxis] * E_field).sum(1) * self.skin_rho * water_conductivity
            )
            assert (transdermal_signal.shape[0] == self.receptors_locations.shape[0]) and (
                transdermal_signal.shape[1] == E_field.shape[2]
            ), "Transdermal signal shape should be the number of fish receptors by number of time points (NxT)."
        else:
            raise ValueError("Electric field input should be shape (N x 3) or (N x 3 x T).")
        return transdermal_signal

    def compute_receptors_responses(self, transdermal_signal: np.ndarray) -> np.ndarray:
        """Apply the receptor filters to the transdermal signal to compute individual responses of the fish receptors.
        Typically, filters correspond to the A- and B-type cells responses.

        Args:
            transdermal_signal (np.ndarray): Wave form of the signal at each receptor on the fish's skin.

        Returns:
            np.ndarray: Responses of the receptors.
        """
        assert (
            self.eod_length == self.receptor_filters.shape[1]
        ), "The receptor length must match the eod length for proper computation."
        responses = np.inner(transdermal_signal[:, -self.eod_length :], self.receptor_filters)  # noqa: E203
        return responses

    def update_receptors(
        self, receptors_locations: np.ndarray | None = None, receptors_normals: np.ndarray | None = None
    ):
        """Update the receptors of the fish, i.e. move the fish in space.

        Args:
            receptors_locations (np.ndarray | None, optional): New receptors locations. Defaults to None.
            receptors_normals (np.ndarray | None, optional): Corresponding normals to the skin surface for
                new locations. Defaults to None.
        """
        self.receptors_locations = self.receptors_locations if receptors_locations is None else receptors_locations
        self.receptors_normals = self.receptors_normals if receptors_normals is None else receptors_normals
        assert (
            self.receptors_locations.shape[0] == self.receptors_normals.shape[0]
        ), "Number of receptors locations and normals must match."
        assert (self.receptors_locations.shape[1] == 3) and (
            self.receptors_normals.shape[1] == 3
        ), "All locations and normals must be 3-dimensional (Nx3)."
        if (self.receptors_locations.shape[0] == 0) or (self.receptors_normals.shape[0] == 0):
            print("Fish receptors were not properly initialized - some have 0 length.")

    def update_point_currents(
        self, point_currents_mag: np.ndarray | None = None, point_currents_loc: np.ndarray | None = None
    ):
        """Update the point currents of the fish, i.e. move the fish in space.

        Args:
            point_currents_mag (np.ndarray | None, optional): New point currents magnitudes. Defaults to None.
            point_currents_loc (np.ndarray | None, optional): New point currents locations. Defaults to None.
        """
        self.point_currents_magnitudes = (
            self.point_currents_magnitudes if point_currents_mag is None else point_currents_mag
        )
        self.point_currents_locations = (
            self.point_currents_locations if point_currents_loc is None else point_currents_loc
        )
        assert (
            self.point_currents_magnitudes.shape[0] == self.point_currents_locations.shape[0]
        ), "Number of point currents locations and magnitudes must match."
        assert self.point_currents_locations.shape[1] == 3, "All locations must be 3-dimensional (Nx3)."
        if (self.point_currents_magnitudes.shape[0] == 0) or (self.point_currents_locations.shape[0] == 0):
            print("Fish point currents were not properly initialized - some have 0 length.")

    def update_eod_wave_form_and_delay(
        self,
        eod_wave_form: np.ndarray | None = None,
        sampling_rate: float | None = None,
        eod_delay: float | None = None,
    ):
        """Update the EOD wave form of the fish and the delay of the EOD.
        Delay appends 0s at the beginning of the wave form.

        Args:
            eod_wave_form (np.ndarray | None, optional): New EOD wave form. Defaults to None.
            sampling_rate (float | None, optional): Sampling rate of the new EOD wave form. Defaults to None.
            eod_delay (float | None, optional): Delay of the new EOD wave form. Defaults to None.
        """
        self.sampling_rate = self.sampling_rate if sampling_rate is None else convert2mainSI(sampling_rate)
        self.eod_delay = 0 if eod_delay is None else convert2mainSI(eod_delay)
        self.eod_wave_form = (
            self.eod_wave_form[-self.eod_length :] if eod_wave_form is None else eod_wave_form  # noqa: E203
        )
        self.eod_length = self.eod_length if eod_wave_form is None else eod_wave_form.shape[0]
        self.eod_wave_max = np.array(self.eod_wave_form).max()
        # include the eod delay at the beginning of the wave-form
        self.eod_wave_form = np.hstack(
            [
                np.zeros(np.int64(self.sampling_rate * self.eod_delay)),
                np.array(self.eod_wave_form) / (self.eod_wave_max if self.eod_wave_max > 0 else 1),
            ]
        )
        self.time_stamps = np.arange(self.eod_wave_form.shape[0]) / self.sampling_rate

    def update_receptors_filters(self, new_receptor_filters: np.ndarray):
        """
        Update the receptor filters of the fish.

        Args:
            new_receptor_filters (np.ndarray): The new receptor filters to be assigned.
        """
        self.receptor_filters = new_receptor_filters

    def is_equal(self, other: Self, simple_return: bool = True) -> bool | tuple:
        """Compare current fish to another fish for equality. See parent class "ElectricObject" for more details."""

        _, truth_values, comparison = super().is_equal(other, simple_return=False)  # type: ignore
        if isinstance(other, self.__class__):
            truth_values.append(np.abs(self.ID - other.get_ID()) < self.assert_err)
            truth_values.append(np.abs(self.sampling_rate - other.get_sampling_rate()) < self.assert_err)
            truth_values.append(np.abs(self.skin_rho - other.get_skin_resistivity()) < self.assert_err)

            comparison.append("ID")
            comparison.append("sampling_rate")
            comparison.append("skin_resistivity")
        truth_value = False if len(truth_values) == 0 else np.array(truth_values).all()
        if simple_return:
            return truth_value  # type: ignore
        else:
            return truth_value, truth_values, comparison

    def visualize_scatter(
        self,
        fig: go.Figure | None = None,
        intensity: np.ndarray | None = None,
        cbar_N_ticks: int = 10,
        marker_size: int = 10,
        marker_alpha: float = 0.8,
        color_map: str = "Viridis",
        fig_width: float | None = None,
        fig_height: float | None = None,
        units_prefix: str = "m",
        xaxis_title: str = "X AXIS TITLE",
        yaxis_title: str = "Y AXIS TITLE",
        zaxis_title: str = "z AXIS TITLE",
        show_normals: int = 5,
        show_point_currents: int = 30,
        update_layout: bool = True,
    ) -> go.Figure:
        """Visualize the fish as scatter points. If provided, the intensity of the perturbation is reflected in the
        color of each scatter point.

        Args:
            fig (go.Figure | None, optional): Figure to draw the visualization. Defaults to None.
            intensity (np.ndarray | None, optional): Intensity for coloring the receptors. Defaults to None.
            cbar_N_ticks (int, optional): Number of ticks for the colorbar. Defaults to 10.
            marker_size (int, optional):  Size of the scatter points that represent the fish receptors. Defaults to 10.
            marker_alpha (float, optional): Transparency of the scatter points. Defaults to 0.8.
            color_map (str, optional): Colormap for intensity visualization. Defaults to "Viridis".
            fig_width (float | None, optional): Figure width (px). Defaults to None.
            fig_height (float | None, optional): Figure height (px). Defaults to None.
            units_prefix (str, optional): Length prefixes to use for the (xyz)-axis labels. Defaults to "m".
            xaxis_title (str, optional): Label of the x-axis. Defaults to "X AXIS TITLE".
            yaxis_title (str, optional): Label of the y-axis. Defaults to "Y AXIS TITLE".
            zaxis_title (str, optional): Label of the z-axis. Defaults to "z AXIS TITLE".
            show_normals (int, optional): Whether or not to show normals to the skin surface. Defaults to 5.
            show_point_currents (int, optional): Whether or not to show points currents as scatter. Defaults to 30.
            update_layout (bool, optional): Whether or not to update go.Figure layout to predetermined style.
                Defaults to True.

        Returns:
            go.Figure: go.Figure object with the visualization.
        """
        if fig is None:
            fig = go.Figure()

        if show_point_currents:
            fig.add_trace(  # type: ignore
                self.create_point_currents_graph_obj(marker_size=show_point_currents, units_prefix=units_prefix)
            )

        if show_normals:
            fig.add_trace(  # type: ignore
                self.create_normals_graph_obj(size_scale=show_normals, units_prefix=units_prefix)
            )

        intensity_range = None if intensity is None else [intensity.min(), intensity.max()]
        fig.add_trace(  # type: ignore
            self.create_scatter_graph_obj(
                intensity=intensity,
                intensity_range=intensity_range,
                units_prefix=units_prefix,
                marker_size=marker_size,
                color_map=color_map,
                marker_alpha=marker_alpha,
                cbar_N_ticks=cbar_N_ticks,
            )
        )

        if update_layout:
            fig.update_layout(  # type: ignore
                scene=dict(
                    xaxis=dict(
                        backgroundcolor="rgb(220, 220, 240)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                    ),
                    yaxis=dict(
                        backgroundcolor="rgb(240, 220, 240)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                    ),
                    zaxis=dict(
                        backgroundcolor="rgb(240, 240, 220)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                    ),
                    xaxis_title=xaxis_title + f" ({units_prefix}m)",
                    yaxis_title=yaxis_title + f" ({units_prefix}m)",
                    zaxis_title=zaxis_title + f" ({units_prefix}m)",
                    aspectmode="data",  # aspectratio=dict(x=1, y=1, z=2),
                ),
                width=fig_width,
                height=fig_height,
                margin=dict(r=0, l=0, b=0, t=0),
            )
        fig.show()  # type: ignore
        return fig

    def create_scatter_graph_obj(
        self,
        intensity: np.ndarray | None = None,
        intensity_range: np.ndarray | list | None = None,
        units_prefix: str = "",
        marker_size: int = 10,
        color_map: str = "Viridis",
        marker_alpha: float = 0.7,
        cbar_N_ticks: int = 10,
    ) -> go.Scatter3d:
        """Create a plotly graphical object to insert into a plot at a later time. This GO shows the fish as scattered
        points at the receptor locations. The scatter points are colored according to the provided intensity values
        with extreme colormap values provided by the intensity range. (If intensity not provided, colormap defaults
        to receptor locations along the x-axis.)

        Args:
            intensity (np.ndarray | None, optional): Intensity of colormap for each receptor. Defaults to None.
            intensity_range (np.ndarray | list | None, optional): Extreme values for the colormap limits.
                Defaults to None.
            units_prefix (str, optional): Length prefixes to use for the (xyz)-axis labels. Defaults to "".
            marker_size (int, optional): Size of the scatter points that represent the fish receptors. Defaults to 10.
            color_map (str, optional): Colormap for intensity visualization. Defaults to "Viridis".
            marker_alpha (float, optional): Transparency of the scatter points. Defaults to 0.7.
            cbar_N_ticks (int, optional): Number of ticks on the intensity colorbar. Defaults to 10.

        Returns:
            go.Scatter3d: Scatter plotly object to be inserted into a plot at a later time.
        """
        if intensity is None:
            colors = self.receptors_locations[:, 0]
        else:
            colors = intensity
        if intensity_range is None:
            intensity_range = [colors.min(), colors.max()]
            if colors.shape[0] > 1:
                colors = (colors - colors.min()) / colors.ptp()
        else:
            colors = (colors - intensity_range[0]) / (intensity_range[1] - intensity_range[0])

        graph_obj = go.Scatter3d(
            x=add_prefix(self.receptors_locations[:, 0], units_prefix),
            y=add_prefix(self.receptors_locations[:, 1], units_prefix),
            z=add_prefix(self.receptors_locations[:, 2], units_prefix),
            mode="markers",
            showlegend=False,
            marker=dict(
                size=marker_size,
                color=colors,
                colorscale=color_map,
                opacity=marker_alpha,
                colorbar=dict(
                    len=0.8,
                    tickmode="array",
                    tickvals=np.linspace(0, 1, cbar_N_ticks),
                    ticktext=(
                        np.linspace(0, 1, cbar_N_ticks)
                        if intensity_range is None
                        else np.linspace(intensity_range[0], intensity_range[1], cbar_N_ticks)
                    ),
                ),
                cmin=0,
                cmax=1,
            ),
        )
        return graph_obj

    def create_normals_graph_obj(self, size_scale: int = 5, units_prefix: str = "") -> go.Cone:
        """Create a plotly graphical object to insert into a plot at a later time. This GO shows normals on
        the skin of the fish at the receptor locations.

        Args:
            size_scale (int, optional): Size of the scatter cones that represent the normals to the skin surface of
                the fish. Defaults to 5.
            units_prefix (str, optional): Length prefixes to use for the (xyz)-axis labels. Defaults to "".

        Returns:
            go.Cone: Plotly object to be inserted into a plot at a later time.
        """
        graph_obj = go.Cone(
            x=add_prefix(self.receptors_locations[:, 0], units_prefix),
            y=add_prefix(self.receptors_locations[:, 1], units_prefix),
            z=add_prefix(self.receptors_locations[:, 2], units_prefix),
            u=self.receptors_normals[:, 0],
            v=self.receptors_normals[:, 1],
            w=self.receptors_normals[:, 2],
            colorscale="Blues",
            sizemode="absolute",
            sizeref=size_scale,
            showscale=False,
            showlegend=False,
        )
        return graph_obj

    def create_point_currents_graph_obj(self, marker_size: int = 15, units_prefix: str = "") -> go.Scatter3d:
        """Create a plotly graphical object to insert into a plot at a later time. This GO shows point currents
        within the fish.

        Args:
            marker_size (int, optional): Scatter point size of the point currents to show. Defaults to 15.
            units_prefix (str, optional): Length prefixes to use for the (xyz)-axis labels. Defaults to "".

        Returns:
            go.Scatter3d: Plotly object to be inserted into a plot at a later time.
        """
        graph_obj = go.Scatter3d(
            x=add_prefix(self.point_currents_locations[:, 0], units_prefix),
            y=add_prefix(self.point_currents_locations[:, 1], units_prefix),
            z=add_prefix(self.point_currents_locations[:, 2], units_prefix),
            mode="markers",
            showlegend=False,
            marker=dict(
                size=marker_size,
                color=self.point_currents_magnitudes,
                colorscale="Bluered",
                opacity=1,
            ),
        )
        return graph_obj

    def visualize_triangulation(self):
        """Visualize the surface of the fish by triangulation method, typically used in computer graphics."""
        raise NotImplementedError(
            "Triangulation visualization of fish not yet implemented - easier to export to Matlab..."
        )

    def details(self) -> str:
        """Provides the details of the fish. See parent class "ElectricObject" for more details."""

        details_string = super().details()
        details_string += (
            f"This fish has:\n"
            f"skin resistivity {self.skin_rho} (unit: Ohm * meter^2)\n"
            f"conductivity {self.sig} (unit: Siemens / meter) / relative permittivity {self.eps_r}\n"
            f"EOD wave form sampling rate {self.sampling_rate} (unit: Hertz)\n"
            f"number of point currents {self.point_currents_magnitudes.shape[0]} / "
            f"number of receptors: {self.receptors_locations.shape[0]}\n"
        )
        return details_string

    def get_ID(self) -> int:
        """Extract the named property of the fish object (immutable).

        Returns:
            int
        """
        return self.ID

    def get_receptors_locations(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray
        """
        return self.receptors_locations

    def get_receptors_normals(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray
        """
        return self.receptors_normals

    def get_sampling_rate(self) -> float | np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            float | np.ndarray
        """
        return self.sampling_rate

    def get_eod_wave_form(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray
        """
        return self.eod_wave_form

    def get_time_stamps(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray
        """
        return self.time_stamps

    def get_skin_resistivity(self) -> float | np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            float | np.ndarray
        """
        return self.skin_rho

    def get_point_currents_magnitude(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray
        """
        return self.point_currents_magnitudes

    def get_point_currents_location(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray
        """
        return self.point_currents_locations

    def get_N_receptors(self) -> int:
        """Extract the named property of the fish object (immutable).

        Returns:
            int
        """
        return self.receptors_locations.shape[0]

    def get_receptors_filters(self) -> np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            np.ndarray
        """
        return self.receptor_filters

    def get_N_filters(self) -> int:
        """Extract the named property of the fish object (immutable).

        Returns:
            int
        """
        return self.receptor_filters.shape[0]

    def get_EOD_length(self) -> int:
        """Extract the named property of the fish object (immutable).
        Refers to the length of the EOD wave form in number of timestamps (not real time)..

        Returns:
            int
        """
        return self.eod_length

    @classmethod
    def _initialize_input_argument_names(cls) -> list[str]:
        inp_args = super()._initialize_input_argument_names()
        inp_args += [
            "receptors_locations=np.array([]).reshape(0,3)",
            "receptors_normals=np.array([]).reshape(0,3)",
            "point_currents_magnitudes=np.array([]).reshape(0)",
            "point_currents_locations=np.array([]).reshape(0,3)",
            "receptor_filters=np.array([]).reshape(0,1000)",
            "eod_wave_form=np.arange(1000)",
            "skin_resistivity=1",
            'sampling_rate=(2.5, "M"),  # MHz',
            "eod_delay=(0," "), # s",
        ]
        return inp_args

    def run_tests(self) -> str:
        super().run_tests()
        assert self.skin_rho == self.get_skin_resistivity(), "Fish skin resistivity does not match."
        assert self.sampling_rate == self.get_sampling_rate(), "Fish sampling rate does not match."

        assert (
            self.receptors_locations == self.get_receptors_locations()
        ).all(), "Fish receptors locations do not match."
        assert (self.receptors_normals == self.get_receptors_normals()).all(), "Fish receptors normals do not match."

        assert (self.time_stamps == self.get_time_stamps()).all(), "Fish time stamps does not match."
        assert (self.eod_wave_form == self.get_eod_wave_form()).all(), "Fish eod wave form does not match."
        assert (
            self.point_currents_magnitudes == self.get_point_currents_magnitude()
        ).all(), "Fish point currents magnitude does not match."
        assert (
            self.point_currents_locations == self.get_point_currents_location()
        ).all(), "Fish point currents location does not match."

        return "Success!"
