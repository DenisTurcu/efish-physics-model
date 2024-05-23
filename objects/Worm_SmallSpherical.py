import numpy as np
import scipy.signal as Ssignal  # type: ignore
import plotly.graph_objects as go  # type: ignore
import sys

sys.path.append("helper_functions/")
sys.path.append("../helper_functions/")

from prefixes import add_prefix  # noqa: E402
from conversions import convert2mainSI  # noqa: E402
from Worm import Worm  # noqa: E402
from Aquarium import Aquarium  # noqa: E402


class SmallSphericalWorm(Worm):
    """Represents a small, spherical worm of given radius. Parent class "Worm":"""

    __doc__ += Worm.__doc__  # type: ignore

    def __init__(
        self,
        radius: float | tuple = 0.01,
        resistance: float | tuple | None = None,
        capacitance: float | tuple | None = None,
        derivative_filter: list = [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280],
        _init_tests: bool = True,
        **kwds,
    ):
        """Initialize the SmallSphericalWorm object.

        Args:
            radius (float | tuple, optional): Radius if the worm (will be converted to SI, provide it accordingly).
                Defaults to 0.01.
            resistance (float | tuple | None, optional): Resistance of the worm (will be converted to SI, provide it
                accordingly). When "None", it will be inferred from electric properties (conductivity).
                Defaults to None.
            capacitance (float | tuple | None, optional): Capacitance or the worm (will be converted to SI, provide it
                accordingly). When "None", it will be inferred from electric properties (relative permittivity).
                Defaults to None.
            derivative_filter (list, optional): Filter used to compute first derivative of a time series.
                Defaults to [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280].
            _init_tests (bool, optional): Whether to run initialization tests. Defaults to True.

        Parent __doc__:\n
        """
        SmallSphericalWorm.__init__.__doc__ += super().__init__.__doc__  # type: ignore

        super().__init__(**kwds, _init_tests=False)
        self.radius = convert2mainSI(radius)
        self.der_filt = derivative_filter
        self.initiate_electrical_properties(resistance, capacitance)
        # Run assertion tests
        if _init_tests:
            print(self.run_tests())

    def differential_equation_coefficients(self, sig_water: float, eps_r_water: float) -> tuple:
        """Compute the coefficients of the differential equation that solves the distortion waveform based on
        the physics model results (Eq. (7) in "Electric_Fish_Numerically_Stable_Solution" Overleaf).

        Args:
            sig_water (float): Conductivity of the water.
            eps_r_water (float): Relative permittivity of the water.

        Returns:
            tuple: Coefficients of the differential equation.
        """
        Rw = 2 / (np.pi * self.radius * sig_water)
        Cw = np.pi * self.radius * self.eps0 * eps_r_water / 2
        f0 = 1 / self.R - 1 / Rw
        f1 = self.C - Cw
        p0 = 1 / self.R + 2 / Rw
        p1 = self.C + 2 * Cw
        return f0, f1, p0, p1

    def dipole_waveform_single_fish(
        self, wave_form: np.ndarray, sig_water: float, eps_r_water: float, sampling_frequency: int
    ) -> np.ndarray:
        """Compute the wave form of the induced dipole (different than the waveform of the EOD) based on
        relevant computations from the physics model.
        (
            Solve Eq. (7) in "Electric_Fish_Numerically_Stable_Solution" Overleaf
                p0 * phi(t) + p1 * (d phi)/dt = f0 * f(t) + f1 (d f)/dt
        )

        Args:
            wave_form (np.ndarray): Original (EOD) wave form.
            sig_water (float): Conductivity of the water.
            eps_r_water (float): Relative permittivity of the water.
            sampling_frequency (int): Sampling frequency of the wave form.

        Returns:
            np.ndarray: Wave form of the induced dipole.
        """
        assert (
            wave_form.shape == wave_form.reshape(-1).shape
        ), "Wave form should be shape (T,) - i.e. not (T,1) or (1,T)."

        wave_form_der = np.convolve(self.der_filt[::-1], wave_form, "same") * sampling_frequency
        f0, f1, p0, p1 = self.differential_equation_coefficients(sig_water, eps_r_water)
        dipole_wave_form = np.zeros(wave_form.shape[0])
        old_settings = np.seterr(over="ignore", invalid="ignore")
        for j in range(1, wave_form.shape[0]):
            dipole_wave_form[j] = dipole_wave_form[j - 1] + (
                f0 * wave_form[j - 1] + f1 * wave_form_der[j - 1] - p0 * dipole_wave_form[j - 1]
            ) / (p1 * sampling_frequency)
        np.seterr(**old_settings)
        # Ensure proper solution for very small tau=RC
        dipole_wave_form = self.dipole_waveform_quality_check(
            dipole_wave_form, wave_form, wave_form_der, sampling_frequency, f0, f1, p0, p1
        )
        return dipole_wave_form

    def dipole_waveform_quality_check(
        self,
        dipole_wave_form: np.ndarray,
        wave_form: np.ndarray,
        wave_form_der: np.ndarray,
        sampling_frequency: float,
        f0: float,
        f1: float,
        p0: float,
        p1: float,
        resolution_multiplication: int = 5,
        runtime_err_factor: int = 5000,
        check_err: float = 1.1e-2,
    ) -> np.ndarray:
        """Check the quality of the solution of the induced dipole wave form.
        (
            Equation to solve:
                p0 * phi(t) + p1 * (d phi)/dt = f0 * f(t) + f1 (d f)/dt
        )

        Args:
            dipole_wave_form (np.ndarray): Induced dipole wave form.
            wave_form (np.ndarray): Original (EOD) wave form.
            wave_form_der (np.ndarray): Derivative of original wave form.
            sampling_frequency (float): Sampling frequency of the wave form.
            f0 (float): f0 coefficient of the differential equation that solves the distortion waveform.
            f1 (float): f1 coefficient of the differential equation that solves the distortion waveform.
            p0 (float): p0 coefficient of the differential equation that solves the distortion waveform.
            p1 (float): p1 coefficient of the differential equation that solves the distortion waveform.
            resolution_multiplication (int, optional): Factor that multiplies the sampling rate on each iteration to
                attempt a better solution for the differential equation based on smaller "dt". Defaults to 5.
            runtime_err_factor (int, optional): Maximum multiplication factor allowed. Defaults to 5000.
            check_err (float, optional): Maximum discrepancy size allowed. Defaults to 1.1e-2.

        Returns:
            np.ndarray: ave form of the induced dipole adjusted as much as possible to a better solution using a
                smaller "dt" to solve the differential equation.
        """
        resolution_factor = 1
        discrepancy_size = np.abs(
            p0 * dipole_wave_form
            + p1 * np.convolve(self.der_filt[::-1], dipole_wave_form, "same") * sampling_frequency
            - f0 * wave_form
            - f1 * wave_form_der
        ).max()

        # attempt to solve the differential equation again with a smaller "dt" to get a better solution
        #       attempt this until error is small enough or until the maximum resolution factor is reached
        while (discrepancy_size > check_err) or np.isnan(discrepancy_size):
            # print message to inform of poor estimate of the dipole waveform
            resolution_factor *= resolution_multiplication
            if resolution_factor > runtime_err_factor:
                print(
                    f'____________ Could not solve differential equation (7) in "Electric_Fish_Numerically_Stable'
                    f'_Solution" Overleaf with small enough error. Failed due to radius {self.radius}, resistance '
                    f"{self.R} and capacitance {self.C} with maximum resolution factor of "
                    f"{resolution_factor/resolution_multiplication}. Discrepancy size was {discrepancy_size}. "
                    'Returning best estimated "dipole_wave_form".'
                )
                break
            wave_form_resampled = Ssignal.resample(wave_form, wave_form.shape[0] * resolution_factor)
            wave_form_der_resampled = (
                np.convolve(self.der_filt[::-1], wave_form_resampled, "same") * sampling_frequency * resolution_factor
            )
            dipole_wave_form = np.zeros(wave_form_resampled.shape[0])
            old_settings = np.seterr(over="ignore", invalid="ignore")
            for j in range(1, wave_form_resampled.shape[0]):
                dipole_wave_form[j] = dipole_wave_form[j - 1] + (
                    f0 * wave_form_resampled[j - 1] + f1 * wave_form_der_resampled[j - 1] - p0 * dipole_wave_form[j - 1]
                ) / (p1 * sampling_frequency * resolution_factor)
            np.seterr(**old_settings)
            dipole_wave_form = dipole_wave_form[::resolution_factor]
            discrepancy_size = np.abs(
                p0 * dipole_wave_form
                + p1 * np.convolve(self.der_filt[::-1], dipole_wave_form, "same") * sampling_frequency
                - f0 * wave_form
                - f1 * wave_form_der
            ).max()
        return dipole_wave_form

    def perturbation_magnitude_single_fish(
        self,
        receptor_locs: np.ndarray,
        E_field: np.ndarray,
        return_potential: bool = True,
        return_field: bool = True,
        include_radius_factor: bool = True,
    ) -> tuple:
        """Returns the electric potential/field perturbation magnitude, i.e. without the temporal trace, at given
        locations.

        Args:
            receptor_locs (np.ndarray): Locations of the fish receptors, where the perturbation is computed.
            E_field (np.ndarray): Electric field at the center of the Worm, shape (3,), that polarizes the worm.
            return_potential (bool, optional): Whether or not to return the electric POTENTIAL at the provided
                locations. Defaults to True.
            return_field (bool, optional): Whether or not to return the electric FIELD at the provided locations.
                Defaults to True.
            include_radius_factor (bool, optional): Whether to include the radius factor of the worm in the magnitude
                of the result. Defaults to True.

        Returns:
            tuple: Tuple returning (potential perturbation, E field perturbation) at the given locations.
                Includes only magnitude of the perturbation without the temporal trace.
        """
        E_field = E_field.reshape(3)
        assert (
            E_field.shape == np.zeros(3).shape
        ), "Electric field at the center of the Worm should be shape (3,) - i.e. not (3,1) or (1,3)."

        r_relative = receptor_locs - self.r_vec
        assert (
            r_relative.shape == receptor_locs.shape
        ), "The relative distance should match the shape of all fish receptor locations."
        assert (
            r_relative[0, :] == (receptor_locs[0, :] - self.r_vec)
        ).all(), "Individual examples of relative distance should match values."

        radius_factor = 1 if not include_radius_factor else np.power(self.radius, 3)

        potential_perturbation = None
        E_perturbation = None

        if return_potential:
            # compute electric potential perturbation at given locations according to EM computations
            potential_perturbation = (
                radius_factor * (E_field * r_relative).sum(1) / np.power(np.linalg.norm(r_relative, axis=1), 3)
            )
            assert (
                potential_perturbation.shape == receptor_locs[:, 0].shape
            ), "Electric potential should be shape (N,) at this stage"

        if return_field:
            # compute electric field perturbation at given locations according to EM computations
            E_perturbation = (
                3 * (E_field * r_relative).sum(1, keepdims=True) * r_relative
                - np.power(np.linalg.norm(r_relative, axis=1, keepdims=True), 2) * E_field
            )
            E_perturbation *= radius_factor / np.power(np.linalg.norm(r_relative, axis=1, keepdims=True), 5)
            assert (
                E_perturbation.shape == receptor_locs.shape
            ), "The E field perturbation should match the shape of all fish points."

        return potential_perturbation, E_perturbation

    def perturbation_trace(
        self,
        receptor_locs: np.ndarray,
        aquarium_obj: Aquarium,
        return_potential: bool = True,
        return_field: bool = True,
    ) -> tuple:
        """Returns electric potential/field perturbations computed at given locations.

        Args:
            receptor_locs (np.ndarray): Locations of the fish receptors, where the perturbation is computed.
            aquarium_obj (Aquarium): Aquarium object that contains at least one fish and the worm.
            return_potential (bool, optional): Whether or not to return the electric POTENTIAL at the provided
                locations. Defaults to True.
            return_field (bool, optional): Whether or not to return the electric FIELD at the provided locations.
                Defaults to True.

        Returns:
            tuple: Tuple returning (potential perturbation, E field perturbation) at the given locations,
                including the temporal distortions
        """
        potential_perturbation_total = 0
        E_perturbation_total = 0
        for i, fish in enumerate(aquarium_obj.get_fish()):
            # compute electric field at self.r_vec
            _, E_field, wave_form = aquarium_obj.electric_potential_and_field_single_fish(
                self.r_vec[np.newaxis], fish_id=i, return_potential=False, return_field=True
            )

            potential_perturbation, E_perturbation = self.perturbation_magnitude_single_fish(
                receptor_locs=receptor_locs,
                E_field=E_field,  # type: ignore
                return_potential=return_potential,
                return_field=return_field,
                include_radius_factor=True,
            )

            dipole_wave_form = self.dipole_waveform_single_fish(
                wave_form=wave_form,
                sig_water=aquarium_obj.get_conductivity(),
                eps_r_water=aquarium_obj.get_relative_permittivity(),
                sampling_frequency=int(fish.get_sampling_rate()),
            )

            if return_potential:
                potential_perturbation = potential_perturbation[:, np.newaxis] * dipole_wave_form
                assert (
                    potential_perturbation.shape[0] == receptor_locs.shape[0]
                ), "Electric potential should be shape (N,T) at this stage"
                assert (
                    potential_perturbation.shape[1] == wave_form.shape[0]
                ), "Electric potential should be shape (N,T) at this stage"
                potential_perturbation_total += potential_perturbation

            if return_field:
                E_perturbation = E_perturbation[:, :, np.newaxis] * dipole_wave_form
                assert (
                    E_perturbation.shape == np.array([receptor_locs.shape[0], 3, wave_form.shape[0]])
                ).all(), "The E field perturbation should be shape (N x 3 x T)."
                E_perturbation_total += E_perturbation

        return potential_perturbation_total, E_perturbation_total

    def initiate_electrical_properties(self, resistance: float | tuple | None, capacitance: float | tuple | None):
        """Parse electric properties of the worm.

        Args:
            resistance (float | tuple | None): Resistance of the worm (will be converted to SI, provide it accordingly).
                When "None", it will be inferred from conductivity.
            capacitance (float | tuple | None, optional): Capacitance or the worm (will be converted to SI, provide it
                accordingly). When "None", it will be inferred from relative permittivity.
        """
        if resistance is None:
            print("!!! Using the worm material conductivity to estimate its resistance.")
            self.R = 2 / (np.pi * self.radius * self.sig)
        else:
            self.R = convert2mainSI(resistance)
        if capacitance is None:
            print("!!! Using the worm material relative permittivity to estimate its capacitance.")
            self.C = np.pi * self.radius * self.eps0 * self.eps_r / 2
        else:
            self.C = convert2mainSI(capacitance)
        pass

    def create_graph_obj(self, size_scale: float = 10, units_prefix: str = "") -> go.Scatter3d:
        """Create a plotly graphical object to insert into a plot at a later time.

        Args:
            size_scale (float, optional): Dictates the size of the scatter point marker. Defaults to 10.
            units_prefix (str, optional): Prefix of the distance units used and plotted as (xyz)-labels. Defaults to "".

        Returns:
            go.Scatter3d: Plotly graphical object representing the worm.
        """
        graph_obj = go.Scatter3d(
            x=[add_prefix(self.r_vec[0], units_prefix)],
            y=[add_prefix(self.r_vec[1], units_prefix)],
            z=[add_prefix(self.r_vec[2], units_prefix)],
            showlegend=False,
            mode="markers",
            marker=dict(size=self.radius * size_scale, color="brown", opacity=0.9),
        )
        return graph_obj

    def is_equal(self, other: Worm, simple_return: bool = True) -> bool | tuple:
        SmallSphericalWorm.is_equal.__doc__ = super().is_equal.__doc__  # type: ignore

        _, truth_values, comparison = super().is_equal(other, simple_return=False)  # type: ignore
        if isinstance(other, self.__class__):
            truth_values.append(np.abs(self.radius - other.get_radius()) < np.min([self.assert_err, other.assert_err]))
            comparison.append("radius")
        truth_value = False if len(truth_values) == 0 else np.array(truth_values).all()
        if simple_return:
            return truth_value  # type: ignore
        else:
            return truth_value, truth_values, comparison

    def details(self) -> str:
        """Provides the details of the small spherical worm. Parent __doc__:\n"""
        SmallSphericalWorm.details.__doc__ += super().details.__doc__  # type: ignore

        details_string = super().details()
        details_string += f"This worm type is small and spherical, with radius {self.radius}.\n"
        return details_string

    def get_radius(self) -> float | np.ndarray:
        """Extract the worm's radius. (immutable)

        Returns:
            float: Radius of worm.
        """
        return self.radius

    @classmethod
    def _initialize_input_argument_names(cls) -> list[str]:
        SmallSphericalWorm._initialize_input_argument_names.__func__.__doc__ = (
            super()._initialize_input_argument_names.__doc__
        )

        inp_args = super()._initialize_input_argument_names()
        inp_args += ["radius", "resistance", "capacitance"]
        return inp_args
