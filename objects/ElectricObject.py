import numpy as np
import sys
from typing import Self

sys.path.append("helper_functions/")
sys.path.append("../helper_functions/")

from conversions import rho24sig, convert2mainSI  # noqa: E402


class ElectricObject:
    """
    Represents an electric object that has limited functionality.
    Call function "get_input_argument_names()" to check the inputs to this class.
    """

    eps0 = 8.854187812813 * 1e-12  # absolute value for the permittivity of vacuum

    def __init__(
        self,
        conductivity: float | tuple | np.ndarray = 1,
        relative_permittivity: float | tuple | np.ndarray = 1,
        assert_err: float = 1e-12,
        _init_tests: bool = True,
    ):
        """Initialize the ElectricObject.

        Args:
            conductivity (float | tuple | np.ndarray, optional): Electric conductivity value - can be float (make sure
                it's in SI units), tuple (the first element is the value and the rest are prefixes), or np.ndarray
                (values in SI units). Defaults to 1.
            relative_permittivity (float | tuple | np.ndarray, optional): Relative permittivity "epsilon" - same
                conditions as for "conductivity". Defaults to 1.
            assert_err (float, optional): Assertion error value for what counts as "0" error. Defaults to 1e-12.
            _init_tests (bool, optional): Run init tests or not. Defaults to True.
        """
        self.sig = convert2mainSI(conductivity)
        self.eps_r = relative_permittivity
        self.assert_err = assert_err
        self.arguments = self._initialize_input_argument_names()
        # Run assertion tests
        if _init_tests:
            print(self.run_tests())

    def __str__(self) -> str:
        """Provides convenience of using the print function to show details of object.

        Returns:
            str: Details of the object.
        """
        return self.details()

    def details(self) -> str:
        """Provides the details of the object. Can be adjusted in child classes to include more information.

        Returns:
            str: Details string of the object.
        """
        details_string = (
            f"This electric object has conductivity {self.sig} and relative permittivity {self.eps_r}.\n"
            f"Assertion allowed error is {self.assert_err}.\n"
        )
        return details_string

    def is_equal(self, other: Self, simple_return: bool = True) -> bool | tuple:
        """Compare two objects along many available variables. Needs to be updated/implemented in child classes.

        Args:
            other (_type_): Another ElectricObject to compare with.
            simple_return (bool, optional): Whether to return simple bool comparison, or detailed comparison that
                includes all compared features and the comparison results for each compared feature. Defaults to True.

        Returns:
            bool | tuple: Simple True/False comparison or detailed comparison.
        """
        truth_values = []
        comparison = []
        if isinstance(other, self.__class__):
            truth_values.append(
                np.abs((self.sig - other.get_conductivity())) < np.min([self.assert_err, other.assert_err])
            )
            truth_values.append(
                np.abs((self.eps_r - other.get_relative_permittivity()))  # type: ignore
                < np.min([self.assert_err, other.assert_err])  # type: ignore
            )
            comparison.append("conductivity")
            comparison.append("relative_permitivity")
        truth_value = False if len(truth_values) == 0 else np.array(truth_values).all()
        if simple_return:
            return truth_value  # type: ignore
        else:
            return truth_value, truth_values, comparison

    def get_conductivity(self) -> float:
        """Extract the electric conductivity, sigma. (immutable)"""
        return self.sig  # type: ignore

    def get_resistivity(self) -> float:
        """Extract the electric resistivity, rho. (immutable)"""
        return rho24sig(self.sig)  # type: ignore

    def get_relative_permittivity(self) -> float:
        """Extract the relative permittivity eps_relative. (immutable)"""
        return self.eps_r  # type: ignore

    def get_input_argument_names(self, print_arguments=True):
        """Provides the names of the input arguments required to create a new object."""
        if print_arguments:
            print("Input arguments:")
            for argument in self.arguments:
                print(argument)
        return self.arguments

    def run_tests(self) -> str:
        """Run tests to ensure functionality of the object.

        Returns:
            str: "Success!" if all tests passed
        """
        assert self.sig == self.get_conductivity(), "ElectricObject conductivity does not match."
        assert self.eps_r == self.get_relative_permittivity(), "ElectricObject permittivity does not match."
        return "Success!"

    @classmethod
    def _initialize_input_argument_names(cls):
        """Keep track of the required input arguments."""
        return ["conductivity=1", "relative_permittivity=1"]
