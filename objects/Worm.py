import numpy as np
import sys
from typing import Self

sys.path.append("helper_functions/")
sys.path.append("../helper_functions/")

from conversions import convert2mainSI  # noqa: E402
from ElectricObject import ElectricObject  # noqa: E402


class Worm(ElectricObject):
    """Represents a foreign object to the fish, that is close-by. Parent class "ElectricObject":"""

    __doc__ += ElectricObject.__doc__  # type: ignore

    def __init__(self, center_position: list | np.ndarray = [0, 0, 0], _init_tests: bool = True, **kwds):
        """Initialize the Worm (nearby object w.r.t. the fish).

        Args:
            center_position (list | np.ndarray, optional): The center position of the worm in world coordinates.
                Defaults to [0, 0, 0].
            _init_tests (bool, optional): Run init tests or not. Defaults to True.

        See parent class "ElectricObject" for more details.
        """
        super().__init__(**kwds, _init_tests=False)
        self.r_vec = np.array(convert2mainSI(center_position))  # type: ignore
        self._initialize_input_argument_names()
        # Run assertion tests
        if _init_tests:
            print(self.run_tests())

    def is_equal(self, other: Self, simple_return: bool = True) -> bool | tuple:
        """Compare current worm to another worm for equality. See parent class "ElectricObject" for more details."""

        _, truth_values, comparison = super().is_equal(other, simple_return=False)  # type: ignore
        if isinstance(other, self.__class__):
            truth_values.append(
                (np.abs(self.r_vec - other.get_position()) < np.min([self.assert_err, other.assert_err])).all()
            )
            comparison.append("center_position")
        truth_value = False if len(truth_values) == 0 else np.array(truth_values).all()
        if simple_return:
            return truth_value  # type: ignore
        else:
            return truth_value, truth_values, comparison

    def create_graph_obj(self):
        raise NotImplementedError(
            'Need to implement the "create_graph_obj" function according to the geometry of the specified object. '
        )

    def details(self) -> str:
        """Provides the details of the worm. See parent class "ElectricObject" for more details."""

        details_string = super().details()
        details_string += f"This worm has center position {self.r_vec}.\n"
        return details_string

    def get_position(self) -> np.ndarray:
        """Extract the worm's position. (immutable)

        Returns:
            np.ndarray: [x, y, z].
        """
        return self.r_vec

    def get_points(self) -> np.ndarray:
        """Extract the worm's surface as sampled points. Point worm returns its center position.
        Large worms must return many points from their surface, and this method must be overwritten accordingly.

        Returns:
            np.ndarray: [x, y, z] for each sampled point (shape Nx3).
        """
        return np.array([self.get_position()])

    @classmethod
    def _initialize_input_argument_names(cls) -> list[str]:
        inp_args = super()._initialize_input_argument_names()
        inp_args += ["center_position=[0,0,0]"]
        return inp_args

    def run_tests(self):
        super().run_tests()
        assert (self.r_vec == self.get_position()).all(), "Worm position does not match."
        return "Success!"
