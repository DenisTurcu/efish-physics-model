import sys
import numpy as np

sys.path.append("helper_functions/")
sys.path.append("../helper_functions/")

from conversions import convert2mainSI  # noqa: E402
from FishGeneration import FishGeneration  # noqa: E402
from Fish import Fish  # noqa: E402


class IceCreamConeFish(FishGeneration):
    """Represents an ice-cream cone shaped fish. Parent class "FishGeneration":"""

    __doc__ += FishGeneration.__doc__  # type: ignore

    def __init__(
        self,
        vertical_semi_axis: float | tuple = 0.1,
        lateral_semi_axis: float | tuple | None = None,
        rostrocaudal_semi_axis: float | tuple | None = None,
        _init_tests: bool = True,
        _super_init: bool = False,
        **kwds,
    ):
        """This class represents an ice-cream cone shaped fish. The receptors are placed on the surface of the fish,
        mimicking the shape of an ice-cream cone. The cone has elliptical cross-sections given by the vertical and
        lateral semi-axes. The cone is then extended with a ellipsoidal cap in the rostro-caudal direction to form
        the body of the fish. The length of the fish is made up by the rostro-caudal semi-axis (the cone cap) and
        the height of the cone.

        Args:
            vertical_semi_axis (float | tuple, optional): The vertical semi-axis of the cone. Defaults to 0.1.
            lateral_semi_axis (float | tuple | None, optional): The lateral semi-axis of the cone (typically smaller
                than the vertical). Defaults to None.
            rostrocaudal_semi_axis (float | tuple | None, optional): The rostro-caudal semi-axis of the cone's cap.
                Defaults to None.
            _init_tests (bool, optional): Run init tests or not. Defaults to True.
            _super_init (bool, optional): Whether the super was initialized or not. Defaults to False.

        Parent __doc__:\n
        """
        IceCreamConeFish.__init__.__doc__ += super().__init__.__doc__  # type: ignore

        super().__init__(**kwds, _init_tests=False)

        # Fish Head Semi-Axes
        self.ver_ax = convert2mainSI(vertical_semi_axis)
        self.lat_ax = self.ver_ax if lateral_semi_axis is None else convert2mainSI(lateral_semi_axis)
        self.roc_ax = self.ver_ax if rostrocaudal_semi_axis is None else convert2mainSI(rostrocaudal_semi_axis)

        if not _super_init:
            self.initialize_receptors_and_normals(self.receptors_init)

        self._initialize_input_argument_names()

        # Run assertion tests
        if _init_tests:
            print(self.run_tests())

    def is_equal(self, other: Fish, simple_return: bool = True):
        IceCreamConeFish.is_equal.__doc__ = super().is_equal.__doc__  # type: ignore

        _, truth_values, comparison = super().is_equal(other, simple_return=False)  # type: ignore
        if isinstance(other, self.__class__):
            truth_values.append(np.abs(self.ver_ax - other.get_vertical_semi_axis()) < self.assert_err)
            truth_values.append(np.abs(self.lat_ax - other.get_lateral_semi_axis()) < self.assert_err)
            truth_values.append(np.abs(self.roc_ax - other.get_rostro_caudal_semi_axis()) < self.assert_err)

            comparison.append("vertical_semi_axis")
            comparison.append("lateral_semi_axis")
            comparison.append("rostro_caudal_semi_axis")
        truth_value = False if len(truth_values) == 0 else np.array(truth_values).all()
        if simple_return:
            return truth_value
        else:
            return truth_value, truth_values, comparison

    def init_receptors_and_normals_random(self, receptors_init: dict) -> tuple[np.ndarray, np.ndarray]:
        """Initialize the receptors and normals randomly on the fish surface, i.e. the surface of the ice-cream cone.
        This initialization is randomly uniform in parametrization space of the surface.

        Args:
            receptors_init (dict): Dictionary containing the number of receptors to initialize on the "head" and "body"
                of the fish. "head" corresponds to the cap of the cone, and "body" corresponds to the cone itself.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing the receptor locations and normals.
        """
        # HEAD POINTS
        #       construct unit radius semi-sphere with base on zy plane
        # start by generating points in 2D, and then expand to 3D
        # randomly generate 1.5x the number of receptors needed, and then filter out the ones outside the unit circle
        head_points_2D = np.random.rand(int(1.5 * receptors_init["head"]), 2) * 2 - 1
        head_points_2D = head_points_2D[np.power(head_points_2D, 2).sum(1) < 1]
        head_points_2D = head_points_2D[: receptors_init["head"]]
        # expand the HEAD POINTS into 3D
        head_points_3D = np.zeros([head_points_2D.shape[0], 3])
        head_points_3D[:, 1:] = head_points_2D
        head_points_3D[:, 0] = np.sqrt(1 - np.power(head_points_2D, 2).sum(1))
        # scale from unit radius sphere to ellipsoidal cap of the cone
        head_points_3D *= np.array([self.roc_ax, self.lat_ax, self.ver_ax])
        # HEAD NORMALS
        head_normals = head_points_3D / np.power(np.array([self.roc_ax, self.lat_ax, self.ver_ax]), 2)
        head_normals = head_normals / np.sqrt(np.power(head_normals, 2).sum(1, keepdims=True))

        # BODY POINTS
        #       construct cone with unit height and unit base radius, with base on zy plane
        # start by generating points in 2D, and then expand to 3D
        # randomly generate 1.5x the number of receptors needed, and then filter out the ones outside the unit circle
        body_points_2D = np.random.rand(int(1.5 * receptors_init["body"]), 2) * 2 - 1
        body_points_2D = body_points_2D[np.power(body_points_2D, 2).sum(1) < 1]
        body_points_2D = body_points_2D[: receptors_init["body"]]
        body_points_3D = np.zeros([body_points_2D.shape[0], 3])
        body_points_3D[:, 1:] = body_points_2D
        body_points_3D[:, 0] = 1 - np.sqrt(np.power(body_points_2D, 2).sum(1))
        # scale from unit radius & height cone to arbitrary cone
        #       '-' for the x axis ensures the head and body are continuous
        body_points_3D *= np.array([-(self.length - self.roc_ax), self.lat_ax, self.ver_ax])
        # BODY NORMALS
        body_normals = body_points_3D / np.power(np.array([-(self.length - self.roc_ax), self.lat_ax, self.ver_ax]), 2)
        body_normals[:, 0] = -body_normals[:, 0] - 1 / (self.length - self.roc_ax)
        body_normals = body_normals / np.sqrt(np.power(body_normals, 2).sum(1, keepdims=True))

        # stack all points and normals together
        locations = np.vstack([head_points_3D, body_points_3D])
        normals = np.vstack([head_normals, body_normals])
        assert locations.shape[1] == 3, "Fish points should be shape (N,3)."
        assert normals.shape[1] == 3, "Fish normals should be shape (N,3)."
        assert locations.shape[0] == normals.shape[0], "Number of points and normals should match."
        assert (
            (np.power(normals, 2).sum(1) - 1) < self.assert_err
        ).all(), "Fish normals should be unit length vectors."

        # translate the fish: after below line, the fish will be straight along x axis; tail is at [0,0,0] and nose
        # is at [self.length,0,0]
        locations += np.array([(self.length - self.roc_ax), 0, 0])
        assert locations[:, 0].min() > 0, "Fish should currently be positioned in the +x semi-space"
        return locations, normals

    def details(self) -> str:
        """Provides the details of the IceCreamConeFish class. Parent __doc__:\n"""
        IceCreamConeFish.details.__doc__ += super().details.__doc__  # type: ignore

        details_string = super().details()
        details_string += (
            f"head semi-axes: vertical {self.ver_ax}, lateral {self.lat_ax}, "
            f"rostro-caudal {self.roc_ax} (unit: meter)\n"
        )
        return details_string

    def get_vertical_semi_axis(self) -> float | np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            float | np.ndarray
        """
        return self.ver_ax

    def get_lateral_semi_axis(self) -> float | np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            float | np.ndarray
        """
        return self.lat_ax

    def get_rostro_caudal_semi_axis(self) -> float | np.ndarray:
        """Extract the named property of the fish object (immutable).

        Returns:
            float | np.ndarray
        """
        return self.roc_ax

    @classmethod
    def _initialize_input_argument_names(cls) -> list:
        IceCreamConeFish._initialize_input_argument_names.__func__.__doc__ = (
            super()._initialize_input_argument_names.__doc__
        )
        inp_args = super()._initialize_input_argument_names()
        inp_args += ["vertical_semi_axis", "lateral_semi_axis=None", "rostrocaudal_semi_axis=None"]
        return inp_args

    def run_tests(self) -> str:
        IceCreamConeFish.run_tests.__doc__ = super().run_tests.__doc__

        super().run_tests()
        return "Success!"
