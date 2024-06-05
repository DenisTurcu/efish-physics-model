from scipy.spatial.transform import Rotation as R  # type: ignore
import numpy as np
import sys

sys.path.append("../helper_functions/")

from conversions import convert2mainSI  # noqa: E402


class Boundary:
    """Represents a parent class for aquarium boundary types."""

    def __init__(self, central_point: list | np.ndarray, assert_err: float = 1e-12):
        """Initialize the boundary for the aquarium.

        Args:
            central_point (list | np.ndarray): Reference point of the boundary. E.g. for planar boundary this would
                help compute normals, for spherical boundary this would be the center etc.
            assert_err (float, optional): Assertion error value for what counts as "0" error. Defaults to 1e-12.
        """
        self.central_point = convert2mainSI(central_point)
        self.assert_err = assert_err
        self.type = "boundary"

    def get_reference_point(self):
        return self.central_point

    def get_type(self):
        return self.type

    def verify_points(self, points: list | np.ndarray, simple_return: bool = True) -> bool | tuple:
        """Verify that provided points are located within boundary limits. E.g. for planar boundary this means on the
        correct side of the plane, for spherical this means inside the sphere etc.

        Args:
            points (list | np.ndarray): Given points locations to test for positioning with respect to the boundary.
            simple_return (bool, optional): Whether to return the truth value of the test, or return the processed
                points as well with individual test for each point (helps identify points that failed the boundary
                test). Defaults to True.

        Returns:
            bool | tuple: Truth value of test, or itemized truth value for each point with the points themselves.
        """
        points = np.array(points)
        if (len(points.shape) == 1) and (points.shape[0] == 3):
            points = points.reshape(-1, 3)
        assert points.shape[1] == 3, "Given points must be shape (N,3)."  # type: ignore
        if simple_return:
            return False
        else:
            return False, points


class Plane(Boundary):
    """Represents a planar boundary, defined by a normal vector and a central point that lies on the plane.
    Parent class "Boundary":"""

    __doc__ += Boundary.__doc__  # type: ignore

    def __init__(self, normal: list | np.ndarray, **kwds):
        """Initialize a planar boundary for the aquarium.

        Args:
            normal (list | np.ndarray): The normal vector that defines the direction of the half-space that is within
                boundary limits.
        """
        super().__init__(**kwds)
        assert np.abs(normal).sum() > 0, "Normal vector cannot be the 0 vector."
        self.normal = np.array(normal) / np.linalg.norm(normal)  # normalize normal vector
        assert self.normal.shape == np.array([0, 0, 0]).shape, "Normal shape should be (3,), i.e. not (3,1) or (1,3)."
        assert (np.linalg.norm(self.normal) - 1) < self.assert_err, "Normal should be unit vector."
        self.type = "plane"

    def verify_points(self, points, simple_return=True):
        """Verify points for the specific "Plane" boundary. See parent class "Boundary" for more details."""

        _, points = super().verify_points(points, simple_return=False)  # type: ignore
        if simple_return:
            return (((points - self.central_point) @ self.normal) >= 0).all()
        else:
            return (((points - self.central_point) @ self.normal) >= 0), points

    def get_normal(self) -> np.ndarray:
        """Extract the normal to the planar boundary. (immutable)"""
        return self.normal


class Ellipsoid(Boundary):
    """Represents an ellipsoid boundary, defined by a three semi-axes and three rotation angles (Euler, zyx) and
    a central point. See parent class "Boundary" for more details."""

    def __init__(
        self,
        axis_x: float,
        axis_y: float,
        axis_z: float,
        yaw: float | None = None,
        pitch: float | None = None,
        roll: float | None = None,
        **kwds
    ):
        """Initialize an ellipsoidal boundary for the aquarium. The ellipsoid is defined by three semi-axes and
        three rotation angles.

        Args:
            axis_x (float): Size of the semi-axis in the x-direction.
            axis_y (float): Size of the semi-axis in the y-direction.
            axis_z (float): Size of the semi-axis in the z-direction.
            yaw (float | None): Angle of rotation of the ellipsoid around the yaw-axis (~horizontal rotation).
                Defaults to "None".
            pitch (float | None): Angle of rotation of the ellipsoid around the pitch-axis (~nosedive rotation).
                Defaults to "None".
            roll (float | None): Angle of rotation of the ellipsoid around the roll-axis (~left/right rotation).
                Defaults to "None".
        """
        super().__init__(**kwds)
        self.axis_x = convert2mainSI(axis_x)
        self.axis_y = convert2mainSI(axis_y)
        self.axis_z = convert2mainSI(axis_z)
        self.yaw = convert2mainSI(yaw)  # type: ignore
        self.pitch = convert2mainSI(pitch)  # type: ignore
        self.roll = convert2mainSI(roll)  # type: ignore
        if (yaw is None) and (pitch is None) and (roll is None):
            self.rotation = None
        else:
            self.rotation = R.from_euler("zyx", [-self.yaw, -self.pitch, -self.roll])
        self.type = "ellipsoid"

    def verify_points(self, points, simple_return=True):
        """Verify points for the specific "Ellipsoid" boundary. See parent class "Boundary" for more details."""

        _, points = super().verify_points(points, simple_return=False)  # type: ignore
        # relative points w.r.t ellipsoid center
        rel_points = points - self.central_point
        # rotate relative points to align coordinate frame with the ellipsoid's frame
        if self.rotation is not None:
            rel_points = rel_points @ self.rotation.inv().as_matrix()
        # proportionally divide the points to the corresponding semi-axis
        rel_points = rel_points / np.array([self.axis_x, self.axis_y, self.axis_z])
        # check that the transformed relative points lie within a sphere of unit radius
        if simple_return:
            return (np.power(rel_points, 2).sum(1) < 1).all()
        else:
            return (np.power(rel_points, 2).sum(1) < 1), points


class Sphere(Ellipsoid):
    """Represents a spherical boundary, defined by a radius and a central point which lies on the plane.
    Parent class "Ellipsoid":"""

    __doc__ += Ellipsoid.__doc__  # type: ignore

    def __init__(self, radius: float, **kwds):
        """Same as "Ellipsoid", but with equal semi-axes and no rotations.
        See parent class "Ellipsoid" for more details."""

        super().__init__(axis_x=radius, axis_y=radius, axis_z=radius, **kwds)  # type: ignore
        self.type = "sphere"


# class Cylinder(Sphere):


# class EllipseCylinder(Cylinder):
