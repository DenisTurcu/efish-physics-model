from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
sys.path.append('../helper_functions/')

from prefixes import add_prefix, remove_prefix, return_numerical_prefix, parse_prefix
from conversions import rho24sig, convert2mainSI


class Boundary:
    ''' Represents a parent class for aquarium boundary types. '''

    def __init__(self, central_point, assert_err=1e-15):
        self.central_point = convert2mainSI(central_point)
        self.assert_err    = assert_err
        self.type          = 'boundary'

    def get_reference_point(self):
        return self.central_point

    def get_type(self):
        return self.type

    def verify_points(self, points, simple_return=True):
        points = np.array(points)
        if (len(points.shape) == 1) and (points.shape[0] == 3):
            points = points.reshape(-1,3)
        assert points.shape[1] == 3, 'Given points must be shape (N,3).'
        if simple_return:
            return False
        else:
            return False, points


class Plane(Boundary):
    ''' Represents a planar boundary, defined by a normal vector and a central point which lies on the plane. Parent class "Boundary": '''
    __doc__ += Boundary.__doc__

    def __init__(self, normal, **kwds):
        super().__init__(**kwds)
        assert np.abs(normal).sum()>0, 'Normal vector cannot be the 0 vector.'
        self.normal = np.array(normal) / np.sqrt(np.power(normal,2).sum())
        assert self.normal.shape == np.array([0,0,0]).shape, 'Normal shape should be (3,), i.e. not (3,1) or (1,3).'
        assert (np.power(self.normal,2).sum() - 1) < self.assert_err, 'Normal should be unit vector.'
        self.type = 'plane'

    def verify_points(self, points, simple_return=True):
        ''' Check that given points lie within the boundary. '''
        _, points = super().verify_points(points, simple_return=False)
        if simple_return:
            return (((points - self.central_point) @ self.normal) >= 0).all()
        else:
            return (((points - self.central_point) @ self.normal) >= 0), points

    def get_normal(self):
        return self.normal


class Ellipsoid(Boundary):
    ''' Represents an ellipsoid boundary, defined by a three semi-axes and three rotation angles (Euler, zyx) and a central point. Parent class "Boundary": '''
    __doc__ += Boundary.__doc__

    def __init__(self, axis_x, axis_y, axis_z, yaw, pitch, roll, **kwds):
        super().__init__(**kwds)
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z
        if (yaw is None) and (pitch is None) and (roll is None):
            self.rotation = None
        else:
            self.rotation = R.from_euler('zyx', [-self.yaw, -self.pitch, -self.roll])
        self.type = 'ellipsoid'

    def verify_points(self, points, simple_return=True):
        ''' Check that given points lie within the boundary. '''
        _, points = super().verify_points(points, simple_return=False)
        # relative points w.r.t ellipsoid center
        rel_points = (points - self.central_point)
        # rotate relative points to allign coordinate frame with the ellipsoid's frame
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
    ''' Represents a spherical boundary, defined by a radius and a central point which lies on the plane. Parent class "Ellipsoid": '''
    __doc__ += Ellipsoid.__doc__

    def __init__(self, radius, **kwds):
        super().__init__(axis_x=radius, axis_y=radius, axis_z=radius, yaw=None, pitch=None, roll=None, **kwds)
        self.type = 'sphere'


# class Cylinder(Sphere):


# class EllipseCylinder(Cylinder):


