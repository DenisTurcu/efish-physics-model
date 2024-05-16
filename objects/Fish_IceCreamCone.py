import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('../helper_functions/')

from prefixes import add_prefix, remove_prefix, return_numerical_prefix, parse_prefix
from conversions import rho24sig, convert2mainSI
from FishGeneration import FishGeneration


class IceCreamConeFish(FishGeneration):
    ''' Represents an ice-cream cone shaped fish. Parent class "FishGeneration": '''
    __doc__ += FishGeneration.__doc__

    def __init__(self, vertical_semi_axis=0.1, lateral_semi_axis=None, rostrocaudal_semi_axis=None,
                       _init_tests=True, _super_init=False, **kwds):

        super().__init__(**kwds, _init_tests=False)

        # Fish Head Semi-Axes
        self.ver_ax = convert2mainSI(vertical_semi_axis)
        self.lat_ax = self.ver_ax if lateral_semi_axis      is None else convert2mainSI(lateral_semi_axis)
        self.roc_ax = self.ver_ax if rostrocaudal_semi_axis is None else convert2mainSI(rostrocaudal_semi_axis)

        if not _super_init:
            self.initialize_receptors_and_normals(self.receptors_init)

        self._initialize_input_argument_names()

        # Run assertion tests
        if _init_tests:
            print(self.run_tests())


    def is_equal(self, other, simple_return=True):
        ''' Compare current worm to another worm for equality. '''
        _, thruth_values, comparison  = super().is_equal(other, simple_return=False)
        if isinstance(other, self.__class__):
            thruth_values.append(np.abs(self.ver_ax            - other.get_vertical_semi_axis()) < self.assert_err)
            thruth_values.append(np.abs(self.lat_ax            - other.get_lateral_semi_axis()) < self.assert_err)
            thruth_values.append(np.abs(self.roc_ax            - other.get_rostro_caudal_semi_axis()) < self.assert_err)

            comparison.append('vertical_semi_axis')
            comparison.append('lateral_semi_axis')
            comparison.append('rostro_caudal_semi_axis')
        thruth_value = False if len(thruth_values) == 0 else np.array(thruth_values).all()
        if simple_return:
            return thruth_value 
        else: 
            return thruth_value, thruth_values, comparison


    def init_receptors_and_normals_random(self, receptors_init):
        ''' Initialize points and normals random on the fish surface. (uniform in parametrization space) '''
        # HEAD POINTS
        # construct unit radius hemi-sphere with base on zy plane
        head_points_2D = np.random.rand(int(1.5*receptors_init['head']),2) * 2 - 1
        head_points_2D = head_points_2D[np.power(head_points_2D,2).sum(1)<1]
        head_points_2D = head_points_2D[:receptors_init['head']]
        head_points_3D = np.zeros([head_points_2D.shape[0],3])
        head_points_3D[:,1:] = head_points_2D
        head_points_3D[:,0]  = np.sqrt(1-np.power(head_points_2D,2).sum(1))
        # scale from unit radius sphere to ellipsoid
        head_points_3D*= np.array([self.roc_ax, self.lat_ax, self.ver_ax])  
        # HEAD NORMALS
        head_normals = head_points_3D / np.power(np.array([self.roc_ax, self.lat_ax, self.ver_ax]),2)
        head_normals = head_normals / np.sqrt(np.power(head_normals,2).sum(1, keepdims=True))

        # BODY POINTS
        # construct cone with unit height and unit base radius, with base on zy plane
        body_points_2D = np.random.rand(int(1.5*receptors_init['body']),2) * 2 - 1
        body_points_2D = body_points_2D[np.power(body_points_2D,2).sum(1)<1]
        body_points_2D = body_points_2D[:receptors_init['body']]
        body_points_3D = np.zeros([body_points_2D.shape[0],3])
        body_points_3D[:,1:] = body_points_2D
        body_points_3D[:,0]  = 1-np.sqrt(np.power(body_points_2D,2).sum(1))
        # scale from unit radius & height cone to arbitrary cone
        # '-' for the x axis ensures the head and body are continuous
        body_points_3D*= np.array([-(self.length-self.roc_ax), self.lat_ax, self.ver_ax])
        # BODY NORMALS
        body_normals = body_points_3D / np.power(np.array([-(self.length-self.roc_ax), self.lat_ax, self.ver_ax]),2)
        body_normals[:,0] = -body_normals[:,0] - 1/(self.length-self.roc_ax)
        body_normals = body_normals / np.sqrt(np.power(body_normals, 2).sum(1,keepdims=True))

        # stack all points and normals together
        locations  = np.vstack([head_points_3D, body_points_3D])
        normals = np.vstack([head_normals, body_normals])
        assert locations.shape[1] == 3, 'Fish points should be shape (N,3).'
        assert normals.shape[1] == 3, 'Fish normals should be shape (N,3).'
        assert locations.shape[0] == normals.shape[0], 'Number of points and normals should match.'
        assert ((np.power(normals,2).sum(1) - 1) < self.assert_err).all(), 'Fish normals should be unit length vectors.'

        # translate the fish: after below line, the fish will be straight along x axis; tail is at [0,0,0] and nose is at [self.length,0,0]
        locations += np.array([(self.length-self.roc_ax),0,0])
        assert locations[:,0].min() > 0, 'Fish should currently be positioned in the positive x semi-space'
        return locations, normals


    def details(self):
        ''' Provides the details of the object. '''
        details_string  = super().details()
        details_string +=(f'head semi-axes: vertical {self.ver_ax}, lateral {self.lat_ax}, rostro-caudal {self.roc_ax} (unit: meter)\n')
        return details_string


    def get_vertical_semi_axis(self):
        return self.ver_ax


    def get_lateral_semi_axis(self):
        return self.lat_ax


    def get_rostro_caudal_semi_axis(self):
        return self.roc_ax


    @classmethod
    def _initialize_input_argument_names(cls):
        inp_args  = super()._initialize_input_argument_names()
        inp_args += ['vertical_semi_axis', 'lateral_semi_axis=None', 'rostrocaudal_semi_axis=None']
        return inp_args


    def run_tests(self):
        ''' Sanity assertion checks to ensure code robustness. '''
        super().run_tests()
        return 'Success!'
