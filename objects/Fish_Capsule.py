import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('../helper_functions/')
sys.path.append('../uniform_points_generation/')

from param_tools import r_surface
from prefixes import add_prefix, remove_prefix, return_numerical_prefix, parse_prefix
from conversions import rho24sig, convert2mainSI
from Fish_IceCreamCone import IceCreamConeFish


class CapsuleFish(IceCreamConeFish):
    ''' Represents a capsul shaped fish. Parent class "IceCreamConeFishGeneration": '''
    __doc__ += IceCreamConeFish.__doc__

    def __init__(self, rostrocaudal_semi_axis_tail=None,
                       _init_tests=True, _super_init=False, **kwds):

        super().__init__(**kwds, _init_tests=False, _super_init=True)

        # Fish TAIL Semi-Axes
        self.roc_ax_t = self.roc_ax if rostrocaudal_semi_axis_tail is None else convert2mainSI(rostrocaudal_semi_axis_tail)
        assert self.length - self.roc_ax - self.roc_ax_t > 0, 'Fish mid-body length must be positive.'
        
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
            thruth_values.append(np.abs(self.roc_ax_t            - other.get_rostro_caudal_semi_axis_tail()) < self.assert_err)

            comparison.append('rostro_caudal_semi_axis_tail')
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
        head_normals = head_normals / np.linalg.norm(head_normals, axis=1, keepdims=True)

        # TAIL POINTS
        # construct unit radius hemi-sphere with base on zy plane
        tail_points_2D = np.random.rand(int(1.5*receptors_init['tail']),2) * 2 - 1
        tail_points_2D = tail_points_2D[np.power(tail_points_2D,2).sum(1)<1]
        tail_points_2D = tail_points_2D[:receptors_init['tail']]
        tail_points_3D = np.zeros([tail_points_2D.shape[0],3])
        tail_points_3D[:,1:] = tail_points_2D
        tail_points_3D[:,0]  = -np.sqrt(1-np.power(tail_points_2D,2).sum(1))
        # scale from unit radius sphere to ellipsoid
        tail_points_3D*= np.array([self.roc_ax_t, self.lat_ax, self.ver_ax])  
        # TAIL NORMALS
        tail_normals = tail_points_3D / np.power(np.array([self.roc_ax_t, self.lat_ax, self.ver_ax]),2)
        tail_normals = tail_normals / np.linalg.norm(tail_normals, axis=1, keepdims=True)
        # translate the tail points to the tail position
        tail_points_3D[:,0]-= self.length-self.roc_ax-self.roc_ax_t

        # BODY POINTS
        # construct cylinder with unit height and unit radius, with base on zy plane
        body_points = np.zeros([receptors_init['body'], 3])
        body_points[:,0] = - np.random.rand(receptors_init['body'])
        phi = np.random.rand(receptors_init['body']) * 2 * np.pi
        body_points[:,1] = np.sin(phi)
        body_points[:,2] = np.cos(phi)
        # scale the unit cylinder to the appropriate body dimensions
        body_points*= np.array([self.length-self.roc_ax-self.roc_ax_t, self.lat_ax, self.ver_ax])
        # BODY Normals
        body_normals = body_points / np.power(np.array([1, self.lat_ax, self.ver_ax]), 2)
        body_normals[:,0] = 0
        body_normals = body_normals / np.linalg.norm(body_normals, axis=1, keepdims=True)

        # stack all points and normals together
        points  = np.vstack([head_points_3D, body_points, tail_points_3D])
        normals = np.vstack([head_normals, body_normals, tail_normals])
        assert points.shape[1] == 3, 'Fish points should be shape (N,3).'
        assert normals.shape[1] == 3, 'Fish normals should be shape (N,3).'
        assert points.shape[0] == normals.shape[0], 'Number of points and normals should match.'
        assert ((np.power(normals,2).sum(1) - 1) < self.assert_err).all(), 'Fish normals should be unit length vectors.'

        # translate the fish: after below line, the fish will be straight along x axis; tail is at [0,0,0] and nose is at [self.length,0,0]
        points[:,0] += self.length-self.roc_ax
        assert points[:,0].min() >= 0, 'Fish should currently be positioned in the positive x semi-space'
        return points, normals


    def init_receptors_and_normals_grid_uniDense(self, N_receptors):
        # def brief helper functions
        def ellipsoid(t, u, a=6, b=4, c=8):
            return np.array([a * np.cos(u), 
                             b * np.sin(u) * np.sin(t), 
                             c * np.sin(u) * np.cos(t)])
        domain_t_ellipsoid = [0, 2*np.pi]
        domain_u_ellipsoid = [0, np.pi/2]

        def elliptic_cyllinder(t, u, h=1, a=2, b=3):
            return np.array([h * t,
                             a * np.cos(u),
                             b * np.sin(u)])
        domain_t_cyllinder = [0, 1]
        domain_u_cyllinder = [0, 2*np.pi]


        locations = np.array([]).reshape(0,3)
        normals   = np.array([]).reshape(0,3)
        
        # HEAD
        if (N_receptors['head_t']>0) and (N_receptors['head_u']>0): 
            # POINTS
            head_points, _, _, _, _ = r_surface(0, 
                                                (lambda t, u: ellipsoid(t, u, self.roc_ax, self.lat_ax, self.ver_ax)),
                                                *domain_t_ellipsoid,
                                                *domain_u_ellipsoid,
                                                100, 100, 
                                                grid_like=dict(N_points_t=N_receptors['head_t']+1, N_points_u=N_receptors['head_u']))
            head_points = head_points.reshape(3,-1).T
            head_points = head_points[:-(head_points.shape[0]//(N_receptors['head_t']+1))]
            locations  = np.vstack([locations, head_points])

            # NORMALS
            head_normals = head_points / np.power(np.array([self.roc_ax, self.lat_ax, self.ver_ax]),2)
            head_normals = head_normals / np.linalg.norm(head_normals, axis=1, keepdims=True)
            normals = np.vstack([normals, head_normals])

        # TAIL 
        if (N_receptors['tail_t']>0) and (N_receptors['tail_u']>0): 
            # POINTS
            tail_points, _, _, _, _ = r_surface(0, 
                                                (lambda t, u: ellipsoid(t, u, -self.roc_ax_t, self.lat_ax, self.ver_ax)),
                                                *domain_t_ellipsoid,
                                                *domain_u_ellipsoid,
                                                100, 100, 
                                                grid_like=dict(N_points_t=N_receptors['tail_t']+1, N_points_u=N_receptors['tail_u']))
            tail_points = tail_points.reshape(3,-1).T
            tail_points = tail_points[:-(tail_points.shape[0]//(N_receptors['tail_t']+1))]
            tail_points[:,0]-= self.length-self.roc_ax-self.roc_ax_t
            locations  = np.vstack([locations, tail_points])

            # NORMALS
            tail_normals = tail_points / np.power(np.array([self.roc_ax_t, self.lat_ax, self.ver_ax]),2)
            tail_normals = tail_normals / np.linalg.norm(tail_normals, axis=1, keepdims=True)
            normals = np.vstack([normals, tail_normals])

        # BODY 
        if (N_receptors['body_t']>0) and (N_receptors['body_u']>0): 
            # POINTS
            body_points, _, _, _, _ = r_surface(0, 
                                                (lambda t, u: elliptic_cyllinder(t, u, -(self.length-self.roc_ax-self.roc_ax_t), self.lat_ax, self.ver_ax)),
                                                *domain_t_cyllinder,
                                                *domain_u_cyllinder,
                                                100, 100, 
                                                grid_like=dict(N_points_t=N_receptors['body_t']+1, N_points_u=N_receptors['body_u']+1))
            body_points = body_points.reshape(3,-1).T
            body_points = body_points[:-(body_points.shape[0]//(N_receptors['body_u']+1))]
            locations  = np.vstack([locations, body_points])

            # NORMALS
            body_normals = body_points / np.power(np.array([1, self.lat_ax, self.ver_ax]), 2)
            body_normals[:,0] = 0
            body_normals = body_normals / np.linalg.norm(body_normals, axis=1, keepdims=True)
            normals = np.vstack([normals, body_normals])

        # stack all points and normals together
        assert locations.shape[1] == 3, 'Fish points should be shape (N,3).'
        assert normals.shape[1] == 3, 'Fish normals should be shape (N,3).'
        assert locations.shape[0] == normals.shape[0], 'Number of points and normals should match.'
        assert ((np.power(normals,2).sum(1) - 1) < self.assert_err).all(), 'Fish normals should be unit length vectors.'

        # translate the fish: after below line, the fish will be straight along x axis; tail is at [0,0,0] and nose is at [self.length,0,0]
        locations[:,0] += self.length-self.roc_ax
        assert locations[:,0].min() >= 0, 'Fish should currently be positioned in the positive x semi-space'

        return locations, normals


    def details(self):
        ''' Provides the details of the object. '''
        details_string  = super().details()
        details_string +=('tail semi-axes: vertical %r, lateral %r, rostro-caudal %r (unit: meter)\n') % \
                          (self.ver_ax, self.lat_ax, self.roc_ax_t)
        return details_string


    def get_rostro_caudal_semi_axis_tail(self):
        return self.roc_ax_t


    @classmethod
    def _initialize_input_argument_names(cls):
        inp_args  = super()._initialize_input_argument_names()
        inp_args += ['rostrocaudal_semi_axis_tail=None']
        return inp_args


    def run_tests(self):
        ''' Sanity assertion checks to ensure code robustness. '''
        super().run_tests()
        return 'Success!'

