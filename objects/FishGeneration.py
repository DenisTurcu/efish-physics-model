import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


import sys
sys.path.append('../helper_functions/')

from prefixes import add_prefix, remove_prefix, return_numerical_prefix, parse_prefix
from conversions import rho24sig, convert2mainSI
from Fish import Fish


class FishGeneration(Fish):
    ''' Represents a fish with specific manual generation of receptors and point currents. Parent class "Fish": '''
    __doc__ += Fish.__doc__

    def __init__(self, nose_position=[0,0,0],
                       fish_length=1,
                       angle_yaw=0,
                       angle_pitch=0,
                       angle_roll=0,
                       relative_bend_locations_percentage=np.array([]),
                       relative_bend_angle_lateral=(np.array([]),'deg'),
                       relative_bend_angle_dorso_ventral =(np.array([]),'deg'), 
                       point_currents_range_percentage=dict(start=0, end=100),
                       N_point_currents=101,
                       point_currents_magnitude_scale=1,
                       receptors_init=dict(method='random', head=100, body=200),
                       _init_tests=True, **kwds):
        super().__init__(**kwds, _init_tests=False)
        assert N_point_currents >= 2, 'At least two point currents must exist. '

        # position related attributes
        self.nose_position       = np.array(convert2mainSI(nose_position))
        self.length              = convert2mainSI(fish_length)

        # Fish Receptors
        self.receptors_init = receptors_init

        # rotation related atributes
        self.yaw                 = convert2mainSI(angle_yaw)
        self.pitch               = convert2mainSI(angle_pitch)
        self.roll                = convert2mainSI(angle_roll)
        self.initialize_fish_bend_details(relative_bend_locations_percentage, relative_bend_angle_lateral, relative_bend_angle_dorso_ventral)

        # initialization methods
        self.initialize_main_rotation()
        self._initialize_input_argument_names()

        ####################
        self.N_point_currents                = N_point_currents
        self.point_currents_range_percentage = point_currents_range_percentage
        self.point_currents_magnitude_scale  = point_currents_magnitude_scale
        self.initialize_point_currents(point_currents_magnitude_scale)

        # Run assertion tests
        if _init_tests:
            print(self.run_tests())
    

    def update_receptors(self, receptors_locations=None, receptors_normals=None, base_receptors=True):
        ''' Update the receptors of the fish, i.e. move the fish in space. '''
        if base_receptors:
            self.base_receptors_locations        = self.base_receptors_locations       if receptors_locations is None else receptors_locations + np.array([self.length, 0, 0])
            self.base_receptors_normals          = self.base_receptors_normals         if receptors_normals   is None else receptors_normals
            assert self.base_receptors_locations.shape[0]       == self.base_receptors_normals.shape[0],        'Number of receptors locations and normals must match.'
            assert (self.base_receptors_locations.shape[1] == 3) and (self.base_receptors_normals.shape[1] == 3), 'All locations and normals must be 3-dimensional.'
            if (self.base_receptors_locations.shape[0] == 0) or (self.base_receptors_normals.shape[0] == 0):
                print('Fish receptors were not properly innitialized - some have 0 length.')
            self.update_parameters()
        else:
            super().update_receptors(receptors_locations=receptors_locations, receptors_normals=receptors_normals)


    def update_parameters(self, 
                          nose_position=None,
                          angle_yaw=None, angle_pitch=None, angle_roll=None,
                          relative_bend_locations_percentage=None, relative_bend_angle_lateral=None, relative_bend_angle_dorso_ventral=None, 
                          point_currents_range_percentage=None, N_point_currents=None, point_currents_magnitude_scale=None):
        # receptors parameters
        self.nose_position  = self.nose_position  if nose_position  is None else nose_position

        # rotation parameters
        self.yaw      = self.yaw   if angle_yaw   is None else convert2mainSI(angle_yaw)
        self.pitch    = self.pitch if angle_pitch is None else convert2mainSI(angle_pitch)
        self.roll     = self.roll  if angle_roll  is None else convert2mainSI(angle_roll)
        rel_b_loc_per = self.relative_bend_locations_percentage if relative_bend_locations_percentage is None else relative_bend_locations_percentage
        rel_b_ang_lat = self.relative_bend_angle_lateral        if relative_bend_angle_lateral        is None else relative_bend_angle_lateral
        rel_b_ang_dv  = self.relative_bend_angle_dorso_ventral  if relative_bend_angle_dorso_ventral  is None else relative_bend_angle_dorso_ventral
        self.initialize_main_rotation()
        self.initialize_fish_bend_details(rel_b_loc_per, rel_b_ang_lat, rel_b_ang_dv)

        # point currents parameters
        self.N_point_currents                = self.N_point_currents                if N_point_currents                is None else N_point_currents
        self.point_currents_range_percentage = self.point_currents_range_percentage if point_currents_range_percentage is None else point_currents_range_percentage
        self.point_currents_magnitude_scale  = self.point_currents_magnitude_scale  if point_currents_magnitude_scale  is None else point_currents_magnitude_scale

        # update the point currents and receptors
        self.initialize_point_currents(self.point_currents_magnitude_scale)
        self.update_receptors(*self.bend_and_rigid_transform(self.base_receptors_locations, self.base_receptors_normals), base_receptors=False)


    def initialize_point_currents(self, magnitude_scale):
        f''' Define the point currents. Uniform distribution of charges in range {self.point_currents_range_percentage} % of fish length, on the center-line of the fish. '''
        point_currents_locations      = np.zeros([self.N_point_currents, 3])
        point_currents_locations[:,0] = np.linspace(self.length * (100 - self.point_currents_range_percentage['start']) / 100, 
                                                    self.length * (100 - self.point_currents_range_percentage['end'])   / 100, 
                                                    self.N_point_currents)
        point_currents_magnitudes     = np.ones(self.N_point_currents) / (self.N_point_currents-1)
        point_currents_magnitudes[-1] = -1
        assert point_currents_magnitudes.sum() < self.assert_err, 'Total sum of the point currents should be 0.'

        # store base, un-bent point currents locations
        self.base_point_currents_locations = point_currents_locations

        # Bend, rotate and translate the point currents (in this order specifically)
        point_currents_locations, _ = self.bend_and_rigid_transform(point_currents_locations)
        point_currents_magnitudes   = point_currents_magnitudes * magnitude_scale
        
        # update the point currents
        self.update_point_currents(point_currents_mag=point_currents_magnitudes, point_currents_loc=point_currents_locations)


    def initialize_receptors_and_normals(self, receptors_init):
        ''' Define the locations and fish surface normals of the receptors. '''
        if receptors_init is None:
            locations, normals = self.receptors_locations.copy(), self.receptors_normals.copy()
        elif receptors_init['method'] == 'random':
            locations, normals = self.init_receptors_and_normals_random(receptors_init)
        elif receptors_init['method'] == 'grid':
            locations, normals = self.init_receptors_and_normals_grid()
        elif receptors_init['method'] == 'random_uniDense':
            locations, normals = self.init_receptors_and_normals_random_uniDense()
        elif receptors_init['method'] == 'grid_uniDense':
            locations, normals = self.init_receptors_and_normals_grid_uniDense(receptors_init)
        elif receptors_init['method'] == 'manual':
            locations, normals = self.init_receptors_and_normals_manual(receptors_init)
        else:
            raise ValueError('Initialization method not defined.')

        # store the base, un-bent fish - fish is oriented in the positive x direction, tail is at [0,0,0] and nose is at [self.length, 0, 0]
        self.base_receptors_locations = locations
        self.base_receptors_normals   = normals

        # Bend, rotate and translate the fish (in this order specifically)
        locations, normals = self.bend_and_rigid_transform(locations, normals)

        # update the receptors
        self.update_receptors(receptors_locations=locations, receptors_normals=normals, base_receptors=False)


    def init_receptors_and_normals_random(self):
        ''' Initialize receptors and normals random on the fish surface. (uniform in parametrization space) '''
        raise NotImplementedError('Function not yet implemented.')


    def init_receptors_and_normals_random_uniDense(self):
        ''' Initialize receptors and normals uniformaly random, with uniform density on the fish surface. '''
        raise NotImplementedError('Function not yet implemented.')


    def init_receptors_and_normals_grid(self):
        ''' Initialize receptors and normals on grid on the fish surface. (uniform in parametrization space) '''
        raise NotImplementedError('Function not yet implemented.')


    def init_receptors_and_normals_grid_uniDense(self):
        ''' Initialize receptors and normals on grid, with uniform density, on the fish surface. '''
        raise NotImplementedError('Function not yet implemented.')


    # def init_receptors_and_normals_manual(self):
    #     ''' Initialize receptors and normals from manually given receptors locations and normals. '''
    #     raise NotImplementedError('Function not yet implemented.')
    def init_receptors_and_normals_manual(self, receptors_and_normals_dict):
        ''' Initialize points and normals from manually given points and normals. Consider straight fish with tail at (0,0,0) and head on the +x axis.'''
        locations  = receptors_and_normals_dict['points'].copy()
        normals = receptors_and_normals_dict['normals'].copy()
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        assert locations.shape[1] == 3, 'Fish points should be shape (N,3).'
        assert normals.shape[1] == 3, 'Fish normals should be shape (N,3).'
        assert locations.shape[0] == normals.shape[0], 'Number of points and normals should match.'
        assert ((np.power(normals,2).sum(1) - 1) < self.assert_err).all(), 'Fish normals should be unit length vectors.'

        return locations, normals


    def bend_and_rigid_transform(self, locations_, normals_=None):
        ''' Space transformation which first bends the given points according to fish bends, and then applies an overall rigid transformation. '''    
        # use copies to avoid modifying the unbend arrays
        locations = locations_.copy()
        normals = None if normals_ is None else normals_.copy()

        # BEND 
        # Start bending at each fish bend location, starting from the tail to the head
        for i in range(self.relative_bend_locations_percentage.shape[0]):
            segment_length = self.bend_lengths[-i-1]  # get the fish segment length to be bend
            rotation = self.bend_rotations[-i-1]  # get the rotation of the current segment
            locations -= np.array([segment_length,0,0])  # translate the current segment to origin
            locations[ locations[:,0]<0] = locations[ locations[:,0]<0] @ rotation.as_matrix()
            if normals is not None:
                normals[locations[:,0]<0] = normals[locations[:,0]<0] @ rotation.as_matrix()
        locations -= np.array([self.bend_lengths[0],0,0])  # translate: currently, i.e. after this line, nose is at [0,0,0] and receptors live in the negative x-semi-space

        # RIGID TRANSFORMs
        locations  = locations @ self.main_rotation.as_matrix()  # rotate
        locations += self.nose_position  # translate
        if normals is not None:
            normals = normals @ self.main_rotation.as_matrix()

        return locations, normals


    def initialize_fish_bend_details(self, relative_bend_locations_percentage, relative_bend_angle_lateral, relative_bend_angle_dorso_ventral):
        relative_bend_angle_lateral       = convert2mainSI(relative_bend_angle_lateral)
        relative_bend_angle_dorso_ventral = convert2mainSI(relative_bend_angle_dorso_ventral)
        assert (relative_bend_locations_percentage.shape == relative_bend_angle_lateral.shape),  'Number of bend points must match number of lateral bend angles.'
        assert (relative_bend_locations_percentage.shape == relative_bend_angle_dorso_ventral .shape),  'Number of bend points must match number of dorso-ventral bend angles.'

        self.relative_bend_locations_percentage = relative_bend_locations_percentage
        self.bend_lengths                       = np.diff(self.relative_bend_locations_percentage, prepend=0, append=100) * self.length / 100
        assert np.abs(self.bend_lengths.sum() - self.length) < self.assert_err, 'The bend lenghts must sum to the total fish length.'

        self.relative_bend_angle_lateral        = relative_bend_angle_lateral
        self.relative_bend_angle_dorso_ventral  = relative_bend_angle_dorso_ventral

        self.initialize_bend_rotations()
        

#########################
# CHECK ON THE RELEVANCE OF THE '-' SIGN FOR ALL ROTATIONS ANGLES
#########################
    def initialize_bend_rotations(self):
        ''' Initialize the rotation objects corresponding to fish bends. Note '-' sign convention for all angles. '''
        self.bend_rotations = []
        for i in range(self.relative_bend_locations_percentage.shape[0]):
            self.bend_rotations.append(R.from_euler('zy', [-self.relative_bend_angle_lateral[i], -self.relative_bend_angle_dorso_ventral[i]]))
        pass


    def initialize_main_rotation(self):
        ''' Initialize the rotation objects corresponding to overall rotation. Note '-' sign convention for all angles. '''
        self.main_rotation = R.from_euler('zyx', [-self.yaw, -self.pitch, -self.roll])
        return


    def is_equal(self, other, simple_return=True):
        ''' Compare current worm to another worm for equality. '''
        _, thruth_values, comparison  = super().is_equal(other, simple_return=False)
        if isinstance(other, self.__class__):
            thruth_values.append((np.abs(self.nose_position    - other.get_nose_position()) < self.assert_err).all())
            thruth_values.append(np.abs(self.length            - other.get_length()) < self.assert_err)
            thruth_values.append(np.abs(self.yaw               - other.get_yaw()) < self.assert_err)
            thruth_values.append(np.abs(self.pitch             - other.get_pitch()) < self.assert_err)
            thruth_values.append(np.abs(self.roll              - other.get_roll()) < self.assert_err)

            comparison.append('nose_position')
            comparison.append('length')
            comparison.append('yaw')
            comparison.append('pitch')
            comparison.append('roll')
        thruth_value = False if len(thruth_values) == 0 else np.array(thruth_values).all()
        if simple_return:
            return thruth_value  
        else:   
            return thruth_value, thruth_values, comparison


    def details(self):
        ''' Provides the details of the object. Can be adjusted in child classes to include more information. '''
        details_string  =  super().details()
        details_string +=(f'This fish has:'\
                          f'nose_position {self.nose_position} / length {self.length} (unit: meter)\n'\
                          f'yaw {add_prefix(self.yaw,"deg")}, pitch {add_prefix(self.pitch,"deg")}, roll {add_prefix(self.roll,"deg")} (unit: degree)\n'\
                          f'relative bend locations {self.relative_bend_locations_percentage} (unit %), bend lengths {self.bend_lengths} (unit: meter) \n'\
                          f'relative bend angles: lateral {add_prefix(self.relative_bend_angle_lateral,"deg")}, dorso-ventral {add_prefix(self.relative_bend_angle_dorso_ventral,"deg")} (unit: degree)\n'\
                          f'range of point currents from {self.point_currents_range_percentage["start"]} to {self.point_currents_range_percentage["end"]} % of length from nose\n'\
                          f'distribution of receptors {self.receptors_init["method"]}\n')
        return details_string


    def get_nose_position(self):
        return self.nose_position


    def get_length(self):
        return self.length


    def get_yaw(self):
        return self.yaw


    def get_pitch(self):
        return self.pitch


    def get_roll(self):
        return self.roll


    def get_bend_locations(self):
        return self.relative_bend_locations_percentage


    def get_bend_angle_lateral(self):
        return self.relative_bend_angle_lateral


    def get_bend_angle_dorso_ventral(self):
        return self.relative_bend_angle_dorso_ventral


    @classmethod
    def _initialize_input_argument_names(cls):
        inp_args  = super()._initialize_input_argument_names()
        inp_args += ['nose_position=[0,0,0]',
                       'fish_length=1',
                       'angle_yaw=0',
                       'angle_pitch=0',
                       'angle_roll=0',
                       'relative_bend_locations_percentage=np.array([])',
                       'relative_bend_angle_lateral=(np.array([]),"deg")',
                       'relative_bend_angle_dorso_ventral =(np.array([]),"deg")', 
                       'point_currents_range_percentage=dict(start=0, end=100)',
                       'N_point_currents=101',
                       'point_currents_magnitude_scale=1',
                       'receptors_init=dict(method="random", head=100, body=200)',]
        return inp_args


    def run_tests(self):
        ''' Sanity assertion checks to ensure code robustness. '''
        super().run_tests()
        ### assert self.skin_sig        == self.get_skin_conductivity(), 'Fish skin conductivity does not match.'
        assert self.length          == self.get_length(), 'Fish length does not match.'
        assert self.yaw             == self.get_yaw(), 'Fish yaw does not match.'
        assert self.pitch           == self.get_pitch(), 'Fish pitch does not match.'
        assert self.roll            == self.get_roll(), 'Fish roll does not match.'

        assert (self.nose_position  == self.get_nose_position()).all(), 'Fish position does not match.'

        return 'Success!'


