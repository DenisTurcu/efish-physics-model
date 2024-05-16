import numpy as np

import sys
sys.path.append('../helper_functions/')

from prefixes import add_prefix, remove_prefix, return_numerical_prefix, parse_prefix
from conversions import rho24sig, convert2mainSI
from ElectricObject import ElectricObject


class Worm(ElectricObject):
    ''' Represents a foreign object to the fish, which is closeby. Parent class "ElectricObject": '''
    __doc__ += ElectricObject.__doc__

    def __init__(self, center_position=[0,0,0], _init_tests=True, **kwds):
        super().__init__(**kwds, _init_tests=False)
        self.r_vec = np.array(convert2mainSI(center_position))
        self._initialize_input_argument_names()
        # Run assertion tests
        if _init_tests:
            print(self.run_tests())


    def is_equal(self, other, simple_return=True):
        ''' Compare current worm to another worm for equality. '''
        _, thruth_values, comparison  = super().is_equal(other, simple_return=False)
        if isinstance(other, self.__class__):
            thruth_values.append((np.abs(self.r_vec - other.get_position()) < np.min([self.assert_err, other.assert_err])).all())
            comparison.append('center_position')
        thruth_value = False if len(thruth_values) == 0 else np.array(thruth_values).all()
        if simple_return:
            return thruth_value 
        else: 
            return thruth_value, thruth_values, comparison


    def create_graph_obj(self):
        raise NotImplementedError('Need to implement the "create_graph_obj" function according to the geometry of the specified object. ')


    def details(self):
        ''' Provides the details of the worm. Can be adjusted in child classes to include more information. '''
        details_string  =  super().details()
        details_string += f'This worm has center position {self.r_vec}.\n'
        return details_string


    def get_position(self):
        ''' Extract the worm's position [x, y, z]. '''
        return self.r_vec


    def get_points(self):
        ''' Extract the worm's surface as sampled points. Point worm returns its center position. 
        Large worms must return many points from their surface, and this method must be overwritten accordingly. '''
        return self.get_position()


    @classmethod
    def _initialize_input_argument_names(cls):
        inp_args  = super()._initialize_input_argument_names()
        inp_args += ['center_position']
        return inp_args


    def run_tests(self):
        super().run_tests()
        assert (self.r_vec == self.get_position()).all(), 'Worm position does not match.'
        return 'Success!'


