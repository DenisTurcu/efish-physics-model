import numpy as np 
import sys
sys.path.append('../helper_functions/')

from conversions import rho24sig, convert2mainSI


class ElectricObject():
    ''' 
    Represents an electric object which has limited functionality.
    Call function "get_input_argument_names()" to check the inputs to this class.
    '''

    def __init__(self, conductivity=1, 
                       relative_permitivity=1,
                       assert_err=1e-12,
                       _init_tests=True):
        self.sig   = convert2mainSI(conductivity)
        self.eps_r = relative_permitivity
        self.eps0  = 8.854187812813 * 1e-12
        self.assert_err = assert_err
        self.arguments = self._initialize_input_argument_names()
        # Run assertion tests
        if _init_tests:
            print(self.run_tests())


    def __str__(self):
        ''' Provides convenience of using the print function to show details of object. '''
        return self.details()


    def details(self):
        ''' Provides the details of the object. Can be adjusted in child classes to include more information. '''
        details_string = f'This electric object has conductivity  {self.sig} and relative permitivity {self.eps_r}.\nAssertion allowed error is {self.assert_err}.\n'
        return details_string


    def is_equal(self, other, simple_return=True):
        ''' Compare two objects. Needs to be updated/implemented in child classes. '''
        thruth_values  = []
        comparison     = []
        if isinstance(other, self.__class__):
            thruth_values.append(np.abs((self.sig             - other.get_conductivity()))         < np.min([self.assert_err, other.assert_err]))
            thruth_values.append(np.abs((self.eps_r           - other.get_relative_permitivity())) < np.min([self.assert_err, other.assert_err]))
            comparison.append('conductivity')
            comparison.append('relative_permitivity')
        thruth_value = False if len(thruth_values) == 0 else np.array(thruth_values).all()
        if simple_return:
            return thruth_value 
        else: 
            return thruth_value, thruth_values, comparison


    def get_conductivity(self):
        ''' Extract the conductivity, sigma. '''
        return self.sig


    def get_resistivity(self):
        ''' Extract the resistivity, rho. '''
        return rho24sig(self.sig)


    def get_relative_permitivity(self):
        ''' Extract the relative permitivity eps_relative. '''
        return self.eps_r

    @classmethod
    def _initialize_input_argument_names(cls):
        ''' Keep track of the required input arguments required. '''
        return ['conductivity=1', 'relative_permitivity=1']


    def get_input_argument_names(self, print_arguments=True):
        ''' Provides the names of the input arguments required to create a new object. '''
        if print_arguments:
            print('Input arguments:')
            for argument in self.arguments:
                print(argument)
        return self.arguments


    def run_tests(self):
        assert self.sig   == self.get_conductivity(), 'ElectricObject conductivity does not match.'
        assert self.eps_r == self.get_relative_permitivity(), 'ElectricObject permitivity does not match.'
        return 'Success!'

