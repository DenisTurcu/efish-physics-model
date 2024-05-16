import numpy as np
from prefixes import *


def rho24sig(value):
    ''' Converts conductivity <--> resistivity. '''
    assert np.abs(value) > 1e-20, 'Conductivity or resistivity cannot be 0.'
    return 1/value


def convert2mainSI(value):
    ''' Converts values with potentially given prefix to the main SI units. '''
    if type(value) == type(()):
        temp_val = np.array(value[0])
        for i in range(1,len(value)):
            temp_val = remove_prefix(temp_val, value[i])
        return temp_val
    else:
        return np.array(value)*1.
    
