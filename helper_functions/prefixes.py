import numpy as np


def parse_prefix(prefix, direction=1):
    assert type(prefix) == type(''), 'Prefix must be a string of type "str".'
    assert len(prefix) == 1 or prefix == 'deg', 'Prefix must be a single appropriate letter character (or be "deg"): {d,c,m,u,n,p,f,a,z,y, D,h,k,M,G,T,P,E,Z,Y, deg}.'
    if prefix == '':
        return 1
    elif prefix == 'd':
        return (1e-1)**direction
    elif prefix == 'c':
        return (1e-2)**direction
    elif prefix == 'm':
        return (1e-3)**direction
    elif prefix == 'u':
        return (1e-6)**direction
    elif prefix == 'n':
        return (1e-9)**direction
    elif prefix == 'p':
        return (1e-12)**direction
    elif prefix == 'f':
        return (1e-15)**direction
    elif prefix == 'a':
        return (1e-18)**direction
    elif prefix == 'z':
        return (1e-21)**direction
    elif prefix == 'y':
        return (1e-24)**direction
    elif prefix == 'D':
        return (1e1)**direction
    elif prefix == 'h':
        return (1e2)**direction
    elif prefix == 'k':
        return (1e3)**direction
    elif prefix == 'M':
        return (1e6)**direction
    elif prefix == 'G':
        return (1e9)**direction
    elif prefix == 'T':
        return (1e12)**direction
    elif prefix == 'P':
        return (1e15)**direction
    elif prefix == 'E':
        return (1e18)**direction
    elif prefix == 'Z':
        return (1e21)**direction
    elif prefix == 'Y':
        return (1e24)**direction
    elif prefix == 'deg':
        return (np.pi / 180)**direction
    else:
        raise ValueError('Prefix not valid. Must be in {d,c,m,u,n,p,f,a,z,y, D,h,k,M,G,T,P,E,Z,Y, deg}.')


def return_numerical_prefix(prefix):
    assert type(prefix) == type(''), 'Prefix must be a string of type "str".'
    if prefix == '':
        return 1
    else:
        direction = 1 
        if '.' in prefix:
            direction = int(prefix.split('.')[1])
            prefix = prefix.split('.')[0]
        return parse_prefix(prefix, direction)


def remove_prefix(value, prefix):
    ''' Convert a value from a sub-unit to the main SI unit. '''
    return value * return_numerical_prefix(prefix)
    

def add_prefix(value, prefix):
    ''' Convert a value from the main SI unit to the given sub-unit '''
    return value / return_numerical_prefix(prefix)
