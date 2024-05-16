import numpy as np

from conversions import rho24sig, convert2mainSI
from prefixes import remove_prefix, add_prefix


def run_tests(N=10,err=1e-12):
    for i in np.random.randn(N):
        assert np.abs(remove_prefix(i,'d')-i/10)<err, f'"remove_prefix" function fails on "d" example for value {i}'
        assert np.abs(   add_prefix(i,'d')-i*10)<err, f'"add_prefix" function fails on "d" example for value {i}'

        assert np.abs(add_prefix(i,'c')-i*100)<err, f'"add_prefix" function fails on "c" example for value {i}'

        if np.abs(i)>err:
            assert np.abs(rho24sig(i)-1/i)<err, f'"rho24sig" fails on {i}'
            assert np.abs(rho24sig(np.array(i))-1/i)<err, f'"rho24sig" fails on "np.array({i})"'
            assert np.abs(rho24sig(np.array([i]))-1/i)<err, f'"rho24sig" fails on "np.array([{i}])"'

        assert np.abs(convert2mainSI(i)-i)<err,                     f'"convert2mainSI" fails on {i}'
        assert np.abs(convert2mainSI((i,'d'))-i/10)<err,            f'"convert2mainSI" fails on ({i},"d")'
        assert np.abs(convert2mainSI((i,'c'))-i/100)<err,           f'"convert2mainSI" fails on ({i},"c")'
        assert np.abs(convert2mainSI((i,'u','c.-1'))-i/10000)<err,   f'"convert2mainSI" fails on ({i},("u","c.-"))'
        assert np.abs(convert2mainSI((i,'k','c.2'))-i/10)<err,      f'"convert2mainSI" fails on ({i},("k","c.2"))'

    assert np.abs(remove_prefix(90,'deg')-np.pi/2) < err, '"remove_prefix" function fails on degree <-> rad conversion.'
    assert np.abs(add_prefix(np.pi/4,'deg')-45) < err,    '"add_prefix" function fails on degree <-> rad conversion.'
    return 'Success!'

