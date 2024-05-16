import numpy as np

from conversions import rho24sig, convert2mainSI
from prefixes import remove_prefix, add_prefix


def run_tests(N: int = 10, err: float = 1e-12) -> str:
    """Run tests for the helper functions in the "conversions" and "prefixes" modules.

    Args:
        N (int, optional): Number of test trials. Defaults to 10.
        err (float, optional): Error threshold for deciding what counts as "0" error. Defaults to 1e-12.

    Returns:
        str: "Success!" if all tests pass. Assertion error otherwise.
    """
    for test_val in np.random.randn(N):
        assert (
            np.abs(remove_prefix(test_val, "d") - test_val / 10) < err
        ), f'"remove_prefix" function fails on "d" example for value {test_val}'
        assert (
            np.abs(add_prefix(test_val, "d") - test_val * 10) < err
        ), f'"add_prefix" function fails on "d" example for value {test_val}'

        assert (
            np.abs(add_prefix(test_val, "c") - test_val * 100) < err
        ), f'"add_prefix" function fails on "c" example for value {test_val}'

        if np.abs(test_val) > err:
            assert np.abs(rho24sig(test_val) - 1 / test_val) < err, f'"rho24sig" fails on {test_val}'
            assert (
                np.abs(rho24sig(np.array(test_val)) - 1 / test_val) < err
            ), f'"rho24sig" fails on "np.array({test_val})"'
            assert (
                np.abs(rho24sig(np.array([test_val])) - 1 / test_val) < err
            ), f'"rho24sig" fails on "np.array([{test_val}])"'

        assert np.abs(convert2mainSI(test_val) - test_val) < err, f'"convert2mainSI" fails on {test_val}'
        assert (
            np.abs(convert2mainSI((test_val, "d")) - test_val / 10) < err
        ), f'"convert2mainSI" fails on ({test_val},"d")'
        assert (
            np.abs(convert2mainSI((test_val, "c")) - test_val / 100) < err
        ), f'"convert2mainSI" fails on ({test_val},"c")'
        assert (
            np.abs(convert2mainSI((test_val, "u", "c.-1")) - test_val / 10000) < err
        ), f'"convert2mainSI" fails on ({test_val},("u","c.-"))'
        assert (
            np.abs(convert2mainSI((test_val, "k", "c.2")) - test_val / 10) < err
        ), f'"convert2mainSI" fails on ({test_val},("k","c.2"))'

    assert (
        np.abs(remove_prefix(90, "deg") - np.pi / 2) < err
    ), '"remove_prefix" function fails on degree <-> rad conversion.'
    assert np.abs(add_prefix(np.pi / 4, "deg") - 45) < err, '"add_prefix" function fails on degree <-> rad conversion.'
    return "Success!"
