import numpy as np
from prefixes import remove_prefix


def rho24sig(value: float | np.ndarray) -> float | np.ndarray:
    """Converts conductivity <--> resistivity. "value must be provided in SI units."
    "24": "to and from"

    Args:
        value (float | np.ndarray): Provided value(s) to be converted.

    Returns:
        float | np.ndarray: Inverse of the provided value(s).
    """
    """ Converts conductivity <--> resistivity. """
    assert np.abs(value) > 1e-20, "Conductivity or resistivity cannot be 0."
    return 1 / value


def convert2mainSI(value: tuple | float | np.ndarray) -> float | np.ndarray:
    """Convert a given value to the main SI units. If a tuple is provided, the first element is
    the value and the rest are prefixes.

    Args:
        value (tuple | float | np.ndarray): Value to be converted. If a tuple is provided, the first element
            is the value and the rest are prefixes.

    Returns:
        float | np.ndarray: Converted value.
    """
    """Converts values with potentially given prefix to the main SI units."""
    if type(value) is type(()):
        temp_val = np.array(value[0])  # type: ignore
        for i in range(1, len(value)):  # type: ignore
            temp_val = remove_prefix(temp_val, value[i])  # type: ignore
        return temp_val
    else:
        return np.array(value) * 1.0
