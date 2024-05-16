import numpy as np


def parse_prefix(prefix: str, direction: int = 1) -> float:
    """Parse the prefix string to a numerical value to convert between SI prefixes.

    Args:
        prefix (str): "prefix" must be one of {d,c,m,u,n,p,f,a,z,y, D,h,k,M,G,T,P,E,Z,Y, deg} and represents common
            SI prefixes. For example, "m" represents milli, "k" represents kilo, and "deg" represents degrees.
        direction (int, optional): The "direction" in which to perform the conversion. For example "prefix='m'" and
            "direction=1" will convert from milli(grams/meters/etc.) to (grams/meters/etc.), but "direction=-1"
            will reverse the conversion direction. Defaults to 1.

    Raises:
        ValueError: If "prefix" is poorly provided and not in {d,c,m,u,n,p,f,a,z,y, D,h,k,M,G,T,P,E,Z,Y, deg}.

    Returns:
        float: The numerical value needed for conversion based on the prefix.
    """
    if prefix == "":
        return 1
    elif prefix == "d":
        return (1e-1) ** direction
    elif prefix == "c":
        return (1e-2) ** direction
    elif prefix == "m":
        return (1e-3) ** direction
    elif prefix == "u":
        return (1e-6) ** direction
    elif prefix == "n":
        return (1e-9) ** direction
    elif prefix == "p":
        return (1e-12) ** direction
    elif prefix == "f":
        return (1e-15) ** direction
    elif prefix == "a":
        return (1e-18) ** direction
    elif prefix == "z":
        return (1e-21) ** direction
    elif prefix == "y":
        return (1e-24) ** direction
    elif prefix == "D":
        return (1e1) ** direction
    elif prefix == "h":
        return (1e2) ** direction
    elif prefix == "k":
        return (1e3) ** direction
    elif prefix == "M":
        return (1e6) ** direction
    elif prefix == "G":
        return (1e9) ** direction
    elif prefix == "T":
        return (1e12) ** direction
    elif prefix == "P":
        return (1e15) ** direction
    elif prefix == "E":
        return (1e18) ** direction
    elif prefix == "Z":
        return (1e21) ** direction
    elif prefix == "Y":
        return (1e24) ** direction
    elif prefix == "deg":
        return (np.pi / 180) ** direction
    else:
        raise ValueError("Prefix not valid. Must be in {d,c,m,u,n,p,f,a,z,y, D,h,k,M,G,T,P,E,Z,Y, deg}.")


def return_numerical_prefix(prefix: str) -> float:
    """Return the numerical value of the SI prefix, parsing the embedded direction as well, if provided.

    Args:
        prefix (str): "prefix" must be one of {d,c,m,u,n,p,f,a,z,y, D,h,k,M,G,T,P,E,Z,Y, deg} and represents common
            SI prefixes. For example, "m" represents milli, "k" represents kilo, and "deg" represents degrees.
            Additionally, if a direction is embedded, it will be parsed and used to determine the full conversion.
            The "direction" should be embedded as a decimal point followed by an integer. For example, "c.1" will
            convert from centi to the main SI unit, and "c.-1" will convert from the main SI unit to centi.

    Returns:
        float: The numerical value needed for conversion based on the prefix (with embedded direction, if provided).
    """
    if prefix == "":
        return 1
    else:
        direction = 1
        if "." in prefix:
            direction = int(prefix.split(".")[1])
            prefix = prefix.split(".")[0]
        return parse_prefix(prefix, direction)


def remove_prefix(value: int | float | np.ndarray, prefix: str) -> float | np.ndarray:
    """Convert a value from a sub-unit to the main SI unit. This is opposite of "add_prefix".

    Args:
        value (int | float | np.ndarray): The value to be converted.
        prefix (str): Provided as per instructions in "return_numerical_prefix".

    Returns:
        float | np.ndarray: Converted value.
    """
    return value * return_numerical_prefix(prefix)


def add_prefix(value: int | float | np.ndarray, prefix: str) -> float | np.ndarray:
    """Convert a value from the main SI unit to the given prefixed-unit"

    Args:
        value (int | float | np.ndarray): The value to be converted.
        prefix (str): Provided as per instructions in "return_numerical_prefix".

    Returns:
        float | np.ndarray: Converted value.
    """
    return value / return_numerical_prefix(prefix)
