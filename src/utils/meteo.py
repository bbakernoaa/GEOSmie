"""
Standard meteorological calculations for micro-meteorology and atmospheric science.
Supports both Eager (NumPy) and Lazy (Dask) evaluation via Xarray.
"""

import numpy as np
import xarray as xr
from typing import Union
from src.utils.constants import R_UNIVERSAL, G, MW_H2O

# Constants specific to meteorology
P0 = 100000.0  # Reference pressure [Pa]
RD = 287.052874  # Gas constant for dry air [J kg-1 K-1]
CPD = 1004.64  # Specific heat of dry air at constant pressure [J kg-1 K-1]
KAPPA = RD / CPD  # Poisson constant (~0.286)
EPSILON = 0.62198  # Ratio of gas constants (Rd/Rv)

def potential_temperature(temp: xr.DataArray, pres: xr.DataArray) -> xr.DataArray:
    """
    Calculate potential temperature (theta).

    Parameters
    ----------
    temp : xr.DataArray
        Air temperature [K].
    pres : xr.DataArray
        Air pressure [Pa].

    Returns
    -------
    xr.DataArray
        Potential temperature [K].
    """
    theta = temp * (P0 / pres) ** KAPPA
    theta.attrs = {
        "units": "K",
        "long_name": "potential temperature",
        "history": "Calculated via Poisson's equation"
    }
    return theta

def virtual_temperature(temp: xr.DataArray, mixing_ratio: xr.DataArray) -> xr.DataArray:
    """
    Calculate virtual temperature.

    Parameters
    ----------
    temp : xr.DataArray
        Air temperature [K].
    mixing_ratio : xr.DataArray
        Water vapor mixing ratio [kg kg-1].

    Returns
    -------
    xr.DataArray
        Virtual temperature [K].
    """
    v_temp = temp * (1.0 + 0.61 * mixing_ratio)
    v_temp.attrs = {
        "units": "K",
        "long_name": "virtual temperature",
        "history": "Calculated via Tv = T(1 + 0.61q)"
    }
    return v_temp

def saturation_vapor_pressure(temp: xr.DataArray) -> xr.DataArray:
    """
    Calculate saturation vapor pressure over liquid water (Tetens formula).

    Parameters
    ----------
    temp : xr.DataArray
        Air temperature [K].

    Returns
    -------
    xr.DataArray
        Saturation vapor pressure [Pa].
    """
    # temp in Celsius for Tetens
    temp_c = temp - 273.15
    es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    es.attrs = {
        "units": "Pa",
        "long_name": "saturation vapor pressure",
        "history": "Calculated via Tetens formula"
    }
    return es

def mixing_ratio_from_relative_humidity(relhum: xr.DataArray, temp: xr.DataArray, pres: xr.DataArray) -> xr.DataArray:
    """
    Calculate mixing ratio from relative humidity.

    Parameters
    ----------
    relhum : xr.DataArray
        Relative humidity [0-1].
    temp : xr.DataArray
        Air temperature [K].
    pres : xr.DataArray
        Air pressure [Pa].

    Returns
    -------
    xr.DataArray
        Mixing ratio [kg kg-1].
    """
    es = saturation_vapor_pressure(temp)
    e = relhum * es
    w = EPSILON * e / (pres - e)
    w.attrs = {
        "units": "kg kg-1",
        "long_name": "water vapor mixing ratio",
        "history": "Calculated from RH and saturation vapor pressure"
    }
    return w

def specific_humidity_from_mixing_ratio(mixing_ratio: xr.DataArray) -> xr.DataArray:
    """
    Convert mixing ratio to specific humidity.

    Parameters
    ----------
    mixing_ratio : xr.DataArray
        Water vapor mixing ratio [kg kg-1].

    Returns
    -------
    xr.DataArray
        Specific humidity [kg kg-1].
    """
    q = mixing_ratio / (1.0 + mixing_ratio)
    q.attrs = {
        "units": "kg kg-1",
        "long_name": "specific humidity",
        "history": "Calculated from mixing ratio"
    }
    return q
