"""
Particle Size Distribution (PSD) utilities.
Supports backend-agnostic calculations for size bins and distributions.
"""

import numpy as np
import xarray as xr
from typing import Tuple, Union
from src.utils.constants import FOUR_THIRDS_PI

def carma_bins(
    nbin: int,
    rmrat: float,
    rmin: float,
    rhop: Union[xr.DataArray, float] = 1.0
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Generate CARMA-like radius bins.

    Parameters
    ----------
    nbin : int
        Number of size bins.
    rmrat : float
        Ratio of volume (mass) between size bins.
    rmin : float
        Radius of the smallest bin.
    rhop : xr.DataArray or float
        Particle density.

    Returns
    -------
    r : xr.DataArray
        Median radius of the bin.
    dr : xr.DataArray
        Width of the bin (rup - rlow).
    rlow : xr.DataArray
        Lower edge radius.
    rup : xr.DataArray
        Upper edge radius.
    rmassup : xr.DataArray
        Mass at the upper edge.
    """
    rmassmin = FOUR_THIRDS_PI * rhop * rmin**3
    vrfact = ((3.0 / 2.0 / np.pi / (rmrat + 1))**(1.0 / 3.0)) * (rmrat**(1.0 / 3.0) - 1.0)

    # Vectorized bin generation
    bins = np.arange(nbin)
    rmass = rmassmin * (rmrat**bins)

    rmassup = 2.0 * rmrat / (rmrat + 1.0) * rmass
    r = (rmass / rhop / FOUR_THIRDS_PI)**(1.0 / 3.0)
    rup = (rmassup / rhop / FOUR_THIRDS_PI)**(1.0 / 3.0)
    dr = vrfact * (rmass / rhop)**(1.0 / 3.0)
    rlow = rup - dr

    # Convert to DataArray if they aren't already
    return (
        xr.DataArray(r, coords={"bin": bins}, dims="bin", name="r"),
        xr.DataArray(dr, coords={"bin": bins}, dims="bin", name="dr"),
        xr.DataArray(rlow, coords={"bin": bins}, dims="bin", name="rlow"),
        xr.DataArray(rup, coords={"bin": bins}, dims="bin", name="rup"),
        xr.DataArray(rmassup, coords={"bin": bins}, dims="bin", name="rmassup")
    )

def find_rmin(nbin: int, rlow: float, rup: float) -> Tuple[float, float]:
    """
    Find rmin and rmrat for a desired size range.

    Parameters
    ----------
    nbin : int
        Number of bins.
    rlow : float
        Desired lower edge of first bin.
    rup : float
        Desired upper edge of last bin.

    Returns
    -------
    rmin : float
    rmrat : float
    """
    rmrat = (rup**3 / rlow**3)**(1.0 / nbin)
    vrfact = ((3.0 / 2.0 / np.pi / (rmrat + 1))**(1.0 / 3.0)) * (rmrat**(1.0 / 3.0) - 1.0)
    f = 2.0 * rmrat / (rmrat + 1.0)
    rmin = 1.0 / (f**(1.0 / 3.0) - (FOUR_THIRDS_PI)**(1.0 / 3.0) * vrfact) * rlow
    return float(rmin), float(rmrat)

def lognormal_dndr(
    r: xr.DataArray,
    rm: Union[xr.DataArray, float],
    sigma: Union[xr.DataArray, float],
    n_total: Union[xr.DataArray, float] = 1.0
) -> xr.DataArray:
    """
    Evaluate a lognormal distribution (dN/dr).

    Parameters
    ----------
    r : xr.DataArray
        Radii to evaluate at.
    rm : xr.DataArray or float
        Median radius.
    sigma : xr.DataArray or float
        Geometric standard deviation.
    n_total : xr.DataArray or float
        Total number concentration.

    Returns
    -------
    xr.DataArray
        dN/dr at given radii.
    """
    log_sigma = np.log(sigma)
    term1 = n_total / (r * log_sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * (np.log(r / rm) / log_sigma)**2)

    dn_dr = term1 * term2
    dn_dr.attrs = {"units": "m-4", "long_name": "Number size distribution"}
    return dn_dr

def lognormal_dndlnr(
    r: xr.DataArray,
    rm: Union[xr.DataArray, float],
    sigma: Union[xr.DataArray, float],
    n_total: Union[xr.DataArray, float] = 1.0
) -> xr.DataArray:
    """
    Evaluate a lognormal distribution (dN/dlnr).

    Parameters
    ----------
    r : xr.DataArray
        Radii to evaluate at.
    rm : xr.DataArray or float
        Median radius.
    sigma : xr.DataArray or float
        Geometric standard deviation.
    n_total : xr.DataArray or float
        Total number concentration.

    Returns
    -------
    xr.DataArray
        dN/dlnr at given radii.
    """
    log_sigma = np.log(sigma)
    term1 = n_total / (log_sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * (np.log(r / rm) / log_sigma)**2)

    dn_dlnr = term1 * term2
    dn_dlnr.attrs = {"units": "1", "long_name": "Log-normal number distribution"}
    return dn_dlnr
