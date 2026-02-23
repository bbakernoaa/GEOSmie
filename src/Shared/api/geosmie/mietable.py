"""
Implements API to access Mie LUTs for a single species.
Supports both Eager (NumPy) and Lazy (Dask) evaluation via Xarray.
"""

import numpy as np
import xarray as xr
from typing import List, Optional, Union, Tuple

__VERSION__ = "0.9.1"

class MieTABLE:
    """
    Interface for Mie look-up tables stored in NetCDF format.
    """

    def __init__(self, filename: str, wavelengths: Optional[List[float]] = None):
        """
        Initialize MieTABLE by loading a NetCDF file.

        Parameters
        ----------
        filename : str
            Path to the Mie Table NetCDF file.
        wavelengths : list of float, optional
            Desired wavelengths [m]. If omitted, all wavelengths in the file are used.
        """
        self.ds = xr.open_dataset(filename)

        # Standardize dimension names
        rename_dict = {}
        if "radius" in self.ds.dims:
            rename_dict["radius"] = "bin"
        if "lambda" in self.ds.dims:
            rename_dict["lambda"] = "wavelength"
        if "nPol" in self.ds.dims:
            rename_dict["nPol"] = "p"

        if rename_dict:
            self.ds = self.ds.rename(rename_dict)

        if wavelengths is not None:
            self.ds = self.ds.sel(wavelength=wavelengths, method="nearest")

        # Expose coordinates as attributes for convenience
        self.wavelengths = self.ds.wavelength
        self.rh = self.ds.rh
        self.bins = self.ds.bin

    def _interpolate(self, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.Dataset:
        """
        Perform interpolation on RH and selection on bin/wavelength.
        """
        # Select bin (bins are 1-indexed in the file, but let's be careful)
        # The 'bin' coordinate in the file is [1, 2, 3, ...]
        subset = self.ds.sel(bin=bin_idx)

        if wavelength is not None:
            subset = subset.sel(wavelength=wavelength, method="nearest")

        # Interpolate RH. rh input should be [0, 1].
        # Clip RH to the range available in the table to avoid NaNs.
        rh_min = float(self.rh.min())
        rh_max = float(self.rh.max())
        rh_clipped = rh.clip(rh_min, rh_max)

        return subset.interp(rh=rh_clipped)

    def getBEXT(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.DataArray:
        """Calculate mass extinction: bext * q_mass"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        return interp_ds.bext * q_mass

    def getBSCA(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.DataArray:
        """Calculate mass scattering: bsca * q_mass"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        return interp_ds.bsca * q_mass

    def getAOT(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.DataArray:
        """AOT is the same as getBEXT (integrated over layers, but here we return profile)"""
        return self.getBEXT(q_mass, rh, bin_idx, wavelength)

    def getSSA(self, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.DataArray:
        """Get Single Scattering Albedo"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        return interp_ds.ssa

    def getGASYM(self, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.DataArray:
        """Get Asymmetry Parameter"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        return interp_ds.g

    def getScalar(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Get AOT, SSA, and GASYM simultaneously"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        aot = interp_ds.bext * q_mass
        return aot, interp_ds.ssa, interp_ds.g

    def getREFF(self, rh: xr.DataArray, bin_idx: int) -> xr.DataArray:
        """Get Effective Radius"""
        interp_ds = self._interpolate(rh, bin_idx)
        return interp_ds.rEff

    def getGF(self, rh: xr.DataArray, bin_idx: int) -> xr.DataArray:
        """Get Growth Factor"""
        interp_ds = self._interpolate(rh, bin_idx)
        return interp_ds.growth_factor

    def getRHOP(self, rh: xr.DataArray, bin_idx: int) -> xr.DataArray:
        """Get Wet Particle Density"""
        interp_ds = self._interpolate(rh, bin_idx)
        return interp_ds.rhop

    def getRHOD(self, bin_idx: int) -> xr.DataArray:
        """Get Dry Particle Density (at rh=0)"""
        return self.ds.rhop.sel(bin=bin_idx, rh=0, method="nearest")

    def getBBCK(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.DataArray:
        """Get Mass Backscatter: bbck * q_mass"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        return interp_ds.bbck * q_mass

    def getPMOM(self, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> xr.DataArray:
        """Get Phase Function moments"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        return interp_ds.pmom

    def getVOLUME(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int) -> xr.DataArray:
        """Get Total Volume: volume * q_mass"""
        interp_ds = self._interpolate(rh, bin_idx)
        return interp_ds.volume * q_mass

    def getAREA(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int) -> xr.DataArray:
        """Get Total Cross Section: area * q_mass"""
        interp_ds = self._interpolate(rh, bin_idx)
        return interp_ds.area * q_mass

    def getRefIndex(self, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> Tuple[xr.DataArray, xr.DataArray]:
        """Get Real and Imaginary Refractive Indices"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        return interp_ds.refreal, interp_ds.refimag

    def getVector(self, q_mass: xr.DataArray, rh: xr.DataArray, bin_idx: int, wavelength: Optional[float] = None) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Get AOT, SSA, and Phase Function moments (PMOM)"""
        interp_ds = self._interpolate(rh, bin_idx, wavelength)
        aot = interp_ds.bext * q_mass
        # pmom might not be in all files, handle gracefully
        pmom = interp_ds.get("pmom", None)
        return aot, interp_ds.ssa, pmom
