import numpy as np
import xarray as xr
import pytest
import dask.array as da
from src.utils.meteo import (
    potential_temperature, virtual_temperature, saturation_vapor_pressure,
    mixing_ratio_from_relative_humidity, specific_humidity_from_mixing_ratio
)

def test_potential_temperature_backends():
    temp_np = np.array([280.0, 300.0])
    pres_np = np.array([100000.0, 50000.0])

    # Eager
    res_eager = potential_temperature(xr.DataArray(temp_np), xr.DataArray(pres_np))

    # Lazy
    temp_da = xr.DataArray(da.from_array(temp_np, chunks=1))
    pres_da = xr.DataArray(da.from_array(pres_np, chunks=1))
    res_lazy = potential_temperature(temp_da, pres_da)

    assert isinstance(res_lazy.data, da.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

def test_meteo_formulas():
    # Test values
    temp = xr.DataArray([300.0]) # ~27 C
    pres = xr.DataArray([100000.0]) # 1000 hPa
    relhum = xr.DataArray([0.5]) # 50%

    es = saturation_vapor_pressure(temp)
    assert es > 3000.0 # Standard value around 27C is ~3500 Pa

    w = mixing_ratio_from_relative_humidity(relhum, temp, pres)
    assert w > 0
    assert w < 0.1

    tv = virtual_temperature(temp, w)
    assert tv > temp

    q = specific_humidity_from_mixing_ratio(w)
    assert q < w
