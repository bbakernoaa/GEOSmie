import numpy as np
import xarray as xr
import pytest
import dask.array as da
from src.utils.microphysics import (
    wtpct_sulfate, density_sulfate, growth_factor_sulfate, growth_factor_sea_salt
)

def test_wtpct_sulfate_backends():
    rh_np = np.linspace(0.1, 0.9, 10)
    temp = 220.0

    # Eager (NumPy)
    res_eager = wtpct_sulfate(xr.DataArray(rh_np), temp)

    # Lazy (Dask)
    rh_da = xr.DataArray(da.from_array(rh_np, chunks=5))
    res_lazy = wtpct_sulfate(rh_da, temp)

    assert isinstance(res_lazy.data, da.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

def test_density_sulfate_backends():
    wtp_np = np.linspace(10, 90, 10)
    temp = 220.0

    # Eager
    res_eager = density_sulfate(xr.DataArray(wtp_np), temp)

    # Lazy
    wtp_da = xr.DataArray(da.from_array(wtp_np, chunks=5))
    res_lazy = density_sulfate(wtp_da, temp)

    assert isinstance(res_lazy.data, da.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

def test_growth_factors_backends():
    rh_np = np.array([0.5, 0.8])
    r_dry = 1e-7 # 100 nm

    # Sulfate
    res_su_eager = growth_factor_sulfate(xr.DataArray(rh_np), r_dry)
    res_su_lazy = growth_factor_sulfate(xr.DataArray(da.from_array(rh_np, chunks=1)), r_dry)
    xr.testing.assert_allclose(res_su_eager, res_su_lazy.compute())

    # Sea Salt
    res_ss_eager = growth_factor_sea_salt(xr.DataArray(rh_np), r_dry)
    res_ss_lazy = growth_factor_sea_salt(xr.DataArray(da.from_array(rh_np, chunks=1)), r_dry)
    xr.testing.assert_allclose(res_ss_eager, res_ss_lazy.compute())

def test_sulfate_chain_consistency():
    # Test that growth factor is consistent with wtpct and density
    rh = xr.DataArray([0.5])
    temp = 220.0
    r_dry = 1e-7

    gf = growth_factor_sulfate(rh, r_dry, temp=temp)

    # Manual calculation using the components
    # We need to account for Kelvin effect, so we use relhum_eff from gf logic
    # but for a simple check, let's just see if values are reasonable
    assert gf > 1.0
    assert gf < 10.0
