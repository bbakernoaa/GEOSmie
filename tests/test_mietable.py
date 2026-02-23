import numpy as np
import xarray as xr
import pytest
import os
from src.Shared.api.geosmie.mietable import MieTABLE

@pytest.fixture
def dummy_mie_file(tmp_path):
    fn = tmp_path / "dummy_optics.nc"

    rh = np.linspace(0, 1, 11)
    wavelength = [470e-9, 550e-9]
    bin_idx = [1, 2, 3]

    ds = xr.Dataset(
        data_vars={
            "bext": (("bin", "wavelength", "rh"), np.random.rand(3, 2, 11)),
            "ssa": (("bin", "wavelength", "rh"), np.random.rand(3, 2, 11)),
            "g": (("bin", "wavelength", "rh"), np.random.rand(3, 2, 11)),
            "rhop": (("bin", "rh"), np.random.rand(3, 11)),
            "growth_factor": (("bin", "rh"), np.random.rand(3, 11)),
        },
        coords={
            "bin": bin_idx,
            "wavelength": wavelength,
            "rh": rh,
        }
    )
    ds.to_netcdf(fn)
    return str(fn)

def test_mietable_basic(dummy_mie_file):
    mie = MieTABLE(dummy_mie_file)

    q_mass = xr.DataArray(np.ones((5, 5)), dims=("lat", "lon"))
    rh = xr.DataArray(np.full((5, 5), 0.5), dims=("lat", "lon"))

    aot = mie.getAOT(q_mass, rh, bin_idx=2, wavelength=550e-9)

    assert aot.shape == (5, 5)
    assert not np.isnan(aot).any()

def test_mietable_lazy(dummy_mie_file):
    import dask.array as da
    mie = MieTABLE(dummy_mie_file)

    q_mass = xr.DataArray(da.ones((5, 5), chunks=2), dims=("lat", "lon"))
    rh = xr.DataArray(da.from_array(np.full((5, 5), 0.5), chunks=2), dims=("lat", "lon"))

    aot = mie.getAOT(q_mass, rh, bin_idx=2, wavelength=550e-9)

    assert isinstance(aot.data, da.Array)
    res = aot.compute()
    assert res.shape == (5, 5)
