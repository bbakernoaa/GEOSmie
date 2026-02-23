"""
Microphysical calculations for aerosols and cloud droplets.
Supports both Eager (NumPy) and Lazy (Dask) evaluation via Xarray.
"""

import numpy as np
import xarray as xr
from typing import Union
from src.utils.constants import (
    AVOGADRO, K_BOLTZMANN, MW_H2O, MW_H2SO4, R_UNIVERSAL, R_VAPOR,
    L_V, T_TRIPLE, E_STR, RHO_H2SO4_DRY
)

def wtpct_sulfate(relhum: xr.DataArray, temp: Union[xr.DataArray, float] = 220.0) -> xr.DataArray:
    """
    Calculate the weight percentage of sulfuric acid in an aqueous solution.
    Based on Tabazadeh et al. (1994), valid for 185 K <= T <= 260 K.

    Parameters
    ----------
    relhum : xr.DataArray
        Relative humidity in the range [0, 1].
    temp : xr.DataArray or float
        Temperature in Kelvin.

    Returns
    -------
    xr.DataArray
        Weight percentage of H2SO4 [1-100].
    """
    # Ensure input is DataArray for attribute handling
    if not isinstance(relhum, xr.DataArray):
        relhum = xr.DataArray(relhum)

    activ = relhum.clip(1e-6, 1.0)

    # Coefficients for different RH ranges
    # Range 1: activ < 0.05
    at1_1, bt1_1, ct1_1, dt1_1 = 12.37208932, -0.16125516114, -30.490657554, -2.1133114241
    at2_1, bt2_1, ct2_1, dt2_1 = 13.455394705, -0.1921312255, -34.285174607, -1.7620073078

    # Range 2: 0.05 <= activ <= 0.85
    at1_2, bt1_2, ct1_2, dt1_2 = 11.820654354, -0.20786404244, -4.807306373, -5.1727540348
    at2_2, bt2_2, ct2_2, dt2_2 = 12.891938068, -0.23233847708, -6.4261237757, -4.9005471319

    # Range 3: activ > 0.85
    at1_3, bt1_3, ct1_3, dt1_3 = -180.06541028, -0.38601102592, -93.317846778, 273.88132245
    at2_3, bt2_3, ct2_3, dt2_3 = -176.95814097, -0.36257048154, -90.469744201, 267.45509988

    def calc_cont(at, bt, ct, dt, a):
        return at * (a**bt) + ct * a + dt

    # Range 1
    contl1 = calc_cont(at1_1, bt1_1, ct1_1, dt1_1, activ)
    conth1 = calc_cont(at2_1, bt2_1, ct2_1, dt2_1, activ)

    # Range 2
    contl2 = calc_cont(at1_2, bt1_2, ct1_2, dt1_2, activ)
    conth2 = calc_cont(at2_2, bt2_2, ct2_2, dt2_2, activ)

    # Range 3
    contl3 = calc_cont(at1_3, bt1_3, ct1_3, dt1_3, activ)
    conth3 = calc_cont(at2_3, bt2_3, ct2_3, dt2_3, activ)

    contl = xr.where(activ < 0.05, contl1, xr.where(activ <= 0.85, contl2, contl3))
    conth = xr.where(activ < 0.05, conth1, xr.where(activ <= 0.85, conth2, conth3))

    contt = contl + (conth - contl) * ((temp - 190.0) / 70.0)
    conwtp = (contt * 98.0) + 1000.0

    wtpct = (100.0 * contt * 98.0) / conwtp
    wtpct = wtpct.clip(1.0, 100.0)

    wtpct.attrs = {
        "units": "percent",
        "long_name": "sulfuric acid weight percentage",
        "history": "Calculated via Tabazadeh et al. (1994)"
    }
    return wtpct

def density_sulfate(wtp: xr.DataArray, temp: Union[xr.DataArray, float] = 220.0) -> xr.DataArray:
    """
    Calculate the density of aqueous sulfuric acid solution.
    Piecewise linear interpolation based on CARMA data.

    Parameters
    ----------
    wtp : xr.DataArray
        Weight percentage of H2SO4 [0, 100].
    temp : xr.DataArray or float
        Temperature in Kelvin.

    Returns
    -------
    xr.DataArray
        Density of solution [g cm-3].
    """
    dnwtp = np.array([0., 1., 5., 10., 20., 25., 30., 35., 40., 41., 45., 50., 53., 55., 56.,
                      60., 65., 66., 70., 72., 73., 74., 75., 76., 78., 79., 80., 81., 82.,
                      83., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96.,
                      97., 98., 100.])

    dnc0 = np.array([1., 1.13185, 1.17171, 1.22164, 1.3219, 1.37209, 1.42185, 1.4705, 1.51767,
                     1.52731, 1.56584, 1.61834, 1.65191, 1.6752, 1.68708, 1.7356, 1.7997,
                     1.81271, 1.86696, 1.89491, 1.9092, 1.92395, 1.93904, 1.95438, 1.98574,
                     2.00151, 2.01703, 2.03234, 2.04716, 2.06082, 2.07363, 2.08461, 2.09386,
                     2.10143, 2.10764, 2.11283, 2.11671, 2.11938, 2.12125, 2.1219, 2.12723,
                     2.12654, 2.12621, 2.12561, 2.12494, 2.12093])

    dnc1 = np.array([0., -0.000435022, -0.000479481, -0.000531558, -0.000622448, -0.000660866,
                     -0.000693492, -0.000718251, -0.000732869, -0.000735755, -0.000744294,
                     -0.000761493, -0.000774238, -0.00078392, -0.000788939, -0.00080946,
                     -0.000839848, -0.000845825, -0.000874337, -0.000890074, -0.00089873,
                     -0.000908778, -0.000920012, -0.000932184, -0.000959514, -0.000974043,
                     -0.000988264, -0.00100258, -0.00101634, -0.00102762, -0.00103757,
                     -0.00104337, -0.00104563, -0.00104458, -0.00104144, -0.00103719,
                     -0.00103089, -0.00102262, -0.00101355, -0.00100249, -0.00100934,
                     -0.000998299, -0.000990961, -0.000985845, -0.000984529, -0.000989315])

    # We want to interpolate den = dnc0 + dnc1 * temp
    # This is slightly tricky because dnc0 and dnc1 are both functions of wtp.

    def interp_vec(v_wtp):
        c0 = np.interp(v_wtp, dnwtp, dnc0)
        c1 = np.interp(v_wtp, dnwtp, dnc1)
        return c0 + c1 * temp

    # Use xarray.apply_ufunc for dask support
    rho = xr.apply_ufunc(
        interp_vec,
        wtp,
        input_core_dims=[[]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float]
    )

    rho.attrs = {
        "units": "g cm-3",
        "long_name": "sulfuric acid solution density",
        "history": "Interpolated from CARMA density tables"
    }
    return rho

def growth_factor_sulfate(relhum: xr.DataArray, r_dry: Union[xr.DataArray, float], temp: Union[xr.DataArray, float] = 220.0) -> xr.DataArray:
    """
    Calculate the hygroscopic growth factor for sulfate particles,
    accounting for the Kelvin effect.

    Parameters
    ----------
    relhum : xr.DataArray
        Relative humidity in the range [0, 1].
    r_dry : xr.DataArray or float
        Dry particle radius in meters.
    temp : xr.DataArray or float
        Temperature in Kelvin.

    Returns
    -------
    xr.DataArray
        Growth factor (r_wet / r_dry).
    """
    # r_dry in cm for internal consistency with legacy code if needed,
    # but we'll try to stay in SI as much as possible.
    # Legacy code uses g cm-3 and cm.

    # Saturation vapor pressure (Legacy constants)
    llv_legacy = 2.501e6
    rv_legacy = 461.0
    ttr_legacy = 273.16
    estr_legacy = 611.0
    es = estr_legacy * np.exp(llv_legacy / rv_legacy * (1.0 / ttr_legacy - 1.0 / temp))

    # Mass concentration of water (vapor) using legacy constants
    k_legacy = 1.38e-23
    navogad_legacy = 6.022e23
    mw_h2o_legacy = 0.018
    n_v = relhum * es / (k_legacy * temp)
    h2o_mass_g_cm3 = n_v / navogad_legacy * mw_h2o_legacy * 1000.0 / 1e6

    # Kelvin effect iteration (Simplified from legacy grow_v75)
    # Start with an assumption of 80 wt % H2SO4
    wtp_init = 80.0
    den1 = 2.00151 - 0.000974043 * temp  # density at 79 wt %
    den2 = 2.01703 - 0.000988264 * temp  # density at 80 wt %
    drho_dwt = den2 - den1

    sig1 = 79.3556 - 0.0267212 * temp    # surface tension at 79.432 wt %
    sig2 = 75.608  - 0.0269204 * temp    # surface tension at 85.9195 wt %
    dsigma_dwt = (sig2 - sig1) / (85.9195 - 79.432)
    sigkelv = sig1 + dsigma_dwt * (80.0 - 79.432)

    # Conversion to meters for Kelvin term
    # R_UNIVERSAL is J mol-1 K-1.
    # mw_h2so4 is kg mol-1.
    # sigkelv is dyne cm-1 = 1e-3 N m-1
    sig_si = sigkelv * 1e-3
    den_si = den2 * 1000.0 # g cm-3 to kg m-3

    # Kelvin factor for water
    # Legacy uses mw_h2so4 (98 g/mol) instead of mw_h2o for Kelvin effect of water
    # and uses den1 (density at 79 wt%) instead of den2 (80 wt%) in the denominator.
    # We follow this exactly to match upstream results.

    # Dry density used in legacy for sulfate growth factor calculation
    rhopdry_legacy = 1.923

    r_dry_cm = r_dry * 100.0
    r_wet_init_cm = r_dry_cm * ( (100.0 * rhopdry_legacy) / (wtp_init * den2) )**(1.0 / 3.0)

    # Kelvin terms using legacy constants and units (CGS-like)
    mw_h2so4_legacy = 98.0
    rgas_legacy = 8.31447e7

    r_kelvin_h2o_a = (2.0 * mw_h2so4_legacy * sigkelv) / (den1 * rgas_legacy * temp * r_wet_init_cm)

    # Legacy 'b' term
    r_kelvin_h2o_b = 1.0 + wtp_init * drho_dwt / den2 - 3.0 * wtp_init * dsigma_dwt / (2.0 * sigkelv)

    r_kelvin_h2o = np.exp(r_kelvin_h2o_a * r_kelvin_h2o_b)

    # Effective RH adjusted by Kelvin effect
    # To match legacy exactly, we use their slightly inconsistent constants
    k_cgs_legacy = 1.3807e-16
    mw_h2o_cgs_legacy = 18.0
    relhum_eff = h2o_mass_g_cm3 / r_kelvin_h2o * navogad_legacy / mw_h2o_cgs_legacy * k_cgs_legacy * temp / (es * 10.0)
    relhum_eff = relhum_eff.clip(1e-6, 0.999)

    # Legacy calls wtpct WITHOUT temp in the final step, using default temp=220
    # To match exactly, we should check if we should pass temp or use 220.
    # Legacy grow_v75: rwet = rdry * (100. * rhopdry / legacy_wtpct(relhum_) / rhopwet)**(1. / 3.)
    # where legacy_wtpct(relhum_) uses default temp=220.
    wtp_final = wtpct_sulfate(relhum_eff, temp=220.0)
    rho_final = density_sulfate(wtp_final, temp)

    gf = ( rhopdry_legacy / ( (wtp_final / 100.0) * rho_final ) )**(1.0 / 3.0)

    gf.attrs = {
        "units": "1",
        "long_name": "hygroscopic growth factor",
        "history": "Calculated including Kelvin effect for sulfate"
    }
    return gf

def growth_factor_sea_salt(relhum: xr.DataArray, r_dry: Union[xr.DataArray, float]) -> xr.DataArray:
    """
    Calculate the hygroscopic growth factor for sea salt particles.
    Based on Gerber (1985).

    Parameters
    ----------
    relhum : xr.DataArray
        Relative humidity in the range [0, 1].
    r_dry : xr.DataArray or float
        Dry particle radius in meters.

    Returns
    -------
    xr.DataArray
        Growth factor (r_wet / r_dry).
    """
    # Convert to cm as the formula expects cm
    s_cm = r_dry * 100.0

    c1, c2, c3, c4 = 0.7674, 3.079, 2.573e-11, -1.424

    # Handle RH=0
    log_rh = np.log10(relhum.where(relhum > 0, 1e-10))

    r_wet_cm = (c1 * s_cm**c2 / (c3 * s_cm**c4 - log_rh) + s_cm**3.0)**(1.0 / 3.0)
    gf = r_wet_cm / s_cm

    # For RH=0, GF=1
    gf = xr.where(relhum <= 0, 1.0, gf)

    gf.attrs = {
        "units": "1",
        "long_name": "hygroscopic growth factor",
        "history": "Calculated via Gerber (1985) for sea salt"
    }
    return gf
