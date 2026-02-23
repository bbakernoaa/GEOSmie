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

    # Saturation vapor pressure
    es = E_STR * np.exp(L_V / R_VAPOR * (1.0 / T_TRIPLE - 1.0 / temp))

    # Mass concentration of water (vapor)
    # n_v = relhum * es / (k * temp)
    h2o_mass_kg_m3 = (relhum * es) / (R_VAPOR * temp)
    h2o_mass_g_cm3 = h2o_mass_kg_m3 * 1e3 / 1e6

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
    # Legacy: rkelvinH2O_a = 2. * mw_h2so4 * sigkelv / (den1 * rgas * temp * rwet)
    # Note: Legacy uses mw_h2so4 (98) and rgas (8.31e7) which are CGS-ish.

    r_dry_m = r_dry
    r_wet_init = r_dry_m * ( (RHO_H2SO4_DRY / 1000.0) / (wtp_init / 100.0) / den2 )**(1.0 / 3.0)

    # Corrected Kelvin terms in SI
    # r_kelvin = exp(2 * Sigma * MW / (rho * R * T * r))
    # Using water properties for Kelvin effect on water activity?
    # Legacy uses mw_h2so4 which is weird if it's Kelvin effect for water.
    # Actually, legacy says "rkelvinH2O".

    r_kelvin_h2o_a = 2.0 * MW_H2O * sig_si / (den_si * R_UNIVERSAL * temp * r_wet_init)
    # Legacy has a 'b' term: 1. + wtpkelv * drho_dwt / den2 - 3. * wtpkelv * dsigma_dwt / (2.*sigkelv)
    # This 'b' term looks like a derivative correction for concentration-dependent surface tension/density.
    r_kelvin_h2o_b = 1.0 + wtp_init * drho_dwt / den2 - 3.0 * wtp_init * dsigma_dwt / (2.0 * sigkelv)

    r_kelvin_h2o = np.exp(r_kelvin_h2o_a * r_kelvin_h2o_b)

    # Effective RH adjusted by Kelvin effect
    relhum_eff = relhum / r_kelvin_h2o
    relhum_eff = relhum_eff.clip(1e-6, 0.999)

    wtp_final = wtpct_sulfate(relhum_eff, temp)
    rho_final = density_sulfate(wtp_final, temp)

    # r_wet = r_dry * (rho_dry / (wtp/100 * rho_wet))^(1/3)
    # Legacy uses 1.923 for rhopdry.
    gf = ( 1.923 / ( (wtp_final / 100.0) * rho_final ) )**(1.0 / 3.0)

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
