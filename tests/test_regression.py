import numpy as np
import xarray as xr
import pytest
from src.utils.microphysics import wtpct_sulfate, density_sulfate, growth_factor_sulfate

# Legacy implementations for comparison
def legacy_wtpct(relhum, temp=220.):
    activ = relhum
    if activ < 0.05:
        activ   = np.max([activ,1.e-6])
        atab1, btab1, ctab1, dtab1 = 12.37208932, -0.16125516114, -30.490657554, -2.1133114241
        atab2, btab2, ctab2, dtab2 = 13.455394705, -0.1921312255, -34.285174607, -1.7620073078
    elif (activ >= 0.05) & (activ <= 0.85):
        atab1, btab1, ctab1, dtab1 = 11.820654354, -0.20786404244, -4.807306373, -5.1727540348
        atab2, btab2, ctab2, dtab2 = 12.891938068, -0.23233847708, -6.4261237757, -4.9005471319
    else:
        activ   = np.min([activ,1.])
        atab1, btab1, ctab1, dtab1 = -180.06541028, -0.38601102592, -93.317846778, 273.88132245
        atab2, btab2, ctab2, dtab2 = -176.95814097, -0.36257048154, -90.469744201, 267.45509988

    contl = atab1*(activ**btab1)+ctab1*activ+dtab1
    conth = atab2*(activ**btab2)+ctab2*activ+dtab2
    contt = contl + (conth-contl) * ((temp -190.)/70.)
    conwtp = (contt*98.) + 1000.
    wtpct_tabaz = (100.*contt*98.)/conwtp
    return np.min([np.max([wtpct_tabaz,1.]),100.])

def legacy_dens(relhum, temp=220.):
    wtp = legacy_wtpct(relhum, temp=temp)
    dnwtp = np.array([ 0., 1., 5., 10., 20., 25., 30., 35., 40.,
     41., 45., 50., 53., 55., 56., 60., 65., 66., 70.,
     72., 73., 74., 75., 76., 78., 79., 80., 81., 82.,
     83., 84., 85., 86., 87., 88., 89., 90., 91., 92.,
     93., 94., 95., 96., 97., 98., 100. ])
    dnc0 = np.array([ 1., 1.13185, 1.17171, 1.22164, 1.3219, 1.37209,
     1.42185, 1.4705, 1.51767, 1.52731, 1.56584, 1.61834, 1.65191,
     1.6752, 1.68708, 1.7356, 1.7997, 1.81271, 1.86696, 1.89491,
     1.9092, 1.92395, 1.93904, 1.95438, 1.98574, 2.00151, 2.01703,
     2.03234, 2.04716, 2.06082, 2.07363, 2.08461, 2.09386, 2.10143,
     2.10764, 2.11283, 2.11671, 2.11938, 2.12125, 2.1219, 2.12723,
     2.12654, 2.12621, 2.12561, 2.12494, 2.12093 ])
    dnc1 = np.array([ 0.,  -0.000435022, -0.000479481, -0.000531558, -0.000622448,
     -0.000660866, -0.000693492, -0.000718251, -0.000732869, -0.000735755,
     -0.000744294, -0.000761493, -0.000774238, -0.00078392, -0.000788939,
     -0.00080946, -0.000839848, -0.000845825, -0.000874337, -0.000890074,
     -0.00089873, -0.000908778, -0.000920012, -0.000932184, -0.000959514,
     -0.000974043, -0.000988264, -0.00100258, -0.00101634, -0.00102762,
     -0.00103757, -0.00104337, -0.00104563, -0.00104458, -0.00104144,
     -0.00103719, -0.00103089, -0.00102262, -0.00101355, -0.00100249,
     -0.00100934, -0.000998299, -0.000990961, -0.000985845, -0.000984529,
     -0.000989315 ])
    i=0
    while wtp > dnwtp[i]: i += 1
    den2 = dnc0[i]+dnc1[i]*temp
    if (i == 0) or (wtp == dnwtp[i]): dens = den2
    else:
        den1=dnc0[i-1]+dnc1[i-1]*temp
        frac=(dnwtp[i]-wtp)/(dnwtp[i]-dnwtp[i-1])
        dens=den1*frac+den2*(1.0-frac)
    return dens

def legacy_getLogNormPSD(rmode, sigma, xxArr, lambd, rmax, rmin):
    xconv = 2 * np.pi / lambd
    xmode = rmode * xconv
    xmax = rmax * xconv
    xmin = rmin * xconv
    dNdx = 1./(xxArr * (2*np.pi) ** 0.5 * np.log(sigma)) * np.exp(-(np.log(xxArr/xmode)**2) / (2. * np.log(sigma) ** 2))
    dNdx[np.where(xxArr >= xmax)] = 0.
    dNdx[np.where(xxArr <= xmin)] = 0.
    return dNdx

def legacy_grow_v75(relhum, rd, temp=220.):
    rdry = rd*100.
    rhopdry = 1.923
    mw_h2so4 = 98.
    rgas = 8.31447e7
    llv = 2.501e6 # J kg-1
    rv  = 461.    # J K-1 kg-1
    ttr = 273.16  # triple point [K]
    estr = 611.   # Pa
    es = estr*np.exp(llv/rv*(1./ttr - 1/temp))
    k   = 1.38e-23             # Boltzman constant mks
    n_v = relhum*es / (k*temp) # number per m-3
    navogad = 6.022e23         # mole-1
    mw_h2o  = 0.018            # kg mole-1
    h2o_mass = n_v / navogad *mw_h2o * 1000. / 1.e6
    wtpkelv = 80.
    den1 = 2.00151 - 0.000974043 * temp
    den2 = 2.01703 - 0.000988264 * temp
    drho_dwt = den2-den1
    sig1 = 79.3556 - 0.0267212 * temp
    sig2 = 75.608  - 0.0269204 * temp
    dsigma_dwt = (sig2-sig1) / (85.9195 - 79.432)
    sigkelv = sig1 + dsigma_dwt * (80.0 - 79.432)
    rwet = rdry * (100. * rhopdry / wtpkelv / den2)**(1. / 3.)
    rkelvinH2O_b = 1. + wtpkelv * drho_dwt / den2 - 3. * wtpkelv * dsigma_dwt / (2.*sigkelv)
    rkelvinH2O_a = 2. * mw_h2so4 * sigkelv / (den1 * rgas * temp * rwet)
    rkelvinH2O = np.exp(rkelvinH2O_a*rkelvinH2O_b)
    h2o_kelv = h2o_mass / rkelvinH2O
    k_cgs = 1.3807e-16 # cm2 g s-2 K-1
    mw_h2o_cgs = 18.   # g mol-1
    relhum_ = h2o_kelv*navogad/mw_h2o_cgs*k_cgs*temp / (es*10.)
    rhopwet = legacy_dens(relhum_,temp=temp)
    rwet    = rdry * (100. * rhopdry / legacy_wtpct(relhum_) / rhopwet)**(1. / 3.)
    return rwet/rdry

# Regression Tests
def test_wtpct_regression():
    rhs = np.linspace(0, 1, 100)
    for rh in rhs:
        l = legacy_wtpct(rh)
        n = float(wtpct_sulfate(xr.DataArray(rh)))
        np.testing.assert_allclose(l, n, rtol=1e-7)

def test_dens_regression():
    rhs = np.linspace(0, 1, 100)
    for rh in rhs:
        l = legacy_dens(rh)
        w = wtpct_sulfate(xr.DataArray(rh))
        n = float(density_sulfate(w))
        np.testing.assert_allclose(l, n, rtol=1e-7)

def test_grow_v75_regression():
    rhs = np.linspace(0.1, 0.95, 20)
    rds = [1e-8, 1e-7, 1e-6]
    for rh in rhs:
        for rd in rds:
            l = legacy_grow_v75(rh, rd)
            n = float(growth_factor_sulfate(xr.DataArray(rh), rd))
            np.testing.assert_allclose(l, n, rtol=1e-7)

def test_psd_regression():
    from src.utils.psd import lognormal_dndr
    rmode = 1e-7
    sigma = 1.5
    lambd = 550e-9
    rmin = 1e-8
    rmax = 1e-5

    xconv = 2 * np.pi / lambd
    xxArr = np.logspace(np.log10(rmin*xconv), np.log10(rmax*xconv), 100)

    l_dNdx = legacy_getLogNormPSD(rmode, sigma, xxArr, lambd, rmax, rmin)

    # New implementation dN/dr
    r = xr.DataArray(xxArr / xconv)
    n_dndr = lognormal_dndr(r, rmode, sigma)

    # Compare dN: legacy dN = dNdx * dx; new dN = dndr * dr
    # Wait, easier to compare dN/dx * dx/dr with dN/dr
    l_dndr = l_dNdx * (2 * np.pi / lambd)

    # Only compare non-zero values due to truncation
    mask = (xxArr > rmin*xconv) & (xxArr < rmax*xconv)
    np.testing.assert_allclose(l_dndr[mask], n_dndr.values[mask], rtol=1e-7)
