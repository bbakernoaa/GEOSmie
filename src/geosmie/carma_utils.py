#!/usr/bin/env python3

# Utilities for calculating properties for CARMA
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
from src.utils.psd import carma_bins, find_rmin, lognormal_dndlnr
from src.utils.microphysics import wtpct_sulfate, density_sulfate, growth_factor_sulfate
from src.utils.constants import FOUR_THIRDS_PI as cpi

def tune():
    """
    Use this function to select appropriate lognormal width sigma
    to satisfy your needs for describing CARMA size bins with
    individual lognormal distributions.
    Given nbin, rhop, rlow, rup and sigma prints out a table of
    bin #, median radius, fraction of number distribution contained,
    and effective radius from default CARMA and a higher bin resolution
    Also produces an illustrative plot.
    """
    nbin = 24
    rhop0 = 1700.
    num0 = 800
    sigma0 = 1.1
    rlow = 4.25e-10
    rup  = 2.87e-5
    rhop = np.ones(nbin)*rhop0
    sigma = np.ones(nbin)*sigma0
    num = 800
    numperdecade = np.ones(nbin)*num0
    fracs = np.ones(nbin)
   
    rmin, rmrat = find_rmin(nbin, rlow, rup)
    r, dr, rlow, rup, rmassup = carma_bins(nbin, rmrat, rmin)
#   Compute rmed given r = reff
    rmed = r*np.exp(-5.*np.log(sigma)*np.log(sigma)/2.)

#   Now compute the number in the default intervals but at a higher res
    for ibin in range(0,nbin):
        nb = num
        rl = rlow[ibin].values if hasattr(rlow[ibin], 'values') else rlow[ibin]
        ru = rup[ibin].values if hasattr(rup[ibin], 'values') else rup[ibin]
        rm, rmr = find_rmin(nb, rl, ru)
        r_, dr_, rlow_, rup_, rmassup_ = carma_bins(nb, rmr, rm)
        dndlogr = lognormal_dndlnr(r_, rmed[ibin], sigma[ibin])
        reff_ = np.sum(r_**3*dndlogr*dr_/r_)/np.sum(r_**2*dndlogr*dr_/r_)
        print(ibin, rmed[ibin], np.sum(dndlogr*dr_/r_), reff_, r[ibin])
        plt.bar(rlow[ibin], 1, width=dr[ibin], align='edge', color='blue', edgecolor='black', label='CARMA Number', alpha=0.5)
        plt.plot(r_,dndlogr/np.max(dndlogr),color='black')

    plt.xscale('log')
    plt.show()

    return

def carmabins(nbin, rmrat, rmin, rhop=1.):
    r, dr, rlow, rup, rmassup = carma_bins(nbin, rmrat, rmin, rhop)
    return r.values, dr.values, rlow.values, rup.values, rmassup.values

def findrmin(nbin,rlow,rup):
    return find_rmin(nbin, rlow, rup)

def lognormal(r, rm, sigma, N=1.):
    return lognormal_dndlnr(r, rm, sigma, N).values

def wtpct(relhum, temp=220.):
    return float(wtpct_sulfate(relhum, temp))

def dens(relhum,temp=220.):
    wtp = wtpct_sulfate(relhum, temp)
    return float(density_sulfate(wtp, temp))

def grow_v75(relhum, rd, temp=220.):
    
    '''
    Explanatory comments
    rd dry radius in meters
    relhum relative humidity in range 0 - 1
    temp temperature in K
    '''

#   rdry in cm
    rdry = rd*100.
    
#   Assumed dry density of sulfate [g cm-3]
    rhopdry = 1.923 

#   molecular weight of H2SO4 [g mol-1]
    mw_h2so4 = 98.

#   Ideal gas constant [erg mol-1 K-1]
    rgas = 8.31447e7


#   Calculate the mass concentration of water given rh and temp
#   first, saturation vapor pressure (Curry & Webster, 4.31)
    llv = 2.501e6 # J kg-1, Curry & Webster Table 4.2 @ 0 C
    rv  = 461.    # J K-1 kg-1, Curry & Webster pg. 438, gas constant for water
    ttr = 273.16  # triple point [K]
    estr = 611.   # Pa
    es = estr*np.exp(llv/rv*(1./ttr - 1/temp))
#   ideal gas law, pv = nkT, rearrange to n/v = p/(kT)
    k   = 1.38e-23             # Boltzman constant mks
    n_v = relhum*es / (k*temp) # number per m-3
    navogad = 6.022e23         # mole-1
    mw_h2o  = 0.018            # kg mole-1
#   And finally to mass concentration, and note conversion to g cm-3
    h2o_mass = n_v / navogad *mw_h2o * 1000. / 1.e6

#   Adjust calculation for the Kelvin effect of H2O:
    wtpkelv = 80.                        # start with assumption of 80 wt % H2SO4 
    den1 = 2.00151 - 0.000974043 * temp  # density at 79 wt %
    den2 = 2.01703 - 0.000988264 * temp  # density at 80 wt %
    drho_dwt = den2-den1                 # change in density for change in 1 wt %
      
    sig1 = 79.3556 - 0.0267212 * temp    # surface tension at 79.432 wt %
    sig2 = 75.608  - 0.0269204 * temp    # surface tension at 85.9195 wt %      
    dsigma_dwt = (sig2-sig1) / (85.9195 - 79.432) # change in density for change in 1 wt %
    sigkelv = sig1 + dsigma_dwt * (80.0 - 79.432)
      
    rwet = rdry * (100. * rhopdry / wtpkelv / den2)**(1. / 3.)

    rkelvinH2O_b = 1. + wtpkelv * drho_dwt / den2 - 3. * wtpkelv * dsigma_dwt / (2.*sigkelv)

    rkelvinH2O_a = 2. * mw_h2so4 * sigkelv / (den1 * rgas * temp * rwet)     

    rkelvinH2O = np.exp(rkelvinH2O_a*rkelvinH2O_b)
            
    h2o_kelv = h2o_mass / rkelvinH2O

#   wtpct just wants a relative humidity, but it is recalculated here
#   based on the calculated terms from above
    k_cgs = 1.3807e-16 # cm2 g s-2 K-1
    mw_h2o_cgs = 18.   # g mol-1
    relhum_ = h2o_kelv*navogad/mw_h2o_cgs*k_cgs*temp / (es*10.) # es now dyne cm-2
    rhopwet = dens(relhum_,temp=temp)
    rwet    = rdry * (100. * rhopdry / wtpct(relhum_) / rhopwet)**(1. / 3.)   


    return float(growth_factor_sulfate(relhum, rd, temp))

def printfield(f,nbin,title,field,close=False):

#   Find the blank spaces
    head = " "*(len(title)+5)

#   Practice printing in groups of n for readability
    n = 5
    if (title == 'rh') | (title == 'gf'):
        n = 10
    ngrp = nbin//n
    nexa = nbin % n
    print('"%s": ['%(title),end="",file=f)
    for ibin in range(0,n-1):
        print(field[ibin]+",",end="",file=f)
    ibin = n-1
    print(field[ibin]+",",file=f)
    for igrp in range(1,ngrp):
        ibs = igrp*n
        print('%s'%(head)+field[ibs]+",",end="",file=f)
        for ibin in range(ibs+1,np.min([nbin,ibs+n-1])):
            print(field[ibin]+",",end="",file=f)
        ibin = ibs+n-1
        if ibin == nbin-1:
            if close:
                print(field[ibin]+"]}},",file=f)
            else:
                print(field[ibin]+"],",file=f)
        else:
            print(field[ibin]+",",file=f)
    if nexa == 0:
        return
    if nexa == 1:
        ibin = nbin-1
        print('%s'%(head)+field[ibin]+"],",file=f)
        return
    else:
        ibs = ngrp*n
        print('%s'%(head)+field[ibs]+",",end="",file=f)
        for ibin in range(ibs+1,nbin-1):
            print(field[ibin]+",",end="",file=f)
        ibin = nbin-1
        if close:
            print(field[ibin]+"]}},",file=f)
        else:
            print(field[ibin]+"],",file=f)
    print('',file=f)
    return

def printjson(nbin, rlow, rup, rhop0, species='SU', sigma0=1.1,num0=800,version='v2.0.0'):
    '''
    Given particle properties and a species identifier (for RH growth handling)
    write out a CARMA style json file suitable to run runoptics.py
    Assumes single density rhop0 valid at all bins
    All input fields in MKS:
     rlow  - left edge lower limit of size grid
     rup   - right edge upper limit of size grid
     rhop0 - particle dry density
    '''
#   Density
    rhop = np.ones(nbin)*rhop0

#   Lognormal width
    sigma = np.ones(nbin)*sigma0

#   Resolution of calculation
    num = np.ones(nbin)*num0

#   Assume single lognormal mode per bin
    fracs = np.ones(nbin)

#   Find particle grid
    rmin, rmrat = find_rmin(nbin, rlow, rup)
    r, dr, rlow, rup, rmassup = carma_bins(nbin, rmrat, rmin)

#   Compute rmed given r = reff and sigma
    rmed = r*np.exp(-5.*np.log(sigma)*np.log(sigma)/2.)

#   Define RH grid
    rh = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                   0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85,
                   0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                   0.97, 0.98, 0.99])

#   Print
    f = open('carma%s.%s.json'%(species,version),'w')
    print('{', file=f)
    print('',file=f)
    printfield(f,nbin,'rhop0',["%5.1f"%(x) for x in rhop])
    printfield(f,len(rh),'rh',["%4.2f"%(x) for x in rh])

#   Species RH handling
    if species == 'SU':
        print('"rhDep": {"type": "su", "params":{',file=f)
        print('"temp": 220.0}},',file=f)
    elif species == 'SS':
        print('"maxrh": 0.95,',file=f)
        print('"rhDep": {"type": "ss", "params":{',file=f)
        print('"c1": 0.7674,',file=f)
        print('"c2": 3.079,',file=f)
        print('"c3": 2.573e-11,',file=f)
        print('"c4": -1.424}},',file=f)
    elif species == 'DU':
        print('"rhDep": {"type": "trivial", "params":{',file=f)
        print('"gf": [1.0]}},',file=f)
    elif species == 'OC':
        print('"rhDep": {"type": "simple", "params":{',file=f)
        gf = np.array([1.00, 1.02, 1.05, 1.07, 1.09, 1.12, 1.14, 1.17, 1.19, 1.21, 1.24,
                       1.26, 1.29, 1.32, 1.34, 1.39, 1.44, 1.46, 1.47, 1.49, 1.50, 1.52,
                       1.54, 1.56, 1.58, 1.61, 1.64, 1.67, 1.71, 1.76, 1.81, 1.88, 1.97,
                       2.08, 2.25, 2.52])
        printfield(f,len(gf), 'gf',["%4.2f"%(x) for x in gf],close=True)
    elif species == 'BC':
        print('"rhDep": {"type": "simple", "params":{',file=f)
        gf = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
                       1.00, 1.01, 1.01, 1.03, 1.10, 1.19, 1.21, 1.23, 1.25, 1.27, 1.30,
                       1.32, 1.34, 1.36, 1.38, 1.41, 1.43, 1.46, 1.48, 1.52, 1.55, 1.59,
                       1.65, 1.72, 1.89])
        printfield(f,len(gf), 'gf',["%4.2f"%(x) for x in gf],close=True)
    else:
        print('Species unsupported!!!!')

    print('',file=f)
    print('"psd": {"type": "lognorm", "params":{',file=f)
    printfield(f,nbin,'r0',['['+"%12.6e"%(x)+']' for x in rmed])
    printfield(f,nbin,'rmin0',['['+"%12.6e"%(x)+']' for x in rlow])
    printfield(f,nbin,'rmax0',['['+"%12.6e"%(x)+']' for x in rup])
    printfield(f,nbin,'sigma',['['+"%5.3f"%(x)+']' for x in sigma])
    printfield(f,nbin,'numperdec',["%d"%(x) for x in num])
    printfield(f,nbin,'fracs',['['+"%3.1f"%(x)+']' for x in fracs],close=True)

#   Species RH handling
    if species == 'SU':
        print('"ri": {"format": "gads", "path": ["data/suso00"]}',file=f)
    elif species == 'SS':
        print('"ri": {"format": "gads", "path": ["data/sscm00"]}',file=f)
    elif species == 'DU':
        print('"ri": {"format": "wsv", "path": ["data/ri-dust.wsv"]}',file=f)
    elif species == 'OC':
        print('"ri": {"format": "gads", "path": ["data/waso00"]}',file=f)
    elif species == 'BC':
        print('"ri": {"format": "gads", "path": ["data/soot00"]}',file=f)
    else:
        print('Species unsupported!!!!')
    print('}',file=f)
    f.close()



if __name__ == "__main__":
    parser = OptionParser(usage="Usage: %prog",
                          version='0.0.1' )
    parser.add_option("--nbin", type=int, dest="nbin", default=24,
                      help="Number of bins (default=24)")
    parser.add_option("--rlow", type=float, dest="rlow", default=4.25e-08,
                      help="Lower left edge radius (default=4.25e-08 m)")
    parser.add_option("--rup", type=float, dest="rup", default=2.87e-05,
                      help="Upper right edge radius (default=2.87e-05 m)")
    parser.add_option("--rhop", type=float, dest="rhop", default=1700.0,
                      help="Dry particle density (default=1700 kg m-3)")
    parser.add_option("--sigma", type=float, dest="sigma", default=1.1,
                      help="Width of lognormal (default=1.1)")
    parser.add_option("--num", type=int, dest="num", default=800,
                      help="Number of sub-bins (default=800)")
    parser.add_option("--species", dest="species", default="SU",
                      help="Species (default=SU)")
    parser.add_option("--versionid", dest="version", default="v2.0.0",
                      help="Version ID (default=v2.0.0)")

    (options, args) = parser.parse_args()
    
    printjson(options.nbin,options.rlow,options.rup,options.rhop,
              sigma0=options.sigma,num0=options.num,species=options.species,
              version=options.version)
