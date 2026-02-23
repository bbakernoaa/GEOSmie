"""
Physical constants for meteorological and microphysical calculations.
Values are in SI units unless otherwise specified.
"""

import numpy as np

# Mathematical constants
PI = np.pi
FOUR_THIRDS_PI = 4.0 / 3.0 * np.pi

# Gas constants
R_UNIVERSAL = 8.314462618  # Universal gas constant [J mol-1 K-1]
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant [J K-1]
AVOGADRO = 6.02214076e23  # Avogadro constant [mol-1]

# Water properties
MW_H2O = 0.01801528  # Molecular weight of water [kg mol-1]
R_VAPOR = 461.5  # Gas constant for water vapor [J K-1 kg-1]
L_V = 2.501e6  # Latent heat of vaporization at 0°C [J kg-1]
T_TRIPLE = 273.16  # Triple point of water [K]
E_STR = 611.65  # Saturation vapor pressure at triple point [Pa]

# Sulfuric acid properties
MW_H2SO4 = 0.098079  # Molecular weight of H2SO4 [kg mol-1]
RHO_H2SO4_DRY = 1830.0  # Density of dry sulfuric acid [kg m-3] (Tabazadeh value approx)

# Standard gravity
G = 9.80665  # [m s-2]
