/* SPECTRUM
- Spectral data and helpers (host side).
- All per-band quantities are band averages over the NB bands defined in common.hpp.
*/

#pragma once
#include "common.hpp"

namespace spectrum {

// Averages of tabulated data (linearly interpolated) over each band
void cie_xyz_weights(float wx[NB], float wy[NB], float wz[NB]);   // integral of cmf over band (nm)
void solar_irradiance_1au(float out[NB]);                          // W m^-2 nm^-1 at 1 AU
void ozone_cross_section(float out[NB]);                           // m^2 / molecule
void smits_basis(float out[7][NB]);
void water_reflectance(float out[NB]);                             // deep ocean diffuse reflectance

double planck(double lambda_nm, double T);                         // W m^-2 sr^-1 nm^-1
void   planck_bands(double T, float out[NB]);
double air_refractivity_std(double lambda_nm);                     // (n-1) of dry air at 15 C, 101325 Pa
double rayleigh_cross_section(double lambda_nm);                   // m^2 per molecule (standard air)
float  solar_limb_darkening(double lambda_nm);                     // linear law coefficient

// XYZ (CIE 1931 2 deg) -> linear sRGB (D65)
void xyz_to_srgb(const double xyz[3], double rgb[3]);
}
