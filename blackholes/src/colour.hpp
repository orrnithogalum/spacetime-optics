/* COLOUR
- Blackbody colorimetry (host).
- The renderer represents every light source as a scaled blackbody s * B_nu(T): redshifting by g
  maps it exactly to s * B_nu(g T), so colour shifts are a table look-up of XYZ(T).
*/

#pragma once
#include <vector>

namespace colour {
constexpr double BB_T0 = 300.0, BB_T1 = 3e7;
constexpr int BB_N = 2048;
double planck_lambda(double lambda_nm, double T);            // W m^-2 sr^-1 nm^-1
std::vector<float> blackbody_xyz_table(float* l0, float* dl);   // 3 floats per entry, ln T grid
void blackbody_xyz(double T, double xyz[3]);
void xyz_to_srgb(const double xyz[3], double rgb[3]);
void srgb_to_xyz(const double rgb[3], double xyz[3]);
}
