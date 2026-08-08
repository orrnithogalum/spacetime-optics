/* ATMOSPHERE MODEL
- Neutral atmosphere density: U.S.
- Standard Atmosphere 1976, computed exactly from its layer definitions below 86 km and interpolated
  (monotone cubic in ln rho) from the published tables up to 1000 km.
- Density matters for two things here:
    - Rayleigh scattering and extinction (0-100 km)
    - Refraction, which dominates gravitational bending below ~110 km and is still a few percent of
      it at 150 km, so the thermosphere has to be modelled properly.
*/

#pragma once
#include <vector>

namespace atmo {
constexpr double RHO0 = 1.2250;   // kg/m^3, sea-level density of the 1976 standard

double density(double z_m);                        // kg/m^3 at geometric altitude z
std::vector<float> lnrho_table(int* n_out);        // ln(rho/RHO0), LNRHO_H0 + i*LNRHO_DH
}
