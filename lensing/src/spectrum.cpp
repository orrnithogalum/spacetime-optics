#include "spectrum.hpp"

#include <cmath>

namespace spectrum {
namespace {

// CIE 1931 2-degree colour matching functions, 380..780 nm in 10 nm steps
const double CIE_X[41] = {0.001368, 0.004243, 0.014310, 0.043510, 0.134380, 0.283900, 0.348280,
                          0.336200, 0.290800, 0.195360, 0.095640, 0.032010, 0.004900, 0.009300,
                          0.063270, 0.165500, 0.290400, 0.433450, 0.594500, 0.762100, 0.916300,
                          1.026300, 1.062200, 1.002600, 0.854450, 0.642400, 0.447900, 0.283500,
                          0.164900, 0.087400, 0.046770, 0.022700, 0.011359, 0.005790, 0.002899,
                          0.001440, 0.000690, 0.000332, 0.000166, 0.000083, 0.000042};
const double CIE_Y[41] = {0.000039, 0.000120, 0.000396, 0.001210, 0.004000, 0.011600, 0.023000,
                          0.038000, 0.060000, 0.090980, 0.139020, 0.208020, 0.323000, 0.503000,
                          0.710000, 0.862000, 0.954000, 0.994950, 0.995000, 0.952000, 0.870000,
                          0.757000, 0.631000, 0.503000, 0.381000, 0.265000, 0.175000, 0.107000,
                          0.061000, 0.032000, 0.017000, 0.008210, 0.004102, 0.002091, 0.001047,
                          0.000520, 0.000249, 0.000120, 0.000060, 0.000030, 0.000015};
const double CIE_Z[41] = {0.006450, 0.020050, 0.067850, 0.207400, 0.645600, 1.385600, 1.747060,
                          1.772110, 1.669200, 1.287640, 0.812950, 0.465180, 0.272000, 0.158200,
                          0.078250, 0.042160, 0.020300, 0.008750, 0.003900, 0.002100, 0.001650,
                          0.001100, 0.000800, 0.000340, 0.000190, 0.000050, 0.000020, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/* SOLAR
- Extraterrestrial solar spectral irradiance at 1 AU (W m^-2 nm^-1), 360..830 nm, 10 nm steps (as
  used by Bruneton's precomputed atmospheric scattering reference implementation).
*/
const double SOLAR[48] = {1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
                          1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
                          1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158,
                          1.7533, 1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113,
                          1.474, 1.4482, 1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758,
                          1.2367, 1.2082, 1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992};

/* OZONE
- Ozone absorption cross section (m^2/molecule), 360..830 nm, 10 nm steps (Serdyuchenko 2014, 293 K,
  as tabulated by Bruneton).
*/
const double OZONE[48] = {1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27,
                          5.52e-27, 8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26,
                          7.752e-26, 9.016e-26, 1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25,
                          3.091e-25, 3.5e-25, 4.266e-25, 4.672e-25, 4.398e-25, 4.701e-25,
                          5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25, 2.238e-25,
                          1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26,
                          5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
                          2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27};

// Smits (1999) RGB -> reflectance basis, 10 bins spanning 380..720 nm
const double SMITS[7][10] = {
    {1.0000, 1.0000, 0.9999, 0.9993, 0.9992, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000},   // white
    {0.9710, 0.9426, 1.0007, 1.0007, 1.0007, 1.0007, 0.1564, 0.0000, 0.0000, 0.0000},   // cyan
    {1.0000, 1.0000, 0.9685, 0.2229, 0.0000, 0.0458, 0.8369, 1.0000, 1.0000, 0.9959},   // magenta
    {0.0001, 0.0000, 0.1088, 0.6651, 1.0000, 1.0000, 0.9996, 0.9586, 0.9685, 0.9840},   // yellow
    {0.1012, 0.0515, 0.0000, 0.0000, 0.0000, 0.0000, 0.8325, 1.0149, 1.0149, 1.0149},   // red
    {0.0000, 0.0000, 0.0273, 0.7937, 1.0000, 0.9418, 0.1719, 0.0000, 0.0000, 0.0025},   // green
    {1.0000, 1.0000, 0.8916, 0.3323, 0.0000, 0.0000, 0.0003, 0.0369, 0.0483, 0.0496}};  // blue

double interp(const double* t, int n, double l0, double dl, double lambda) {
    double x = (lambda - l0) / dl;
    if (x <= 0) return t[0];
    if (x >= n - 1) return t[n - 1];
    int i = (int)x;
    double f = x - i;
    return t[i] * (1 - f) + t[i + 1] * f;
}

// Band average of a tabulated function, integrated with 1 nm sub-steps
double band_avg(const double* t, int n, double l0, double dl, int k) {
    double a = LAMBDA0_NM + k * DLAMBDA_NM, s = 0;
    int m = (int)DLAMBDA_NM;
    for (int i = 0; i < m; ++i) s += interp(t, n, l0, dl, a + i + 0.5);
    return s / m;
}
}  // namespace

void cie_xyz_weights(float wx[NB], float wy[NB], float wz[NB]) {
    for (int k = 0; k < NB; ++k) {
        wx[k] = (float)(band_avg(CIE_X, 41, 380, 10, k) * DLAMBDA_NM);
        wy[k] = (float)(band_avg(CIE_Y, 41, 380, 10, k) * DLAMBDA_NM);
        wz[k] = (float)(band_avg(CIE_Z, 41, 380, 10, k) * DLAMBDA_NM);
    }
}

void solar_irradiance_1au(float out[NB]) {
    for (int k = 0; k < NB; ++k) out[k] = (float)band_avg(SOLAR, 48, 360, 10, k);
}

void ozone_cross_section(float out[NB]) {
    for (int k = 0; k < NB; ++k) out[k] = (float)band_avg(OZONE, 48, 360, 10, k);
}

void smits_basis(float out[7][NB]) {
    // Smits' bins are centred at 380 + (i+0.5)*34 nm
    for (int b = 0; b < 7; ++b)
        for (int k = 0; k < NB; ++k) {
            double l = band_center_nm(k);
            out[b][k] = (float)interp(SMITS[b], 10, 380 + 17, 34, l);
        }
}

void water_reflectance(float out[NB]) {
    /* R
    - Diffuse reflectance of clear open-ocean water (Morel-type case 1 water, low chlorophyll):
      blue-violet peak, almost black beyond 600 nm.
    */
    const double R[9] = {0.060, 0.055, 0.045, 0.030, 0.015, 0.007, 0.003, 0.002, 0.0015};
    for (int k = 0; k < NB; ++k) out[k] = (float)interp(R, 9, 400, 50, band_center_nm(k));
}

double planck(double lambda_nm, double T) {
    double l = lambda_nm * 1e-9;
    double c1 = 2.0 * phys::H_PLANCK * phys::C * phys::C;
    double c2 = phys::H_PLANCK * phys::C / phys::K_BOLTZ;
    double B = c1 / (l * l * l * l * l) / std::expm1(c2 / (l * T));   // W m^-2 sr^-1 m^-1
    return B * 1e-9;
}

void planck_bands(double T, float out[NB]) {
    for (int k = 0; k < NB; ++k) {
        double s = 0, a = LAMBDA0_NM + k * DLAMBDA_NM;
        for (int i = 0; i < (int)DLAMBDA_NM; ++i) s += planck(a + i + 0.5, T);
        out[k] = (float)(s / DLAMBDA_NM);
    }
}

double air_refractivity_std(double lambda_nm) {
    // Ciddor (1996) dispersion of standard dry air (15 C, 101325 Pa, 450 ppm CO2)
    double s2 = 1.0 / (lambda_nm * 1e-3);
    s2 *= s2;
    return (5792105.0 / (238.0185 - s2) + 167917.0 / (57.362 - s2)) * 1e-8;
}

double rayleigh_cross_section(double lambda_nm) {
    // Sigma = 24 pi^3 (n^2-1)^2 / (lambda^4 N^2 (n^2+2)^2) * F_K, standard air number density
    const double N = 2.546899e25;   // m^-3 at 15 C, 101325 Pa
    double n = 1.0 + air_refractivity_std(lambda_nm);
    double l = lambda_nm * 1e-9;
    double n2 = n * n;
    const double FK = 1.048;   // King correction factor for air (depolarisation ~0.028)
    return 24.0 * phys::PI * phys::PI * phys::PI * (n2 - 1) * (n2 - 1) /
           (l * l * l * l * N * N * (n2 + 2) * (n2 + 2)) * FK;
}

float solar_limb_darkening(double lambda_nm) {
    // Linear limb-darkening coefficient of the Sun (Allen's Astrophysical Quantities), smooth fit
    double u = 0.83 - 0.00110 * (lambda_nm - 400.0);
    return (float)(u < 0.35 ? 0.35 : (u > 0.95 ? 0.95 : u));
}

void xyz_to_srgb(const double xyz[3], double rgb[3]) {
    rgb[0] = 3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2];
    rgb[1] = -0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2];
    rgb[2] = 0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2];
}
}  // namespace spectrum
