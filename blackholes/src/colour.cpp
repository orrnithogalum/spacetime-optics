#include "colour.hpp"

#include <cmath>

namespace colour {
namespace {
// CIE 1931 2-degree colour matching functions, 380..780 nm, 10 nm steps
const double CX[41] = {0.001368, 0.004243, 0.014310, 0.043510, 0.134380, 0.283900, 0.348280, 0.336200, 0.290800,
                       0.195360, 0.095640, 0.032010, 0.004900, 0.009300, 0.063270, 0.165500, 0.290400, 0.433450,
                       0.594500, 0.762100, 0.916300, 1.026300, 1.062200, 1.002600, 0.854450, 0.642400, 0.447900,
                       0.283500, 0.164900, 0.087400, 0.046770, 0.022700, 0.011359, 0.005790, 0.002899, 0.001440,
                       0.000690, 0.000332, 0.000166, 0.000083, 0.000042};
const double CY[41] = {0.000039, 0.000120, 0.000396, 0.001210, 0.004000, 0.011600, 0.023000, 0.038000, 0.060000,
                       0.090980, 0.139020, 0.208020, 0.323000, 0.503000, 0.710000, 0.862000, 0.954000, 0.994950,
                       0.995000, 0.952000, 0.870000, 0.757000, 0.631000, 0.503000, 0.381000, 0.265000, 0.175000,
                       0.107000, 0.061000, 0.032000, 0.017000, 0.008210, 0.004102, 0.002091, 0.001047, 0.000520,
                       0.000249, 0.000120, 0.000060, 0.000030, 0.000015};
const double CZ[41] = {0.006450, 0.020050, 0.067850, 0.207400, 0.645600, 1.385600, 1.747060, 1.772110, 1.669200,
                       1.287640, 0.812950, 0.465180, 0.272000, 0.158200, 0.078250, 0.042160, 0.020300, 0.008750,
                       0.003900, 0.002100, 0.001650, 0.001100, 0.000800, 0.000340, 0.000190, 0.000050, 0.000020,
                       0,        0,        0,        0,        0,        0,        0,        0,        0,
                       0,        0,        0,        0,        0};
double cmf(const double* t, double lam) {
    double x = (lam - 380.0) / 10.0;
    if (x <= 0) return t[0];
    if (x >= 40) return t[40];
    int i = (int)x;
    double f = x - i;
    return t[i] * (1 - f) + t[i + 1] * f;
}
}  // namespace

double planck_lambda(double lambda_nm, double T) {
    const double h = 6.62607015e-34, c = 299792458.0, k = 1.380649e-23;
    double l = lambda_nm * 1e-9;
    return 2.0 * h * c * c / (l * l * l * l * l) / std::expm1(h * c / (l * k * T)) * 1e-9;
}

void blackbody_xyz(double T, double xyz[3]) {
    xyz[0] = xyz[1] = xyz[2] = 0;
    for (double lam = 380.5; lam < 780.0; lam += 1.0) {
        double b = planck_lambda(lam, T);
        xyz[0] += b * cmf(CX, lam);
        xyz[1] += b * cmf(CY, lam);
        xyz[2] += b * cmf(CZ, lam);
    }
}

std::vector<float> blackbody_xyz_table(float* l0, float* dl) {
    double a = std::log(BB_T0), b = std::log(BB_T1), d = (b - a) / (BB_N - 1);
    std::vector<float> t(3 * BB_N);
    for (int i = 0; i < BB_N; ++i) {
        double xyz[3];
        blackbody_xyz(std::exp(a + i * d), xyz);
        for (int c = 0; c < 3; ++c) t[3 * i + c] = (float)xyz[c];
    }
    *l0 = (float)a;
    *dl = (float)d;
    return t;
}

void xyz_to_srgb(const double xyz[3], double rgb[3]) {
    rgb[0] = 3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2];
    rgb[1] = -0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2];
    rgb[2] = 0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2];
}

void srgb_to_xyz(const double rgb[3], double xyz[3]) {
    xyz[0] = 0.4124564 * rgb[0] + 0.3575761 * rgb[1] + 0.1804375 * rgb[2];
    xyz[1] = 0.2126729 * rgb[0] + 0.7151522 * rgb[1] + 0.0721750 * rgb[2];
    xyz[2] = 0.0193339 * rgb[0] + 0.1191920 * rgb[1] + 0.9503041 * rgb[2];
}
}  // namespace colour
