#include "atmosphere_model.hpp"

#include <cmath>

#include "scene.hpp"

namespace atmo {
namespace {

// Lower atmosphere (geopotential layers)
const double HB[8] = {0, 11000, 20000, 32000, 47000, 51000, 71000, 84852};
const double LB[7] = {-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002};
const double TB[8] = {288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946};
const double PB[8] = {101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420, 0.3734};

// Upper atmosphere, published 1976 densities (km, kg/m^3)
const double UZ[30] = {86,  90,  95,  100, 105, 110, 115, 120, 130, 140, 150, 160, 170, 180, 190,
                       200, 220, 240, 260, 280, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000};
const double UR[30] = {6.958e-6,  3.416e-6,  1.393e-6,  5.604e-7,  2.23e-7,   9.708e-8,
                       4.289e-8,  2.222e-8,  8.152e-9,  3.831e-9,  2.076e-9,  1.233e-9,
                       7.815e-10, 5.194e-10, 3.581e-10, 2.541e-10, 1.367e-10, 7.858e-11,
                       4.742e-11, 2.971e-11, 1.916e-11, 7.014e-12, 2.803e-12, 1.184e-12,
                       5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15};

double lower_density(double z) {
    const double r0 = 6356766.0, g0 = 9.80665, M = 0.0289644, Rstar = 8.31432;
    double H = r0 * z / (r0 + z);
    int b = 0;
    while (b < 6 && H >= HB[b + 1]) ++b;
    double T = TB[b] + LB[b] * (H - HB[b]);
    double P;
    if (LB[b] == 0.0)
        P = PB[b] * std::exp(-g0 * M * (H - HB[b]) / (Rstar * TB[b]));
    else
        P = PB[b] * std::pow(TB[b] / T, g0 * M / (Rstar * LB[b]));
    return P * M / (Rstar * T);
}

// Monotone (Fritsch-Carlson) cubic Hermite interpolation of ln(rho) on the upper table
double upper_lnrho(double zkm) {
    const int n = 30;
    if (zkm <= UZ[0]) return std::log(UR[0]);
    if (zkm >= UZ[n - 1]) {
        double s = (std::log(UR[n - 1]) - std::log(UR[n - 2])) / (UZ[n - 1] - UZ[n - 2]);
        return std::log(UR[n - 1]) + s * (zkm - UZ[n - 1]);
    }
    int i = 0;
    while (UZ[i + 1] < zkm) ++i;
    auto y = [](int j) { return std::log(UR[j]); };
    auto sec = [&](int j) { return (y(j + 1) - y(j)) / (UZ[j + 1] - UZ[j]); };
    auto tangent = [&](int j) {
        if (j == 0) return sec(0);
        if (j == n - 1) return sec(n - 2);
        double a = sec(j - 1), b = sec(j);
        if (a * b <= 0) return 0.0;
        return 2.0 / (1.0 / a + 1.0 / b);   // harmonic mean keeps monotonicity
    };
    double h = UZ[i + 1] - UZ[i], t = (zkm - UZ[i]) / h;
    double h00 = (1 + 2 * t) * (1 - t) * (1 - t), h10 = t * (1 - t) * (1 - t);
    double h01 = t * t * (3 - 2 * t), h11 = t * t * (t - 1);
    return h00 * y(i) + h10 * h * tangent(i) + h01 * y(i + 1) + h11 * h * tangent(i + 1);
}
}  // namespace

double density(double z) {
    if (z < 86000.0) return lower_density(z);
    return std::exp(upper_lnrho(z * 1e-3));
}

std::vector<float> lnrho_table(int* n_out) {
    const double zmax = 1000e3;
    int n = (int)((zmax - LNRHO_H0) / LNRHO_DH) + 2;
    std::vector<float> t(n);
    for (int i = 0; i < n; ++i) t[i] = (float)std::log(density(LNRHO_H0 + i * LNRHO_DH) / RHO0);
    *n_out = n;
    return t;
}
}  // namespace atmo
