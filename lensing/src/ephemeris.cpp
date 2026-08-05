#include "ephemeris.hpp"

#include <cmath>
#include <cstdio>

static constexpr double DEG = phys::PI / 180.0;

static double days_from_civil(int y, int m, int d) {
    // Howard Hinnant's algorithm: days since 1970-01-01
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097.0 + (double)doe - 719468.0;
}

bool parse_iso8601(const std::string& s, Epoch* out) {
    int Y = 0, M = 0, D = 0, h = 0, mi = 0;
    double sec = 0;
    int n = std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%lf", &Y, &M, &D, &h, &mi, &sec);
    if (n < 3) n = std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%lf", &Y, &M, &D, &h, &mi, &sec);
    if (n < 3 || M < 1 || M > 12 || D < 1 || D > 31) return false;
    double days = days_from_civil(Y, M, D);
    out->jd_utc = 2440587.5 + days + (h + mi / 60.0 + sec / 3600.0) / 24.0;
    return true;
}

std::string format_epoch(const Epoch& e) {
    double days = e.jd_utc - 2440587.5;
    double fd = std::floor(days);
    long z = (long)fd + 719468;
    long era = (z >= 0 ? z : z - 146096) / 146097;
    unsigned doe = (unsigned)(z - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long y = (long)yoe + era * 400;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned mp = (5 * doy + 2) / 153;
    unsigned d = doy - (153 * mp + 2) / 5 + 1;
    unsigned m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
    double frac = (days - fd) * 24.0;
    int hh = (int)frac;
    int mm = (int)((frac - hh) * 60.0);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04ld-%02u-%02u %02d:%02d UTC", y, m, d, hh, mm);
    return buf;
}

dvec3 radec_to_unit(double ra_deg, double dec_deg) {
    double a = ra_deg * DEG, d = dec_deg * DEG;
    return {std::cos(d) * std::cos(a), std::cos(d) * std::sin(a), std::sin(d)};
}

dvec3 precess_j2000_to_date(dvec3 v, double jd) {
    // IAU 1976 precession angles (Lieske et al. 1977)
    double T = (jd - 2451545.0) / 36525.0;
    double as = DEG / 3600.0;
    double zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T * T * T) * as;
    double z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T * T * T) * as;
    double th = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T * T * T) * as;
    double cz = std::cos(zeta), sz = std::sin(zeta), cZ = std::cos(z), sZ = std::sin(z);
    double ct = std::cos(th), st = std::sin(th);
    dmat3 P = {{cZ * ct * cz - sZ * sz, -cZ * ct * sz - sZ * cz, -cZ * st},
               {sZ * ct * cz + cZ * sz, -sZ * ct * sz + cZ * cz, -sZ * st},
               {st * cz, -st * sz, ct}};
    return mul(P, v);
}

double gmst_rad(double jd) {
    double n = jd - 2451545.0;
    double T = n / 36525.0;
    double g = 280.46061837 + 360.98564736629 * n + 0.000387933 * T * T - T * T * T / 38710000.0;
    g = std::fmod(g, 360.0);
    if (g < 0) g += 360.0;
    return g * DEG;
}

double obliquity_rad(double jd) {
    double n = jd - 2451545.0;
    return (23.439291 - 0.0000003563 * n) * DEG;
}

dvec3 sun_position_eci(double jd, double* dist_au) {
    double n = jd - 2451545.0;
    double L = std::fmod(280.460 + 0.9856474 * n, 360.0);
    double g = std::fmod(357.528 + 0.9856003 * n, 360.0) * DEG;
    double lam = (L + 1.915 * std::sin(g) + 0.020 * std::sin(2 * g)) * DEG;
    double R = 1.00014 - 0.01671 * std::cos(g) - 0.00014 * std::cos(2 * g);
    double eps = obliquity_rad(jd);
    if (dist_au) *dist_au = R;
    return {std::cos(lam), std::cos(eps) * std::sin(lam), std::sin(eps) * std::sin(lam)};
}
