/* GEOMETRY
- Near-Earth geometry in the LENS frame.
- Rays from the telescope reach Earth with transverse slopes of ~1e-8 rad, so across the few
  thousand kilometres around Earth they are parallel to -z to better than a centimetre.
- A ray is therefore fully described near Earth by its impact vector b = (bx, by):
    p(t) = (bx, by, -t),   t increasing away from the telescope.
*/

#pragma once
#include "device_util.cuh"

// Map a point into the space where the (inflated) ellipsoid becomes the unit sphere
DEVI dvec3 ell_scale(dvec3 p, double invA, double invC) {
    const dvec3 s = c_S.spin_L;
    return p * invA + s * (dot(p, s) * (invC - invA));
}

// Roots of the ray with the ellipsoid inflated by 'delta' metres (t0 <= t1)
DEVI bool ellipsoid_roots(double bx, double by, double delta, double* t0, double* t1) {
    double invA = 1.0 / (c_S.ell_a + delta), invC = 1.0 / (c_S.ell_c + delta);
    dvec3 qB = ell_scale(dvec3{bx, by, 0.0}, invA, invC);
    dvec3 qw = ell_scale(dvec3{0.0, 0.0, -1.0}, invA, invC);
    double A = dot(qw, qw), B = dot(qB, qw), C = dot(qB, qB) - 1.0;
    double disc = B * B - A * C;
    if (disc < 0.0) return false;
    double sq = sqrt(disc);
    *t0 = (-B - sq) / A;
    *t1 = (-B + sq) / A;
    return true;
}

/* altitude_d
- Altitude above the ellipsoid along the geocentric radial (difference to the geodetic altitude is <
  0.01% of the altitude).
*/
DEVI double altitude_d(dvec3 p) {
    double r = length(p);
    double s = dot(p, c_S.spin_L) / r;
    double a = c_S.ell_a, c = c_S.ell_c;
    double re = a * c / sqrt(c * c + (a - c) * (a + c) * s * s);
    return r - re;
}

DEVI float altitude_f(vec3 p, vec3 spin, float* r_out) {
    float r = length(p);
    float s = dot(p, spin) / r;
    float a = (float)c_S.ell_a, c = (float)c_S.ell_c;
    float re = a * c * rsqrtf(c * c + (a - c) * (a + c) * s * s);
    *r_out = r;
    return r - re;
}

// Point of minimum altitude along the ray (the "tangent point"). Returns that altitude
DEVI double tangent_point(double bx, double by, double* t_out) {
    double invA = 1.0 / c_S.ell_a, invC = 1.0 / c_S.ell_c;
    dvec3 qB = ell_scale(dvec3{bx, by, 0.0}, invA, invC);
    dvec3 qw = ell_scale(dvec3{0.0, 0.0, -1.0}, invA, invC);
    double t = -dot(qB, qw) / dot(qw, qw);
    const double d = 2000.0;
    for (int it = 0; it < 2; ++it) {
        double hm = altitude_d(dvec3{bx, by, -(t - d)});
        double h0 = altitude_d(dvec3{bx, by, -t});
        double hp = altitude_d(dvec3{bx, by, -(t + d)});
        double g = (hp - hm) / (2.0 * d), c2 = (hp - 2.0 * h0 + hm) / (d * d);
        if (c2 > 0.0) t -= g / c2;
    }
    *t_out = t;
    return altitude_d(dvec3{bx, by, -t});
}

// Outward ellipsoid normal (LENS frame) at p
DEVI dvec3 ellipsoid_normal(dvec3 p) {
    double ia2 = 1.0 / (c_S.ell_a * c_S.ell_a), ic2 = 1.0 / (c_S.ell_c * c_S.ell_c);
    dvec3 s = c_S.spin_L;
    return normalize(p * ia2 + s * (dot(p, s) * (ic2 - ia2)));
}

// Geodetic latitude / longitude (radians) of a LENS-frame point, plus its ECEF position
DEVI void lens_to_geo(dvec3 pL, float* lat, float* lon, dvec3* ecef) {
    dvec3 e = mul(c_S.M_LE, pL);
    double a = c_S.ell_a, c = c_S.ell_c;
    double e2 = 1.0 - (c * c) / (a * a);
    *lon = (float)atan2(e.y, e.x);
    *lat = (float)atan2(e.z, (1.0 - e2) * sqrt(e.x * e.x + e.y * e.y));
    if (ecef) *ecef = e;
}

DEVI float2 geo_to_uv(float lat, float lon) {
    return make_float2(lon * (0.5f / 3.14159265f) + 0.5f, 0.5f - lat * (1.0f / 3.14159265f));
}

// ECEF direction -> LENS frame (float)
DEVI vec3 ecef_dir_to_lens(vec3 d) {
    const dmat3& M = c_S.M_LE;
    return vec3{(float)(M.r0.x * d.x + M.r1.x * d.y + M.r2.x * d.z),
                (float)(M.r0.y * d.x + M.r1.y * d.y + M.r2.y * d.z),
                (float)(M.r0.z * d.x + M.r1.z * d.y + M.r2.z * d.z)};
}
