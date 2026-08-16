/* TRACE
- The lens itself: one camera sample -> spectral radiance.
- Gravity (weak field, thin lens, exact to O(Rs/b) ~ 1e-9 relative):
  monopole     dk = -2 Rs b / |b|^2
  oblateness   dk = 2 Q (u + i w)^3 / |b|^6,  Q = Rs J2 Req^2 sin^2(beta_s), (u,w) relative to
               the projected spin axis -- the quadrupole that turns the focal point into an
               astroid caustic a few km across.
  solar tide   dk = T b  (differential deflection by the Sun across the Earth lens)
- Atmosphere: the refraction angle, the true tangent altitude and the optical depth along the bent
  path are tabulated per band as functions of the apparent tangent altitude (computed with Bouguer's
  invariant in double precision); the in-scattered light is ray-marched.
*/

#pragma once
#include "sources.cuh"

DEVI dvec2 grav_deflection(double bx, double by) {
    double b2 = bx * bx + by * by;
    double k = -2.0 * c_S.Rs / b2;
    dvec2 a = {k * bx, k * by};
    if (c_S.use_j2) {
        double ex = c_S.j2_ex, ey = c_S.j2_ey;
        double u = bx * ex + by * ey, w = -bx * ey + by * ex;
        double re = u * u * u - 3.0 * u * w * w, im = 3.0 * u * u * w - w * w * w;
        double s = 2.0 * c_S.j2_Q / (b2 * b2 * b2);
        double du = s * re, dw = s * im;
        a.x += du * ex - dw * ey;
        a.y += du * ey + dw * ex;
    }
    if (c_S.use_tide) {
        a.x += c_S.tide[0] * bx + c_S.tide[1] * by;
        a.y += c_S.tide[2] * bx + c_S.tide[3] * by;
    }
    return a;
}

/* turbulence
- Frozen density perturbations of the upper atmosphere (gravity waves): a few percent of structure
  on ~100 km horizontal / few km vertical scales, fixed in Earth's frame so that the refracted ring
  evolves coherently as the telescope moves.
*/
DEVI float turbulence(dvec3 pL, float h) {
    if (c_S.turb_amp <= 0.f) return 1.f;
    dvec3 e = mul(c_S.M_LE, pL);
    float ih = 1.f / c_S.turb_lh;
    vec3 q = {(float)e.x * ih, (float)e.y * ih, (float)e.z * ih};
    float v = h / c_S.turb_lv;
    float n = fbm(vec3{q.x + v * 0.37f, q.y - v * 0.21f, q.z + v * 0.93f}, 4, 0xA77u);
    return 1.f + c_S.turb_amp * 2.f * n;
}

DEVI void shade_limb(double bx, double by, double gvx, double gvy, double t_tan, double h_app,
                     float jitter, float L[NB]) {
    float Ls[NB], Ts[NB];
#pragma unroll
    for (int k = 0; k < NB; ++k) { Ls[k] = 0.f; Ts[k] = 1.f; }
    if (h_app < c_S.h_top) {
        double ta, tb;
        if (ellipsoid_roots(bx, by, c_S.h_top + 1000.0, &ta, &tb))
            march((float)bx, (float)by, (float)ta, (float)tb, (float)t_tan, 1, 40, jitter, Ls, Ts);
    }
    dvec3 pt = {bx, by, -t_tan};
    dvec3 nrm = ellipsoid_normal(pt);
    double nl = rsqrt(nrm.x * nrm.x + nrm.y * nrm.y);
    double nx = nrm.x * nl, ny = nrm.y * nl;

    float x = refr_x((float)h_app);
    float h_true = refr_lookup(c_S.refr_htrue, 0, x);

    float fb = 0.f, Lc[NB];
    if (c_S.clouds && c_S.has_textures && h_true < c_S.cloud_alt + 2000.f) {
        float c = cloud_at(pt, 1.f);
        fb = c * smoothstepf(c_S.cloud_alt + 2000.f, c_S.cloud_alt - 1000.f, h_true);
        if (fb > 0.f) cloud_radiance(pt, Lc);
    }
    float pert = turbulence(pt, h_true);

#pragma unroll 1
    for (int k = 0; k < NB; ++k) {
        float bg = 0.f;
        if ((float)h_app >= c_S.h_ground_k[k]) {
            double alpha = exp((double)refr_lookup(c_S.refr_lna, k, x)) * pert;
            float tau = refr_lookup(c_S.refr_tau, k, x);
            float tmp[NB];
            background(bx, by, gvx - alpha * nx, gvy - alpha * ny, k, tmp);
            bg = tmp[k] * __expf(-tau);
        }
        L[k] = Ls[k] + (1.f - fb) * bg + (fb > 0.f ? fb * Lc[k] : 0.f);
    }
}

// Trace one sample at image angle (tx, ty) relative to the direction of Earth's centre
DEVI void trace_sample(double tx, double ty, float jitter, float L[NB]) {
    const double D = c_F.D;
    double bx = D * tx, by = D * ty;                       // impact vector at the lens plane
    double vx = tx - c_F.rho_x / D, vy = ty - c_F.rho_y / D; // ray slope w.r.t. the focal line
    dvec2 ag = grav_deflection(bx, by);
    double gvx = vx + ag.x, gvy = vy + ag.y;
    double b2 = bx * bx + by * by;
    double rlim = c_S.ell_a + c_S.h_top_refr;
    if (b2 > rlim * rlim) {
        background(bx, by, gvx, gvy, -1, L);
        return;
    }
    double t0, t1;
    if (ellipsoid_roots(bx, by, 0.0, &t0, &t1)) {
        shade_earth(bx, by, t0, jitter, L);
        return;
    }
    double tt;
    double h_app = tangent_point(bx, by, &tt);
    if (!c_S.atmosphere || h_app >= c_S.h_top_refr) {
        background(bx, by, gvx, gvy, -1, L);
        return;
    }
    shade_limb(bx, by, gvx, gvy, tt, h_app, jitter, L);
}
