/* TRACE
- One camera sample: backward null geodesic through Kerr spacetime, with
    - The thin disk hit at every equatorial crossing (dense-output root find)
    - Front-to-back radiative transfer through the hot flow
    - The celestial sphere for escaping rays, sampled with ray-footprint mip filtering
- The photon-ring *order* n = number of equatorial crossings before the emission (n = 0 direct
  image, n = 1 first ring, ...), accumulated as an emission-weighted average.
*/

#pragma once
#include "emission.cuh"

struct SkyGrad { float2 dx, dy; };

DEVI float hermite(float p0, float p1, float m0, float m1, float s) {
    float s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1;
}
DEVI double hermite_d(double p0, double p1, double m0, double m1, double s) {
    double s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1;
}

// Galactic texture coordinates of an escape direction (theta, phi at infinity)
DEVI float2 sky_uv(double th, double ph) {
    double st = sin(th);
    dvec3 d = {st * cos(ph), st * sin(ph), cos(th)};
    dvec3 g = mul(c_S.sky_rot, d);
    double l = atan2(g.y, g.x), b = asin(fmax(-1.0, fmin(1.0, g.z)));
    float u = (float)(0.5 - l * (0.5 / PI));
    u -= floorf(u);
    return make_float2(u, (float)(0.5 - b / PI));
}

DEVI float3 sky_radiance(float2 uv, const SkyGrad& gr, double Ecam) {
    float2 v = tex2DGrad<float2>((cudaTextureObject_t)c_S.sky_tex, uv.x, uv.y, gr.dx, gr.dy);
    if (v.x <= 0.f) return make_float3(0.f, 0.f, 0.f);
    float T = 1e4f * v.y / v.x;
    float g = (float)(1.0 / Ecam);   // nu_cam / nu_inf: blueshift deep in the well / Doppler
    float3 xyz = blackbody_xyz(g * T);
    float s = v.x * c_S.sky_scale;
    return make_float3(xyz.x * s, xyz.y * s, xyz.z * s);
}

/* escape_angles
- Asymptotic escape direction: integrate the remaining bending analytically to first order
  (dtheta/dr ~ 1/r^2 at large r, so the rest of the deflection is r * dtheta/dr).
*/
DEVI void escape_angles(const double* y, const double* k, double* th, double* ph) {
    double s = sqrt(fmax(1.0 - y[2] * y[2], 1e-30));
    double t0 = acos(fmax(-1.0, fmin(1.0, y[2])));
    double dthdr = (-y[3] / s) / y[1], dphdr = k[4] / y[1];
    *th = t0 + y[0] * dthdr;
    *ph = y[4] + y[0] * dphdr;
}

/* trace_ray
- Returns accumulated values: visible -> (X, Y, Z, order*Y); radio -> (I_nu, I_nu of the n = 1, 2
  and >= 3 photon-ring orders). *fate: +1 escaped, -1 captured, 0 step limit.  *uv_out: sky
  coordinates if escaped.
*/
DEVI float4 trace_ray(double x, double yv, const SkyGrad& gr, bool want_sky, int* fate, float2* uv_out) {
    const double a = c_S.a;
    dvec3 d = normalize(c_F.F + c_F.R * (c_F.cx + x) + c_F.U * (c_F.cy + yv));
    Photon p = camera_photon(a, c_F.kp, c_F.ph, c_F.t, d, c_F.beta);
    double* y = p.y;
    const double lam = p.lam, eta = p.eta, Ecam = p.E;
    double k1[6], yn[6], k7[6];
    geodesic_rhs(a, lam, eta, y, k1);
    double h = initial_step(y);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    double trans = 1.0;
    int ncross = 0;   // equatorial crossings: photon-ring order for the thin (equatorial) disk
    *fate = 0;
    const bool radio = c_S.mode == MODE_RADIO;
    for (int it = 0; it < c_S.max_steps; ++it) {
        // Cap the step inside emitting matter (volumetric sampling) / near the disk plane
        double hc = h;
        if (radio && c_S.flow_on && y[0] < c_S.r_flow_max) {
            double s = sqrt(fmax(1.0 - y[2] * y[2], 0.0));
            double cap = c_S.step_frac * fmin(y[0] / (fabs(y[1]) + 1e-30), fmax(s, 0.05) / (fabs(y[3]) + 1e-30));
            hc = fmin(hc, cap);
        } else if (c_S.disk_on) {
            hc = fmin(hc, 0.3 / (fabs(y[3]) + 1e-30));
        }
        double err = dp54_step(a, lam, eta, y, k1, hc, c_S.tol, yn, k7);
        if (!(err <= 1.0)) {
            h = hc * fmax(0.2, 0.9 * pow(fmax(err, 1e-30), -0.2));
            if (!(err == err)) h = hc * 0.1;   // NaN guard
            continue;
        }
        // Equatorial crossing
        if ((y[2] > 0.0) != (yn[2] > 0.0)) {
            ++ncross;
            if (c_S.disk_on) {
                double lo = 0.0, hi = 1.0, u0 = y[2], u1 = yn[2], m0 = y[3] * hc, m1 = yn[3] * hc;
                for (int b = 0; b < 40; ++b) {
                    double mid = 0.5 * (lo + hi);
                    double um = hermite_d(u0, u1, m0, m1, mid);
                    if ((um > 0.0) == (u0 > 0.0)) lo = mid;
                    else hi = mid;
                }
                double sc = 0.5 * (lo + hi);
                double rc = hermite_d(y[0], yn[0], y[1] * hc, yn[1] * hc, sc);
                double pc = hermite_d(y[4], yn[4], k1[4] * hc, k7[4] * hc, sc);
                double tc = hermite_d(y[5], yn[5], k1[5] * hc, k7[5] * hc, sc);
                float4 c = disk_shade(rc, pc, tc, lam, eta, Ecam);
                if (c.w > 0.f) {
                    float tf = (float)trans;
                    acc.x += tf * c.x;
                    acc.y += tf * c.y;
                    acc.z += tf * c.z;
                    acc.w += tf * c.y * (float)(ncross - 1);
                    trans *= 1.0 - c.w;
                    if (trans < 1e-4) { *fate = -1; return acc; }
                }
            }
        }
        // Hot flow
        if (radio && c_S.flow_on && yn[0] < c_S.r_flow_max) {
            double dI, dtau;
            flow_step(yn[0], yn[2], yn[4], yn[5], lam, Ecam, hc, &dI, &dtau);
            float c = (float)(trans * dI);
            // Total, then the light that has circled the hole n = 1, 2, >= 3 times
            acc.x += c;
            if (ncross == 1) acc.y += c;
            else if (ncross == 2) acc.z += c;
            else if (ncross >= 3) acc.w += c;
            trans *= exp(-dtau);
            if (trans < 1e-5) { *fate = -1; return acc; }
        }
        for (int i = 0; i < 6; ++i) { y[i] = yn[i]; k1[i] = k7[i]; }
        project_radial(a, lam, eta, y, k1);
        h = hc * fmin(5.0, fmax(0.2, 0.9 * pow(fmax(err, 1e-30), -0.2)));
        if (y[0] < c_S.r_stop) { *fate = -1; return acc; }
        if (y[0] > c_S.r_escape && y[1] > 0.0) {
            *fate = 1;
            double th, ph;
            escape_angles(y, k1, &th, &ph);
            float2 uv = sky_uv(th, ph);
            if (uv_out) *uv_out = uv;
            if (want_sky && c_S.sky_on && !radio) {
                float3 s = sky_radiance(uv, gr, Ecam);
                float tf = (float)trans;
                acc.x += tf * s.x;
                acc.y += tf * s.y;
                acc.z += tf * s.z;
            }
            return acc;
        }
    }
    return acc;
}
