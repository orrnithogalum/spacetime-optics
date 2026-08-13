/* ATMO
- Atmospheric radiative transfer along view rays: spectral single scattering from the Sun with
  Earth's shadow, plus the Hillaire (2020) multiple-scattering approximation, ozone absorption and
  the 557.7 nm OI airglow layer.
*/

#pragma once
#include "geometry.cuh"

/* airglow_emission
- Volume emission of the OI 557.7 nm green line (W m^-3 sr^-1 nm^-1, spread over its band): peak
  ~100 photons cm^-3 s^-1 at 97 km, ~10 km FWHM.
*/
DEVI float airglow_emission(float h) {
    float x = (h - 97000.f) * (1.f / 4500.f);
    return c_S.airglow * 1.13e-13f * __expf(-0.5f * x * x);
}

/* march
- March the straight segment p(t) = (ox, oy, -t), t in [t0, t1]. mode 0: samples concentrated
  towards t1 (ray ending on the ground / cloud deck) mode 1: samples concentrated around tm (ray
  grazing the limb at its tangent point) Accumulates in-scattered radiance into L and multiplies
  transmittance into T.
*/
DEVI void march(float ox, float oy, float t0, float t1, float tm, int mode, int n, float jitter,
                float L[NB], float T[NB]) {
    if (t1 <= t0) return;
    const vec3 spin = to_f(c_S.spin_L);
    const vec3 sun = c_S.sun_L;
    // Light propagates along -sun and is scattered towards the telescope (+z)
    const float mu = -sun.z;
    const float PR = phase_rayleigh(mu), PM = phase_mie(mu, c_S.mie_g);
    const float h_top = c_S.h_top;
    for (int i = 0; i < n; ++i) {
        float u = (i + jitter) / n, t, dt;
        if (mode == 0) {
            float w = 1.f - u;
            t = t0 + (t1 - t0) * (1.f - w * w);
            dt = 2.f * (t1 - t0) * w / n;
        } else {
            float w = 2.f * u - 1.f;
            if (w >= 0.f) { t = tm + (t1 - tm) * w * w; dt = 4.f * (t1 - tm) * w / n; }
            else          { t = tm - (tm - t0) * w * w; dt = -4.f * (tm - t0) * w / n; }
        }
        vec3 p = {ox, oy, -t};
        float r;
        float h = altitude_f(p, spin, &r);
        if (h > h_top) continue;
        float mu_s = dot(p, sun) / r;
        float rR = dens_R(h), rM = dens_M(h), rO = dens_O(h);
        float Ts[NB], ms[NB];
        float rl = c_S.R_lut + fmaxf(h, 0.f);
        sun_transmittance(rl, mu_s, Ts);
        ms_lookup(h, mu_s, ms);
        float glow = (c_S.airglow > 0.f) ? airglow_emission(h) : 0.f;
#pragma unroll
        for (int k = 0; k < NB; ++k) {
            float sR = c_S.sigR[k] * rR, sMs = c_S.sigMs[k] * rM;
            float st = sR + c_S.sigMe[k] * rM + c_S.sigO[k] * rO;
            float S = c_S.E_sun[k] * ((sR * PR + sMs * PM) * Ts[k] + (sR + sMs) * ms[k]);
            if (k == c_S.airglow_band) S += glow;
            float tr = __expf(-st * dt);
            float integ = st > 1e-12f ? (1.f - tr) / st : dt;
            L[k] += T[k] * S * integ;
            T[k] *= tr;
        }
    }
}
