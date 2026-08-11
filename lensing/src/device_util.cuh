/* DEVICE UTIL
- Device-side helpers: hashing RNG, procedural noise, table/LUT access.
- Included only from kernels.cu (single translation unit, so everything inlines without RDC).
*/

#pragma once
#include <cuda_runtime.h>
#include "scene.hpp"

// Defined here (not extern): the whole device program is one translation unit
__constant__ SceneGPU c_S;
__constant__ FrameGPU c_F;

#define DEVI __device__ __forceinline__

// Random numbers
DEVI uint32_t pcg(uint32_t v) {
    uint32_t s = v * 747796405u + 2891336453u;
    uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
DEVI float rnd(uint32_t& s) {
    s = pcg(s);
    return (s >> 8) * (1.0f / 16777216.0f);
}
DEVI float frac1(float x) { return x - floorf(x); }

// R2 low-discrepancy sequence (Roberts 2018), Cranley-Patterson rotated per pixel
DEVI void r2_sample(uint32_t n, float r0, float r1, float* x, float* y) {
    const double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
    *x = frac1(r0 + (float)fmod(n * a1, 1.0));
    *y = frac1(r1 + (float)fmod(n * a2, 1.0));
}

/* hash3
- Procedural noise (value noise / fBm / Worley), deterministic in space so that animations are
  temporally coherent.
*/
DEVI float hash3(int x, int y, int z, uint32_t seed) {
    uint32_t h = pcg((uint32_t)x * 73856093u ^ pcg((uint32_t)y * 19349663u ^ pcg((uint32_t)z * 83492791u ^ seed)));
    return (h >> 8) * (1.0f / 16777216.0f);
}

DEVI float vnoise(vec3 p, uint32_t seed) {
    float fx = floorf(p.x), fy = floorf(p.y), fz = floorf(p.z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float ux = p.x - fx, uy = p.y - fy, uz = p.z - fz;
    ux = ux * ux * ux * (ux * (ux * 6.f - 15.f) + 10.f);
    uy = uy * uy * uy * (uy * (uy * 6.f - 15.f) + 10.f);
    uz = uz * uz * uz * (uz * (uz * 6.f - 15.f) + 10.f);
    float c000 = hash3(ix, iy, iz, seed), c100 = hash3(ix + 1, iy, iz, seed);
    float c010 = hash3(ix, iy + 1, iz, seed), c110 = hash3(ix + 1, iy + 1, iz, seed);
    float c001 = hash3(ix, iy, iz + 1, seed), c101 = hash3(ix + 1, iy, iz + 1, seed);
    float c011 = hash3(ix, iy + 1, iz + 1, seed), c111 = hash3(ix + 1, iy + 1, iz + 1, seed);
    float x00 = lerpf(c000, c100, ux), x10 = lerpf(c010, c110, ux);
    float x01 = lerpf(c001, c101, ux), x11 = lerpf(c011, c111, ux);
    return 2.f * lerpf(lerpf(x00, x10, uy), lerpf(x01, x11, uy), uz) - 1.f;
}

DEVI float fbm(vec3 p, int octaves, uint32_t seed) {
    float s = 0.f, a = 0.5f, norm = 0.f;
    for (int i = 0; i < octaves; ++i) {
        s += a * vnoise(p, seed + 101u * i);
        norm += a;
        // Rotate + scale between octaves to break lattice alignment
        p = vec3{0.80f * p.x + 0.60f * p.z, p.y, -0.60f * p.x + 0.80f * p.z} * 2.03f;
        p = vec3{p.x, 0.80f * p.y - 0.60f * p.z, 0.60f * p.y + 0.80f * p.z};
        a *= 0.5f;
    }
    return s / norm;
}

// Worley noise: returns F2 - F1 (cell-edge distance), the classic granulation pattern
DEVI float worley_edge(vec3 p, uint32_t seed) {
    float fx = floorf(p.x), fy = floorf(p.y), fz = floorf(p.z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float f1 = 1e9f, f2 = 1e9f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int cx = ix + dx, cy = iy + dy, cz = iz + dz;
                vec3 q = {cx + hash3(cx, cy, cz, seed), cy + hash3(cx, cy, cz, seed ^ 0x9e3779b9u),
                          cz + hash3(cx, cy, cz, seed ^ 0x7f4a7c15u)};
                vec3 d = q - p;
                float dd = dot(d, d);
                if (dd < f1) { f2 = f1; f1 = dd; }
                else if (dd < f2) f2 = dd;
            }
    return sqrtf(f2) - sqrtf(f1);
}

// Density profiles (normalised to sea level)
DEVI float lnrho_at(float h) {
    float x = (h - LNRHO_H0) * (1.f / LNRHO_DH);
    x = fminf(fmaxf(x, 0.f), (float)(c_S.lnrho_n - 2) + 0.9999f);
    int i = (int)x;
    float f = x - i;
    float a = __ldg(c_S.lnrho + i), b = __ldg(c_S.lnrho + i + 1);
    return a + (b - a) * f;
}
DEVI float dens_R(float h) { return __expf(lnrho_at(h)); }
DEVI float dens_M(float h) { return __expf(-fmaxf(h, 0.f) / c_S.mie_H); }
DEVI float dens_O(float h) { return fmaxf(0.f, 1.f - fabsf(h - 25000.f) * (1.f / 15000.f)); }

// Double-precision variant with derivative d(ln rho)/dh, used to build the refraction tables
__device__ inline double lnrho_d(double h, double* dlnrho) {
    double x = (h - LNRHO_H0) / LNRHO_DH;
    x = fmin(fmax(x, 0.0), (double)(c_S.lnrho_n - 2) + 0.999999);
    int i = (int)x;
    double f = x - i;
    double a = c_S.lnrho[i], b = c_S.lnrho[i + 1];
    if (dlnrho) *dlnrho = (b - a) / LNRHO_DH;
    return a + (b - a) * f;
}

// Phase functions
DEVI float phase_rayleigh(float mu) { return 3.f / (16.f * 3.14159265f) * (1.f + mu * mu); }
DEVI float phase_mie(float mu, float g) {
    float g2 = g * g;
    float k = 3.f / (8.f * 3.14159265f) * (1.f - g2) / (2.f + g2);
    return k * (1.f + mu * mu) / powf(fmaxf(1.f + g2 - 2.f * g * mu, 1e-6f), 1.5f);
}

// Bruneton transmittance LUT parameterisation (r, mu) <-> (u, v)
DEVI float tc_from_unit(float x, int n) { return 0.5f / n + x * (1.f - 1.f / n); }
DEVI float unit_from_tc(float u, int n) { return (u - 0.5f / n) / (1.f - 1.f / n); }

DEVI float2 trans_uv(float r, float mu) {
    const float Rb = c_S.R_lut, Rt = c_S.R_lut + c_S.h_top;
    float H = sqrtf((Rt - Rb) * (Rt + Rb));
    float rho = sqrtf(fmaxf((r - Rb) * (r + Rb), 0.f));
    float disc = (Rt - r) * (Rt + r) + r * r * mu * mu;
    float d = fmaxf(-r * mu + sqrtf(fmaxf(disc, 0.f)), 0.f);
    float dmin = Rt - r, dmax = rho + H;
    float xmu = (dmax > dmin) ? (d - dmin) / (dmax - dmin) : 0.f;
    return make_float2(tc_from_unit(saturate(xmu), TRANS_W), tc_from_unit(saturate(rho / H), TRANS_H));
}

DEVI void lut4(tex_handle t, float u, float v, float out[NB]) {
#pragma unroll
    for (int l = 0; l < NB / 4; ++l) {
        float4 q = tex2DLayered<float4>((cudaTextureObject_t)t, u, v, l);
        out[4 * l + 0] = q.x; out[4 * l + 1] = q.y; out[4 * l + 2] = q.z; out[4 * l + 3] = q.w;
    }
}

/* SUN_ANG_RADIUS
- Transmittance from radius r to the Sun at cosine zenith mu_s, including Earth's shadow with a soft
  edge the size of the solar disk.
*/
constexpr float SUN_ANG_RADIUS = 0.004652f;
DEVI float sun_visibility(float r, float mu_s) {
    float sin_h = fminf(c_S.R_lut / r, 1.f);
    float cos_h = -sqrtf(fmaxf(1.f - sin_h * sin_h, 0.f));
    float w = sin_h * SUN_ANG_RADIUS;
    return smoothstepf(-w, w, mu_s - cos_h);
}
DEVI void sun_transmittance(float r, float mu_s, float out[NB]) {
    float vis = sun_visibility(r, mu_s);
    if (vis <= 0.f) {
#pragma unroll
        for (int k = 0; k < NB; ++k) out[k] = 0.f;
        return;
    }
    float2 uv = trans_uv(r, mu_s);
    lut4(c_S.trans_tex, uv.x, uv.y, out);
#pragma unroll
    for (int k = 0; k < NB; ++k) out[k] *= vis;
}

DEVI void ms_lookup(float h, float mu_s, float out[NB]) {
    float u = tc_from_unit(saturate(mu_s * 0.5f + 0.5f), MS_N);
    float v = tc_from_unit(saturate(h / c_S.h_top), MS_N);
    lut4(c_S.ms_tex, u, v, out);
}

DEVI void sky_irradiance(float mu_s, float out[NB]) {
    float x = saturate(mu_s * 0.5f + 0.5f) * (SKY_N - 1);
    int i = min((int)x, SKY_N - 2);
    float f = x - i;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        float a = __ldg(c_S.sky_irr + i * NB + k), b = __ldg(c_S.sky_irr + (i + 1) * NB + k);
        out[k] = a + (b - a) * f;
    }
}

// Refraction tables, indexed by apparent (straight-line) tangent altitude
DEVI float refr_x(float h_app) { return h_app / c_S.refr_dh; }
DEVI float refr_lookup(const float* t, int k, float x) {
    x = fminf(fmaxf(x, 0.f), (float)(c_S.refr_n - 2) + 0.9999f);
    int i = (int)x;
    float f = x - i;
    const float* p = t + (size_t)k * c_S.refr_n + i;
    float a = __ldg(p), b = __ldg(p + 1);
    return a + (b - a) * f;
}
