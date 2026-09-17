/* DEVICE UTIL
- Device helpers: RNG, low-discrepancy sampling, procedural noise, table look-ups.
- Included only from render.cu (single translation unit).
*/

#pragma once
#include <cuda_runtime.h>
#include "scene.hpp"

__constant__ SceneGPU c_S;
__constant__ FrameGPU c_F;

#define DEVI __device__ __forceinline__

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
DEVI void r2_sample(uint32_t n, float r0, float r1, float* x, float* y) {
    const double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
    *x = frac1(r0 + (float)fmod(n * a1, 1.0));
    *y = frac1(r1 + (float)fmod(n * a2, 1.0));
}

// Gradient noise (Perlin-style, quintic fade)
DEVI float grad3(int ix, int iy, int iz, uint32_t seed, float fx, float fy, float fz) {
    uint32_t h = pcg((uint32_t)ix * 73856093u ^ pcg((uint32_t)iy * 19349663u ^ pcg((uint32_t)iz * 83492791u ^ seed)));
    // 12 edge directions of a cube
    switch (h % 12u) {
        case 0: return fx + fy;   case 1: return -fx + fy;  case 2: return fx - fy;   case 3: return -fx - fy;
        case 4: return fx + fz;   case 5: return -fx + fz;  case 6: return fx - fz;   case 7: return -fx - fz;
        case 8: return fy + fz;   case 9: return -fy + fz;  case 10: return fy - fz;  default: return -fy - fz;
    }
}
DEVI float gnoise(float x, float y, float z, uint32_t seed) {
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float ux = x - fx, uy = y - fy, uz = z - fz;
    float wx = ux * ux * ux * (ux * (ux * 6.f - 15.f) + 10.f);
    float wy = uy * uy * uy * (uy * (uy * 6.f - 15.f) + 10.f);
    float wz = uz * uz * uz * (uz * (uz * 6.f - 15.f) + 10.f);
    float n000 = grad3(ix, iy, iz, seed, ux, uy, uz);
    float n100 = grad3(ix + 1, iy, iz, seed, ux - 1, uy, uz);
    float n010 = grad3(ix, iy + 1, iz, seed, ux, uy - 1, uz);
    float n110 = grad3(ix + 1, iy + 1, iz, seed, ux - 1, uy - 1, uz);
    float n001 = grad3(ix, iy, iz + 1, seed, ux, uy, uz - 1);
    float n101 = grad3(ix + 1, iy, iz + 1, seed, ux - 1, uy, uz - 1);
    float n011 = grad3(ix, iy + 1, iz + 1, seed, ux, uy - 1, uz - 1);
    float n111 = grad3(ix + 1, iy + 1, iz + 1, seed, ux - 1, uy - 1, uz - 1);
    float x00 = lerpf(n000, n100, wx), x10 = lerpf(n010, n110, wx);
    float x01 = lerpf(n001, n101, wx), x11 = lerpf(n011, n111, wx);
    return lerpf(lerpf(x00, x10, wy), lerpf(x01, x11, wy), wz);   // ~[-1, 1]
}
// Normalised fBm with roughly unit variance
DEVI float fbm3(float x, float y, float z, int oct, uint32_t seed) {
    float s = 0.f, amp = 1.f, norm = 0.f;
    for (int i = 0; i < oct; ++i) {
        s += amp * gnoise(x, y, z, seed + 1013u * i);
        norm += amp * amp;
        float nx = 0.80f * x + 0.60f * z, nz = -0.60f * x + 0.80f * z;
        x = nx * 2.03f; y = y * 2.03f; z = nz * 2.03f;
        amp *= 0.55f;
    }
    return s * 2.2f / sqrtf(norm);
}

// Tables
DEVI float table_lerp(const float* t, int n, float x0, float dx, float x) {
    float f = (x - x0) / dx;
    f = fminf(fmaxf(f, 0.f), (float)(n - 2) + 0.9999f);
    int i = (int)f;
    f -= i;
    return __ldg(t + i) + (__ldg(t + i + 1) - __ldg(t + i)) * f;
}
// CIE XYZ (radiance-weighted, W m^-2 sr^-1) of a blackbody at temperature T
DEVI float3 blackbody_xyz(float T) {
    float f = (logf(fmaxf(T, 1.f)) - c_S.bb_l0) / c_S.bb_dl;
    f = fminf(fmaxf(f, 0.f), (float)(c_S.bb_n - 2) + 0.9999f);
    int i = (int)f;
    f -= i;
    const float* p = c_S.bb_xyz + 3 * i;
    return make_float3(__ldg(p) + (__ldg(p + 3) - __ldg(p)) * f, __ldg(p + 1) + (__ldg(p + 4) - __ldg(p + 1)) * f,
                       __ldg(p + 2) + (__ldg(p + 5) - __ldg(p + 2)) * f);
}
