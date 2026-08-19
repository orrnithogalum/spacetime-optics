/* KERNELS
- All device code (one translation unit so every helper inlines without relocatable device code)
  plus the host-side Renderer that owns GPU resources.
*/

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "atmosphere_model.hpp"
#include "font_atlas.h"
#include "renderer.hpp"
#include "trace.cuh"

#define CK(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,          \
                         cudaGetErrorString(e_));                                                 \
            std::abort();                                                                         \
        }                                                                                         \
    } while (0)
#define CF(x)                                                                                     \
    do {                                                                                          \
        cufftResult r_ = (x);                                                                     \
        if (r_ != CUFFT_SUCCESS) {                                                                \
            std::fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__);       \
            std::abort();                                                                         \
        }                                                                                         \
    } while (0)

struct ColorGPU { float wx[NB], wy[NB], wz[NB]; float white[3]; };
__constant__ ColorGPU c_C;

constexpr int CHUNK = 8;        // samples per work item in the adaptive pass
constexpr int SCAN_B = 1024;
constexpr float REFR_TOP = 300e3f;
constexpr float REFR_DH = 10.f;

// Look-up table construction
struct RefrParams { double Nk[NB]; };

/* k_build_refraction
- Refraction angle and optical depth of a ray grazing a spherical atmosphere, per band, as a
  function of its apparent (vacuum) tangent altitude h_app = p - R, with p = n(r_t) r_t the Bouguer
  invariant.
- Abel-type integrals with the substitution r = r_t + x^2 (removes the inverse-square-root
  singularity), composite Simpson, all in double precision.
*/
__global__ void k_build_refraction(RefrParams P, float* lna, float* tau, float* htrue, int n, float dh) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (j >= n) return;
    const double R = c_S.R_lut, Nk = P.Nk[k];
    const double h_app = j * (double)dh;
    double ht = h_app;
    for (int it = 0; it < 50; ++it) {
        double dl, rho = exp(lnrho_d(ht, &dl));
        double nn = 1.0 + Nk * rho;
        double f = nn * (R + ht) - R - h_app;
        double fp = nn + (R + ht) * Nk * rho * dl;
        double step = f / fp;
        ht -= step;
        if (fabs(step) < 1e-7) break;
    }
    double rt = R + ht;
    double p = (1.0 + Nk * exp(lnrho_d(ht, nullptr))) * rt;
    double h_up = fmin(ht + 400e3, 990e3);
    if (h_up < ht + 100.0) h_up = ht + 100.0;
    const int M = 6000;
    double X = sqrt(h_up - ht), dx = X / M;
    double sa = 0.0, st = 0.0;
    for (int i = 0; i <= M; ++i) {
        double x = i * dx, r = rt + x * x, h = r - R;
        double dl, rho = exp(lnrho_d(h, &dl));
        double nn = 1.0 + Nk * rho, dndr = Nk * rho * dl;
        double g;
        if (i == 0) g = 1.0 / sqrt(2.0 * p * fmax(nn + r * dndr, 1e-3));
        else g = x / sqrt(fmax((nn * r - p) * (nn * r + p), 1e-30));
        double sig = c_S.sigR[k] * rho + c_S.sigMe[k] * exp(-fmax(h, 0.0) / c_S.mie_H) +
                     c_S.sigO[k] * fmax(0.0, 1.0 - fabs(h - 25000.0) / 15000.0);
        double w = (i == 0 || i == M) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
        sa += w * (-4.0 * p * dndr / nn * g);
        st += w * (4.0 * sig * nn * r * g);
    }
    double alpha = sa * dx / 3.0, tv = st * dx / 3.0;
    lna[(size_t)k * n + j] = (float)log(fmax(alpha, 1e-35));
    tau[(size_t)k * n + j] = (float)tv;
    if (k == 7) htrue[j] = (float)ht;
}

__global__ void k_build_transmittance(float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= TRANS_W || y >= TRANS_H) return;
    double Rb = c_S.R_lut, Rt = Rb + c_S.h_top;
    double xmu = unit_from_tc((x + 0.5f) / TRANS_W, TRANS_W);
    double xr = unit_from_tc((y + 0.5f) / TRANS_H, TRANS_H);
    double H = sqrt(Rt * Rt - Rb * Rb), rho = H * xr, r = sqrt(rho * rho + Rb * Rb);
    double dmin = Rt - r, dmax = rho + H, d = dmin + xmu * (dmax - dmin);
    double mu = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = fmin(fmax(mu, -1.0), 1.0);
    const int S = 500;
    double ds = d / S, tv[NB];
    for (int k = 0; k < NB; ++k) tv[k] = 0.0;
    for (int i = 0; i < S; ++i) {
        double s = (i + 0.5) * ds;
        double h = sqrt(r * r + s * s + 2.0 * r * mu * s) - Rb;
        double rR = exp(lnrho_d(h, nullptr)), rM = exp(-fmax(h, 0.0) / c_S.mie_H);
        double rO = fmax(0.0, 1.0 - fabs(h - 25000.0) / 15000.0);
        for (int k = 0; k < NB; ++k) tv[k] += (c_S.sigR[k] * rR + c_S.sigMe[k] * rM + c_S.sigO[k] * rO) * ds;
    }
    for (int l = 0; l < NB / 4; ++l)
        for (int c = 0; c < 4; ++c)
            out[((size_t)(l * TRANS_H + y) * TRANS_W + x) * 4 + c] = (float)exp(-tv[4 * l + c]);
}

// Hillaire (2020) multiple-scattering transfer: Psi_ms = L_2nd / (1 - f_ms), per unit irradiance
__global__ void k_build_ms(float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= MS_N || y >= MS_N) return;
    float mu_s = 2.f * x / (MS_N - 1) - 1.f;
    float h = fmaxf((float)y / (MS_N - 1) * c_S.h_top, 20.f);
    float Rb = c_S.R_lut, Rt = Rb + c_S.h_top, r = Rb + h;
    vec3 sun = {sqrtf(fmaxf(1.f - mu_s * mu_s, 0.f)), 0.f, mu_s};
    const int NDIR = 64, NS = 20;
    float L2[NB], F[NB];
    for (int k = 0; k < NB; ++k) { L2[k] = 0.f; F[k] = 0.f; }
    for (int d = 0; d < NDIR; ++d) {
        float z = 1.f - 2.f * (d + 0.5f) / NDIR, ph = d * 2.39996323f, s = sqrtf(fmaxf(1.f - z * z, 0.f));
        vec3 w = {s * cosf(ph), s * sinf(ph), z};
        float b = r * w.z;
        float dg = b * b - (r - Rb) * (r + Rb);
        bool ground = dg >= 0.f && b < 0.f;
        float tmax = ground ? (-b - sqrtf(dg)) : (-b + sqrtf(fmaxf(b * b - (r - Rt) * (r + Rt), 0.f)));
        float dt = tmax / NS, tv[NB];
        for (int k = 0; k < NB; ++k) tv[k] = 0.f;
        for (int i = 0; i < NS; ++i) {
            float t = (i + 0.5f) * dt;
            vec3 pos = {w.x * t, w.y * t, r + w.z * t};
            float rr = length(pos), hh = rr - Rb, mus = dot(pos, sun) / rr;
            float Ts[NB];
            sun_transmittance(rr, mus, Ts);
            float rR = dens_R(hh), rM = dens_M(hh), rO = dens_O(hh);
            for (int k = 0; k < NB; ++k) {
                float ss = c_S.sigR[k] * rR + c_S.sigMs[k] * rM;
                float stt = c_S.sigR[k] * rR + c_S.sigMe[k] * rM + c_S.sigO[k] * rO;
                float Tm = __expf(-(tv[k] + 0.5f * stt * dt));
                L2[k] += Tm * ss * Ts[k] * (1.f / (4.f * 3.14159265f)) * dt;
                F[k] += Tm * ss * dt;
                tv[k] += stt * dt;
            }
        }
        if (ground) {
            vec3 pos = {w.x * tmax, w.y * tmax, r + w.z * tmax};
            float rr = length(pos), mug = dot(pos, sun) / rr;
            float Ts[NB];
            sun_transmittance(Rb, mug, Ts);
            for (int k = 0; k < NB; ++k)
                L2[k] += __expf(-tv[k]) * c_S.ground_albedo_ms * (1.f / 3.14159265f) * fmaxf(mug, 0.f) * Ts[k];
        }
    }
    for (int k = 0; k < NB; ++k) {
        float l2 = L2[k] / NDIR, f = F[k] / NDIR;
        float psi = l2 / fmaxf(1.f - f, 1e-3f);
        out[((size_t)((k / 4) * MS_N + y) * MS_N + x) * 4 + (k & 3)] = psi;
    }
}

// Sky irradiance on a horizontal ground surface vs cos(sun zenith), absolute units
__global__ void k_build_sky(float* out) {
    __shared__ float red[NB][256];
    int i = blockIdx.x;
    float mu_s = 2.f * i / (SKY_N - 1) - 1.f;
    vec3 sun = {sqrtf(fmaxf(1.f - mu_s * mu_s, 0.f)), 0.f, mu_s};
    float Rb = c_S.R_lut, Rt = Rb + c_S.h_top, r = Rb + 2.f;
    const int NDIR = 512, NS = 32;
    float acc[NB];
    for (int k = 0; k < NB; ++k) acc[k] = 0.f;
    for (int d = threadIdx.x; d < NDIR; d += blockDim.x) {
        float u = (d + 0.5f) / NDIR, st = sqrtf(u), ct = sqrtf(1.f - u), ph = d * 2.39996323f;
        vec3 w = {st * cosf(ph), st * sinf(ph), ct};
        float mu = dot(w, sun);
        float PR = phase_rayleigh(mu), PM = phase_mie(mu, c_S.mie_g);
        float b = r * w.z;
        float tmax = -b + sqrtf(fmaxf(b * b - (r - Rt) * (r + Rt), 0.f));
        float dt = tmax / NS, T[NB], L[NB];
        for (int k = 0; k < NB; ++k) { T[k] = 1.f; L[k] = 0.f; }
        for (int s = 0; s < NS; ++s) {
            float t = (s + 0.5f) * dt;
            vec3 pos = {w.x * t, w.y * t, r + w.z * t};
            float rr = length(pos), hh = rr - Rb, mus = dot(pos, sun) / rr;
            float Ts[NB], ms[NB];
            sun_transmittance(rr, mus, Ts);
            ms_lookup(hh, mus, ms);
            float rR = dens_R(hh), rM = dens_M(hh), rO = dens_O(hh);
            for (int k = 0; k < NB; ++k) {
                float sR = c_S.sigR[k] * rR, sM = c_S.sigMs[k] * rM;
                float stt = sR + c_S.sigMe[k] * rM + c_S.sigO[k] * rO;
                float S = c_S.E_sun[k] * ((sR * PR + sM * PM) * Ts[k] + (sR + sM) * ms[k]);
                float tr = __expf(-stt * dt);
                L[k] += T[k] * S * (stt > 1e-12f ? (1.f - tr) / stt : dt);
                T[k] *= tr;
            }
        }
        for (int k = 0; k < NB; ++k) acc[k] += L[k];
    }
    for (int k = 0; k < NB; ++k) red[k][threadIdx.x] = acc[k];
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off)
            for (int k = 0; k < NB; ++k) red[k][threadIdx.x] += red[k][threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x < NB) out[i * NB + threadIdx.x] = red[threadIdx.x][0] * 3.14159265f / NDIR;
}

// Texture mip generation
__device__ __forceinline__ float srgb_to_lin(float c) {
    return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}
__device__ __forceinline__ float lin_to_srgb(float c) {
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
}

__global__ void k_mip(cudaSurfaceObject_t src, int sw, int sh, cudaSurfaceObject_t dst, int dw, int dh, int srgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float a[4] = {0, 0, 0, 0};
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int sx = min(2 * x + dx, sw - 1), sy = min(2 * y + dy, sh - 1);
            uchar4 v;
            surf2Dread(&v, src, sx * 4, sy);
            float c[4] = {v.x / 255.f, v.y / 255.f, v.z / 255.f, v.w / 255.f};
            for (int i = 0; i < 4; ++i) a[i] += (srgb && i < 3) ? srgb_to_lin(c[i]) : c[i];
        }
    unsigned char o[4];
    for (int i = 0; i < 4; ++i) {
        float v = a[i] * 0.25f;
        if (srgb && i < 3) v = lin_to_srgb(v);
        o[i] = (unsigned char)fminf(fmaxf(v * 255.f + 0.5f, 0.f), 255.f);
    }
    surf2Dwrite(make_uchar4(o[0], o[1], o[2], o[3]), dst, x * 4, y);
}

// Rendering: pass 1 (uniform), adaptive refinement, resolve
__device__ __forceinline__ void pixel_seed(int p, uint32_t* h, float* r0, float* r1) {
    *h = pcg((uint32_t)p * 0x9E3779B9u ^ c_F.seed);
    *r0 = rnd(*h);
    *r1 = rnd(*h);
}

__device__ __forceinline__ void sample_pixel(int px, int py, int s, uint32_t h, float r0, float r1, float L[NB]) {
    float jx, jy;
    r2_sample((uint32_t)s, r0, r1, &jx, &jy);
    double tx = c_F.cx0 + (px + jx - 0.5 * c_F.W) * c_F.pix;
    double ty = c_F.cy0 + (0.5 * c_F.H - (py + jy)) * c_F.pix;
    uint32_t hs = pcg(h + (uint32_t)s * 0x632BE5ABu);
    trace_sample(tx, ty, (hs >> 8) * (1.f / 16777216.f), L);
}

__global__ void __launch_bounds__(128) k_pass1(float* __restrict__ accum, float* __restrict__ ystat,
                                               uint32_t* __restrict__ nsamp, double* __restrict__ frame_sum) {
    const int W = c_F.W, H = c_F.H;
    int px = blockIdx.x * 16 + (threadIdx.x & 15), py = blockIdx.y * 8 + (threadIdx.x >> 4);
    float ys = 0.f;
    if (px < W && py < H) {
        int p = py * W + px;
        size_t npix = (size_t)W * H;
        uint32_t h;
        float r0, r1;
        pixel_seed(p, &h, &r0, &r1);
        float acc[NB];
#pragma unroll
        for (int k = 0; k < NB; ++k) acc[k] = 0.f;
        float yq = 0.f;
        for (int s = 0; s < c_F.spp; ++s) {
            float L[NB];
            sample_pixel(px, py, s, h, r0, r1, L);
            float Y = 0.f;
#pragma unroll
            for (int k = 0; k < NB; ++k) { acc[k] += L[k]; Y += L[k] * c_C.wy[k]; }
            ys += Y;
            yq += Y * Y;
        }
#pragma unroll
        for (int k = 0; k < NB; ++k) accum[k * npix + p] = acc[k];
        ystat[p] = ys;
        ystat[npix + p] = yq;
        nsamp[p] = c_F.spp;
    }
    // Block reduction of the luminance sum (frame mean drives the adaptive noise floor)
    for (int off = 16; off > 0; off >>= 1) ys += __shfl_down_sync(0xffffffffu, ys, off);
    __shared__ float wsum[4];
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ys;
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(frame_sum, (double)(wsum[0] + wsum[1] + wsum[2] + wsum[3]));
}

__global__ void k_adapt(const float* __restrict__ ystat, const uint32_t* __restrict__ nsamp, int npix,
                        float yref, float target, int extra_max, uint32_t* __restrict__ extra) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix) return;
    float n = (float)nsamp[p];
    float mean = ystat[p] / n;
    float var = fmaxf(ystat[npix + p] / n - mean * mean, 0.f);
    float se = sqrtf(var / n);
    float denom = target * (mean + yref);
    float want = n * (se / denom) * (se / denom);
    float e = fminf(fmaxf(ceilf(want) - n, 0.f), (float)extra_max);
    extra[p] = (uint32_t)e;
}

// 3x3 dilation (catches thin features missed by every first-pass sample) -> chunk counts
__global__ void k_dilate(const uint32_t* __restrict__ extra, uint32_t* __restrict__ chunks, int W, int H, int extra_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uint32_t m = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = min(max(x + dx, 0), W - 1), yy = min(max(y + dy, 0), H - 1);
            m = max(m, extra[yy * W + xx]);
        }
    m = min(m, (uint32_t)extra_max);
    chunks[y * W + x] = (m + CHUNK - 1) / CHUNK;
}

__global__ void k_scan_block(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, uint32_t* __restrict__ sums, int n) {
    __shared__ uint32_t s[SCAN_B];
    int i = blockIdx.x * SCAN_B + threadIdx.x;
    uint32_t v = i < n ? in[i] : 0u;
    s[threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < SCAN_B; off <<= 1) {
        uint32_t t = (int)threadIdx.x >= off ? s[threadIdx.x - off] : 0u;
        __syncthreads();
        s[threadIdx.x] += t;
        __syncthreads();
    }
    if (i < n) out[i] = s[threadIdx.x] - v;
    if (threadIdx.x == SCAN_B - 1) sums[blockIdx.x] = s[SCAN_B - 1];
}

__global__ void k_scan_top(uint32_t* sums, int nb) {
    __shared__ uint32_t s[SCAN_B];
    __shared__ uint32_t carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < nb; base += SCAN_B) {
        int i = base + threadIdx.x;
        uint32_t v = i < nb ? sums[i] : 0u;
        s[threadIdx.x] = v;
        __syncthreads();
        for (int off = 1; off < SCAN_B; off <<= 1) {
            uint32_t t = (int)threadIdx.x >= off ? s[threadIdx.x - off] : 0u;
            __syncthreads();
            s[threadIdx.x] += t;
            __syncthreads();
        }
        if (i < nb) sums[i] = carry + s[threadIdx.x] - v;
        __syncthreads();
        if (threadIdx.x == SCAN_B - 1) carry += s[SCAN_B - 1];
        __syncthreads();
    }
}

__global__ void k_scan_add(uint32_t* out, const uint32_t* sums, int n) {
    int i = blockIdx.x * SCAN_B + threadIdx.x;
    if (i < n) out[i] += sums[blockIdx.x];
}

__global__ void __launch_bounds__(128) k_pass2(float* __restrict__ accum, uint32_t* __restrict__ nsamp,
                                               const uint32_t* __restrict__ offsets, int npix, uint32_t total) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    int lo = 0, hi = npix;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(offsets + mid) <= g) lo = mid + 1;
        else hi = mid;
    }
    int p = lo - 1;
    uint32_t j = g - __ldg(offsets + p);
    int px = p % c_F.W, py = p / c_F.W;
    uint32_t h;
    float r0, r1;
    pixel_seed(p, &h, &r0, &r1);
    float acc[NB];
#pragma unroll
    for (int k = 0; k < NB; ++k) acc[k] = 0.f;
    for (int i = 0; i < CHUNK; ++i) {
        float L[NB];
        sample_pixel(px, py, c_F.spp + (int)j * CHUNK + i, h, r0, r1, L);
#pragma unroll
        for (int k = 0; k < NB; ++k) acc[k] += L[k];
    }
#pragma unroll
    for (int k = 0; k < NB; ++k) atomicAdd(accum + (size_t)k * npix + p, acc[k]);
    atomicAdd(nsamp + p, (uint32_t)CHUNK);
}

__global__ void k_resolve(float* __restrict__ spec, const float* __restrict__ accum, const uint32_t* __restrict__ nsamp, int npix) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix) return;
    float inv = 1.f / (float)nsamp[p];
#pragma unroll
    for (int k = 0; k < NB; ++k) spec[(size_t)k * npix + p] = accum[(size_t)k * npix + p] * inv;
}

// Spectral radiance -> white-balanced linear sRGB in luminance units (cd/m^2)
__global__ void k_to_rgb(const float* __restrict__ spec, int npix, float* __restrict__ rgb) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix) return;
    float X = 0.f, Y = 0.f, Z = 0.f;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        float v = spec[(size_t)k * npix + p];
        X += v * c_C.wx[k]; Y += v * c_C.wy[k]; Z += v * c_C.wz[k];
    }
    float r = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float b = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
    rgb[3 * p + 0] = 683.f * r / c_C.white[0];
    rgb[3 * p + 1] = 683.f * g / c_C.white[1];
    rgb[3 * p + 2] = 683.f * b / c_C.white[2];
}

/* k_resample
- Telescope optics: resample to the detector, convolve with the polychromatic diffraction PSF of the
  actual pupil (central obstruction + spider vanes), optional photon/read noise.
*/
__global__ void k_resample(const float* __restrict__ src, int W, int H, double pi, float* __restrict__ dst,
                           int Wc, int Hc, double pc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, k = blockIdx.z;
    if (i >= Wc || j >= Hc) return;
    double s = pc / pi;
    double x0 = (i - 0.5 * Wc) * s + 0.5 * W, y0 = (j - 0.5 * Hc) * s + 0.5 * H;
    const float* S = src + (size_t)k * W * H;
    float v = 0.f;
    if (s >= 1.0) {
        double x1 = x0 + s, y1 = y0 + s;
        int ix0 = max((int)floor(x0), 0), ix1 = min((int)ceil(x1), W);
        int iy0 = max((int)floor(y0), 0), iy1 = min((int)ceil(y1), H);
        double acc = 0.0;
        for (int y = iy0; y < iy1; ++y) {
            double wy = fmin(y + 1.0, y1) - fmax((double)y, y0);
            for (int x = ix0; x < ix1; ++x) {
                double wx = fmin(x + 1.0, x1) - fmax((double)x, x0);
                acc += wx * wy * S[y * W + x];
            }
        }
        v = (float)(acc / (s * s));
    } else {
        double cx = x0 + 0.5 * s - 0.5, cy = y0 + 0.5 * s - 0.5;
        int ix = (int)floor(cx), iy = (int)floor(cy);
        float fx = (float)(cx - ix), fy = (float)(cy - iy);
        auto at = [&](int x, int y) { return (x < 0 || y < 0 || x >= W || y >= H) ? 0.f : S[y * W + x]; };
        v = (1 - fy) * ((1 - fx) * at(ix, iy) + fx * at(ix + 1, iy)) + fy * ((1 - fx) * at(ix, iy + 1) + fx * at(ix + 1, iy + 1));
    }
    dst[(size_t)k * Wc * Hc + j * Wc + i] = v;
}

__global__ void k_pupil(cufftComplex* buf, int N, double dx, double A, double eps, int spiders, double sw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    double xi = (i < N / 2 ? i : i - N) * dx, yj = (j < N / 2 ? j : j - N) * dx;
    float t = 0.f;
    for (int sy = 0; sy < 2; ++sy)
        for (int sx = 0; sx < 2; ++sx) {
            double x = xi + (sx - 0.5) * 0.5 * dx, y = yj + (sy - 0.5) * 0.5 * dx;
            double r = sqrt(x * x + y * y);
            bool in = r <= 0.5 * A && r >= 0.5 * A * eps;
            for (int m = 0; in && m < spiders; ++m) {
                double th = 1.5707963267948966 + 6.283185307179586 * m / spiders;
                double c = cos(th), s = sin(th);
                double along = x * c + y * s, across = -x * s + y * c;
                if (along > 0 && fabs(across) < 0.5 * sw * A) in = false;
            }
            t += in ? 0.25f : 0.f;
        }
    buf[(size_t)j * N + i] = make_float2(t, 0.f);
}

__global__ void k_abs2(cufftComplex* buf, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 v = buf[i];
    buf[i] = make_float2(v.x * v.x + v.y * v.y, 0.f);
}

__global__ void k_load_pad(cufftComplex* buf, int N, const float* __restrict__ src, int Wc, int Hc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    float v = (i < Wc && j < Hc) ? src[j * Wc + i] : 0.f;
    buf[(size_t)j * N + i] = make_float2(v, 0.f);
}

__global__ void k_mul_otf(cufftComplex* img, const cufftComplex* otf, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv = 1.f / otf[0].x;
    float2 a = img[i], b = otf[i];
    img[i] = make_float2((a.x * b.x - a.y * b.y) * inv, (a.x * b.y + a.y * b.x) * inv);
}

// The PSF sits at index (0,0) with wrap-around, so the convolved image is aligned with the input
__global__ void k_store_crop(const cufftComplex* buf, int N, float* dst, int Wc, int Hc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Wc || j >= Hc) return;
    dst[j * Wc + i] = fmaxf(buf[(size_t)j * N + i].x / ((float)N * (float)N), 0.f);
}

__global__ void k_noise(float* spec, int npix, int k, double conv, double read_sigma, uint32_t seed) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix) return;
    float* v = spec + (size_t)k * npix + p;
    double n = *v * conv;   // expected photo-electrons
    uint32_t h = pcg((uint32_t)p ^ pcg(seed + 977u * k));
    float u1 = fmaxf(rnd(h), 1e-7f), u2 = rnd(h);
    float g = sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
    double e;
    if (n > 30.0) e = n + sqrt(n) * g;
    else {   // Knuth Poisson sampling for faint pixels
        double Lm = exp(-n), pr = 1.0;
        int c = -1;
        do { ++c; pr *= rnd(h); } while (pr > Lm && c < 200);
        e = c;
    }
    float u3 = fmaxf(rnd(h), 1e-7f), u4 = rnd(h);
    e += read_sigma * sqrtf(-2.f * logf(u3)) * cosf(6.2831853f * u4);
    *v = (float)(e / conv);
}

// Display: filmic tone mapping and HUD
__device__ __forceinline__ float3 aces_fit(float3 c) {
    float3 v = {0.59719f * c.x + 0.35458f * c.y + 0.04823f * c.z,
                0.07600f * c.x + 0.90834f * c.y + 0.01566f * c.z,
                0.02840f * c.x + 0.13383f * c.y + 0.83777f * c.z};
    auto rrt = [](float x) {
        float a = x * (x + 0.0245786f) - 0.000090537f;
        float b = x * (0.983729f * x + 0.4329510f) + 0.238081f;
        return a / b;
    };
    v = make_float3(rrt(v.x), rrt(v.y), rrt(v.z));
    return make_float3(1.60475f * v.x - 0.53108f * v.y - 0.07367f * v.z,
                       -0.10208f * v.x + 1.10813f * v.y - 0.00605f * v.z,
                       -0.00327f * v.x - 0.07276f * v.y + 1.07602f * v.z);
}

__global__ void k_tonemap(const float* __restrict__ rgb, int npix, float exposure, uint32_t seed, uchar4* __restrict__ out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix) return;
    float3 c = make_float3(fmaxf(rgb[3 * p], 0.f) * exposure, fmaxf(rgb[3 * p + 1], 0.f) * exposure,
                           fmaxf(rgb[3 * p + 2], 0.f) * exposure);
    c = aces_fit(c);
    uint32_t h = pcg((uint32_t)p ^ seed);
    float d0 = rnd(h) - rnd(h), d1 = rnd(h) - rnd(h), d2 = rnd(h) - rnd(h);   // triangular dither
    auto q = [](float v, float d) {
        v = lin_to_srgb(fminf(fmaxf(v, 0.f), 1.f));
        return (unsigned char)fminf(fmaxf(v * 255.f + 0.5f + d, 0.f), 255.f);
    };
    out[p] = make_uchar4(q(c.x, d0), q(c.y, d1), q(c.z, d2), 255);
}

__global__ void k_hud_glyphs(uchar4* img, int W, int H, const HudGlyph* g, const uint8_t* atlas, float dx, float dy, int shadow) {
    HudGlyph G = g[blockIdx.x];
    int cw = (int)ceilf(FONT_CELL_W * G.scale) + 1, ch = (int)ceilf(FONT_CELL_H * G.scale) + 1;
    float r = (G.rgba & 255) / 255.f, gg = ((G.rgba >> 8) & 255) / 255.f, b = ((G.rgba >> 16) & 255) / 255.f;
    float alpha_mul = ((G.rgba >> 24) & 255) / 255.f;
    if (shadow) { r = gg = b = 0.f; alpha_mul *= 0.75f; }
    for (int idx = threadIdx.x; idx < cw * ch; idx += blockDim.x) {
        int ix = idx % cw, iy = idx / cw;
        int X = (int)floorf(G.x + dx) + ix, Y = (int)floorf(G.y + dy) + iy;
        if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
        float u = (ix + 0.5f) / G.scale - 0.5f, v = (iy + 0.5f) / G.scale - 0.5f;
        int u0 = (int)floorf(u), v0 = (int)floorf(v);
        float fu = u - u0, fv = v - v0;
        auto A = [&](int x, int y) -> float {
            if (x < 0 || y < 0 || x >= FONT_CELL_W || y >= FONT_CELL_H) return 0.f;
            return atlas[(G.glyph * FONT_CELL_H + y) * FONT_CELL_W + x] / 255.f;
        };
        float a = (1 - fv) * ((1 - fu) * A(u0, v0) + fu * A(u0 + 1, v0)) + fv * ((1 - fu) * A(u0, v0 + 1) + fu * A(u0 + 1, v0 + 1));
        a *= alpha_mul;
        if (a <= 0.f) continue;
        uchar4 o = img[Y * W + X];
        auto bl = [&](unsigned char d, float s) { return (unsigned char)fminf(d * (1 - a) + s * 255.f * a + 0.5f, 255.f); };
        img[Y * W + X] = make_uchar4(bl(o.x, r), bl(o.y, gg), bl(o.z, b), 255);
    }
}

__global__ void k_hud_rects(uchar4* img, int W, int H, const HudRect* rects) {
    HudRect R = rects[blockIdx.x];
    int w = R.x1 - R.x0, h = R.y1 - R.y0;
    float a = ((R.rgba >> 24) & 255) / 255.f;
    for (int idx = threadIdx.x; idx < w * h; idx += blockDim.x) {
        int X = R.x0 + idx % w, Y = R.y0 + idx / w;
        if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
        uchar4 o = img[Y * W + X];
        auto bl = [&](unsigned char d, int s) { return (unsigned char)fminf(d * (1 - a) + s * a + 0.5f, 255.f); };
        img[Y * W + X] = make_uchar4(bl(o.x, R.rgba & 255), bl(o.y, (R.rgba >> 8) & 255), bl(o.z, (R.rgba >> 16) & 255), 255);
    }
}

// Host side
namespace {
template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

cudaTextureObject_t make_layered_tex(cudaArray_t* arr_out, const float* dbuf, int w, int h, int layers) {
    cudaChannelFormatDesc fd = cudaCreateChannelDesc<float4>();
    cudaExtent ext = make_cudaExtent(w, h, layers);
    cudaArray_t arr;
    CK(cudaMalloc3DArray(&arr, &fd, ext, cudaArrayLayered));
    cudaMemcpy3DParms cp = {};
    cp.srcPtr = make_cudaPitchedPtr((void*)dbuf, w * sizeof(float4), w, h);
    cp.dstArray = arr;
    cp.extent = ext;
    cp.kind = cudaMemcpyDeviceToDevice;
    CK(cudaMemcpy3D(&cp));
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 1;
    cudaTextureObject_t t;
    CK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
    *arr_out = arr;
    return t;
}

cudaTextureObject_t make_mip_tex(cudaMipmappedArray_t* mip_out, const uint8_t* rgba, int w, int h, bool srgb, cudaStream_t st) {
    int levels = 1;
    while ((w >> levels) > 0 || (h >> levels) > 0) ++levels;
    cudaChannelFormatDesc fd = cudaCreateChannelDesc<uchar4>();
    cudaMipmappedArray_t mip;
    CK(cudaMallocMipmappedArray(&mip, &fd, make_cudaExtent(w, h, 0), levels, cudaArraySurfaceLoadStore));
    cudaArray_t l0;
    CK(cudaGetMipmappedArrayLevel(&l0, mip, 0));
    CK(cudaMemcpy2DToArray(l0, 0, 0, rgba, (size_t)w * 4, (size_t)w * 4, h, cudaMemcpyHostToDevice));
    for (int l = 1; l < levels; ++l) {
        cudaArray_t a, b;
        CK(cudaGetMipmappedArrayLevel(&a, mip, l - 1));
        CK(cudaGetMipmappedArrayLevel(&b, mip, l));
        int sw = std::max(w >> (l - 1), 1), sh = std::max(h >> (l - 1), 1);
        int dw = std::max(w >> l, 1), dh = std::max(h >> l, 1);
        cudaResourceDesc ra = {}, rb = {};
        ra.resType = rb.resType = cudaResourceTypeArray;
        ra.res.array.array = a;
        rb.res.array.array = b;
        cudaSurfaceObject_t sa, sb;
        CK(cudaCreateSurfaceObject(&sa, &ra));
        CK(cudaCreateSurfaceObject(&sb, &rb));
        dim3 bs(16, 16), gs((dw + 15) / 16, (dh + 15) / 16);
        k_mip<<<gs, bs, 0, st>>>(sa, sw, sh, sb, dw, dh, srgb ? 1 : 0);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(st));
        cudaDestroySurfaceObject(sa);
        cudaDestroySurfaceObject(sb);
    }
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeMipmappedArray;
    rd.res.mipmap.mipmap = mip;
    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeWrap;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeNormalizedFloat;
    td.normalizedCoords = 1;
    td.sRGB = srgb ? 1 : 0;
    td.mipmapFilterMode = cudaFilterModeLinear;
    td.maxMipmapLevelClamp = (float)(levels - 1);
    td.minMipmapLevelClamp = 0.f;
    td.maxAnisotropy = 8;
    cudaTextureObject_t t;
    CK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
    *mip_out = mip;
    return t;
}

cudaTextureObject_t make_image_tex(cudaArray_t* arr_out, const ImageTexData& img) {
    cudaChannelFormatDesc fd = cudaCreateChannelDesc<uchar4>();
    cudaArray_t arr;
    CK(cudaMallocArray(&arr, &fd, img.w, img.h));
    CK(cudaMemcpy2DToArray(arr, 0, 0, img.rgba.data(), (size_t)img.w * 4, (size_t)img.w * 4, img.h, cudaMemcpyHostToDevice));
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeBorder;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeNormalizedFloat;
    td.normalizedCoords = 1;
    td.sRGB = 1;
    cudaTextureObject_t t;
    CK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
    *arr_out = arr;
    return t;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int lensim_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

struct Renderer::Impl {
    int device = 0;
    cudaStream_t st = nullptr;
    SceneGPU S{};
    RenderSettings rs;
    ColorSettings cs;
    double Nk[NB];
    int W = 0, H = 0;
    size_t npix = 0;

    // Tables
    float *d_lnrho = nullptr, *d_refr_lna = nullptr, *d_refr_tau = nullptr, *d_refr_ht = nullptr, *d_sky = nullptr;
    cudaArray_t trans_arr = nullptr, ms_arr = nullptr;
    cudaMipmappedArray_t alb_mip = nullptr, aux_mip = nullptr;
    std::vector<cudaArray_t> img_arrs;

    // Frame buffers
    float *d_accum = nullptr, *d_ystat = nullptr, *d_spec = nullptr, *d_rgb = nullptr;
    uint32_t *d_nsamp = nullptr, *d_extra = nullptr, *d_chunks = nullptr, *d_offsets = nullptr, *d_bsums = nullptr;
    double* d_fsum = nullptr;

    // Telescope
    int N = 0;
    cufftHandle plan = 0;
    cufftComplex *d_f0 = nullptr, *d_f1 = nullptr;
    float *d_tspec = nullptr, *d_trgb = nullptr;

    // Display
    uint8_t* d_atlas = nullptr;
    uchar4* d_ldr = nullptr;
    float* d_hdr_in = nullptr;
    int ldr_cap = 0;
    HudGlyph* d_glyphs = nullptr;
    HudRect* d_rects = nullptr;
    uint32_t frame_counter = 0;

    void upload_scene() { CK(cudaMemcpyToSymbolAsync(c_S, &S, sizeof(SceneGPU), 0, cudaMemcpyHostToDevice, st)); }
    void hdr_from_device(const float* d_rgb_src, int w, int h, double pix, double percentile, HdrFrame* out);
    void telescope_view(const FrameGPU& f, double tpix, uint32_t seed);
};

Renderer::Renderer() : p_(new Impl) {}

Renderer::~Renderer() {
    if (!p_) return;
    cudaDeviceSynchronize();
    delete p_;
}

bool Renderer::init(int device, const SceneGPU& scene, const double refractivity_k[NB], const EarthTextureData* earth,
                    const std::vector<ImageTexData>& images, const RenderSettings& rs, const ColorSettings& cs,
                    std::string* err) {
    Impl& I = *p_;
    I.device = device;
    CK(cudaSetDevice(device));
    CK(cudaStreamCreateWithFlags(&I.st, cudaStreamNonBlocking));
    I.S = scene;
    I.rs = rs;
    I.cs = cs;
    for (int k = 0; k < NB; ++k) I.Nk[k] = refractivity_k[k];

    // Density table
    {
        int n = 0;
        std::vector<float> t = atmo::lnrho_table(&n);
        I.d_lnrho = dalloc<float>(n);
        CK(cudaMemcpy(I.d_lnrho, t.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        I.S.lnrho = I.d_lnrho;
        I.S.lnrho_n = n;
    }
    // Refraction tables
    {
        int n = (int)(REFR_TOP / REFR_DH) + 2;
        I.d_refr_lna = dalloc<float>((size_t)n * NB);
        I.d_refr_tau = dalloc<float>((size_t)n * NB);
        I.d_refr_ht = dalloc<float>(n);
        I.S.refr_lna = I.d_refr_lna;
        I.S.refr_tau = I.d_refr_tau;
        I.S.refr_htrue = I.d_refr_ht;
        I.S.refr_n = n;
        I.S.refr_dh = REFR_DH;
        I.S.h_top_refr = REFR_TOP - 2 * REFR_DH;
        I.upload_scene();
        RefrParams P;
        for (int k = 0; k < NB; ++k) P.Nk[k] = I.Nk[k];
        dim3 gs((n + 127) / 128, NB);
        k_build_refraction<<<gs, 128, 0, I.st>>>(P, I.d_refr_lna, I.d_refr_tau, I.d_refr_ht, n, REFR_DH);
        CK(cudaGetLastError());
    }
    // Scattering LUTs
    {
        float* d_tr = dalloc<float>((size_t)TRANS_W * TRANS_H * NB);
        dim3 bs(16, 16), gs((TRANS_W + 15) / 16, (TRANS_H + 15) / 16);
        k_build_transmittance<<<gs, bs, 0, I.st>>>(d_tr);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(I.st));
        I.S.trans_tex = make_layered_tex(&I.trans_arr, d_tr, TRANS_W, TRANS_H, NB / 4);
        cudaFree(d_tr);
        I.upload_scene();

        float* d_ms = dalloc<float>((size_t)MS_N * MS_N * NB);
        dim3 gm((MS_N + 15) / 16, (MS_N + 15) / 16);
        k_build_ms<<<gm, bs, 0, I.st>>>(d_ms);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(I.st));
        I.S.ms_tex = make_layered_tex(&I.ms_arr, d_ms, MS_N, MS_N, NB / 4);
        cudaFree(d_ms);
        I.upload_scene();

        I.d_sky = dalloc<float>((size_t)SKY_N * NB);
        k_build_sky<<<SKY_N, 256, 0, I.st>>>(I.d_sky);
        CK(cudaGetLastError());
        I.S.sky_irr = I.d_sky;
    }
    // Textures
    if (earth && earth->w > 0) {
        I.S.albedo_tex = make_mip_tex(&I.alb_mip, earth->albedo.data(), earth->w, earth->h, true, I.st);
        I.S.aux_tex = make_mip_tex(&I.aux_mip, earth->aux.data(), earth->w, earth->h, false, I.st);
        I.S.tex_w = earth->w;
        I.S.tex_h = earth->h;
        I.S.has_textures = 1;
    } else {
        I.S.has_textures = 0;
    }
    int img_i = 0;
    for (int i = 0; i < I.S.nsrc; ++i) {
        if (I.S.src[i].type != SRC_IMAGE) continue;
        if (img_i >= (int)images.size()) {
            if (err) *err = "image source without texture";
            return false;
        }
        cudaArray_t a;
        I.S.src[i].tex = make_image_tex(&a, images[img_i++]);
        I.img_arrs.push_back(a);
    }
    I.upload_scene();
    ColorGPU cg;
    std::memcpy(cg.wx, cs.wx, sizeof cg.wx);
    std::memcpy(cg.wy, cs.wy, sizeof cg.wy);
    std::memcpy(cg.wz, cs.wz, sizeof cg.wz);
    std::memcpy(cg.white, cs.white, sizeof cg.white);
    CK(cudaMemcpyToSymbolAsync(c_C, &cg, sizeof cg, 0, cudaMemcpyHostToDevice, I.st));

    // Frame buffers
    I.W = rs.W;
    I.H = rs.H;
    I.npix = (size_t)I.W * I.H;
    I.d_accum = dalloc<float>(I.npix * NB);
    I.d_spec = dalloc<float>(I.npix * NB);
    I.d_ystat = dalloc<float>(I.npix * 2);
    I.d_rgb = dalloc<float>(I.npix * 3);
    I.d_nsamp = dalloc<uint32_t>(I.npix);
    I.d_extra = dalloc<uint32_t>(I.npix);
    I.d_chunks = dalloc<uint32_t>(I.npix);
    I.d_offsets = dalloc<uint32_t>(I.npix);
    I.d_bsums = dalloc<uint32_t>((I.npix + SCAN_B - 1) / SCAN_B + 1);
    I.d_fsum = dalloc<double>(1);

    if (rs.tele.enabled) {
        int m = std::max(rs.tele.W, rs.tele.H) * 2;
        I.N = 256;
        while (I.N < m) I.N <<= 1;
        CF(cufftPlan2d(&I.plan, I.N, I.N, CUFFT_C2C));
        CF(cufftSetStream(I.plan, I.st));
        I.d_f0 = dalloc<cufftComplex>((size_t)I.N * I.N);
        I.d_f1 = dalloc<cufftComplex>((size_t)I.N * I.N);
        I.d_tspec = dalloc<float>((size_t)rs.tele.W * rs.tele.H * NB);
        I.d_trgb = dalloc<float>((size_t)rs.tele.W * rs.tele.H * 3);
    }
    I.d_atlas = dalloc<uint8_t>(sizeof(FONT_ATLAS));
    CK(cudaMemcpy(I.d_atlas, FONT_ATLAS, sizeof(FONT_ATLAS), cudaMemcpyHostToDevice));
    I.d_glyphs = dalloc<HudGlyph>(4096);
    I.d_rects = dalloc<HudRect>(256);
    CK(cudaStreamSynchronize(I.st));
    return true;
}

void Renderer::Impl::hdr_from_device(const float* d_src, int w, int h, double pix, double percentile, HdrFrame* out) {
    out->W = w;
    out->H = h;
    out->px.resize((size_t)w * h * 3);
    CK(cudaMemcpyAsync(out->px.data(), d_src, out->px.size() * sizeof(float), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    // Luminance statistics on the host (Grace cores are idle anyway)
    std::vector<float> lum;
    lum.reserve((size_t)w * h);
    double sum = 0.0, fl = 0.0, mx = 0.0;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        const float* c = &out->px[3 * i];
        double Y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        sum += Y;
        mx = std::max(mx, Y);
        if (Y > 0) lum.push_back((float)Y);
    }
    fl = sum * pix * pix;
    std::vector<float> sig;
    for (float v : lum)
        if (v > mx * 1e-6) sig.push_back(v);
    double pv = 0.0;
    if (!sig.empty()) {
        size_t idx = std::min(sig.size() - 1, (size_t)(percentile * (sig.size() - 1)));
        std::nth_element(sig.begin(), sig.begin() + idx, sig.end());
        pv = sig[idx];
    }
    out->lum_percentile = pv;
    out->lum_mean = sum / ((double)w * h);
    out->flux = fl;
}

void Renderer::Impl::telescope_view(const FrameGPU& f, double tpix, uint32_t seed) {
    const TelescopeSettings& T = rs.tele;
    int Wc = T.W, Hc = T.H;
    size_t nc = (size_t)Wc * Hc;
    dim3 bs(16, 16), gr((Wc + 15) / 16, (Hc + 15) / 16, NB);
    k_resample<<<gr, bs, 0, st>>>(d_spec, W, H, f.pix, d_tspec, Wc, Hc, tpix);
    CK(cudaGetLastError());
    size_t nn = (size_t)N * N;
    dim3 gN((N + 15) / 16, (N + 15) / 16);
    int t1 = 256;
    unsigned gl = (unsigned)((nn + t1 - 1) / t1);
    for (int k = 0; k < NB; ++k) {
        double lam = band_center_nm(k) * 1e-9;
        double dx = lam / (N * tpix);           // pupil-plane sampling for this band
        double samples = T.aperture / dx;       // pupil diameter in samples
        if (samples > N / 2) continue;          // PSF narrower than a detector pixel: leave as is
        k_pupil<<<gN, bs, 0, st>>>(d_f0, N, dx, T.aperture, T.obstruction, T.spiders, T.spider_width);
        CF(cufftExecC2C(plan, d_f0, d_f0, CUFFT_FORWARD));
        k_abs2<<<gl, t1, 0, st>>>(d_f0, nn);
        CF(cufftExecC2C(plan, d_f0, d_f0, CUFFT_FORWARD));   // -> OTF (unnormalised)
        k_load_pad<<<gN, bs, 0, st>>>(d_f1, N, d_tspec + (size_t)k * nc, Wc, Hc);
        CF(cufftExecC2C(plan, d_f1, d_f1, CUFFT_FORWARD));
        k_mul_otf<<<gl, t1, 0, st>>>(d_f1, d_f0, nn);
        CF(cufftExecC2C(plan, d_f1, d_f1, CUFFT_INVERSE));
        dim3 gc((Wc + 15) / 16, (Hc + 15) / 16);
        k_store_crop<<<gc, bs, 0, st>>>(d_f1, N, d_tspec + (size_t)k * nc, Wc, Hc);
    }
    if (T.noise) {
        double area = 3.14159265358979 * 0.25 * T.aperture * T.aperture * (1.0 - T.obstruction * T.obstruction);
        for (int k = 0; k < NB; ++k) {
            double lam = band_center_nm(k) * 1e-9;
            double ephot = phys::H_PLANCK * phys::C / lam;
            double conv = DLAMBDA_NM * tpix * tpix * area * T.exposure_s * T.qe / ephot;
            k_noise<<<(unsigned)((nc + 255) / 256), 256, 0, st>>>(d_tspec, (int)nc, k, conv, T.read_noise / std::sqrt((double)NB), seed);
        }
    }
    CK(cudaGetLastError());
    k_to_rgb<<<(unsigned)((nc + 255) / 256), 256, 0, st>>>(d_tspec, (int)nc, d_trgb);
    CK(cudaGetLastError());
}

void Renderer::render(const FrameGPU& f_in, double tele_pix, double percentile, HdrFrame* ideal, HdrFrame* tele, RenderStats* stats) {
    Impl& I = *p_;
    FrameGPU f = f_in;
    f.W = I.W;
    f.H = I.H;
    f.spp = I.rs.spp;
    CK(cudaMemcpyToSymbolAsync(c_F, &f, sizeof f, 0, cudaMemcpyHostToDevice, I.st));
    CK(cudaMemsetAsync(I.d_fsum, 0, sizeof(double), I.st));
    double t0 = now_ms();
    dim3 g1((I.W + 15) / 16, (I.H + 7) / 8);
    k_pass1<<<g1, 128, 0, I.st>>>(I.d_accum, I.d_ystat, I.d_nsamp, I.d_fsum);
    CK(cudaGetLastError());
    double fsum = 0;
    CK(cudaMemcpyAsync(&fsum, I.d_fsum, sizeof fsum, cudaMemcpyDeviceToHost, I.st));
    CK(cudaStreamSynchronize(I.st));
    double t1 = now_ms();
    long long samples = (long long)I.npix * f.spp;

    int extra_max = std::max(I.rs.spp_max - f.spp, 0);
    uint32_t total = 0;
    if (extra_max > 0) {
        float yref = (float)(0.02 * fsum / ((double)I.npix * f.spp) + 1e-30);
        int npix = (int)I.npix;
        k_adapt<<<(npix + 255) / 256, 256, 0, I.st>>>(I.d_ystat, I.d_nsamp, npix, yref, I.rs.target_err, extra_max, I.d_extra);
        dim3 bs(16, 16), gs((I.W + 15) / 16, (I.H + 15) / 16);
        k_dilate<<<gs, bs, 0, I.st>>>(I.d_extra, I.d_chunks, I.W, I.H, extra_max);
        int nb = (npix + SCAN_B - 1) / SCAN_B;
        k_scan_block<<<nb, SCAN_B, 0, I.st>>>(I.d_chunks, I.d_offsets, I.d_bsums, npix);
        k_scan_top<<<1, SCAN_B, 0, I.st>>>(I.d_bsums, nb);
        k_scan_add<<<nb, SCAN_B, 0, I.st>>>(I.d_offsets, I.d_bsums, npix);
        CK(cudaGetLastError());
        uint32_t last[2];
        CK(cudaMemcpyAsync(&last[0], I.d_offsets + npix - 1, 4, cudaMemcpyDeviceToHost, I.st));
        CK(cudaMemcpyAsync(&last[1], I.d_chunks + npix - 1, 4, cudaMemcpyDeviceToHost, I.st));
        CK(cudaStreamSynchronize(I.st));
        total = last[0] + last[1];
        if (total > 0) {
            k_pass2<<<(total + 127) / 128, 128, 0, I.st>>>(I.d_accum, I.d_nsamp, I.d_offsets, npix, total);
            CK(cudaGetLastError());
        }
        samples += (long long)total * CHUNK;
    }
    int npix = (int)I.npix;
    k_resolve<<<(npix + 255) / 256, 256, 0, I.st>>>(I.d_spec, I.d_accum, I.d_nsamp, npix);
    k_to_rgb<<<(npix + 255) / 256, 256, 0, I.st>>>(I.d_spec, npix, I.d_rgb);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(I.st));
    double t2 = now_ms();
    I.hdr_from_device(I.d_rgb, I.W, I.H, f.pix, percentile, ideal);
    if (I.rs.tele.enabled && tele) {
        I.telescope_view(f, tele_pix, f.seed ^ 0x5bd1e995u);
        I.hdr_from_device(I.d_trgb, I.rs.tele.W, I.rs.tele.H, tele_pix, percentile, tele);
    }
    double t3 = now_ms();
    if (stats) {
        stats->ms_pass1 = t1 - t0;
        stats->ms_pass2 = t2 - t1;
        stats->ms_post = t3 - t2;
        stats->samples = samples;
    }
}

void Renderer::update_geometry(const SceneGPU& g) {
    Impl& I = *p_;
    I.S.M_LE = g.M_LE;
    I.S.spin_L = g.spin_L;
    I.S.sun_L = g.sun_L;
    I.S.sun_pos_L = g.sun_pos_L;
    I.S.j2_Q = g.j2_Q;
    I.S.j2_ex = g.j2_ex;
    I.S.j2_ey = g.j2_ey;
    for (int i = 0; i < 4; ++i) I.S.tide[i] = g.tide[i];
    I.upload_scene();
}

void Renderer::tonemap(const HdrFrame& in, float exposure, const std::vector<HudGlyph>& glyphs,
                       const std::vector<HudRect>& rects, std::vector<uint8_t>* rgb8) {
    Impl& I = *p_;
    int n = in.W * in.H;
    if (n > I.ldr_cap) {
        if (I.d_ldr) cudaFree(I.d_ldr);
        if (I.d_hdr_in) cudaFree(I.d_hdr_in);
        I.d_ldr = dalloc<uchar4>(n);
        I.d_hdr_in = dalloc<float>((size_t)n * 3);
        I.ldr_cap = n;
    }
    CK(cudaMemcpyAsync(I.d_hdr_in, in.px.data(), (size_t)n * 3 * sizeof(float), cudaMemcpyHostToDevice, I.st));
    k_tonemap<<<(n + 255) / 256, 256, 0, I.st>>>(I.d_hdr_in, n, exposure, 0x1234567u + I.frame_counter++, I.d_ldr);
    if (!rects.empty()) {
        // One launch per layer, so that overlapping rectangles blend in a defined order
        std::vector<HudRect> sorted(rects);
        std::stable_sort(sorted.begin(), sorted.end(), [](const HudRect& a, const HudRect& b) { return a.layer < b.layer; });
        int nr = std::min((int)sorted.size(), 256);
        CK(cudaMemcpyAsync(I.d_rects, sorted.data(), nr * sizeof(HudRect), cudaMemcpyHostToDevice, I.st));
        for (int a = 0; a < nr;) {
            int b = a;
            while (b < nr && sorted[b].layer == sorted[a].layer) ++b;
            k_hud_rects<<<b - a, 256, 0, I.st>>>(I.d_ldr, in.W, in.H, I.d_rects + a);
            a = b;
        }
    }
    if (!glyphs.empty()) {
        int ng = std::min((int)glyphs.size(), 4096);
        CK(cudaMemcpyAsync(I.d_glyphs, glyphs.data(), ng * sizeof(HudGlyph), cudaMemcpyHostToDevice, I.st));
        float off = std::max(1.5f, in.H / 720.f);
        k_hud_glyphs<<<ng, 256, 0, I.st>>>(I.d_ldr, in.W, in.H, I.d_glyphs, I.d_atlas, off, off, 1);
        k_hud_glyphs<<<ng, 256, 0, I.st>>>(I.d_ldr, in.W, in.H, I.d_glyphs, I.d_atlas, 0.f, 0.f, 0);
    }
    CK(cudaGetLastError());
    std::vector<uchar4> tmp(n);
    CK(cudaMemcpyAsync(tmp.data(), I.d_ldr, (size_t)n * sizeof(uchar4), cudaMemcpyDeviceToHost, I.st));
    CK(cudaStreamSynchronize(I.st));
    rgb8->resize((size_t)n * 3);
    for (int i = 0; i < n; ++i) {
        (*rgb8)[3 * i] = tmp[i].x;
        (*rgb8)[3 * i + 1] = tmp[i].y;
        (*rgb8)[3 * i + 2] = tmp[i].z;
    }
}

void Renderer::report(FILE* out) {
    Impl& I = *p_;
    int n = I.S.refr_n;
    std::vector<float> lna((size_t)n * NB), tau((size_t)n * NB), ht(n), sky((size_t)SKY_N * NB);
    CK(cudaMemcpy(lna.data(), I.d_refr_lna, lna.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tau.data(), I.d_refr_tau, tau.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ht.data(), I.d_refr_ht, ht.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sky.data(), I.d_sky, sky.size() * 4, cudaMemcpyDeviceToHost));
    const int k = 7;
    std::fprintf(out, "Refraction of a grazing ray, band %d (%.1f nm), vs gravity (%.3e rad at Req):\n", k,
                 band_center_nm(k), 2.0 * I.S.Rs / I.S.ell_a);
    std::fprintf(out, "  h_app[km]  h_true[km]  alpha_refr[rad]  alpha_refr/alpha_grav  tau\n");
    const double hs[] = {2, 5, 10, 20, 30, 50, 70, 90, 100, 110, 120, 130, 150, 200, 250};
    for (double h : hs) {
        int j = (int)(h * 1e3 / REFR_DH);
        if (j >= n) continue;
        double a = std::exp((double)lna[(size_t)k * n + j]);
        double ag = 2.0 * I.S.Rs / (I.S.R_lut + h * 1e3);
        std::fprintf(out, "  %8.1f  %9.3f   %13.4e   %12.4e   %9.3e\n", h, ht[j] * 1e-3, a, a / ag, tau[(size_t)k * n + j]);
    }
    std::fprintf(out, "  ground-hit threshold (apparent altitude) band 0/%d/15: %.0f / %.0f / %.0f m\n", k,
                 I.S.h_ground_k[0], I.S.h_ground_k[k], I.S.h_ground_k[NB - 1]);
    std::fprintf(out, "Sky irradiance at the ground, sun at zenith, band %d: %.4f W/m^2/nm (direct %.4f)\n", k,
                 sky[(SKY_N - 1) * NB + k], I.S.E_sun[k]);
}
