/* RENDER
- Device kernels and the host-side Renderer (single translation unit).
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

constexpr int CHUNK = 8;
constexpr int SCAN_B = 1024;
constexpr int MAX_LEVELS = 9;

// Sampling passes
/* k_corners
- Escape direction of every pixel corner: gives each pixel its footprint on the sky, which drives
  anisotropic mip filtering of the star map (essential where lensing squeezes the whole sky into a
  thin ring around the shadow).
*/
__global__ void k_corners(float2* __restrict__ uv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    int W = c_F.W, H = c_F.H;
    if (x > W || y > H) return;
    double tx = (x - 0.5 * W) * c_F.pix, ty = (0.5 * H - y) * c_F.pix;
    SkyGrad g = {make_float2(0, 0), make_float2(0, 0)};
    int fate;
    float2 o = make_float2(-1.f, -1.f);
    trace_ray(tx, ty, g, false, &fate, &o);
    uv[y * (W + 1) + x] = fate == 1 ? o : make_float2(-1.f, -1.f);
}

__device__ __forceinline__ float wrapd(float d) { return d > 0.5f ? d - 1.f : (d < -0.5f ? d + 1.f : d); }

__device__ __forceinline__ SkyGrad pixel_grad(const float2* uv, int px, int py, float scale) {
    int W1 = c_F.W + 1;
    float2 a = uv[py * W1 + px], b = uv[py * W1 + px + 1], c = uv[(py + 1) * W1 + px], d = uv[(py + 1) * W1 + px + 1];
    SkyGrad g;
    if (a.x < 0.f || b.x < 0.f || c.x < 0.f || d.x < 0.f) {   // footprint straddles the shadow edge
        g.dx = make_float2(0.02f, 0.f);
        g.dy = make_float2(0.f, 0.02f);
        return g;
    }
    float dux = 0.5f * (wrapd(b.x - a.x) + wrapd(d.x - c.x)), dvx = 0.5f * (b.y - a.y + d.y - c.y);
    float duy = 0.5f * (wrapd(c.x - a.x) + wrapd(d.x - b.x)), dvy = 0.5f * (c.y - a.y + d.y - b.y);
    g.dx = make_float2(dux * scale, dvx * scale);
    g.dy = make_float2(duy * scale, dvy * scale);
    return g;
}

__device__ __forceinline__ void pixel_seed(int p, uint32_t* h, float* r0, float* r1) {
    *h = pcg((uint32_t)p * 0x9E3779B9u ^ c_F.seed);
    *r0 = rnd(*h);
    *r1 = rnd(*h);
}

__device__ __forceinline__ float4 sample_pixel(int px, int py, int s, float r0, float r1, const SkyGrad& g, int* lim) {
    float jx, jy;
    r2_sample((uint32_t)s, r0, r1, &jx, &jy);
    double tx = (px + jx - 0.5 * c_F.W) * c_F.pix;
    double ty = (0.5 * c_F.H - (py + jy)) * c_F.pix;
    int fate;
    float4 v = trace_ray(tx, ty, g, true, &fate, nullptr);
    if (fate == 0) ++*lim;
    return v;
}

__device__ __forceinline__ float sample_lum(float4 v) { return c_S.mode == MODE_RADIO ? v.x : v.y; }

__global__ void __launch_bounds__(128) k_pass1(float4* __restrict__ accum, float2* __restrict__ ystat,
                                               uint32_t* __restrict__ nsamp, const float2* __restrict__ uv,
                                               double* __restrict__ fsum, unsigned long long* __restrict__ nlim) {
    const int W = c_F.W, H = c_F.H;
    int px = blockIdx.x * 16 + (threadIdx.x & 15), py = blockIdx.y * 8 + (threadIdx.x >> 4);
    float ys = 0.f;
    int lim = 0;
    if (px < W && py < H) {
        int p = py * W + px;
        uint32_t h;
        float r0, r1;
        pixel_seed(p, &h, &r0, &r1);
        SkyGrad g = {make_float2(0, 0), make_float2(0, 0)};
        if (c_S.sky_on && uv) g = pixel_grad(uv, px, py, 0.7f);
        float4 a = make_float4(0, 0, 0, 0);
        float yq = 0.f;
        for (int s = 0; s < c_F.spp; ++s) {
            float4 v = sample_pixel(px, py, s, r0, r1, g, &lim);
            a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
            float Y = sample_lum(v);
            ys += Y;
            yq += Y * Y;
        }
        accum[p] = a;
        ystat[p] = make_float2(ys, yq);
        nsamp[p] = c_F.spp;
    }
    for (int off = 16; off > 0; off >>= 1) {
        ys += __shfl_down_sync(0xffffffffu, ys, off);
        lim += __shfl_down_sync(0xffffffffu, lim, off);
    }
    __shared__ float wsum[4];
    __shared__ int wlim[4];
    if ((threadIdx.x & 31) == 0) { wsum[threadIdx.x >> 5] = ys; wlim[threadIdx.x >> 5] = lim; }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(fsum, (double)(wsum[0] + wsum[1] + wsum[2] + wsum[3]));
        atomicAdd(nlim, (unsigned long long)(wlim[0] + wlim[1] + wlim[2] + wlim[3]));
    }
}

__global__ void k_adapt(const float2* __restrict__ ystat, const uint32_t* __restrict__ nsamp, int npix, float yref,
                        float target, int extra_max, uint32_t* __restrict__ extra) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix) return;
    float n = (float)nsamp[p];
    float2 s = ystat[p];
    float mean = s.x / n, var = fmaxf(s.y / n - mean * mean, 0.f);
    float se = sqrtf(var / n), den = target * (mean + yref);
    float want = n * (se / den) * (se / den);
    extra[p] = (uint32_t)fminf(fmaxf(ceilf(want) - n, 0.f), (float)extra_max);
}

__global__ void k_dilate(const uint32_t* __restrict__ extra, uint32_t* __restrict__ chunks, int W, int H, int extra_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uint32_t m = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            m = max(m, extra[min(max(y + dy, 0), H - 1) * W + min(max(x + dx, 0), W - 1)]);
    chunks[y * W + x] = (min(m, (uint32_t)extra_max) + CHUNK - 1) / CHUNK;
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

__global__ void __launch_bounds__(128) k_pass2(float4* __restrict__ accum, uint32_t* __restrict__ nsamp,
                                               const uint32_t* __restrict__ offsets, const float2* __restrict__ uv,
                                               int npix, uint32_t total) {
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
    SkyGrad gr = {make_float2(0, 0), make_float2(0, 0)};
    if (c_S.sky_on && uv) gr = pixel_grad(uv, px, py, 0.5f);
    float4 a = make_float4(0, 0, 0, 0);
    int lim = 0;
    for (int i = 0; i < CHUNK; ++i) {
        float4 v = sample_pixel(px, py, c_F.spp + (int)j * CHUNK + i, r0, r1, gr, &lim);
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    float* A = (float*)(accum + p);
    atomicAdd(A, a.x); atomicAdd(A + 1, a.y); atomicAdd(A + 2, a.z); atomicAdd(A + 3, a.w);
    atomicAdd(nsamp + p, (uint32_t)CHUNK);
}

// Mean per pixel; visible: XYZ -> linear sRGB (D65, "daylight" camera), order -> mean order
__global__ void k_resolve(float4* __restrict__ img, const float4* __restrict__ accum, const uint32_t* __restrict__ nsamp, int npix) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix) return;
    float inv = 1.f / (float)nsamp[p];
    float4 a = accum[p];
    a.x *= inv; a.y *= inv; a.z *= inv; a.w *= inv;
    if (c_S.mode == MODE_VISIBLE) {
        float X = a.x, Y = a.y, Z = a.z;
        float r = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
        float g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
        float b = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
        img[p] = make_float4(683.f * r, 683.f * g, 683.f * b, 683.f * a.w);   // keep order*Y weighting
    } else {
        img[p] = a;
    }
}

// Pyramid filters: camera glare (visible) and the interferometer beam (radio)
__global__ void k_down(const float4* __restrict__ src, int sw, int sh, float4* __restrict__ dst, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float4 a = make_float4(0, 0, 0, 0);
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            float4 v = src[min(2 * y + j, sh - 1) * sw + min(2 * x + i, sw - 1)];
            a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
        }
    dst[y * dw + x] = make_float4(0.25f * a.x, 0.25f * a.y, 0.25f * a.z, 0.25f * a.w);
}

__device__ __forceinline__ float4 bilinear(const float4* L, int w, int h, float x, float y) {
    x = fminf(fmaxf(x - 0.5f, 0.f), w - 1.001f);
    y = fminf(fmaxf(y - 0.5f, 0.f), h - 1.001f);
    int ix = (int)x, iy = (int)y;
    float fx = x - ix, fy = y - iy;
    int ix1 = min(ix + 1, w - 1), iy1 = min(iy + 1, h - 1);
    float4 a = L[iy * w + ix], b = L[iy * w + ix1], c = L[iy1 * w + ix], d = L[iy1 * w + ix1];
    float4 o;
    o.x = (1 - fy) * ((1 - fx) * a.x + fx * b.x) + fy * ((1 - fx) * c.x + fx * d.x);
    o.y = (1 - fy) * ((1 - fx) * a.y + fx * b.y) + fy * ((1 - fx) * c.y + fx * d.y);
    o.z = (1 - fy) * ((1 - fx) * a.z + fx * b.z) + fy * ((1 - fx) * c.z + fx * d.z);
    o.w = (1 - fy) * ((1 - fx) * a.w + fx * b.w) + fy * ((1 - fx) * c.w + fx * d.w);
    return o;
}

struct LevelPtrs { const float4* p[MAX_LEVELS]; int w[MAX_LEVELS], h[MAX_LEVELS]; int n; };

/* k_bloom
- Glare: energy-conserving mix of the image with a sum of ever wider blurs (power-law-like PSF wings
  of a real camera lens).
- Only colour channels; the order channel is untouched.
*/
__global__ void k_bloom(float4* __restrict__ img, int W, int H, LevelPtrs L, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float3 b = make_float3(0, 0, 0);
    float ws = 0.f;
    for (int k = 1; k < L.n; ++k) {
        float sx = (x + 0.5f) * L.w[k] / W, sy = (y + 0.5f) * L.h[k] / H;
        float4 v = bilinear(L.p[k], L.w[k], L.h[k], sx, sy);
        float w = 1.f / (float)k;
        b.x += w * v.x; b.y += w * v.y; b.z += w * v.z;
        ws += w;
    }
    float4 o = img[y * W + x];
    float s = strength;
    img[y * W + x] = make_float4((1 - s) * o.x + s * b.x / ws, (1 - s) * o.y + s * b.y / ws, (1 - s) * o.z + s * b.z / ws, o.w);
}

__global__ void k_gauss(const float4* __restrict__ src, float4* __restrict__ dst, int w, int h, float sigma, int horiz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int R = (int)ceilf(3.f * sigma);
    float4 a = make_float4(0, 0, 0, 0);
    float ws = 0.f;
    for (int i = -R; i <= R; ++i) {
        int xx = horiz ? min(max(x + i, 0), w - 1) : x, yy = horiz ? y : min(max(y + i, 0), h - 1);
        float wgt = __expf(-0.5f * i * i / (sigma * sigma));
        float4 v = src[yy * w + xx];
        a.x += wgt * v.x; a.y += wgt * v.y; a.z += wgt * v.z; a.w += wgt * v.w;
        ws += wgt;
    }
    dst[y * w + x] = make_float4(a.x / ws, a.y / ws, a.z / ws, a.w / ws);
}

__global__ void k_upsample(const float4* __restrict__ L, int lw, int lh, float4* __restrict__ img, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    img[y * W + x] = bilinear(L, lw, lh, (x + 0.5f) * lw / W, (y + 0.5f) * lh / H);
}

// Display
__device__ __forceinline__ float lin_to_srgb(float c) {
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
}
__device__ __forceinline__ float3 aces_fit(float3 c) {
    float3 v = {0.59719f * c.x + 0.35458f * c.y + 0.04823f * c.z, 0.07600f * c.x + 0.90834f * c.y + 0.01566f * c.z,
                0.02840f * c.x + 0.13383f * c.y + 0.83777f * c.z};
    auto rrt = [](float x) {
        return (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
    };
    v = make_float3(rrt(v.x), rrt(v.y), rrt(v.z));
    return make_float3(1.60475f * v.x - 0.53108f * v.y - 0.07367f * v.z, -0.10208f * v.x + 1.10813f * v.y - 0.00605f * v.z,
                       -0.00327f * v.x - 0.07276f * v.y + 1.07602f * v.z);
}

// Colormaps for radio brightness (input 0..1): afmhot (the EHT palette) and inferno
__device__ float3 colormap(int which, float x) {
    x = saturate(x);
    if (which == 0) return make_float3(saturate(2.f * x), saturate(2.f * x - 0.5f), saturate(2.f * x - 1.f));
    const float T[9][3] = {{0.0015f, 0.0005f, 0.014f}, {0.087f, 0.045f, 0.224f}, {0.258f, 0.039f, 0.406f},
                           {0.416f, 0.090f, 0.433f},   {0.578f, 0.148f, 0.404f}, {0.735f, 0.216f, 0.330f},
                           {0.865f, 0.317f, 0.226f},   {0.955f, 0.497f, 0.086f}, {0.988f, 0.998f, 0.645f}};
    float f = x * 8.f;
    int i = min((int)f, 7);
    f -= i;
    return make_float3(lerpf(T[i][0], T[i + 1][0], f), lerpf(T[i][1], T[i + 1][1], f), lerpf(T[i][2], T[i + 1][2], f));
}

// Photon-ring order tint: n = 0 untouched, n = 1 cyan, 2 magenta, 3 gold, 4+ lime
__device__ float3 order_tint(float n) {
    const float C[5][3] = {{1.f, 1.f, 1.f}, {0.25f, 0.9f, 1.f}, {1.f, 0.3f, 0.95f}, {1.f, 0.82f, 0.2f}, {0.55f, 1.f, 0.3f}};
    n = fminf(fmaxf(n, 0.f), 4.f);
    int i = min((int)n, 3);
    float f = n - i;
    return make_float3(lerpf(C[i][0], C[i + 1][0], f), lerpf(C[i][1], C[i + 1][1], f), lerpf(C[i][2], C[i + 1][2], f));
}

__global__ void k_tonemap(const float4* __restrict__ hdr, int W, int H, ToneParams tp, uchar4* __restrict__ out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int p = y * W + x;
    float4 v = hdr[p];
    float3 c;
    float order = 0.f;
    if (tp.mode == MODE_RADIO) {
        float I0 = fmaxf(v.x - v.y - v.z - v.w, 0.f);
        float Id = fmaxf(v.x - tp.hide_direct * I0, 0.f);   // displayed intensity
        float I = Id * tp.exposure;
        float xx = powf(I, tp.gamma);
        c = colormap(tp.colormap, xx);
        float over = fmaxf(xx - 1.f, 0.f);   // highlights burn to white
        c = make_float3(fminf(c.x + 0.25f * over, 1.f), fminf(c.y + 0.25f * over, 1.f), fminf(c.z + 0.25f * over, 1.f));
        if (tp.overlay > 0.f && Id > 0.f) {
            // Tint each pixel by its true mix of photon-ring orders
            float f1 = saturate(v.y / Id), f2 = saturate(v.z / Id), f3 = saturate(v.w / Id), f0 = saturate(1.f - f1 - f2 - f3);
            float lum = fmaxf(0.3f * c.x + 0.59f * c.y + 0.11f * c.z, 0.f);
            float3 t1 = order_tint(1.f), t2 = order_tint(2.f), t3 = order_tint(3.f);
            float L = saturate(lum * 1.3f + 0.05f);
            float3 m = make_float3(f0 * c.x + L * (f1 * t1.x + f2 * t2.x + f3 * t3.x), f0 * c.y + L * (f1 * t1.y + f2 * t2.y + f3 * t3.y),
                                   f0 * c.z + L * (f1 * t1.z + f2 * t2.z + f3 * t3.z));
            float k = tp.overlay;
            c = make_float3(lerpf(c.x, saturate(m.x), k), lerpf(c.y, saturate(m.y), k), lerpf(c.z, saturate(m.z), k));
        }
    } else {
        float Y = 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
        order = Y > 0.f ? v.w / Y : 0.f;
        c = aces_fit(make_float3(fmaxf(v.x, 0.f) * tp.exposure, fmaxf(v.y, 0.f) * tp.exposure, fmaxf(v.z, 0.f) * tp.exposure));
        c = make_float3(lin_to_srgb(saturate(c.x)), lin_to_srgb(saturate(c.y)), lin_to_srgb(saturate(c.z)));
    }
    if (tp.mode == MODE_VISIBLE && tp.overlay > 0.f && order > 0.05f) {
        float3 t = order_tint(order);
        float lum = 0.3f * c.x + 0.59f * c.y + 0.11f * c.z;
        float k = tp.overlay * smoothstepf(0.05f, 0.6f, order);
        c = make_float3(lerpf(c.x, saturate(lum * 1.4f * t.x), k), lerpf(c.y, saturate(lum * 1.4f * t.y), k),
                        lerpf(c.z, saturate(lum * 1.4f * t.z), k));
    }
    if (tp.vignette > 0.f) {
        float dx = (x + 0.5f) / W - 0.5f, dy = ((y + 0.5f) / H - 0.5f) * H / W;
        float v2 = 1.f - tp.vignette * smoothstepf(0.25f, 0.75f, sqrtf(dx * dx + dy * dy));
        c.x *= v2; c.y *= v2; c.z *= v2;
    }
    uint32_t hs = pcg((uint32_t)p ^ tp.seed);
    float gr = tp.grain * (rnd(hs) + rnd(hs) - 1.f);
    float d0 = rnd(hs) - rnd(hs), d1 = rnd(hs) - rnd(hs), d2 = rnd(hs) - rnd(hs);
    auto q = [&](float v, float d) {
        v = (v + gr) * tp.fade;
        return (unsigned char)fminf(fmaxf(v * 255.f + 0.5f + d, 0.f), 255.f);
    };
    out[p] = make_uchar4(q(c.x, d0), q(c.y, d1), q(c.z, d2), 255);
}

__global__ void k_hud_glyphs(uchar4* img, int W, int H, const HudGlyph* g, const uint8_t* atlas, float dx, float dy, int shadow) {
    HudGlyph G = g[blockIdx.x];
    int cw = (int)ceilf(FONT_CELL_W * G.scale) + 1, ch = (int)ceilf(FONT_CELL_H * G.scale) + 1;
    float r = (G.rgba & 255) / 255.f, gg = ((G.rgba >> 8) & 255) / 255.f, b = ((G.rgba >> 16) & 255) / 255.f;
    float am = ((G.rgba >> 24) & 255) / 255.f;
    if (shadow) { r = gg = b = 0.f; am *= 0.75f; }
    for (int idx = threadIdx.x; idx < cw * ch; idx += blockDim.x) {
        int ix = idx % cw, iy = idx / cw;
        int X = (int)floorf(G.x + dx) + ix, Y = (int)floorf(G.y + dy) + iy;
        if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
        float u = (ix + 0.5f) / G.scale - 0.5f, v = (iy + 0.5f) / G.scale - 0.5f;
        int u0 = (int)floorf(u), v0 = (int)floorf(v);
        float fu = u - u0, fv = v - v0;
        auto A = [&](int xx, int yy) -> float {
            if (xx < 0 || yy < 0 || xx >= FONT_CELL_W || yy >= FONT_CELL_H) return 0.f;
            return atlas[(G.glyph * FONT_CELL_H + yy) * FONT_CELL_W + xx] / 255.f;
        };
        float a = ((1 - fv) * ((1 - fu) * A(u0, v0) + fu * A(u0 + 1, v0)) + fv * ((1 - fu) * A(u0, v0 + 1) + fu * A(u0 + 1, v0 + 1))) * am;
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

// Baked flow turbulence
constexpr int TURB_NPHI = 512, TURB_NLR = 192, TURB_NZ = 48, TURB_NT = 24;
__global__ void k_bake_turb(cudaSurfaceObject_t surf, float delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, l = blockIdx.z;
    if (i >= TURB_NPHI || j >= TURB_NLR) return;
    double phi = (i + 0.5) / TURB_NPHI * 2.0 * PI;
    double R = exp(c_S.turb_lr0 + (j + 0.5) / TURB_NLR * (c_S.turb_lr1 - c_S.turb_lr0));
    double z = (-1.0 + 2.0 * (l + 0.5) / TURB_NZ) * c_S.turb_zmax * R;
    double r = sqrt(R * R + z * z);
    double t = retarded_reference(r) + delta;
    float n = sheared_noise(R, phi, z, t, c_S.flow_turb_scale, c_S.flow_tcorr, c_S.seed ^ 0x51f15eedu);
    float a = c_S.flow_turb_amp;
    surf3Dwrite(__expf(a * n - 0.5f * a * a), surf, i * 4, j, l);
}

// Sky texture mips (half2, box filter)
__global__ void k_mip_half2(cudaSurfaceObject_t src, int sw, int sh, cudaSurfaceObject_t dst, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float a = 0.f, b = 0.f;
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            ushort2 v;
            surf2Dread(&v, src, min(2 * x + i, sw - 1) * 4, min(2 * y + j, sh - 1));
            a += __half2float(__ushort_as_half(v.x));
            b += __half2float(__ushort_as_half(v.y));
        }
    surf2Dwrite(make_ushort2(__half_as_ushort(__float2half(0.25f * a)), __half_as_ushort(__float2half(0.25f * b))), dst, x * 4, y);
}

// Host side
namespace {
template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
dim3 grid2(int w, int h) { return dim3((w + 15) / 16, (h + 15) / 16); }
}  // namespace

int bh_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

struct Renderer::Impl {
    cudaStream_t st = nullptr;
    SceneGPU S{};
    RenderSettings rs;
    int W = 0, H = 0;
    size_t npix = 0;
    float *d_disk = nullptr, *d_k2 = nullptr, *d_bb = nullptr;
    cudaMipmappedArray_t sky_mip = nullptr;
    float4 *d_accum = nullptr, *d_img = nullptr, *d_tmp = nullptr;
    float4* d_lev[MAX_LEVELS] = {};
    int lw[MAX_LEVELS], lh[MAX_LEVELS], nlev = 0;
    float2 *d_ystat = nullptr, *d_uv = nullptr;
    uint32_t *d_nsamp = nullptr, *d_extra = nullptr, *d_chunks = nullptr, *d_offsets = nullptr, *d_bsums = nullptr;
    double* d_fsum = nullptr;
    unsigned long long* d_nlim = nullptr;
    cudaArray_t turb_arr[TURB_NT] = {};
    cudaSurfaceObject_t turb_surf[TURB_NT] = {};
    uint8_t* d_atlas = nullptr;
    uchar4* d_ldr = nullptr;
    HudGlyph* d_glyphs = nullptr;
    HudRect* d_rects = nullptr;

    void upload_scene() { CK(cudaMemcpyToSymbolAsync(c_S, &S, sizeof(SceneGPU), 0, cudaMemcpyHostToDevice, st)); }
    void trace(const FrameGPU& f, bool adaptive, RenderStats* stats);
    void radio_blur(double sigma);
    void bloom(float strength);
};

Renderer::Renderer() : p_(new Impl) {}
Renderer::~Renderer() {
    cudaDeviceSynchronize();
    delete p_;
}

bool Renderer::init(int device, const SceneGPU& scene, const SceneTables& T, const SkyTexture* sky,
                    const RenderSettings& rs, std::string* err) {
    Impl& I = *p_;
    CK(cudaSetDevice(device));
    CK(cudaStreamCreateWithFlags(&I.st, cudaStreamNonBlocking));
    I.S = scene;
    I.rs = rs;
    auto up = [&](const std::vector<float>& v) {
        float* d = dalloc<float>(v.size());
        CK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    };
    I.d_disk = up(T.disk_lnT);
    I.d_k2 = up(T.k2_ln);
    I.d_bb = up(T.bb_xyz);
    I.S.disk_lnT = I.d_disk;
    I.S.k2_lntab = I.d_k2;
    I.S.bb_xyz = I.d_bb;

    if (sky && sky->w > 0 && I.S.sky_on) {
        int w = sky->w, h = sky->h, levels = 1;
        while ((w >> levels) > 0 || (h >> levels) > 0) ++levels;
        cudaChannelFormatDesc fd = cudaCreateChannelDescHalf2();
        CK(cudaMallocMipmappedArray(&I.sky_mip, &fd, make_cudaExtent(w, h, 0), levels, cudaArraySurfaceLoadStore));
        cudaArray_t l0;
        CK(cudaGetMipmappedArrayLevel(&l0, I.sky_mip, 0));
        CK(cudaMemcpy2DToArray(l0, 0, 0, sky->data.data(), (size_t)w * 4, (size_t)w * 4, h, cudaMemcpyHostToDevice));
        for (int l = 1; l < levels; ++l) {
            cudaArray_t a, b;
            CK(cudaGetMipmappedArrayLevel(&a, I.sky_mip, l - 1));
            CK(cudaGetMipmappedArrayLevel(&b, I.sky_mip, l));
            cudaResourceDesc ra = {}, rb = {};
            ra.resType = rb.resType = cudaResourceTypeArray;
            ra.res.array.array = a;
            rb.res.array.array = b;
            cudaSurfaceObject_t sa, sb;
            CK(cudaCreateSurfaceObject(&sa, &ra));
            CK(cudaCreateSurfaceObject(&sb, &rb));
            int sw = std::max(w >> (l - 1), 1), sh = std::max(h >> (l - 1), 1), dw = std::max(w >> l, 1), dh = std::max(h >> l, 1);
            k_mip_half2<<<grid2(dw, dh), dim3(16, 16), 0, I.st>>>(sa, sw, sh, sb, dw, dh);
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(I.st));
            cudaDestroySurfaceObject(sa);
            cudaDestroySurfaceObject(sb);
        }
        cudaResourceDesc rd = {};
        rd.resType = cudaResourceTypeMipmappedArray;
        rd.res.mipmap.mipmap = I.sky_mip;
        cudaTextureDesc td = {};
        td.addressMode[0] = cudaAddressModeWrap;
        td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode = cudaFilterModeLinear;
        td.readMode = cudaReadModeElementType;
        td.normalizedCoords = 1;
        td.mipmapFilterMode = cudaFilterModeLinear;
        td.maxMipmapLevelClamp = (float)(levels - 1);
        td.maxAnisotropy = 16;
        cudaTextureObject_t t;
        CK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
        I.S.sky_tex = t;
    } else {
        I.S.sky_on = 0;
    }
    // Baked, time-sliced turbulence for the hot flow
    I.S.turb_nt = 0;
    if (I.S.flow_on && I.S.mode == MODE_RADIO && I.S.flow_turb_amp > 0.f) {
        cudaChannelFormatDesc fd = cudaCreateChannelDesc<float>();
        for (int k = 0; k < TURB_NT; ++k) {
            CK(cudaMalloc3DArray(&I.turb_arr[k], &fd, make_cudaExtent(TURB_NPHI, TURB_NLR, TURB_NZ), cudaArraySurfaceLoadStore));
            cudaResourceDesc rd = {};
            rd.resType = cudaResourceTypeArray;
            rd.res.array.array = I.turb_arr[k];
            CK(cudaCreateSurfaceObject(&I.turb_surf[k], &rd));
            cudaTextureDesc td = {};
            td.addressMode[0] = cudaAddressModeWrap;
            td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
            td.filterMode = cudaFilterModeLinear;
            td.readMode = cudaReadModeElementType;
            td.normalizedCoords = 1;
            cudaTextureObject_t t;
            CK(cudaCreateTextureObject(&t, &rd, &td, nullptr));
            I.S.turb_tex[k] = t;
        }
        I.S.turb_nt = TURB_NT;
        I.S.turb_lr0 = (float)std::log(0.8 * I.S.rh);
        I.S.turb_lr1 = (float)std::log(I.S.r_flow_max * 1.05);
        I.S.turb_zmax = (float)std::min(2.0, 5.0 * I.S.h_r);
        I.S.turb_d0 = -250.f;
        I.S.turb_dd = 260.f / (TURB_NT - 1);
    }
    I.upload_scene();

    I.W = rs.W;
    I.H = rs.H;
    I.npix = (size_t)I.W * I.H;
    I.d_accum = dalloc<float4>(I.npix);
    I.d_img = dalloc<float4>(I.npix);
    I.d_tmp = dalloc<float4>(I.npix);
    I.d_ystat = dalloc<float2>(I.npix);
    I.d_uv = dalloc<float2>((size_t)(I.W + 1) * (I.H + 1));
    I.d_nsamp = dalloc<uint32_t>(I.npix);
    I.d_extra = dalloc<uint32_t>(I.npix);
    I.d_chunks = dalloc<uint32_t>(I.npix);
    I.d_offsets = dalloc<uint32_t>(I.npix);
    I.d_bsums = dalloc<uint32_t>(I.npix / SCAN_B + 2);
    I.d_fsum = dalloc<double>(1);
    I.d_nlim = dalloc<unsigned long long>(1);
    int w = I.W, h = I.H;
    I.nlev = 0;
    while (I.nlev < MAX_LEVELS && w >= 2 && h >= 2) {
        I.lw[I.nlev] = w;
        I.lh[I.nlev] = h;
        I.d_lev[I.nlev] = I.nlev == 0 ? I.d_img : dalloc<float4>((size_t)w * h);
        ++I.nlev;
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
    I.d_atlas = dalloc<uint8_t>(sizeof(FONT_ATLAS));
    CK(cudaMemcpy(I.d_atlas, FONT_ATLAS, sizeof(FONT_ATLAS), cudaMemcpyHostToDevice));
    I.d_ldr = dalloc<uchar4>(I.npix);
    I.d_glyphs = dalloc<HudGlyph>(4096);
    I.d_rects = dalloc<HudRect>(1024);
    CK(cudaStreamSynchronize(I.st));
    (void)err;
    return true;
}

void Renderer::update_scene(const SceneGPU& s) {
    Impl& I = *p_;
    SceneGPU keep = I.S;
    I.S = s;
    I.S.disk_lnT = keep.disk_lnT;
    I.S.k2_lntab = keep.k2_lntab;
    I.S.bb_xyz = keep.bb_xyz;
    I.S.sky_tex = keep.sky_tex;
    I.S.sky_on = keep.sky_on;
    for (int k = 0; k < 24; ++k) I.S.turb_tex[k] = keep.turb_tex[k];
    I.S.turb_nt = keep.turb_nt;
    I.S.turb_lr0 = keep.turb_lr0;
    I.S.turb_lr1 = keep.turb_lr1;
    I.S.turb_zmax = keep.turb_zmax;
    I.S.turb_d0 = keep.turb_d0;
    I.S.turb_dd = keep.turb_dd;
    I.upload_scene();
}

void Renderer::Impl::trace(const FrameGPU& f, bool adaptive, RenderStats* stats) {
    CK(cudaMemcpyToSymbolAsync(c_F, &f, sizeof f, 0, cudaMemcpyHostToDevice, st));
    for (int k = 0; k < S.turb_nt; ++k) {   // re-bake the turbulence for this frame's time
        dim3 g((TURB_NPHI + 15) / 16, (TURB_NLR + 15) / 16, TURB_NZ);
        k_bake_turb<<<g, dim3(16, 16), 0, st>>>(turb_surf[k], S.turb_d0 + k * S.turb_dd);
    }
    CK(cudaGetLastError());
    CK(cudaMemsetAsync(d_fsum, 0, sizeof(double), st));
    CK(cudaMemsetAsync(d_nlim, 0, sizeof(unsigned long long), st));
    const float2* uv = nullptr;
    if (S.sky_on && S.mode == MODE_VISIBLE) {
        k_corners<<<grid2(f.W + 1, f.H + 1), dim3(16, 16), 0, st>>>(d_uv);
        CK(cudaGetLastError());
        uv = d_uv;
    }
    dim3 g1((f.W + 15) / 16, (f.H + 7) / 8);
    k_pass1<<<g1, 128, 0, st>>>(d_accum, d_ystat, d_nsamp, uv, d_fsum, d_nlim);
    CK(cudaGetLastError());
    double fsum = 0;
    CK(cudaMemcpyAsync(&fsum, d_fsum, sizeof fsum, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    int n = f.W * f.H;
    long long samples = (long long)n * f.spp;
    int extra_max = std::max(rs.spp_max - f.spp, 0);
    if (adaptive && extra_max > 0) {
        float yref = (float)(0.03 * fsum / ((double)n * f.spp) + 1e-37);
        k_adapt<<<(n + 255) / 256, 256, 0, st>>>(d_ystat, d_nsamp, n, yref, rs.target_err, extra_max, d_extra);
        k_dilate<<<grid2(f.W, f.H), dim3(16, 16), 0, st>>>(d_extra, d_chunks, f.W, f.H, extra_max);
        int nb = (n + SCAN_B - 1) / SCAN_B;
        k_scan_block<<<nb, SCAN_B, 0, st>>>(d_chunks, d_offsets, d_bsums, n);
        k_scan_top<<<1, SCAN_B, 0, st>>>(d_bsums, nb);
        k_scan_add<<<nb, SCAN_B, 0, st>>>(d_offsets, d_bsums, n);
        CK(cudaGetLastError());
        uint32_t last[2];
        CK(cudaMemcpyAsync(&last[0], d_offsets + n - 1, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(&last[1], d_chunks + n - 1, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        uint32_t total = last[0] + last[1];
        if (total > 0) {
            k_pass2<<<(total + 127) / 128, 128, 0, st>>>(d_accum, d_nsamp, d_offsets, uv, n, total);
            CK(cudaGetLastError());
        }
        samples += (long long)total * CHUNK;
    }
    k_resolve<<<(n + 255) / 256, 256, 0, st>>>(d_img, d_accum, d_nsamp, n);
    CK(cudaGetLastError());
    unsigned long long lim = 0;
    CK(cudaMemcpyAsync(&lim, d_nlim, sizeof lim, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    if (stats) {
        stats->samples = samples;
        stats->step_limited = (long long)lim;
    }
}

void Renderer::Impl::bloom(float strength) {
    for (int k = 1; k < nlev; ++k)
        k_down<<<grid2(lw[k], lh[k]), dim3(16, 16), 0, st>>>(d_lev[k - 1], lw[k - 1], lh[k - 1], d_lev[k], lw[k], lh[k]);
    LevelPtrs L;
    L.n = nlev;
    for (int k = 0; k < nlev; ++k) { L.p[k] = d_lev[k]; L.w[k] = lw[k]; L.h[k] = lh[k]; }
    k_bloom<<<grid2(W, H), dim3(16, 16), 0, st>>>(d_img, W, H, L, strength);
    CK(cudaGetLastError());
}

void Renderer::Impl::radio_blur(double sigma) {
    int k = 0;
    while (k + 1 < nlev && sigma / (1 << (k + 1)) > 3.0) ++k;
    for (int j = 1; j <= k; ++j)
        k_down<<<grid2(lw[j], lh[j]), dim3(16, 16), 0, st>>>(d_lev[j - 1], lw[j - 1], lh[j - 1], d_lev[j], lw[j], lh[j]);
    float s = (float)(sigma / (1 << k));
    float4* L = d_lev[k];
    k_gauss<<<grid2(lw[k], lh[k]), dim3(16, 16), 0, st>>>(L, d_tmp, lw[k], lh[k], s, 1);
    k_gauss<<<grid2(lw[k], lh[k]), dim3(16, 16), 0, st>>>(d_tmp, L, lw[k], lh[k], s, 0);
    if (k > 0) k_upsample<<<grid2(W, H), dim3(16, 16), 0, st>>>(L, lw[k], lh[k], d_img, W, H);
    CK(cudaGetLastError());
}

void Renderer::render(const FrameGPU& f_in, const PostParams& post, double percentile, HdrFrame* out, RenderStats* stats) {
    Impl& I = *p_;
    FrameGPU f = f_in;
    f.W = I.W;
    f.H = I.H;
    f.spp = I.rs.spp;
    double t0 = now_ms();
    I.trace(f, true, stats);
    double t1 = now_ms();
    if (I.S.mode == MODE_VISIBLE && post.bloom > 0.f) I.bloom(post.bloom);
    if (I.S.mode == MODE_RADIO && post.blur_sigma_px > 0.5) I.radio_blur(post.blur_sigma_px);
    out->W = I.W;
    out->H = I.H;
    out->px.resize(I.npix * 4);
    CK(cudaMemcpyAsync(out->px.data(), I.d_img, I.npix * sizeof(float4), cudaMemcpyDeviceToHost, I.st));
    CK(cudaStreamSynchronize(I.st));
    std::vector<float> lum;
    lum.reserve(I.npix);
    double mx = 0;
    for (size_t i = 0; i < I.npix; ++i) {
        const float* c = &out->px[4 * i];
        double Y = I.S.mode == MODE_RADIO ? c[0] - post.hide_direct * std::max(c[0] - c[1] - c[2] - c[3], 0.f)
                                          : 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        if (Y > 0) { lum.push_back((float)Y); mx = std::max(mx, Y); }
    }
    std::vector<float> sig;
    for (float v : lum)
        if (v > mx * 1e-5) sig.push_back(v);
    double pv = 0;
    if (!sig.empty()) {
        size_t idx = std::min(sig.size() - 1, (size_t)(percentile * (sig.size() - 1)));
        std::nth_element(sig.begin(), sig.begin() + idx, sig.end());
        pv = sig[idx];
    }
    out->lum_percentile = pv;
    out->lum_max = mx;
    if (stats) {
        stats->ms_trace = t1 - t0;
        stats->ms_post = now_ms() - t1;
    }
}

double Renderer::radio_flux(FrameGPU f, int n, double pix_sr) {
    Impl& I = *p_;
    f.W = n;
    f.H = n;
    f.spp = 2;
    I.trace(f, false, nullptr);
    std::vector<float4> img((size_t)n * n);
    CK(cudaMemcpyAsync(img.data(), I.d_img, img.size() * sizeof(float4), cudaMemcpyDeviceToHost, I.st));
    CK(cudaStreamSynchronize(I.st));
    double s = 0;
    for (const float4& v : img) s += v.x;
    return s * pix_sr / cgs::JY;
}

void Renderer::tonemap(const HdrFrame& in, const ToneParams& tp, const std::vector<HudGlyph>& glyphs,
                       const std::vector<HudRect>& rects, std::vector<uint8_t>* rgb8) {
    Impl& I = *p_;
    int W = in.W, H = in.H, n = W * H;
    CK(cudaMemcpyAsync(I.d_tmp, in.px.data(), (size_t)n * sizeof(float4), cudaMemcpyHostToDevice, I.st));
    k_tonemap<<<grid2(W, H), dim3(16, 16), 0, I.st>>>(I.d_tmp, W, H, tp, I.d_ldr);
    if (!rects.empty()) {
        std::vector<HudRect> sorted(rects);
        std::stable_sort(sorted.begin(), sorted.end(), [](const HudRect& a, const HudRect& b) { return a.layer < b.layer; });
        int nr = std::min((int)sorted.size(), 1024);
        CK(cudaMemcpyAsync(I.d_rects, sorted.data(), nr * sizeof(HudRect), cudaMemcpyHostToDevice, I.st));
        for (int a = 0; a < nr;) {
            int b = a;
            while (b < nr && sorted[b].layer == sorted[a].layer) ++b;
            k_hud_rects<<<b - a, 256, 0, I.st>>>(I.d_ldr, W, H, I.d_rects + a);
            a = b;
        }
    }
    if (!glyphs.empty()) {
        int ng = std::min((int)glyphs.size(), 4096);
        CK(cudaMemcpyAsync(I.d_glyphs, glyphs.data(), ng * sizeof(HudGlyph), cudaMemcpyHostToDevice, I.st));
        float off = std::max(1.5f, H / 720.f);
        k_hud_glyphs<<<ng, 256, 0, I.st>>>(I.d_ldr, W, H, I.d_glyphs, I.d_atlas, off, off, 1);
        k_hud_glyphs<<<ng, 256, 0, I.st>>>(I.d_ldr, W, H, I.d_glyphs, I.d_atlas, 0.f, 0.f, 0);
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
