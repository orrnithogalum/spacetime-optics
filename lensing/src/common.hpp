/* COMMON
- Shared host/device definitions: physical constants, spectral bands, small vector math.
*/

#pragma once
#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

/* NB
- Spectral sampling: NB contiguous bands covering 380-780 nm.
- Everything radiometric is carried per band (W m^-2 sr^-1 nm^-1) and only converted to colour at
  the very end.
*/
constexpr int   NB          = 16;
constexpr float LAMBDA0_NM  = 380.0f;
constexpr float DLAMBDA_NM  = 25.0f;
HD float band_center_nm(int k) { return LAMBDA0_NM + (k + 0.5f) * DLAMBDA_NM; }

namespace phys {
constexpr double PI        = 3.14159265358979323846;
constexpr double C         = 299792458.0;               // m/s
constexpr double GM_EARTH  = 3.986004418e14;            // m^3/s^2 (IERS / WGS84)
constexpr double GM_SUN    = 1.32712440018e20;          // m^3/s^2
constexpr double AU        = 1.495978707e11;            // m
constexpr double PC        = 3.0856775814913673e16;     // m
constexpr double R_EQ      = 6378137.0;                 // WGS84 semi-major axis
constexpr double FLAT      = 1.0 / 298.257223563;       // WGS84 flattening
constexpr double R_POL     = R_EQ * (1.0 - FLAT);       // semi-minor axis
constexpr double R_MEAN    = 6371008.8;                 // IUGG mean radius
constexpr double J2_EARTH  = 1.08262668e-3;             // EGM2008 (unnormalised C20)
constexpr double R_SUN     = 6.957e8;                   // IAU nominal solar radius
constexpr double LIGHTYEAR = 9.4607304725808e15;
constexpr double RS_EARTH  = 2.0 * GM_EARTH / (C * C);  // Schwarzschild radius, ~8.87 mm
constexpr double RS_SUN    = 2.0 * GM_SUN / (C * C);    // ~2.95 km
constexpr double H_PLANCK  = 6.62607015e-34;
constexpr double K_BOLTZ   = 1.380649e-23;
constexpr double RAD2MAS   = 180.0 / PI * 3600.0e3;
}

// Minimal vector types usable on host and device without pulling in CUDA headers
struct dvec2 { double x, y; };
struct dvec3 { double x, y, z; };
struct vec3  { float x, y, z; };

HD dvec2 operator+(dvec2 a, dvec2 b) { return {a.x + b.x, a.y + b.y}; }
HD dvec2 operator-(dvec2 a, dvec2 b) { return {a.x - b.x, a.y - b.y}; }
HD dvec2 operator*(dvec2 a, double s) { return {a.x * s, a.y * s}; }
HD double dot(dvec2 a, dvec2 b) { return a.x * b.x + a.y * b.y; }

HD dvec3 operator+(dvec3 a, dvec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
HD dvec3 operator-(dvec3 a, dvec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
HD dvec3 operator-(dvec3 a) { return {-a.x, -a.y, -a.z}; }
HD dvec3 operator*(dvec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
HD double dot(dvec3 a, dvec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD dvec3 cross(dvec3 a, dvec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
HD double length(dvec3 a) { return sqrt(dot(a, a)); }
HD dvec3 normalize(dvec3 a) { return a * (1.0 / length(a)); }

HD vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
HD vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
HD vec3 operator-(vec3 a) { return {-a.x, -a.y, -a.z}; }
HD vec3 operator*(vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
HD float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD vec3 cross(vec3 a, vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
HD float length(vec3 a) { return sqrtf(dot(a, a)); }
HD vec3 normalize(vec3 a) { return a * (1.0f / sqrtf(dot(a, a))); }
HD vec3 to_f(dvec3 a) { return {(float)a.x, (float)a.y, (float)a.z}; }

// Row-major 3x3 rotation
struct dmat3 { dvec3 r0, r1, r2; };
HD dvec3 mul(const dmat3& m, dvec3 v) { return {dot(m.r0, v), dot(m.r1, v), dot(m.r2, v)}; }
HD dvec3 mulT(const dmat3& m, dvec3 v) { return m.r0 * v.x + m.r1 * v.y + m.r2 * v.z; }
struct mat3 { vec3 r0, r1, r2; };
HD vec3 mul(const mat3& m, vec3 v) { return {dot(m.r0, v), dot(m.r1, v), dot(m.r2, v)}; }
HD vec3 mulT(const mat3& m, vec3 v) { return m.r0 * v.x + m.r1 * v.y + m.r2 * v.z; }

HD float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
HD float saturate(float x) { return clampf(x, 0.0f, 1.0f); }
HD float lerpf(float a, float b, float t) { return a + (b - a) * t; }
HD float smoothstepf(float a, float b, float x) {
    float t = saturate((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}
