/* COMMON
- Shared host/device definitions: units, physical constants, small vector math.
- Geometric units throughout the ray tracer: G = c = M = 1 (lengths and times in GM/c^2).
*/

#pragma once
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

namespace cgs {   // physical constants (CGS), for radiative transfer and unit conversion
constexpr double C = 2.99792458e10;
constexpr double G = 6.67430e-8;
constexpr double MSUN = 1.98847e33;
constexpr double KB = 1.380649e-16;
constexpr double H = 6.62607015e-27;
constexpr double ME = 9.1093837e-28;
constexpr double MP = 1.67262192e-24;
constexpr double QE = 4.80320471e-10;   // esu
constexpr double PC = 3.0856775814913673e18;
constexpr double JY = 1e-23;
}
constexpr double PI = 3.14159265358979323846;
constexpr double RAD2UAS = 180.0 / PI * 3600.0e6;

struct dvec3 { double x, y, z; };
HD dvec3 operator+(dvec3 a, dvec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
HD dvec3 operator-(dvec3 a, dvec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
HD dvec3 operator-(dvec3 a) { return {-a.x, -a.y, -a.z}; }
HD dvec3 operator*(dvec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
HD double dot(dvec3 a, dvec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD dvec3 cross(dvec3 a, dvec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
HD double length(dvec3 a) { return sqrt(dot(a, a)); }
HD dvec3 normalize(dvec3 a) { return a * (1.0 / length(a)); }

struct dmat3 { dvec3 r0, r1, r2; };   // row-major
HD dvec3 mul(const dmat3& m, dvec3 v) { return {dot(m.r0, v), dot(m.r1, v), dot(m.r2, v)}; }

HD float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
HD float saturate(float x) { return clampf(x, 0.f, 1.f); }
HD float lerpf(float a, float b, float t) { return a + (b - a) * t; }
HD float smoothstepf(float a, float b, float x) {
    float t = saturate((x - a) / (b - a));
    return t * t * (3.f - 2.f * t);
}
