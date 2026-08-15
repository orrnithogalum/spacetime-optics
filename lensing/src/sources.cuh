/* SOURCES
- Everything behind the lens: the Sun (at its ephemeris position) and up to MAX_SOURCES target
  objects at their true distances.
- A ray leaving the lens region at impact vector b with slope v reaches the plane of an object at
  distance d at transverse position b + d*v (exact for straight propagation; the lens equation with
  finite source distance falls out of this).
*/

#pragma once
#include "earth.cuh"

DEVI bool band_on(int band, int k) { return band < 0 || band == k; }

DEVI void star_radiance(const SourceGPU& s, float qx, float qy, int band, float out[NB]) {
    float r2 = qx * qx + qy * qy;
    float mu = sqrtf(fmaxf(1.f - r2, 0.f));
    vec3 n = {qx, qy, mu};
    vec3 b = mul(s.rot, n);
    float gpat = 0.f;
    if (s.gran_contrast > 0.f) {
        float e = worley_edge(b * s.gran_cells, s.seed);
        float pat = smoothstepf(0.0f, 0.42f, e);
        float meso = fbm(b * (s.gran_cells * 0.11f), 3, s.seed + 17u);
        gpat = (pat - 0.6f) * 1.6f + 0.35f * meso;
    }
    float wu = 0.f, wp = 0.f;
    for (int i = 0; i < s.nspots; ++i) {
        float d = acosf(clampf(dot(b, s.spot_dir[i]), -1.f, 1.f)) / s.spot_size[i];
        if (d < 1.f) {
            wu = fmaxf(wu, 1.f - smoothstepf(0.30f, 0.45f, d));
            wp = fmaxf(wp, 1.f - smoothstepf(0.85f, 1.0f, d));
        }
    }
    float sq = sqrtf(mu);
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        if (!band_on(band, k)) continue;
        float lam = band_center_nm(k);
        float ld = 1.f - s.ldk[k] * (1.f - mu);
        float g = 1.f + s.gran_contrast * (500.f / lam) * sq * gpat;
        float spot = lerpf(1.f, lerpf(0.5f * (1.f + s.spot_tk[k]), s.spot_tk[k], wu), wp);
        out[k] = s.Bk[k] * ld * fmaxf(g, 0.f) * spot;
    }
}

DEVI void planet_radiance(const SourceGPU& s, float qx, float qy, int band, float out[NB]) {
    float r2 = qx * qx + qy * qy;
    float mu_v = sqrtf(fmaxf(1.f - r2, 0.f));
    vec3 n = {qx, qy, mu_v};
    float mu_l = dot(n, s.light);
#pragma unroll
    for (int k = 0; k < NB; ++k) if (band_on(band, k)) out[k] = 0.f;
    if (mu_l <= 0.f) return;
    vec3 b = mul(s.rot, n);
    float lat = asinf(clampf(b.z, -1.f, 1.f)), lon = atan2f(b.y, b.x);
    float rgb[3], water, cloud;
    if (s.surface == SURF_EARTH && c_S.has_textures) {
        float2 uv = geo_to_uv(lat, lon);
        float4 a = tex2DLod<float4>((cudaTextureObject_t)c_S.albedo_tex, uv.x, uv.y, 3.f);
        float4 x = tex2DLod<float4>((cudaTextureObject_t)c_S.aux_tex, uv.x, uv.y, 3.f);
        rgb[0] = a.x; rgb[1] = a.y; rgb[2] = a.z;
        water = a.w;
        cloud = cloud_cover_from_tex(x.x);
    } else {
        vec3 p = b * 1.7f;
        float warp = fbm(p * 1.3f + vec3{3.1f, 1.7f, 0.4f}, 3, s.seed + 5u);
        float e = fbm(p + vec3{warp, warp, warp} * 0.6f, 6, s.seed);
        float moist = fbm(p * 2.5f, 4, s.seed + 3u);
        float ice = smoothstepf(1.0f, 1.2f, fabsf(lat) + 0.18f * fbm(p * 4.f, 3, s.seed + 9u));
        water = e < s.ocean_level ? 1.f : 0.f;
        float dry = smoothstepf(-0.15f, 0.25f, moist - 0.4f * cosf(2.f * lat));
        float land[3] = {lerpf(0.07f, 0.42f, dry), lerpf(0.11f, 0.33f, dry), lerpf(0.05f, 0.21f, dry)};
        float hi = smoothstepf(0.25f, 0.45f, e - s.ocean_level);
        for (int c = 0; c < 3; ++c) land[c] = lerpf(land[c], 0.24f - 0.02f * c, hi);
        for (int c = 0; c < 3; ++c) rgb[c] = lerpf(land[c], 0.86f + 0.03f * c, ice);
        water *= (1.f - ice);
        float cn = fbm(b * 3.2f + vec3{warp, -warp, warp} * 0.8f, 5, s.seed + 11u) + 0.12f * cosf(6.f * lat);
        cloud = smoothstepf(0.0f, 0.40f, cn + (s.cloud_cover - 0.5f) * 0.6f);
    }
    float glint = water > 0.f ? glint_brdf(n, vec3{0.f, 0.f, 1.f}, s.light, 7.f) * mu_l : 0.f;
    float Fv = fresnel_water(mu_v);
    float haze = 0.03f * sqrtf(mu_l) / (mu_v + 0.25f);
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        if (!band_on(band, k)) continue;
        float E = s.Bk[k];
        float lam = band_center_nm(k);
        float rho = 0.94f * rgb_to_spec(rgb[0], rgb[1], rgb[2], k) * s.albedo_scale;
        float Ls = (1.f - water) * rho * (1.f / 3.14159265f) * E * mu_l +
                   water * ((1.f - Fv) * c_S.water_refl[k] * (1.f / 3.14159265f) * E * mu_l + glint * E);
        float Lc = 0.85f * (1.f / 3.14159265f) * E * mu_l;
        float ray = haze * powf(450.f / lam, 4.f) * (1.f / 3.14159265f) * E;
        out[k] = lerpf(Ls, Lc, cloud) + ray;
    }
}

DEVI void image_radiance(const SourceGPU& s, float u, float v, int band, float out[NB]) {
    float4 c = tex2D<float4>((cudaTextureObject_t)s.tex, u, v);
#pragma unroll
    for (int k = 0; k < NB; ++k)
        if (band_on(band, k)) out[k] = s.image_scale * rgb_to_spec(c.x, c.y, c.z, k);
}

/* background
- Radiance arriving from behind the lens along the outgoing ray (impact b, slope v). band < 0: all
  bands; otherwise only out[band] is written.
*/
DEVI void background(double bx, double by, double vx, double vy, int band, float out[NB]) {
#pragma unroll
    for (int k = 0; k < NB; ++k) if (band_on(band, k)) out[k] = 0.f;

    // The Sun: exact ray-sphere test at its ephemeris position
    {
        double inv = rsqrt(vx * vx + vy * vy + 1.0);
        dvec3 d = {vx * inv, vy * inv, -inv};
        dvec3 oc = c_S.sun_pos_L - dvec3{bx, by, 0.0};
        double tca = dot(oc, d);
        if (tca > 0.0) {
            dvec3 perp = oc - d * tca;
            double d2 = dot(perp, perp), R2 = c_S.sun_radius * c_S.sun_radius;
            if (d2 < R2) {
                float mu = (float)sqrt(1.0 - d2 / R2);
#pragma unroll
                for (int k = 0; k < NB; ++k)
                    if (band_on(band, k)) out[k] = c_S.sun_Lk[k] * (1.f - c_S.sun_ld[k] * (1.f - mu));
                return;
            }
        }
    }
    for (int i = 0; i < c_S.nsrc; ++i) {
        const SourceGPU& s = c_S.src[i];
        double xs = bx + s.dist * vx - s.cx;
        double ys = by + s.dist * vy - s.cy;
        if (s.type == SRC_IMAGE) {
            if (fabs(xs) < s.radius && fabs(ys) < s.half_h) {
                image_radiance(s, (float)(0.5 + 0.5 * xs / s.radius), (float)(0.5 - 0.5 * ys / s.half_h), band, out);
                return;
            }
            continue;
        }
        double r2 = xs * xs + ys * ys;
        if (r2 >= s.radius * s.radius) continue;
        float qx = (float)(xs / s.radius), qy = (float)(ys / s.radius);
        if (s.type == SRC_STAR) star_radiance(s, qx, qy, band, out);
        else if (s.type == SRC_PLANET) planet_radiance(s, qx, qy, band, out);
        return;
    }
}
