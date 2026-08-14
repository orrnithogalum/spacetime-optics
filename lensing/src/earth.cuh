/* EARTH
- Earth's surface and cloud deck: NASA Blue Marble albedo, GEBCO relief, Blue Marble clouds, Black
  Marble city lights, Cox-Munk ocean glint, all lit by the attenuated spectral Sun plus skylight,
  then seen through the atmosphere.
*/

#pragma once
#include "atmo.cuh"

// RGB (linear sRGB) -> smooth reflectance spectrum, band k (Smits 1999)
DEVI float rgb_to_spec(float r, float g, float b, int k) {
    const float(*S)[NB] = c_S.smits;
    float v;
    if (r <= g && r <= b) {
        v = r * S[0][k];
        if (g <= b) v += (g - r) * S[1][k] + (b - g) * S[6][k];
        else        v += (b - r) * S[1][k] + (g - b) * S[5][k];
    } else if (g <= r && g <= b) {
        v = g * S[0][k];
        if (r <= b) v += (r - g) * S[2][k] + (b - r) * S[6][k];
        else        v += (b - g) * S[2][k] + (r - b) * S[4][k];
    } else {
        v = b * S[0][k];
        if (r <= g) v += (r - b) * S[3][k] + (g - r) * S[5][k];
        else        v += (g - b) * S[3][k] + (r - g) * S[4][k];
    }
    return fmaxf(v, 0.f);
}

DEVI float fresnel_water(float cos_i) {
    const float n = 1.334f;
    cos_i = saturate(cos_i);
    float sin_t2 = (1.f - cos_i * cos_i) / (n * n);
    if (sin_t2 >= 1.f) return 1.f;
    float cos_t = sqrtf(1.f - sin_t2);
    float rs = (cos_i - n * cos_t) / (cos_i + n * cos_t);
    float rp = (n * cos_i - cos_t) / (n * cos_i + cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

DEVI float smith_lambda_beckmann(float mu, float alpha) {
    mu = fmaxf(mu, 1e-4f);
    float a = mu / (alpha * sqrtf(fmaxf(1.f - mu * mu, 1e-8f)));
    if (a >= 1.6f) return 0.f;
    return (1.f - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}

// Cox-Munk wind-roughened sea surface BRDF (isotropic), with Smith masking
DEVI float glint_brdf(vec3 N, vec3 V, vec3 Ls, float wind) {
    float mu_i = dot(N, Ls), mu_v = dot(N, V);
    if (mu_i <= 0.f || mu_v <= 0.f) return 0.f;
    vec3 Hh = normalize(V + Ls);
    float mu_h = fmaxf(dot(N, Hh), 1e-4f);
    float s2 = 0.003f + 0.00512f * wind;   // total mean-square slope
    float mu_h2 = mu_h * mu_h;
    float tan2 = (1.f - mu_h2) / mu_h2;
    float Dn = __expf(-tan2 / s2) / (3.14159265f * s2 * mu_h2 * mu_h2);
    float alpha = sqrtf(s2);
    float G = 1.f / (1.f + smith_lambda_beckmann(mu_i, alpha) + smith_lambda_beckmann(mu_v, alpha));
    float F = fresnel_water(dot(V, Hh));
    return F * Dn * G / (4.f * mu_i * mu_v);
}

struct TexFootprint { float2 du, dv; };

// Texture-space footprint of one image pixel around a surface point with LENS normal N
DEVI TexFootprint footprint(vec3 N, dvec3 ecef, float lat, float lon, float scale) {
    float db = (float)(c_F.D * c_F.pix) * scale;
    float nz = fmaxf(N.z, 0.03f);
    // One-pixel steps along image x and y, projected onto the tangent plane
    vec3 dpx = {db, 0.f, -N.x * db / nz};
    vec3 dpy = {0.f, db, -N.y * db / nz};
    const dmat3& M = c_S.M_LE;
    auto to_ecef = [&](vec3 v) {
        return vec3{(float)(M.r0.x * v.x + M.r0.y * v.y + M.r0.z * v.z),
                    (float)(M.r1.x * v.x + M.r1.y * v.y + M.r1.z * v.z),
                    (float)(M.r2.x * v.x + M.r2.y * v.y + M.r2.z * v.z)};
    };
    vec3 ex = to_ecef(dpx), ey = to_ecef(dpy);
    float sl = sinf(lon), cl = cosf(lon), sp = sinf(lat), cp = cosf(lat);
    vec3 east = {-sl, cl, 0.f}, north = {-sp * cl, -sp * sl, cp};
    float rxy = fmaxf((float)sqrt(ecef.x * ecef.x + ecef.y * ecef.y), 1000.f);
    float R = (float)c_S.ell_a;
    TexFootprint f;
    f.du = make_float2(dot(ex, east) / rxy * (0.5f / 3.14159265f), dot(ey, east) / rxy * (0.5f / 3.14159265f));
    f.dv = make_float2(-dot(ex, north) / R * (1.f / 3.14159265f), -dot(ey, north) / R * (1.f / 3.14159265f));
    return f;
}

DEVI float4 sample_albedo(float2 uv, const TexFootprint& f) {
    return tex2DGrad<float4>((cudaTextureObject_t)c_S.albedo_tex, uv.x, uv.y,
                             make_float2(f.du.x, f.dv.x), make_float2(f.du.y, f.dv.y));
}
DEVI float4 sample_aux(float2 uv, const TexFootprint& f) {
    return tex2DGrad<float4>((cudaTextureObject_t)c_S.aux_tex, uv.x, uv.y,
                             make_float2(f.du.x, f.dv.x), make_float2(f.du.y, f.dv.y));
}
DEVI float cloud_cover_from_tex(float v) {
    return saturate(powf(saturate(v * 1.05f), 1.25f) * c_S.cloud_opacity);
}
DEVI float cloud_at(dvec3 pL, float lod) {
    float lat, lon;
    lens_to_geo(pL, &lat, &lon, nullptr);
    float2 uv = geo_to_uv(lat, lon);
    float4 a = tex2DLod<float4>((cudaTextureObject_t)c_S.aux_tex, uv.x, uv.y, lod);
    return cloud_cover_from_tex(a.x);
}

// Lit cloud-top radiance at LENS point p (cloud deck treated as a bright diffuse layer)
DEVI void cloud_radiance(dvec3 p, float L[NB]) {
    vec3 pf = to_f(p);
    float r = length(pf);
    vec3 up = pf * (1.f / r);
    float mu_s = dot(up, c_S.sun_L);
    float Ts[NB], Es[NB];
    sun_transmittance(c_S.R_lut + c_S.cloud_alt, mu_s, Ts);
    sky_irradiance(mu_s, Es);
    float m = fmaxf(mu_s, 0.f);
    // Mild forward-scattering brightening when back-lit, as for real cloud decks
    float fwd = 1.f + 0.6f * powf(saturate(-c_S.sun_L.z), 8.f);
#pragma unroll
    for (int k = 0; k < NB; ++k)
        L[k] = c_S.cloud_albedo * (1.f / 3.14159265f) * (c_S.E_sun[k] * Ts[k] * m * fwd + 0.7f * Es[k]);
}

/* ground_radiance
- Radiance leaving the ground at LENS point pg (before the atmosphere between ground and telescope
  is applied).
*/
DEVI void ground_radiance(dvec3 pg, float L[NB]) {
    dvec3 ecef;
    float lat, lon;
    lens_to_geo(pg, &lat, &lon, &ecef);
    vec3 N = to_f(ellipsoid_normal(pg));
    const vec3 V = {0.f, 0.f, 1.f};
    const vec3 Ls = c_S.sun_L;
    float mu_s_geo = dot(N, Ls);

    float Ts[NB], Es[NB];
    if (c_S.atmosphere) {
        sun_transmittance(c_S.R_lut, mu_s_geo, Ts);
        sky_irradiance(mu_s_geo, Es);
    } else {
#pragma unroll
        for (int k = 0; k < NB; ++k) { Ts[k] = mu_s_geo > 0.f ? 1.f : 0.f; Es[k] = 0.f; }
    }

    if (!c_S.has_textures) {
        // Untextured fallback: grey land-ish sphere
#pragma unroll
        for (int k = 0; k < NB; ++k)
            L[k] = 0.3f / 3.14159265f * (c_S.E_sun[k] * Ts[k] * fmaxf(mu_s_geo, 0.f) + Es[k]);
        return;
    }

    float2 uv = geo_to_uv(lat, lon);
    TexFootprint fp = footprint(N, ecef, lat, lon, 0.6f);
    float4 alb = sample_albedo(uv, fp);
    float4 aux = sample_aux(uv, fp);

    // Relief: perturb the normal with the GEBCO elevation gradient
    vec3 Nr = N;
    if (c_S.relief > 0.f) {
        float su = fmaxf(fabsf(fp.du.x) + fabsf(fp.du.y), 1.f / c_S.tex_w);
        float sv = fmaxf(fabsf(fp.dv.x) + fabsf(fp.dv.y), 1.f / c_S.tex_h);
        float he = tex2DGrad<float4>((cudaTextureObject_t)c_S.aux_tex, uv.x + su, uv.y, make_float2(fp.du.x, fp.dv.x), make_float2(fp.du.y, fp.dv.y)).z;
        float hw = tex2DGrad<float4>((cudaTextureObject_t)c_S.aux_tex, uv.x - su, uv.y, make_float2(fp.du.x, fp.dv.x), make_float2(fp.du.y, fp.dv.y)).z;
        float hn = tex2DGrad<float4>((cudaTextureObject_t)c_S.aux_tex, uv.x, uv.y - sv, make_float2(fp.du.x, fp.dv.x), make_float2(fp.du.y, fp.dv.y)).z;
        float hs = tex2DGrad<float4>((cudaTextureObject_t)c_S.aux_tex, uv.x, uv.y + sv, make_float2(fp.du.x, fp.dv.x), make_float2(fp.du.y, fp.dv.y)).z;
        const float HMAX = 6400.f;
        float rxy = fmaxf((float)sqrt(ecef.x * ecef.x + ecef.y * ecef.y), 1000.f);
        float dE = (he - hw) * HMAX / (2.f * su * 2.f * 3.14159265f * rxy);
        float dN = (hn - hs) * HMAX / (2.f * sv * 3.14159265f * (float)c_S.ell_a);
        float sl = sinf(lon), cl = cosf(lon), sp = sinf(lat), cp = cosf(lat);
        vec3 east = ecef_dir_to_lens(vec3{-sl, cl, 0.f});
        vec3 north = ecef_dir_to_lens(vec3{-sp * cl, -sp * sl, cp});
        Nr = normalize(N - (east * dE + north * dN) * c_S.relief);
    }
    float mu_s = dot(Nr, Ls);

    // Cloud shadow: look up the deck along the sun direction
    float shadow = 1.f;
    if (c_S.clouds && mu_s_geo > 0.02f) {
        dvec3 ps = pg + dvec3{Ls.x, Ls.y, Ls.z} * (double)(c_S.cloud_alt / fmaxf(mu_s_geo, 0.05f));
        shadow = 1.f - 0.85f * cloud_at(ps, 2.f);
    }

    float water = alb.w;
    float lum = 0.2126f * alb.x + 0.7152f * alb.y + 0.0722f * alb.z;
    float ice = water * smoothstepf(0.10f, 0.30f, lum);   // sea ice / ice shelves over "ocean"
    float wat = water - ice;

    float glint = 0.f, Fv = 0.f, Fi = 0.f, whitecap = 0.f;
    if (wat > 0.f) {
        glint = glint_brdf(N, V, Ls, c_S.ocean_wind) * fmaxf(dot(N, Ls), 0.f);
        Fv = fresnel_water(N.z);
        Fi = fresnel_water(fmaxf(dot(N, Ls), 0.f));
        whitecap = fminf(2.95e-6f * powf(c_S.ocean_wind, 3.52f), 0.1f);
    }
    float lights = 0.f;
    if (c_S.night_lights) lights = powf(aux.y, 2.2f) * c_S.night_scale;

    float dir_cos = fmaxf(mu_s, 0.f) * shadow;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        float Edir = c_S.E_sun[k] * Ts[k];
        float E = Edir * dir_cos + Es[k];
        float rho_land = 0.94f * rgb_to_spec(alb.x, alb.y, alb.z, k);
        float Lland = rho_land * (1.f / 3.14159265f) * E;
        float Lwat = (1.f - Fv) * (1.f - Fi) * c_S.water_refl[k] * (1.f / 3.14159265f) * E
                   + glint * Edir * shadow + Fv * Es[k] * (1.f / 3.14159265f)
                   + whitecap * 0.22f * (1.f / 3.14159265f) * E;
        L[k] = (1.f - wat) * Lland + wat * Lwat + lights * c_S.city_spec[k];
    }
}

// Full radiance towards the telescope for a ray that strikes the ellipsoid at t_ground
DEVI void shade_earth(double bx, double by, double t_ground, float jitter, float L[NB]) {
    dvec3 pg = {bx, by, -t_ground};
    float Lg[NB];
    ground_radiance(pg, Lg);
    if (!c_S.atmosphere) {
        float c = 0.f;
        if (c_S.clouds && c_S.has_textures) c = cloud_at(pg, 0.f);
        float Lc[NB];
        if (c > 0.f) cloud_radiance(pg, Lc);
#pragma unroll
        for (int k = 0; k < NB; ++k) L[k] = c > 0.f ? lerpf(Lg[k], Lc[k], c) : Lg[k];
        return;
    }
    double ta, tb;
    if (!ellipsoid_roots(bx, by, c_S.h_top + 1000.0, &ta, &tb)) ta = t_ground;
    float T[NB];
#pragma unroll
    for (int k = 0; k < NB; ++k) { L[k] = 0.f; T[k] = 1.f; }

    double tc0, tc1;
    bool deck = c_S.clouds && c_S.has_textures && ellipsoid_roots(bx, by, c_S.cloud_alt, &tc0, &tc1) &&
                tc0 < t_ground && tc0 > ta;
    if (!deck) {
        march((float)bx, (float)by, (float)ta, (float)t_ground, 0.f, 0, 24, jitter, L, T);
#pragma unroll
        for (int k = 0; k < NB; ++k) L[k] += T[k] * Lg[k];
        return;
    }
    // Above the deck
    march((float)bx, (float)by, (float)ta, (float)tc0, 0.f, 0, 16, jitter, L, T);
    dvec3 pc = {bx, by, -tc0};
    float c = cloud_at(pc, 0.5f);
    float L1[NB], T1[NB], Lc[NB];
#pragma unroll
    for (int k = 0; k < NB; ++k) { L1[k] = L[k]; T1[k] = T[k]; }
    if (c > 0.f) cloud_radiance(pc, Lc);
    // Below the deck
    march((float)bx, (float)by, (float)tc0, (float)t_ground, 0.f, 0, 10, jitter, L, T);
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        float below = (L[k] - L1[k]) + T[k] * Lg[k];
        float cl = c > 0.f ? Lc[k] : 0.f;
        L[k] = L1[k] + T1[k] * c * cl + (1.f - c) * below;
    }
}
