#include "scene_build.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "atmosphere_model.hpp"
#include "image_io.hpp"
#include "spectrum.hpp"

static constexpr double DEG = phys::PI / 180.0;

double einstein_angle(double D, double d) {
    return std::sqrt(2.0 * phys::RS_EARTH * d / (D * (D + d)));
}

namespace {
dmat3 rot_x(double a) { return {{1, 0, 0}, {0, std::cos(a), -std::sin(a)}, {0, std::sin(a), std::cos(a)}}; }
dmat3 rot_z(double a) { return {{std::cos(a), -std::sin(a), 0}, {std::sin(a), std::cos(a), 0}, {0, 0, 1}}; }
dmat3 matmul(const dmat3& A, const dmat3& B) {
    dvec3 c0 = {B.r0.x, B.r1.x, B.r2.x}, c1 = {B.r0.y, B.r1.y, B.r2.y}, c2 = {B.r0.z, B.r1.z, B.r2.z};
    return {{dot(A.r0, c0), dot(A.r0, c1), dot(A.r0, c2)},
            {dot(A.r1, c0), dot(A.r1, c1), dot(A.r1, c2)},
            {dot(A.r2, c0), dot(A.r2, c1), dot(A.r2, c2)}};
}
mat3 to_f(const dmat3& m) {
    return {{(float)m.r0.x, (float)m.r0.y, (float)m.r0.z},
            {(float)m.r1.x, (float)m.r1.y, (float)m.r1.z},
            {(float)m.r2.x, (float)m.r2.y, (float)m.r2.z}};
}

/* body_rotation
- Object frame (z towards Earth, y = projected celestial north) -> body frame of a sphere whose pole
  is tilted 'tilt' towards the viewer and that has rotated by 'spin' about its pole.
*/
dmat3 body_rotation(double tilt, double spin) {
    // B0^T maps object -> body for an untilted body with its pole along object +y
    dmat3 B0T = {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}};
    return matmul(rot_z(spin), matmul(B0T, rot_x(-tilt)));
}

double distance_m(const Config& c, const std::string& sec, double def_pc) {
    if (c.has(sec + ".distance_au")) return c.num(sec + ".distance_au", 1) * phys::AU;
    if (c.has(sec + ".distance_ly")) return c.num(sec + ".distance_ly", 1) * phys::LIGHTYEAR;
    return c.num(sec + ".distance_pc", def_pc) * phys::PC;
}

double radius_m(const Config& c, const std::string& sec, double def_km) {
    if (c.has(sec + ".radius_rsun")) return c.num(sec + ".radius_rsun", 1) * phys::R_SUN;
    if (c.has(sec + ".radius_rearth")) return c.num(sec + ".radius_rearth", 1) * phys::R_MEAN;
    return c.num(sec + ".radius_km", def_km) * 1e3;
}

void fill_star(SourceGPU& s, const Config& c, const std::string& sec, double R, const float solar_1au[NB]) {
    double teff = c.num(sec + ".teff", 5772);
    // Spectral shape: the observed solar spectrum (with its absorption lines averaged per band),
    // re-weighted by the Planck ratio for other temperatures
    float pl[NB], pls[NB];
    spectrum::planck_bands(teff, pl);
    spectrum::planck_bands(5772.0, pls);
    double omega_sun = phys::PI * std::pow(phys::R_SUN / phys::AU, 2);
    for (int k = 0; k < NB; ++k) {
        double lam = band_center_nm(k);
        float u = spectrum::solar_limb_darkening(lam) * (float)c.num(sec + ".limb_darkening_scale", 1.0);
        s.ldk[k] = u;
        double I0_sun = solar_1au[k] / (omega_sun * (1.0 - u / 3.0));
        s.Bk[k] = (float)(I0_sun * pl[k] / pls[k]);
        double tspot = teff - c.num(sec + ".spot_dt", 1700);
        s.spot_tk[k] = (float)(spectrum::planck(lam, tspot) / spectrum::planck(lam, teff));
    }
    s.gran_contrast = (float)c.num(sec + ".granulation", 0.14);
    s.gran_cells = (float)(R / (c.num(sec + ".granule_km", 1300) * 1e3));
    std::mt19937 rng((uint32_t)c.integer(sec + ".seed", 7));
    std::uniform_real_distribution<double> U(0, 1);
    s.nspots = std::min(c.integer(sec + ".spots", 3), MAX_SPOTS);
    for (int i = 0; i < s.nspots; ++i) {
        double lat = (U(rng) * 2 - 1) * 32 * DEG, lon = U(rng) * 2 * phys::PI;
        s.spot_dir[i] = {(float)(std::cos(lat) * std::cos(lon)), (float)(std::cos(lat) * std::sin(lon)), (float)std::sin(lat)};
        s.spot_size[i] = (float)((0.025 + 0.05 * U(rng)) * c.num(sec + ".spot_scale", 1.0));
    }
    s.rot = to_f(body_rotation(c.num(sec + ".pole_tilt_deg", 20) * DEG, c.num(sec + ".rotation_deg", 0) * DEG));
    s.seed = (uint32_t)c.integer(sec + ".seed", 7) * 2654435761u;
}
}  // namespace

bool load_earth_textures(const std::string& dir, EarthTextureData* out, std::string* err) {
    int w, h, c, w2, h2, c2;
    if (!read_texture(dir + "/albedo.lgtx", &w, &h, &c, &out->albedo) || c != 4) {
        if (err) *err = "cannot read " + dir + "/albedo.lgtx (run scripts/fetch_earth_data.sh)";
        return false;
    }
    if (!read_texture(dir + "/aux.lgtx", &w2, &h2, &c2, &out->aux) || c2 != 4 || w2 != w || h2 != h) {
        if (err) *err = "cannot read " + dir + "/aux.lgtx";
        return false;
    }
    out->w = w;
    out->h = h;
    return true;
}

void scene_geometry(const Config& cfg, double jd, SceneGPU* S, SceneInfo* info) {
    double ra = cfg.angle_hours_deg("target.ra", 217.4289), dec = cfg.angle_deg("target.dec", -62.6795);
    dvec3 t_eci = precess_j2000_to_date(radec_to_unit(ra, dec), jd);
    dvec3 ez = -t_eci;
    dvec3 north = {0, 0, 1};
    dvec3 ey = normalize(north - ez * dot(north, ez));
    dvec3 ex = cross(ey, ez);
    // LENS -> ECI has columns ex, ey, ez; ECI -> ECEF is a rotation by -GMST about z
    double g = gmst_rad(jd), cg = std::cos(g), sg = std::sin(g);
    auto eci_to_ecef = [&](dvec3 v) { return dvec3{cg * v.x + sg * v.y, -sg * v.x + cg * v.y, v.z}; };
    dvec3 cx = eci_to_ecef(ex), cy = eci_to_ecef(ey), cz = eci_to_ecef(ez);
    S->M_LE = {{cx.x, cy.x, cz.x}, {cx.y, cy.y, cz.y}, {cx.z, cy.z, cz.z}};
    S->spin_L = {ex.z, ey.z, ez.z};   // ECI z-axis expressed in LENS

    double sun_au = 1.0;
    dvec3 sun_eci = sun_position_eci(jd, &sun_au);
    dvec3 sunL = {dot(sun_eci, ex), dot(sun_eci, ey), dot(sun_eci, ez)};
    S->sun_L = {(float)sunL.x, (float)sunL.y, (float)sunL.z};
    S->sun_pos_L = sunL * (sun_au * phys::AU);
    S->sun_radius = phys::R_SUN;
    info->phase_deg = std::acos(std::max(-1.0, std::min(1.0, sunL.z))) / DEG;
    info->sub_lat = std::asin(cz.z) / DEG;
    info->sub_lon = std::atan2(cz.y, cz.x) / DEG;

    // Gravity
    S->ell_a = phys::R_EQ;
    S->ell_c = phys::R_POL;
    S->Rs = phys::RS_EARTH;
    S->use_j2 = cfg.flag("gravity.j2", true) ? 1 : 0;
    double sx = S->spin_L.x, sy = S->spin_L.y, sp = std::sqrt(sx * sx + sy * sy);
    S->j2_Q = phys::RS_EARTH * phys::J2_EARTH * phys::R_EQ * phys::R_EQ * sp * sp;
    S->j2_ex = sp > 0 ? sx / sp : 1.0;
    S->j2_ey = sp > 0 ? sy / sp : 0.0;
    S->use_tide = cfg.flag("gravity.solar_tide", true) ? 1 : 0;
    {
        double Bx = -S->sun_pos_L.x, By = -S->sun_pos_L.y, B2 = Bx * Bx + By * By;
        double ux = Bx / std::sqrt(B2), uy = By / std::sqrt(B2), k = -2.0 * phys::RS_SUN / B2;
        S->tide[0] = k * (1 - 2 * ux * ux);
        S->tide[1] = k * (-2 * ux * uy);
        S->tide[2] = k * (-2 * ux * uy);
        S->tide[3] = k * (1 - 2 * uy * uy);
        info->sun_sep_au = std::sqrt(B2) / phys::AU;
    }
    float solar[NB];
    spectrum::solar_irradiance_1au(solar);
    double omega_sun = phys::PI * std::pow(phys::R_SUN / phys::AU, 2);
    for (int k = 0; k < NB; ++k) {
        double lam = band_center_nm(k);
        S->E_sun[k] = (float)(solar[k] / (sun_au * sun_au));
        S->sun_ld[k] = spectrum::solar_limb_darkening(lam);
        S->sun_Lk[k] = (float)(solar[k] / (omega_sun * (1.0 - S->sun_ld[k] / 3.0)));
    }
}

bool build_scene(const Config& cfg, SceneGPU* S, SceneInfo* info, std::string* err) {
    std::memset(S, 0, sizeof(SceneGPU));
    if (!parse_iso8601(cfg.str("time.utc", "2026-05-20T09:37:00"), &info->epoch)) {
        if (err) *err = "bad time.utc";
        return false;
    }
    scene_geometry(cfg, info->epoch.jd_utc, S, info);

    // Spectra
    float solar[NB], ozone[NB];
    spectrum::solar_irradiance_1au(solar);
    spectrum::ozone_cross_section(ozone);
    spectrum::smits_basis(S->smits);
    spectrum::water_reflectance(S->water_refl);
    float city[NB];
    spectrum::planck_bands(cfg.num("earth.city_light_temp", 2700), city);
    double cm = 0;
    for (int k = 0; k < NB; ++k) cm += city[k];
    for (int k = 0; k < NB; ++k) S->city_spec[k] = (float)(city[k] * NB / cm);

    // Atmosphere
    S->atmosphere = cfg.flag("earth.atmosphere", true) ? 1 : 0;
    S->R_lut = (float)phys::R_MEAN;
    S->h_top = 100e3f;
    S->h_top_refr = 300e3;
    S->mie_H = (float)(cfg.num("earth.aerosol_scale_height_km", 1.6) * 1e3);
    S->mie_g = (float)cfg.num("earth.aerosol_g", 0.80);
    double aod = cfg.num("earth.aod550", 0.08), ang = cfg.num("earth.angstrom", 1.3);
    double ssa = cfg.num("earth.aerosol_ssa", 0.92);
    double ozone_du = cfg.num("earth.ozone_du", 300);
    double n_o3_peak = ozone_du * 2.6867e20 / 15000.0;   // tent profile 10-40 km
    const double N_air = atmo::RHO0 * 6.02214076e23 / 0.0289644;
    for (int k = 0; k < NB; ++k) {
        double lam = band_center_nm(k);
        S->sigR[k] = (float)(spectrum::rayleigh_cross_section(lam) * N_air);
        double bm = aod / S->mie_H * std::pow(lam / 550.0, -ang);
        S->sigMs[k] = (float)(bm * ssa);
        S->sigMe[k] = (float)bm;
        S->sigO[k] = (float)(ozone[k] * n_o3_peak);
        info->refractivity[k] = spectrum::air_refractivity_std(lam);
        S->h_ground_k[k] = (float)(info->refractivity[k] * S->R_lut);
    }
    S->turb_amp = (float)cfg.num("earth.turbulence", 0.03);
    S->turb_lh = (float)(cfg.num("earth.turbulence_lh_km", 120) * 1e3);
    S->turb_lv = (float)(cfg.num("earth.turbulence_lv_km", 4) * 1e3);
    S->airglow = (float)cfg.num("earth.airglow", 1.0);
    S->airglow_band = (int)((557.7f - LAMBDA0_NM) / DLAMBDA_NM);
    S->ground_albedo_ms = 0.3f;

    // Surface
    S->cloud_alt = (float)(cfg.num("earth.cloud_alt_km", 6.0) * 1e3);
    S->cloud_opacity = (float)cfg.num("earth.cloud_opacity", 1.0);
    S->cloud_albedo = (float)cfg.num("earth.cloud_albedo", 0.80);
    S->night_scale = (float)cfg.num("earth.night_light_radiance", 4e-6);
    S->ocean_wind = (float)cfg.num("earth.ocean_wind", 7.0);
    S->relief = (float)cfg.num("earth.relief", 1.0);
    S->clouds = cfg.flag("earth.clouds", true) ? 1 : 0;
    S->night_lights = cfg.flag("earth.night_lights", true) ? 1 : 0;
    std::string tdir = cfg.str("earth.textures", "data/earth");
    if (!tdir.empty() && tdir[0] != '/') {
        // Resolve relative to the working directory first, then to the scene file
        FILE* f = std::fopen((tdir + "/albedo.lgtx").c_str(), "rb");
        if (f) std::fclose(f);
        else tdir = cfg.dir() + "/" + tdir;
    }
    info->earth_dir = tdir;

    // Targets
    info->target_name = cfg.str("target.name", "target");
    info->target_type = cfg.str("target.type", "planet");
    double d = distance_m(cfg, "target", 1.3012);
    info->target_dist = d;
    std::vector<SourceGPU> srcs;
    SourceGPU t{};
    t.dist = d;
    t.seed = (uint32_t)cfg.integer("target.seed", 7) * 2246822519u;
    if (info->target_type == "star") {
        t.type = SRC_STAR;
        t.radius = radius_m(cfg, "target", 696000);
        fill_star(t, cfg, "target", t.radius, solar);
    } else if (info->target_type == "planet") {
        t.type = SRC_PLANET;
        t.radius = radius_m(cfg, "target", 7160);
        t.surface = cfg.str("target.surface", "procedural") == "earth" ? SURF_EARTH : SURF_PROCEDURAL;
        double phase = cfg.num("target.phase_deg", 60) * DEG, pa = cfg.num("target.phase_pa_deg", 120) * DEG;
        dvec3 L = {std::sin(phase) * std::cos(pa), std::sin(phase) * std::sin(pa), std::cos(phase)};
        t.light = {(float)L.x, (float)L.y, (float)L.z};
        double th = cfg.num("target.host_teff", 3042), hr = cfg.num("target.host_radius_rsun", 0.1542) * phys::R_SUN;
        double a = cfg.num("target.orbit_au", 0.04857) * phys::AU;
        float pl[NB];
        spectrum::planck_bands(th, pl);
        for (int k = 0; k < NB; ++k) t.Bk[k] = (float)(phys::PI * pl[k] * (hr / a) * (hr / a));
        t.albedo_scale = (float)cfg.num("target.albedo_scale", 1.0);
        t.cloud_cover = (float)cfg.num("target.cloud_cover", 0.5);
        t.ocean_level = (float)cfg.num("target.ocean_level", 0.0);
        t.rot = to_f(body_rotation(cfg.num("target.pole_tilt_deg", 25) * DEG, cfg.num("target.rotation_deg", 0) * DEG));
        if (cfg.flag("target.show_host", true)) {
            SourceGPU h{};
            h.type = SRC_STAR;
            h.radius = hr;
            h.dist = d - a * L.z;
            h.cx = a * L.x;
            h.cy = a * L.y;
            Config hc;
            hc.set("h.teff", std::to_string(th));
            hc.set("h.spots", "2");
            fill_star(h, hc, "h", hr, solar);
            srcs.push_back(h);
        }
    } else if (info->target_type == "image") {
        t.type = SRC_IMAGE;
        PngImage png;
        std::string file = cfg.str("target.file");
        if (!file.empty() && file[0] != '/') file = cfg.dir() + "/" + file;
        if (!read_png(file, &png, err)) return false;
        ImageTexData img;
        img.w = png.w;
        img.h = png.h;
        img.rgba.resize((size_t)png.w * png.h * 4);
        for (size_t i = 0; i < (size_t)png.w * png.h; ++i) {
            const uint8_t* s = &png.data[i * png.channels];
            uint8_t r = s[0], g2 = png.channels >= 3 ? s[1] : s[0], b = png.channels >= 3 ? s[2] : s[0];
            img.rgba[4 * i] = r; img.rgba[4 * i + 1] = g2; img.rgba[4 * i + 2] = b; img.rgba[4 * i + 3] = 255;
        }
        info->images.push_back(img);
        double wmas = cfg.num("target.width_mas", 1.0);
        t.radius = 0.5 * wmas / phys::RAD2MAS * d;
        t.half_h = t.radius * png.h / (double)png.w;
        t.image_scale = (float)cfg.num("target.radiance", 0.05);
    } else {
        if (err) *err = "unknown target.type " + info->target_type;
        return false;
    }
    info->target_radius = t.radius;
    srcs.push_back(t);
    std::sort(srcs.begin(), srcs.end(), [](const SourceGPU& a, const SourceGPU& b) { return a.dist < b.dist; });
    S->nsrc = (int)std::min<size_t>(srcs.size(), MAX_SOURCES);
    for (int i = 0; i < S->nsrc; ++i) S->src[i] = srcs[i];

    // Focal distances
    auto focal = [&](double b) { return b * b * d / (2.0 * phys::RS_EARTH * d - b * b) / phys::AU; };
    info->focal_au = focal(phys::R_EQ);
    info->focal_atm_au = focal(phys::R_EQ + 110e3);

    // Colour
    ColorSettings& cs = info->color;
    spectrum::cie_xyz_weights(cs.wx, cs.wy, cs.wz);
    // White balance "sun": sunlight (as seen from space) renders neutral, like a camera set to
    // daylight; "none": raw sRGB with its D65 white point
    std::string wb = cfg.str("camera.white_balance", "sun");
    double xyz[3] = {0, 0, 0};
    for (int k = 0; k < NB; ++k) {
        xyz[0] += solar[k] * cs.wx[k]; xyz[1] += solar[k] * cs.wy[k]; xyz[2] += solar[k] * cs.wz[k];
    }
    double rgb[3];
    spectrum::xyz_to_srgb(xyz, rgb);
    for (int c = 0; c < 3; ++c) cs.white[c] = wb == "none" ? 1.f : (float)(rgb[c] / xyz[1]);
    return true;
}
