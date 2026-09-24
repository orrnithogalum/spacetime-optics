#include "scene_build.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "colour.hpp"

namespace {
// Circular equatorial prograde orbits (Bardeen, Press & Teukolsky 1972), M = 1
double orb_E(double a, double r) {
    double sr = std::sqrt(r);
    return (r * sr - 2.0 * sr + a) / (std::pow(r, 0.75) * std::sqrt(r * sr - 3.0 * sr + 2.0 * a));
}
double orb_L(double a, double r) {
    double sr = std::sqrt(r);
    return (r * r - 2.0 * a * sr + a * a) / (std::pow(r, 0.75) * std::sqrt(r * sr - 3.0 * sr + 2.0 * a));
}
double orb_Om(double a, double r) { return 1.0 / (r * std::sqrt(r) + a); }

/* nt_flux
- Novikov-Thorne / Page-Thorne flux for Mdot = 1, M = 1: F(r) = -Om_r / (4 pi r (E - Om L)^2) *
  Int_{r_isco}^{r} (E - Om L) L_r dr'
*/
std::vector<double> nt_flux(double a, const std::vector<double>& r) {
    double risco = isco_radius(a);
    std::vector<double> F(r.size(), 0.0);
    // Fine integration grid from the ISCO
    const int N = 200000;
    double rmax = r.back();
    double lr0 = std::log(risco), lr1 = std::log(rmax);
    std::vector<double> rg(N), I(N, 0.0);
    for (int i = 0; i < N; ++i) rg[i] = std::exp(lr0 + (lr1 - lr0) * i / (N - 1));
    auto integrand = [&](double x) {
        double h = 1e-6 * x;
        double Lr = (orb_L(a, x + h) - orb_L(a, x - h)) / (2 * h);
        return (orb_E(a, x) - orb_Om(a, x) * orb_L(a, x)) * Lr;
    };
    double prev = integrand(rg[0] * (1 + 1e-9));
    for (int i = 1; i < N; ++i) {
        double cur = integrand(rg[i]);
        I[i] = I[i - 1] + 0.5 * (prev + cur) * (rg[i] - rg[i - 1]);
        prev = cur;
    }
    for (size_t j = 0; j < r.size(); ++j) {
        double x = r[j];
        if (x <= risco) continue;
        double lx = std::log(x);
        double fi = (lx - lr0) / (lr1 - lr0) * (N - 1);
        int i = std::min(std::max((int)fi, 0), N - 2);
        double Ix = I[i] + (I[i + 1] - I[i]) * (fi - i);
        double Omr = -1.5 * std::sqrt(x) / std::pow(x * std::sqrt(x) + a, 2);
        double EmOL = orb_E(a, x) - orb_Om(a, x) * orb_L(a, x);
        F[j] = -Omr / (4.0 * PI * x * EmOL * EmOL) * Ix;
    }
    return F;
}
}  // namespace

bool load_sky(const std::string& path, SkyTexture* sky, float* scale, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open sky texture " + path + " (run scripts/fetch_sky.sh + prepare)";
        return false;
    }
    uint32_t hdr[3];
    float sc;
    if (std::fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x4b534842u || std::fread(&sc, 4, 1, f) != 1) {
        std::fclose(f);
        if (err) *err = "bad sky texture " + path;
        return false;
    }
    sky->w = (int)hdr[1];
    sky->h = (int)hdr[2];
    sky->data.resize((size_t)sky->w * sky->h * 2);
    size_t got = std::fread(sky->data.data(), 2, sky->data.size(), f);
    std::fclose(f);
    *scale = sc;
    return got == sky->data.size();
}

bool build_scene(const Config& c, SceneGPU* S, SceneTables* T, SceneInfo* info, std::string* err) {
    std::memset(S, 0, sizeof(SceneGPU));
    double a = c.num("bh.spin", 0.94);
    if (a <= 0.0) a = 1e-6;   // the formulas use 1/a; this is Schwarzschild to 1e-6
    if (a > 0.9999) a = 0.9999;
    S->a = a;
    S->rh = horizon_radius(a);
    S->r_isco = isco_radius(a);
    S->tol = c.num("render.tolerance", 1e-9);
    S->max_steps = c.integer("render.max_steps", 20000);
    S->r_escape = c.num("render.r_escape", 2e4);
    S->r_stop = S->rh + c.num("render.horizon_margin", 0.01);
    S->step_frac = c.num("render.step_fraction", 0.04);
    std::string mode = c.str("render.mode", "visible");
    S->mode = mode == "radio" ? MODE_RADIO : MODE_VISIBLE;
    S->seed = (uint32_t)c.integer("render.seed", 1) * 2654435761u;

    info->mass_msun = c.num("bh.mass_msun", 6.5e9);
    double dist_pc = c.has("bh.distance_mpc") ? c.num("bh.distance_mpc", 16.8) * 1e6 : c.num("bh.distance_pc", 16.8e6);
    info->distance_cm = dist_pc * cgs::PC;
    info->rg_cm = cgs::G * info->mass_msun * cgs::MSUN / (cgs::C * cgs::C);
    info->tg_s = info->rg_cm / cgs::C;
    info->uas_per_m = info->rg_cm / info->distance_cm * RAD2UAS;

    // Thin disk
    S->disk_on = c.flag("disk.enabled", S->mode == MODE_VISIBLE) ? 1 : 0;
    S->disk_rin = std::max(c.num("disk.r_in", 0.0), S->r_isco);
    S->disk_rout = c.num("disk.r_out", 40.0);
    S->disk_turb_amp = (float)c.num("disk.turbulence", 0.45);
    S->disk_turb_scale = (float)c.num("disk.turbulence_scale", 0.35);
    S->disk_tcorr = (float)c.num("disk.correlation_orbits", 0.6);
    S->disk_ld = c.flag("disk.limb_darkening", true) ? 1.f : 0.f;
    S->disk_fcol = (float)c.num("disk.color_correction", 1.0);
    {
        const int N = 1024;
        double l0 = std::log(S->r_isco * 1.0001), l1 = std::log(S->disk_rout * 1.05);
        std::vector<double> r(N);
        for (int i = 0; i < N; ++i) r[i] = std::exp(l0 + (l1 - l0) * i / (N - 1));
        std::vector<double> F = nt_flux(a, r);
        // Physical scale: F_phys = F * Mdot c^2 / r_g^2, Mdot = f_Edd L_Edd / (eta c^2)
        double eta = 1.0 - orb_E(a, S->r_isco);
        double LEdd = 1.26e38 * info->mass_msun;
        double fedd = c.num("disk.eddington_ratio", 0.1);
        double Mdot = fedd * LEdd / (eta * cgs::C * cgs::C);
        const double sigma = 5.670374419e-5;
        double fmax = *std::max_element(F.begin(), F.end());
        double Tmax_phys = std::pow(fmax * Mdot * cgs::C * cgs::C / (info->rg_cm * info->rg_cm) / sigma, 0.25);
        double Tmax = c.num("disk.t_max", Tmax_phys);
        info->disk_tmax = Tmax;
        info->disk_mdot_edd = fedd * std::pow(Tmax / Tmax_phys, 4.0);
        T->disk_lnT.resize(N);
        for (int i = 0; i < N; ++i) {
            double Ti = Tmax * std::pow(std::max(F[i], 1e-300) / fmax, 0.25);
            T->disk_lnT[i] = (float)std::log(std::max(Ti, 1.0));
        }
        S->disk_n = N;
        S->disk_lr0 = l0;
        S->disk_dlr = (l1 - l0) / (N - 1);
    }

    // Hot flow
    S->flow_on = c.flag("flow.enabled", S->mode == MODE_RADIO) ? 1 : 0;
    S->n0 = c.num("flow.n0", 1e5);
    S->n_index = c.num("flow.n_index", 1.1);
    S->theta0 = c.num("flow.theta0", 25.0);
    S->theta_index = c.num("flow.theta_index", 0.84);
    S->h_r = c.num("flow.h_over_r", 0.4);
    S->beta_p = c.num("flow.beta", 10.0);
    S->r_flow_max = c.num("flow.r_max", 60.0);
    S->nu_obs = c.num("flow.nu_ghz", 230.0) * 1e9;
    S->rg_cm = info->rg_cm;
    S->flow_kep = c.num("flow.keplerian_fraction", 0.9);
    S->flow_turb_amp = (float)c.num("flow.turbulence", 0.7);
    S->flow_turb_scale = (float)c.num("flow.turbulence_scale", 0.45);
    S->flow_tcorr = (float)c.num("flow.correlation_orbits", 0.5);
    info->radio_flux_target_jy = c.num("flow.flux_jy", 0.0);
    {
        const int N = 512;
        double l0 = std::log(0.02), l1 = std::log(1e4);
        T->k2_ln.resize(N);
        for (int i = 0; i < N; ++i) {
            double th = std::exp(l0 + (l1 - l0) * i / (N - 1));
            T->k2_ln[i] = (float)std::log(std::cyl_bessel_k(2.0, 1.0 / th));
        }
        S->k2_n = N;
        S->k2_l0 = (float)l0;
        S->k2_dl = (float)((l1 - l0) / (N - 1));
    }

    // Sky
    S->sky_on = c.flag("sky.enabled", S->mode == MODE_VISIBLE) ? 1 : 0;
    info->sky_file = c.str("sky.file", "data/sky/sky.bhsk");
    if (!info->sky_file.empty() && info->sky_file[0] != '/') {
        FILE* f = std::fopen(info->sky_file.c_str(), "rb");
        if (f) std::fclose(f);
        else info->sky_file = c.dir() + "/" + info->sky_file;
    }
    S->sky_scale = (float)c.num("sky.brightness", 1.0);
    {
        // Black-hole frame -> galactic: the direction straight behind the hole as seen from a
        // camera at (incl, phi) points at galactic (l, b); 'roll' turns the sky around it
        double l = c.num("sky.l_deg", 0.0) * PI / 180, b = c.num("sky.b_deg", 0.0) * PI / 180;
        double incl = c.num("camera.incl", 80.0) * PI / 180, ph = c.num("camera.phi", 0.0) * PI / 180;
        double roll = c.num("sky.roll_deg", 0.0) * PI / 180;
        dvec3 behind = -dvec3{std::sin(incl) * std::cos(ph), std::sin(incl) * std::sin(ph), std::cos(incl)};
        dvec3 up = normalize(dvec3{0, 0, 1} - behind * dot(dvec3{0, 0, 1}, behind));
        if (!(length(up) > 0.5)) up = {1, 0, 0};
        dvec3 side = cross(up, behind);
        dvec3 gt = {std::cos(b) * std::cos(l), std::cos(b) * std::sin(l), std::sin(b)};
        dvec3 gn = normalize(dvec3{0, 0, 1} - gt * dot(dvec3{0, 0, 1}, gt));
        dvec3 gs = cross(gn, gt);
        dvec3 gu = gn * std::cos(roll) + gs * std::sin(roll);
        dvec3 gv = cross(gu, gt);
        // M maps behind->gt, up->gu, side->gv:  M = gt behind^T + gu up^T + gv side^T
        auto row = [&](int i) {
            auto comp = [&](dvec3 v) { return i == 0 ? v.x : i == 1 ? v.y : v.z; };
            return dvec3{comp(gt) * behind.x + comp(gu) * up.x + comp(gv) * side.x,
                         comp(gt) * behind.y + comp(gu) * up.y + comp(gv) * side.y,
                         comp(gt) * behind.z + comp(gu) * up.z + comp(gv) * side.z};
        };
        S->sky_rot = {row(0), row(1), row(2)};
    }

    // Colour table
    float bl0, bdl;
    T->bb_xyz = colour::blackbody_xyz_table(&bl0, &bdl);
    S->bb_n = colour::BB_N;
    S->bb_l0 = bl0;
    S->bb_dl = bdl;
    (void)err;
    return true;
}
