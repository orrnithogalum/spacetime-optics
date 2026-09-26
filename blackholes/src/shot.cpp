#include "shot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace {
bool parse_track(const std::string& s, bool log, Track* tr, std::string* err) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) continue;
        size_t c = item.find(':');
        if (c == std::string::npos) {
            if (err) *err = "track entry '" + item + "' needs time:value";
            return false;
        }
        double t = std::strtod(item.substr(0, c).c_str(), nullptr), v = std::strtod(item.substr(c + 1).c_str(), nullptr);
        tr->t.push_back(t);
        tr->v.push_back(log ? std::log(v) : v);
    }
    tr->log = log;
    return !tr->t.empty();
}

double param(const Config& c, const std::string& key, double def, double time, bool log, bool* animated = nullptr) {
    std::string tk = "track." + key;
    if (c.has(tk)) {
        Track tr;
        std::string err;
        if (parse_track(c.str(tk), log, &tr, &err)) {
            if (animated) *animated = true;
            return tr.value(time);
        }
        std::fprintf(stderr, "warning: %s\n", err.c_str());
    }
    return c.num(key, def);
}

dvec3 rotate(dvec3 v, dvec3 axis, double ang) {   // Rodrigues
    double cs = std::cos(ang), sn = std::sin(ang);
    return v * cs + cross(axis, v) * sn + axis * (dot(axis, v) * (1 - cs));
}
}  // namespace

double Track::value(double time) const {
    size_t n = t.size();
    double out;
    if (n == 1 || time <= t[0]) out = v[0];
    else if (time >= t[n - 1]) out = v[n - 1];
    else {
        size_t i = 0;
        while (i + 2 < n && time > t[i + 1]) ++i;
        // Fritsch-Carlson monotone tangents; zero tangents at the ends ease in and out
        auto sec = [&](size_t k) { return (v[k + 1] - v[k]) / (t[k + 1] - t[k]); };
        auto tan = [&](size_t k) {
            if (k == 0 || k == n - 1) return 0.0;
            double a = sec(k - 1), b = sec(k);
            if (a * b <= 0) return 0.0;
            double h0 = t[k] - t[k - 1], h1 = t[k + 1] - t[k];
            return 3.0 * (h0 + h1) / ((2.0 * h1 + h0) / a + (h1 + 2.0 * h0) / b);   // weighted harmonic mean
        };
        double h = t[i + 1] - t[i], s = (time - t[i]) / h;
        double h00 = (1 + 2 * s) * (1 - s) * (1 - s), h10 = s * (1 - s) * (1 - s), h01 = s * s * (3 - 2 * s), h11 = s * s * (s - 1);
        out = h00 * v[i] + h10 * h * tan(i) + h01 * v[i + 1] + h11 * h * tan(i + 1);
    }
    return log ? std::exp(out) : out;
}

bool build_shot(const Config& c, Shot* shot, std::string* err) {
    shot->name = c.str("shot.name", "shot");
    shot->type = c.str("shot.type", "render");
    shot->duration = c.num("shot.duration", 5);
    shot->fps = c.num("shot.fps", 30);
    shot->motion = c.str("camera.motion", "static");
    int n = std::max(1, (int)std::lround(shot->duration * shot->fps));
    if (c.has("shot.frames")) n = c.integer("shot.frames", n);
    const double D2R = PI / 180;
    for (int i = 0; i < n; ++i) {
        ShotFrame f;
        f.index = i;
        f.t = i / shot->fps;
        double t = f.t;
        f.r = param(c, "camera.r", 1e6, t, true);
        f.incl = param(c, "camera.incl", 80, t, false) * D2R;
        f.phi = param(c, "camera.phi", 0, t, false) * D2R;
        f.roll = param(c, "camera.roll", 0, t, false) * D2R;
        f.yaw = param(c, "camera.yaw", 0, t, false) * D2R;
        f.pitch = param(c, "camera.pitch", 0, t, false) * D2R;
        f.fov_in_m = c.has("camera.fov_m") || c.has("track.camera.fov_m");
        f.fov = f.fov_in_m ? param(c, "camera.fov_m", 30, t, true) : param(c, "camera.fov_deg", 40, t, true) * D2R;
        f.zoom = param(c, "camera.zoom", 1, t, true);
        f.target = param(c, "camera.target", 0, t, false);
        f.off_x = param(c, "camera.offset_x_m", 0, t, false);
        f.off_y = param(c, "camera.offset_y_m", 0, t, false);
        f.time_m = param(c, "time.t_m", 0, t, false);
        f.blur = param(c, "post.blur", 0, t, false);
        f.overlay = param(c, "post.overlay", 0, t, false);
        f.hide_direct = param(c, "post.hide_direct", 0, t, false);
        f.exposure_ev = param(c, "post.exposure_ev", 0, t, false);
        double fi = c.num("shot.fade_in", 0.0), fo = c.num("shot.fade_out", 0.0);
        double fade = param(c, "post.fade", 1, t, false);
        if (fi > 0) fade *= std::min(1.0, t / fi);
        if (fo > 0) fade *= std::min(1.0, (shot->duration - t) / fo);
        f.fade = std::max(0.0, fade);
        f.hud = param(c, "hud.alpha", 1, t, false);
        shot->frames.push_back(f);
    }
    (void)err;
    return true;
}

FrameGPU make_frame(const Shot& shot, const ShotFrame& f, double a, int W, uint32_t seed) {
    FrameGPU F{};
    F.r = f.r;
    F.th = f.incl;
    F.ph = f.phi;
    F.t = f.time_m;
    F.kp = kerr_point(a, f.r, f.incl);
    // Looking at the hole: forward = -e_r, up = -e_theta (towards the spin axis), right = e_phi
    dvec3 Fw = {-1, 0, 0}, Up = {0, -1, 0}, Rt = {0, 0, 1};
    if (f.yaw != 0) { Fw = rotate(Fw, Up, -f.yaw); Rt = rotate(Rt, Up, -f.yaw); }
    if (f.pitch != 0) { Fw = rotate(Fw, Rt, f.pitch); Up = rotate(Up, Rt, f.pitch); }
    if (f.roll != 0) { Rt = rotate(Rt, Fw, f.roll); Up = rotate(Up, Fw, f.roll); }
    F.F = Fw;
    F.R = Rt;
    F.U = Up;
    double width = f.fov_in_m ? f.fov / f.r : 2.0 * std::tan(0.5 * f.fov);
    F.pix = width / f.zoom / W;
    if (f.fov_in_m) {
        F.cx = (f.off_x + f.target * shot.target_x) / f.r;
        F.cy = (f.off_y + f.target * shot.target_y) / f.r;
    }
    F.beta = {0, 0, 0};
    if (shot.motion == "orbit") {
        // Circular Keplerian speed measured by the ZAMO (exact in the equatorial plane)
        double Om = 1.0 / (std::pow(f.r, 1.5) + a);
        double v = (Om - F.kp.omega) * F.kp.e_psi / F.kp.e_nu;
        F.beta = {0, 0, std::max(-0.98, std::min(0.98, v))};
    }
    F.seed = seed;
    return F;
}

double find_critical_point(const Shot& shot, const ShotFrame& f, double a, double psi) {
    FrameGPU F = make_frame(shot, f, a, 1, 0);
    auto fate = [&](double rho) {
        double x = rho * std::cos(psi) / f.r, y = rho * std::sin(psi) / f.r;
        dvec3 d = normalize(F.F + F.R * x + F.U * y);
        Photon p = camera_photon(a, F.kp, F.ph, F.t, d, F.beta);
        return vacuum_fate(a, p, 1e-12, 200.0, 2000000);
    };
    double lo = 0.5, hi = 15.0;   // captured at lo, escapes at hi
    if (fate(lo) != -1 || fate(hi) != 1) return 0.0;
    for (int it = 0; it < 64; ++it) {
        double mid = 0.5 * (lo + hi);
        if (fate(mid) == 1) hi = mid;
        else lo = mid;
    }
    return 0.5 * (lo + hi);
}
