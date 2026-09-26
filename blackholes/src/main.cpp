/* MAIN
- bhrt: cinematic, physically based renderer of black-hole photon rings.
- One MPI rank per GPU; a scheduler thread on rank 0 hands out frames (see FrameScheduler).
*/

#ifndef BH_NO_MPI
#include <mpi.h>
#endif
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "config.hpp"
#include "font_atlas.h"
#include "image_io.hpp"
#include "renderer.hpp"
#include "scene_build.hpp"
#include "shot.hpp"

namespace {
int g_rank = 0, g_size = 1;
bool g_threaded = false;

#ifdef BH_NO_MPI
void mpi_init(int*, char***) {}
void mpi_finalize() {}
void barrier() {}
#else
void mpi_init(int* argc, char*** argv) {
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
    g_threaded = provided >= MPI_THREAD_SERIALIZED;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
}
void mpi_finalize() { MPI_Finalize(); }
void barrier() { MPI_Barrier(MPI_COMM_WORLD); }
#endif

/* FrameScheduler
- Rank 0 runs a scheduler thread answering two-sided work requests; its own loop takes frames from
  the same atomic counter.
- SERIALIZED suffices: rank 0's main thread makes no MPI calls meanwhile.
*/
class FrameScheduler {
public:
    FrameScheduler(long first, long end) : next_(first), first_(first), end_(end) {
#ifndef BH_NO_MPI
        if (g_size > 1 && g_rank == 0 && g_threaded) thread_ = std::thread([this] { serve(); });
#endif
    }
    ~FrameScheduler() {
        if (thread_.joinable()) thread_.join();
    }
    long next() {
#ifndef BH_NO_MPI
        if (g_size > 1 && !g_threaded) return first_ + g_rank + (long)g_size * (count_++);
        if (g_size > 1 && g_rank != 0) {
            long req = 0, idx = end_;
            MPI_Send(&req, 1, MPI_LONG, 0, 101, MPI_COMM_WORLD);
            MPI_Recv(&idx, 1, MPI_LONG, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return idx;
        }
#endif
        return next_.fetch_add(1);
    }

private:
#ifndef BH_NO_MPI
    void serve() {
        int finished = 0;
        while (finished < g_size - 1) {
            long req;
            MPI_Status st;
            MPI_Recv(&req, 1, MPI_LONG, MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &st);
            long idx = next_.fetch_add(1);
            if (idx >= end_) { idx = end_; ++finished; }
            MPI_Send(&idx, 1, MPI_LONG, st.MPI_SOURCE, 102, MPI_COMM_WORLD);
        }
    }
#endif
    std::atomic<long> next_;
    long first_, end_, count_ = 0;
    std::thread thread_;
};

void mkdirs(const std::string& p) {
    std::string cur = p[0] == '/' ? "/" : "", part;
    std::stringstream ss(p);
    while (std::getline(ss, part, '/'))
        if (!part.empty()) { cur += part + "/"; mkdir(cur.c_str(), 0755); }
}
double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Text overlay
std::string fmt_int(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%.0f", std::fabs(v));
    std::string s = b, o;
    int n = (int)s.size();
    for (int i = 0; i < n; ++i) {
        o += s[i];
        if ((n - i - 1) % 3 == 0 && i != n - 1) o += ',';
    }
    return (v < 0 ? "-" : "") + o;
}
std::string fmt_g(double v, const char* f = "%.3g") {
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}
std::string fmt_uas(double uas) {
    if (uas >= 1000) return fmt_g(uas / 1000, "%.3g") + " mas";
    if (uas >= 0.9995) return fmt_g(uas, "%.3g") + " \xC2\xB5" "as";
    if (uas >= 1e-3) return fmt_g(uas * 1e3, "%.3g") + " nas";
    return fmt_g(uas * 1e6, "%.3g") + " pas";
}

struct Hud {
    std::vector<HudGlyph> glyphs;
    std::vector<HudRect> rects;
    static constexpr float ADV = 21.6f;
    float alpha = 1.f;
    static int decode(const std::string& s, size_t* i) {
        unsigned char c = s[*i];
        if (c < 0x80) { ++*i; return c >= 32 && c < 127 ? c - 32 : 0; }
        unsigned char d = *i + 1 < s.size() ? s[*i + 1] : 0;
        *i += 2;
        if (c == 0xC2 && d == 0xB5) return 95;
        if (c == 0xC2 && d == 0xB0) return 96;
        if (c == 0xC3 && d == 0x97) return 97;
        if (c == 0xC2 && d == 0xB1) return 98;
        return 0;
    }
    static int length(const std::string& s) {
        int n = 0;
        for (size_t i = 0; i < s.size();) { decode(s, &i); ++n; }
        return n;
    }
    uint32_t col(uint32_t rgba) const {
        uint32_t a = (uint32_t)(((rgba >> 24) & 255) * alpha);
        return (rgba & 0x00ffffffu) | (a << 24);
    }
    void text(float x, float y, float scale, const std::string& s, uint32_t rgba, int align = 0) {
        float w = length(s) * ADV * scale;
        if (align == 1) x -= w;
        if (align == 2) x -= 0.5f * w;
        grow(x, y + 0.12f * FONT_CELL_H * scale, x + w, y + 0.95f * FONT_CELL_H * scale);
        for (size_t i = 0; i < s.size();) {
            int g = decode(s, &i);
            if (g) glyphs.push_back({x, y, scale, g, col(rgba)});
            x += ADV * scale;
        }
    }
    void rect(int x0, int y0, int x1, int y1, uint32_t rgba) {
        rects.push_back({x0, y0, x1, y1, col(rgba), 1});
        grow((float)x0, (float)y0, (float)x1, (float)y1);
    }
    void begin_block() { bx0 = by0 = 1e9f; bx1 = by1 = -1e9f; }
    void end_block(float pad) {
        if (bx1 < bx0) return;
        rects.push_back({(int)(bx0 - pad), (int)(by0 - pad), (int)(bx1 + pad), (int)(by1 + pad), col(0x55000000u), 0});
    }

private:
    float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
    void grow(float x0, float y0, float x1, float y1) {
        bx0 = std::min(bx0, x0); by0 = std::min(by0, y0);
        bx1 = std::max(bx1, x1); by1 = std::max(by1, y1);
    }
};

constexpr uint32_t WHITE = 0xF0FFFFFFu, DIM = 0xC8D8DDE2u, ACCENT = 0xF070C8FFu;
const uint32_t ORDER_COL[5] = {0xF0FFFFFFu, 0xF0FFE640u, 0xF0F24DFFu, 0xF033D1FFu, 0xF04DFF8Cu};   // ABGR

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> o;
    std::stringstream ss(s);
    std::string it;
    while (std::getline(ss, it, sep)) o.push_back(trim(it));
    return o;
}

Hud make_hud(const Config& c, const Shot& shot, const ShotFrame& f, const FrameGPU& F, const SceneInfo& si,
             const SceneGPU& S, int W, int H) {
    Hud h;
    h.alpha = (float)(f.hud * f.fade);
    if (h.alpha <= 0.01f) return h;
    float s = H / 1080.f, m = 40 * s, pad = 10 * s;
    float big = 0.85f * s, mid = 0.6f * s, small = 0.48f * s;
    float lh = FONT_CELL_H * small * 1.12f;
    std::string style = c.str("hud.style", "visible");

    if (shot.type == "title") {
        std::vector<std::string> lines;
        for (int i = 1; i <= 6; ++i) {
            std::string k = "title.line" + std::to_string(i);
            if (c.has(k)) lines.push_back(c.str(k));
        }
        float y = 0.5f * H - 0.5f * lines.size() * FONT_CELL_H * 0.9f * s;
        for (size_t i = 0; i < lines.size(); ++i) {
            float sc = (i == 0 ? 1.6f : 0.62f) * s;
            uint32_t colr = i == 0 ? WHITE : DIM;
            h.text(0.5f * W, y, sc, lines[i], colr, 2);
            y += FONT_CELL_H * sc * (i == 0 ? 1.3f : 1.25f);
        }
        return h;
    }
    // Top-left: title block
    h.begin_block();
    float y = m;
    if (c.has("hud.title")) { h.text(m, y, mid, c.str("hud.title"), WHITE); y += FONT_CELL_H * mid * 1.1f; }
    for (const std::string& l : split(c.str("hud.lines"), '|'))
        if (!l.empty()) { h.text(m, y, small, l, DIM); y += lh; }
    h.end_block(pad);

    // Top-right: live numbers
    h.begin_block();
    y = m;
    double uas_m = si.uas_per_m;
    double width_m = f.fov_in_m ? f.fov / f.zoom : 0.0;
    if (style == "radio" || style == "zoom") {
        if (style == "zoom") {
            h.text(W - m, y, big, "zoom \xC3\x97" + fmt_int(f.zoom), WHITE, 1);
            y += FONT_CELL_H * big * 1.05f;
            h.text(W - m, y, small, "field of view " + fmt_uas(width_m * uas_m), ACCENT, 1);
            y += lh;
            h.text(W - m, y, small, "= " + fmt_g(width_m, "%.3g") + " M (gravitational radii)", DIM, 1);
            y += lh;
        } else {
            h.text(W - m, y, big, fmt_g(S.nu_obs / 1e9, "%.0f") + " GHz", WHITE, 1);
            y += FONT_CELL_H * big * 1.05f;
            double beam = c.num("post.beam_uas", 20.0) * f.blur;
            h.text(W - m, y, small, beam > 0.5 ? "resolution " + fmt_g(beam, "%.0f") + " \xC2\xB5" "as (Event Horizon Telescope)"
                                               : "resolution: unlimited", ACCENT, 1);
            y += lh;
        }
        double days = f.time_m * si.tg_s / 86400.0;
        h.text(W - m, y, small, "elapsed " + fmt_g(days, days < 10 ? "%.1f" : "%.0f") + " days", DIM, 1);
    } else {
        double km = f.r * si.rg_cm * 1e-5;
        h.text(W - m, y, big, "r = " + fmt_g(f.r, f.r < 100 ? "%.1f" : "%.0f") + " M", WHITE, 1);
        y += FONT_CELL_H * big * 1.05f;
        std::string dist = km < 1e8 ? fmt_int(km) + " km" : fmt_g(km / 1.495978707e8, km < 1.5e10 ? "%.1f" : "%.0f") + " AU";
        h.text(W - m, y, small, dist + " from the centre", ACCENT, 1);
        y += lh;
        double v = std::sqrt(dot(F.beta, F.beta));
        if (v > 1e-3) { h.text(W - m, y, small, "orbital speed " + fmt_g(v, "%.2f") + " c", DIM, 1); y += lh; }
        h.text(W - m, y, small, "inclination " + fmt_g(f.incl * 180 / PI, "%.0f") + "\xC2\xB0", DIM, 1);
    }
    h.end_block(pad);

    // Bottom-left: note
    std::vector<std::string> notes = split(c.str("hud.note"), '|');
    if (f.hide_direct > 0.5) notes = {"direct image removed: only light that circled the black hole is shown"};
    if (!notes.empty() && !notes[0].empty()) {
        h.begin_block();
        float yb = H - m - lh * notes.size();
        for (const std::string& l : notes) { h.text(m, yb, small, l, DIM); yb += lh; }
        h.end_block(pad);
    }
    // Order legend
    if (f.overlay > 0.05) {
        h.begin_block();
        float yb = H - m - lh * 2.2f, x = 0.5f * W - 4 * 140 * s * 0.5f;
        float a0 = h.alpha;
        h.alpha = a0 * (float)std::min(1.0, f.overlay * 1.5);
        h.text(0.5f * W, yb - lh, small, "photon-ring order n: half-orbits the light made around the hole", DIM, 2);
        x = 0.5f * W - 3 * 150 * s * 0.5f;
        for (int n = 1; n <= 3; ++n)
            h.text(x + (n - 1) * 150 * s, yb + 0.2f * lh, mid, n < 3 ? "n = " + std::to_string(n) : "n = 3+", ORDER_COL[n]);
        h.end_block(pad);
        h.alpha = a0;
    }
    // Bottom-right: scale bar (image-plane width known)
    if (width_m > 0 && uas_m > 0) {
        double pix_uas = width_m * uas_m / W;
        double target = pix_uas * W / 6.0, e = std::pow(10.0, std::floor(std::log10(target))), nice = e;
        for (double k : {1.0, 2.0, 5.0, 10.0})
            if (k * e <= target) nice = k * e;
        int bar = (int)std::lround(nice / pix_uas);
        int x1 = (int)(W - m), x0 = x1 - bar, yy = (int)(H - m - lh * 0.4f);
        h.begin_block();
        h.rect(x0, yy, x1, yy + std::max(2, (int)(3 * s)), 0xE0FFFFFFu);
        h.rect(x0, yy - (int)(8 * s), x0 + std::max(2, (int)(2 * s)), yy + (int)(3 * s), 0xE0FFFFFFu);
        h.rect(x1 - std::max(2, (int)(2 * s)), yy - (int)(8 * s), x1, yy + (int)(3 * s), 0xE0FFFFFFu);
        h.text(0.5f * (x0 + x1), yy - lh * 1.2f, small, fmt_uas(nice), WHITE, 2);
        h.end_block(pad);
    }
    return h;
}

std::vector<double> smooth_exposure(const std::vector<double>& lp, double key, double sigma) {
    size_t n = lp.size();
    std::vector<double> out(n, 1.0);
    for (size_t i = 0; i < n; ++i) {
        double ws = 0, acc = 0;
        int r = (int)std::ceil(3 * sigma);
        for (int d = -r; d <= r; ++d) {
            long j = (long)i + d;
            if (j < 0 || j >= (long)n || lp[j] <= 0) continue;
            double w = sigma > 0 ? std::exp(-0.5 * d * d / (sigma * sigma)) : (d == 0);
            ws += w;
            acc += w * std::log(lp[j]);
        }
        if (ws > 0) out[i] = key / std::exp(acc / ws);
    }
    return out;
}

// Self-test of the geodesic integrator against analytic Kerr results (host only)
void selftest() {
    std::printf("self-test: shadow edge from ray tracing vs analytic critical curve\n");
    for (double a : {1e-6, 0.5, 0.94, 0.99}) {
        for (double incl_deg : {90.0, 17.0}) {
            Shot sh;
            sh.motion = "static";
            ShotFrame f;
            f.r = 1e7;
            f.incl = incl_deg * PI / 180;
            f.fov_in_m = true;
            f.fov = 30;
            // Analytic: Bardeen curve, horizontal extent (beta = 0 crossings) for alpha < 0 and > 0
            double amin = 1e9, amax = -1e9;
            double rp0 = 2.0 * (1.0 + std::cos(2.0 / 3.0 * std::acos(-a))), rp1 = 2.0 * (1.0 + std::cos(2.0 / 3.0 * std::acos(a)));
            // The curve is symmetric in beta, so its extreme alphas lie on the horizontal axis
            for (int i = 0; i <= 200000; ++i) {
                double rp = rp0 + (rp1 - rp0) * i / 200000.0, al, be;
                if (critical_curve_point(a, f.incl, rp, &al, &be)) {
                    amin = std::min(amin, al);
                    amax = std::max(amax, al);
                }
            }
            // Numerical: bisection along +x (psi = 0) and -x (psi = pi); image +x is Bardeen's +alpha
            double xp = find_critical_point(sh, f, a, 0.0), xm = find_critical_point(sh, f, a, PI);
            std::printf("  a=%.2f incl=%4.0f  ray-traced edges %+.6f %+.6f   analytic %+.6f %+.6f  (M)\n", a,
                        incl_deg, -xm, xp, amin, amax);
        }
    }
    // Constants of motion: drift of R(r) - (dr/dtau)^2 over a long integration
    {
        double a = 0.94;
        KerrPoint kp = kerr_point(a, 50.0, 1.2);
        Photon p = camera_photon(a, kp, 0, 0, normalize(dvec3{-1, 0.13, 0.11}), dvec3{0, 0, 0});
        double* y = p.y, k1[6], yn[6], k7[6];
        geodesic_rhs(a, p.lam, p.eta, y, k1);
        double h = initial_step(y), worst = 0;
        int steps = 0;
        for (int it = 0; it < 200000 && steps < 3000; ++it) {
            double err = dp54_step(a, p.lam, p.eta, y, k1, h, 1e-10, yn, k7);
            if (err > 1) { h *= std::max(0.2, 0.9 * std::pow(err, -0.2)); continue; }
            for (int i = 0; i < 6; ++i) { y[i] = yn[i]; k1[i] = k7[i]; }
            h *= std::min(5.0, std::max(0.2, 0.9 * std::pow(std::max(err, 1e-30), -0.2)));
            ++steps;
            double R = radial_potential(a, p.lam, p.eta, y[0]);
            worst = std::max(worst, std::fabs(R - y[1] * y[1]) / std::pow(y[0], 4.0));
            if (y[0] < horizon_radius(a) + 0.01 || y[0] > 1e4) break;
        }
        std::printf("  radial constraint |R - r'^2| / r^4, worst over %d steps (no projection): %.2e\n", steps, worst);
    }
}

void usage() {
    std::fprintf(stderr, "usage: bhrt shot.ini [--out DIR] [--set section.key=value ...] [--frames A:B] [--list] [--selftest]\n");
}
}  // namespace

int main(int argc, char** argv) {
    mpi_init(&argc, &argv);
    double t_start = wall();
    if (argc < 2) { if (g_rank == 0) usage(); mpi_finalize(); return 1; }
    if (std::string(argv[1]) == "--selftest") { if (g_rank == 0) selftest(); mpi_finalize(); return 0; }
    Config cfg;
    std::string err;
    if (!cfg.load(argv[1], &err)) { if (g_rank == 0) std::fprintf(stderr, "bhrt: %s\n", err.c_str()); mpi_finalize(); return 1; }
    bool list = false;
    int fa = 0, fb = -1;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--set" && i + 1 < argc) {
            std::string kv = argv[++i];
            size_t eq = kv.find('=');
            if (eq != std::string::npos) cfg.set(trim(kv.substr(0, eq)), trim(kv.substr(eq + 1)));
        } else if (a == "--out" && i + 1 < argc) cfg.set("output.dir", argv[++i]);
        else if (a == "--frames" && i + 1 < argc) std::sscanf(argv[++i], "%d:%d", &fa, &fb);
        else if (a == "--list") list = true;
        else { if (g_rank == 0) usage(); mpi_finalize(); return 1; }
    }

    SceneGPU S;
    SceneTables T;
    SceneInfo si;
    Shot shot;
    if (!build_scene(cfg, &S, &T, &si, &err) || !build_shot(cfg, &shot, &err)) {
        if (g_rank == 0) std::fprintf(stderr, "bhrt: %s\n", err.c_str());
        mpi_finalize();
        return 1;
    }
    RenderSettings rs;
    rs.W = cfg.integer("camera.width", 1920);
    rs.H = cfg.integer("camera.height", 1080);
    rs.spp = cfg.integer("render.spp", 8);
    rs.spp_max = cfg.integer("render.spp_max", 128);
    rs.target_err = (float)cfg.num("render.target_error", 0.03);
    int nframes = (int)shot.frames.size();
    if (fb < 0 || fb > nframes) fb = nframes;
    fa = std::max(0, std::min(fa, fb));

    // Zoom target on the critical curve (deterministic, computed by every rank)
    if (shot.type == "render" && cfg.has("camera.target_psi_deg")) {
        double psi = cfg.num("camera.target_psi_deg", 0) * PI / 180;
        ShotFrame f0 = shot.frames[0];
        double rho = find_critical_point(shot, f0, S.a, psi) + cfg.num("camera.target_offset_m", 0.0);
        shot.target_x = rho * std::cos(psi);
        shot.target_y = rho * std::sin(psi);
        if (g_rank == 0) std::printf("  zoom target: critical curve at psi = %.1f deg, rho = %.12f M\n", psi * 180 / PI, rho);
    }
    if (g_rank == 0) {
        std::printf("bhrt | %d rank(s) | shot '%s' (%s) %d frames %dx%d spp %d..%d\n", g_size, shot.name.c_str(),
                    shot.type.c_str(), nframes, rs.W, rs.H, rs.spp, rs.spp_max);
        std::printf("  a = %.4f, r_h = %.4f M, r_isco = %.4f M, %s, M = %.3g Msun (GM/c^2 = %.4g km, %.3g uas)\n", S.a, S.rh,
                    S.r_isco, S.mode == MODE_RADIO ? "radio" : "visible", si.mass_msun, si.rg_cm * 1e-5, si.uas_per_m);
        if (S.disk_on) std::printf("  thin disk %.2f..%.1f M, T_max %.0f K\n", S.disk_rin, S.disk_rout, si.disk_tmax);
        if (list)
            for (const ShotFrame& f : shot.frames)
                std::printf("  frame %4d t=%6.2f r=%.4g incl=%.1f phi=%.1f fov=%.4g zoom=%.4g target=%.2f time=%.1f\n", f.index, f.t,
                            f.r, f.incl * 180 / PI, f.phi * 180 / PI, f.fov_in_m ? f.fov : f.fov * 180 / PI, f.zoom, f.target, f.time_m);
        std::fflush(stdout);
    }
    if (list) { mpi_finalize(); return 0; }

    int local = g_rank;
#ifndef BH_NO_MPI
    {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &local);
        MPI_Comm_free(&node);
    }
#endif
    Renderer R;
    SkyTexture sky;
    bool is_title = shot.type == "title";
    int ndev = bh_device_count();
    if (ndev <= 0) { std::fprintf(stderr, "[rank %d] no CUDA device\n", g_rank); mpi_finalize(); return 1; }
    if (is_title) {
        S.sky_on = 0;
        R.init(local % ndev, S, T, nullptr, rs, &err);   // tone mapping + text run on the GPU
    } else {
        if (S.sky_on) {
            float scale = 1;
            if (load_sky(si.sky_file, &sky, &scale, &err)) S.sky_scale *= scale;
            else {
                if (g_rank == 0) std::fprintf(stderr, "warning: %s -- black sky\n", err.c_str());
                S.sky_on = 0;
            }
        }
        double t0 = wall();
        R.init(local % ndev, S, T, &sky, rs, &err);
        sky.data.clear();
        sky.data.shrink_to_fit();
        if (g_rank == 0) std::printf("  GPU init %.2f s\n", wall() - t0);
        // Normalise the hot flow to an observed flux density (e.g. 0.6 Jy for M87* at 230 GHz)
        if (S.mode == MODE_RADIO && si.radio_flux_target_jy > 0) {
            double n0 = S.n0;
            if (g_rank == 0) {
                ShotFrame f0 = shot.frames[0];
                f0.zoom = 1;
                f0.target = 0;
                f0.off_x = f0.off_y = 0;
                f0.fov = cfg.num("flow.normalize_fov_m", 40.0);
                f0.fov_in_m = true;
                const int n = 160;
                FrameGPU F = make_frame(shot, f0, S.a, n, 1);
                double pix_sr = std::pow(f0.fov / n * si.rg_cm / si.distance_cm, 2.0);
                double x0 = std::log(n0), f0v = std::log(std::max(R.radio_flux(F, n, pix_sr), 1e-30));
                double target = std::log(si.radio_flux_target_jy);
                double x1 = x0 + (target - f0v) / 1.5, f1v = 0;
                for (int it = 0; it < 8; ++it) {
                    S.n0 = std::exp(x1);
                    R.update_scene(S);
                    f1v = std::log(std::max(R.radio_flux(F, n, pix_sr), 1e-30));
                    if (std::fabs(f1v - target) < 0.01) break;
                    double slope = (f1v - f0v) / (x1 - x0);
                    if (!(slope > 0.2)) slope = 1.5;
                    x0 = x1; f0v = f1v;
                    x1 = x1 + (target - f1v) / slope;
                }
                n0 = std::exp(x1);
                std::printf("  flow normalised: n0 = %.4g cm^-3 for %.3g Jy (got %.3g Jy)\n", n0, si.radio_flux_target_jy, std::exp(f1v));
            }
#ifndef BH_NO_MPI
            MPI_Bcast(&n0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
            S.n0 = n0;
            R.update_scene(S);
        }
    }
    std::string out = cfg.str("output.dir", "out/" + shot.name);
    if (g_rank == 0) mkdirs(out + "/frames");
    barrier();

    // Phase 1: render
    struct Done { int index; HdrFrame hdr; FrameGPU F; };
    std::vector<Done> done;
    double percentile = cfg.num("post.exposure_percentile", 0.995);
    double beam_px_uas = cfg.num("post.beam_uas", 20.0);
    long long my_samples = 0, my_limited = 0;
    double t_render = wall();
    {
        FrameScheduler sched(fa, fb);
        for (;;) {
            long idx = sched.next();
            if (idx >= fb) break;
            const ShotFrame& f = shot.frames[idx];
            Done d;
            d.index = (int)idx;
            d.F = make_frame(shot, f, S.a, rs.W, 0x9E3779B9u * (uint32_t)(idx + 1));
            if (is_title) {
                d.hdr.W = rs.W;
                d.hdr.H = rs.H;
                d.hdr.px.assign((size_t)rs.W * rs.H * 4, 0.f);
                done.push_back(std::move(d));
                continue;
            }
            PostParams pp;
            pp.bloom = (float)cfg.num("post.bloom", 0.0);
            pp.hide_direct = (float)f.hide_direct;
            if (S.mode == MODE_RADIO && f.blur > 0 && f.fov_in_m) {
                double pix_uas = f.fov / f.zoom * si.uas_per_m / rs.W;
                pp.blur_sigma_px = f.blur * beam_px_uas / 2.3548 / pix_uas;
            }
            RenderStats st;
            double t0 = wall();
            R.render(d.F, pp, percentile, &d.hdr, &st);
            my_samples += st.samples;
            my_limited += st.step_limited;
            if (S.mode == MODE_RADIO && cfg.flag("render.order_stats", false)) {
                double s[4] = {0, 0, 0, 0};
                for (size_t i = 0; i < d.hdr.px.size(); i += 4)
                    for (int k = 0; k < 4; ++k) s[k] += d.hdr.px[i + k];
                std::printf("[rank %3d] frame %4ld order fractions: n=0 %.3f  n=1 %.3f  n=2 %.3f  n>=3 %.3f\n", g_rank, idx,
                            (s[0] - s[1] - s[2] - s[3]) / s[0], s[1] / s[0], s[2] / s[0], s[3] / s[0]);
            }
            std::printf("[rank %3d] %s frame %4ld  %6.1f Msamples  %7.1f ms  (step-limited %lld)\n", g_rank, shot.name.c_str(), idx,
                        st.samples * 1e-6, (wall() - t0) * 1e3, st.step_limited);
            std::fflush(stdout);
            done.push_back(std::move(d));
        }
    }
    barrier();
    double t_render_end = wall();

    // Exposure
    std::vector<double> lp(nframes, 0.0), mine(nframes, 0.0);
    for (const Done& d : done) mine[d.index] = d.hdr.lum_percentile;
#ifndef BH_NO_MPI
    MPI_Allreduce(mine.data(), lp.data(), nframes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    lp = mine;
#endif
    double key = cfg.num("post.exposure_key", S.mode == MODE_RADIO ? 0.92 : 0.8);
    double sigma = std::min(cfg.num("post.exposure_smooth", 4.0), nframes / 30.0);
    std::vector<double> ex = smooth_exposure(lp, key, sigma);
    if (cfg.str("post.exposure", "auto") != "auto") ex.assign(nframes, key / cfg.num("post.exposure", 1.0));

    // Phase 2: tone map, HUD, write
    std::vector<std::future<void>> writes;
    for (Done& d : done) {
        const ShotFrame& f = shot.frames[d.index];
        ToneParams tp;
        tp.mode = S.mode;
        tp.exposure = (float)(ex[d.index] * std::pow(2.0, f.exposure_ev));
        tp.colormap = cfg.str("post.colormap", "afmhot") == "inferno" ? 1 : 0;
        tp.gamma = (float)cfg.num("post.gamma", 1.0);
        tp.overlay = (float)f.overlay;
        tp.hide_direct = (float)f.hide_direct;
        tp.fade = (float)f.fade;
        tp.vignette = (float)cfg.num("post.vignette", 0.15);
        tp.grain = (float)cfg.num("post.grain", 0.006);
        tp.seed = 0x51ED27u + (uint32_t)d.index * 7919u;
        Hud h;
        if (cfg.flag("hud.enabled", true)) h = make_hud(cfg, shot, f, d.F, si, S, d.hdr.W, d.hdr.H);
        auto rgb = std::make_shared<std::vector<uint8_t>>();
        R.tonemap(d.hdr, tp, h.glyphs, h.rects, rgb.get());
        char name[64];
        std::snprintf(name, sizeof name, "/frames/frame_%05d.png", d.index);
        std::string path = out + name;
        int W = d.hdr.W, H = d.hdr.H;
        writes.push_back(std::async(std::launch::async, [path, W, H, rgb]() { write_png(path, W, H, 3, rgb->data()); }));
        if (writes.size() > 16) { writes.front().get(); writes.erase(writes.begin()); }
        d.hdr.px.clear();
        d.hdr.px.shrink_to_fit();
    }
    for (auto& w : writes) w.get();

    long long all_samples = my_samples, all_lim = my_limited;
#ifndef BH_NO_MPI
    MPI_Reduce(&my_samples, &all_samples, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_limited, &all_lim, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    if (g_rank == 0) {
        FILE* m = std::fopen((out + "/meta.json").c_str(), "w");
        if (m) {
            std::fprintf(m, "{\"shot\": \"%s\", \"frames\": %d, \"fps\": %g, \"width\": %d, \"height\": %d}\n", shot.name.c_str(),
                         nframes, shot.fps, rs.W, rs.H);
            std::fclose(m);
        }
        double tr = t_render_end - t_render;
        std::printf("done: %d frames, %.3g samples (%lld step-limited) in %.1f s render (%.3g Msamples/s), %.1f s total\n",
                    fb - fa, (double)all_samples, all_lim, tr, all_samples / std::max(tr, 1e-9) * 1e-6, wall() - t_start);
    }
    mpi_finalize();
    return 0;
}
