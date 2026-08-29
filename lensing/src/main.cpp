/* MAIN
- lensim: what a telescope sees from Earth's gravitational focal line.
- One MPI rank per GPU.
- Frames are handed out dynamically by a scheduler thread on rank 0, rendered to HDR in host memory,
  then -- once every frame's luminance statistics are known -- exposure is smoothed along the
  sequence, and each rank tone-maps and writes its own frames.
*/

#ifndef LENSIM_NO_MPI
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

namespace {

int g_rank = 0, g_size = 1;
bool g_threaded_mpi = false;

#ifdef LENSIM_NO_MPI
void mpi_init(int*, char***) {}
void mpi_finalize() {}
#else
void mpi_init(int* argc, char*** argv) {
    // SERIALIZED is enough: while the scheduler thread runs, rank 0's main thread makes no MPI
    // calls (and many UCX builds do not support MPI_THREAD_MULTIPLE)
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
    g_threaded_mpi = provided >= MPI_THREAD_SERIALIZED;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
}
void mpi_finalize() { MPI_Finalize(); }
#endif

/* FrameScheduler
- Dynamic frame distribution.
- Rank 0 runs a scheduler thread that answers work requests from the other ranks with plain
  two-sided messages, while its own render loop takes frames from the same atomic counter.
- (One-sided MPI_Fetch_and_op would be the textbook tool, but Open MPI 4.1 over UCX does not
  reliably progress remote atomics between nodes.) Without MPI_THREAD_SERIALIZED the frames are
  dealt round-robin instead.
*/
class FrameScheduler {
public:
    FrameScheduler(long first, long end) : next_(first), first_(first), end_(end) {
#ifndef LENSIM_NO_MPI
        if (g_size > 1 && g_rank == 0 && g_threaded_mpi) thread_ = std::thread([this] { serve(); });
#endif
    }
    ~FrameScheduler() {
        if (thread_.joinable()) thread_.join();
    }
    long next() {
#ifndef LENSIM_NO_MPI
        if (g_size > 1 && !g_threaded_mpi) return first_ + g_rank + (long)g_size * (count_++);
        if (g_size > 1 && g_rank != 0) {
            long req = 0, idx = end_;
            MPI_Send(&req, 1, MPI_LONG, 0, TAG_REQ, MPI_COMM_WORLD);
            MPI_Recv(&idx, 1, MPI_LONG, 0, TAG_REP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return idx;
        }
#endif
        return next_.fetch_add(1);
    }

private:
#ifndef LENSIM_NO_MPI
    void serve() {
        int finished = 0;
        while (finished < g_size - 1) {
            long req;
            MPI_Status st;
            MPI_Recv(&req, 1, MPI_LONG, MPI_ANY_SOURCE, TAG_REQ, MPI_COMM_WORLD, &st);
            long idx = next_.fetch_add(1);
            if (idx >= end_) {
                idx = end_;
                ++finished;   // that rank stops after this reply
            }
            MPI_Send(&idx, 1, MPI_LONG, st.MPI_SOURCE, TAG_REP, MPI_COMM_WORLD);
        }
    }
#endif
    static constexpr int TAG_REQ = 101, TAG_REP = 102;
    std::atomic<long> next_;
    long first_, end_;
    long count_ = 0;
    std::thread thread_;
};

void mkdirs(const std::string& p) {
    std::string cur;
    std::stringstream ss(p);
    std::string part;
    if (!p.empty() && p[0] == '/') cur = "/";
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        cur += part + "/";
        mkdir(cur.c_str(), 0755);
    }
}

double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Frames
struct FrameSpec {
    int index = 0, id = 0, il = 0;
    double D = 0;              // m
    double x = 0, y = 0;       // lateral offset (m)
    double fov_half = 0;       // rad (horizontal half-width)
    double tele_pix = 0;       // rad
    double t = 0;              // path time (s)
    double jd = 0;             // epoch of this frame (time-lapse); 0 = scene epoch
};

struct Sequence {
    std::string mode;
    int nd = 1, nl = 1;
    double fps = 30;
    std::vector<FrameSpec> frames;
};

double auto_fov_half(double D, const SceneInfo& si, double lat, double margin) {
    double th_earth = (phys::R_EQ + 100e3) / D;
    double d = si.target_dist;
    double thE = einstein_angle(D, d);
    double beta = si.target_radius / (D + d) + lat * d / (D * (D + d));
    beta = std::min(beta, 1.5 * thE);
    double th_out = 0.5 * (beta + std::sqrt(beta * beta + 4 * thE * thE));
    return margin * std::max(th_earth, th_out);
}

bool build_sequence(const Config& c, const SceneInfo& si, const RenderSettings& rs, Sequence* seq, std::string* err) {
    seq->mode = c.str("sweep.mode", "sweep");
    double aspect = (double)rs.W / rs.H;
    std::string fovs = c.str("camera.fov", "auto");
    double margin = c.num("camera.fov_margin", 1.35);
    auto fov_for = [&](double D, double lat) {
        if (fovs == "auto") {
            double h = auto_fov_half(D, si, lat, margin);
            return aspect >= 1 ? h * aspect : h;   // fit the vertical extent
        }
        return 0.5 * std::strtod(fovs.c_str(), nullptr) / phys::RAD2MAS;
    };
    auto push = [&](FrameSpec f) {
        f.index = (int)seq->frames.size();
        f.fov_half = fov_for(f.D, std::hypot(f.x, f.y));
        if (rs.tele.enabled) {
            std::string tf = c.str("telescope.fov_mas", "auto");
            double full = tf == "auto" ? std::max(2 * f.fov_half, 14.0 * 780e-9 / rs.tele.aperture)
                                       : std::strtod(tf.c_str(), nullptr) / phys::RAD2MAS;
            f.tele_pix = full / rs.tele.W;
        }
        seq->frames.push_back(f);
    };
    const double jd0 = si.epoch.jd_utc;
    if (seq->mode == "sweep") {
        double hpf = c.num("sweep.hours_per_frame", 0);
        double d0 = c.num("sweep.d_min_au", 4000), d1 = c.num("sweep.d_max_au", 80000);
        seq->nd = std::max(1, c.integer("sweep.d_steps", 240));
        seq->nl = std::max(1, c.integer("sweep.lat_steps", 1));
        double lmax = c.num("sweep.lat_max_km", 0) * 1e3, ang = c.num("sweep.lat_angle_deg", 0) * phys::PI / 180;
        for (int il = 0; il < seq->nl; ++il)
            for (int id = 0; id < seq->nd; ++id) {
                FrameSpec f;
                f.id = id;
                f.il = il;
                double u = seq->nd > 1 ? (double)id / (seq->nd - 1) : 0.0;
                f.D = d0 * std::pow(d1 / d0, u) * phys::AU;
                double l = seq->nl > 1 ? -lmax + 2 * lmax * il / (seq->nl - 1) : c.num("sweep.lat_km", 0) * 1e3;
                f.x = l * std::cos(ang);
                f.y = l * std::sin(ang);
                if (hpf != 0) f.jd = jd0 + id * hpf / 24.0;
                push(f);
            }
    } else if (seq->mode == "path") {
        // Keyframes: "t D_au x_km y_km; t D_au x_km y_km; ..."
        struct K { double t, D, x, y; };
        std::vector<K> ks;
        for (const std::string& item : c.list("sweep.keyframes", ';')) {
            K k{};
            if (std::sscanf(item.c_str(), "%lf %lf %lf %lf", &k.t, &k.D, &k.x, &k.y) < 2) {
                if (err) *err = "bad keyframe '" + item + "'";
                return false;
            }
            ks.push_back(k);
        }
        if (ks.size() < 2) { if (err) *err = "path mode needs >= 2 keyframes"; return false; }
        seq->fps = c.num("sweep.fps", 30);
        double hps = c.num("sweep.hours_per_second", 0);   // time-lapse: Earth turns while we fly
        int n = (int)std::ceil(ks.back().t * seq->fps) + 1;
        seq->nd = n;
        for (int i = 0; i < n; ++i) {
            double t = std::min(i / seq->fps, ks.back().t);
            size_t s = 0;
            while (s + 2 < ks.size() && t > ks[s + 1].t) ++s;
            double u = (t - ks[s].t) / std::max(ks[s + 1].t - ks[s].t, 1e-9);
            u = std::min(std::max(u, 0.0), 1.0);
            u = u * u * (3 - 2 * u);   // ease in/out between keyframes
            FrameSpec f;
            f.id = i;
            f.t = t;
            f.D = std::exp(std::log(ks[s].D) + (std::log(ks[s + 1].D) - std::log(ks[s].D)) * u) * phys::AU;
            f.x = (ks[s].x + (ks[s + 1].x - ks[s].x) * u) * 1e3;
            f.y = (ks[s].y + (ks[s + 1].y - ks[s].y) * u) * 1e3;
            if (hps != 0) f.jd = jd0 + t * hps / 24.0;
            push(f);
        }
    } else if (seq->mode == "still") {
        FrameSpec f;
        f.D = c.num("sweep.d_au", 20000) * phys::AU;
        f.x = c.num("sweep.x_km", 0) * 1e3;
        f.y = c.num("sweep.y_km", 0) * 1e3;
        push(f);
    } else {
        if (err) *err = "unknown sweep.mode " + seq->mode;
        return false;
    }
    return true;
}

// HUD
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

std::string fmt_angle(double rad) {
    double mas = rad * phys::RAD2MAS;
    char b[64];
    if (mas < 0.001) std::snprintf(b, sizeof b, "%.3g nas", mas * 1e6);
    else if (mas < 0.9995) std::snprintf(b, sizeof b, "%.0f \xC2\xB5" "as", mas * 1e3);
    else if (mas < 999.5) std::snprintf(b, sizeof b, "%.3g mas", mas);
    else std::snprintf(b, sizeof b, "%.3g arcsec", mas * 1e-3);
    return b;
}

struct Hud {
    std::vector<HudGlyph> glyphs;
    std::vector<HudRect> rects;
    static constexpr float ADV = 21.6f;

    static int decode(const std::string& s, size_t* i) {
        unsigned char c = s[*i];
        if (c < 0x80) { ++*i; return c >= 32 && c < 127 ? c - 32 : 0; }
        unsigned char d = *i + 1 < s.size() ? s[*i + 1] : 0;
        *i += 2;
        if (c == 0xC2 && d == 0xB5) return 95;   // µ
        if (c == 0xC2 && d == 0xB0) return 96;   // °
        if (c == 0xC3 && d == 0x97) return 97;   // ×
        if (c == 0xC2 && d == 0xB1) return 98;   // ±
        return 0;
    }
    static int length(const std::string& s) {
        int n = 0;
        for (size_t i = 0; i < s.size();) { decode(s, &i); ++n; }
        return n;
    }
    // Align: 0 left, 1 right, 2 centre. y is the top of the text cell
    void text(float x, float y, float scale, const std::string& s, uint32_t rgba, int align = 0) {
        float w = length(s) * ADV * scale;
        if (align == 1) x -= w;
        if (align == 2) x -= 0.5f * w;
        grow(x, y + 0.12f * FONT_CELL_H * scale, x + w, y + 0.95f * FONT_CELL_H * scale);
        for (size_t i = 0; i < s.size();) {
            int g = decode(s, &i);
            if (g) glyphs.push_back({x, y, scale, g, rgba});
            x += ADV * scale;
        }
    }
    void rect(int x0, int y0, int x1, int y1, uint32_t rgba) {
        rects.push_back({x0, y0, x1, y1, rgba, 1});
        grow((float)x0, (float)y0, (float)x1, (float)y1);
    }
    /* begin_block
    - Everything drawn since begin_block() gets a translucent backing panel, so the HUD stays
      legible over bright stellar photospheres (invisible over black space).
    */
    void begin_block() { bx0 = by0 = 1e9f; bx1 = by1 = -1e9f; }
    void end_block(float pad) {
        if (bx1 < bx0) return;
        rects.push_back({(int)(bx0 - pad), (int)(by0 - pad), (int)(bx1 + pad), (int)(by1 + pad), 0x66000000u, 0});
    }

private:
    float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
    void grow(float x0, float y0, float x1, float y1) {
        bx0 = std::min(bx0, x0); by0 = std::min(by0, y0);
        bx1 = std::max(bx1, x1); by1 = std::max(by1, y1);
    }
};

constexpr uint32_t WHITE = 0xF0FFFFFFu, DIM = 0xC0D8E0E8u, ACCENT = 0xF0A8D8FFu;

Hud make_hud(const FrameSpec& f, const SceneInfo& si, int W, int H, double pix, bool tele, const TelescopeSettings& ts) {
    Hud h;
    float s = H / 1080.f;
    float m = 34 * s;
    float big = 0.95f * s, mid = 0.62f * s, small = 0.50f * s;
    float lh_mid = FONT_CELL_H * mid * 1.05f, lh_small = FONT_CELL_H * small * 1.1f;
    double Dau = f.D / phys::AU;
    const float pad = 10 * s;
    // Top-left: what we are looking at
    h.begin_block();
    h.text(m, m, mid, "EARTH AS A GRAVITATIONAL LENS", WHITE);
    char b[256];
    std::snprintf(b, sizeof b, "background: %s, %.3g pc", si.target_name.c_str(), si.target_dist / phys::PC);
    h.text(m, m + lh_mid, small, b, DIM);
    if (tele) std::snprintf(b, sizeof b, "telescope view: %.3g m aperture, \xCE\xBB/D = %s", ts.aperture, fmt_angle(550e-9 / ts.aperture).c_str());
    else std::snprintf(b, sizeof b, "ideal view: unlimited aperture");
    std::string tv = b;
    // Replace the Greek lambda (not in the atlas) by "lambda"
    size_t lp = tv.find("\xCE\xBB");
    if (lp != std::string::npos) tv.replace(lp, 2, "lambda");
    h.text(m, m + lh_mid + lh_small, small, tv, DIM);
    h.end_block(pad);

    // Top-right: where the telescope is
    h.begin_block();
    h.text(W - m, m, big, fmt_int(Dau) + " AU", WHITE, 1);
    float y = m + FONT_CELL_H * big * 1.02f;
    std::snprintf(b, sizeof b, "%.3f \xC3\x97 focal distance", Dau / si.focal_atm_au);
    h.text(W - m, y, small, b, ACCENT, 1);
    y += lh_small;
    double off = std::hypot(f.x, f.y);
    std::snprintf(b, sizeof b, "%s km off the focal line", fmt_int(off * 1e-3).c_str());
    h.text(W - m, y, small, b, DIM, 1);
    y += lh_small;
    std::snprintf(b, sizeof b, "light time %.1f days", f.D / phys::C / 86400.0);
    h.text(W - m, y, small, b, DIM, 1);
    h.end_block(pad);

    // Bottom-left: optics
    double thE = einstein_angle(f.D, si.target_dist);
    float yb = H - m - 2 * lh_small;
    h.begin_block();
    std::snprintf(b, sizeof b, "Einstein ring %s   Earth disk %s   phase %.0f\xC2\xB0",
                  fmt_angle(2 * thE).c_str(), fmt_angle(2 * phys::R_EQ / f.D).c_str(), si.phase_deg);
    h.text(m, yb, small, b, WHITE);
    std::snprintf(b, sizeof b, "%s   GR + J2 + solar tide + refraction", format_epoch(si.epoch).c_str());
    h.text(m, yb + lh_small, small, b, DIM);
    h.end_block(pad);

    // Bottom-right: scale bar
    double target = pix * W / 6.0 * phys::RAD2MAS;   // in mas, so the bar gets a round label
    double e = std::pow(10.0, std::floor(std::log10(target)));
    double nice = e;
    for (double k : {1.0, 2.0, 5.0, 10.0})
        if (k * e <= target) nice = k * e;
    nice /= phys::RAD2MAS;
    int bar = (int)std::lround(nice / pix);
    int x1 = (int)(W - m), x0 = x1 - bar, yy = (int)(H - m - lh_small * 0.6f);
    h.begin_block();
    h.rect(x0, yy, x1, yy + std::max(2, (int)(3 * s)), 0xE0FFFFFFu);
    h.rect(x0, yy - (int)(8 * s), x0 + std::max(2, (int)(2 * s)), yy + (int)(3 * s), 0xE0FFFFFFu);
    h.rect(x1 - std::max(2, (int)(2 * s)), yy - (int)(8 * s), x1, yy + (int)(3 * s), 0xE0FFFFFFu);
    h.text(0.5f * (x0 + x1), yy - lh_small * 1.15f, small, fmt_angle(nice), WHITE, 2);
    h.end_block(pad);
    return h;
}

// Exposure smoothing along the sequence (in log space)
std::vector<double> smooth_exposure(const Sequence& seq, const std::vector<double>& lp, double key, double sigma) {
    size_t n = lp.size();
    std::vector<double> lg(n), out(n);
    for (size_t i = 0; i < n; ++i) lg[i] = std::log(std::max(lp[i], 1e-30));
    for (size_t i = 0; i < n; ++i) {
        const FrameSpec& f = seq.frames[i];
        double ws = 0, acc = 0;
        int r = (int)std::ceil(3 * sigma);
        for (int d = -r; d <= r; ++d) {
            int id = f.id + d;
            if (id < 0 || id >= seq.nd) continue;
            size_t j = (size_t)f.il * seq.nd + id;
            if (lp[j] <= 0) continue;
            double w = sigma > 0 ? std::exp(-0.5 * d * d / (sigma * sigma)) : (d == 0);
            ws += w;
            acc += w * lg[j];
        }
        out[i] = ws > 0 ? key / std::exp(acc / ws) : 1.0;
    }
    return out;
}

void usage() {
    std::fprintf(stderr,
                 "usage: lensim scene.ini [--out DIR] [--set section.key=value ...] [--frames A:B]\n"
                 "              [--report] [--list]\n");
}
}  // namespace

int main(int argc, char** argv) {
    mpi_init(&argc, &argv);
    double t_start = wall();
    if (argc < 2) {
        if (g_rank == 0) usage();
        mpi_finalize();
        return 1;
    }
    Config cfg;
    std::string err;
    if (!cfg.load(argv[1], &err)) {
        if (g_rank == 0) std::fprintf(stderr, "lensim: %s\n", err.c_str());
        mpi_finalize();
        return 1;
    }
    bool report = false, list = false;
    int fa = 0, fb = -1;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--set" && i + 1 < argc) {
            std::string kv = argv[++i];
            size_t eq = kv.find('=');
            if (eq != std::string::npos) cfg.set(trim(kv.substr(0, eq)), trim(kv.substr(eq + 1)));
        } else if (a == "--out" && i + 1 < argc) {
            cfg.set("output.dir", argv[++i]);
        } else if (a == "--frames" && i + 1 < argc) {
            std::sscanf(argv[++i], "%d:%d", &fa, &fb);
        } else if (a == "--report") {
            report = true;
        } else if (a == "--list") {
            list = true;
        } else {
            if (g_rank == 0) usage();
            mpi_finalize();
            return 1;
        }
    }

    SceneGPU S;
    SceneInfo si;
    if (!build_scene(cfg, &S, &si, &err)) {
        if (g_rank == 0) std::fprintf(stderr, "lensim: %s\n", err.c_str());
        mpi_finalize();
        return 1;
    }
    RenderSettings rs;
    rs.W = cfg.integer("camera.width", 1920);
    rs.H = cfg.integer("camera.height", 1080);
    rs.spp = cfg.integer("camera.spp", 16);
    rs.spp_max = cfg.integer("camera.spp_max", 256);
    rs.target_err = (float)cfg.num("camera.target_error", 0.02);
    rs.tele.aperture = cfg.num("telescope.aperture_m", 0);
    rs.tele.enabled = rs.tele.aperture > 0;
    rs.tele.obstruction = cfg.num("telescope.obstruction", 0.12);
    rs.tele.spiders = cfg.integer("telescope.spiders", 4);
    rs.tele.spider_width = cfg.num("telescope.spider_width", 0.012);
    rs.tele.W = cfg.integer("telescope.width", rs.W);
    rs.tele.H = cfg.integer("telescope.height", rs.H);
    rs.tele.noise = cfg.flag("telescope.noise", false);
    rs.tele.exposure_s = cfg.num("telescope.exposure_s", 1.0);
    rs.tele.qe = cfg.num("telescope.qe", 0.8);
    rs.tele.read_noise = cfg.num("telescope.read_noise", 3.0);

    Sequence seq;
    if (!build_sequence(cfg, si, rs, &seq, &err)) {
        if (g_rank == 0) std::fprintf(stderr, "lensim: %s\n", err.c_str());
        mpi_finalize();
        return 1;
    }
    int nframes = (int)seq.frames.size();
    if (fb < 0 || fb > nframes) fb = nframes;
    fa = std::max(0, std::min(fa, fb));

    if (g_rank == 0) {
        std::printf("lensim | %d rank(s) | %d frame(s) [%d:%d) at %dx%d, spp %d..%d\n", g_size, nframes, fa, fb, rs.W,
                    rs.H, rs.spp, rs.spp_max);
        std::printf("  epoch %s, target %s (%s) at %.4g pc, radius %.4g km\n", format_epoch(si.epoch).c_str(),
                    si.target_name.c_str(), si.target_type.c_str(), si.target_dist / phys::PC, si.target_radius * 1e-3);
        std::printf("  telescope sees Earth from lat %+.2f lon %+.2f, Sun-Earth-telescope phase %.1f deg\n", si.sub_lat,
                    si.sub_lon, si.phase_deg);
        std::printf("  focal distance: %.0f AU (ring at equator), %.0f AU (ring at 110 km)\n", si.focal_au, si.focal_atm_au);
        std::printf("  J2 quadrupole %s, solar tide %s (Sun %.3f AU off the line of sight)\n", S.use_j2 ? "on" : "off",
                    S.use_tide ? "on" : "off", si.sun_sep_au);
        if (list)
            for (const FrameSpec& f : seq.frames)
                std::printf("  frame %5d  D = %10.1f AU  offset (%8.1f, %8.1f) km  fov %.4g mas\n", f.index, f.D / phys::AU,
                            f.x * 1e-3, f.y * 1e-3, 2 * f.fov_half * phys::RAD2MAS);
        std::fflush(stdout);
    }
    if (list) { mpi_finalize(); return 0; }

    // GPU set-up
    int local_rank = g_rank;
#ifndef LENSIM_NO_MPI
    {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &local_rank);
        MPI_Comm_free(&node);
    }
#endif
    int ndev = lensim_device_count();
    if (ndev <= 0) {
        std::fprintf(stderr, "[rank %d] no CUDA device\n", g_rank);
        mpi_finalize();
        return 1;
    }
    EarthTextureData earth;
    bool tex_ok = load_earth_textures(si.earth_dir, &earth, &err);
    if (!tex_ok && g_rank == 0)
        std::fprintf(stderr, "warning: %s -- rendering an untextured Earth\n", err.c_str());
    Renderer R;
    double t_init = wall();
    if (!R.init(local_rank % ndev, S, si.refractivity, tex_ok ? &earth : nullptr, si.images, rs, si.color, &err)) {
        std::fprintf(stderr, "[rank %d] init failed: %s\n", g_rank, err.c_str());
        mpi_finalize();
        return 1;
    }
    earth.albedo.clear(); earth.albedo.shrink_to_fit();
    earth.aux.clear(); earth.aux.shrink_to_fit();
    if (g_rank == 0) {
        std::printf("  GPU initialisation (tables + textures): %.2f s\n", wall() - t_init);
        if (report) R.report(stdout);
        std::fflush(stdout);
    }
    if (report) { mpi_finalize(); return 0; }

    std::string out = cfg.str("output.dir", "out");
    bool write_exr_files = cfg.flag("output.exr", false);
    if (g_rank == 0) {
        mkdirs(out + "/ideal");
        if (rs.tele.enabled) mkdirs(out + "/telescope");
        if (write_exr_files) mkdirs(out + "/hdr");
    }
#ifndef LENSIM_NO_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Phase 1: render
    struct Done { int index; HdrFrame ideal, tele; };
    std::vector<Done> done;
    double percentile = cfg.num("camera.exposure_percentile", 0.995);
    long long my_samples = 0;
    double t_render = wall();
    {
    FrameScheduler sched(fa, fb);
    for (;;) {
        long idx = sched.next();
        if (idx >= fb) break;
        const FrameSpec& f = seq.frames[idx];
        if (f.jd != 0) {
            SceneGPU g = S;
            SceneInfo tmp = si;
            scene_geometry(cfg, f.jd, &g, &tmp);
            R.update_geometry(g);
        }
        FrameGPU F{};
        F.D = f.D;
        F.rho_x = f.x;
        F.rho_y = f.y;
        F.pix = 2.0 * f.fov_half / rs.W;
        F.seed = 0x9E3779B9u * (uint32_t)(idx + 1);
        Done d;
        d.index = (int)idx;
        RenderStats st;
        double t0 = wall();
        R.render(F, f.tele_pix, percentile, &d.ideal, rs.tele.enabled ? &d.tele : nullptr, &st);
        my_samples += st.samples;
        std::printf("[rank %3d] frame %5ld  D %9.1f AU  %6.1f Msamples  %7.1f ms (pass1 %.0f, adapt %.0f, post %.0f)\n",
                    g_rank, idx, f.D / phys::AU, st.samples * 1e-6, (wall() - t0) * 1e3, st.ms_pass1, st.ms_pass2, st.ms_post);
        std::fflush(stdout);
        done.push_back(std::move(d));
    }
    }   // scheduler thread joins here: every rank has received its final "no more frames"
#ifndef LENSIM_NO_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double t_render_end = wall();

    // Exposure
    std::vector<double> lp(2 * (size_t)nframes, 0.0);
    {
        std::vector<double> mine(2 * (size_t)nframes, 0.0);
        for (const Done& d : done) {
            mine[d.index] = d.ideal.lum_percentile;
            mine[nframes + d.index] = d.tele.lum_percentile;
        }
#ifndef LENSIM_NO_MPI
        MPI_Allreduce(mine.data(), lp.data(), (int)lp.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
        lp = mine;
#endif
    }
    std::string emode = cfg.str("camera.exposure", "auto");
    double key = cfg.num("camera.exposure_key", 0.85);
    // Eye-adaptation smoothing (frames)
    double sigma = cfg.num("camera.exposure_smooth", 6);
    std::vector<double> ex_ideal, ex_tele;
    {
        std::vector<double> a(lp.begin(), lp.begin() + nframes), b(lp.begin() + nframes, lp.end());
        if (emode == "auto") {
            ex_ideal = smooth_exposure(seq, a, key, sigma);
            ex_tele = smooth_exposure(seq, b, key, sigma);
        } else {
            double Lref = std::strtod(emode.c_str(), nullptr);   // cd/m^2 mapped to the key
            ex_ideal.assign(nframes, key / Lref);
            ex_tele.assign(nframes, key / Lref);
        }
    }

    // Phase 2: tone map, HUD, write
    bool hud = cfg.flag("camera.hud", true);
    std::vector<std::future<void>> writes;
    auto flush_writes = [&](size_t keep) {
        while (writes.size() > keep) {
            writes.front().get();
            writes.erase(writes.begin());
        }
    };
    for (Done& d : done) {
        const FrameSpec& f = seq.frames[d.index];
        SceneInfo fsi = si;
        if (f.jd != 0) {
            SceneGPU g = S;
            scene_geometry(cfg, f.jd, &g, &fsi);
            fsi.epoch.jd_utc = f.jd;
        }
        char name[64];
        std::snprintf(name, sizeof name, "frame_%05d", d.index);
        for (int view = 0; view < (rs.tele.enabled ? 2 : 1); ++view) {
            HdrFrame& hf = view == 0 ? d.ideal : d.tele;
            double pix = view == 0 ? 2.0 * f.fov_half / rs.W : f.tele_pix;
            Hud h;
            if (hud) h = make_hud(f, fsi, hf.W, hf.H, pix, view == 1, rs.tele);
            auto rgb = std::make_shared<std::vector<uint8_t>>();
            R.tonemap(hf, (float)(view == 0 ? ex_ideal[d.index] : ex_tele[d.index]), h.glyphs, h.rects, rgb.get());
            std::string path = out + (view == 0 ? "/ideal/" : "/telescope/") + name + ".png";
            int W = hf.W, H = hf.H;
            writes.push_back(std::async(std::launch::async, [path, W, H, rgb]() {
                if (!write_png(path, W, H, 3, rgb->data())) std::fprintf(stderr, "cannot write %s\n", path.c_str());
            }));
            if (write_exr_files) {
                std::string ep = out + "/hdr/" + name + (view == 0 ? "_ideal.exr" : "_telescope.exr");
                write_exr(ep, hf.W, hf.H, hf.px.data());
            }
            flush_writes(16);
        }
    }
    flush_writes(0);

    // Metadata & summary
    long long all_samples = my_samples;
#ifndef LENSIM_NO_MPI
    MPI_Reduce(&my_samples, &all_samples, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    if (g_rank == 0) {
        FILE* m = std::fopen((out + "/meta.json").c_str(), "w");
        if (m) {
            std::fprintf(m, "{\n  \"mode\": \"%s\", \"nd\": %d, \"nl\": %d, \"fps\": %g,\n", seq.mode.c_str(), seq.nd, seq.nl, seq.fps);
            std::fprintf(m, "  \"width\": %d, \"height\": %d, \"telescope\": %s,\n", rs.W, rs.H, rs.tele.enabled ? "true" : "false");
            std::fprintf(m, "  \"target\": \"%s\", \"target_pc\": %.6g, \"epoch\": \"%s\", \"phase_deg\": %.3f,\n",
                         si.target_name.c_str(), si.target_dist / phys::PC, format_epoch(si.epoch).c_str(), si.phase_deg);
            std::fprintf(m, "  \"focal_au\": %.3f, \"focal_atm_au\": %.3f,\n  \"frames\": [\n", si.focal_au, si.focal_atm_au);
            for (int i = 0; i < nframes; ++i) {
                const FrameSpec& f = seq.frames[i];
                std::fprintf(m,
                             "    {\"i\": %d, \"id\": %d, \"il\": %d, \"D_au\": %.4f, \"x_km\": %.4f, \"y_km\": %.4f, "
                             "\"fov_mas\": %.6g, \"thetaE_mas\": %.6g, \"earth_mas\": %.6g, \"light_days\": %.4f, "
                             "\"exposure\": %.6g}%s\n",
                             i, f.id, f.il, f.D / phys::AU, f.x * 1e-3, f.y * 1e-3, 2 * f.fov_half * phys::RAD2MAS,
                             einstein_angle(f.D, si.target_dist) * phys::RAD2MAS, 2 * phys::R_EQ / f.D * phys::RAD2MAS,
                             f.D / phys::C / 86400, ex_ideal[i], i + 1 < nframes ? "," : "");
            }
            std::fprintf(m, "  ]\n}\n");
            std::fclose(m);
        }
        double tr = t_render_end - t_render;
        std::printf("done: %d frames, %.3g samples in %.1f s render (%.3g Msamples/s aggregate), %.1f s total\n", fb - fa,
                    (double)all_samples, tr, all_samples / std::max(tr, 1e-9) * 1e-6, wall() - t_start);
    }
    mpi_finalize();
    return 0;
}
