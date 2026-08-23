/* PREP TEXTURES
- lensim-prep: packs NASA Earth imagery into the two RGBA textures the renderer streams to the GPU.
  albedo.lgtx : RGB = Blue Marble Next Generation surface colour (sRGB), A = water mask
  aux.lgtx    : R = Blue Marble cloud cover, G = Black Marble night lights, B = GEBCO elevation
*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "image_io.hpp"

namespace {
struct Gray { int w = 0, h = 0; std::vector<uint8_t> v; };

template <typename F>
void parallel_rows(int h, F f) {
    int nt = std::max(1u, std::thread::hardware_concurrency());
    nt = std::min(nt, 64);
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t]() { for (int y = t; y < h; y += nt) f(y); });
    for (auto& x : th) x.join();
}

// Area-average (downsampling) or bilinear (upsampling) resample of one channel
std::vector<uint8_t> resample(const uint8_t* src, int sw, int sh, int stride, int W, int H) {
    std::vector<uint8_t> out((size_t)W * H);
    double fx = (double)sw / W, fy = (double)sh / H;
    parallel_rows(H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            double v;
            if (fx >= 1.0 && fy >= 1.0) {
                int x0 = (int)(x * fx), x1 = std::max(x0 + 1, (int)((x + 1) * fx));
                int y0 = (int)(y * fy), y1 = std::max(y0 + 1, (int)((y + 1) * fy));
                x1 = std::min(x1, sw); y1 = std::min(y1, sh);
                double s = 0;
                for (int yy = y0; yy < y1; ++yy)
                    for (int xx = x0; xx < x1; ++xx) s += src[((size_t)yy * sw + xx) * stride];
                v = s / ((x1 - x0) * (y1 - y0));
            } else {
                double sx = (x + 0.5) * fx - 0.5, sy = (y + 0.5) * fy - 0.5;
                int ix = (int)std::floor(sx), iy = (int)std::floor(sy);
                double ax = sx - ix, ay = sy - iy;
                auto at = [&](int xx, int yy) {
                    xx = (xx % sw + sw) % sw;
                    yy = std::min(std::max(yy, 0), sh - 1);
                    return (double)src[((size_t)yy * sw + xx) * stride];
                };
                v = (1 - ay) * ((1 - ax) * at(ix, iy) + ax * at(ix + 1, iy)) + ay * ((1 - ax) * at(ix, iy + 1) + ax * at(ix + 1, iy + 1));
            }
            out[(size_t)y * W + x] = (uint8_t)std::min(255.0, std::max(0.0, v + 0.5));
        }
    });
    return out;
}

bool load_channel(const std::string& path, int W, int H, std::vector<uint8_t>* out) {
    PngImage img;
    std::string err;
    if (!read_png(path, &img, &err)) {
        std::fprintf(stderr, "lensim-prep: %s\n", err.c_str());
        return false;
    }
    std::printf("  %s: %dx%d, %d channel(s)\n", path.c_str(), img.w, img.h, img.channels);
    if (img.channels >= 3) {   // luminance
        std::vector<uint8_t> g((size_t)img.w * img.h);
        for (size_t i = 0; i < g.size(); ++i) {
            const uint8_t* p = &img.data[i * img.channels];
            g[i] = (uint8_t)((54 * p[0] + 183 * p[1] + 19 * p[2]) >> 8);
        }
        *out = resample(g.data(), img.w, img.h, 1, W, H);
    } else {
        *out = resample(img.data.data(), img.w, img.h, img.channels, W, H);
    }
    return true;
}

double sample_ll(const std::vector<uint8_t>& c, int W, int H, double lon, double lat) {
    int x = (int)((lon + 180.0) / 360.0 * W), y = (int)((90.0 - lat) / 180.0 * H);
    x = std::min(std::max(x, 0), W - 1);
    y = std::min(std::max(y, 0), H - 1);
    return c[(size_t)y * W + x];
}

void usage() {
    std::fprintf(stderr,
                 "usage: lensim-prep --albedo world.png --clouds clouds.png --lights lights.png\n"
                 "                   --elev elev.png --bathy bathy.png --out DIR [--width W]\n");
}
}  // namespace

int main(int argc, char** argv) {
    std::string albedo, clouds, lights, elev, bathy, out;
    int W = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--albedo") albedo = v;
        else if (a == "--clouds") clouds = v;
        else if (a == "--lights") lights = v;
        else if (a == "--elev") elev = v;
        else if (a == "--bathy") bathy = v;
        else if (a == "--out") out = v;
        else if (a == "--width") W = std::atoi(v.c_str());
        else { usage(); return 1; }
    }
    if (albedo.empty() || out.empty()) { usage(); return 1; }

    PngImage alb;
    std::string err;
    if (!read_png(albedo, &alb, &err)) { std::fprintf(stderr, "lensim-prep: %s\n", err.c_str()); return 1; }
    if (alb.channels < 3) { std::fprintf(stderr, "lensim-prep: albedo must be RGB\n"); return 1; }
    if (W <= 0) W = alb.w;
    int H = W / 2;
    std::printf("output %dx%d\n  %s: %dx%d\n", W, H, albedo.c_str(), alb.w, alb.h);
    std::vector<uint8_t> r = resample(alb.data.data() + 0, alb.w, alb.h, alb.channels, W, H);
    std::vector<uint8_t> g = resample(alb.data.data() + 1, alb.w, alb.h, alb.channels, W, H);
    std::vector<uint8_t> b = resample(alb.data.data() + 2, alb.w, alb.h, alb.channels, W, H);
    alb.data.clear();
    alb.data.shrink_to_fit();

    std::vector<uint8_t> cl((size_t)W * H, 0), li((size_t)W * H, 0), el((size_t)W * H, 0), ba;
    if (!clouds.empty() && !load_channel(clouds, W, H, &cl)) return 1;
    if (!lights.empty() && !load_channel(lights, W, H, &li)) return 1;
    if (!elev.empty() && !load_channel(elev, W, H, &el)) return 1;

    // Water mask from GEBCO bathymetry; polarity and threshold are detected from known points
    std::vector<uint8_t> water((size_t)W * H, 0);
    if (!bathy.empty()) {
        if (!load_channel(bathy, W, H, &ba)) return 1;
        const double ocean_pts[][2] = {{-150, 0}, {-30, -10}, {80, -25}, {-140, 35}, {160, -40}, {-20, 40}};
        const double land_pts[][2] = {{15, 23}, {100, 62}, {-60, -10}, {135, -25}, {-100, 40}, {30, 50}};
        double mo = 0, ml = 0;
        for (auto& p : ocean_pts) mo += sample_ll(ba, W, H, p[0], p[1]);
        for (auto& p : land_pts) ml += sample_ll(ba, W, H, p[0], p[1]);
        mo /= 6;
        ml /= 6;
        bool ocean_bright = mo > ml;
        std::printf("  bathymetry: ocean %.1f, land %.1f -> water where value %s land level\n", mo, ml,
                    ocean_bright ? "above" : "below");
        parallel_rows(H, [&](int y) {
            for (int x = 0; x < W; ++x) {
                size_t i = (size_t)y * W + x;
                bool wet = ocean_bright ? ba[i] > ml + 0.5 : ba[i] < ml - 0.5;
                // Anything with land elevation is land
                if (el[i] > 0) wet = false;
                water[i] = wet ? 255 : 0;
            }
        });
    }

    std::vector<uint8_t> A((size_t)W * H * 4), X((size_t)W * H * 4);
    parallel_rows(H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            size_t i = (size_t)y * W + x;
            A[4 * i] = r[i]; A[4 * i + 1] = g[i]; A[4 * i + 2] = b[i]; A[4 * i + 3] = water[i];
            X[4 * i] = cl[i]; X[4 * i + 1] = li[i]; X[4 * i + 2] = el[i]; X[4 * i + 3] = 0;
        }
    });
    if (!write_texture(out + "/albedo.lgtx", W, H, 4, A.data()) || !write_texture(out + "/aux.lgtx", W, H, 4, X.data())) {
        std::fprintf(stderr, "lensim-prep: cannot write to %s\n", out.c_str());
        return 1;
    }
    // Small previews for a sanity check
    int pw = 1024, ph = 512;
    std::vector<uint8_t> prev((size_t)pw * ph * 3);
    auto pr = resample(A.data(), W, H, 4, pw, ph), pg = resample(A.data() + 1, W, H, 4, pw, ph);
    auto pb = resample(A.data() + 2, W, H, 4, pw, ph), pc = resample(X.data(), W, H, 4, pw, ph);
    auto pwm = resample(A.data() + 3, W, H, 4, pw, ph);
    for (size_t i = 0; i < (size_t)pw * ph; ++i) {
        prev[3 * i] = std::max(pr[i], pc[i]);
        prev[3 * i + 1] = std::max(pg[i], pc[i]);
        prev[3 * i + 2] = std::max(pb[i], pc[i]);
    }
    write_png(out + "/preview_albedo_clouds.png", pw, ph, 3, prev.data());
    write_png(out + "/preview_water.png", pw, ph, 1, pwm.data());
    std::printf("wrote %s/albedo.lgtx and %s/aux.lgtx\n", out.c_str(), out.c_str());
    return 0;
}
