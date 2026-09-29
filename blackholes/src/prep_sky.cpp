/* PREP SKY
- bh-prep-sky: NASA SVS Deep Star Map (OpenEXR, ZIP-compressed half RGB, galactic equirectangular)
  -> renderer sky texture.
- Every texel is re-expressed as a scaled blackbody s * B(T): the colour temperature T comes from
  the blue/red ratio on the blackbody locus, s from the luminance.
- The renderer can then Doppler/gravitationally shift the whole sky exactly (B(T) -> B(gT)).
- Bh-prep-sky starmap_2020_32k_gal.exr data/sky/sky.bhsk
*/

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "colour.hpp"

namespace {
float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else {
            e = 1;
            while (!(m & 1024)) { m <<= 1; --e; }
            m &= 1023;
            f = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) f = (s << 31) | 0x7f800000 | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
uint16_t float_to_half(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    uint32_t s = (f >> 16) & 0x8000;
    int e = (int)((f >> 23) & 255) - 127 + 15;
    uint32_t m = f & 0x7fffff;
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m |= 0x800000;
        uint32_t t = 14 - e;
        uint32_t hm = m >> t;
        if ((m >> (t - 1)) & 1) ++hm;
        return (uint16_t)(s | hm);
    }
    if (e >= 31) return (uint16_t)(s | 0x7bff);   // clamp to max finite
    uint32_t hm = m >> 13;
    if (m & 0x1000) ++hm;
    if (hm & 0x400) { hm = 0; ++e; if (e >= 31) return (uint16_t)(s | 0x7bff); }
    return (uint16_t)(s | (e << 10) | hm);
}

struct Exr {
    int w = 0, h = 0;
    int compression = -1;
    std::vector<std::string> chans;
    std::vector<int> types;
    size_t header_end = 0;
};

bool parse_header(const std::vector<uint8_t>& d, Exr* e) {
    if (d.size() < 8 || d[0] != 0x76 || d[1] != 0x2f || d[2] != 0x31 || d[3] != 0x01) return false;
    size_t p = 8;
    while (p < d.size() && d[p] != 0) {
        std::string name((const char*)&d[p]);
        p += name.size() + 1;
        std::string type((const char*)&d[p]);
        p += type.size() + 1;
        int32_t size;
        std::memcpy(&size, &d[p], 4);
        p += 4;
        const uint8_t* v = &d[p];
        if (name == "compression") e->compression = v[0];
        if (name == "dataWindow") {
            int32_t b[4];
            std::memcpy(b, v, 16);
            e->w = b[2] - b[0] + 1;
            e->h = b[3] - b[1] + 1;
        }
        if (name == "channels") {
            size_t q = 0;
            while (q < (size_t)size && v[q]) {
                std::string cn((const char*)&v[q]);
                q += cn.size() + 1;
                int32_t t;
                std::memcpy(&t, &v[q], 4);
                q += 16;
                e->chans.push_back(cn);
                e->types.push_back(t);
            }
        }
        p += size;
    }
    e->header_end = p + 1;
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: bh-prep-sky <starmap.exr> <out.bhsk>\n");
        return 1;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror(argv[1]); return 1; }
    std::fseek(f, 0, SEEK_END);
    size_t n = (size_t)std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> d(n);
    if (std::fread(d.data(), 1, n, f) != n) { std::fprintf(stderr, "short read\n"); return 1; }
    std::fclose(f);
    Exr e;
    if (!parse_header(d, &e) || e.compression != 3 || e.chans.size() != 3) {
        std::fprintf(stderr, "expected a ZIP-compressed 3-channel EXR (got compression %d, %zu channels)\n",
                     e.compression, e.chans.size());
        return 1;
    }
    for (int t : e.types)
        if (t != 1) { std::fprintf(stderr, "expected HALF channels\n"); return 1; }
    const int W = e.w, H = e.h, LINES = 16, nblocks = (H + LINES - 1) / LINES;
    std::printf("%s: %dx%d, channels %s%s%s, ZIP\n", argv[1], W, H, e.chans[0].c_str(), e.chans[1].c_str(), e.chans[2].c_str());
    int ci[3];   // index of R, G, B in the file's channel order
    for (int c = 0; c < 3; ++c) {
        const char* want = c == 0 ? "R" : c == 1 ? "G" : "B";
        ci[c] = (int)(std::find(e.chans.begin(), e.chans.end(), want) - e.chans.begin());
    }
    std::vector<uint64_t> offs(nblocks);
    std::memcpy(offs.data(), &d[e.header_end], 8 * (size_t)nblocks);

    // Blackbody locus in linear sRGB: ln(b/r) is monotonic in T
    const int NT = 4096;
    std::vector<double> lt(NT), q(NT), yb(NT);
    for (int i = 0; i < NT; ++i) {
        double T = std::exp(std::log(1200.0) + i * (std::log(4.5e4) - std::log(1200.0)) / (NT - 1));
        double xyz[3], rgb[3];
        colour::blackbody_xyz(T, xyz);
        colour::xyz_to_srgb(xyz, rgb);
        lt[i] = T;
        q[i] = std::log(std::max(rgb[2], 1e-30) / std::max(rgb[0], 1e-30));
        yb[i] = xyz[1];
    }

    std::vector<float> s_all((size_t)W * H), T_all((size_t)W * H);
    std::vector<std::thread> pool;
    int nt = std::min(64, (int)std::max(1u, std::thread::hardware_concurrency()));
    for (int t = 0; t < nt; ++t) {
        pool.emplace_back([&, t]() {
            std::vector<uint8_t> raw((size_t)W * LINES * 6), tmp(raw.size());
            for (int b = t; b < nblocks; b += nt) {
                const uint8_t* c = &d[offs[b]];
                int32_t y0, size;
                std::memcpy(&y0, c, 4);
                std::memcpy(&size, c + 4, 4);
                int lines = std::min(LINES, H - y0);
                size_t want = (size_t)W * lines * 6;
                if ((size_t)size < want) {
                    uLongf dl = (uLongf)want;
                    if (uncompress(tmp.data(), &dl, c + 8, (uLong)size) != Z_OK || dl != want) {
                        std::fprintf(stderr, "zlib error in block %d\n", b);
                        continue;
                    }
                    for (size_t i = 1; i < want; ++i) tmp[i] = (uint8_t)(tmp[i - 1] + tmp[i] - 128);
                    size_t half = (want + 1) / 2;
                    for (size_t i = 0; i < want; ++i) raw[i] = (i & 1) ? tmp[half + i / 2] : tmp[i / 2];
                } else {
                    std::memcpy(raw.data(), c + 8, want);
                }
                for (int ly = 0; ly < lines; ++ly) {
                    const uint16_t* line = (const uint16_t*)&raw[(size_t)ly * W * 6];
                    size_t row = (size_t)(y0 + ly) * W;
                    for (int x = 0; x < W; ++x) {
                        double rgb[3];
                        for (int k = 0; k < 3; ++k) rgb[k] = std::max(0.0f, half_to_float(line[(size_t)ci[k] * W + x]));
                        double xyz[3];
                        colour::srgb_to_xyz(rgb, xyz);
                        float s = 0.f, T = 5800.f;
                        if (xyz[1] > 0) {
                            double qq = std::log(std::max(rgb[2], 1e-12 * xyz[1] + 1e-30) / std::max(rgb[0], 1e-12 * xyz[1] + 1e-30));
                            int i = (int)(std::lower_bound(q.begin(), q.end(), qq) - q.begin());
                            i = std::min(std::max(i, 0), NT - 1);
                            T = (float)lt[i];
                            s = (float)(xyz[1] / yb[i]);
                        }
                        s_all[row + x] = s;
                        T_all[row + x] = T;
                    }
                }
            }
        });
    }
    for (auto& th : pool) th.join();
    float smax = 0;
    for (float s : s_all) smax = std::max(smax, s);
    double S0 = 1.5e4 / std::max(smax, 1e-30f);   // keep s and s*T/1e4 inside half-float range
    std::vector<uint16_t> out((size_t)W * H * 2);
    for (size_t i = 0; i < s_all.size(); ++i) {
        float s = (float)(s_all[i] * S0);
        out[2 * i] = float_to_half(s);
        out[2 * i + 1] = float_to_half(s * T_all[i] * 1e-4f);
    }
    FILE* o = std::fopen(argv[2], "wb");
    if (!o) { std::perror(argv[2]); return 1; }
    uint32_t hdr[3] = {0x4b534842u /* "BHSK" */, (uint32_t)W, (uint32_t)H};
    float scale = (float)(1.0 / S0);
    std::fwrite(hdr, 4, 3, o);
    std::fwrite(&scale, 4, 1, o);
    std::fwrite(out.data(), 2, out.size(), o);
    std::fclose(o);
    std::printf("wrote %s (%dx%d half2, scale %.4g)\n", argv[2], W, H, scale);
    return 0;
}
