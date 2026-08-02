#include "image_io.hpp"

#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace {
void put32be(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}
uint32_t get32be(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

void png_chunk(FILE* f, const char* type, const uint8_t* data, uint32_t len) {
    uint8_t hdr[8] = {(uint8_t)(len >> 24), (uint8_t)(len >> 16), (uint8_t)(len >> 8), (uint8_t)len,
                      (uint8_t)type[0], (uint8_t)type[1], (uint8_t)type[2], (uint8_t)type[3]};
    std::fwrite(hdr, 1, 8, f);
    if (len) std::fwrite(data, 1, len, f);
    uLong crc = crc32(0L, (const Bytef*)type, 4);
    if (len) crc = crc32(crc, data, len);
    uint8_t c[4] = {(uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc};
    std::fwrite(c, 1, 4, f);
}

uint8_t paeth(int a, int b, int c) {
    int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return (uint8_t)a;
    return (uint8_t)(pb <= pc ? b : c);
}
}  // namespace

bool write_png(const std::string& path, int w, int h, int channels, const uint8_t* data, int level) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    std::fwrite(sig, 1, 8, f);
    std::vector<uint8_t> ihdr;
    put32be(ihdr, w);
    put32be(ihdr, h);
    ihdr.push_back(8);
    ihdr.push_back(channels == 1 ? 0 : (channels == 3 ? 2 : 6));
    ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    png_chunk(f, "IHDR", ihdr.data(), (uint32_t)ihdr.size());

    // Sub filter on every row: cheap and compresses smooth renders well
    size_t stride = (size_t)w * channels;
    std::vector<uint8_t> raw((stride + 1) * h);
    for (int y = 0; y < h; ++y) {
        uint8_t* o = &raw[y * (stride + 1)];
        const uint8_t* r = data + y * stride;
        o[0] = 1;
        for (size_t i = 0; i < stride; ++i) o[1 + i] = r[i] - (i >= (size_t)channels ? r[i - channels] : 0);
    }
    uLongf clen = compressBound(raw.size());
    std::vector<uint8_t> comp(clen);
    if (compress2(comp.data(), &clen, raw.data(), raw.size(), level) != Z_OK) {
        std::fclose(f);
        return false;
    }
    png_chunk(f, "IDAT", comp.data(), (uint32_t)clen);
    png_chunk(f, "IEND", nullptr, 0);
    return std::fclose(f) == 0;
}

bool read_png(const std::string& path, PngImage* img, std::string* err, bool keep16) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { if (err) *err = "cannot open " + path; return false; }
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (file.size() < 8 || std::memcmp(file.data(), sig, 8) != 0) { if (err) *err = path + ": not a PNG"; return false; }
    size_t pos = 8;
    int w = 0, h = 0, depth = 0, ctype = 0, interlace = 0;
    std::vector<uint8_t> idat, plte;
    while (pos + 8 <= file.size()) {
        uint32_t len = get32be(&file[pos]);
        const char* type = (const char*)&file[pos + 4];
        const uint8_t* d = &file[pos + 8];
        if (pos + 12 + len > file.size()) break;
        if (!std::memcmp(type, "IHDR", 4)) {
            w = get32be(d); h = get32be(d + 4); depth = d[8]; ctype = d[9]; interlace = d[12];
        } else if (!std::memcmp(type, "PLTE", 4)) {
            plte.assign(d, d + len);
        } else if (!std::memcmp(type, "IDAT", 4)) {
            idat.insert(idat.end(), d, d + len);
        } else if (!std::memcmp(type, "IEND", 4)) {
            break;
        }
        pos += 12 + len;
    }
    if (interlace) { if (err) *err = path + ": interlaced PNG not supported"; return false; }
    if (depth != 8 && depth != 16) { if (err) *err = path + ": unsupported bit depth"; return false; }
    int ch = ctype == 0 ? 1 : ctype == 2 ? 3 : ctype == 3 ? 1 : ctype == 4 ? 2 : ctype == 6 ? 4 : 0;
    if (!ch) { if (err) *err = path + ": bad colour type"; return false; }
    int bpp = ch * depth / 8;
    size_t stride = (size_t)w * bpp;
    std::vector<uint8_t> raw((stride + 1) * h);
    z_stream zs = {};
    inflateInit(&zs);
    zs.next_in = idat.data();
    zs.avail_in = (uInt)idat.size();
    size_t out = 0;
    int ret = Z_OK;
    while (ret != Z_STREAM_END && out < raw.size()) {
        size_t chunk = std::min<size_t>(raw.size() - out, 1u << 30);
        zs.next_out = raw.data() + out;
        zs.avail_out = (uInt)chunk;
        ret = inflate(&zs, Z_NO_FLUSH);
        out += chunk - zs.avail_out;
        if (ret != Z_OK && ret != Z_STREAM_END) break;
    }
    inflateEnd(&zs);
    if (out != raw.size()) { if (err) *err = path + ": truncated image data"; return false; }

    std::vector<uint8_t> pix(stride * h);
    for (int y = 0; y < h; ++y) {
        uint8_t ft = raw[y * (stride + 1)];
        const uint8_t* s = &raw[y * (stride + 1) + 1];
        uint8_t* o = &pix[y * stride];
        const uint8_t* up = y ? &pix[(y - 1) * stride] : nullptr;
        for (size_t i = 0; i < stride; ++i) {
            int a = i >= (size_t)bpp ? o[i - bpp] : 0, b = up ? up[i] : 0, c = (up && i >= (size_t)bpp) ? up[i - bpp] : 0;
            uint8_t v = s[i];
            switch (ft) {
                case 1: v += a; break;
                case 2: v += b; break;
                case 3: v += (uint8_t)((a + b) >> 1); break;
                case 4: v += paeth(a, b, c); break;
                default: break;
            }
            o[i] = v;
        }
    }
    img->w = w;
    img->h = h;
    if (ctype == 3) {   // palette -> RGB
        img->channels = 3;
        img->depth = 8;
        img->data.resize((size_t)w * h * 3);
        for (size_t i = 0; i < (size_t)w * h; ++i)
            for (int c = 0; c < 3; ++c) {
                size_t j = 3 * (size_t)pix[i] + c;
                img->data[3 * i + c] = j < plte.size() ? plte[j] : 0;
            }
        return true;
    }
    img->channels = ch;
    if (depth == 16 && !keep16) {
        img->depth = 8;
        img->data.resize((size_t)w * h * ch);
        for (size_t i = 0; i < img->data.size(); ++i) img->data[i] = pix[2 * i];
    } else {
        img->depth = depth;
        img->data.swap(pix);
    }
    return true;
}

bool write_exr(const std::string& path, int w, int h, const float* rgb) {
    std::vector<uint8_t> hd;
    auto put = [&](const void* p, size_t n) { hd.insert(hd.end(), (const uint8_t*)p, (const uint8_t*)p + n); };
    auto put32 = [&](int32_t v) { put(&v, 4); };
    auto putstr = [&](const char* s) { put(s, std::strlen(s) + 1); };
    auto attr = [&](const char* name, const char* type, const std::vector<uint8_t>& val) {
        putstr(name); putstr(type); put32((int32_t)val.size()); put(val.data(), val.size());
    };
    const uint8_t magic[4] = {0x76, 0x2f, 0x31, 0x01};
    put(magic, 4);
    put32(2);
    std::vector<uint8_t> ch;
    for (const char* n : {"B", "G", "R"}) {
        ch.insert(ch.end(), n, n + 2);
        int32_t t = 2;   // FLOAT
        ch.insert(ch.end(), (uint8_t*)&t, (uint8_t*)&t + 4);
        uint8_t pl[4] = {0, 0, 0, 0};
        ch.insert(ch.end(), pl, pl + 4);
        int32_t s = 1;
        ch.insert(ch.end(), (uint8_t*)&s, (uint8_t*)&s + 4);
        ch.insert(ch.end(), (uint8_t*)&s, (uint8_t*)&s + 4);
    }
    ch.push_back(0);
    attr("channels", "chlist", ch);
    attr("compression", "compression", {0});
    int32_t box[4] = {0, 0, w - 1, h - 1};
    std::vector<uint8_t> bv((uint8_t*)box, (uint8_t*)box + 16);
    attr("dataWindow", "box2i", bv);
    attr("displayWindow", "box2i", bv);
    attr("lineOrder", "lineOrder", {0});
    float one = 1.f, zero[2] = {0.f, 0.f};
    attr("pixelAspectRatio", "float", std::vector<uint8_t>((uint8_t*)&one, (uint8_t*)&one + 4));
    attr("screenWindowCenter", "v2f", std::vector<uint8_t>((uint8_t*)zero, (uint8_t*)zero + 8));
    attr("screenWindowWidth", "float", std::vector<uint8_t>((uint8_t*)&one, (uint8_t*)&one + 4));
    hd.push_back(0);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(hd.data(), 1, hd.size(), f);
    uint64_t line_bytes = 8 + (uint64_t)w * 3 * 4;
    uint64_t base = hd.size() + (uint64_t)h * 8;
    for (int y = 0; y < h; ++y) {
        uint64_t off = base + (uint64_t)y * line_bytes;
        std::fwrite(&off, 8, 1, f);
    }
    std::vector<float> line((size_t)w * 3);
    for (int y = 0; y < h; ++y) {
        int32_t yy = y, sz = w * 3 * 4;
        std::fwrite(&yy, 4, 1, f);
        std::fwrite(&sz, 4, 1, f);
        for (int c = 0; c < 3; ++c)
            for (int x = 0; x < w; ++x) line[(size_t)c * w + x] = rgb[((size_t)y * w + x) * 3 + (2 - c)];
        std::fwrite(line.data(), 4, line.size(), f);
    }
    return std::fclose(f) == 0;
}

bool write_texture(const std::string& path, int w, int h, int channels, const uint8_t* data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t hd[4] = {0x5854474cu /* "LGTX" */, (uint32_t)w, (uint32_t)h, (uint32_t)channels};
    std::fwrite(hd, 4, 4, f);
    std::fwrite(data, 1, (size_t)w * h * channels, f);
    return std::fclose(f) == 0;
}

bool read_texture(const std::string& path, int* w, int* h, int* channels, std::vector<uint8_t>* data) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t hd[4];
    if (std::fread(hd, 4, 4, f) != 4 || hd[0] != 0x5854474cu) { std::fclose(f); return false; }
    *w = (int)hd[1]; *h = (int)hd[2]; *channels = (int)hd[3];
    data->resize((size_t)*w * *h * *channels);
    size_t got = std::fread(data->data(), 1, data->size(), f);
    std::fclose(f);
    return got == data->size();
}
