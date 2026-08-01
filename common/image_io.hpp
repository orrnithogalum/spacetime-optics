#pragma once
#include <cstdint>
#include <string>
#include <vector>

// 8-bit PNG writer (zlib). channels = 1, 3 or 4
bool write_png(const std::string& path, int w, int h, int channels, const uint8_t* data, int level = 4);

/* PngImage
- PNG reader: non-interlaced, any colour type, bit depth 8 or 16 (16-bit samples are reduced to 8
  bits unless keep16 is set, in which case data holds big-endian 16-bit samples).
*/
struct PngImage {
    int w = 0, h = 0, channels = 0, depth = 8;
    std::vector<uint8_t> data;
};
bool read_png(const std::string& path, PngImage* img, std::string* err, bool keep16 = false);

// Scanline OpenEXR, uncompressed FLOAT RGB
bool write_exr(const std::string& path, int w, int h, const float* rgb);

// Prepared texture container: "LGTX" + uint32 w, h, channels + raw bytes
bool write_texture(const std::string& path, int w, int h, int channels, const uint8_t* data);
bool read_texture(const std::string& path, int* w, int* h, int* channels, std::vector<uint8_t>* data);
