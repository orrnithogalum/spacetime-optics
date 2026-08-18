/* RENDERER
- GPU renderer interface (implementation in kernels.cu).
- One Renderer per GPU / MPI rank.
*/

#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "scene.hpp"

struct EarthTextureData {
    int w = 0, h = 0;
    std::vector<uint8_t> albedo;   // RGBA8: sRGB albedo, A = water mask
    std::vector<uint8_t> aux;      // RGBA8: R clouds, G night lights, B elevation, A unused
};

struct ImageTexData {
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;     // RGBA8 sRGB
};

struct ColorSettings {
    float wx[NB], wy[NB], wz[NB];  // CIE weights per band (nm)
    float white[3];                // linear-sRGB of the white reference (normalised to Y = 1)
};

struct TelescopeSettings {
    bool   enabled = false;
    double aperture = 0;           // m
    double obstruction = 0.12;     // central obstruction (fraction of diameter)
    int    spiders = 4;
    double spider_width = 0.01;    // fraction of diameter
    int    W = 1024, H = 1024;
    bool   noise = false;
    double exposure_s = 1.0;
    double qe = 0.8;
    double read_noise = 3.0;       // electrons rms
};

struct RenderSettings {
    int   W = 1920, H = 1080;
    int   spp = 16;                // first-pass samples per pixel
    int   spp_max = 256;           // adaptive cap
    float target_err = 0.02f;      // relative standard error that stops refinement
    TelescopeSettings tele;
};

/* HdrFrame
- Linear HDR frame: white-balanced linear sRGB scaled so that Y is luminance in cd/m^2.
- (float32: stellar surfaces reach 1e9 cd/m^2, beyond half-float range.)
*/
struct HdrFrame {
    int W = 0, H = 0;
    std::vector<float> px;         // RGB triplets
    double lum_percentile = 0;     // luminance at the exposure percentile (cd/m^2)
    double lum_mean = 0;
    double flux = 0;               // integrated luminance x solid angle (cd/m^2 sr) over the frame
};

struct HudGlyph { float x, y, scale; int glyph; uint32_t rgba; };
struct HudRect { int x0, y0, x1, y1; uint32_t rgba; int layer; };   // layer 0 drawn before 1

struct RenderStats {
    double ms_pass1 = 0, ms_pass2 = 0, ms_post = 0;
    long long samples = 0;
};

int lensim_device_count();

class Renderer {
public:
    Renderer();
    ~Renderer();
    bool init(int device, const SceneGPU& scene, const double refractivity_k[NB],
              const EarthTextureData* earth, const std::vector<ImageTexData>& images,
              const RenderSettings& rs, const ColorSettings& cs, std::string* err);

    /* render
    - Renders the ideal (infinite-aperture) view and, if enabled, the telescope view whose pixel
      scale is tele_pix (rad).
    */
    void render(const FrameGPU& f, double tele_pix, double percentile, HdrFrame* ideal,
                HdrFrame* tele, RenderStats* stats);

    // Replace the epoch-dependent fields (Earth orientation, Sun, J2 projection, solar tide)
    void update_geometry(const SceneGPU& g);

    // Exposure multiplies luminance in cd/m^2 before the filmic curve
    void tonemap(const HdrFrame& in, float exposure, const std::vector<HudGlyph>& glyphs,
                 const std::vector<HudRect>& rects, std::vector<uint8_t>* rgb8);

    // Diagnostics computed from the device tables (refraction etc.)
    void report(FILE* out);

private:
    struct Impl;
    Impl* p_;
};
