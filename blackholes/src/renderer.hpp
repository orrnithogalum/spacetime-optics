/* RENDERER
- GPU renderer interface (implementation in render.cu).
- One Renderer per GPU / MPI rank.
*/

#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "scene.hpp"

struct SkyTexture {
    int w = 0, h = 0;
    std::vector<uint16_t> data;   // half2 per texel: (s, s*T/1e4)
};

struct SceneTables {
    std::vector<float> disk_lnT;   // ln T(r) on the scene's ln r grid
    std::vector<float> k2_ln;      // ln K2(1/Theta)
    std::vector<float> bb_xyz;     // blackbody XYZ per ln T step (3 floats each)
};

struct RenderSettings {
    int W = 1920, H = 1080;
    int spp = 8, spp_max = 128;
    float target_err = 0.03f;
};

struct PostParams {
    float  bloom = 0.f;            // fraction of light scattered into the glare (visible mode)
    double blur_sigma_px = 0.0;    // gaussian beam (radio mode), in pixels
    float  hide_direct = 0.f;      // radio: fraction of the n = 0 (direct) light removed (for statistics)
};

/* HdrFrame
- Linear HDR frame, 4 floats per pixel: visible: linear sRGB (luminance units) + mean photon-ring
  order radio  : I_nu (cgs) + mean order, 0, 0
*/
struct HdrFrame {
    int W = 0, H = 0;
    std::vector<float> px;
    double lum_percentile = 0;
    double lum_max = 0;
};

struct ToneParams {
    int   mode = MODE_VISIBLE;
    float exposure = 1.f;
    int   colormap = 0;            // radio: 0 afmhot (EHT), 1 inferno
    float gamma = 1.f;             // radio display gamma
    float overlay = 0.f;           // photon-ring order tint
    float hide_direct = 0.f;       // radio: remove the direct (n = 0) image, keep light that orbited
    float fade = 1.f;              // fade to black
    float vignette = 0.f;
    float grain = 0.f;
    uint32_t seed = 0;
};

struct HudGlyph { float x, y, scale; int glyph; uint32_t rgba; };
struct HudRect { int x0, y0, x1, y1; uint32_t rgba; int layer; };

struct RenderStats {
    double ms_trace = 0, ms_post = 0;
    long long samples = 0;
    long long step_limited = 0;
};

int bh_device_count();

class Renderer {
public:
    Renderer();
    ~Renderer();
    bool init(int device, const SceneGPU& scene, const SceneTables& tables, const SkyTexture* sky,
              const RenderSettings& rs, std::string* err);
    void update_scene(const SceneGPU& s);   // scalar parameters only (tables stay)
    void render(const FrameGPU& f, const PostParams& post, double percentile, HdrFrame* out, RenderStats* st);
    // Total flux density (Jy) of a quick low-resolution radio render; pix_sr = pixel solid angle seen from Earth
    double radio_flux(FrameGPU f, int n, double pix_sr);
    void tonemap(const HdrFrame& in, const ToneParams& tp, const std::vector<HudGlyph>& glyphs,
                 const std::vector<HudRect>& rects, std::vector<uint8_t>* rgb8);

private:
    struct Impl;
    Impl* p_;
};
