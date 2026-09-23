#pragma once
#include <string>

#include "config.hpp"
#include "renderer.hpp"
#include "scene.hpp"

struct SceneInfo {
    double mass_msun = 0;
    double distance_cm = 0;
    double rg_cm = 0;
    double tg_s = 0;             // GM/c^3
    double uas_per_m = 0;        // angular size of 1 M at the given distance
    double disk_tmax = 0;
    double disk_mdot_edd = 0;
    std::string sky_file;
    double radio_flux_target_jy = 0;
};

bool build_scene(const Config& cfg, SceneGPU* S, SceneTables* T, SceneInfo* info, std::string* err);
bool load_sky(const std::string& path, SkyTexture* sky, float* scale, std::string* err);
