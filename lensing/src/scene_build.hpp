/* SCENE BUILD
- Builds the device scene from a configuration: ephemerides -> lens frame, gravity terms, atmosphere
  coefficients per band, Sun and target spectra.
*/

#pragma once
#include <string>
#include <vector>

#include "config.hpp"
#include "ephemeris.hpp"
#include "renderer.hpp"
#include "scene.hpp"

struct SceneInfo {
    std::string target_name;
    std::string target_type;
    double target_dist = 0;       // m
    double target_radius = 0;     // m
    Epoch  epoch{};
    double phase_deg = 0;         // Sun-Earth-telescope angle
    double sub_lat = 0, sub_lon = 0;     // sub-telescope point on Earth (deg)
    double sun_sep_au = 0;        // projected Sun-Earth separation across the line of sight (AU)
    double focal_au = 0;          // distance at which the Einstein ring grazes the equator (vacuum)
    double focal_atm_au = 0;      // same, for a ring at 110 km altitude (refraction ~ gravity)
    double refractivity[NB];      // (n-1) of standard air per band
    ColorSettings color;
    std::vector<ImageTexData> images;
    std::string earth_dir;
};

bool build_scene(const Config& cfg, SceneGPU* S, SceneInfo* info, std::string* err);
/* scene_geometry
- Epoch-dependent part only (Earth orientation, Sun, J2 projection, solar tide, solar flux): used
  per frame for time-lapse paths.
*/
void scene_geometry(const Config& cfg, double jd, SceneGPU* S, SceneInfo* info);
bool load_earth_textures(const std::string& dir, EarthTextureData* out, std::string* err);

// Einstein angle for a telescope at distance D (m) behind Earth, source at distance d (m)
double einstein_angle(double D, double d);
