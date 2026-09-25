/* SHOT
- Shots: a camera + post-processing timeline.
- Any parameter listed below may be a constant (in its section) or animated in [track] as
  "time:value, time:value, ..." (seconds), interpolated with a monotone cubic (no overshoot), in log
  space for scale-like quantities.
*/

#pragma once
#include <string>
#include <vector>

#include "config.hpp"
#include "scene.hpp"

struct Track {
    std::vector<double> t, v;
    bool log = false;
    double value(double time) const;
};

struct ShotFrame {
    int index = 0;
    double t = 0;                    // seconds into the shot
    double r = 0, incl = 0, phi = 0; // camera position (M, rad, rad)
    double roll = 0, yaw = 0, pitch = 0;
    double fov = 0;                  // image width: M (far camera) or radians (near camera)
    bool fov_in_m = false;
    double zoom = 1;
    double target = 0;               // 0..1: view centre blends towards the zoom target
    double off_x = 0, off_y = 0;     // manual view-centre offset (M)
    double time_m = 0;               // simulation (coordinate) time, M
    double blur = 0, overlay = 0, hide_direct = 0, exposure_ev = 0, fade = 1, hud = 1;
};

struct Shot {
    std::string name, type;          // type: render | title
    double duration = 5, fps = 30;
    std::string motion;              // static | orbit
    double target_x = 0, target_y = 0;   // zoom target in the image plane (M)
    std::vector<ShotFrame> frames;
};

bool build_shot(const Config& c, Shot* shot, std::string* err);
// Camera of a frame -> device frame description (W, H, spp filled by the renderer)
FrameGPU make_frame(const Shot& shot, const ShotFrame& f, double a, int W, uint32_t seed);
/* find_critical_point
- Locate the shadow edge (critical curve) along image direction psi (rad) by bisection on the vacuum
  fate of camera rays; returns the distance from the image centre in M.
*/
double find_critical_point(const Shot& shot, const ShotFrame& f, double a, double psi);
