/* SCENE
- Plain-old-data scene and frame descriptions shared by host and device.
*/

#pragma once
#include "kerr.hpp"

typedef unsigned long long tex_handle;   // == cudaTextureObject_t

enum RenderMode : int { MODE_VISIBLE = 0, MODE_RADIO = 1 };

struct SceneGPU {
    // Black hole & integrator
    double a, rh, r_isco;
    double tol;              // DP54 relative tolerance
    int    max_steps;
    double r_escape;         // rays moving outwards beyond this radius have escaped
    double r_stop;           // rays below this radius are captured (just outside the horizon)
    double step_frac;        // max fractional change of r / theta per step inside emitting matter
    int    mode;

    // Thin accretion disk (Novikov-Thorne, optically thick, true colour)
    int    disk_on;
    double disk_rin, disk_rout;
    const float* disk_lnT;   // ln T_eff(r) on a ln r grid
    int    disk_n;
    double disk_lr0, disk_dlr;
    float  disk_turb_amp;    // rms of ln(flux) fluctuations
    float  disk_turb_scale;  // feature size in units of r
    float  disk_tcorr;       // correlation time in local orbital periods
    float  disk_ld;          // 1 = electron-scattering limb darkening
    float  disk_fcol;        // colour-correction factor (spectral hardening)
    float  disk_plunge;      // relative emissivity of the plunging region inside the ISCO

    // Hot accretion flow (RIAF, thermal synchrotron, optically thin-ish at mm)
    int    flow_on;
    double n0;               // electron density scale (cm^-3) at r = 1
    double n_index, theta_index;
    double theta0;           // dimensionless electron temperature kT/mc^2 at r = 1
    double h_r;              // scale height H/R
    double beta_p;           // plasma beta
    double r_flow_max;
    double nu_obs;           // observing frequency (Hz) at the camera
    double rg_cm;            // GM/c^2 in cm
    double flow_kep;         // angular velocity as a fraction of Keplerian
    float  flow_turb_amp, flow_turb_scale, flow_tcorr;
    // Per-frame baked turbulence: density factor on (phi, ln R, z/R) for TURB_NT slices of retarded
    // time delta = t - t_ref(r) (t_ref = arrival time of a radial ray), so photon-ring echoes keep
    // their light-travel delays. turb_nt = 0 -> evaluate the noise inline
    tex_handle turb_tex[24];
    int    turb_nt;
    float  turb_lr0, turb_lr1, turb_zmax, turb_d0, turb_dd;
    const float* k2_lntab;   // ln K2(1/Theta) on a ln Theta grid
    int    k2_n;
    float  k2_l0, k2_dl;

    // Celestial sphere (NASA SVS Deep Star Maps, galactic)
    int    sky_on;
    tex_handle sky_tex;      // half2: (s, s*T/1e4), radiance = s * B(T)
    float  sky_scale;
    dmat3  sky_rot;          // black-hole frame -> galactic

    // Blackbody colour table
    const float* bb_xyz;     // XYZ (W m^-2 sr^-1 weighted) of B(T), on a ln T grid
    int    bb_n;
    float  bb_l0, bb_dl;

    uint32_t seed;
};

struct FrameGPU {
    double r, th, ph, t;     // camera position (Boyer-Lindquist) and coordinate time
    KerrPoint kp;            // metric functions at the camera
    dvec3  F, R, U;          // camera forward/right/up in the ZAMO frame (e_r, e_theta, e_phi)
    dvec3  beta;             // camera velocity relative to the ZAMO
    double cx, cy;           // view-centre offset (tangent-plane units)
    double pix;              // tangent-plane units per pixel
    int    W, H, spp;
    uint32_t seed;
};
