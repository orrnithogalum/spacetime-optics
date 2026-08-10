/* SCENE
- Plain-old-data scene description shared by host and device.
- Everything the kernels need lives in SceneGPU (uploaded once to __constant__ memory) and FrameGPU
  (uploaded per frame).
- Coordinate frames
  ECI   : geocentric equatorial frame of date (x = equinox, z = celestial north pole).
- ECEF  : Earth-fixed frame, ECI rotated by Greenwich sidereal time.
- LENS  : origin at Earth's centre, +z points from Earth towards the telescope (i.e. away from
          the target), +y is the projection of celestial north, +x completes a right-handed
          frame. Looking from the telescope back at Earth, image x = +x and image y = +y.
*/

#pragma once
#include "common.hpp"

typedef unsigned long long tex_handle;   // == cudaTextureObject_t, kept opaque for host code

enum SourceType : int { SRC_NONE = 0, SRC_STAR = 1, SRC_PLANET = 2, SRC_IMAGE = 3 };
enum PlanetSurface : int { SURF_PROCEDURAL = 0, SURF_EARTH = 1 };

constexpr int MAX_SOURCES = 4;
constexpr int MAX_SPOTS = 8;

struct SourceGPU {
    int    type;
    int    surface;          // planets: procedural or Earth textures
    double dist;             // distance behind the lens (m), measured along -z
    double cx, cy;           // transverse centre at that distance (m)
    double radius;           // physical radius (m); images: half width
    double half_h;           // images: half height (m)
    float  Bk[NB];           // star: disk-centre radiance; planet: host irradiance (W/m^2/nm)
    float  ldk[NB];          // star: linear limb-darkening coefficient per band
    float  spot_tk[NB];      // star: spot/photosphere intensity ratio per band
    vec3   light;            // planet: unit vector towards host star (object frame, z -> Earth)
    mat3   rot;              // object frame -> body frame
    float  gran_contrast;    // star: granulation rms contrast at 500 nm
    float  gran_cells;       // star: granulation cells per stellar radius
    int    nspots;
    vec3   spot_dir[MAX_SPOTS];
    float  spot_size[MAX_SPOTS];   // angular radius (rad on the sphere)
    float  albedo_scale;     // planet
    float  cloud_cover;      // planet procedural
    float  ocean_level;      // planet procedural
    float  image_scale;      // image: radiance of white (W/m^2/sr/nm)
    tex_handle tex;          // image texture
    uint32_t seed;
};

struct SceneGPU {
    // Geometry
    dmat3  M_LE;             // LENS -> ECEF rotation
    dvec3  spin_L;           // Earth rotation axis in LENS frame
    vec3   sun_L;            // unit vector Earth -> Sun (LENS)
    dvec3  sun_pos_L;        // Sun centre (m, LENS)
    double sun_radius;
    double ell_a, ell_c;     // ellipsoid semi-axes (m)
    double Rs;               // Earth Schwarzschild radius (m)
    int    use_j2;
    double j2_Q;             // Rs * J2 * Req^2 * sin^2(beta_s)
    double j2_ex, j2_ey;     // projected spin axis direction in the transverse plane
    int    use_tide;
    double tide[4];          // 2x2 solar tidal deflection matrix (rad per m of impact parameter)

    // Atmosphere
    int    atmosphere;
    float  R_lut;            // reference radius used by the spherical LUTs (m)
    float  h_top;            // top of the scattering atmosphere (m)
    double h_top_refr;       // top of the refraction model (m)
    float  sigR[NB];         // Rayleigh scattering at sea level (1/m)
    float  sigMs[NB];        // aerosol scattering at sea level (1/m)
    float  sigMe[NB];        // aerosol extinction at sea level (1/m)
    float  sigO[NB];         // ozone absorption at the profile peak (1/m)
    float  mie_g, mie_H;
    float  E_sun[NB];        // solar spectral irradiance at Earth (W/m^2/nm)
    float  sun_Lk[NB];       // solar disk-centre spectral radiance (W/m^2/sr/nm)
    float  sun_ld[NB];       // solar limb darkening (linear law)
    float  h_ground_k[NB];   // apparent tangent altitude below which rays refract into the ground
    float  turb_amp;         // relative amplitude of density perturbations (gravity waves)
    float  turb_lh, turb_lv; // horizontal / vertical correlation lengths (m)
    float  airglow;          // airglow scale (1 = typical)
    int    airglow_band;
    float  ground_albedo_ms; // average ground albedo used by the multiple-scattering LUT

    // LUT storage (device pointers / texture objects)
    const float* lnrho;      // ln(rho/rho0) table, h from LNRHO_H0 in steps of LNRHO_DH
    int    lnrho_n;
    const float* refr_lna;   // [NB][refr_n] ln(refraction angle), indexed by apparent altitude
    const float* refr_tau;   // [NB][refr_n] optical depth along the refracted limb path
    const float* refr_htrue; // [refr_n] true tangent altitude (reference band)
    int    refr_n;
    float  refr_dh;
    tex_handle trans_tex;    // layered float4 x 4 : transmittance to top of atmosphere
    tex_handle ms_tex;       // layered float4 x 4 : multiple scattering (Hillaire 2020)
    const float* sky_irr;    // [SKY_N][NB] ground sky irradiance vs cos(sun zenith)

    // Earth surface
    int    has_textures;
    tex_handle albedo_tex;   // RGBA8 sRGB albedo + A = water mask
    tex_handle aux_tex;      // RGBA8: R clouds, G night lights, B elevation, A unused
    int    tex_w, tex_h;
    float  cloud_alt, cloud_opacity, cloud_albedo;
    float  night_scale;      // radiance of a saturated night-light pixel (W/m^2/sr/nm)
    float  ocean_wind;       // m/s for Cox-Munk
    float  relief;           // topographic normal exaggeration
    int    clouds, night_lights;
    float  city_spec[NB];
    float  water_refl[NB];   // diffuse water-leaving reflectance
    float  smits[7][NB];     // RGB -> spectrum basis (white, cyan, magenta, yellow, red, green, blue)

    // Background sources
    int       nsrc;
    SourceGPU src[MAX_SOURCES];
};

struct FrameGPU {
    double D;                // telescope distance from Earth's centre (m)
    double rho_x, rho_y;     // telescope lateral offset from the focal line (m)
    double pix;              // pixel scale (rad)
    double cx0, cy0;         // angle of the image centre relative to Earth's centre (rad)
    int    W, H;
    uint32_t seed;
    int    spp;
};

// Table layouts shared by host and device code
constexpr float LNRHO_H0  = -2000.0f;
constexpr float LNRHO_DH  = 250.0f;
constexpr int   TRANS_W = 256, TRANS_H = 64;
constexpr int   MS_N = 32;
constexpr int   SKY_N = 64;
