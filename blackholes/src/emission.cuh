/* EMISSION
- Emitting matter around the black hole.
- Thin disk: Novikov-Thorne (Page-Thorne flux, computed on the host), optically thick, each annulus
  radiating a (colour-corrected, limb-darkened) blackbody. Because I_nu/nu^3 is invariant, the
  observed spectrum of a blackbody of temperature T seen with redshift factor g = nu_obs/nu_emit is
  *exactly* a blackbody of temperature g*T: the observed colour is a table look-up, Doppler boosting
  and gravitational redshift included.
- Hot flow (RIAF, as around M87* and Sgr A*): thermal relativistic electrons emitting and absorbing
  synchrotron radiation (Leung et al. 2011 emissivity, Kirchhoff absorption) in a toroidal magnetic
  field, density/temperature power laws, Keplerian rotation.
- Both carry MHD-like turbulence: noise advected with the local orbital shear and renewed on a
  correlation time of a fraction of the orbital period, so features wind up into trailing spirals
  and decorrelate realistically. Light-travel-time delays are automatic because the noise is
  evaluated at the emission time t along each ray.
*/

#pragma once
#include "device_util.cuh"

DEVI double keplerian_omega(double R) { return 1.0 / (R * sqrt(R) + c_S.a); }

/* retarded_reference
- Coordinate time at which a radial light ray from radius r reaches the camera's distance, minus the
  camera time: the reference for retarded-time slices of baked turbulence.
*/
DEVI double retarded_reference(double r) {
    return c_F.t - (c_F.r - r + 2.0 * log(c_F.r / r));
}

/* sheared_noise
- Sheared, finite-lifetime turbulence at cylindrical radius R, azimuth phi, height z, time t.
- Returns a zero-mean, unit-variance field.
*/
DEVI float sheared_noise(double R, double phi, double z, double t, float scale, float tcorr, uint32_t seed) {
    double Om = keplerian_omega(fmax(R, c_S.r_isco));
    double P = 6.283185307179586 / Om * tcorr;   // decorrelation time
    double s = t / P;
    double k = floor(s);
    float f = (float)(s - k);
    float acc = 0.f;
    for (int j = 0; j < 2; ++j) {
        double tb = (k + j) * P;                  // birth time of this noise generation
        double ph = phi - Om * (t - tb);          // comoving azimuth at birth
        float inv = 1.f / (scale * (float)R);
        float x = (float)(R * cos(ph)) * inv, y = (float)(R * sin(ph)) * inv, zz = (float)z * inv * 2.f;
        float n = fbm3(x, y, zz, 4, seed + 7919u * (uint32_t)(long long)(k + j));
        float w = j == 0 ? 1.f - f : f;
        acc += w * n;
    }
    return acc * rsqrtf((1.f - f) * (1.f - f) + f * f);
}

// Thin disk
DEVI float disk_temperature(double r) {
    return __expf(table_lerp(c_S.disk_lnT, c_S.disk_n, (float)c_S.disk_lr0, (float)c_S.disk_dlr, (float)log(r)));
}

// Observed XYZ radiance and opacity of the disk at a crossing (r, phi, t) for photon (lam, eta)
DEVI float4 disk_shade(double r, double phi, double t, double lam, double eta, double Ecam) {
    if (r < c_S.disk_rin || r > c_S.disk_rout) return make_float4(0.f, 0.f, 0.f, 0.f);
    // Circular equatorial (prograde) orbit: u^t = (r^1.5 + a) / (r^0.75 sqrt(r^1.5 - 3 r^0.5 + 2a))
    double sr = sqrt(r);
    double ut = (r * sr + c_S.a) / (sqrt(sr) * sqrt(sr) * sqrt(sr) * sqrt(fmax(r * sr - 3.0 * sr + 2.0 * c_S.a, 1e-12)));
    double Om = keplerian_omega(r);
    double kdotu = ut * (1.0 - Om * lam);          // -k.u for E = 1
    double g = 1.0 / (Ecam * kdotu);               // nu_cam / nu_emit
    float T = disk_temperature(r);
    float m = 1.f;
    if (c_S.disk_turb_amp > 0.f) {
        float n = sheared_noise(r, phi, 0.0, t, c_S.disk_turb_scale, c_S.disk_tcorr, c_S.seed);
        m = __expf(c_S.disk_turb_amp * n - 0.5f * c_S.disk_turb_amp * c_S.disk_turb_amp);
    }
    T *= sqrtf(sqrtf(m));                          // flux fluctuation -> temperature
    // Limb darkening of an electron-scattering atmosphere, emission angle in the fluid frame
    float ld = 1.f;
    if (c_S.disk_ld > 0.f) {
        float mu = (float)fmin(sqrt(fmax(eta, 0.0)) / (r * kdotu), 1.0);
        ld = (1.f + 2.06f * mu) / (1.f + 2.06f * 2.f / 3.f);
    }
    float fc = c_S.disk_fcol;
    float3 xyz = blackbody_xyz((float)g * T * fc);
    float s = ld / (fc * fc * fc * fc);            // diluted, colour-corrected blackbody
    // Soft outer edge
    float edge = 1.f - smoothstepf((float)(0.75 * c_S.disk_rout), (float)c_S.disk_rout, (float)r);
    return make_float4(xyz.x * s * edge, xyz.y * s * edge, xyz.z * s * edge, edge);
}

// Hot accretion flow: thermal synchrotron
DEVI double bessel_k2_inv(double theta) {   // K2(1/Theta)
    return exp((double)table_lerp(c_S.k2_lntab, c_S.k2_n, c_S.k2_l0, c_S.k2_dl, (float)log(theta)));
}

/* flow_step
- Emission/absorption at (r, u, phi, t) for photon (lam, eta, Ecam) whose covariant momentum (E = 1)
  is k = (-1, k_r, k_theta, lam).
- Returns observed-frame source term: *dI_src  = g^3 S_nu (1 - e^-dtau),  *dtau = optical depth of
  the step of Mino length h.
*/
DEVI void flow_step(double r, double u, double phi, double t, double lam, double Ecam, double h,
                    double* dI, double* dtau) {
    *dI = 0.0;
    *dtau = 0.0;
    const double a = c_S.a;
    double s2 = fmax(1.0 - u * u, 1e-12), s = sqrt(s2);
    double R = r * s, z = r * u;
    if (r > c_S.r_flow_max || R < 1e-3) return;
    double zh = z / (c_S.h_r * R);
    double vert = exp(-0.5 * zh * zh);
    if (vert < 1e-7) return;   // far above the flow: nothing to emit or absorb
    double ne = c_S.n0 * exp(-c_S.n_index * log(r)) * vert;
    if (c_S.flow_turb_amp > 0.f) {
        if (c_S.turb_nt > 0) {
            // Baked turbulence: two trilinear fetches, interpolated in retarded time
            float u = (float)((log(R) - c_S.turb_lr0) / (c_S.turb_lr1 - c_S.turb_lr0));
            float v = (float)(phi * (0.5 / PI));
            float w = (float)(0.5 + 0.5 * (z / R) / c_S.turb_zmax);
            float s = (float)((t - retarded_reference(r)) - c_S.turb_d0) / c_S.turb_dd;
            s = fminf(fmaxf(s, 0.f), (float)(c_S.turb_nt - 1) - 1e-4f);
            int k = (int)s;
            float f = s - k;
            float A = tex3D<float>((cudaTextureObject_t)c_S.turb_tex[k], v, u, w);
            float B = tex3D<float>((cudaTextureObject_t)c_S.turb_tex[k + 1], v, u, w);
            ne *= A + (B - A) * f;
        } else {
            float n = sheared_noise(R, phi, z, t, c_S.flow_turb_scale, c_S.flow_tcorr, c_S.seed ^ 0x51f15eedu);
            ne *= exp((double)(c_S.flow_turb_amp * n - 0.5f * c_S.flow_turb_amp * c_S.flow_turb_amp));
        }
    }
    // Soften the inner edge near the horizon (the plunging gas is hot but tenuous)
    ne *= 1.0 - exp(-(r - c_S.rh) * (r - c_S.rh) / 0.25);
    if (ne < 1e-30) return;
    double Th = c_S.theta0 * exp(-c_S.theta_index * log(r));
    // Equipartition-like field: B^2/8pi = ne m_p c^2 / (6 r beta)
    double B = sqrt(8.0 * PI * ne * cgs::MP * cgs::C * cgs::C / (6.0 * r * c_S.beta_p));

    // Fluid four-velocity: rotation at a fraction of Keplerian (cylindrical), clamped to the
    // allowed range around frame dragging near the horizon
    double r2 = r * r, a2 = a * a;
    double Sigma = r2 + a2 * u * u, Delta = r2 - 2.0 * r + a2;
    double A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;
    double g_tt = -(1.0 - 2.0 * r / Sigma), g_tph = -2.0 * a * r * s2 / Sigma, g_phph = A * s2 / Sigma;
    double omz = 2.0 * a * r / A;
    double Om = c_S.flow_kep * keplerian_omega(fmax(R, c_S.r_isco));
    // Keep the orbit timelike: |Om - omega| < lapse / sqrt(g_phph)
    double vmax = sqrt(Sigma * Delta / A) / sqrt(g_phph) * 0.999;
    Om = omz + fmax(-vmax, fmin(vmax, Om - omz));
    double norm = -(g_tt + 2.0 * g_tph * Om + g_phph * Om * Om);
    if (norm <= 0.0) return;
    double ut = 1.0 / sqrt(norm);
    double kdotu = ut * (1.0 - Om * lam);               // -k.u
    double nu_f = c_S.nu_obs * Ecam * kdotu;            // fluid-frame frequency
    double g = c_S.nu_obs / nu_f;
    // Pitch angle w.r.t. a toroidal field: b ~ projection of d/dphi orthogonal to u
    double u_ph = ut * (g_tph + g_phph * Om);
    double kb = lam - kdotu * u_ph;                     // k.b with b = e_phi + u_phi u
    double bb = g_phph + u_ph * u_ph;
    double cosb = kb / (kdotu * sqrt(bb));
    double sinb = sqrt(fmax(1.0 - cosb * cosb, 1e-6));
    // Thermal synchrotron (Leung et al. 2011)
    double nuc = cgs::QE * B / (2.0 * PI * cgs::ME * cgs::C);
    double nus = 2.0 / 9.0 * nuc * Th * Th * sinb;
    double X = nu_f / nus;
    double x3 = cbrt(X);
    double k2 = bessel_k2_inv(Th);
    double brk = sqrt(X) + 1.887748625363387 * sqrt(x3);          // X^1/2 + 2^(11/12) X^1/6
    double j = ne * 1.4142135623730951 * PI * cgs::QE * cgs::QE * nus / (3.0 * k2 * cgs::C) * brk * brk * exp(-x3);
    double Te = Th * cgs::ME * cgs::C * cgs::C / cgs::KB;
    double hnu = cgs::H * nu_f / (cgs::KB * Te);
    double Bnu = 2.0 * cgs::H * nu_f * nu_f * nu_f / (cgs::C * cgs::C) / expm1(hnu);
    double alpha = j / Bnu;
    double dl = kdotu * Sigma * h * c_S.rg_cm;          // fluid-frame path length (cm)
    double tau = alpha * dl;
    *dtau = tau;
    *dI = g * g * g * Bnu * (-expm1(-tau));
}
