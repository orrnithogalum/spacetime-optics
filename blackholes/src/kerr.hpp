/* KERR
- Null geodesics of the Kerr metric (Boyer-Lindquist coordinates, G = c = M = 1).
- Photons are described by their constants of motion: energy E (normalised to 1), angular momentum
  lambda = L/E and Carter constant eta = Q/E^2.
- In Mino time tau (d tau = d affine/Sigma) the radial and polar motions decouple:
    (dr/dtau)^2 = R(r) = (r^2 + a^2 - a lam)^2 - Delta (eta + (lam - a)^2)
    (du/dtau)^2 = U(u) = (1 - u^2)(eta + a^2 u^2) - lam^2 u^2,      u = cos(theta)
- We integrate the *second-order* forms  r'' = R'(r)/2,  u'' = U'(u)/2, whose right-hand sides are
  polynomials: no square roots, no sign flips at turning points, perfectly smooth through the many
  polar oscillations of a photon-ring orbit. phi and t follow by quadrature.
- Rays are traced backwards from the camera; the state is
    y = (r, dr/dtau, u, du/dtau, phi, t)   (derivatives along the *backward* direction).
- Everything here is __host__ __device__ so the host can run the same integrator (critical-curve
  search, self-tests).
*/

#pragma once
#include "common.hpp"

struct KerrPoint {
    double r, s, c;                 // r, sin(theta), cos(theta)
    double Sigma, Delta, A, omega;  // metric functions; omega = frame-dragging angular velocity
    double e_nu, e_psi;             // ZAMO lapse and sqrt(g_phiphi)
    double sqrtSigma, sqrtSigmaDelta;
};

HD KerrPoint kerr_point(double a, double r, double th) {
    KerrPoint k;
    k.r = r;
    k.s = sin(th);
    k.c = cos(th);
    double a2 = a * a, r2 = r * r;
    k.Sigma = r2 + a2 * k.c * k.c;
    k.Delta = r2 - 2.0 * r + a2;
    k.A = (r2 + a2) * (r2 + a2) - a2 * k.Delta * k.s * k.s;
    k.omega = 2.0 * a * r / k.A;
    k.e_nu = sqrt(k.Sigma * k.Delta / k.A);
    k.e_psi = sqrt(k.A / k.Sigma) * k.s;
    k.sqrtSigma = sqrt(k.Sigma);
    k.sqrtSigmaDelta = sqrt(k.Sigma * k.Delta);
    return k;
}

HD double horizon_radius(double a) { return 1.0 + sqrt(1.0 - a * a); }

HD double isco_radius(double a) {   // prograde
    double z1 = 1.0 + cbrt(1.0 - a * a) * (cbrt(1.0 + a) + cbrt(1.0 - a));
    double z2 = sqrt(3.0 * a * a + z1 * z1);
    return 3.0 + z2 - sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2));
}

struct Photon {
    double lam, eta;   // L/E, Q/E^2
    double E;          // energy at infinity per unit camera-frame energy (nu_cam / nu_inf = 1/E)
    double y[6];
};

/* camera_photon
- Photon arriving at a camera at (r, theta, phi, t) travelling along -d, where d (the viewing
  direction) and beta (camera velocity) are given in the local ZAMO orthonormal frame with
  components (e_r, e_theta, e_phi).
*/
HD Photon camera_photon(double a, const KerrPoint& kp, double phi0, double t0, dvec3 d, dvec3 beta) {
    double k0 = 1.0;
    dvec3 k = -d;
    double b2 = dot(beta, beta);
    if (b2 > 0.0) {   // Lorentz boost camera frame -> ZAMO frame (aberration + Doppler)
        double b = sqrt(b2), g = 1.0 / sqrt(1.0 - b2);
        dvec3 bh = beta * (1.0 / b);
        double kpar = dot(k, bh);
        double k0z = g * (k0 + b * kpar);
        k = k + bh * ((g - 1.0) * kpar + g * b * k0);
        k0 = k0z;
    }
    double L = kp.e_psi * k.z;
    double E = kp.e_nu * k0 + kp.omega * L;
    double pth = kp.sqrtSigma * k.y;
    double Q = pth * pth + kp.c * kp.c * (L * L / (kp.s * kp.s) - a * a * E * E);
    Photon p;
    p.lam = L / E;
    p.eta = Q / (E * E);
    p.E = E;
    double dr = kp.sqrtSigmaDelta * k.x / E;   // photon dr/dtau (forward in time)
    double dth = pth / E;                       // photon dtheta/dtau
    p.y[0] = kp.r;
    p.y[1] = -dr;                               // backward tracing
    p.y[2] = kp.c;
    p.y[3] = kp.s * dth;                        // backward du/dtau = +sin(theta) dtheta/dtau
    p.y[4] = phi0;
    p.y[5] = t0;
    return p;
}

HD double radial_potential(double a, double lam, double eta, double r) {
    double P = r * r + a * a - a * lam, D = r * r - 2.0 * r + a * a;
    return P * P - D * (eta + (lam - a) * (lam - a));
}
HD double polar_potential(double a, double lam, double eta, double u) {
    return (1.0 - u * u) * (eta + a * a * u * u) - lam * lam * u * u;
}

// Backward-direction derivatives of the state
HD void geodesic_rhs(double a, double lam, double eta, const double* y, double* d) {
    double r = y[0], u = y[2];
    double r2 = r * r, a2 = a * a;
    double Delta = r2 - 2.0 * r + a2;
    double P = r2 + a2 - a * lam;
    double K = eta + (lam - a) * (lam - a);
    double s2 = fmax(1.0 - u * u, 1e-300);
    d[0] = y[1];
    d[1] = 2.0 * r * P - (r - 1.0) * K;
    d[2] = y[3];
    d[3] = u * (a2 * (1.0 - 2.0 * u * u) - eta - lam * lam);
    d[4] = -(a * P / Delta + lam / s2 - a);
    d[5] = -((r2 + a2) * P / Delta + a * (lam - a * s2));
}

/* dp54_step
- One Dormand-Prince 5(4) step (FSAL). k1 = f(y) on entry; on exit k7 = f(yout).
- Returns the scaled error norm (<= 1 means the step meets the tolerance).
*/
HD double dp54_step(double a, double lam, double eta, const double* y, const double* k1, double h, double tol,
                    double* yout, double* k7) {
    const double a21 = 1.0 / 5, a31 = 3.0 / 40, a32 = 9.0 / 40, a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
    const double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561, a54 = -212.0 / 729;
    const double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247, a64 = 49.0 / 176, a65 = -5103.0 / 18656;
    const double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192, b5 = -2187.0 / 6784, b6 = 11.0 / 84;
    const double e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920, e5 = -17253.0 / 339200, e6 = 22.0 / 525,
                 e7 = -1.0 / 40;
    double k2[6], k3[6], k4[6], k5[6], k6[6], t[6];
    for (int i = 0; i < 6; ++i) t[i] = y[i] + h * a21 * k1[i];
    geodesic_rhs(a, lam, eta, t, k2);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    geodesic_rhs(a, lam, eta, t, k3);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    geodesic_rhs(a, lam, eta, t, k4);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    geodesic_rhs(a, lam, eta, t, k5);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    geodesic_rhs(a, lam, eta, t, k6);
    for (int i = 0; i < 6; ++i) yout[i] = y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
    geodesic_rhs(a, lam, eta, yout, k7);
    // Scaled error: r relative, dr/dtau relative to r^2 (its natural size), u absolute,
    // du/dtau relative to max(1, |du|), phi absolute. t is not error-controlled
    double sc[5];
    sc[0] = tol * (1e-6 + fmax(fabs(y[0]), fabs(yout[0])));
    sc[1] = tol * (1e-6 + fmax(fmax(fabs(y[1]), fabs(yout[1])), y[0] * y[0]));
    sc[2] = tol;
    sc[3] = tol * fmax(1.0, fmax(fabs(y[3]), fabs(yout[3])));
    sc[4] = tol * 4.0;
    double m = 0.0;
    for (int i = 0; i < 5; ++i) {
        double e = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
        m = fmax(m, fabs(e) / sc[i]);
    }
    return m;
}

/* project_radial
- Constraint stabilisation. r'' = R'/2 conserves r'^2 - R(r) only to integration accuracy, and far
  from the hole the angular-momentum part of R (~ b^2 r^2) is a tiny fraction of r^4 (1e-13 at r =
  1e7): a 1e-13 drift would fabricate a turning point thousands of M away.
- Away from radial turning points we therefore re-project r' = +-sqrt(R(r)) after every step (k[0] =
  r' as well).
*/
HD void project_radial(double a, double lam, double eta, double* y, double* k) {
    double r = y[0], R = radial_potential(a, lam, eta, r);
    if (R > 0.25 * r * r * r * r) {
        double v = sqrt(R);
        y[1] = y[1] < 0.0 ? -v : v;
        k[0] = y[1];
    }
}

HD double initial_step(const double* y) {
    double r = y[0];
    return 0.02 * fmin(r / (fabs(y[1]) + 1e-12), 1.0 / (fabs(y[3]) + 1e-3));
}

/* vacuum_fate
- Plain vacuum trace (no emission): returns +1 escaped, -1 captured, 0 undecided.
- Used on the host to locate the critical curve (shadow edge) by bisection.
*/
HD int vacuum_fate(double a, Photon p, double tol, double r_escape, int max_steps) {
    double rh = horizon_radius(a);
    double* y = p.y;
    double k1[6], yn[6], k7[6];
    geodesic_rhs(a, p.lam, p.eta, y, k1);
    double h = initial_step(y);
    for (int it = 0; it < max_steps; ++it) {
        double err = dp54_step(a, p.lam, p.eta, y, k1, h, tol, yn, k7);
        if (err > 1.0) {
            h *= fmax(0.2, 0.9 * pow(err, -0.2));
            continue;
        }
        for (int i = 0; i < 6; ++i) { y[i] = yn[i]; k1[i] = k7[i]; }
        project_radial(a, p.lam, p.eta, y, k1);
        h *= fmin(5.0, fmax(0.2, 0.9 * pow(fmax(err, 1e-30), -0.2)));
        if (y[0] < rh * 1.0001 + 1e-3) return -1;
        if (y[0] > r_escape && y[1] > 0.0) return 1;
    }
    return 0;
}

/* critical_curve_point
- Bardeen critical curve (shadow edge) seen from infinity at inclination theta_o, parametrised by
  the radius rp of the spherical photon orbit.
- Returns false where the orbit is not visible.
*/
HD bool critical_curve_point(double a, double theta_o, double rp, double* alpha, double* beta_plus) {
    double D = rp * rp - 2.0 * rp + a * a;
    double lam = a + rp / a * (rp - 2.0 * D / (rp - 1.0));
    double eta = rp * rp * rp / (a * a) * (4.0 * D / ((rp - 1.0) * (rp - 1.0)) - rp);
    double so = sin(theta_o), co = cos(theta_o);
    double b2 = eta + a * a * co * co - lam * lam * co * co / (so * so);
    if (b2 < 0.0) return false;
    *alpha = -lam / so;
    *beta_plus = sqrt(b2);
    return true;
}
