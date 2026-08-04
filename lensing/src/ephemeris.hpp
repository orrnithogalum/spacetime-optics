/* EPHEMERIS
- Low-precision ephemerides (Astronomical Almanac level, ~0.01 deg for the Sun), sidereal time and
  IAU 1976 precession.
- Good enough to place the terminator, the sub-solar point and the telescope's view of Earth
  correctly to a fraction of a pixel at the scales simulated here.
*/

#pragma once
#include <string>
#include "common.hpp"

struct Epoch {
    double jd_utc;   // Julian date (UTC, used as UT1)
};

bool parse_iso8601(const std::string& s, Epoch* out);     // "2026-05-20T09:37:00"
std::string format_epoch(const Epoch& e);

dvec3  radec_to_unit(double ra_deg, double dec_deg);
dvec3  precess_j2000_to_date(dvec3 v, double jd);
double gmst_rad(double jd_ut1);
dvec3  sun_position_eci(double jd, double* dist_au);       // unit vector Earth->Sun (of date)
double obliquity_rad(double jd);
