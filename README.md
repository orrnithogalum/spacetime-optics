# Spacetime Optics

Spacetime Optics is a physically based GPU renderer for light bent by gravity, written in C++ and CUDA.  
It shows what a telescope would see from Earth's gravitational focal point.

## Earth as a gravitational lens

A telescope 4,000 to 90,000 AU behind Earth, lined up with an exoplanet (here Proxima b), sees the planet's light bent into an Einstein ring around Earth.

### Features

- General relativity deflection, with Earth's oblateness (J2) and the Sun's tidal field
- Atmospheric refraction up to 1000 km, which beats gravity below ~107 km
- Earth rendered from NASA data:
  - Blue Marble surface, clouds and city lights
  - Ocean sun glint
  - Atmosphere scattering, ozone and airglow
- Spectral rendering (16 bands)
