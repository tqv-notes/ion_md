#ifndef IONMD_CONSTANTS_HPP
#define IONMD_CONSTANTS_HPP

/**
 * Physical constants (SI units, CODATA 2018).
 */
namespace ionmd { namespace constants {

constexpr double pi = 3.14159265358979323846264338327950;

/// Atomic mass unit [kg]
constexpr double amu = 1.66053906660e-27;

/// Elementary charge [C]
constexpr double q_e = 1.602176634e-19;

/// Speed of light [m/s]
constexpr double c = 2.99792458e8;

/// Vacuum permittivity [F/m]
constexpr double epsilon0 = 8.8541878128e-12;

/// Coulomb constant 1/(4 pi epsilon0) [N m^2 / C^2]
constexpr double k_e = 8.9875517923e9;

/// Reduced Planck constant [J s]
constexpr double hbar = 1.054571817e-34;

/// Boltzmann constant [J/K]
constexpr double kB = 1.380649e-23;

// Legacy aliases used by the original IonMD code.
constexpr double OOFPEN = k_e;
constexpr double HBAR = hbar;

} }  // namespace ionmd::constants

#endif
