#ifndef IONMD_UTIL_HPP
#define IONMD_UTIL_HPP

#include <cmath>
#include <cstdint>
#include <ctime>
#include <string>
#include <ionmd/constants.hpp>
#include <ionmd/vec3.hpp>

namespace ionmd {

inline std::string timestamp_str()
{
    const auto now = std::time(nullptr);
    char buff[64];
    std::strftime(buff, sizeof(buff), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    return std::string(buff);
}

/**
 * Counter-based random numbers.
 *
 * Every random number is a pure function of (seed, step, ion, stream), so
 * results are bit-for-bit identical regardless of the number of OpenMP
 * threads and of the order in which ions are processed.
 */
namespace rng {

inline std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline std::uint64_t hash(std::uint64_t seed, std::uint64_t a,
                          std::uint64_t b, std::uint64_t c)
{
    std::uint64_t h = splitmix64(seed);
    h = splitmix64(h ^ a);
    h = splitmix64(h ^ (b * 0xD6E8FEB86659FD93ull));
    h = splitmix64(h ^ (c * 0xA0761D6478BD642Full));
    return h;
}

/// Uniform double in [0, 1).
inline double uniform(std::uint64_t seed, std::uint64_t a,
                      std::uint64_t b, std::uint64_t c)
{
    return (hash(seed, a, b, c) >> 11) * (1.0/9007199254740992.0);
}

/// Uniformly distributed random unit vector.
inline Vec3 unit_vector(std::uint64_t seed, std::uint64_t step, std::uint64_t ion)
{
    const double u = 2*uniform(seed, step, ion, 0) - 1;      // cos(theta)
    const double phi = 2*constants::pi*uniform(seed, step, ion, 1);
    const double s = std::sqrt(1 - u*u);
    return Vec3(s*std::cos(phi), s*std::sin(phi), u);
}

/// Standard normal deviate (Box-Muller).
inline double normal(std::uint64_t seed, std::uint64_t a,
                     std::uint64_t b, std::uint64_t c)
{
    double u1 = uniform(seed, a, b, 2*c);
    const double u2 = uniform(seed, a, b, 2*c + 1);
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2*std::log(u1))*std::cos(2*constants::pi*u2);
}

}  // namespace rng

}  // namespace ionmd

#endif
