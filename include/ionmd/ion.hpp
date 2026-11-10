#ifndef IONMD_ION_HPP
#define IONMD_ION_HPP

#include <string>
#include <ionmd/constants.hpp>
#include <ionmd/vec3.hpp>

namespace ionmd {

/**
 * Description of a single ion. The dynamical state of all ions is stored by
 * the Simulation in structure-of-arrays form for performance (OpenMP and CUDA);
 * this struct is only used to define ions and to read them back.
 */
struct Ion
{
    /// Mass [kg]
    double m = 40*constants::amu;

    /// Charge in units of e
    double Z = 1.0;

    /// Whether the cooling lasers act on this ion.
    bool laser_cooled = true;

    /// Initial position [m]
    Vec3 x0;

    /// Initial velocity [m/s]
    Vec3 v0;

    /**
     * Stochastic heating rate h [K/s] for this ion (Zhang et al. Eq. 12,
     * h = m v_kick^2 / (2 dt kB)). Negative means "use the global value".
     */
    double heating_rate = -1.0;

    /// Species label (ions with equal labels are grouped in the output).
    std::string label;

    Ion() = default;

    /**
     * @param m Mass [kg]
     * @param Z Charge in units of e
     * @param x0 Initial position [m]
     * @param laser_cooled Whether lasers act on this ion
     * @param label Species label (defaults to "m<mass in amu>")
     */
    Ion(double m, double Z, Vec3 x0 = Vec3(), bool laser_cooled = true,
        std::string label = "")
        : m(m), Z(Z), laser_cooled(laser_cooled), x0(x0), label(std::move(label))
    {}

    double charge() const { return Z*constants::q_e; }
    double mass_amu() const { return m/constants::amu; }
};

}  // namespace ionmd

#endif
