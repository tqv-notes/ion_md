#ifndef IONMD_LASER_HPP
#define IONMD_LASER_HPP

#include <memory>
#include <vector>
#include <ionmd/constants.hpp>
#include <ionmd/vec3.hpp>

namespace ionmd {

/**
 * Doppler cooling laser using the damping model of Zhang et al.,
 * PRA 76, 012719 (2007), Eq. 14:
 *
 *     F_laser = -beta v + F0 khat
 *
 * i.e. a viscous damping force plus a constant light pressure force along the
 * beam. With `directional = true` only the velocity component along the beam
 * is damped: F = -beta (v . khat) khat + F0 khat (a single beam; the radial
 * motion is then only cooled indirectly through the Coulomb coupling).
 *
 * Lasers act on all ions flagged as `laser_cooled`.
 */
class Laser
{
public:
    /// Unit vector along the beam
    Vec3 khat{0, 0, 1};

    /// Damping (friction) coefficient [kg/s]
    double beta = 2e-22;

    /// Constant light pressure force [N]
    double F0 = 0.0;

    /// Damp only the velocity component along khat.
    bool directional = false;

    /// Angular frequency detuning (informational only in the damping model)
    double detuning = 0.0;

    Laser() = default;

    /**
     * @param beta Damping coefficient
     * @param F0 Constant light pressure force
     * @param khat Beam direction (normalised internally)
     */
    Laser(double beta, double F0, Vec3 khat, bool directional = false)
        : khat(khat.normalized()), beta(beta), F0(F0), directional(directional)
    {}

    Vec3 force(const Vec3 &v) const
    {
        if (directional) {
            return (F0 - beta*dot(v, khat))*khat;
        }
        return F0*khat - beta*v;
    }
};

typedef std::shared_ptr<Laser> laser_ptr;
typedef std::vector<laser_ptr> lasers_ptr;

}  // namespace ionmd

#endif
