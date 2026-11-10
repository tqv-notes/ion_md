#ifndef IONMD_TRAP_HPP
#define IONMD_TRAP_HPP

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <ionmd/constants.hpp>
#include <ionmd/vec3.hpp>

namespace ionmd {

/**
 * Linear Paul trap.
 *
 * The electric potential near the trap centre is (Zhang et al., PRA 76,
 * 012719 (2007), Eq. 2; P. K. Ghosh, "Ion Traps"):
 *
 *     phi(x,y,z,t) = (U_dc + V_rf cos(Omega t + phase)) (x^2 - y^2) / (2 r0^2)
 *                  + kappa U_ec (2 z^2 - x^2 - y^2) / (2 z0^2)
 *
 * so that the Mathieu parameters are
 *
 *     q = 2 Q V_rf / (m r0^2 Omega^2),   a = 4 Q U_dc / (m r0^2 Omega^2)
 *
 * and the secular (pseudopotential) frequencies are
 *
 *     omega_z^2 = 2 Q kappa U_ec / (m z0^2)
 *     omega_x^2 = (Omega^2/4)( a + q^2/2) - omega_z^2/2
 *     omega_y^2 = (Omega^2/4)(-a + q^2/2) - omega_z^2/2
 *
 * All quantities are SI. Q is the ion charge in Coulombs, m the mass in kg.
 *
 * Note: the original IonMD pseudopotential force was a factor 4 larger in the
 * rf term than the convention used for q in paultrap.py and in the paper; this
 * implementation uses one consistent convention for the full rf force, the
 * pseudopotential and the Mathieu parameters.
 */
struct Trap
{
    /// Trap radius (centre to electrode surface) [m]
    double r0 = 3.5e-3;

    /// Half the distance between end caps [m]
    double z0 = 2.7e-3;

    /// Geometric factor for the end cap potential (determined empirically)
    double kappa = 0.248;

    /// rf angular frequency [rad/s]
    double omega_rf = 2*constants::pi*4.972e6;

    /// rf amplitude [V]
    double V_rf = 300.0;

    /// dc offset of the rf electrodes [V]
    double U_dc = 0.0;

    /// End cap voltage [V]
    double U_ec = 1.0;

    /// rf phase at t = 0 [rad]
    double rf_phase = 0.0;

    // ------------------------------------------------------------------
    // Stability parameters and secular frequencies
    // ------------------------------------------------------------------

    double mathieu_q(double m, double Q) const
    {
        return 2*Q*V_rf/(m*r0*r0*omega_rf*omega_rf);
    }

    double mathieu_a(double m, double Q) const
    {
        return 4*Q*U_dc/(m*r0*r0*omega_rf*omega_rf);
    }

    /// Spring constants k = m omega^2 of the pseudopotential [N/m].
    Vec3 spring_constants(double m, double Q) const
    {
        const double rf = Q*Q*V_rf*V_rf/(2*m*omega_rf*omega_rf*r0*r0*r0*r0);
        const double dc_rad = Q*U_dc/(r0*r0);
        const double ec = Q*kappa*U_ec/(z0*z0);
        return Vec3(rf + dc_rad - ec, rf - dc_rad - ec, 2*ec);
    }

    /// Secular angular frequencies (NaN along an unstable direction).
    Vec3 secular_frequencies(double m, double Q) const
    {
        const Vec3 k = spring_constants(m, Q);
        auto w = [m](double kk) { return kk >= 0 ? std::sqrt(kk/m) : std::nan(""); };
        return Vec3(w(k.x), w(k.y), w(k.z));
    }

    // ------------------------------------------------------------------
    // Forces and potential energies
    // ------------------------------------------------------------------

    /// Time-averaged (pseudopotential) force.
    Vec3 pseudo_force(const Vec3 &r, double m, double Q) const
    {
        const Vec3 k = spring_constants(m, Q);
        return Vec3(-k.x*r.x, -k.y*r.y, -k.z*r.z);
    }

    /// Pseudopotential energy [J].
    double pseudo_potential(const Vec3 &r, double m, double Q) const
    {
        const Vec3 k = spring_constants(m, Q);
        return 0.5*(k.x*r.x*r.x + k.y*r.y*r.y + k.z*r.z*r.z);
    }

    /// Full time-dependent force including rf micromotion.
    Vec3 rf_force(const Vec3 &r, double Q, double t) const
    {
        const double c = (U_dc + V_rf*std::cos(omega_rf*t + rf_phase))/(r0*r0);
        const double d = kappa*U_ec/(z0*z0);
        return Vec3(-Q*(c - d)*r.x, Q*(c + d)*r.y, -2*Q*d*r.z);
    }

    /// Instantaneous potential energy in the full time-dependent field [J].
    double rf_potential(const Vec3 &r, double Q, double t) const
    {
        const double c = (U_dc + V_rf*std::cos(omega_rf*t + rf_phase))/(r0*r0);
        const double d = kappa*U_ec/(z0*z0);
        return Q*(0.5*c*(r.x*r.x - r.y*r.y)
                  + 0.5*d*(2*r.z*r.z - r.x*r.x - r.y*r.y));
    }

    /// rf period [s]
    double rf_period() const { return 2*constants::pi/omega_rf; }

    std::string to_string() const
    {
        std::stringstream s;
        s << "Trap parameters:\n"
          << "  r0 = " << r0 << " m\n"
          << "  z0 = " << z0 << " m\n"
          << "  kappa = " << kappa << "\n"
          << "  f_rf = " << omega_rf/(2*constants::pi) << " Hz\n"
          << "  V_rf = " << V_rf << " V\n"
          << "  U_dc = " << U_dc << " V\n"
          << "  U_ec = " << U_ec << " V\n";
        return s.str();
    }
};

typedef std::shared_ptr<Trap> trap_ptr;

}  // namespace ionmd

#endif
