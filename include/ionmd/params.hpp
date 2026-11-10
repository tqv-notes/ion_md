#ifndef IONMD_PARAMS_HPP
#define IONMD_PARAMS_HPP

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <ionmd/vec3.hpp>

namespace ionmd {

/**
 * Uniform ac "tickle" field used for secular excitation of the ions
 * (thesis of C. B. Zhang, Ch. 10): F = Q E0 cos(2 pi f t) ehat for
 * t_start <= t < t_stop.
 */
struct Excitation
{
    double amplitude = 0.0;      ///< Field amplitude E0 [V/m]
    double frequency = 0.0;      ///< Frequency [Hz]
    Vec3 direction{1, 0, 0};     ///< Field direction (normalised on use)
    double t_start = 0.0;        ///< Switch-on time [s]
    double t_stop = 1e300;       ///< Switch-off time [s]
};

/**
 * Container structure for all parameters of a simulation.
 */
struct SimParams
{
    // --- Integration ---------------------------------------------------

    /// Time step [s]. Use ~1/50 of the fastest secular period with the
    /// pseudopotential, and ~1/50-1/100 of the rf period with micromotion.
    double dt = 20e-9;

    /// Total number of production time steps.
    unsigned long num_steps = 20000;

    /**
     * Number of "minimization" steps run before production (as in the
     * original IonMD minimize()): pseudopotential only, no heating, and a
     * strong damping `equilibrate_beta` applied to all ions. Velocities are
     * reset to zero afterwards.
     */
    unsigned long equilibrate_steps = 0;

    /// Damping coefficient used during equilibration [kg/s].
    double equilibrate_beta = 1e-20;

    /// If > 0, draw Maxwell-Boltzmann velocities at this temperature [K]
    /// once the equilibration is complete (before production).
    double initial_temperature = 0.0;

    // --- Forces --------------------------------------------------------

    /// Time-averaged pseudopotential trap force (ignored when micromotion is on)
    bool secular_enabled = true;

    /// Full time-dependent rf trap force (requires small dt)
    bool micromotion_enabled = false;

    /// Ion-ion Coulomb repulsion
    bool coulomb_enabled = true;

    /// Random velocity kicks modelling collisions / heating (Zhang Eq. 10)
    bool stochastic_enabled = false;

    /// Doppler cooling lasers (damping model)
    bool doppler_enabled = false;

    /// Secular excitation (ac tickle field)
    bool excitation_enabled = false;

    /// Global stochastic heating rate per ion [K/s]
    double heating_rate = 0.0;

    /// Excitation field parameters
    Excitation excitation;

    /// Abort if any ion leaves this radius (or becomes NaN). 0 disables.
    double abort_radius = 2e-3;

    /// Seed for the counter-based random number generator.
    std::uint64_t seed = 12345;

    // --- Parallelism ---------------------------------------------------

    /// Coulomb backend: "cpu", "cuda" or "auto".
    std::string backend = "auto";

    /// OpenMP threads (0 = OpenMP default / OMP_NUM_THREADS).
    int num_threads = 0;

    /// Number of GPUs to use with the CUDA backend (0 = all visible).
    int num_gpus = 0;

    /// With backend "auto", only use the GPU when N >= gpu_min_ions.
    unsigned int gpu_min_ions = 256;

    // --- Output --------------------------------------------------------

    /// Directory to write data to
    std::string path = "output";

    /// Allow writing into an existing output directory.
    bool overwrite = true;

    /// Write positions every `record_every` steps (0 disables trajectories).
    unsigned long record_every = 100;

    /// Only record trajectories from this step on.
    unsigned long record_start = 0;

    /// Also write velocities with the trajectory frames.
    bool record_velocities = false;

    /// Write energies/temperatures every `diag_every` steps (0 disables).
    unsigned long diag_every = 100;

    /// Write per-species centre-of-mass every `com_every` steps (0 disables).
    unsigned long com_every = 0;

    /// Simulated CCD image (2-D histogram of projected positions per species).
    bool ccd_enabled = false;
    unsigned int ccd_bins = 256;
    double ccd_extent = 300e-6;     ///< half width of the image [m]
    unsigned long ccd_every = 10;   ///< accumulate every n steps
    unsigned long ccd_start = 0;    ///< start accumulating at this step

    /// Print progress every `progress_every` steps (0 disables).
    unsigned long progress_every = 0;

    /// Verbosity of output. Higher numbers increase verbosity.
    unsigned int verbosity = 1;

    std::string to_string() const
    {
        std::stringstream s;
        s << "Simulation parameters:\n"
          << "  dt = " << dt << " s\n"
          << "  num_steps = " << num_steps << "\n"
          << "  equilibrate_steps = " << equilibrate_steps << "\n"
          << "  secular: " << secular_enabled << "\n"
          << "  micromotion: " << micromotion_enabled << "\n"
          << "  coulomb: " << coulomb_enabled << "\n"
          << "  stochastic: " << stochastic_enabled
          << " (heating_rate = " << heating_rate << " K/s)\n"
          << "  doppler: " << doppler_enabled << "\n"
          << "  excitation: " << excitation_enabled << "\n"
          << "  backend: " << backend << ", threads: " << num_threads
          << ", gpus: " << num_gpus << "\n"
          << "  path: " << path << "\n";
        return s.str();
    }
};

typedef std::shared_ptr<SimParams> params_ptr;

}  // namespace ionmd

#endif
