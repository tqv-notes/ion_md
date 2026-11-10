#ifndef IONMD_SIMULATION_HPP
#define IONMD_SIMULATION_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <ionmd/coulomb.hpp>
#include <ionmd/ion.hpp>
#include <ionmd/laser.hpp>
#include <ionmd/params.hpp>
#include <ionmd/trap.hpp>
#include <ionmd/vec3.hpp>

namespace ionmd {

class DataWriter;

/// Simulation status
enum class SimStatus { IDLE, RUNNING, FINISHED, ERRORED, ABORTED };

std::string to_string(SimStatus s);

/// Group of ions sharing the same label (mass/charge species).
struct Species
{
    std::string label;
    double m = 0;            ///< mass [kg] (of the first member)
    double Z = 0;            ///< charge [e] (of the first member)
    std::vector<std::size_t> members;
};

/// Instantaneous energies of the ensemble [J].
struct Energies
{
    double kinetic = 0;
    double trap = 0;
    double coulomb = 0;
    double total() const { return kinetic + trap + coulomb; }
};

/// Information about a finished run.
struct RunSummary
{
    unsigned long steps_done = 0;
    double wall_time = 0;          ///< total wall time [s]
    double equilibrate_time = 0;   ///< wall time spent equilibrating [s]
    double production_time = 0;    ///< wall time of production steps [s]
    std::string backend;
    int threads = 1;
    std::string message;
};

/**
 * Molecular dynamics simulation of ions in a linear Paul trap.
 *
 * Equations of motion (Zhang et al., PRA 76, 012719, Eq. 1)
 *
 *     m_i a_i = F_trap + F_Coulomb + F_laser + F_excitation
 *
 * are integrated with the velocity Verlet (leapfrog) scheme. Stochastic
 * heating is modelled by velocity kicks of constant magnitude and random
 * direction (Eq. 10-12).
 *
 * The O(N) parts are parallelised with OpenMP; the O(N^2) Coulomb sum is
 * delegated to a CoulombSolver (OpenMP CPU or multi-GPU CUDA).
 */
class Simulation
{
public:
    Simulation();
    Simulation(SimParams p, Trap trap);
    Simulation(SimParams p, Trap trap, std::vector<Ion> ions);
    ~Simulation();

    Simulation(const Simulation &) = delete;
    Simulation &operator=(const Simulation &) = delete;

    // ---- Configuration ------------------------------------------------

    SimParams get_params() const { return *p; }
    void set_params(SimParams new_params);

    Trap get_trap() const { return *trap; }
    void set_trap(Trap new_trap);

    std::vector<Laser> get_lasers() const;
    void set_lasers(const std::vector<Laser> &lasers);
    void add_laser(const Laser &laser);

    /**
     * Make an ion with given initial position and zero velocity.
     * @param m Ion mass in amu
     * @param Z Ion charge in units of e
     * @param x0 Initial position [m]
     */
    Ion make_ion(double m, double Z, const std::vector<double> &x0) const;

    /// Add an ion (mass in amu, charge in e, position in m).
    void add_ion(double m, double Z, const std::vector<double> &x0);

    void add_ion(const Ion &ion);
    void set_ions(const std::vector<Ion> &ions);

    /// Ions with their *current* positions/velocities as x0/v0.
    std::vector<Ion> get_ions() const;

    std::size_t num_ions() const { return ions.size(); }

    // ---- Running ------------------------------------------------------

    /** Run the simulation (equilibration + production + output). Blocking. */
    void run();

    /** Start the simulation in a background thread. */
    void start();

    /** Wait for a background run to finish. */
    void wait();

    SimStatus status() const { return status_.load(); }

    /// Fraction of production steps completed (0..1).
    double progress() const;

    const RunSummary &summary() const { return summary_; }

    // ---- Low level interface (used by tests and bindings) --------------

    /**
     * Allocate the state arrays from the ion list and compute the initial
     * accelerations. Called automatically by run().
     */
    void initialize();

    /// Advance one production step (no output).
    void step();

    /// Run the equilibration ("minimization") phase.
    void equilibrate(unsigned long steps);

    double time() const { return t_; }
    unsigned long step_count() const { return step_index_; }

    Vec3 position(std::size_t i) const { return Vec3(x_[i], y_[i], z_[i]); }
    Vec3 velocity(std::size_t i) const { return Vec3(vx_[i], vy_[i], vz_[i]); }
    void set_velocity(std::size_t i, const Vec3 &v);

    /// Compute instantaneous energies of the current state.
    Energies energies();

    const std::vector<Species> &species() const { return species_; }

    std::string backend_name() const;

private:
    enum class Mode { PRODUCTION, EQUILIBRATION };

    params_ptr p;
    trap_ptr trap;
    std::vector<Laser> lasers;
    std::vector<Ion> ions;

    std::atomic<SimStatus> status_{SimStatus::IDLE};
    std::atomic<unsigned long> prod_done_{0};
    std::thread worker_;
    RunSummary summary_;

    // ---- Structure-of-arrays state ----
    bool initialized_ = false;
    std::size_t n_ = 0;
    double t_ = 0;
    unsigned long step_index_ = 0;     // counts all steps (RNG counter)
    std::vector<double> m_, q_, vkick_;
    std::vector<std::uint8_t> lc_;
    std::vector<int> species_of_;
    std::vector<double> x_, y_, z_, vx_, vy_, vz_, ax_, ay_, az_;
    std::vector<double> fcx_, fcy_, fcz_, phi_;
    std::vector<Species> species_;
    std::unique_ptr<CoulombSolver> solver_;

    // Diagnostics accumulators
    std::vector<double> vsx_, vsy_, vsz_;  // velocity sums over an rf period
    std::vector<double> sec_acc_;           // per species sum m <v>^2
    std::vector<double> ke_acc_;            // per species sum m v^2
    unsigned long sec_periods_ = 0, ke_samples_ = 0, avg_count_ = 0;
    unsigned long avg_len_ = 1;
    std::vector<double> ccd_;               // per species bins*bins

    void build_species();
    void compute_coulomb(bool with_phi);
    long accelerations(double t, Mode mode, bool finish_kick);
    void run_impl();
    bool advance(Mode mode, bool with_phi);
    void kick_velocities();
    bool check_bounds(std::string &msg) const;
    void thermalize(double T);

    void accumulate_diagnostics();
    void write_diagnostics(DataWriter &w);
    void accumulate_ccd();
    std::vector<double> centre_of_mass() const;
};

}  // namespace ionmd

#endif
