#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ionmd/json.hpp>

#include <ionmd/config.hpp>
#include <ionmd/constants.hpp>
#include <ionmd/data.hpp>
#include <ionmd/simulation.hpp>
#include <ionmd/util.hpp>

namespace ionmd {

using constants::kB;

namespace {

double wall_seconds()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

std::string default_label(const Ion &ion)
{
    std::ostringstream os;
    os << "m" << std::lround(ion.mass_amu());
    if (ion.Z != 1.0) os << "_Z" << ion.Z;
    return os.str();
}

const double NaN = std::numeric_limits<double>::quiet_NaN();

/// O(N) loops are only parallelised above this size: for small systems the
/// OpenMP fork/join overhead (~10-100 us) exceeds the work.
constexpr long OMP_MIN_N = 2048;

}  // namespace


std::string to_string(SimStatus s)
{
    switch (s) {
    case SimStatus::IDLE: return "IDLE";
    case SimStatus::RUNNING: return "RUNNING";
    case SimStatus::FINISHED: return "FINISHED";
    case SimStatus::ERRORED: return "ERRORED";
    case SimStatus::ABORTED: return "ABORTED";
    }
    return "UNKNOWN";
}


// ---------------------------------------------------------------------------
// Construction and configuration
// ---------------------------------------------------------------------------

Simulation::Simulation()
    : p(std::make_shared<SimParams>()), trap(std::make_shared<Trap>())
{}

Simulation::Simulation(SimParams p_, Trap trap_)
    : p(std::make_shared<SimParams>(p_)), trap(std::make_shared<Trap>(trap_))
{}

Simulation::Simulation(SimParams p_, Trap trap_, std::vector<Ion> ions_)
    : Simulation(p_, trap_)
{
    set_ions(ions_);
}

Simulation::~Simulation()
{
    if (worker_.joinable()) worker_.join();
}

void Simulation::set_params(SimParams new_params)
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Can't change parameters while the simulation is running");
    }
    const bool backend_changed = new_params.backend != p->backend
        || new_params.num_gpus != p->num_gpus;
    p = std::make_shared<SimParams>(new_params);
    if (backend_changed) solver_.reset();
}

void Simulation::set_trap(Trap new_trap)
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Can't change the trap while the simulation is running");
    }
    trap = std::make_shared<Trap>(new_trap);
}

std::vector<Laser> Simulation::get_lasers() const { return lasers; }

void Simulation::set_lasers(const std::vector<Laser> &l)
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Can't change lasers while the simulation is running");
    }
    lasers = l;
}

void Simulation::add_laser(const Laser &laser)
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Can't change lasers while the simulation is running");
    }
    lasers.push_back(laser);
}

Ion Simulation::make_ion(double m, double Z, const std::vector<double> &x0) const
{
    if (x0.size() != 3) {
        throw std::invalid_argument("Initial position must have 3 components");
    }
    return Ion(m*constants::amu, Z, Vec3(x0[0], x0[1], x0[2]));
}

void Simulation::add_ion(double m, double Z, const std::vector<double> &x0)
{
    add_ion(make_ion(m, Z, x0));
}

void Simulation::add_ion(const Ion &ion)
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Can't add ions while the simulation is running");
    }
    ions.push_back(ion);
    initialized_ = false;
}

void Simulation::set_ions(const std::vector<Ion> &new_ions)
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Can't change ions while the simulation is running");
    }
    ions = new_ions;
    initialized_ = false;
}

std::vector<Ion> Simulation::get_ions() const
{
    std::vector<Ion> out = ions;
    if (initialized_) {
        for (std::size_t i = 0; i < n_; i++) {
            out[i].x0 = position(i);
            out[i].v0 = velocity(i);
        }
    }
    return out;
}

void Simulation::set_velocity(std::size_t i, const Vec3 &v)
{
    vx_[i] = v.x; vy_[i] = v.y; vz_[i] = v.z;
}

std::string Simulation::backend_name() const
{
    if (!p->coulomb_enabled) return "none (Coulomb disabled)";
    return solver_ ? solver_->name() : std::string("(not initialized)");
}

double Simulation::progress() const
{
    const auto n = p->num_steps;
    return n ? double(prod_done_.load())/double(n) : 1.0;
}


// ---------------------------------------------------------------------------
// State setup
// ---------------------------------------------------------------------------

void Simulation::build_species()
{
    species_.clear();
    species_of_.assign(n_, 0);
    std::map<std::string, int> index;

    for (std::size_t i = 0; i < n_; i++) {
        std::string label = ions[i].label.empty() ? default_label(ions[i]) : ions[i].label;
        auto it = index.find(label);
        int s;
        if (it == index.end()) {
            s = static_cast<int>(species_.size());
            index[label] = s;
            Species sp;
            sp.label = label;
            sp.m = ions[i].m;
            sp.Z = ions[i].Z;
            species_.push_back(sp);
        }
        else {
            s = it->second;
        }
        species_[s].members.push_back(i);
        species_of_[i] = s;
    }
}

void Simulation::initialize()
{
    n_ = ions.size();
    if (n_ == 0) {
        throw std::runtime_error("No ions set!");
    }

    m_.resize(n_); q_.resize(n_); vkick_.resize(n_); lc_.resize(n_);
    x_.resize(n_); y_.resize(n_); z_.resize(n_);
    vx_.resize(n_); vy_.resize(n_); vz_.resize(n_);
    ax_.assign(n_, 0); ay_.assign(n_, 0); az_.assign(n_, 0);
    fcx_.assign(n_, 0); fcy_.assign(n_, 0); fcz_.assign(n_, 0); phi_.assign(n_, 0);

    for (std::size_t i = 0; i < n_; i++) {
        const Ion &ion = ions[i];
        if (!(ion.m > 0)) {
            throw std::runtime_error("Ion " + std::to_string(i) + " has non-positive mass");
        }
        m_[i] = ion.m;
        q_[i] = ion.charge();
        vkick_[i] = ion.heating_rate;   // per-ion heating rate (<0: global)
        lc_[i] = ion.laser_cooled ? 1 : 0;
        x_[i] = ion.x0.x; y_[i] = ion.x0.y; z_[i] = ion.x0.z;
        vx_[i] = ion.v0.x; vy_[i] = ion.v0.y; vz_[i] = ion.v0.z;
    }

    build_species();

    if (p->num_threads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(p->num_threads);
#endif
    }

    t_ = 0;
    step_index_ = 0;
    prod_done_ = 0;
    solver_.reset();
    if (p->coulomb_enabled) {
        solver_ = make_coulomb_solver(p->backend, n_, p->num_gpus, p->gpu_min_ions);
    }

    compute_coulomb(false);
    accelerations(t_, Mode::PRODUCTION, false);
    initialized_ = true;
}


// ---------------------------------------------------------------------------
// Forces and integration
// ---------------------------------------------------------------------------

void Simulation::compute_coulomb(bool with_phi)
{
    if (!p->coulomb_enabled || n_ < 2) {
        std::fill(fcx_.begin(), fcx_.end(), 0.0);
        std::fill(fcy_.begin(), fcy_.end(), 0.0);
        std::fill(fcz_.begin(), fcz_.end(), 0.0);
        std::fill(phi_.begin(), phi_.end(), 0.0);
        return;
    }
    if (!solver_) {
        solver_ = make_coulomb_solver(p->backend, n_, p->num_gpus, p->gpu_min_ions);
    }
    solver_->compute(n_, x_.data(), y_.data(), z_.data(), q_.data(),
                     fcx_.data(), fcy_.data(), fcz_.data(),
                     with_phi ? phi_.data() : nullptr);
}

/**
 * Compute accelerations at time t from the current positions, the
 * precomputed Coulomb forces and the current velocities (used by the
 * velocity dependent laser force).
 *
 * If `finish_kick` is true this is the second half of a velocity Verlet step:
 * on entry v holds the half step velocity v(t - dt/2) + ..., and on exit
 * v(t) = v_half + a(t) dt/2. Stochastic kicks are applied here as well.
 */
long Simulation::accelerations(double t, Mode mode, bool finish_kick)
{
    const SimParams &P = *p;
    const Trap &T = *trap;
    const double dt = P.dt;
    const long n = static_cast<long>(n_);

    const bool equil = (mode == Mode::EQUILIBRATION);
    const bool use_rf = P.micromotion_enabled && !equil;
    const bool use_pseudo = equil || (!P.micromotion_enabled && P.secular_enabled);
    const bool use_laser = P.doppler_enabled && !lasers.empty();
    const bool use_exc = P.excitation_enabled && !equil
        && t >= P.excitation.t_start && t < P.excitation.t_stop;
    const bool use_kicks = finish_kick && !equil && P.stochastic_enabled;

    Vec3 exc_field;
    if (use_exc) {
        exc_field = P.excitation.direction.normalized()
            * (P.excitation.amplitude*std::cos(2*constants::pi*P.excitation.frequency*t));
    }

    Vec3 light_pressure;  // sum of F0 khat over all lasers
    for (const auto &l : lasers) light_pressure += l.F0*l.khat;

    const unsigned long step = step_index_;
    const double R2 = P.abort_radius > 0 ? P.abort_radius*P.abort_radius
                                         : std::numeric_limits<double>::infinity();
    long bad = 0;

    #pragma omp parallel for schedule(static) reduction(+:bad) if(n >= OMP_MIN_N)
    for (long i = 0; i < n; i++) {
        const Vec3 r(x_[i], y_[i], z_[i]);
        // Bounds check (also catches NaN): positions are final for this step.
        const double r2 = r.norm2();
        if (!(r2 <= R2)) bad++;

        const Vec3 v(vx_[i], vy_[i], vz_[i]);
        const double m = m_[i];
        const double q = q_[i];

        Vec3 F(fcx_[i], fcy_[i], fcz_[i]);

        if (use_rf) {
            F += T.rf_force(r, q, t);
        }
        else if (use_pseudo) {
            F += T.pseudo_force(r, m, q);
        }

        if (equil) {
            F -= P.equilibrate_beta*v;
            if (use_laser && lc_[i]) F += light_pressure;
        }
        else if (use_laser && lc_[i]) {
            for (const auto &l : lasers) F += l.force(v);
        }

        if (use_exc) {
            F += q*exc_field;
        }

        const Vec3 a = F/m;
        ax_[i] = a.x; ay_[i] = a.y; az_[i] = a.z;

        if (finish_kick) {
            vx_[i] += 0.5*a.x*dt;
            vy_[i] += 0.5*a.y*dt;
            vz_[i] += 0.5*a.z*dt;
        }

        if (use_kicks) {
            const double h = vkick_[i] >= 0 ? vkick_[i] : P.heating_rate;
            if (h > 0) {
                // Zhang et al. Eq. 12: h kB = m v0^2 / (2 dt)
                const double v0 = std::sqrt(2*kB*h*dt/m);
                const Vec3 w = rng::unit_vector(P.seed, step, static_cast<std::uint64_t>(i));
                vx_[i] += v0*w.x;
                vy_[i] += v0*w.y;
                vz_[i] += v0*w.z;
            }
        }
    }
    return bad;
}

bool Simulation::advance(Mode mode, bool with_phi)
{
    const double dt = p->dt;
    const long n = static_cast<long>(n_);

    // Drift: x(t+dt) = x + v dt + a dt^2/2 ; half kick v -> v + a dt/2
    #pragma omp parallel for schedule(static) if(n >= OMP_MIN_N)
    for (long i = 0; i < n; i++) {
        x_[i] += (vx_[i] + 0.5*ax_[i]*dt)*dt;
        y_[i] += (vy_[i] + 0.5*ay_[i]*dt)*dt;
        z_[i] += (vz_[i] + 0.5*az_[i]*dt)*dt;
        vx_[i] += 0.5*ax_[i]*dt;
        vy_[i] += 0.5*ay_[i]*dt;
        vz_[i] += 0.5*az_[i]*dt;
    }
    t_ += dt;

    compute_coulomb(with_phi);
    const long bad = accelerations(t_, mode, true);
    step_index_++;
    return bad == 0;
}

bool Simulation::check_bounds(std::string &msg) const
{
    const double R = p->abort_radius;
    const long n = static_cast<long>(n_);
    long bad = 0;
    long first = -1;

    for (long i = 0; i < n; i++) {
        const double r2 = x_[i]*x_[i] + y_[i]*y_[i] + z_[i]*z_[i];
        if (!std::isfinite(r2) || (R > 0 && r2 > R*R)) {
            if (first < 0) first = i;
            bad++;
        }
    }
    if (bad == 0) return true;
    std::ostringstream os;
    os << bad << " ion(s) out of bounds (abort_radius = " << R << " m), e.g. ion "
       << first << " at (" << x_[first] << ", " << y_[first] << ", " << z_[first]
       << ") at t = " << t_ << " s. Is the time step too large or the trap unstable?";
    msg = os.str();
    return false;
}

void Simulation::step()
{
    if (!initialized_) initialize();
    if (!advance(Mode::PRODUCTION, false)) {
        std::string msg;
        check_bounds(msg);
        throw std::runtime_error(msg);
    }
}

void Simulation::equilibrate(unsigned long steps)
{
    if (!initialized_) initialize();
    if (steps == 0) return;

    for (unsigned long k = 0; k < steps; k++) {
        if (!advance(Mode::EQUILIBRATION, false)) {
            std::string msg;
            check_bounds(msg);
            throw std::runtime_error("During equilibration: " + msg);
        }
        if (p->verbosity >= 1 && p->progress_every && (k + 1) % p->progress_every == 0) {
            std::cout << "  equilibration " << (k + 1) << "/" << steps << std::endl;
        }
    }

    // Reset velocities and the clock (as the original minimize()).
    std::fill(vx_.begin(), vx_.end(), 0.0);
    std::fill(vy_.begin(), vy_.end(), 0.0);
    std::fill(vz_.begin(), vz_.end(), 0.0);
    t_ = 0;
    compute_coulomb(false);
    accelerations(t_, Mode::PRODUCTION, false);
}

void Simulation::thermalize(double T)
{
    const long n = static_cast<long>(n_);
    const std::uint64_t tag = 0x7E3A11F0ull;
    #pragma omp parallel for if(n >= OMP_MIN_N)
    for (long i = 0; i < n; i++) {
        const double s = std::sqrt(kB*T/m_[i]);
        vx_[i] = s*rng::normal(p->seed, tag, i, 0);
        vy_[i] = s*rng::normal(p->seed, tag, i, 1);
        vz_[i] = s*rng::normal(p->seed, tag, i, 2);
    }
    compute_coulomb(false);
    accelerations(t_, Mode::PRODUCTION, false);
}

Energies Simulation::energies()
{
    if (!initialized_) initialize();
    compute_coulomb(true);
    Energies E;
    const bool rf = p->micromotion_enabled;
    for (std::size_t i = 0; i < n_; i++) {
        const Vec3 r = position(i);
        E.kinetic += 0.5*m_[i]*velocity(i).norm2();
        E.trap += rf ? trap->rf_potential(r, q_[i], t_)
                     : trap->pseudo_potential(r, m_[i], q_[i]);
        E.coulomb += 0.5*q_[i]*phi_[i];
    }
    return E;
}


// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

void Simulation::accumulate_diagnostics()
{
    const std::size_t S = species_.size();
    if (sec_acc_.size() != S) {
        sec_acc_.assign(S, 0);
        ke_acc_.assign(S, 0);
    }

    for (std::size_t i = 0; i < n_; i++) {
        const int s = species_of_[i];
        const double mv2 = m_[i]*(vx_[i]*vx_[i] + vy_[i]*vy_[i] + vz_[i]*vz_[i]);
        ke_acc_[s] += mv2;
        if (avg_len_ == 1) {
            sec_acc_[s] += mv2;
        }
        else {
            vsx_[i] += vx_[i]; vsy_[i] += vy_[i]; vsz_[i] += vz_[i];
        }
    }
    ke_samples_++;

    if (avg_len_ == 1) {
        sec_periods_++;
        return;
    }

    // Secular velocity = average over one rf period (Zhang Eq. 6)
    if (++avg_count_ == avg_len_) {
        const double inv = 1.0/double(avg_len_);
        for (std::size_t i = 0; i < n_; i++) {
            const double ux = vsx_[i]*inv, uy = vsy_[i]*inv, uz = vsz_[i]*inv;
            sec_acc_[species_of_[i]] += m_[i]*(ux*ux + uy*uy + uz*uz);
            vsx_[i] = vsy_[i] = vsz_[i] = 0;
        }
        avg_count_ = 0;
        sec_periods_++;
    }
}

void Simulation::write_diagnostics(DataWriter &w)
{
    // Instantaneous energies use the potential computed in this step.
    Energies E;
    const bool rf = p->micromotion_enabled;
    for (std::size_t i = 0; i < n_; i++) {
        const Vec3 r = position(i);
        E.kinetic += 0.5*m_[i]*velocity(i).norm2();
        E.trap += rf ? trap->rf_potential(r, q_[i], t_)
                     : trap->pseudo_potential(r, m_[i], q_[i]);
        E.coulomb += 0.5*q_[i]*phi_[i];
    }

    const std::size_t S = species_.size();
    std::vector<double> row;
    row.push_back(double(prod_done_.load()));
    row.push_back(t_);

    double sec_all = 0, ke_all = 0;
    for (std::size_t s = 0; s < S; s++) {
        sec_all += sec_acc_[s];
        ke_all += ke_acc_[s];
    }
    const double N = double(n_);
    row.push_back(sec_periods_ ? sec_all/(3*kB*N*sec_periods_) : NaN);
    for (std::size_t s = 0; s < S; s++) {
        const double Ns = double(species_[s].members.size());
        row.push_back(sec_periods_ ? sec_acc_[s]/(3*kB*Ns*sec_periods_) : NaN);
    }
    row.push_back(ke_samples_ ? ke_all/(3*kB*N*ke_samples_) : NaN);
    row.push_back(E.kinetic);
    row.push_back(E.trap);
    row.push_back(E.coulomb);
    row.push_back(E.total());
    w.write_diagnostics(row);

    std::fill(sec_acc_.begin(), sec_acc_.end(), 0.0);
    std::fill(ke_acc_.begin(), ke_acc_.end(), 0.0);
    sec_periods_ = 0;
    ke_samples_ = 0;
}

void Simulation::accumulate_ccd()
{
    const unsigned int B = p->ccd_bins;
    const double ext = p->ccd_extent;
    const double scale = B/(2*ext);
    for (std::size_t i = 0; i < n_; i++) {
        // Camera looks along y: horizontal axis = z (trap axis), vertical = x.
        const long ih = static_cast<long>(std::floor((z_[i] + ext)*scale));
        const long iv = static_cast<long>(std::floor((x_[i] + ext)*scale));
        if (ih < 0 || iv < 0 || ih >= long(B) || iv >= long(B)) continue;
        ccd_[(std::size_t(species_of_[i])*B + iv)*B + ih] += 1.0;
    }
}

std::vector<double> Simulation::centre_of_mass() const
{
    std::vector<double> com(3*species_.size(), 0.0);
    for (std::size_t s = 0; s < species_.size(); s++) {
        double M = 0;
        for (auto i : species_[s].members) {
            com[3*s] += m_[i]*x_[i];
            com[3*s + 1] += m_[i]*y_[i];
            com[3*s + 2] += m_[i]*z_[i];
            M += m_[i];
        }
        for (int k = 0; k < 3; k++) com[3*s + k] /= M;
    }
    return com;
}


// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

void Simulation::run()
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Simulation is already running");
    }
    if (worker_.joinable()) worker_.join();
    status_ = SimStatus::RUNNING;
    run_impl();
}

void Simulation::run_impl()
{
    summary_ = RunSummary();
    const double t_wall0 = wall_seconds();

    try {
        const SimParams &P = *p;
        if (!(P.dt > 0)) throw std::runtime_error("dt must be positive");

        initialize();

        DataWriter w(P.path, P.overwrite);
        w.write_text("config.json", config_to_json(P, *trap, lasers, ions));

        if (P.verbosity >= 1) {
            std::cout << "IonMD: " << n_ << " ions, " << species_.size() << " species\n"
                      << "  Coulomb backend: " << backend_name() << "\n"
                      << "  OpenMP threads: " << max_threads() << "\n";
            for (const auto &sp : species_) {
                const Vec3 w0 = trap->secular_frequencies(sp.m, sp.Z*constants::q_e);
                std::cout << "  species " << sp.label << ": N = " << sp.members.size()
                          << ", q = " << trap->mathieu_q(sp.m, sp.Z*constants::q_e)
                          << ", f_sec = (" << w0.x/(2*constants::pi) << ", "
                          << w0.y/(2*constants::pi) << ", "
                          << w0.z/(2*constants::pi) << ") Hz\n";
            }
            std::cout << std::flush;
        }

        // ---- Equilibration ("minimization") ----
        const double t_eq0 = wall_seconds();
        if (P.equilibrate_steps > 0) {
            if (P.verbosity >= 1) {
                std::cout << "Equilibrating for " << P.equilibrate_steps << " steps..." << std::endl;
            }
            equilibrate(P.equilibrate_steps);
        }
        if (P.initial_temperature > 0) {
            thermalize(P.initial_temperature);
        }
        summary_.equilibrate_time = wall_seconds() - t_eq0;

        w.write_state_csv("initial_state.csv", *this);

        // ---- Output setup ----
        const bool do_traj = P.record_every > 0;
        const bool do_diag = P.diag_every > 0;
        const bool do_com = P.com_every > 0;
        const bool do_ccd = P.ccd_enabled && P.ccd_bins > 0 && P.ccd_every > 0;

        avg_len_ = 1;
        if (P.micromotion_enabled) {
            avg_len_ = std::max<unsigned long>(1, std::lround(trap->rf_period()/P.dt));
        }
        vsx_.assign(n_, 0); vsy_.assign(n_, 0); vsz_.assign(n_, 0);
        sec_acc_.assign(species_.size(), 0);
        ke_acc_.assign(species_.size(), 0);
        sec_periods_ = ke_samples_ = avg_count_ = 0;
        if (do_ccd) ccd_.assign(species_.size()*P.ccd_bins*P.ccd_bins, 0.0);

        if (do_diag) w.open_diagnostics(species_);
        if (do_traj) {
            w.open_trajectory(P.record_velocities);
            if (P.record_start == 0) {
                w.write_frame(n_, x_.data(), y_.data(), z_.data());
                if (P.record_velocities) w.write_velocity_frame(n_, vx_.data(), vy_.data(), vz_.data());
            }
        }
        if (do_com) {
            w.open_com();
            w.write_com(centre_of_mass());
        }

        // ---- Production ----
        if (P.verbosity >= 1) {
            std::cout << "Running " << P.num_steps << " steps (dt = " << P.dt << " s)..." << std::endl;
        }
        const double t_prod0 = wall_seconds();
        bool ok = true;
        std::string abort_msg;

        for (unsigned long k = 1; k <= P.num_steps; k++) {
            const bool diag_now = do_diag && (k % P.diag_every == 0);
            ok = advance(Mode::PRODUCTION, diag_now);
            prod_done_ = k;

            if (!ok) {
                check_bounds(abort_msg);
                break;
            }

            if (do_diag) {
                accumulate_diagnostics();
                if (diag_now) write_diagnostics(w);
            }
            if (do_traj && k >= P.record_start && k % P.record_every == 0) {
                w.write_frame(n_, x_.data(), y_.data(), z_.data());
                if (P.record_velocities) w.write_velocity_frame(n_, vx_.data(), vy_.data(), vz_.data());
            }
            if (do_com && k % P.com_every == 0) {
                w.write_com(centre_of_mass());
            }
            if (do_ccd && k >= P.ccd_start && k % P.ccd_every == 0) {
                accumulate_ccd();
            }
            if (P.verbosity >= 1 && P.progress_every && k % P.progress_every == 0) {
                const double el = wall_seconds() - t_prod0;
                std::ostringstream os;
                os << "  step " << k << "/" << P.num_steps << std::fixed
                   << std::setprecision(1) << " (" << 100.0*k/P.num_steps << "%), "
                   << std::setprecision(2) << el << " s elapsed, "
                   << std::setprecision(1) << el/k*(P.num_steps - k) << " s left";
                std::cout << os.str() << std::endl;
            }
        }
        summary_.production_time = wall_seconds() - t_prod0;
        summary_.steps_done = prod_done_.load();

        // ---- Final output ----
        w.write_state_csv("final_state.csv", *this);
        w.write_xyz("final.xyz", *this);
        if (do_ccd) w.write_ccd(ccd_);

        summary_.wall_time = wall_seconds() - t_wall0;
        summary_.backend = backend_name();
        summary_.threads = max_threads();
        summary_.message = ok ? "completed" : abort_msg;

        using nlohmann::json;
        json info;
        info["n_ions"] = n_;
        info["n_species"] = species_.size();
        json sp = json::array();
        for (const auto &s : species_) {
            sp.push_back({{"label", s.label}, {"mass_amu", s.m/constants::amu},
                          {"charge", s.Z}, {"count", s.members.size()}});
        }
        info["species"] = sp;
        info["ion_species"] = species_of_;
        info["dt"] = P.dt;
        info["num_steps"] = P.num_steps;
        info["steps_done"] = summary_.steps_done;
        info["record_every"] = P.record_every;
        info["record_start"] = P.record_start;
        info["frames"] = w.frames();
        info["record_velocities"] = P.record_velocities;
        info["diag_every"] = P.diag_every;
        info["com_every"] = P.com_every;
        info["com_frames"] = w.com_frames();
        info["rf_average_steps"] = avg_len_;
        info["ccd"] = {{"enabled", do_ccd}, {"bins", P.ccd_bins},
                       {"extent", P.ccd_extent}, {"every", P.ccd_every},
                       {"start", P.ccd_start},
                       {"horizontal_axis", "z"}, {"vertical_axis", "x"}};
        info["status"] = ok ? "FINISHED" : "ABORTED";
        info["message"] = summary_.message;
        info["backend"] = summary_.backend;
        info["omp_threads"] = summary_.threads;
        info["wall_time"] = summary_.wall_time;
        info["equilibrate_time"] = summary_.equilibrate_time;
        info["production_time"] = summary_.production_time;
        info["steps_per_second"] = summary_.production_time > 0
            ? summary_.steps_done/summary_.production_time : 0.0;
        info["finished_at"] = timestamp_str();
        w.write_text("run_info.json", info.dump(2));

        if (P.verbosity >= 1) {
            std::cout << (ok ? "Finished" : "ABORTED: " + abort_msg) << "\n"
                      << "  wall time " << summary_.wall_time << " s ("
                      << info["steps_per_second"].get<double>() << " steps/s)\n"
                      << "  output written to " << w.path() << std::endl;
        }
        status_ = ok ? SimStatus::FINISHED : SimStatus::ABORTED;
    }
    catch (const std::exception &e) {
        summary_.message = e.what();
        summary_.wall_time = wall_seconds() - t_wall0;
        std::cerr << "IonMD error: " << e.what() << std::endl;
        status_ = SimStatus::ERRORED;
    }
}

void Simulation::start()
{
    if (status() == SimStatus::RUNNING) {
        throw std::logic_error("Simulation is already running");
    }
    if (worker_.joinable()) worker_.join();
    status_ = SimStatus::RUNNING;   // visible immediately to the caller
    worker_ = std::thread([this]() { this->run_impl(); });
}

void Simulation::wait()
{
    if (worker_.joinable()) worker_.join();
}

}  // namespace ionmd
