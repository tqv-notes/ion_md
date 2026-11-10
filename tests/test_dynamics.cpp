#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <ionmd/constants.hpp>
#include <ionmd/simulation.hpp>
#include <ionmd/util.hpp>

using namespace ionmd;
using constants::amu;
using constants::kB;
using constants::k_e;
using constants::pi;
using constants::q_e;

namespace {

SimParams quiet_params()
{
    SimParams p;
    p.verbosity = 0;
    p.backend = "cpu";
    p.record_every = 0;
    p.diag_every = 0;
    return p;
}

std::vector<Ion> random_ions(std::size_t n, double r = 20e-6, double zl = 60e-6,
                             std::uint64_t seed = 99)
{
    std::vector<Ion> ions;
    for (std::size_t i = 0; i < n; i++) {
        Vec3 x(r*(2*rng::uniform(seed, 1, i, 0) - 1), r*(2*rng::uniform(seed, 1, i, 1) - 1),
               zl*(2*rng::uniform(seed, 1, i, 2) - 1));
        ions.emplace_back(40*amu, 1.0, x);
    }
    return ions;
}

}  // namespace


TEST_CASE("Single ion oscillates at the axial secular frequency", "[dynamics]")
{
    SimParams p = quiet_params();
    Trap trap;
    const double m = 40*amu;
    const double wz = trap.secular_frequencies(m, q_e).z;
    const double Tz = 2*pi/wz;
    p.dt = Tz/400;

    Simulation sim(p, trap, {Ion(m, 1, Vec3(0, 0, 10e-6))});
    sim.initialize();
    const double E0 = sim.energies().total();
    const unsigned long steps = 400;   // exactly one period
    for (unsigned long k = 0; k < steps; k++) sim.step();

    REQUIRE(sim.position(0).z == Approx(10e-6).epsilon(1e-3));
    REQUIRE(std::abs(sim.velocity(0).z) < 1e-3*wz*10e-6);
    REQUIRE(sim.energies().total() == Approx(E0).epsilon(1e-4));
}


TEST_CASE("Two-ion equilibrium separation", "[dynamics]")
{
    SimParams p = quiet_params();
    Trap trap;
    const double m = 40*amu;
    const double wz = trap.secular_frequencies(m, q_e).z;
    p.dt = 2*pi/wz/200;

    Simulation sim(p, trap, {Ion(m, 1, Vec3(1e-6, 0, -30e-6)), Ion(m, 1, Vec3(0, -1e-6, 5e-6))});
    sim.equilibrate(40000);

    // m wz^2 d/2 = k_e e^2 / d^2  ->  d^3 = 2 k_e e^2 / (m wz^2)
    const double d = std::cbrt(2*k_e*q_e*q_e/(m*wz*wz));
    const double dz = sim.position(1).z - sim.position(0).z;
    REQUIRE(std::abs(dz) == Approx(d).epsilon(1e-4));
    REQUIRE(std::abs(sim.position(0).x) < 1e-9);
}


TEST_CASE("Five-ion chain matches James' equilibrium positions", "[dynamics]")
{
    SimParams p = quiet_params();
    Trap trap;
    const double m = 40*amu;
    const double wz = trap.secular_frequencies(m, q_e).z;
    p.dt = 2*pi/wz/200;

    std::vector<Ion> ions;
    for (int i = 0; i < 5; i++) ions.emplace_back(m, 1, Vec3(0.3e-6*i, -0.2e-6*i, -40e-6 + 20e-6*i));
    Simulation sim(p, trap, ions);
    sim.equilibrate(60000);

    // D. F. V. James, Appl. Phys. B 66, 181 (1998): u = 0, +-0.8221, +-1.7429
    const double l = std::cbrt(k_e*q_e*q_e/(m*wz*wz));
    std::vector<double> z;
    for (int i = 0; i < 5; i++) z.push_back(sim.position(i).z/l);
    std::sort(z.begin(), z.end());
    const double expect[5] = {-1.7429, -0.8221, 0.0, 0.8221, 1.7429};
    for (int i = 0; i < 5; i++) {
        REQUIRE(z[i] == Approx(expect[i]).margin(2e-4));
    }
}


TEST_CASE("Energy conservation without dissipation", "[dynamics]")
{
    SimParams p = quiet_params();
    Trap trap;
    p.dt = 5e-9;
    Simulation sim(p, trap, random_ions(40));
    sim.initialize();
    // give the ions some kinetic energy
    for (std::size_t i = 0; i < sim.num_ions(); i++) {
        sim.set_velocity(i, Vec3(rng::normal(5, 0, i, 0), rng::normal(5, 0, i, 1),
                                 rng::normal(5, 0, i, 2)));
    }
    const Energies E0 = sim.energies();
    for (int k = 0; k < 20000; k++) sim.step();
    const Energies E1 = sim.energies();
    // relative to the total energy
    REQUIRE(E1.total() == Approx(E0.total()).epsilon(2e-5));
    // and the scale of the dynamics (kinetic energy)
    REQUIRE(std::abs(E1.total() - E0.total()) < 1e-2*E0.kinetic);
}


TEST_CASE("Laser damping cools at the rate beta/m", "[dynamics]")
{
    SimParams p = quiet_params();
    p.doppler_enabled = true;
    p.coulomb_enabled = false;
    Trap trap;
    const double m = 40*amu;
    const double beta = 2e-22;
    p.dt = 10e-9;

    Simulation sim(p, trap, {Ion(m, 1, Vec3(5e-6, 3e-6, 20e-6))});
    sim.add_laser(Laser(beta, 0.0, Vec3(0, 0, 1)));
    sim.initialize();
    const double E0 = sim.energies().total();
    const double t = 1e-3;
    const auto steps = static_cast<unsigned long>(t/p.dt);
    for (unsigned long k = 0; k < steps; k++) sim.step();
    // For weak isotropic damping of a harmonic oscillator, E(t) = E0 exp(-beta t / m)
    REQUIRE(sim.energies().total()/E0 == Approx(std::exp(-beta*t/m)).epsilon(0.02));
}


TEST_CASE("Velocity-kick heating rate (Zhang Eq. 12)", "[dynamics]")
{
    SimParams p = quiet_params();
    p.stochastic_enabled = true;
    p.heating_rate = 200.0;   // K/s per ion
    p.dt = 20e-9;
    Trap trap;
    const std::size_t N = 50;
    Simulation sim(p, trap, random_ions(N));
    sim.equilibrate(5000);   // crystallise (velocities reset afterwards)
    const double E0 = sim.energies().total();
    const unsigned long steps = 50000;
    for (unsigned long k = 0; k < steps; k++) sim.step();
    const double dE = sim.energies().total() - E0;
    const double expected = N*kB*p.heating_rate*steps*p.dt;
    REQUIRE(dE == Approx(expected).epsilon(0.1));
}


TEST_CASE("Results are independent of the number of OpenMP threads", "[dynamics][openmp]")
{
    auto run = [](int threads) {
        SimParams p = quiet_params();
        p.num_threads = threads;
        p.stochastic_enabled = true;
        p.heating_rate = 50;
        p.doppler_enabled = true;
        Trap trap;
        // N above the OpenMP thresholds so that every loop runs in parallel
        Simulation sim(p, trap, random_ions(2500, 100e-6, 300e-6));
        sim.add_laser(Laser(2e-22, 1e-20, Vec3(0, 0, 1)));
        sim.initialize();
        for (int k = 0; k < 30; k++) sim.step();
        std::vector<double> out;
        for (std::size_t i = 0; i < sim.num_ions(); i++) {
            out.push_back(sim.position(i).x);
            out.push_back(sim.velocity(i).z);
        }
        return out;
    };
    const auto a = run(1);
    const auto b = run(4);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(a[i] == b[i]);   // bitwise identical
    }
}


TEST_CASE("Micromotion: rf-averaged radial frequency", "[dynamics][micromotion]")
{
    SimParams p = quiet_params();
    p.micromotion_enabled = true;
    Trap trap;
    const double m = 40*amu;
    const int J = 100;   // steps per rf period
    p.dt = trap.rf_period()/J;

    Simulation sim(p, trap, {Ion(m, 1, Vec3(2e-6, 0, 0))});
    sim.initialize();

    // rf-averaged x, sampled once per rf period
    const int periods = 3000;
    std::vector<double> xs;
    for (int k = 0; k < periods; k++) {
        double acc = 0;
        for (int j = 0; j < J; j++) {
            sim.step();
            acc += sim.position(0).x;
        }
        xs.push_back(acc/J);
    }
    // count sign changes -> frequency
    int crossings = 0;
    int first = -1, last = -1;
    for (int k = 1; k < periods; k++) {
        if ((xs[k - 1] < 0) != (xs[k] < 0)) {
            if (first < 0) first = k;
            last = k;
            crossings++;
        }
    }
    REQUIRE(crossings > 10);
    const double f_meas = 0.5*(crossings - 1)/((last - first)*trap.rf_period());
    const double f_theory = trap.secular_frequencies(m, q_e).x/(2*pi);
    // adiabatic approximation is accurate to O(q^2) ~ 1-2 %
    REQUIRE(f_meas == Approx(f_theory).epsilon(0.03));
}


TEST_CASE("Full run writes the expected output files", "[output]")
{
    namespace fs = std::filesystem;
    const std::string dir = "test_output_run";
    std::error_code ec;
    fs::remove_all(dir, ec);

    SimParams p = quiet_params();
    p.path = dir;
    p.num_steps = 1000;
    p.record_every = 100;
    p.record_velocities = true;
    p.diag_every = 50;
    p.com_every = 10;
    p.ccd_enabled = true;
    p.ccd_bins = 32;
    p.ccd_every = 5;
    p.equilibrate_steps = 100;
    Trap trap;

    std::vector<Ion> ions = random_ions(10);
    for (std::size_t i = 7; i < 10; i++) {
        ions[i].m = 28*amu;
        ions[i].label = "N2";
        ions[i].laser_cooled = false;
    }
    Simulation sim(p, trap, ions);
    sim.run();
    REQUIRE(sim.status() == SimStatus::FINISHED);

    for (const char *f : {"config.json", "run_info.json", "initial_state.csv",
                          "final_state.csv", "final.xyz", "diagnostics.csv",
                          "trajectory.bin", "velocities.bin", "com.bin", "ccd.bin"}) {
        INFO(f);
        REQUIRE(fs::exists(fs::path(dir)/f));
    }
    REQUIRE(sim.species().size() == 2);
    // frames 0, 100, ..., 1000
    REQUIRE(fs::file_size(fs::path(dir)/"trajectory.bin") == 11*10*3*sizeof(float));
    REQUIRE(fs::file_size(fs::path(dir)/"com.bin") == 101*2*3*sizeof(double));
    REQUIRE(fs::file_size(fs::path(dir)/"ccd.bin") == 2*32*32*sizeof(double));

    std::ifstream diag((fs::path(dir)/"diagnostics.csv").string());
    std::string line;
    int lines = 0;
    while (std::getline(diag, line)) lines++;
    REQUIRE(lines == 1 + 20);
    diag.close();

    fs::remove_all(dir, ec);   // may fail harmlessly on Windows (virus scanner)
}


TEST_CASE("Unstable trap aborts cleanly", "[dynamics]")
{
    SimParams p = quiet_params();
    p.micromotion_enabled = true;
    p.num_steps = 200000;
    p.path = "test_output_abort";
    p.record_every = 0;
    p.diag_every = 0;
    Trap trap;
    trap.V_rf = 3000;   // q ~ 1.2 > 0.908: unstable
    p.dt = trap.rf_period()/100;
    Simulation sim(p, trap, {Ion(40*amu, 1, Vec3(1e-6, 1e-6, 0))});
    sim.run();
    REQUIRE(sim.status() == SimStatus::ABORTED);
    std::error_code ec;
    std::filesystem::remove_all("test_output_abort", ec);
}
