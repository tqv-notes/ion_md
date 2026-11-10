/**
 * ionmd: command line driver.
 *
 *   ionmd config.json [options]
 *
 * Options override the corresponding entries of the configuration file:
 *   -o, --output DIR      output directory
 *   -b, --backend NAME    Coulomb backend: cpu | cuda | auto
 *   -t, --threads N       OpenMP threads (0 = default / OMP_NUM_THREADS)
 *   -g, --gpus N          number of GPUs for the cuda backend (0 = all)
 *   -n, --steps N         number of production steps
 *   -N, --ions N          rescale the number of ions to N (keeps species
 *                         fractions; positions are re-randomised); useful
 *                         for benchmarks
 *   -q, --quiet           less output
 *       --info            print build/device information and exit
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ionmd/config.hpp>
#include <ionmd/coulomb.hpp>
#include <ionmd/simulation.hpp>
#include <ionmd/util.hpp>

using namespace ionmd;

namespace {

void usage()
{
    std::cout <<
        "Usage: ionmd CONFIG.json [options]\n"
        "  -o, --output DIR     output directory\n"
        "  -b, --backend NAME   cpu | cuda | auto\n"
        "  -t, --threads N      OpenMP threads (0 = OMP_NUM_THREADS/default)\n"
        "  -g, --gpus N         GPUs for the cuda backend (0 = all visible)\n"
        "  -n, --steps N        production steps\n"
        "  -N, --ions N         rescale total ion number (benchmarking)\n"
        "  -q, --quiet          less output\n"
        "      --info           print build and device information\n";
}

void print_info()
{
    std::cout << "IonMD build information\n";
#ifdef _OPENMP
    std::cout << "  OpenMP: yes (" << _OPENMP << "), max threads " << omp_get_max_threads() << "\n";
#else
    std::cout << "  OpenMP: no\n";
#endif
    std::cout << "  CUDA: " << (cuda_compiled() ? "yes" : "no") << "\n";
    if (cuda_compiled()) {
        std::cout << "  " << cuda_device_info() << "\n";
    }
}

/// Rescale the ion list to N ions keeping the species proportions.
void rescale_ions(Config &cfg, std::size_t N)
{
    const auto old = cfg.ions;
    if (old.empty()) return;
    cfg.ions.clear();
    for (std::size_t i = 0; i < N; i++) {
        Ion ion = old[(i*old.size())/N];
        cfg.ions.push_back(ion);
    }
    // Random positions in a volume scaled like a crystal of N ions.
    const double scale = std::cbrt(double(N)/double(old.size()));
    double rmax = 0, zmax = 0;
    for (const auto &ion : old) {
        rmax = std::max(rmax, std::hypot(ion.x0.x, ion.x0.y));
        zmax = std::max(zmax, std::abs(ion.x0.z));
    }
    rmax = std::max(rmax, 10e-6)*scale;
    zmax = std::max(zmax, 10e-6)*scale;
    for (std::size_t i = 0; i < N; i++) {
        const double r = rmax*std::sqrt(rng::uniform(cfg.params.seed, 0xBE4Cull, i, 0));
        const double ph = 2*constants::pi*rng::uniform(cfg.params.seed, 0xBE4Cull, i, 1);
        const double z = zmax*(2*rng::uniform(cfg.params.seed, 0xBE4Cull, i, 2) - 1);
        cfg.ions[i].x0 = Vec3(r*std::cos(ph), r*std::sin(ph), z);
        cfg.ions[i].v0 = Vec3();
    }
}

}  // namespace


int main(int argc, char *argv[])
{
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string config_file;
    std::string output, backend;
    long threads = -1, gpus = -1, steps = -1, n_ions = -1;
    bool quiet = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << what << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--info") { print_info(); return 0; }
        else if (a == "-o" || a == "--output") output = next("--output");
        else if (a == "-b" || a == "--backend") backend = next("--backend");
        else if (a == "-t" || a == "--threads") threads = std::stol(next("--threads"));
        else if (a == "-g" || a == "--gpus") gpus = std::stol(next("--gpus"));
        else if (a == "-n" || a == "--steps") steps = std::stol(next("--steps"));
        else if (a == "-N" || a == "--ions") n_ions = std::stol(next("--ions"));
        else if (a == "-q" || a == "--quiet") quiet = true;
        else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option " << a << "\n";
            usage();
            return 1;
        }
        else config_file = a;
    }

    Config cfg;
    try {
        cfg = load_config(config_file);
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (!output.empty()) cfg.params.path = output;
    if (!backend.empty()) cfg.params.backend = backend;
    if (threads >= 0) cfg.params.num_threads = static_cast<int>(threads);
    if (gpus >= 0) cfg.params.num_gpus = static_cast<int>(gpus);
    if (steps >= 0) cfg.params.num_steps = static_cast<unsigned long>(steps);
    if (n_ions > 0) rescale_ions(cfg, static_cast<std::size_t>(n_ions));
    if (quiet) cfg.params.verbosity = 0;

    Simulation sim(cfg.params, cfg.trap, cfg.ions);
    sim.set_lasers(cfg.lasers);
    sim.run();

    switch (sim.status()) {
    case SimStatus::FINISHED: return 0;
    case SimStatus::ABORTED:
        std::cerr << "Simulation aborted: " << sim.summary().message << std::endl;
        return 2;
    default:
        std::cerr << "Simulation failed: " << sim.summary().message << std::endl;
        return 3;
    }
}
