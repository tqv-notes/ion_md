#ifndef IONMD_CONFIG_HPP
#define IONMD_CONFIG_HPP

#include <string>
#include <vector>
#include <ionmd/ion.hpp>
#include <ionmd/laser.hpp>
#include <ionmd/params.hpp>
#include <ionmd/trap.hpp>

namespace ionmd {

/**
 * Complete description of a simulation, as read from a JSON file.
 *
 * JSON layout (all keys optional, SI units unless noted):
 *
 * {
 *   "simulation": { "dt": 2e-8, "num_steps": 20000, "equilibrate_steps": 0,
 *                   "equilibrate_beta": 1e-20, "initial_temperature": 0,
 *                   "seed": 12345, "abort_radius": 1e-3,
 *                   "backend": "auto", "num_threads": 0, "num_gpus": 0,
 *                   "gpu_min_ions": 256, "progress_every": 0, "verbosity": 1 },
 *   "forces":     { "secular": true, "micromotion": false, "coulomb": true,
 *                   "doppler": false, "stochastic": false, "excitation": false },
 *   "trap":       { "r0": 3.5e-3, "z0": 2.7e-3, "kappa": 0.248,
 *                   "f_rf": 4.972e6, "V_rf": 300, "U_dc": 0, "U_ec": 1,
 *                   "rf_phase": 0 },
 *   "lasers":     [ { "khat": [0,0,1], "beta": 2e-22, "F0": 0,
 *                     "directional": false } ],
 *   "stochastic": { "heating_rate": 0.0 },                  // K/s per ion
 *   "excitation": { "amplitude": 0, "frequency": 0, "direction": [1,0,0],
 *                   "t_start": 0, "t_stop": 1e300 },
 *   "ions":       [ { "label": "Ca40", "mass": 40, "charge": 1, "count": 10,
 *                     "laser_cooled": true, "heating_rate": -1 } ],  // mass in amu
 *   "initial":    { "mode": "random" | "chain" | "file",
 *                   "radial_extent": 20e-6, "axial_extent": 100e-6,
 *                   "temperature": 0.0, "file": "state.csv",
 *                   "positions": [[x,y,z], ...] },
 *   "output":     { "dir": "output", "overwrite": true, "record_every": 100,
 *                   "record_start": 0, "record_velocities": false,
 *                   "diag_every": 100, "com_every": 0,
 *                   "ccd": { "enabled": false, "bins": 256, "extent": 3e-4,
 *                            "every": 10, "start": 0 } }
 * }
 */
struct Config
{
    SimParams params;
    Trap trap;
    std::vector<Laser> lasers;
    std::vector<Ion> ions;
};

/// Parse a configuration from a JSON string. Throws on error.
Config parse_config(const std::string &json_text);

/// Load a configuration from a JSON file. Throws on error.
Config load_config(const std::string &filename);

/// Serialise a configuration (including derived trap frequencies) to JSON.
std::string config_to_json(const SimParams &p, const Trap &trap,
                           const std::vector<Laser> &lasers,
                           const std::vector<Ion> &ions);

}  // namespace ionmd

#endif
