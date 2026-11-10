#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <ionmd/json.hpp>

#include <ionmd/config.hpp>
#include <ionmd/constants.hpp>
#include <ionmd/util.hpp>

namespace ionmd {

using nlohmann::json;
using constants::amu;
using constants::kB;
using constants::pi;
using constants::q_e;

namespace {

template <typename T>
void get(const json &j, const char *key, T &value)
{
    if (j.is_object() && j.count(key) && !j.at(key).is_null()) {
        value = j.at(key).get<T>();
    }
}

Vec3 to_vec3(const json &j, const char *what)
{
    if (!j.is_array() || j.size() != 3) {
        throw std::runtime_error(std::string(what) + " must be an array of 3 numbers");
    }
    return Vec3(j[0].get<double>(), j[1].get<double>(), j[2].get<double>());
}

void get_vec(const json &j, const char *key, Vec3 &value)
{
    if (j.is_object() && j.count(key) && !j.at(key).is_null()) {
        value = to_vec3(j.at(key), key);
    }
}

json vec_json(const Vec3 &v) { return json::array({v.x, v.y, v.z}); }

/// Load positions/velocities from a CSV written by IonMD (initial/final_state.csv).
void load_state_csv(const std::string &fname, std::vector<Ion> &ions)
{
    std::ifstream in(fname);
    if (!in) throw std::runtime_error("Could not open initial state file " + fname);
    std::string line;
    std::getline(in, line);  // header
    std::size_t i = 0;
    while (std::getline(in, line) && i < ions.size()) {
        if (line.empty()) continue;
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) cols.push_back(c);
        if (cols.size() < 12) throw std::runtime_error("Malformed line in " + fname);
        ions[i].x0 = Vec3(std::stod(cols[6]), std::stod(cols[7]), std::stod(cols[8]));
        ions[i].v0 = Vec3(std::stod(cols[9]), std::stod(cols[10]), std::stod(cols[11]));
        i++;
    }
    if (i != ions.size()) {
        throw std::runtime_error("Initial state file " + fname + " has " + std::to_string(i)
                                 + " ions but the configuration defines "
                                 + std::to_string(ions.size()));
    }
}

/// Fisher-Yates shuffle driven by the counter-based RNG (reproducible).
void shuffle_positions(std::vector<Ion> &ions, std::uint64_t seed)
{
    for (std::size_t i = ions.size(); i > 1; i--) {
        const std::size_t j = static_cast<std::size_t>(
            rng::uniform(seed, 0xC0FFEEull, i, 0)*double(i));
        std::swap(ions[i - 1].x0, ions[std::min(j, i - 1)].x0);
    }
}

void initial_conditions(const json &init, Config &cfg)
{
    std::string mode = "random";
    double r_ext = 20e-6, z_ext = 100e-6, T0 = 0.0;
    bool shuffle = true;
    get(init, "mode", mode);
    get(init, "radial_extent", r_ext);
    get(init, "axial_extent", z_ext);
    get(init, "temperature", T0);
    get(init, "shuffle", shuffle);

    auto &ions = cfg.ions;
    const std::size_t N = ions.size();
    const std::uint64_t seed = cfg.params.seed;

    if (init.is_object() && init.count("positions")) {
        const json &pos = init.at("positions");
        if (!pos.is_array() || pos.size() != N) {
            throw std::runtime_error("initial.positions must have one entry per ion");
        }
        for (std::size_t i = 0; i < N; i++) ions[i].x0 = to_vec3(pos[i], "position");
        if (init.count("velocities")) {
            const json &vel = init.at("velocities");
            if (!vel.is_array() || vel.size() != N) {
                throw std::runtime_error("initial.velocities must have one entry per ion");
            }
            for (std::size_t i = 0; i < N; i++) ions[i].v0 = to_vec3(vel[i], "velocity");
        }
    }
    else if (mode == "file") {
        std::string fname;
        get(init, "file", fname);
        load_state_csv(fname, ions);
    }
    else if (mode == "chain") {
        // Ions evenly spaced along z with a small random radial offset.
        for (std::size_t i = 0; i < N; i++) {
            const double zz = N > 1 ? -z_ext + 2*z_ext*double(i)/double(N - 1) : 0.0;
            const double xx = r_ext*(2*rng::uniform(seed, 0xA11CEull, i, 0) - 1);
            const double yy = r_ext*(2*rng::uniform(seed, 0xA11CEull, i, 1) - 1);
            ions[i].x0 = Vec3(xx, yy, zz);
        }
        if (shuffle) shuffle_positions(ions, seed);
    }
    else if (mode == "random") {
        // Uniform in a cylinder |r| < r_ext, |z| < z_ext.
        for (std::size_t i = 0; i < N; i++) {
            const double rr = r_ext*std::sqrt(rng::uniform(seed, 0xB0Bull, i, 0));
            const double ph = 2*pi*rng::uniform(seed, 0xB0Bull, i, 1);
            const double zz = z_ext*(2*rng::uniform(seed, 0xB0Bull, i, 2) - 1);
            ions[i].x0 = Vec3(rr*std::cos(ph), rr*std::sin(ph), zz);
        }
    }
    else {
        throw std::runtime_error("Unknown initial.mode '" + mode +
                                 "' (expected random, chain or file)");
    }

    if (T0 > 0) {
        for (std::size_t i = 0; i < N; i++) {
            const double s = std::sqrt(kB*T0/ions[i].m);
            ions[i].v0 = Vec3(s*rng::normal(seed, 0x5EEDull, i, 0),
                              s*rng::normal(seed, 0x5EEDull, i, 1),
                              s*rng::normal(seed, 0x5EEDull, i, 2));
        }
    }
}

}  // namespace


Config parse_config(const std::string &text)
{
    json j;
    try {
        j = json::parse(text);
    }
    catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid JSON configuration: ") + e.what());
    }

    Config cfg;
    SimParams &P = cfg.params;

    if (j.count("simulation")) {
        const json &s = j["simulation"];
        get(s, "dt", P.dt);
        get(s, "num_steps", P.num_steps);
        get(s, "equilibrate_steps", P.equilibrate_steps);
        get(s, "equilibrate_beta", P.equilibrate_beta);
        get(s, "initial_temperature", P.initial_temperature);
        get(s, "abort_radius", P.abort_radius);
        get(s, "seed", P.seed);
        get(s, "backend", P.backend);
        get(s, "num_threads", P.num_threads);
        get(s, "num_gpus", P.num_gpus);
        get(s, "gpu_min_ions", P.gpu_min_ions);
        get(s, "progress_every", P.progress_every);
        get(s, "verbosity", P.verbosity);
    }

    if (j.count("forces")) {
        const json &f = j["forces"];
        get(f, "secular", P.secular_enabled);
        get(f, "micromotion", P.micromotion_enabled);
        get(f, "coulomb", P.coulomb_enabled);
        get(f, "doppler", P.doppler_enabled);
        get(f, "stochastic", P.stochastic_enabled);
        get(f, "excitation", P.excitation_enabled);
    }

    if (j.count("stochastic")) {
        get(j["stochastic"], "heating_rate", P.heating_rate);
    }

    if (j.count("excitation")) {
        const json &e = j["excitation"];
        get(e, "amplitude", P.excitation.amplitude);
        get(e, "frequency", P.excitation.frequency);
        get_vec(e, "direction", P.excitation.direction);
        get(e, "t_start", P.excitation.t_start);
        get(e, "t_stop", P.excitation.t_stop);
    }

    if (j.count("trap")) {
        const json &t = j["trap"];
        Trap &T = cfg.trap;
        get(t, "r0", T.r0);
        get(t, "z0", T.z0);
        get(t, "kappa", T.kappa);
        if (t.count("f_rf")) T.omega_rf = 2*pi*t["f_rf"].get<double>();
        get(t, "omega_rf", T.omega_rf);
        get(t, "V_rf", T.V_rf);
        get(t, "U_dc", T.U_dc);
        get(t, "U_ec", T.U_ec);
        get(t, "rf_phase", T.rf_phase);
    }

    if (j.count("lasers")) {
        for (const auto &l : j["lasers"]) {
            Laser laser;
            Vec3 khat(0, 0, 1);
            get_vec(l, "khat", khat);
            laser.khat = khat.normalized();
            get(l, "beta", laser.beta);
            get(l, "F0", laser.F0);
            get(l, "directional", laser.directional);
            get(l, "detuning", laser.detuning);
            cfg.lasers.push_back(laser);
        }
    }

    if (j.count("ions")) {
        for (const auto &g : j["ions"]) {
            double mass = 40, charge = 1, heating = -1;
            unsigned long count = 1;
            bool lc = true;
            std::string label;
            get(g, "mass", mass);
            get(g, "charge", charge);
            get(g, "count", count);
            get(g, "laser_cooled", lc);
            get(g, "heating_rate", heating);
            get(g, "label", label);
            for (unsigned long k = 0; k < count; k++) {
                Ion ion(mass*amu, charge, Vec3(), lc, label);
                ion.heating_rate = heating;
                cfg.ions.push_back(ion);
            }
        }
    }

    if (j.count("output")) {
        const json &o = j["output"];
        get(o, "dir", P.path);
        get(o, "overwrite", P.overwrite);
        get(o, "record_every", P.record_every);
        get(o, "record_start", P.record_start);
        get(o, "record_velocities", P.record_velocities);
        get(o, "diag_every", P.diag_every);
        get(o, "com_every", P.com_every);
        if (o.count("ccd")) {
            const json &c = o["ccd"];
            get(c, "enabled", P.ccd_enabled);
            get(c, "bins", P.ccd_bins);
            get(c, "extent", P.ccd_extent);
            get(c, "every", P.ccd_every);
            get(c, "start", P.ccd_start);
        }
    }

    initial_conditions(j.count("initial") ? j["initial"] : json::object(), cfg);
    return cfg;
}


Config load_config(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Could not open configuration file " + filename);
    std::stringstream ss;
    ss << in.rdbuf();
    return parse_config(ss.str());
}


std::string config_to_json(const SimParams &P, const Trap &T,
                           const std::vector<Laser> &lasers,
                           const std::vector<Ion> &ions)
{
    json j;
    j["simulation"] = {
        {"dt", P.dt}, {"num_steps", P.num_steps},
        {"equilibrate_steps", P.equilibrate_steps},
        {"equilibrate_beta", P.equilibrate_beta},
        {"initial_temperature", P.initial_temperature},
        {"abort_radius", P.abort_radius}, {"seed", P.seed},
        {"backend", P.backend}, {"num_threads", P.num_threads},
        {"num_gpus", P.num_gpus}, {"gpu_min_ions", P.gpu_min_ions},
        {"progress_every", P.progress_every}, {"verbosity", P.verbosity}};
    j["forces"] = {
        {"secular", P.secular_enabled}, {"micromotion", P.micromotion_enabled},
        {"coulomb", P.coulomb_enabled}, {"doppler", P.doppler_enabled},
        {"stochastic", P.stochastic_enabled}, {"excitation", P.excitation_enabled}};
    j["stochastic"] = {{"heating_rate", P.heating_rate}};
    j["excitation"] = {
        {"amplitude", P.excitation.amplitude}, {"frequency", P.excitation.frequency},
        {"direction", vec_json(P.excitation.direction)},
        {"t_start", P.excitation.t_start}, {"t_stop", P.excitation.t_stop}};
    j["trap"] = {
        {"r0", T.r0}, {"z0", T.z0}, {"kappa", T.kappa},
        {"f_rf", T.omega_rf/(2*pi)}, {"V_rf", T.V_rf}, {"U_dc", T.U_dc},
        {"U_ec", T.U_ec}, {"rf_phase", T.rf_phase}};

    json lj = json::array();
    for (const auto &l : lasers) {
        lj.push_back({{"khat", vec_json(l.khat)}, {"beta", l.beta}, {"F0", l.F0},
                      {"directional", l.directional}, {"detuning", l.detuning}});
    }
    j["lasers"] = lj;

    // Compress the ion list into groups of consecutive identical ions.
    json groups = json::array();
    json derived = json::object();
    for (std::size_t i = 0; i < ions.size();) {
        std::size_t k = i + 1;
        while (k < ions.size() && ions[k].m == ions[i].m && ions[k].Z == ions[i].Z
               && ions[k].laser_cooled == ions[i].laser_cooled
               && ions[k].label == ions[i].label
               && ions[k].heating_rate == ions[i].heating_rate) {
            k++;
        }
        const Ion &ion = ions[i];
        groups.push_back({{"label", ion.label}, {"mass", ion.mass_amu()},
                          {"charge", ion.Z}, {"count", k - i},
                          {"laser_cooled", ion.laser_cooled},
                          {"heating_rate", ion.heating_rate}});

        const std::string key = ion.label.empty()
            ? ("m" + std::to_string(std::lround(ion.mass_amu())) + "_Z" + std::to_string(ion.Z))
            : ion.label;
        const double Q = ion.charge();
        const Vec3 w = T.secular_frequencies(ion.m, Q);
        derived[key] = {{"mathieu_q", T.mathieu_q(ion.m, Q)},
                        {"mathieu_a", T.mathieu_a(ion.m, Q)},
                        {"f_x", w.x/(2*pi)}, {"f_y", w.y/(2*pi)}, {"f_z", w.z/(2*pi)}};
        i = k;
    }
    j["ions"] = groups;
    j["derived"] = derived;
    j["initial"] = {{"mode", "file"}, {"file", "initial_state.csv"}};
    j["output"] = {
        {"dir", P.path}, {"overwrite", P.overwrite},
        {"record_every", P.record_every}, {"record_start", P.record_start},
        {"record_velocities", P.record_velocities}, {"diag_every", P.diag_every},
        {"com_every", P.com_every},
        {"ccd", {{"enabled", P.ccd_enabled}, {"bins", P.ccd_bins},
                 {"extent", P.ccd_extent}, {"every", P.ccd_every},
                 {"start", P.ccd_start}}}};
    return j.dump(2);
}

}  // namespace ionmd
