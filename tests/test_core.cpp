#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <ionmd/config.hpp>
#include <ionmd/constants.hpp>
#include <ionmd/coulomb.hpp>
#include <ionmd/laser.hpp>
#include <ionmd/trap.hpp>
#include <ionmd/util.hpp>
#include <ionmd/vec3.hpp>

using namespace ionmd;
using constants::amu;
using constants::q_e;
using constants::pi;
using constants::k_e;


TEST_CASE("Vec3 arithmetic", "[vec3]")
{
    Vec3 a(1, 2, 3), b(-2, 0.5, 4);
    REQUIRE(dot(a, b) == Approx(-2 + 1 + 12));
    REQUIRE((a + b).x == Approx(-1));
    REQUIRE((a - b).z == Approx(-1));
    REQUIRE((2*a).y == Approx(4));
    REQUIRE(a.norm() == Approx(std::sqrt(14.0)));
    REQUIRE(a.normalized().norm() == Approx(1.0));
    const Vec3 c = cross(Vec3(1, 0, 0), Vec3(0, 1, 0));
    REQUIRE(c.z == Approx(1.0));
    REQUIRE(Vec3().normalized().norm() == 0.0);
}


TEST_CASE("Trap stability parameters and secular frequencies", "[trap]")
{
    Trap trap;
    const double m = 40*amu, Q = q_e;
    const double q = trap.mathieu_q(m, Q);
    // q = 2 Q V / (m r0^2 Omega^2)
    REQUIRE(q == Approx(2*Q*trap.V_rf/(m*trap.r0*trap.r0*trap.omega_rf*trap.omega_rf)));
    REQUIRE(q > 0.0);
    REQUIRE(q < 0.908);

    const Vec3 w = trap.secular_frequencies(m, Q);
    const double wz2 = 2*Q*trap.kappa*trap.U_ec/(m*trap.z0*trap.z0);
    REQUIRE(w.z == Approx(std::sqrt(wz2)));
    // Adiabatic approximation: omega_r^2 = (q Omega)^2/8 - omega_z^2/2
    const double wr2 = q*q*trap.omega_rf*trap.omega_rf/8 - wz2/2;
    REQUIRE(w.x == Approx(std::sqrt(wr2)));
    REQUIRE(w.y == Approx(std::sqrt(wr2)));

    // dc offset splits the radial frequencies
    trap.U_dc = 1.0;
    const Vec3 w2 = trap.secular_frequencies(m, Q);
    REQUIRE(w2.x > w2.y);
}


TEST_CASE("Trap forces are gradients of the potentials", "[trap]")
{
    Trap trap;
    trap.U_dc = 0.3;
    const double m = 40*amu, Q = q_e;
    const Vec3 r(3e-6, -7e-6, 11e-6);
    const double h = 1e-9;
    const double t = 0.123e-6;

    for (int k = 0; k < 3; k++) {
        Vec3 rp = r, rm = r;
        rp[k] += h;
        rm[k] -= h;
        const double g_ps = (trap.pseudo_potential(rp, m, Q) - trap.pseudo_potential(rm, m, Q))/(2*h);
        REQUIRE(trap.pseudo_force(r, m, Q)[k] == Approx(-g_ps).epsilon(1e-6));
        const double g_rf = (trap.rf_potential(rp, Q, t) - trap.rf_potential(rm, Q, t))/(2*h);
        REQUIRE(trap.rf_force(r, Q, t)[k] == Approx(-g_rf).epsilon(1e-6));
    }
}


TEST_CASE("Laser damping model", "[laser]")
{
    Laser iso(1e-21, 2e-20, Vec3(0, 0, 2));
    REQUIRE(iso.khat.norm() == Approx(1.0));
    const Vec3 v(1, 2, 3);
    const Vec3 F = iso.force(v);
    REQUIRE(F.x == Approx(-1e-21));
    REQUIRE(F.z == Approx(2e-20 - 3e-21));

    Laser dir(1e-21, 0.0, Vec3(0, 0, 1), true);
    const Vec3 Fd = dir.force(v);
    REQUIRE(Fd.x == 0.0);
    REQUIRE(Fd.z == Approx(-3e-21));
}


TEST_CASE("Counter based RNG", "[rng]")
{
    // deterministic
    REQUIRE(rng::uniform(1, 2, 3, 4) == rng::uniform(1, 2, 3, 4));
    REQUIRE(rng::uniform(1, 2, 3, 4) != rng::uniform(1, 2, 3, 5));

    // unit vectors are isotropic: <w> ~ 0, <w_k^2> ~ 1/3
    const int n = 200000;
    Vec3 mean, sq;
    double mean_u = 0, var_n = 0, max_norm_err = 0;
    for (int i = 0; i < n; i++) {
        const Vec3 w = rng::unit_vector(42, 7, i);
        max_norm_err = std::max(max_norm_err, std::abs(w.norm() - 1.0));
        mean += w;
        sq += Vec3(w.x*w.x, w.y*w.y, w.z*w.z);
        mean_u += rng::uniform(42, 1, i, 0);
        const double g = rng::normal(42, 2, i, 0);
        var_n += g*g;
    }
    mean /= n; sq /= n;
    REQUIRE(max_norm_err < 1e-12);
    REQUIRE(std::abs(mean.x) < 0.01);
    REQUIRE(std::abs(mean.z) < 0.01);
    REQUIRE(sq.x == Approx(1.0/3).epsilon(0.02));
    REQUIRE(sq.z == Approx(1.0/3).epsilon(0.02));
    REQUIRE(mean_u/n == Approx(0.5).epsilon(0.01));
    REQUIRE(var_n/n == Approx(1.0).epsilon(0.02));
}


namespace {

struct Cloud
{
    std::vector<double> x, y, z, q;
    explicit Cloud(std::size_t n, std::uint64_t seed = 1) : x(n), y(n), z(n), q(n)
    {
        for (std::size_t i = 0; i < n; i++) {
            x[i] = 50e-6*(2*rng::uniform(seed, 0, i, 0) - 1);
            y[i] = 50e-6*(2*rng::uniform(seed, 0, i, 1) - 1);
            z[i] = 200e-6*(2*rng::uniform(seed, 0, i, 2) - 1);
            q[i] = q_e*(1 + (i % 3 == 0));   // mix of charges 1 and 2
        }
    }
};

}  // namespace


TEST_CASE("CPU Coulomb solver: two ions", "[coulomb]")
{
    auto solver = make_cpu_coulomb_solver();
    std::vector<double> x = {0, 0}, y = {0, 0}, z = {-5e-6, 5e-6}, q = {q_e, 2*q_e};
    std::vector<double> fx(2), fy(2), fz(2), phi(2);
    solver->compute(2, x.data(), y.data(), z.data(), q.data(),
                    fx.data(), fy.data(), fz.data(), phi.data());
    const double d = 10e-6;
    const double F = k_e*q_e*2*q_e/(d*d);
    REQUIRE(fz[0] == Approx(-F));
    REQUIRE(fz[1] == Approx(F));
    REQUIRE(fx[0] == 0.0);
    REQUIRE(phi[0] == Approx(k_e*2*q_e/d));
    REQUIRE(phi[1] == Approx(k_e*q_e/d));
}


TEST_CASE("CPU Coulomb solver: Newton's third law and F = -grad U", "[coulomb]")
{
    const std::size_t n = 300;
    Cloud c(n);
    auto solver = make_cpu_coulomb_solver();
    std::vector<double> fx(n), fy(n), fz(n), phi(n);
    solver->compute(n, c.x.data(), c.y.data(), c.z.data(), c.q.data(),
                    fx.data(), fy.data(), fz.data(), phi.data());

    double sx = 0, sy = 0, sz = 0, fmax = 0;
    for (std::size_t i = 0; i < n; i++) {
        sx += fx[i]; sy += fy[i]; sz += fz[i];
        fmax = std::max(fmax, std::abs(fz[i]));
    }
    REQUIRE(std::abs(sx) < 1e-9*fmax*n);
    REQUIRE(std::abs(sy) < 1e-9*fmax*n);
    REQUIRE(std::abs(sz) < 1e-9*fmax*n);

    // Total energy U = 1/2 sum q_i phi_i; check dU/dx_k = -F_k
    auto energy = [&](const Cloud &cc) {
        std::vector<double> a(n), b(n), d(n), p(n);
        solver->compute(n, cc.x.data(), cc.y.data(), cc.z.data(), cc.q.data(),
                        a.data(), b.data(), d.data(), p.data());
        double U = 0;
        for (std::size_t i = 0; i < n; i++) U += 0.5*cc.q[i]*p[i];
        return U;
    };
    const std::size_t k = 17;
    const double h = 1e-10;
    Cloud cp = c, cm = c;
    cp.x[k] += h; cm.x[k] -= h;
    const double grad = (energy(cp) - energy(cm))/(2*h);
    REQUIRE(fx[k] == Approx(-grad).epsilon(1e-4));
}


TEST_CASE("CUDA Coulomb solver agrees with CPU", "[coulomb][cuda]")
{
    if (!cuda_compiled() || cuda_device_count() == 0) {
        WARN("CUDA not available - skipping GPU test");
        return;
    }
    INFO(cuda_device_info());

    // One solver per GPU count, reused for all sizes so that buffer growth is
    // exercised too. The sizes cover one tile, ragged source chunks, the
    // source-split path and (n >= 4096) two GPUs with uneven slices.
    auto gpu1 = make_cuda_coulomb_solver(1);
    auto gpu_all = make_cuda_coulomb_solver(0);
    for (std::size_t n : {3ul, 257ul, 1000ul, 5000ul, 12345ul, 777ul}) {
        Cloud c(n, 7);
        std::vector<double> fx(n), fy(n), fz(n), phi(n);
        std::vector<double> gx(n), gy(n), gz(n), gphi(n);
        make_cpu_coulomb_solver()->compute(n, c.x.data(), c.y.data(), c.z.data(), c.q.data(),
                                           fx.data(), fy.data(), fz.data(), phi.data());
        for (int gpus : {1, 0}) {
            auto &gpu = (gpus == 1) ? gpu1 : gpu_all;
            gpu->compute(n, c.x.data(), c.y.data(), c.z.data(), c.q.data(),
                         gx.data(), gy.data(), gz.data(), gphi.data());
            double max_err = 0, max_perr = 0;
            for (std::size_t i = 0; i < n; i++) {
                const double scale = std::abs(fx[i]) + std::abs(fy[i]) + std::abs(fz[i]);
                max_err = std::max(max_err, (std::abs(gx[i] - fx[i]) + std::abs(gy[i] - fy[i])
                                             + std::abs(gz[i] - fz[i]))/scale);
                max_perr = std::max(max_perr, std::abs(gphi[i] - phi[i])/std::abs(phi[i]));
            }
            INFO("n = " << n << ", gpus = " << gpus);
            REQUIRE(max_err < 1e-10);
            REQUIRE(max_perr < 1e-10);
        }
    }
}


TEST_CASE("Configuration parsing", "[config]")
{
    const char *text = R"({
        "simulation": {"dt": 1e-8, "num_steps": 10, "seed": 3},
        "forces": {"doppler": true, "micromotion": true},
        "trap": {"f_rf": 2e6, "V_rf": 150, "U_ec": 2},
        "lasers": [{"khat": [0, 0, 5], "beta": 1e-21, "F0": 1e-20}],
        "ions": [{"label": "Ca", "mass": 40, "count": 5},
                 {"label": "N2", "mass": 28, "count": 3, "laser_cooled": false}],
        "initial": {"mode": "chain", "axial_extent": 50e-6, "radial_extent": 0},
        "output": {"dir": "out_x", "ccd": {"enabled": true, "bins": 64}}
    })";
    Config cfg = parse_config(text);
    REQUIRE(cfg.params.dt == Approx(1e-8));
    REQUIRE(cfg.params.num_steps == 10);
    REQUIRE(cfg.params.doppler_enabled);
    REQUIRE(cfg.params.micromotion_enabled);
    REQUIRE(cfg.trap.omega_rf == Approx(2*pi*2e6));
    REQUIRE(cfg.lasers.size() == 1);
    REQUIRE(cfg.lasers[0].khat.z == Approx(1.0));
    REQUIRE(cfg.ions.size() == 8);
    REQUIRE(cfg.ions[7].laser_cooled == false);
    REQUIRE(cfg.ions[7].m == Approx(28*amu));
    REQUIRE(cfg.params.ccd_enabled);
    REQUIRE(cfg.params.ccd_bins == 64);
    double zmin = 1, zmax = -1;
    for (const auto &ion : cfg.ions) {
        zmin = std::min(zmin, ion.x0.z);
        zmax = std::max(zmax, ion.x0.z);
        REQUIRE(ion.x0.x == 0.0);
    }
    REQUIRE(zmin == Approx(-50e-6));
    REQUIRE(zmax == Approx(50e-6));

    // round trip
    const std::string js = config_to_json(cfg.params, cfg.trap, cfg.lasers, cfg.ions);
    REQUIRE(js.find("\"derived\"") != std::string::npos);

    REQUIRE_THROWS(parse_config("{ not json"));
    REQUIRE_THROWS(parse_config(R"({"initial": {"mode": "nonsense"}, "ions": [{"count": 1}]})"));
}
