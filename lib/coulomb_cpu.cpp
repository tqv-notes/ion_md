#include <cmath>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ionmd/constants.hpp>
#include <ionmd/coulomb.hpp>

namespace ionmd {

namespace {

/**
 * Direct O(N^2) summation parallelised over target ions. Each thread owns a
 * disjoint set of targets, so no synchronisation is required and the result
 * is independent of the number of threads (bitwise reproducible).
 */
class CpuCoulombSolver : public CoulombSolver
{
public:
    void compute(std::size_t n,
                 const double *x, const double *y, const double *z,
                 const double *q,
                 double *fx, double *fy, double *fz,
                 double *phi) override
    {
        const long nn = static_cast<long>(n);
        const bool want_phi = (phi != nullptr);

        // Parallelise only when there is enough work to amortise fork/join.
        #pragma omp parallel for schedule(static) if(nn >= 192)
        for (long i = 0; i < nn; i++) {
            const double xi = x[i], yi = y[i], zi = z[i];
            double sx = 0, sy = 0, sz = 0, sp = 0;

            #pragma omp simd reduction(+:sx,sy,sz,sp)
            for (long j = 0; j < nn; j++) {
                const double dx = xi - x[j];
                const double dy = yi - y[j];
                const double dz = zi - z[j];
                const double r2 = dx*dx + dy*dy + dz*dz;
                // Self interaction (j == i) is masked out.
                const double inv_r = (j == i) ? 0.0 : 1.0/std::sqrt(r2);
                const double qj_inv_r = q[j]*inv_r;
                const double s = qj_inv_r*inv_r*inv_r;
                sx += s*dx;
                sy += s*dy;
                sz += s*dz;
                sp += qj_inv_r;
            }

            const double pref = constants::k_e*q[i];
            fx[i] = pref*sx;
            fy[i] = pref*sy;
            fz[i] = pref*sz;
            if (want_phi) {
                phi[i] = constants::k_e*sp;
            }
        }
    }

    std::string name() const override
    {
#ifdef _OPENMP
        return "cpu (OpenMP, " + std::to_string(omp_get_max_threads()) + " threads)";
#else
        return "cpu (serial, no OpenMP)";
#endif
    }
};

}  // namespace


std::unique_ptr<CoulombSolver> make_cpu_coulomb_solver()
{
    return std::unique_ptr<CoulombSolver>(new CpuCoulombSolver());
}


#ifndef IONMD_HAVE_CUDA
bool cuda_compiled() { return false; }
int cuda_device_count() { return 0; }
std::string cuda_device_info() { return "CUDA support not compiled in"; }

std::unique_ptr<CoulombSolver> make_cuda_coulomb_solver(int)
{
    throw std::runtime_error(
        "IonMD was built without CUDA support; reconfigure with -DIONMD_USE_CUDA=ON");
}
#endif


std::unique_ptr<CoulombSolver> make_coulomb_solver(const std::string &backend,
                                                   std::size_t n_ions,
                                                   int num_gpus,
                                                   std::size_t gpu_min_ions)
{
    if (backend == "cpu") {
        return make_cpu_coulomb_solver();
    }
    if (backend == "cuda" || backend == "gpu") {
        return make_cuda_coulomb_solver(num_gpus);
    }
    if (backend == "auto") {
        if (cuda_compiled() && cuda_device_count() > 0 && n_ions >= gpu_min_ions) {
            return make_cuda_coulomb_solver(num_gpus);
        }
        return make_cpu_coulomb_solver();
    }
    throw std::runtime_error("Unknown Coulomb backend '" + backend +
                             "' (expected cpu, cuda or auto)");
}

}  // namespace ionmd
