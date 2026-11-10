#ifndef IONMD_COULOMB_HPP
#define IONMD_COULOMB_HPP

/**
 * Pairwise Coulomb interaction backends.
 *
 * The O(N^2) Coulomb sum dominates the cost of the simulation (Zhang et al.,
 * PRA 76, 012719). It is therefore implemented behind a small interface with
 * an OpenMP CPU implementation and a (multi-)GPU CUDA implementation.
 *
 * For ion i at r_i with charge q_i:
 *
 *     F_i   = k_e q_i  sum_{j != i} q_j (r_i - r_j) / |r_i - r_j|^3
 *     phi_i = k_e      sum_{j != i} q_j / |r_i - r_j|
 *
 * so that the total Coulomb energy is U = 1/2 sum_i q_i phi_i.
 * This header deliberately has no dependencies so it can be included from
 * CUDA translation units.
 */

#include <cstddef>
#include <memory>
#include <string>

namespace ionmd {

class CoulombSolver
{
public:
    virtual ~CoulombSolver() = default;

    /**
     * Compute Coulomb forces (and optionally the electrostatic potential).
     *
     * @param n   number of ions
     * @param x,y,z positions [m] (length n)
     * @param q   charges [C] (length n)
     * @param fx,fy,fz output forces [N] (length n)
     * @param phi output potential [V] at each ion, may be nullptr
     */
    virtual void compute(std::size_t n,
                         const double *x, const double *y, const double *z,
                         const double *q,
                         double *fx, double *fy, double *fz,
                         double *phi) = 0;

    /// Human readable description, e.g. "cpu (8 threads)".
    virtual std::string name() const = 0;
};

/// OpenMP CPU implementation (always available).
std::unique_ptr<CoulombSolver> make_cpu_coulomb_solver();

/// True if the library was compiled with CUDA support.
bool cuda_compiled();

/// Number of usable CUDA devices (0 if not compiled with CUDA).
int cuda_device_count();

/// Description of the visible CUDA devices.
std::string cuda_device_info();

/**
 * CUDA implementation. The target ions are split across `num_gpus` devices
 * (0 = all visible devices). Throws std::runtime_error if CUDA is unavailable.
 */
std::unique_ptr<CoulombSolver> make_cuda_coulomb_solver(int num_gpus);

/**
 * Factory.
 *
 * @param backend "cpu", "cuda" or "auto"
 * @param n_ions number of ions (used by "auto")
 * @param num_gpus number of GPUs for the CUDA backend (0 = all)
 * @param gpu_min_ions with "auto" use the GPU only if n_ions >= gpu_min_ions
 */
std::unique_ptr<CoulombSolver> make_coulomb_solver(const std::string &backend,
                                                   std::size_t n_ions,
                                                   int num_gpus,
                                                   std::size_t gpu_min_ions);

}  // namespace ionmd

#endif
