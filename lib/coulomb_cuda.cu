/**
 * CUDA (multi-GPU) implementation of the O(N^2) Coulomb interaction.
 *
 * Algorithm: tiled all-pairs N-body kernel (GPU Gems 3, ch. 31) in double
 * precision. Every thread owns one target ion i and loops over source ions j,
 * which are staged through shared memory in tiles of BLOCK_SIZE ions.
 *
 * To keep all SMs busy also for moderate N, the source range is split into
 * `nsplit` chunks (grid.y): block (bx, by) sums the contributions of chunk by
 * to the targets of block bx and writes a partial result. A second kernel
 * adds the partial results in a fixed order, so the forces are deterministic
 * (no atomics). Without the split, N = 10000 would only launch 40 blocks on
 * the 80 SMs of a V100.
 *
 * Multiple GPUs: the target ions are split into contiguous slices, one per
 * GPU; every GPU receives the full set of positions, computes the forces on
 * its slice and copies them back. All transfers use pinned host memory and
 * one stream per device, so all GPUs run concurrently.
 */

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <ionmd/constants.hpp>
#include <ionmd/coulomb.hpp>

#define IONMD_CUDA_CHECK(call)                                                \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::ostringstream os_;                                           \
            os_ << "CUDA error '" << cudaGetErrorString(err_) << "' at "      \
                << __FILE__ << ":" << __LINE__ << " (" #call ")";             \
            throw std::runtime_error(os_.str());                              \
        }                                                                     \
    } while (0)

namespace ionmd {

namespace {

constexpr int BLOCK_SIZE = 128;

// Aim for this many blocks per SM in total (several waves of resident blocks).
constexpr int BLOCKS_PER_SM = 16;

// Use an additional GPU only if every GPU gets at least this many targets;
// below that the per-device launch/transfer overhead outweighs the gain.
constexpr std::size_t MIN_TARGETS_PER_DEVICE = 2048;

/**
 * pos[j] = (x, y, z, q). For targets i in [i_begin, i_end) and sources j in
 * chunk blockIdx.y, i.e. [blockIdx.y*j_chunk, min(n, (blockIdx.y+1)*j_chunk)):
 *   out[blockIdx.y*slice + i - i_begin] = (sum_j q_j dx/r^3, ..., sum_j q_j/r)
 * The prefactors k_e q_i and k_e are applied on the host.
 */
__global__ void coulomb_kernel(const double4 *__restrict__ pos, int n,
                               int i_begin, int i_end, int j_chunk,
                               double4 *__restrict__ out)
{
    __shared__ double4 tile[BLOCK_SIZE];

    const int slice = i_end - i_begin;
    const int i = i_begin + blockIdx.x*BLOCK_SIZE + threadIdx.x;
    const bool active = (i < i_end);
    const double4 pi = active ? pos[i] : make_double4(0.0, 0.0, 0.0, 0.0);

    const int j_begin = blockIdx.y*j_chunk;
    const int j_end = min(n, j_begin + j_chunk);

    double sx = 0.0, sy = 0.0, sz = 0.0, sp = 0.0;

    for (int base = j_begin; base < j_end; base += BLOCK_SIZE) {
        const int j_load = base + threadIdx.x;
        tile[threadIdx.x] = (j_load < j_end) ? pos[j_load]
                                             : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        const int lim = min(BLOCK_SIZE, j_end - base);
        #pragma unroll 8
        for (int k = 0; k < lim; k++) {
            const double4 pj = tile[k];
            const double dx = pi.x - pj.x;
            const double dy = pi.y - pj.y;
            const double dz = pi.z - pj.z;
            const double r2 = dx*dx + dy*dy + dz*dz;
            const double inv_r = (base + k == i) ? 0.0 : rsqrt(r2);
            const double qinv = pj.w*inv_r;
            const double s = qinv*inv_r*inv_r;
            sx += s*dx;
            sy += s*dy;
            sz += s*dz;
            sp += qinv;
        }
        __syncthreads();
    }

    if (active) {
        out[static_cast<size_t>(blockIdx.y)*slice + (i - i_begin)] =
            make_double4(sx, sy, sz, sp);
    }
}

/** out[k] = sum_c partial[c*slice + k], summed in a fixed order. */
__global__ void reduce_kernel(const double4 *__restrict__ partial, int slice,
                              int nsplit, double4 *__restrict__ out)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= slice) return;
    double4 s = partial[k];
    for (int c = 1; c < nsplit; c++) {
        const double4 p = partial[static_cast<size_t>(c)*slice + k];
        s.x += p.x;
        s.y += p.y;
        s.z += p.z;
        s.w += p.w;
    }
    out[k] = s;
}


struct DeviceContext
{
    int id = 0;
    std::string name;
    cudaStream_t stream = nullptr;
    int sm_count = 1;
    double4 *d_pos = nullptr;      // all positions (x, y, z, q)
    double4 *d_out = nullptr;      // final sums for this device's targets
    double4 *d_partial = nullptr;  // per-source-chunk partial sums
    std::size_t cap_pos = 0;
    std::size_t cap_out = 0;
    std::size_t cap_partial = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
};


class CudaCoulombSolver : public CoulombSolver
{
public:
    explicit CudaCoulombSolver(int num_gpus)
    {
        int count = 0;
        IONMD_CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count <= 0) {
            throw std::runtime_error("No CUDA devices available");
        }
        const int use = (num_gpus <= 0) ? count : std::min(num_gpus, count);

        for (int d = 0; d < use; d++) {
            DeviceContext ctx;
            ctx.id = d;
            cudaDeviceProp prop;
            IONMD_CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
            ctx.name = prop.name;
            ctx.sm_count = std::max(1, prop.multiProcessorCount);
            IONMD_CUDA_CHECK(cudaSetDevice(d));
            IONMD_CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
            devices_.push_back(ctx);
        }
    }

    ~CudaCoulombSolver() override
    {
        for (auto &d : devices_) {
            cudaSetDevice(d.id);
            if (d.d_pos) cudaFree(d.d_pos);
            if (d.d_out) cudaFree(d.d_out);
            if (d.d_partial) cudaFree(d.d_partial);
            if (d.stream) cudaStreamDestroy(d.stream);
        }
        if (h_pos_) cudaFreeHost(h_pos_);
        if (h_out_) cudaFreeHost(h_out_);
    }

    void compute(std::size_t n,
                 const double *x, const double *y, const double *z,
                 const double *q,
                 double *fx, double *fy, double *fz,
                 double *phi) override
    {
        if (n == 0) return;
        ensure_host_capacity(n);

        for (std::size_t i = 0; i < n; i++) {
            h_pos_[i] = make_double4(x[i], y[i], z[i], q[i]);
        }

        // Split targets across devices. Small systems use a single GPU since
        // the launch/transfer overhead would dominate.
        std::size_t ndev = devices_.size();
        ndev = std::max<std::size_t>(1, std::min(ndev, n/MIN_TARGETS_PER_DEVICE));
        const std::size_t chunk = (n + ndev - 1)/ndev;

        for (std::size_t d = 0; d < devices_.size(); d++) {
            auto &ctx = devices_[d];
            ctx.begin = std::min(n, d*chunk);
            ctx.end = (d < ndev) ? std::min(n, (d + 1)*chunk) : ctx.begin;
        }

        // Launch everything asynchronously ...
        for (auto &ctx : devices_) {
            const std::size_t slice = ctx.end - ctx.begin;
            if (slice == 0) continue;
            IONMD_CUDA_CHECK(cudaSetDevice(ctx.id));

            // Source split: enough blocks to fill the GPU, chunks aligned to
            // whole tiles (at least one tile per chunk).
            const std::size_t target_blocks = (slice + BLOCK_SIZE - 1)/BLOCK_SIZE;
            const std::size_t tiles = (n + BLOCK_SIZE - 1)/BLOCK_SIZE;
            const std::size_t want_blocks =
                static_cast<std::size_t>(BLOCKS_PER_SM)*ctx.sm_count;
            std::size_t nsplit = (want_blocks + target_blocks - 1)/target_blocks;
            nsplit = std::max<std::size_t>(1, std::min(nsplit, tiles));
            const std::size_t j_chunk = ((tiles + nsplit - 1)/nsplit)*BLOCK_SIZE;
            nsplit = (n + j_chunk - 1)/j_chunk;

            ensure_device_capacity(ctx, n, slice, nsplit > 1 ? nsplit*slice : 0);
            IONMD_CUDA_CHECK(cudaMemcpyAsync(ctx.d_pos, h_pos_, n*sizeof(double4),
                                             cudaMemcpyHostToDevice, ctx.stream));

            const dim3 grid(static_cast<unsigned>(target_blocks),
                            static_cast<unsigned>(nsplit));
            double4 *dest = (nsplit > 1) ? ctx.d_partial : ctx.d_out;
            coulomb_kernel<<<grid, BLOCK_SIZE, 0, ctx.stream>>>(
                ctx.d_pos, static_cast<int>(n),
                static_cast<int>(ctx.begin), static_cast<int>(ctx.end),
                static_cast<int>(j_chunk), dest);
            IONMD_CUDA_CHECK(cudaGetLastError());
            if (nsplit > 1) {
                const int rgrid = static_cast<int>((slice + 255)/256);
                reduce_kernel<<<rgrid, 256, 0, ctx.stream>>>(
                    ctx.d_partial, static_cast<int>(slice),
                    static_cast<int>(nsplit), ctx.d_out);
                IONMD_CUDA_CHECK(cudaGetLastError());
            }
            IONMD_CUDA_CHECK(cudaMemcpyAsync(h_out_ + ctx.begin, ctx.d_out,
                                             slice*sizeof(double4),
                                             cudaMemcpyDeviceToHost, ctx.stream));
        }

        // ... then wait for all devices.
        for (auto &ctx : devices_) {
            if (ctx.end == ctx.begin) continue;
            IONMD_CUDA_CHECK(cudaSetDevice(ctx.id));
            IONMD_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        }

        const double ke = constants::k_e;
        for (std::size_t i = 0; i < n; i++) {
            const double4 o = h_out_[i];
            const double pref = ke*q[i];
            fx[i] = pref*o.x;
            fy[i] = pref*o.y;
            fz[i] = pref*o.z;
        }
        if (phi) {
            for (std::size_t i = 0; i < n; i++) {
                phi[i] = ke*h_out_[i].w;
            }
        }
    }

    std::string name() const override
    {
        std::ostringstream os;
        os << "cuda (" << devices_.size() << " GPU" << (devices_.size() > 1 ? "s" : "");
        for (const auto &d : devices_) {
            os << "; " << d.id << ": " << d.name;
        }
        os << ")";
        return os.str();
    }

private:
    std::vector<DeviceContext> devices_;
    double4 *h_pos_ = nullptr;
    double4 *h_out_ = nullptr;
    std::size_t h_cap_ = 0;

    void ensure_host_capacity(std::size_t n)
    {
        if (n <= h_cap_) return;
        if (h_pos_) cudaFreeHost(h_pos_);
        if (h_out_) cudaFreeHost(h_out_);
        h_pos_ = nullptr;
        h_out_ = nullptr;
        IONMD_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&h_pos_),
                                       n*sizeof(double4), cudaHostAllocPortable));
        IONMD_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&h_out_),
                                       n*sizeof(double4), cudaHostAllocPortable));
        h_cap_ = n;
    }

    static void ensure_device_capacity(DeviceContext &ctx, std::size_t n,
                                       std::size_t slice, std::size_t partial)
    {
        if (partial > ctx.cap_partial) {
            if (ctx.d_partial) cudaFree(ctx.d_partial);
            ctx.d_partial = nullptr;
            IONMD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ctx.d_partial),
                                        partial*sizeof(double4)));
            ctx.cap_partial = partial;
        }
        if (n > ctx.cap_pos) {
            if (ctx.d_pos) cudaFree(ctx.d_pos);
            ctx.d_pos = nullptr;
            IONMD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ctx.d_pos),
                                        n*sizeof(double4)));
            ctx.cap_pos = n;
        }
        if (slice > ctx.cap_out) {
            if (ctx.d_out) cudaFree(ctx.d_out);
            ctx.d_out = nullptr;
            IONMD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ctx.d_out),
                                        slice*sizeof(double4)));
            ctx.cap_out = slice;
        }
    }
};

}  // namespace


bool cuda_compiled() { return true; }

int cuda_device_count()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();  // clear the error state
        return 0;
    }
    return count;
}

std::string cuda_device_info()
{
    const int count = cuda_device_count();
    std::ostringstream os;
    os << count << " CUDA device(s)";
    for (int d = 0; d < count; d++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
            os << "\n  [" << d << "] " << prop.name << ", sm_" << prop.major
               << prop.minor << ", " << (prop.totalGlobalMem >> 20) << " MiB";
        }
    }
    return os.str();
}

std::unique_ptr<CoulombSolver> make_cuda_coulomb_solver(int num_gpus)
{
    return std::unique_ptr<CoulombSolver>(new CudaCoulombSolver(num_gpus));
}

}  // namespace ionmd
