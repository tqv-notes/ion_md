# ion_md: methods, algorithms and parallelisation

This document describes how ion_md works internally: the physical model, the
numerical methods, how the work is parallelised with OpenMP and CUDA, and the
measured performance on the target node (8 CPU cores, 2 × Tesla V100-SXM2-32GB).
It ends with a code review: known limitations and possible improvements.

For building and running, see the [README](../README.md).

## Contents

1. [Overview of one time step](#1-overview-of-one-time-step)
2. [Physical model](#2-physical-model)
3. [Time integration](#3-time-integration)
4. [Diagnostics](#4-diagnostics)
5. [Where the time goes: cost model](#5-where-the-time-goes-cost-model)
6. [CPU parallelisation with OpenMP](#6-cpu-parallelisation-with-openmp)
7. [GPU parallelisation with CUDA](#7-gpu-parallelisation-with-cuda)
8. [Reproducibility and precision](#8-reproducibility-and-precision)
9. [Measured performance](#9-measured-performance)
10. [Code review: limitations and possible improvements](#10-code-review-limitations-and-possible-improvements)

Source map:

| File | Content |
|---|---|
| [`include/ionmd/trap.hpp`](../include/ionmd/trap.hpp) | Paul trap potential, rf and pseudopotential forces, Mathieu parameters |
| [`include/ionmd/laser.hpp`](../include/ionmd/laser.hpp) | Doppler cooling (damping) force |
| [`include/ionmd/util.hpp`](../include/ionmd/util.hpp) | counter-based random numbers |
| [`include/ionmd/coulomb.hpp`](../include/ionmd/coulomb.hpp) | Coulomb solver interface and backend selection |
| [`lib/coulomb_cpu.cpp`](../lib/coulomb_cpu.cpp) | OpenMP + SIMD Coulomb solver |
| [`lib/coulomb_cuda.cu`](../lib/coulomb_cuda.cu) | CUDA (multi-GPU) Coulomb solver |
| [`lib/simulation.cpp`](../lib/simulation.cpp) | integrator, forces, diagnostics, run driver |
| [`lib/config.cpp`](../lib/config.cpp) | JSON configuration, initial conditions |
| [`lib/data.cpp`](../lib/data.cpp) | output files |

---

## 1. Overview of one time step

```
           positions x(t), velocities v(t), accelerations a(t)
                                   |
   (1) drift + half kick   x <- x + v dt + a dt^2/2 ,  v <- v + a dt/2      O(N)   OpenMP
                                   |
   (2) Coulomb forces      F_C,i = k_e q_i sum_j q_j (r_i - r_j)/|r_i - r_j|^3    O(N^2) OpenMP or GPU
                                   |
   (3) other forces        trap (rf or pseudo) + laser + excitation        O(N)   OpenMP
       accelerations       a = F/m ;  bounds check
   (4) second half kick    v <- v + a dt/2
   (5) heating kicks       v <- v + v0 * (random unit vector)
                                   |
   (6) diagnostics / output (every k steps)                               O(N)   serial
```

Step (2) is the only O(N²) part. For more than a few hundred ions it is
essentially the whole cost of the simulation, so it is the part that runs on
the GPU.

## 2. Physical model

The equation of motion of ion *i* (mass $m_i$, charge $q_i = Z_i e$) is
(Zhang *et al.*, PRA **76**, 012719, Eq. 1):

$$
m_i \ddot{\vec r}_i = \vec F^{\mathrm{trap}}_i + \vec F^{\mathrm{C}}_i + \vec F^{\mathrm{laser}}_i + \vec F^{\mathrm{exc}}_i + \text{stochastic heating kicks}.
$$

### 2.1 Linear Paul trap ([`trap.hpp`](../include/ionmd/trap.hpp))

The potential near the trap centre is

$$
\phi(\vec r, t) = \frac{(U_{dc} + V_{rf}\cos(\Omega t + \varphi))(x^2 - y^2)}{2 r_0^2} + \frac{\kappa U_{ec}(2z^2 - x^2 - y^2)}{2 z_0^2},
$$

giving the Mathieu parameters $q = 2QV_{rf}/(m r_0^2\Omega^2)$ and $a = 4QU_{dc}/(m r_0^2\Omega^2)$. There are two ways to compute the trap force:

* **Full rf force** (`forces.micromotion`): $\vec F = -Q\nabla\phi(\vec r, t)$,
  evaluated at every step. This reproduces the micromotion at frequency Ω. The
  time step must resolve the rf period, typically $dt \approx T_{rf}/100$
  (4 ns for a 5 MHz trap).
* **Pseudopotential** (`forces.secular`): the time-averaged harmonic potential
  with spring constants $k = m\omega^2$,

$$
\omega_z^2 = \frac{2Q\kappa U_{ec}}{m z_0^2},\qquad
\omega_{x,y}^2 = \frac{\Omega^2}{4}\left(\pm a + \frac{q^2}{2}\right) - \frac{\omega_z^2}{2}.
$$

  There is no micromotion, so the time step only has to resolve the secular
  motion (hundreds of kHz). Steps of 20 ns or more are possible, about 5× fewer
  steps for the same simulated time.

The same convention is used for $q$, the full field and the pseudopotential,
so the two models agree. Unit test: the secular frequency of a single ion
in the full rf field matches the exact Mathieu characteristic exponent β
(test 02: 208.480 kHz vs 208.455 kHz). The original code's pseudopotential
was 4× too strong in the rf term; this was corrected.

### 2.2 Coulomb interaction

$$
\vec F^{C}_i = k_e q_i \sum_{j\ne i} q_j \frac{\vec r_i - \vec r_j}{|\vec r_i - \vec r_j|^3},
\qquad \phi_i = k_e \sum_{j\ne i}\frac{q_j}{|\vec r_i - \vec r_j|}.
$$

The forces come from a **direct, exact pairwise sum**: no cut-off, no
softening and no approximation. Ion crystals are long-range ordered and the
Coulomb force is long-range, so a cut-off would be wrong. Tree or multipole
methods would only pay off at much larger N (see §10). The potential $\phi_i$
is only computed on steps where energies are written.

### 2.3 Laser cooling ([`laser.hpp`](../include/ionmd/laser.hpp))

This is the damping model of PRA Eq. 14:

$$
\vec F^{\mathrm{laser}} = -\beta\vec v + F_0\hat k ,
$$

applied only to ions marked `laser_cooled` (e.g. Ca⁺). It is a viscous
friction (the linearised Doppler force, $\beta$ in kg/s) plus a constant
radiation-pressure force along the beam. With `directional: true`, only
$\vec v\cdot\hat k$ is damped (a single beam along the trap axis); radial
motion is then cooled only through Coulomb coupling, as in experiments.
Ions that are not laser-cooled (N₂⁺, CaO⁺, …) are cooled **sympathetically**,
purely through the Coulomb interaction with the cooled ions.

### 2.4 Stochastic heating

Heating (background-gas collisions, photon recoil, rf noise) is modelled as
in PRA Eqs. 10–12. At every step each ion gets a velocity kick of fixed size
$v_0$ in a uniformly random direction:

$$
v_0 = \sqrt{\frac{2 k_B h\,dt}{m}} \quad\Rightarrow\quad
\frac{dE}{dt} = \frac{m v_0^2}{2\,dt} = k_B h \text{ per ion},
$$

where $h$ is the heating rate in K/s (global or per species). Test 05 checks
that the total energy grows at $N k_B h$.

### 2.5 Secular excitation

A uniform oscillating field $\vec E = E_0\cos(2\pi f t)\,\hat e$ acts between
`t_start` and `t_stop` (thesis Ch. 10). Sweeping $f$, or applying a short
kick with $f = 0$, excites the centre-of-mass modes. Their resonance
frequencies identify the ion species (test 07).

### 2.6 Initial state and "minimisation"

Ions start at random positions in a cylinder, on a chain along z, or at
positions read from a file ([`config.cpp`](../lib/config.cpp)). Before
production, an optional **equilibration** runs with a strong damping force
$-\beta_{eq}\vec v$ on all ions in the pseudopotential, without heating. The
cloud then relaxes to a Coulomb crystal, and the velocities are reset to zero
(the original code's `minimize()`). Optionally the ions are then given
Maxwell–Boltzmann velocities at `initial_temperature`.

## 3. Time integration

ion_md uses the **velocity Verlet** scheme (leapfrog in kick–drift–kick form),
in [`Simulation::advance`](../lib/simulation.cpp):

$$
\begin{aligned}
\vec x(t+dt) &= \vec x(t) + \vec v(t)\,dt + \tfrac12\vec a(t)\,dt^2 \\
\vec v_{1/2} &= \vec v(t) + \tfrac12\vec a(t)\,dt \\
\vec a(t+dt) &= \vec F\big(\vec x(t+dt),\, \vec v_{1/2},\, t+dt\big)/m \\
\vec v(t+dt) &= \vec v_{1/2} + \tfrac12\vec a(t+dt)\,dt
\end{aligned}
$$

Why velocity Verlet:

* **Second-order accurate and symplectic** for position-dependent forces. The
  energy error stays bounded instead of drifting, which matters for runs of
  10⁶–10⁷ steps. The unit tests check energy conservation with only the trap
  and Coulomb forces.
* **One force evaluation per step**, the minimum possible. The Coulomb sum is
  the expensive part, so higher-order schemes such as RK4 (4 evaluations per
  step) would cost about 4× more per step.
* **Time-reversible**, and handles the explicitly time-dependent rf force
  without modification.

The velocity-dependent laser force is evaluated with the half-step velocity
$\vec v_{1/2}$. This keeps the scheme explicit, at the price of a first-order
error proportional to $\beta\,dt/m$. For Ca⁺ with the default β that is
$\approx 3000\ \mathrm{s^{-1}} \times 20\ \mathrm{ns} = 6\times10^{-5}$ per
step, which is negligible. The random heating kicks are added after the
deterministic update, so the energy input per step is exactly $m v_0^2/2$
on average.

All particle data is stored as a **structure of arrays** (`x_[]`, `y_[]`,
`z_[]`, `vx_[]`, …) rather than an array of `Ion` objects. Consecutive ions
are then contiguous in memory, which lets the compiler vectorise the loops
(SIMD) and gives coalesced memory access on the GPU.

## 4. Diagnostics

* **Secular temperature** (PRA Eq. 6): $T_{sec} = \frac{m\langle\bar v^2\rangle}{3k_B}$,
  where $\bar v$ is the velocity **averaged over one rf period**. This removes
  the micromotion, which is driven motion and not heat. In pseudopotential runs
  $\bar v = v$. It is reported per species.
* **Kinetic temperature** $T_{kin}$ from the instantaneous velocities,
  including micromotion (PRA Eq. 7). Test 08 checks $T_{sec}\ll T_{kin}$ in a
  crystal with micromotion.
* **Energies**: kinetic, trap (rf or pseudo-potential) and Coulomb
  ($\tfrac12\sum_i q_i\phi_i$).
* **Centre of mass per species**: its Fourier spectrum gives the secular
  resonances.
* **Simulated CCD image**: per-species 2-D histograms of the positions
  projected on the x–z plane, accumulated over many steps, like a camera
  exposure.

## 5. Where the time goes: cost model

Per time step, with N ions:

| Part | Work | Scaling |
|---|---|---|
| Coulomb sum | N(N−1) pair interactions, ~20 floating-point operations + one $1/\sqrt{r^2}$ each | **O(N²)** |
| Trap, laser, excitation, kicks, drift | a few dozen operations per ion | O(N) |
| Diagnostics, output | per ion, only every k steps | O(N) |

For N = 1000 the Coulomb sum is 10⁶ pair interactions per step, against about
10⁴ operations for everything else. For N = 30 000 it is 9 × 10⁸ pair
interactions per step. The program therefore parallelises **the Coulomb sum
first**. The O(N) parts only need to be cheap enough not to become a
bottleneck (Amdahl's law: if a fraction *s* of the work stays serial, the
speed-up can never exceed 1/*s*).

The pair interactions are **independent of each other** and every ion needs
the same computation. This is an ideal case for data parallelism: thousands
of identical, independent operations, which is exactly what SIMD units, CPU
cores and GPUs are built for.

## 6. CPU parallelisation with OpenMP

OpenMP is a shared-memory threading model. The program runs as one process,
and loops marked with `#pragma omp parallel for` are divided among threads
(one per core). All threads see the same arrays, so nothing needs to be
copied. OpenMP helps at three levels.

### 6.1 Coulomb sum: threads over targets, SIMD over sources

[`coulomb_cpu.cpp`](../lib/coulomb_cpu.cpp):

```cpp
#pragma omp parallel for schedule(static) if(nn >= 192)
for (long i = 0; i < nn; i++) {                     // target ions -> threads
    double sx = 0, sy = 0, sz = 0, sp = 0;
    #pragma omp simd reduction(+:sx,sy,sz,sp)
    for (long j = 0; j < nn; j++) {                 // source ions -> SIMD lanes
        ... inv_r = (j == i) ? 0 : 1/sqrt(r2);  sx += q_j inv_r^3 dx; ...
    }
    fx[i] = k_e q_i sx; ...
}
```

* **Outer loop (target ions) is split across threads.** Thread *t* computes
  the total force on its own block of ions. Each thread writes only to its
  own `fx[i]`, so there are **no race conditions, locks or atomics**, and the
  speed-up can in principle be perfect.
* **Inner loop (source ions) is vectorised with `omp simd`.** Each core
  handles 2 (SSE2), 4 (AVX2) or 8 (AVX-512) ions at once.
  `scripts/build.sh` compiles with `-march=native` so the widest instruction
  set of the node is used. The self-interaction $j=i$ is masked with a select
  instead of an `if`, which keeps the loop branch-free and vectorisable.
* **`schedule(static)`**: every target costs exactly N pair interactions, so
  the load is perfectly balanced and the cheapest scheduling is enough.
* **`if(nn >= 192)`**: starting threads costs about 10 µs. For a handful of
  ions that costs more than the work saved, so small systems stay serial.

**Design choice: the full N² sum instead of N²/2.** Newton's third law
($\vec F_{ij} = -\vec F_{ji}$) would halve the arithmetic. However, each pair
would then write to two ions, and two threads could update the same `fx[j]`
at the same time. That would need atomics or per-thread force buffers plus a
reduction, and the summation order (hence the rounding) would depend on the
thread count. Computing every pair twice costs 2× the arithmetic but gives
conflict-free parallelism, perfect scaling and thread-count-independent
results. The GPU uses the same approach.

### 6.2 The O(N) loops

The drift/kick update, the force assembly in `accelerations()` and the
thermalisation are OpenMP loops over ions. The **bounds check** (an ion
escaping or becoming NaN) is folded into the force loop as a
`reduction(+:bad)`, so it costs no extra pass over memory. These loops are
memory-bound and very cheap, so they are only run in parallel above
`OMP_MIN_N = 2048` ions. Below that, the fork/join cost would exceed the work.

### 6.3 Thread placement

`scripts/hpc_env.sh` sets `OMP_NUM_THREADS` (from PBS `NCPUS` if available),
`OMP_PROC_BIND=close` and `OMP_PLACES=cores`. Each thread is pinned to its own
physical core and stays there, so its data stays in that core's caches.

### 6.4 What OpenMP achieves

The measured speed-up with 8 threads is **8.0× at N = 10 000** (perfect) and
about 5× at N = 300–3000 (§9). The CPU reaches about 2.4 × 10⁹ pair
interactions per second on 8 cores. That is enough for a few thousand ions,
but N = 30 000 needs 0.37 s per step: about 4 days for 10⁶ steps.

## 7. GPU parallelisation with CUDA

A V100 has 80 streaming multiprocessors (SMs) and 5120 CUDA cores, including
2560 FP64 units (7.8 TFLOP/s in double precision), plus 900 GB/s of memory
bandwidth. It can keep tens of thousands of threads in flight. The Coulomb sum
has exactly the structure a GPU wants: very many identical, independent
calculations with a high ratio of arithmetic to memory traffic (each loaded
position is reused by every thread in a block).

### 7.1 Offload model

Only the Coulomb sum runs on the GPU; the integrator stays on the CPU. Each
step, [`CudaCoulombSolver::compute`](../lib/coulomb_cuda.cu):

1. packs (x, y, z, q) into a `double4` array in **pinned** (page-locked) host
   memory, which allows fast DMA transfers;
2. copies it to every GPU (`cudaMemcpyAsync`, 32 bytes per ion);
3. launches the force kernel and, if needed, a reduction kernel;
4. copies the forces back (32 bytes per ion) and waits for all GPUs;
5. multiplies by $k_e q_i$ and unpacks into `fx`, `fy`, `fz`.

All operations on one GPU are queued on its own CUDA **stream**, and launched
on all GPUs before the host waits for any of them, so the GPUs work at the
same time.

### 7.2 The force kernel: tiled all-pairs with shared memory

This is the classic N-body kernel (Nyland, Harris & Prins, *GPU Gems 3*,
ch. 31), in double precision:

```
grid  = (target blocks) × (source chunks),   block = 128 threads
thread (bx, tx):  target ion i = bx*128 + tx
for each tile of 128 source ions in chunk by:
    every thread loads ONE source ion into shared memory   (coalesced load)
    __syncthreads()
    every thread loops over the 128 ions in shared memory  (fast on-chip reads)
    __syncthreads()
write partial sum for (i, chunk by)
```

* **One thread per target ion**, accumulating in registers. As on the CPU,
  each thread writes only its own result, so no atomics are needed.
* **Shared-memory tiling.** Without it, every thread would read all N source
  positions from global memory: N² reads per step. With tiles, the 128
  threads of a block load a tile cooperatively, one position each, then all
  of them reuse it from fast on-chip shared memory. Global memory traffic
  drops 128×, and the kernel becomes limited by arithmetic, not memory.
* **`rsqrt()`** computes $1/\sqrt{r^2}$ with a dedicated hardware
  approximation refined by Newton iterations. This is cheaper than a separate
  square root and division.

### 7.3 Splitting the source range (occupancy)

A GPU is only fast when it has enough parallel work. If each block handled
128 targets against **all** N sources, N = 10 000 would launch only 79 blocks
for 80 SMs, each of which can hold about 8–16 blocks. Most of the GPU would be
idle, and there would be no other blocks to hide memory and instruction
latency. The first version of the kernel (256 targets per block, no split)
launched only 40 blocks at N = 10 000 and 4 blocks at N = 1000.

The kernel therefore also **splits the source ions into chunks** (the second
grid dimension). Block $(b_x, b_y)$ computes the force of source chunk $b_y$
on the targets of block $b_x$. The number of chunks is chosen so that there
are about 16 blocks per SM (1280 blocks on a V100). A small second kernel then
**adds the partial sums for each target in a fixed order**.

Why not `atomicAdd`: atomic additions would be simpler, but floating-point
addition is not associative, and atomics add in an unpredictable order. Each
run would then give results that differ in the last bits. The two-pass sum is
deterministic and just as fast here.

This change made the GPU 2–3.4× faster at N = 1000–10 000 (§9).

### 7.4 Multiple GPUs

The two GPUs use **replicated data with target decomposition**:

* every GPU receives **all** N positions (32 N bytes, small);
* the **target ions are divided** between the GPUs (GPU 0 takes ions
  0…N/2−1, GPU 1 the rest);
* each GPU computes the complete force on its own targets, and the host
  gathers the results.

No data is exchanged between the GPUs, and each GPU does exactly half of the
pair interactions, so the speed-up is close to 2× once the GPUs are busy. A
second GPU is only used if each GPU gets at least 2048 targets
(`MIN_TARGETS_PER_DEVICE`). Below that, the fixed cost of using a device
(launch, copies, synchronisation; about 10–30 µs) outweighs the gain.

### 7.5 Backend selection

`backend: "auto"` (the default) uses the GPU when CUDA is available and
N ≥ `gpu_min_ions` (default 256); otherwise it uses the OpenMP solver. At
N = 300 the GPU already beats 8 CPU threads (43 µs vs 63 µs per step).

### 7.6 What the GPU can and cannot speed up

The GPU removes the O(N²) bottleneck. Two costs remain per step:

* **Fixed latency**: kernel launch, PCIe transfers and synchronisation. This
  is about 30 µs per step and dominates below N ≈ 1000. That is why the GPU
  curve is flat at small N.
* **Transfers and host work**: 64 bytes per ion cross PCIe every step (about
  2 MB at N = 30 000, roughly 0.15 ms), plus the serial pack/unpack loops and
  the O(N) integrator on the CPU. At N = 30 000 on 2 GPUs this is a few
  percent of the 2.9 ms step.

## 8. Reproducibility and precision

### 8.1 Counter-based random numbers

Heating kicks, initial positions and thermal velocities use a
**counter-based** generator ([`util.hpp`](../include/ionmd/util.hpp)). Each
random number is a pure hash function (splitmix64) of
(seed, step, ion index, stream):

```
u = hash(seed, step, i, k) / 2^64
```

A conventional generator has internal state that must be advanced in
sequence. With threads, either they share it (locks, plus results that depend
on which thread came first) or each thread has its own (results that depend
on how ions were split among threads). A counter-based generator has no
state: ion *i* at step *n* always gets the same random numbers, whatever
thread or device computes it.

### 8.2 What is reproducible

| Comparison | Result |
|---|---|
| CPU, any number of threads | **bit-for-bit identical** (each sum is always done in the same order; tested) |
| GPU, repeated runs on the same hardware | identical (fixed-order reduction, no atomics) |
| CPU vs GPU | forces agree to ~10⁻¹⁵ relative (different summation order and `rsqrt`) |

A many-ion simulation is chaotic: rounding differences of 10⁻¹⁶ grow
exponentially, so CPU and GPU **trajectories** separate after enough steps.
Their **statistics** (temperatures, crystal structure, spectra) agree. Test 09
compares CPU and GPU over a short run and finds 10⁻¹² relative difference.

### 8.3 Why double precision

Everything runs in FP64, although a V100 is 2× faster in FP32 and consumer
GPUs are often 32–64× faster. In a crystal, the force on an ion is the small
difference between large trap and Coulomb forces that almost cancel. With
FP32 (7 significant digits), rounding noise in that difference acts like a
random force, which heats the crystal at the mK level we want to resolve.
Mixed precision is possible (e.g. FP32 pair terms with FP64 accumulation),
but would have to be validated against the tests first.

## 9. Measured performance

Tesla V100-SXM2-32GB × 2 and 8 CPU cores, CUDA 12.8, GCC 13.1
(`scripts/run_benchmark.sh`). Wall time per integration step, including the
whole step (Coulomb, other forces and integration):

| N | 1 CPU thread | 8 CPU threads | 1 GPU | 2 GPUs | best vs 1 thread |
|---:|---:|---:|---:|---:|---:|
| 100 | 37 µs | 37 µs | 32 µs | 32 µs | 1.1× |
| 300 | 307 µs | 63 µs | 43 µs | 41 µs | 7.6× |
| 1 000 | 3.4 ms | 0.66 ms | 66 µs | 66 µs | 51× |
| 3 000 | 30 ms | 6.3 ms | 145 µs | 145 µs | 207× |
| 10 000 | 333 ms | 42 ms | 0.73 ms | 0.51 ms | 650× |
| 30 000 | ~3 s (extrapolated) | 375 ms | 5.2 ms | 2.9 ms | ~1000× |

Throughput, in pair interactions per second:

* one CPU core: 3.0 × 10⁸;
* 8 cores: 2.4 × 10⁹;
* one V100: 1.7 × 10¹¹;
* two V100s: 3.1 × 10¹¹.

Each V100 is close to its double-precision arithmetic limit for this kernel.
The single-GPU curve flattens at large N because the kernel is then limited
by arithmetic, not by occupancy or latency.

Physics test suite on the same node: 9/9 tests pass, and the GPU/CPU forces
agree to 10⁻¹². Typical production costs for 10⁶ steps on 2 GPUs: about
9 minutes for 10⁴ ions, 50 minutes for 3 × 10⁴ ions.

## 10. Code review: limitations and possible improvements

The core algorithms are standard and validated against analytic results:
single-ion frequencies, exact Mathieu exponents, James chain equilibria,
cooling and heating rates, and CPU/GPU agreement. The points below are the
known limitations, roughly in order of practical relevance.

### Limitations to be aware of

1. **The time step is not checked against the physics.** With micromotion,
   dt must resolve the rf period (≲ T_rf/50). In secular mode it must resolve
   the highest secular frequency. A too-large dt is only detected indirectly,
   when ions escape past `abort_radius`. A warning at start-up when
   `dt > T_rf/50` would be a cheap safeguard.
2. **No softening of close encounters.** The exact Coulomb force is used.
   This is right for crystals and cold clouds, but in a very hot or
   initially random dense cloud two ions can pass very close and receive a
   huge kick within one step. The usual remedy is the equilibration stage;
   for hot-cloud studies, use a smaller dt.
3. **First-order treatment of the velocity-dependent laser force** (§3).
   This is negligible for realistic β, but would matter for an extreme
   `equilibrate_beta`. That only affects the equilibration stage, whose
   outcome (a relaxed crystal) does not depend on accuracy.
4. **The damping model is not a full Doppler model.** β and F₀ are
   constants; detuning, saturation and photon-recoil heating are not computed
   from laser parameters. This matches PRA Eq. 14. Recoil heating can be
   included through `heating_rate`.
5. **Test 05 tolerance margin.** The measured heating rate (51.8 vs 50 K/s,
   tolerance 5%) is subject to statistical noise from a short run. A longer
   run or more ions would make the test more robust.
6. **CPU scaling at medium N** is about 5× on 8 threads for N = 300–3000, but
   8× at N = 10 000. The cause was not identified; candidates are node
   sharing and CPU frequency changes with load. This doesn't matter in
   practice, because the GPU is used for N ≥ 256.

### Possible performance improvements (only if needed)

| Idea | Gain | Effort | When worth it |
|---|---|---|---|
| Whole time step on the GPU (positions stay on the device; no per-step PCIe transfer) | removes the ~30 µs/step floor: 2–5× for N ≲ 3000 | high (port the integrator, RNG, diagnostics) | many long runs with small crystals |
| OpenMP for the pack/unpack loops in the CUDA solver | a few % at N ≳ 30 000 | very low (needs `-Xcompiler -fopenmp`) | very large N |
| Mixed precision (FP32 pair terms, FP64 sums) | up to ~2× on V100, much more on consumer GPUs | medium + careful validation | if FP64 throughput is the limit |
| Newton's 3rd law on the CPU (per-thread force buffers) | ≤ 2× on CPU | medium; loses thread-count-independent bitwise results | CPU-only machines |
| Tree code (Barnes–Hut) or fast multipole method, O(N log N) / O(N) | large for N ≳ 10⁵ | high; approximation errors must be controlled | very large clouds |
| MPI across several nodes (same target decomposition) | linear in nodes for N ≳ 10⁵ | high | beyond one node's GPUs |

For crystals of up to a few times 10⁴ ions, the current design (exact
double-precision O(N²) sum on 1–2 GPUs, everything else on OpenMP CPU threads)
is near the hardware limit of this node. None of these changes is necessary
for the planned simulations.

## References

1. C. B. Zhang, D. Offenberg, B. Roth, M. A. Wilson and S. Schiller,
   *Molecular-dynamics simulations of cold single-species and multispecies
   ion ensembles in a linear Paul trap*, Phys. Rev. A **76**, 012719 (2007).
2. C. B. Zhang, PhD thesis, Heinrich-Heine-Universität Düsseldorf (2008).
3. L. Nyland, M. Harris and J. Prins, *Fast N-Body Simulation with CUDA*,
   GPU Gems 3, ch. 31 (2007).
4. L. Verlet, Phys. Rev. **159**, 98 (1967); W. C. Swope *et al.*,
   J. Chem. Phys. **76**, 637 (1982) (velocity Verlet).
5. J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, *Parallel random
   numbers: as easy as 1, 2, 3*, SC'11 (2011) (counter-based RNGs).
6. D. F. V. James, Appl. Phys. B **66**, 181 (1998).
