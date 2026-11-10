#ifndef IONMD_DATA_HPP
#define IONMD_DATA_HPP

#include <cstdio>
#include <string>
#include <vector>

namespace ionmd {

class Simulation;
struct Species;

/**
 * Simulation data output.
 *
 * Files written to the output directory:
 *
 *   config.json        complete configuration used for the run
 *   run_info.json      metadata needed to read the binary files, timing
 *   initial_state.csv  state after equilibration (start of production)
 *   final_state.csv    state at the end of the run
 *   final.xyz          final positions in um (xyz format)
 *   diagnostics.csv    energies and temperatures vs time
 *   trajectory.bin     float32 positions [m], shape (frames, N, 3)
 *   velocities.bin     float32 velocities [m/s], shape (frames, N, 3)
 *   com.bin            float64 centre of mass per species, (frames, S, 3)
 *   ccd.bin            float64 histograms, shape (S, bins, bins)
 */
class DataWriter
{
public:
    /**
     * Create (or reuse when overwrite is true) the output directory.
     * @throws std::runtime_error on failure.
     */
    DataWriter(const std::string &path, bool overwrite);
    ~DataWriter();

    DataWriter(const DataWriter &) = delete;
    DataWriter &operator=(const DataWriter &) = delete;

    const std::string &path() const { return path_; }
    std::string file(const std::string &name) const;

    void write_text(const std::string &name, const std::string &content);
    void write_state_csv(const std::string &name, const Simulation &sim);
    void write_xyz(const std::string &name, const Simulation &sim);

    void open_diagnostics(const std::vector<Species> &species);
    void write_diagnostics(const std::vector<double> &row);

    void open_trajectory(bool velocities);
    void write_frame(std::size_t n, const double *x, const double *y, const double *z);
    void write_velocity_frame(std::size_t n, const double *vx, const double *vy,
                              const double *vz);
    unsigned long frames() const { return frames_; }

    void open_com();
    void write_com(const std::vector<double> &com);
    unsigned long com_frames() const { return com_frames_; }

    void write_ccd(const std::vector<double> &ccd);

private:
    std::string path_;
    std::FILE *diag_ = nullptr;
    std::FILE *traj_ = nullptr;
    std::FILE *vel_ = nullptr;
    std::FILE *com_ = nullptr;
    unsigned long frames_ = 0;
    unsigned long com_frames_ = 0;
    std::vector<float> fbuf_;
};

}  // namespace ionmd

#endif
