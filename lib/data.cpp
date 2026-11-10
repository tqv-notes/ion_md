#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <ionmd/constants.hpp>
#include <ionmd/data.hpp>
#include <ionmd/simulation.hpp>

namespace fs = std::filesystem;

namespace ionmd {

namespace {

std::FILE *open_or_throw(const std::string &fname, const char *mode)
{
    std::FILE *f = std::fopen(fname.c_str(), mode);
    if (!f) {
        throw std::runtime_error("Could not open " + fname + " for writing");
    }
    return f;
}

}  // namespace


DataWriter::DataWriter(const std::string &path, bool overwrite)
    : path_(path)
{
    if (fs::exists(path_)) {
        if (!fs::is_directory(path_)) {
            throw std::runtime_error("Output path " + path_ + " is not a directory");
        }
        if (!overwrite) {
            throw std::runtime_error("Output path " + path_ +
                                     " already exists (set output.overwrite = true)");
        }
    }
    else {
        fs::create_directories(path_);
    }
}

DataWriter::~DataWriter()
{
    for (std::FILE *f : {diag_, traj_, vel_, com_}) {
        if (f) std::fclose(f);
    }
}

std::string DataWriter::file(const std::string &name) const
{
    return (fs::path(path_) / name).string();
}

void DataWriter::write_text(const std::string &name, const std::string &content)
{
    std::ofstream out(file(name));
    if (!out) throw std::runtime_error("Could not write " + file(name));
    out << content << "\n";
}

void DataWriter::write_state_csv(const std::string &name, const Simulation &sim)
{
    std::ofstream out(file(name));
    if (!out) throw std::runtime_error("Could not write " + file(name));
    out << "index,species,label,mass_amu,charge,laser_cooled,x,y,z,vx,vy,vz\n";
    out << std::setprecision(12);

    const auto ions = sim.get_ions();
    std::vector<int> species_of(ions.size(), 0);
    const auto &species = sim.species();
    for (std::size_t s = 0; s < species.size(); s++) {
        for (auto i : species[s].members) species_of[i] = static_cast<int>(s);
    }

    for (std::size_t i = 0; i < ions.size(); i++) {
        const Ion &ion = ions[i];
        const int s = species_of[i];
        out << i << "," << s << "," << species[s].label << "," << ion.mass_amu() << ","
            << ion.Z << "," << (ion.laser_cooled ? 1 : 0) << ","
            << ion.x0.x << "," << ion.x0.y << "," << ion.x0.z << ","
            << ion.v0.x << "," << ion.v0.y << "," << ion.v0.z << "\n";
    }
}

void DataWriter::write_xyz(const std::string &name, const Simulation &sim)
{
    std::ofstream out(file(name));
    if (!out) throw std::runtime_error("Could not write " + file(name));
    const auto ions = sim.get_ions();
    out << ions.size() << "\nSimulated ion crystal, positions in microns\n";
    for (const auto &ion : ions) {
        out << std::lround(ion.mass_amu()) << " " << ion.x0.x/1e-6 << " "
            << ion.x0.y/1e-6 << " " << ion.x0.z/1e-6 << "\n";
    }
}

void DataWriter::open_diagnostics(const std::vector<Species> &species)
{
    diag_ = open_or_throw(file("diagnostics.csv"), "w");
    std::fprintf(diag_, "step,t,T_sec");
    for (const auto &s : species) std::fprintf(diag_, ",T_sec_%s", s.label.c_str());
    std::fprintf(diag_, ",T_kin,E_kin,E_trap,E_coulomb,E_total\n");
}

void DataWriter::write_diagnostics(const std::vector<double> &row)
{
    if (!diag_) return;
    for (std::size_t k = 0; k < row.size(); k++) {
        if (k == 0) std::fprintf(diag_, "%.0f", row[k]);
        else std::fprintf(diag_, ",%.10e", row[k]);
    }
    std::fprintf(diag_, "\n");
}

void DataWriter::open_trajectory(bool velocities)
{
    traj_ = open_or_throw(file("trajectory.bin"), "wb");
    if (velocities) vel_ = open_or_throw(file("velocities.bin"), "wb");
}

void DataWriter::write_frame(std::size_t n, const double *x, const double *y, const double *z)
{
    if (!traj_) return;
    fbuf_.resize(3*n);
    for (std::size_t i = 0; i < n; i++) {
        fbuf_[3*i] = static_cast<float>(x[i]);
        fbuf_[3*i + 1] = static_cast<float>(y[i]);
        fbuf_[3*i + 2] = static_cast<float>(z[i]);
    }
    std::fwrite(fbuf_.data(), sizeof(float), fbuf_.size(), traj_);
    frames_++;
}

void DataWriter::write_velocity_frame(std::size_t n, const double *vx, const double *vy,
                                      const double *vz)
{
    if (!vel_) return;
    fbuf_.resize(3*n);
    for (std::size_t i = 0; i < n; i++) {
        fbuf_[3*i] = static_cast<float>(vx[i]);
        fbuf_[3*i + 1] = static_cast<float>(vy[i]);
        fbuf_[3*i + 2] = static_cast<float>(vz[i]);
    }
    std::fwrite(fbuf_.data(), sizeof(float), fbuf_.size(), vel_);
}

void DataWriter::open_com()
{
    com_ = open_or_throw(file("com.bin"), "wb");
}

void DataWriter::write_com(const std::vector<double> &com)
{
    if (!com_) return;
    std::fwrite(com.data(), sizeof(double), com.size(), com_);
    com_frames_++;
}

void DataWriter::write_ccd(const std::vector<double> &ccd)
{
    std::FILE *f = open_or_throw(file("ccd.bin"), "wb");
    std::fwrite(ccd.data(), sizeof(double), ccd.size(), f);
    std::fclose(f);
}

}  // namespace ionmd
