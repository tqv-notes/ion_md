"""
ionmd_io.py - readers for IonMD output directories and shared plot style.

Only numpy and matplotlib are required (both available as HPC modules:
py-numpy/1.26.1-gcc-13.1.0, py-matplotlib/3.8.1-gcc-13.1.0).

Usage::

    from ionmd_io import Run
    run = Run("results/04_laser_cooling")
    run.diag["t"], run.diag["T_sec"]          # diagnostics columns
    run.trajectory()                          # (frames, N, 3) float32 [m]
    run.com()                                 # (frames, S, 3) float64 [m]
    run.ccd()                                 # (S, bins, bins)
    run.final["x"], run.final["species"]      # final state table
"""

import json
import os

import numpy as np

# Physical constants (CODATA 2018), SI
AMU = 1.66053906660e-27
QE = 1.602176634e-19
KB = 1.380649e-23
KE = 8.9875517923e9

# Categorical palette in fixed order (validated: adjacent CVD dE >= 8).
# Species i always gets SERIES[i]; colours are never cycled or re-ranked.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
REFERENCE = "#52514e"   # analytic/expected lines: neutral, dashed


def setup_style():
    """Recessive grid/axes, thin marks, neutral text."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def sequential_cmap():
    """Single-hue (blue) light -> dark map for magnitudes (CCD images)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "ionmd_blue", ["#fcfcfb", "#cfe0f6", "#7fb0ea", "#2a78d6", "#123f78", "#0a1f3d"])


def species_color(index):
    return SERIES[index % len(SERIES)] if index < len(SERIES) else INK_2


def _read_state(fname):
    if not os.path.exists(fname):
        return None
    return np.genfromtxt(fname, delimiter=",", names=True, dtype=None, encoding="utf-8")


class Run:
    """An IonMD output directory."""

    def __init__(self, path):
        self.path = path
        if not os.path.isdir(path):
            raise FileNotFoundError("No such output directory: %s" % path)
        self.info = self._json("run_info.json")
        self.config = self._json("config.json")
        self.n = int(self.info["n_ions"]) if self.info else None
        self.species = self.info["species"] if self.info else []
        self.labels = [s["label"] for s in self.species]
        self.ion_species = np.asarray(self.info.get("ion_species", []), dtype=int) \
            if self.info else None
        self.dt = self.info["dt"] if self.info else None
        self.diag = self._diag()
        self.initial = _read_state(self.file("initial_state.csv"))
        self.final = _read_state(self.file("final_state.csv"))

    # -- helpers ----------------------------------------------------------
    def file(self, name):
        return os.path.join(self.path, name)

    def _json(self, name):
        f = self.file(name)
        if not os.path.exists(f):
            return None
        with open(f) as fh:
            return json.load(fh)

    def _diag(self):
        f = self.file("diagnostics.csv")
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            return None
        d = np.genfromtxt(f, delimiter=",", names=True)
        return np.atleast_1d(d)

    @property
    def ok(self):
        return self.info is not None and self.info.get("status") == "FINISHED"

    @property
    def trap(self):
        return self.config["trap"]

    def derived(self, label):
        """Mathieu q and secular frequencies [Hz] of a species from config.json."""
        return self.config.get("derived", {}).get(label)

    def species_mask(self, label):
        return self.ion_species == self.labels.index(label)

    # -- binary data --------------------------------------------------------
    def trajectory(self):
        """Positions, shape (frames, N, 3), float32 metres (memory mapped)."""
        f = self.file("trajectory.bin")
        if not os.path.exists(f):
            return None
        data = np.memmap(f, dtype=np.float32, mode="r")
        return data.reshape(-1, self.n, 3)

    def velocities(self):
        f = self.file("velocities.bin")
        if not os.path.exists(f):
            return None
        return np.memmap(f, dtype=np.float32, mode="r").reshape(-1, self.n, 3)

    def frame_times(self):
        """Times [s] of the trajectory frames."""
        traj = self.trajectory()
        if traj is None:
            return None
        every = self.info["record_every"]
        start = self.info["record_start"]
        k0 = 0 if start == 0 else int(np.ceil(start / every)) * every
        return (k0 + every * np.arange(traj.shape[0])) * self.dt

    def com(self):
        """Centre of mass per species, shape (frames, S, 3), float64 metres."""
        f = self.file("com.bin")
        if not os.path.exists(f):
            return None
        S = len(self.species)
        return np.fromfile(f, dtype=np.float64).reshape(-1, S, 3)

    def com_times(self):
        c = self.com()
        if c is None:
            return None
        return np.arange(c.shape[0]) * self.info["com_every"] * self.dt

    def ccd(self):
        """Simulated CCD images, shape (S, bins, bins); rows = x, cols = z."""
        f = self.file("ccd.bin")
        if not os.path.exists(f):
            return None
        B = self.info["ccd"]["bins"]
        return np.fromfile(f, dtype=np.float64).reshape(-1, B, B)

    def positions(self, which="final"):
        """(N, 3) positions from the initial or final state table."""
        s = self.final if which == "final" else self.initial
        return np.column_stack([s["x"], s["y"], s["z"]])

    def velocities_table(self, which="final"):
        s = self.final if which == "final" else self.initial
        return np.column_stack([s["vx"], s["vy"], s["vz"]])


def spectrum(signal, dt, window=True):
    """One-sided amplitude spectrum (frequencies [Hz], |FFT|) of a real signal."""
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    if window:
        x = x * np.hanning(len(x))
    amp = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), dt)
    return f, amp


def peak_frequency(f, amp, fmin=0.0, fmax=np.inf):
    """Frequency of the largest peak in [fmin, fmax], refined by a parabolic fit."""
    sel = np.where((f >= fmin) & (f <= fmax))[0]
    if len(sel) < 3:
        return np.nan
    k = sel[np.argmax(amp[sel])]
    if 0 < k < len(amp) - 1:
        a, b, c = np.log(amp[k - 1:k + 2] + 1e-300)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return f[k] + delta * (f[1] - f[0])
    return f[k]


def chain_equilibrium(N, iters=200):
    """
    Dimensionless equilibrium positions u_m of a linear chain of N equal ions
    (James 1998): u_m - sum_{n<m} 1/(u_m-u_n)^2 + sum_{n>m} 1/(u_m-u_n)^2 = 0.
    Physical positions are z_m = l u_m with l^3 = k_e Q^2 / (m w_z^2).
    """
    u = np.linspace(-1, 1, N) * (N ** 0.56)
    for _ in range(iters):
        d = u[:, None] - u[None, :]
        np.fill_diagonal(d, np.inf)
        g = u - np.sum(np.sign(d) / d ** 2, axis=1)
        # Jacobian
        J = -2.0 / np.abs(d) ** 3
        np.fill_diagonal(J, 0.0)
        J[np.diag_indices(N)] = 1 + 2 * np.sum(1.0 / np.abs(d) ** 3, axis=1)
        u = u - np.linalg.solve(J, g)
        if np.max(np.abs(g)) < 1e-13:
            break
    return np.sort(u)
