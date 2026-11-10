#!/usr/bin/env python3
"""
plot_run.py - standard figures for one IonMD output directory.

    python plot_run.py RUN_DIR [RUN_DIR ...] [--out FIGDIR] [--show]

Writes (to RUN_DIR/figures unless --out is given):
    temperature.png   secular temperature per species (and kinetic temperature)
    energy.png        change of kinetic / trap / Coulomb / total energy
    crystal.png       final ion positions (z-x and x-y projections) by species
    trajectory.png    a few ion trajectories (if trajectory.bin was recorded)
    ccd.png           simulated CCD image(s) (if enabled)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ionmd_io import (KB, INK_2, Run, sequential_cmap, setup_style,  # noqa: E402
                      species_color)


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    f = os.path.join(outdir, name)
    fig.savefig(f)
    print("  wrote", f)


def plot_temperature(run, outdir, plt):
    d = run.diag
    if d is None or len(d) == 0:
        return
    t = d["t"] * 1e3
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    S = len(run.labels)
    for s, label in enumerate(run.labels):
        col = "T_sec_" + label
        if col in d.dtype.names:
            ax.plot(t, d[col] * 1e3, color=species_color(s),
                    label="%s secular" % label)
    micromotion = run.config["forces"]["micromotion"]
    if micromotion or S == 1:
        ax.plot(t, d["T_kin"] * 1e3, color=INK_2, linestyle="--", linewidth=1.2,
                label="all ions, kinetic (incl. micromotion)" if micromotion
                else "all ions, kinetic")
    ax.set_yscale("log")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("temperature [mK]")
    ax.set_title("Temperature")
    ax.legend(loc="best")
    _save(fig, outdir, "temperature.png")
    plt.close(fig)


def plot_energy(run, outdir, plt):
    d = run.diag
    if d is None or len(d) < 2:
        return
    t = d["t"] * 1e3
    N = run.n
    # Energy change per ion in units of kB * mK
    scale = 1.0 / (N * KB * 1e-3)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    items = [("E_total", "total", "#0b0b0b"), ("E_kin", "kinetic", "#2a78d6"),
             ("E_trap", "trap potential", "#eb6834"), ("E_coulomb", "Coulomb", "#1baf7a")]
    for key, label, color in items:
        ax.plot(t, (d[key] - d[key][0]) * scale, label=label, color=color,
                linewidth=2.0 if key == "E_total" else 1.4)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$\Delta E$ per ion  [$k_B\,$mK]")
    ax.set_title("Energy change since first sample")
    ax.legend(loc="best")
    _save(fig, outdir, "energy.png")
    plt.close(fig)


def plot_crystal(run, outdir, plt):
    if run.final is None:
        return
    r = run.positions("final") * 1e6
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    for s, label in enumerate(run.labels):
        m = run.ion_species == s
        kw = dict(s=14, color=species_color(s), label="%s (%d)" % (label, m.sum()),
                  edgecolors="#fcfcfb", linewidths=0.5)
        axes[0].scatter(r[m, 2], r[m, 0], **kw)
        axes[1].scatter(r[m, 0], r[m, 1], **kw)
    axes[0].set_xlabel("z (trap axis) [µm]")
    axes[0].set_ylabel("x [µm]")
    axes[0].set_title("Final positions: side view")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set_xlabel("x [µm]")
    axes[1].set_ylabel("y [µm]")
    axes[1].set_title("Axial view")
    axes[1].set_aspect("equal", adjustable="datalim")
    if len(run.labels) > 1:
        axes[0].legend(loc="upper right", markerscale=1.5)
    fig.tight_layout()
    _save(fig, outdir, "crystal.png")
    plt.close(fig)


def plot_trajectory(run, outdir, plt, max_ions=4, max_points=20000):
    traj = run.trajectory()
    if traj is None or traj.shape[0] < 2:
        return
    t = run.frame_times() * 1e3
    stride = max(1, traj.shape[0] // max_points)
    ions = np.linspace(0, run.n - 1, min(max_ions, run.n)).astype(int)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.0), sharex=True)
    for k, (ax, name) in enumerate(zip(axes, "xyz")):
        for j, i in enumerate(ions):
            ax.plot(t[::stride], traj[::stride, i, k] * 1e6, linewidth=1.0,
                    color=species_color(j), label="ion %d" % i)
        ax.set_ylabel("%s [µm]" % name)
    axes[0].set_title("Ion trajectories")
    axes[-1].set_xlabel("time [ms]")
    if len(ions) > 1:
        axes[0].legend(loc="upper right", ncol=len(ions))
    fig.tight_layout()
    _save(fig, outdir, "trajectory.png")
    plt.close(fig)


def plot_ccd(run, outdir, plt):
    ccd = run.ccd()
    if ccd is None:
        return
    ext = run.info["ccd"]["extent"] * 1e6
    S = ccd.shape[0]
    panels = [("all species", ccd.sum(axis=0))]
    if S > 1:
        panels += [(lab, ccd[s]) for s, lab in enumerate(run.labels)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 2.4 * len(panels) + 0.4),
                             squeeze=False)
    cmap = sequential_cmap()
    for ax, (title, img) in zip(axes[:, 0], panels):
        # rows = x (vertical), columns = z (horizontal); gamma for visibility
        v = img / img.max() if img.max() > 0 else img
        ax.imshow(v ** 0.5, origin="lower", extent=[-ext, ext, -ext, ext],
                  cmap=cmap, aspect="equal", interpolation="nearest")
        ax.set_title("Simulated CCD image: %s" % title)
        ax.set_ylabel("x [µm]")
        ax.grid(False)
    axes[-1, 0].set_xlabel("z (trap axis) [µm]")
    fig.tight_layout()
    _save(fig, outdir, "ccd.png")
    plt.close(fig)


def plot_all(run_dir, outdir=None, plt=None):
    run = Run(run_dir)
    outdir = outdir or os.path.join(run_dir, "figures")
    print("Plotting", run_dir, "(status: %s)" % (run.info or {}).get("status"))
    plot_temperature(run, outdir, plt)
    plot_energy(run, outdir, plt)
    plot_crystal(run, outdir, plt)
    plot_trajectory(run, outdir, plt)
    plot_ccd(run, outdir, plt)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="IonMD output directories")
    ap.add_argument("--out", help="figure directory (default RUN/figures)")
    ap.add_argument("--show", action="store_true", help="show figures interactively")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    for r in args.runs:
        out = os.path.join(args.out, os.path.basename(os.path.normpath(r))) if args.out else None
        plot_all(r, out, plt)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
