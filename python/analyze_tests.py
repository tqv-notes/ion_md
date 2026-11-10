#!/usr/bin/env python3
"""
analyze_tests.py - check the physics test runs against analytic expectations
and produce figures.

    python analyze_tests.py RESULTS_DIR [--show]

RESULTS_DIR is the directory written by scripts/run_tests.sh. It must contain
one sub-directory per test (01_single_ion_secular, ...). For the backend
consistency test it looks for 09_backend_consistency_cpu and
09_backend_consistency_cuda.

Writes RESULTS_DIR/figures/<test>/*.png, RESULTS_DIR/summary.md and
RESULTS_DIR/summary.json. Exit code 0 if no test FAILED.
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ionmd_io import (AMU, INK_2, KB, KE, QE, REFERENCE, Run, chain_equilibrium,  # noqa: E402
                      peak_frequency, setup_style, species_color, spectrum)
import plot_run  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Result:
    def __init__(self, name, status, detail, metrics=None):
        self.name, self.status, self.detail = name, status, detail
        self.metrics = metrics or {}


def rel(a, b):
    return abs(a - b) / abs(b)


def mathieu_beta(a, q, n=4000):
    """Exact characteristic exponent beta of x'' + (a - 2q cos 2xi) x = 0 via the
    monodromy matrix over one period (RK4). Secular frequency = beta * f_rf / 2."""
    def rhs(xi, y):
        return np.array([y[1], -(a - 2 * q * np.cos(2 * xi)) * y[0]])
    h = np.pi / n
    M = np.zeros((2, 2))
    for col, y0 in enumerate(([1.0, 0.0], [0.0, 1.0])):
        y = np.array(y0)
        xi = 0.0
        for _ in range(n):
            k1 = rhs(xi, y)
            k2 = rhs(xi + h / 2, y + h / 2 * k1)
            k3 = rhs(xi + h / 2, y + h / 2 * k2)
            k4 = rhs(xi + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            xi += h
        M[:, col] = y
    c = np.clip(0.5 * np.trace(M), -1, 1)
    return np.arccos(c) / np.pi


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_single_ion_secular(run, fig_dir, plt):
    traj = run.trajectory()
    dt = run.dt * run.info["record_every"]
    d = run.derived("Ca40")
    expect = [d["f_x"], d["f_y"], d["f_z"]]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.4))
    meas, errs = [], []
    for k, (ax, name) in enumerate(zip(axes, "xyz")):
        f, A = spectrum(traj[:, 0, k], dt)
        fm = peak_frequency(f, A, 0.3 * expect[k], 3 * expect[k])
        meas.append(fm)
        errs.append(rel(fm, expect[k]))
        ax.semilogy(f / 1e3, A / A.max(), color=species_color(0), linewidth=1.2,
                    label="simulation")
        ax.axvline(expect[k] / 1e3, color=REFERENCE, linestyle="--", linewidth=1,
                   label="pseudopotential theory")
        ax.set_xlim(0, 2.0 * max(expect) / 1e3)
        ax.set_ylim(1e-6, 2)
        ax.set_ylabel("|FFT %s| (norm.)" % name)
        ax.annotate("f = %.2f kHz (theory %.2f)" % (fm / 1e3, expect[k] / 1e3),
                    (0.98, 0.85), xycoords="axes fraction", ha="right", fontsize=9,
                    color=INK_2)
    axes[0].set_title("Single ion: secular spectrum")
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("frequency [kHz]")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spectrum.png"))
    plt.close(fig)
    worst = max(errs)
    status = PASS if worst < 5e-3 else FAIL
    return status, "max relative frequency error %.2e (tol 5e-3)" % worst, \
        {"f_measured": meas, "f_theory": expect}


def test_single_ion_micromotion(run, fig_dir, plt):
    traj = run.trajectory()
    dt = run.dt * run.info["record_every"]
    tr = run.trap
    m, Q = 40 * AMU, QE
    W = 2 * np.pi * tr["f_rf"]
    d_ec = tr["kappa"] * tr["U_ec"] / tr["z0"] ** 2
    a = 4 * Q * (tr["U_dc"] / tr["r0"] ** 2 - d_ec) / (m * W ** 2)
    q = 2 * Q * tr["V_rf"] / (m * tr["r0"] ** 2 * W ** 2)
    f_exact = mathieu_beta(a, q) * tr["f_rf"] / 2
    f_adiab = run.derived("Ca40")["f_x"]

    f, A = spectrum(traj[:, 0, 0], dt)
    f_sec = peak_frequency(f, A, 0.5 * f_adiab, 1.5 * f_adiab)
    f_sb = peak_frequency(f, A, tr["f_rf"] - 1.5 * f_adiab, tr["f_rf"] - 0.5 * f_adiab)
    e_sec = rel(f_sec, f_exact)
    e_sb = rel(f_sb, tr["f_rf"] - f_sec)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogy(f / 1e6, A / A.max(), color=species_color(0), linewidth=1.0,
                label="simulation, x")
    for fx, lab, ls in [(f_exact, "secular frequency (Mathieu)", "--"),
                        (tr["f_rf"], "rf drive", ":"),
                        (tr["f_rf"] - f_exact, "sidebands f_rf ± f_sec", "-."),
                        (tr["f_rf"] + f_exact, None, "-.")]:
        ax.axvline(fx / 1e6, color=REFERENCE, linestyle=ls, linewidth=0.9, label=lab)
    ax.set_xlim(0, 1.3 * tr["f_rf"] / 1e6)
    ax.set_ylim(1e-7, 2)
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("|FFT x| (norm.)")
    ax.set_title("Single ion with micromotion: secular peak and rf sidebands")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spectrum.png"))
    plt.close(fig)

    # zoom of the trajectory: micromotion on top of secular motion
    t = run.frame_times()
    n = min(len(t), int(3 / f_exact / dt))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(t[:n] * 1e6, traj[:n, 0, 0] * 1e6, color=species_color(0), linewidth=1.0)
    ax.set_xlabel("time [µs]")
    ax.set_ylabel("x [µm]")
    ax.set_title("Secular motion with superimposed micromotion")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "trajectory_zoom.png"))
    plt.close(fig)

    status = PASS if (e_sec < 5e-3 and e_sb < 5e-3) else FAIL
    return status, ("secular %.3f kHz vs Mathieu %.3f kHz (err %.1e, adiabatic approx "
                    "%.3f kHz); sideband err %.1e" % (f_sec / 1e3, f_exact / 1e3, e_sec,
                                                      f_adiab / 1e3, e_sb)), \
        {"f_secular": f_sec, "f_mathieu": f_exact, "f_adiabatic": f_adiab, "q": q, "a": a}


def test_linear_chain(run, fig_dir, plt):
    r = run.positions("final")
    N = len(r)
    m = 40 * AMU
    wz = 2 * np.pi * run.derived("Ca40")["f_z"]
    l = (KE * QE ** 2 / (m * wz ** 2)) ** (1 / 3)
    u_th = chain_equilibrium(N)
    z = np.sort(r[:, 2])
    zc = z - r[:, 2].mean()     # light pressure would shift the chain
    err = np.max(np.abs(zc / l - u_th))
    rad = np.max(np.hypot(r[:, 0], r[:, 1]))

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.scatter(u_th * l * 1e6, np.zeros(N) + 0.5, s=60, facecolors="none",
               edgecolors=REFERENCE, linewidths=1.2, label="theory (James 1998)")
    ax.scatter(zc * 1e6, np.zeros(N) - 0.5, s=30, color=species_color(0),
               label="simulation")
    ax.set_ylim(-2, 2)
    ax.set_yticks([])
    ax.set_xlabel("z [µm]")
    ax.set_title("Linear chain of %d ions: max deviation %.1e l" % (N, err))
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "chain.png"))
    plt.close(fig)

    status = PASS if (err < 2e-3 and rad < 0.1e-6) else FAIL
    return status, "max deviation from theory %.2e l (tol 2e-3), max radius %.1e m" % (err, rad), \
        {"length_scale": l, "max_dev": err, "max_radius": rad}


def test_laser_cooling(run, fig_dir, plt):
    d = run.diag
    t, T = d["t"], d["T_sec"]
    beta = run.config["lasers"][0]["beta"]
    m = 40 * AMU
    gamma_th = beta / m
    # fit the first part of the decay (before the crystal reaches its floor)
    sel = (t < 3 / gamma_th) & (T > 0)
    sel[:2] = False
    p = np.polyfit(t[sel], np.log(T[sel]), 1)
    gamma = -p[0]
    ratio = T[-1] / T[:3].mean()

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogy(t * 1e3, T * 1e3, color=species_color(0), label="simulation $T_{sec}$")
    ax.semilogy(t * 1e3, T[1] * 1e3 * np.exp(-gamma_th * (t - t[1])), color=REFERENCE,
                linestyle="--", linewidth=1.2, label=r"$\exp(-\beta t/m)$")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("secular temperature [mK]")
    ax.set_title("Doppler cooling: fitted rate %.0f /s (theory %.0f /s)" % (gamma, gamma_th))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "cooling_rate.png"))
    plt.close(fig)

    e = rel(gamma, gamma_th)
    status = PASS if (e < 0.25 and ratio < 0.02) else FAIL
    return status, ("cooling rate %.0f /s vs beta/m = %.0f /s (err %.0f%%, tol 25%%); "
                    "T_final/T_initial = %.1e" % (gamma, gamma_th, 100 * e, ratio)), \
        {"rate": gamma, "rate_theory": gamma_th, "T_final": T[-1]}


def test_heating_rate(run, fig_dir, plt):
    d = run.diag
    t = d["t"]
    N = run.n
    h = run.config["stochastic"]["heating_rate"]
    p = np.polyfit(t, d["E_total"], 1)
    h_meas = p[0] / (N * KB)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sc = 1 / (N * KB * 1e-3)
    ax.plot(t * 1e3, (d["E_total"] - d["E_total"][0]) * sc, color="#0b0b0b",
            linewidth=2, label="total")
    ax.plot(t * 1e3, (d["E_kin"] - d["E_kin"][0]) * sc, color=species_color(0),
            label="kinetic")
    ax.plot(t * 1e3, (d["E_trap"] + d["E_coulomb"] - d["E_trap"][0] - d["E_coulomb"][0]) * sc,
            color=species_color(1), label="potential (trap + Coulomb)")
    ax.plot(t * 1e3, h * (t - t[0]) * 1e3, color=REFERENCE, linestyle="--", linewidth=1.2,
            label="$k_B h t$ (Eq. 12)")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$\Delta E$ per ion [$k_B\,$mK]")
    ax.set_title("Velocity-kick heating: h = %.1f K/s (set %.1f K/s)" % (h_meas, h))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "heating.png"))
    plt.close(fig)

    e = rel(h_meas, h)
    status = PASS if e < 0.05 else FAIL
    return status, "measured heating rate %.2f K/s vs %.2f K/s (err %.1f%%, tol 5%%)" % (
        h_meas, h, 100 * e), {"h_measured": h_meas, "h_set": h}


def test_sympathetic(run, fig_dir, plt):
    d = run.diag
    n_last = max(1, len(d) // 5)
    r = run.positions("final")
    rad = np.hypot(r[:, 0], r[:, 1])
    temps, radii, masses = {}, {}, {}
    for s, sp in enumerate(run.species):
        lab = sp["label"]
        temps[lab] = float(np.mean(d["T_sec_" + lab][-n_last:]))
        radii[lab] = float(np.mean(rad[run.ion_species == s]))
        masses[lab] = sp["mass_amu"]

    # radial distribution by species
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, rad.max() * 1e6 * 1.05, 40)
    for s, lab in enumerate(run.labels):
        ax.hist(rad[run.ion_species == s] * 1e6, bins=bins, histtype="step",
                linewidth=1.8, color=species_color(s),
                label="%s (%.0f u): T = %.1f mK" % (lab, masses[lab], temps[lab] * 1e3))
    ax.set_xlabel("distance from trap axis [µm]")
    ax.set_ylabel("ions")
    ax.set_title("Radial segregation of species (lighter ions inside)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "radial_distribution.png"))
    plt.close(fig)

    order = sorted(masses, key=masses.get)
    ordered = all(radii[order[i]] < radii[order[i + 1]] for i in range(len(order) - 1))
    cold = all(T < 0.1 for T in temps.values())
    status = PASS if (ordered and cold) else FAIL
    detail = "; ".join("%s: <r> = %.1f um, T = %.2f mK" % (k, radii[k] * 1e6, temps[k] * 1e3)
                       for k in order)
    return status, detail + ("" if ordered else "  [radial ordering violated]"), \
        {"temperatures": temps, "mean_radius": radii}


def test_secular_spectrum(run, fig_dir, plt):
    com = run.com()
    t = run.com_times()
    dt = t[1] - t[0]
    t_stop = run.config["excitation"]["t_stop"]
    k0 = int(np.ceil(t_stop / dt)) + 1
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0))
    found = {}
    ok = True
    for s, lab in enumerate(run.labels):
        der = run.derived(lab)
        for ax, k, key in [(axes[0], 0, "f_x"), (axes[1], 2, "f_z")]:
            f, A = spectrum(com[k0:, s, k], dt)
            fp = peak_frequency(f, A, 0.3 * der[key], 2.0 * der[key])
            found["%s_%s" % (lab, key)] = fp
            ok &= 0.5 * der[key] < fp < 1.5 * der[key]
            ax.semilogy(f / 1e3, A / A.max(), color=species_color(s), linewidth=1.2,
                        label="%s COM" % lab)
            ax.axvline(der[key] / 1e3, color=species_color(s), linestyle=":", linewidth=1.2)
    fmax = max(run.derived(l)["f_x"] for l in run.labels)
    axes[0].set_xlim(0, 1.6 * fmax / 1e3)
    fzmax = max(run.derived(l)["f_z"] for l in run.labels)
    axes[1].set_xlim(0, 3 * fzmax / 1e3)
    for ax, name in zip(axes, ["radial (x)", "axial (z)"]):
        ax.set_ylim(1e-5, 2)
        ax.set_ylabel("|FFT| (norm.)")
        ax.set_title("Secular spectrum, %s; dotted = single-ion frequency" % name)
        ax.legend(loc="upper right")
    axes[1].set_xlabel("frequency [kHz]")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "secular_spectrum.png"))
    plt.close(fig)
    detail = ", ".join("%s %.1f kHz" % (k, v / 1e3) for k, v in found.items())
    return (PASS if ok else FAIL), "COM resonances: " + detail, found


def test_micromotion_crystal(run, fig_dir, plt):
    d = run.diag
    n_last = max(1, len(d) // 4)
    Tsec = float(np.mean(d["T_sec"][-n_last:]))
    Tkin = float(np.mean(d["T_kin"][-n_last:]))
    ratio = Tkin / Tsec
    status = PASS if (ratio > 3 and Tsec < 0.05) else FAIL
    return status, ("T_sec = %.2f mK, T_kin (incl. micromotion) = %.1f mK, ratio %.0f"
                    % (Tsec * 1e3, Tkin * 1e3, ratio)), {"T_sec": Tsec, "T_kin": Tkin}


def test_backend_consistency(results_dir):
    a = os.path.join(results_dir, "09_backend_consistency_cpu")
    b = os.path.join(results_dir, "09_backend_consistency_cuda")
    if not (os.path.isdir(a) and os.path.isdir(b)):
        return Result("09_backend_consistency", SKIP, "needs both cpu and cuda runs "
                      "(CUDA build / GPU node)")
    ra, rb = Run(a), Run(b)
    if not (ra.ok and rb.ok):
        return Result("09_backend_consistency", FAIL, "one of the runs did not finish")
    pa, pb = ra.positions(), rb.positions()
    size = np.max(np.abs(pa))
    err = float(np.max(np.abs(pa - pb)) / size)
    va, vb = ra.velocities_table(), rb.velocities_table()
    verr = float(np.max(np.abs(va - vb)) / np.max(np.abs(va)))
    sp = (rb.info["steps_per_second"] / ra.info["steps_per_second"])
    status = PASS if (err < 1e-8 and verr < 1e-6) else FAIL
    return Result("09_backend_consistency", status,
                  "max position diff %.1e (rel), velocity diff %.1e; GPU speed-up %.1fx "
                  "(%s vs %s)" % (err, verr, sp, rb.info["backend"], ra.info["backend"]),
                  {"pos_err": err, "vel_err": verr, "speedup": sp})


TESTS = [
    ("01_single_ion_secular", test_single_ion_secular),
    ("02_single_ion_micromotion", test_single_ion_micromotion),
    ("03_linear_chain", test_linear_chain),
    ("04_laser_cooling", test_laser_cooling),
    ("05_heating_rate", test_heating_rate),
    ("06_sympathetic_cooling", test_sympathetic),
    ("07_secular_spectrum", test_secular_spectrum),
    ("08_micromotion_crystal", test_micromotion_crystal),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results directory written by run_tests.sh")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-standard-plots", action="store_true",
                    help="skip the generic plot_run.py figures")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    results = []
    for name, fn in TESTS:
        path = os.path.join(args.results, name)
        if not os.path.isdir(path):
            results.append(Result(name, SKIP, "no output directory"))
            continue
        fig_dir = os.path.join(args.results, "figures", name)
        os.makedirs(fig_dir, exist_ok=True)
        try:
            run = Run(path)
            if not run.ok:
                msg = run.info["message"] if run.info else "run_info.json missing"
                results.append(Result(name, FAIL, "run did not finish: " + msg))
                continue
            if not args.no_standard_plots:
                plot_run.plot_all(path, fig_dir, plt)
            status, detail, metrics = fn(run, fig_dir, plt)
            metrics["steps_per_second"] = run.info["steps_per_second"]
            metrics["backend"] = run.info["backend"]
            results.append(Result(name, status, detail, metrics))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results.append(Result(name, FAIL, "analysis error: %s" % e))
    results.append(test_backend_consistency(args.results))

    # ---- summary ----
    lines = ["# IonMD physics test summary", "", "| test | status | details |",
             "|---|---|---|"]
    for r in results:
        lines.append("| %s | %s | %s |" % (r.name, r.status, r.detail))
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.results, "summary.md"), "w") as fh:
        fh.write(text + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(os.path.join(args.results, "summary.json"), "w") as fh:
        json.dump([{"test": r.name, "status": r.status, "detail": r.detail,
                    "metrics": clean(r.metrics)} for r in results], fh, indent=2)

    n_fail = sum(r.status == FAIL for r in results)
    print("\n%d passed, %d failed, %d skipped" % (
        sum(r.status == PASS for r in results), n_fail, sum(r.status == SKIP for r in results)))
    if args.show:
        plt.show()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
