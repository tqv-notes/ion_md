#!/usr/bin/env python3
"""
plot_benchmark.py - plot the output of scripts/run_benchmark.sh.

    python plot_benchmark.py benchmark.csv [--show]

Two panels: wall time per step vs N for each configuration, and speed-up
relative to the single-thread CPU run at the same N.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ionmd_io import REFERENCE, setup_style  # noqa: E402

# Single-hue ramps, light -> dark (all steps readable on the light surface).
CPU_RAMP = ["#7fb0ea", "#4a91e0", "#2a78d6", "#123f78"]
GPU_RAMP = ["#f19a6b", "#eb6834", "#b84a1c", "#7a2e0e"]


def load(csv):
    d = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(d)


def config_key(row):
    if row["backend"] == "cuda":
        return "CUDA, %d GPU%s" % (row["gpus"], "s" if row["gpus"] != 1 else "")
    return "CPU, %d thread%s" % (row["threads"], "s" if row["threads"] != 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    d = load(args.csv)
    keys = []
    for r in d:
        k = config_key(r)
        if k not in keys:
            keys.append(k)
    # stable order: CPU by threads, then CUDA by GPUs
    keys.sort(key=lambda k: (k.startswith("CUDA"), int(k.split(",")[1].split()[0])))

    base = {int(r["n_ions"]): r["time_per_step"] for r in d
            if r["backend"] == "cpu" and r["threads"] == 1}

    # Colour by backend: CPU = blue ramp (darker = more threads), CUDA = orange
    # ramp (darker = more GPUs); CUDA lines are also dashed.
    cpu_keys = [k for k in keys if k.startswith("CPU")]
    gpu_keys = [k for k in keys if k.startswith("CUDA")]

    def ramp(colors, idx, count):
        if count <= 1:
            return colors[-2] if len(colors) > 1 else colors[0]
        return colors[round(idx * (len(colors) - 1) / (count - 1))]

    def color_of(k):
        if k in cpu_keys:
            return ramp(CPU_RAMP, cpu_keys.index(k), len(cpu_keys))
        return ramp(GPU_RAMP, gpu_keys.index(k), len(gpu_keys))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in keys:
        rows = sorted([r for r in d if config_key(r) == k], key=lambda r: r["n_ions"])
        N = np.array([r["n_ions"] for r in rows])
        tps = np.array([r["time_per_step"] for r in rows])
        ls = "--" if k.startswith("CUDA") else "-"
        axes[0].loglog(N, tps * 1e3, ls, marker="o", markersize=5, color=color_of(k),
                       label=k)
        if k == "CPU, 1 thread":
            continue  # the reference itself (speed-up 1) is the grey line
        sp = [(n, base[n] / t) for n, t in zip(N, tps) if n in base]
        if sp:
            n_, s_ = zip(*sp)
            axes[1].loglog(n_, s_, ls, marker="o", markersize=5, color=color_of(k),
                           label=k)

    # O(N^2) guide
    Nall = np.unique(d["n_ions"]).astype(float)
    if len(Nall) > 1 and base:
        n0 = max(base)
        guide = base[n0] * 1e3 * (Nall / n0) ** 2
        axes[0].loglog(Nall, guide, color=REFERENCE, linestyle=":", linewidth=1,
                       label=r"$\propto N^2$")
    axes[0].set_xlabel("number of ions N")
    axes[0].set_ylabel("wall time per step [ms]")
    axes[0].set_title("Time per integration step")
    axes[0].legend(loc="upper left")
    axes[1].axhline(1, color=REFERENCE, linewidth=0.8)
    axes[1].set_xlabel("number of ions N")
    axes[1].set_ylabel("speed-up vs 1 CPU thread")
    axes[1].set_title("Parallel speed-up")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    out = os.path.splitext(args.csv)[0] + ".png"
    fig.savefig(out)
    print("wrote", out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
