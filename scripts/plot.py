#!/usr/bin/env python3
"""Plot GFLOPS vs N for every kernel + cuBLAS from results/results.csv -> results/gflops.png"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["naive", "coalesced", "smem_tiled", "reg_blocked", "vectorized", "cublas"]


def main():
    path = os.path.join(ROOT, "results", "results.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found - run ./build/sgemm_bench first.")
    data = defaultdict(dict)  # kernel -> {N: gflops}
    with open(path) as f:
        for r in csv.DictReader(f):
            data[r["kernel"]][int(r["N"])] = float(r["gflops"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in ORDER:
        if k not in data:
            continue
        ns = sorted(data[k])
        style = dict(linestyle="--", color="black", linewidth=2) if k == "cublas" else {}
        ax.plot(ns, [data[k][n] for n in ns], marker="o", label=k, **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({n for d in data.values() for n in d}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Matrix size N (N x N, fp32)")
    ax.set_ylabel("GFLOPS (median of runs)")
    ax.set_title("SGEMM optimisation ladder vs cuBLAS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = os.path.join(ROOT, "results", "gflops.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
