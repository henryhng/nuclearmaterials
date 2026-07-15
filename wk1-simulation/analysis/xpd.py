#!/usr/bin/env python3
"""Process simulation G(r) through an XPD-equivalent pipeline.

G(r) -> F(Q) -> truncate at Qmax with damping envelope -> back-transform
over the experimental r-window: the PDFgetX3-comparable observable
(Sprouster JNM 527: Qmax 24 1/A, fits over 1.2-50 and 1.2-15 A).
Damping parameters are assumed-typical XPD values, stated in output.

    python3 xpd.py pdf_min_SiC_cascade1.csv --outdir ../figures/pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(csvpath: Path):
    rows = list(csv.DictReader(open(csvpath)))
    r = np.array([float(x["r_A"]) for x in rows])
    gd = np.array([float(x["G_damaged"]) for x in rows])
    gp = np.array([float(x["G_pristine"]) for x in rows])
    return r, gd, gp


def to_fq(r, G, q):
    """F(Q) = int G(r) sin(Qr) dr."""
    return np.array([np.trapezoid(G * np.sin(qi * r), r) for qi in q])


def to_gr(q, F, r_out, qdamp):
    """G(r) = (2/pi) int F(Q) sin(Qr) exp(-Q^2 qdamp^2 / 2) dQ."""
    env = np.exp(-0.5 * (q * qdamp) ** 2)
    return np.array([2 / np.pi * np.trapezoid(F * env * np.sin(qi * q), q)
                     for qi in r_out])


def main():
    parser = argparse.ArgumentParser(
        description="XPD-pipeline processing of simulated G(r).")
    parser.add_argument("csvs", nargs="+", type=Path,
                        help="pdf.py output CSVs")
    parser.add_argument("--qmax", type=float, default=24.0)
    parser.add_argument("--qmin", type=float, default=0.1)
    parser.add_argument("--qdamp", type=float, default=0.04,
                        help="assumed instrument damping [1/A]")
    parser.add_argument("--rmax", type=float, default=15.0)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    q = np.arange(args.qmin, args.qmax, 0.02)
    r_out = np.arange(1.2, args.rmax, 0.02)

    for csvpath in args.csvs:
        r, gd, gp = load(csvpath)
        out = {}
        for name, g in (("damaged", gd), ("pristine", gp)):
            out[name] = to_gr(q, to_fq(r, g, q), r_out, args.qdamp)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1.plot(r_out, out["pristine"], lw=0.9, color="gray",
                 label="pristine (processed)")
        ax1.plot(r_out, out["damaged"], lw=0.9, color="tab:red",
                 label="damaged (processed)")
        ax1.set_ylabel(r"G(r) [$\mathrm{\AA^{-2}}$]")
        ax1.legend(frameon=False)
        ax1.grid(alpha=0.3)
        ax1.set_title(f"{csvpath.stem}  Qmax {args.qmax}  "
                      f"Qdamp {args.qdamp} (assumed)")
        ax2.plot(r_out, out["damaged"] - out["pristine"], lw=0.9,
                 color="tab:blue")
        ax2.axhline(0, color="gray", ls="--", lw=0.7)
        ax2.set_ylabel(r"$\Delta$G(r)")
        ax2.set_xlabel(r"r [$\mathrm{\AA}$]")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        png = args.outdir / f"pipeline_{csvpath.stem}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)

        with open(args.outdir / f"pipeline_{csvpath.stem}.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["r_A", "G_damaged_proc", "G_pristine_proc",
                        f"qmax={args.qmax}", f"qdamp={args.qdamp}"])
            for i in range(len(r_out)):
                w.writerow([f"{r_out[i]:.4f}", f"{out['damaged'][i]:.6f}",
                            f"{out['pristine'][i]:.6f}"])
        print(png)


if __name__ == "__main__":
    main()
