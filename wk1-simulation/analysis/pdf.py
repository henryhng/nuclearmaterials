#!/usr/bin/env python3
"""Reduced pair distribution function G(r) from dumps -> figures + CSV.

Damaged-vs-pristine overlay with total and partial (pair-type) RDFs,
the synchrotron-comparable observable (Sprouster JNM 527, 2019).

    python3 pdf.py traj_SiC_relax_1.lammpstrj \
        --reference perfect_lattice_SiC.data --outdir ../figures/pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier


def rdf(source: Path, frame: int, cutoff: float, bins: int):
    """r, total g(r), {pair: partial g(r)}, number density [1/A^3]."""
    pipeline = import_file(str(source))
    pipeline.modifiers.append(CoordinationAnalysisModifier(
        cutoff=cutoff, number_of_bins=bins, partial=True))
    if frame < 0:
        frame = pipeline.source.num_frames + frame
    data = pipeline.compute(frame)
    table = data.tables["coordination-rdf"]
    r = np.asarray(table.xy()[:, 0])
    rho = data.particles.count / data.cell.volume

    # Key partials by type-id pair: dump vs data type names differ.
    types = np.asarray(data.particles["Particle Type"])
    rev = {(t.name or str(t.id)): t.id
           for t in data.particles["Particle Type"].types}
    partials = {}
    for i, name in enumerate(table.y.component_names):
        a, b = (rev[n] for n in str(name).split("-"))
        partials[tuple(sorted((a, b)))] = np.asarray(table.y[:, i])

    # Total g(r): concentration-weighted partial sum.
    conc = {tid: np.count_nonzero(types == tid) / len(types)
            for tid in rev.values()}
    total = np.zeros_like(r)
    for (a, b), g in partials.items():
        total += conc[a] * conc[b] * (2.0 if a != b else 1.0) * g
    return r, total, partials, rho


def reduced(r, g, rho):
    """G(r) = 4 pi r rho (g(r) - 1)."""
    return 4 * np.pi * r * rho * (g - 1.0)


def plot_pdf(dump: Path, reference: Path, frame: int, cutoff: float,
             bins: int, outdir: Path) -> Path:
    r, g_d, part_d, rho_d = rdf(dump, frame, cutoff, bins)
    _, g_p, part_p, rho_p = rdf(reference, 0, cutoff, bins)
    G_d, G_p = reduced(r, g_d, rho_d), reduced(r, g_p, rho_p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(r, G_p, lw=0.9, color="gray", label="pristine")
    ax1.plot(r, G_d, lw=0.9, color="tab:red", label="damaged")
    ax1.set_ylabel(r"G(r) [$\mathrm{\AA^{-2}}$]")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)
    ax1.set_title(dump.name)

    ax2.plot(r, G_d - G_p, lw=0.9, color="tab:blue")
    ax2.axhline(0, color="gray", ls="--", lw=0.7)
    ax2.set_ylabel(r"$\Delta$G(r) damaged $-$ pristine")
    ax2.set_xlabel(r"r [$\mathrm{\AA}$]")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png = outdir / f"pdf_{dump.stem}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    # Partials: sublattice-resolved damage signature.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pair in sorted(part_d):
        d = reduced(r, part_d[pair], rho_d) - reduced(r, part_p[pair], rho_p)
        ax.plot(r, d, lw=0.9, label="-".join(map(str, pair)))
    ax.axhline(0, color="gray", ls="--", lw=0.7)
    ax.set_xlabel(r"r [$\mathrm{\AA}$]")
    ax.set_ylabel(r"$\Delta$G(r) per pair")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.set_title(f"{dump.name} partial PDFs")
    fig.tight_layout()
    fig.savefig(outdir / f"pdf_partials_{dump.stem}.png", dpi=150)
    plt.close(fig)

    with open(outdir / f"pdf_{dump.stem}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        pairs = sorted(part_d)
        dG = {p: reduced(r, part_d[p], rho_d) - reduced(r, part_p[p], rho_p)
              for p in pairs}
        writer.writerow(["r_A", "G_damaged", "G_pristine"]
                        + ["dG_" + "-".join(map(str, p)) for p in pairs])
        for i in range(len(r)):
            writer.writerow([f"{r[i]:.4f}", f"{G_d[i]:.6f}", f"{G_p[i]:.6f}"]
                            + [f"{dG[p][i]:.6f}" for p in pairs])
    return png


def main():
    parser = argparse.ArgumentParser(
        description="Reduced PDF G(r), damaged vs pristine, from dumps.")
    parser.add_argument("dumps", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, required=True,
                        help="pristine lattice (perfect_lattice_*.data)")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--cutoff", type=float, default=12.0)
    parser.add_argument("--bins", type=int, default=600)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for dump in args.dumps:
        print(plot_pdf(dump, args.reference, args.frame, args.cutoff,
                       args.bins, args.outdir))


if __name__ == "__main__":
    main()
