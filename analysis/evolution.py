#!/usr/bin/env python3
"""Wigner-Seitz defect counts vs time for a cascade trajectory.

    python3 evolution.py traj_SiC_spike_1.lammpstrj \
        --reference perfect_lattice_SiC.data --out defects_run1.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.pipeline import FileSource


def evolution(dump_path: Path, reference: Path, every: int) -> dict:
    pipeline = import_file(str(dump_path))
    ws = WignerSeitzAnalysisModifier(per_type_occupancies=True)
    ws.reference = FileSource()
    ws.reference.load(str(reference))
    pipeline.modifiers.append(ws)

    steps, vac, inter = [], [], []
    for frame in range(0, pipeline.source.num_frames, every):
        data = pipeline.compute(frame)
        steps.append(int(data.attributes.get("Timestep", frame)))
        vac.append(int(data.attributes["WignerSeitz.vacancy_count"]))
        inter.append(int(data.attributes["WignerSeitz.interstitial_count"]))
    return {"step": np.array(steps), "vacancies": np.array(vac),
            "interstitials": np.array(inter)}


def main():
    parser = argparse.ArgumentParser(
        description="Defect count vs time (Wigner-Seitz).")
    parser.add_argument("dumps", nargs="+", type=Path,
                        help="trajectory dumps, plotted in sequence")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1,
                        help="analyze every Nth frame")
    parser.add_argument("--out", type=Path, default=Path("defects_vs_time.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for dump in args.dumps:
        d = evolution(dump, args.reference, args.every)
        ax.plot(d["step"], d["vacancies"], "-o", ms=3, label="vacancies")
        ax.plot(d["step"], d["interstitials"], "-s", ms=3,
                label="interstitials")
        frenkel = np.minimum(d["vacancies"], d["interstitials"])
        peak, final = frenkel.max(), frenkel[-1]
        print(f"{dump.name}: peak Frenkel {peak}, surviving {final}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Defect count")
    ax.set_title("Cascade defect evolution (Wigner-Seitz)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
