#!/usr/bin/env python3
"""Per-sublattice stored-energy split from 0 K per-atom PE dumps.

Excess energy (damaged minus per-species pristine mean) is attributed to
the nearest Wigner-Seitz defect: which species' defects store the energy
(Sprouster PRM 5, 2021: carbon-dominated in SiC).

    python3 energy_split.py --damaged min_SiC_cascade1.lammpstrj \
        --pristine min_SiC_equil.lammpstrj \
        --reference perfect_lattice_SiC.data
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.pipeline import FileSource


def atoms(dump: Path):
    """Positions, types, per-atom PE, cell lengths, type names."""
    data = import_file(str(dump)).compute()
    pe = data.particles["c_pe"]
    names = {t.id: t.name or str(t.id)
             for t in data.particles["Particle Type"].types}
    return (np.asarray(data.particles.positions),
            np.asarray(data.particles["Particle Type"]),
            np.asarray(pe),
            np.diag(np.asarray(data.cell.matrix)[:, :3]),
            names)


def defect_centers(dump: Path, reference: Path):
    """(position, species id, kind) for vacancies, interstitials, antisites."""
    ws = WignerSeitzAnalysisModifier(per_type_occupancies=True,
                                     output_displaced=False)
    ws.reference = FileSource()
    ws.reference.load(str(reference))
    pipeline = import_file(str(dump))
    pipeline.modifiers.append(ws)
    data = pipeline.compute()

    occ = np.asarray(data.particles["Occupancy"])
    site_types = np.asarray(data.particles["Particle Type"])
    site_pos = np.asarray(data.particles.positions)
    total = occ.sum(axis=1)
    own = occ[np.arange(len(site_types)), site_types - 1]

    centers, species, kinds = [], [], []
    for mask, kind, spec in (
            (total == 0, "vac", site_types),
            ((total == 1) & (own == 0), "anti", occ.argmax(axis=1) + 1)):
        centers.append(site_pos[mask])
        species.append(spec[mask])
        kinds += [kind] * int(mask.sum())

    # Interstitials: atoms sitting at multi-occupied sites.
    disp = WignerSeitzAnalysisModifier(per_type_occupancies=False,
                                       output_displaced=True)
    disp.reference = FileSource()
    disp.reference.load(str(reference))
    pipeline = import_file(str(dump))
    pipeline.modifiers.append(disp)
    d = pipeline.compute()
    multi = np.asarray(d.particles["Occupancy"]) > 1
    centers.append(np.asarray(d.particles.positions)[multi])
    species.append(np.asarray(d.particles["Particle Type"])[multi])
    kinds += ["int"] * int(multi.sum())

    return np.vstack(centers), np.concatenate(species), np.array(kinds)


def main():
    parser = argparse.ArgumentParser(
        description="Attribute 0 K stored energy to defects by species.")
    parser.add_argument("--damaged", type=Path, required=True,
                        help="minimized post-cascade dump with c_pe")
    parser.add_argument("--pristine", type=Path, required=True,
                        help="minimized equilibrium dump with c_pe")
    parser.add_argument("--reference", type=Path, required=True,
                        help="ideal lattice for Wigner-Seitz")
    parser.add_argument("--rcut", type=float, default=6.0,
                        help="attribution radius around defects")
    args = parser.parse_args()

    _, ptypes, ppe, _, names = atoms(args.pristine)
    mu = {t: ppe[ptypes == t].mean() for t in np.unique(ptypes)}

    pos, types, pe, box, _ = atoms(args.damaged)
    excess = pe - np.vectorize(mu.get)(types)
    total = excess.sum()

    centers, spec, kinds = defect_centers(args.damaged, args.reference)
    tree = cKDTree(np.mod(centers, box), boxsize=box)
    dist, idx = tree.query(np.mod(pos, box), k=1)
    near = dist <= args.rcut

    print(f"stored 0K total: {total:.1f} eV   "
          f"attributed within {args.rcut} A: {excess[near].sum():.1f} eV  "
          f"({100 * excess[near].sum() / total:.1f}%)")
    print(f"defects: {len(centers)}  "
          + "  ".join(f"{k}:{(kinds == k).sum()}" for k in ("vac", "int",
                                                            "anti")))
    for t in sorted(mu):
        name = names.get(t, str(t))
        m = near & np.isin(idx, np.nonzero(spec == t)[0])
        e = excess[m].sum()
        print(f"{name}-defect share: {e:8.1f} eV  ({100 * e / total:5.1f}% "
              f"of total)")
        for kind in ("vac", "int", "anti"):
            mk = near & np.isin(idx, np.nonzero((spec == t)
                                                & (kinds == kind))[0])
            print(f"    {kind:<5} {excess[mk].sum():8.1f} eV")


if __name__ == "__main__":
    main()
