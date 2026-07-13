#!/usr/bin/env python3
"""Batch Wigner-Seitz defect analysis over cascade dumps -> results.csv.

Per run: surviving Frenkel pairs, vacancy/interstitial cluster distributions,
and (with --material) dpa-normalized survival efficiency and a dispersed-
barrier hardening estimate.

    python3 defects.py traj_SiC_relax_*.lammpstrj \
        --reference perfect_lattice_SiC.data --material Fe --out results.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

from ovito.io import import_file
from ovito.modifiers import (WignerSeitzAnalysisModifier,
                             ExpressionSelectionModifier,
                             ClusterAnalysisModifier)
from ovito.pipeline import FileSource

import materials


def run_label(dump_path: Path) -> str:
    m = re.search(r"_(\d+)\.lammpstrj$", dump_path.name)
    return m.group(1) if m else dump_path.stem


def _ws(reference: Path, displaced: bool) -> WignerSeitzAnalysisModifier:
    ws = WignerSeitzAnalysisModifier(per_type_occupancies=True,
                                     output_displaced=displaced)
    ws.reference = FileSource()
    ws.reference.load(str(reference))
    return ws


def cluster_sizes(dump_path: Path, reference: Path, frame: int, cutoff: float,
                  displaced: bool, expr: str):
    """Cluster sizes in defect units plus the cell volume [A^3].

    Displaced config selects every atom of a multi-occupied site, so a
    lone dumbbell is 2 atoms; weight each atom (occ-1)/occ to count
    defects, not atoms.
    """
    pipeline = import_file(str(dump_path))
    pipeline.modifiers.append(_ws(reference, displaced))
    pipeline.modifiers.append(ExpressionSelectionModifier(expression=expr))
    pipeline.modifiers.append(ClusterAnalysisModifier(
        cutoff=cutoff, only_selected=True, sort_by_size=True))
    data = pipeline.compute(frame)
    n = data.attributes["ClusterAnalysis.cluster_count"]
    if n == 0:
        return np.array([], dtype=int), data.cell.volume
    if not displaced:
        sizes = np.asarray(data.tables["clusters"]["Cluster Size"])
        return sizes.astype(int), data.cell.volume
    cid = np.asarray(data.particles["Cluster"])
    occ = np.asarray(data.particles["Occupancy"])
    if occ.ndim > 1:
        occ = occ.sum(axis=1)
    w = (occ - 1) / occ
    sizes = np.bincount(cid[cid > 0], weights=w[cid > 0])[1:]
    sizes = np.rint(sizes[sizes > 0.49]).astype(int)
    return np.maximum(sizes, 1), data.cell.volume


def cluster_stats(sizes: np.ndarray, prefix: str) -> dict:
    """Count, largest, mono fraction, and clustered (size>=2) fraction."""
    total = int(sizes.sum())
    grouped = sizes[sizes >= 2]
    return {
        f"{prefix}_clusters": int(grouped.size),
        f"{prefix}_largest": int(sizes.max()) if sizes.size else 0,
        f"{prefix}_mono_frac": (float((sizes == 1).sum()) / total
                                if total else 0.0),
        f"{prefix}_clustered_frac": (float(grouped.sum()) / total
                                     if total else 0.0),
    }


def analyze(dump_path: Path, reference: Path, frame: int, mat: dict | None,
            pka_ev: float) -> dict:
    cutoff = mat["cutoff"] if mat else 3.5

    pipeline = import_file(str(dump_path))
    pipeline.modifiers.append(_ws(reference, displaced=False))
    nframes = pipeline.source.num_frames
    if frame < 0:
        frame = nframes + frame
    data = pipeline.compute(frame)

    occupancy = np.asarray(data.particles["Occupancy"])
    if occupancy.ndim == 1:
        occupancy = occupancy[:, np.newaxis]
    site_types = np.asarray(data.particles["Particle Type"])
    total = occupancy.sum(axis=1)

    vacancies = int(np.count_nonzero(total == 0))
    interstitials = int(data.attributes["WignerSeitz.interstitial_count"])
    frenkel = min(vacancies, interstitials)

    # Antisite: singly occupied site, wrong species.
    own = occupancy[np.arange(len(site_types)), site_types - 1]
    anti_mask = (total == 1) & (own == 0)
    type_names = {t.id: t.name or str(t.id)
                  for t in data.particles["Particle Type"].types}
    antisites = {}
    for tid, name in sorted(type_names.items()):
        n = int(np.count_nonzero(anti_mask & (site_types == tid)))
        antisites[f"antisites_on_{name}_sites"] = n

    # Per-species split: vacancies by sublattice; interstitials are the
    # extras at multi-occupied sites after one resident (own species if
    # present, else the most abundant) is removed.
    extras = occupancy.copy()
    resident = np.where(own >= 1, site_types - 1, occupancy.argmax(axis=1))
    extras[np.arange(len(site_types)), resident] -= (total > 0)
    per_species = {}
    for tid, name in sorted(type_names.items()):
        per_species[f"vac_{name}"] = int(
            np.count_nonzero((total == 0) & (site_types == tid)))
        per_species[f"int_{name}"] = int(extras[total > 1, tid - 1].sum())

    # Occupancy is per-type-component when the cell has >1 species.
    ntypes = occupancy.shape[1]
    occ = ("Occupancy" if ntypes == 1 else
           "(" + "+".join(f"Occupancy.{i}" for i in range(1, ntypes + 1)) + ")")
    vac_sizes, vol = cluster_sizes(dump_path, reference, frame, cutoff,
                                   displaced=False, expr=f"{occ}==0")
    int_sizes, _ = cluster_sizes(dump_path, reference, frame, cutoff,
                                 displaced=True, expr=f"{occ}>1")

    row = {
        "run": run_label(dump_path),
        "dump": dump_path.name,
        "frame": frame,
        "n_sites": len(site_types),
        "vacancies": vacancies,
        "interstitials": interstitials,
        "frenkel_pairs": frenkel,
        **antisites,
        **per_species,
        **cluster_stats(vac_sizes, "vac"),
        **cluster_stats(int_sizes, "int"),
    }
    if mat:
        row.update(damage_row(frenkel, int_sizes, vol, mat, pka_ev))
    return row


def damage_row(frenkel: int, int_sizes: np.ndarray, vol_a3: float,
               mat: dict, pka_ev: float) -> dict:
    """Dose-normalized survival efficiency and dispersed-barrier hardening."""
    dpa = materials.dpa_metrics(pka_ev, mat)
    nrt = dpa["nrt_displacements"]
    arc = dpa["arc_displacements"]
    row = {
        "nrt_displacements": round(nrt, 2),
        "survival_eff_nrt": frenkel / nrt if nrt else 0.0,
        "dpa_per_cascade": nrt / (mat["atoms_cell"]
                                  * vol_a3 / mat["a0"] ** 3),
    }
    if arc:
        row["arc_displacements"] = round(arc, 2)
        row["survival_eff_arc"] = frenkel / arc

    # As-simulated cascade-debris hardening: obstacle field is this cascade's
    # loops in the cell volume, not a saturated dose. Density/diameter are the
    # primary quantities; hardening_MPa is the dispersed-barrier estimate.
    if mat["metal"]:
        loops = int_sizes[int_sizes >= 2]
        if loops.size:
            vol_atom = materials.atomic_volume(mat)
            density = loops.size / (vol_a3 * 1e-30)
            diam = materials.cluster_diameter(loops.mean(), vol_atom)
            row["int_density_m3"] = f"{density:.3e}"
            row["int_loop_diam_nm"] = round(diam * 1e9, 3)
            row["hardening_MPa"] = round(
                materials.dispersed_barrier(density, diam, mat), 1)
        else:
            row["int_density_m3"] = "0.000e+00"
            row["int_loop_diam_nm"] = 0.0
            row["hardening_MPa"] = 0.0
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Wigner-Seitz defect + cluster + dpa analysis -> CSV.")
    parser.add_argument("dumps", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, required=True,
                        help="ideal lattice (perfect_lattice_SiC.data)")
    parser.add_argument("--material", choices=sorted(materials.MATERIALS),
                        help="enable dpa normalization and hardening")
    parser.add_argument("--pka-ev", type=float, default=10000.0,
                        help="PKA energy for dpa normalization")
    parser.add_argument("--frame", type=int, default=-1,
                        help="frame to analyze (default: last)")
    parser.add_argument("--out", type=Path, default=Path("results.csv"))
    args = parser.parse_args()

    mat = materials.get(args.material) if args.material else None
    rows = [analyze(d, args.reference, args.frame, mat, args.pka_ev)
            for d in sorted(args.dumps)]

    fields = list(rows[0].keys())
    for r in rows:                        # union in case runs differ
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fp = np.array([r["frenkel_pairs"] for r in rows], dtype=float)
    std = fp.std(ddof=1) if len(fp) > 1 else 0.0
    print(f"Wrote {args.out} ({len(rows)} runs)")
    print(f"Frenkel pairs: mean {fp.mean():.1f}  std {std:.1f}  "
          f"min {fp.min():.0f}  max {fp.max():.0f}")
    if mat and "survival_eff_nrt" in rows[0]:
        eff = np.array([r["survival_eff_nrt"] for r in rows])
        print(f"NRT survival efficiency: mean {eff.mean():.3f}")
        if "hardening_MPa" in rows[0]:
            hard = np.array([r["hardening_MPa"] for r in rows])
            print(f"Hardening: mean {hard.mean():.1f} MPa")


if __name__ == "__main__":
    main()
