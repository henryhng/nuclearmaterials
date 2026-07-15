#!/usr/bin/env python3
"""Electronic-stopping tables for LAMMPS fix electron/stopping.

    python3 elstop.py SiC > ../lammps/elstop_SiC.txt
    python3 elstop.py Fe  > ../lammps/elstop_Fe.txt
    python3 elstop.py W   > ../lammps/elstop_W.txt
"""

import argparse
import sys

import numpy as np

A0 = 0.529177         # Bohr radius, Ang
E2 = 14.3996          # e^2/(4*pi*eps0), eV*Ang
MVV2E = 1.0364269e-4  # (g/mol)(Ang/ps)^2 -> eV

SPECIES = {"Si": (14, 28.085), "C": (6, 12.011),
           "Fe": (26, 55.845), "W": (74, 183.84)}

N_SIC = 4 / 4.3596**3

# gamma_s [g/mol/ps]: Fe Zarkadoula JPCM 26 085401; W Front Phys 13 1592186.
MATERIALS = {
    "SiC": {"types": ["Si", "C"],
            "target": [("Si", N_SIC), ("C", N_SIC)]},
    "Fe": {"types": ["Fe"], "gamma": {"Fe": 55.845}},
    "W": {"types": ["W"], "gamma": {"W": 203.829}},
}


def velocity(mass, energy_ev):
    return np.sqrt(2.0 * energy_ev / (mass * MVV2E))


def dedx_pair(z1, m1, z2, m2, n_tgt, energy_ev):
    """LSS electronic dE/dx [eV/Ang] for one ion-target pair."""
    a = 0.8853 * A0 / (z1 ** (2 / 3) + z2 ** (2 / 3)) ** 0.5
    c = (a / (z1 * z2 * E2)) * (m2 / (m1 + m2))    # reduced energy eps = c*E
    ke = (z1 ** (1 / 6) * 0.0793 * z1 ** 0.5 * z2 ** 0.5 * (m1 + m2) ** 1.5
          / ((z1 ** (2 / 3) + z2 ** (2 / 3)) ** 0.75 * m1 ** 1.5 * m2 ** 0.5))
    prefac = n_tgt * 4 * np.pi * a**2 * m1 * m2 / (m1 + m2) ** 2
    return ke * np.sqrt(energy_ev / c) * prefac


def dedx(material, type_name, energy_ev):
    mat = MATERIALS[material]
    if "gamma" in mat:
        _, m1 = SPECIES[type_name]
        return mat["gamma"][type_name] * velocity(m1, energy_ev) * MVV2E
    z1, m1 = SPECIES[type_name]
    total = np.zeros_like(energy_ev)
    for tgt_name, n_tgt in mat["target"]:
        z2, m2 = SPECIES[tgt_name]
        total += dedx_pair(z1, m1, z2, m2, n_tgt, energy_ev)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Electronic-stopping table -> stdout.")
    parser.add_argument("material", choices=sorted(MATERIALS))
    args = parser.parse_args()

    types = MATERIALS[args.material]["types"]
    energies = np.logspace(np.log10(10.0), np.log10(2.5e4), 200)
    cols = [dedx(args.material, t, energies) for t in types]

    model = ("SRIM-derived friction"
             if "gamma" in MATERIALS[args.material] else "Lindhard-Scharff")
    header = "  ".join(f"dE/dx_{t}[eV/Ang]" for t in types)
    print(f"# {args.material} electronic stopping ({model})")
    print(f"# columns: energy[eV]  {header}")
    for i, e in enumerate(energies):
        row = " ".join(f"{c[i]:12.6f}" for c in cols)
        print(f"{e:12.4f} {row}")

    e10 = np.array([1.0e4])
    at10 = ", ".join(
        f"{t} {dedx(args.material, t, e10)[0]:.2f}" for t in types)
    print(f"# dE/dx at 10 keV: {at10} eV/Ang", file=sys.stderr)


if __name__ == "__main__":
    main()
