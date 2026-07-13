#!/usr/bin/env python3
"""Per-material constants and displacement-damage conversions.

Turns raw cascade defect counts into dose-normalized, comparable quantities:
NRT / arc-dpa displacements and a dispersed-barrier hardening estimate.
"""

import math

# Z, A: recoil atomic number / mass. e_d: threshold displacement energy [eV].
# a0 [A], atoms_cell: lattice for atomic volume. arc_b/arc_c: arc-dpa
# efficiency fit (Nordlund et al., Nat. Commun. 9, 1084, 2018).
# shear [Pa], burgers [m], taylor, alpha: dispersed-barrier hardening.
# cutoff [A]: neighbor distance for defect clustering. metal: glide hardening.
# nn [A]: nearest-neighbor distance (render displacement threshold).
MATERIALS = {
    "Fe": {"Z": 26, "A": 55.845, "e_d": 40.0, "a0": 2.855, "atoms_cell": 2,
           "arc_b": -0.568, "arc_c": 0.286,
           "shear": 82e9, "burgers": 0.2473e-9, "taylor": 3.06, "alpha": 0.4,
           "cutoff": 3.7, "nn": 2.47, "metal": True},
    "W": {"Z": 74, "A": 183.84, "e_d": 90.0, "a0": 3.165, "atoms_cell": 2,
          "arc_b": -0.564, "arc_c": 0.119,
          "shear": 161e9, "burgers": 0.2741e-9, "taylor": 3.06, "alpha": 0.4,
          "cutoff": 4.1, "nn": 2.74, "metal": True},
    # Compound: composition-averaged Z/A, dpa is approximate; covalent
    # ceramic so metal-glide hardening is not applied. e_d per sublattice
    # (convention values; potential-consistent averages in ed_results_*.csv).
    "SiC": {"Z": 10.0, "A": 20.05, "e_d": {"Si": 35.0, "C": 20.0},
            "a0": 4.36, "atoms_cell": 8,
            "arc_b": None, "arc_c": None,
            "shear": 192e9, "burgers": 0.308e-9, "taylor": 3.06, "alpha": 0.4,
            "cutoff": 3.0, "nn": 1.89, "metal": False},
}


def get(material: str) -> dict | None:
    return MATERIALS.get(material)


def effective_e_d(mat: dict) -> float:
    """Scalar e_d, or stoichiometric harmonic mean for compounds."""
    ed = mat["e_d"]
    if isinstance(ed, dict):
        return len(ed) / sum(1.0 / e for e in ed.values())
    return ed


def atomic_volume(mat: dict) -> float:
    """Volume per atom [m^3]."""
    return (mat["a0"] * 1e-10) ** 3 / mat["atoms_cell"]


def damage_energy(pka_ev: float, Z: float, A: float) -> float:
    """Nuclear (damage) energy after Lindhard electronic-loss partition [eV]."""
    e_l = 30.724 * Z * Z * (2 * Z ** (2 / 3)) ** 0.5 * (2 * A) / A
    eps = pka_ev / e_l
    k = 0.1337 * Z ** (2 / 3) / A ** 0.5
    g = 3.4008 * eps ** (1 / 6) + 0.40244 * eps ** (3 / 4) + eps
    return pka_ev / (1 + k * g)


def nrt_displacements(t_dm: float, e_d: float) -> float:
    """Norgett-Robinson-Torrens displacement count."""
    return 0.8 * t_dm / (2 * e_d)


def arc_efficiency(t_dm: float, e_d: float, b, c) -> float | None:
    """Athermal-recombination-corrected surviving fraction of NRT."""
    if b is None or c is None:
        return None
    return (1 - c) * (t_dm / (2 * e_d / 0.8)) ** b + c


def cluster_diameter(n_atoms: int, vol_atom: float) -> float:
    """Spherical-equivalent diameter of an n-atom cluster [m]."""
    return 2 * (3 * n_atoms * vol_atom / (4 * math.pi)) ** (1 / 3)


def dispersed_barrier(density: float, diameter: float, mat: dict) -> float:
    """Irradiation hardening increment [MPa] from obstacle N and size d."""
    dsigma = (mat["taylor"] * mat["alpha"] * mat["shear"] * mat["burgers"]
              * (density * diameter) ** 0.5)
    return dsigma / 1e6


def dpa_metrics(pka_ev: float, mat: dict) -> dict:
    """NRT / arc displacement counts for one PKA in this material."""
    e_d = effective_e_d(mat)
    t_dm = damage_energy(pka_ev, mat["Z"], mat["A"])
    nrt = nrt_displacements(t_dm, e_d)
    xi = arc_efficiency(t_dm, e_d, mat["arc_b"], mat["arc_c"])
    arc = nrt * xi if xi is not None else None
    return {"damage_energy_eV": t_dm, "nrt_displacements": nrt,
            "arc_efficiency": xi, "arc_displacements": arc}
