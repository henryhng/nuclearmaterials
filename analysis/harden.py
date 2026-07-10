#!/usr/bin/env python3
"""Splice ZBL short-range repulsion into a single-element EAM setfl table.

    python3 harden.py ../archive/lammps/Fe_mm.eam.fs \
        ../lammps/Fe_mm_zbl.eam.fs --z 26
"""

import argparse
from pathlib import Path

import numpy as np

E2 = 14.3996
ZBL_C = (0.18175, 0.50986, 0.28022, 0.02817)
ZBL_D = (3.19980, 0.94229, 0.40290, 0.20162)


def zbl(z1, z2, r):
    a = 0.46850 / (z1**0.23 + z2**0.23)
    x = r / a
    phi = sum(c * np.exp(-d * x) for c, d in zip(ZBL_C, ZBL_D))
    return z1 * z2 * E2 / r * phi


def switch(r, r1, r2):
    """1 -> ZBL below r1, 0 -> EAM above r2, cosine blend between."""
    x = np.clip((r - r1) / (r2 - r1), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def main():
    parser = argparse.ArgumentParser(
        description="ZBL-harden a single-element setfl pair function.")
    parser.add_argument("infile", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--r1", type=float, default=1.0)
    parser.add_argument("--r2", type=float, default=2.0)
    args = parser.parse_args()

    lines = args.infile.read_text().splitlines()
    header, tokens = lines[:4], []
    for line in lines[4:]:
        tokens.extend(line.split())

    nrho, drho, nr, dr, cutoff = (
        int(tokens[0]), float(tokens[1]), int(tokens[2]), float(tokens[3]),
        float(tokens[4]))
    elem_header = tokens[5:9]
    body = np.array(tokens[9:], dtype=float)

    npair_off = nrho + nr           # single element: one density function
    assert body.size == npair_off + nr, "not a single-element setfl file"

    r = np.arange(nr) * dr
    rphi = body[npair_off:]
    phi = np.empty_like(rphi)
    phi[1:] = rphi[1:] / r[1:]
    phi[0] = 0.0

    f = switch(r[1:], args.r1, args.r2)
    phi_new = phi.copy()
    phi_new[1:] = f * zbl(args.z, args.z, r[1:]) + (1.0 - f) * phi[1:]
    body[npair_off:] = np.concatenate(([0.0], phi_new[1:] * r[1:]))

    out = header + [
        f"{nrho} {drho:.16e} {nr} {dr:.16e} {cutoff:.10f}",
        " ".join(elem_header),
    ]
    out.extend(f"{v:.16e}" for v in body)
    args.outfile.write_text("\n".join(out) + "\n")
    print(f"{args.outfile}: ZBL below {args.r1} A, blend to {args.r2} A")


if __name__ == "__main__":
    main()
