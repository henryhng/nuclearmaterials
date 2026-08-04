"""Microstructure metrics for BH-2 micrographs.

Usage: python analysis.py [--images DIR] [--out DIR]
"""

import argparse
import csv
import pathlib
import re

import numpy as np
from scipy import spatial
from skimage import color, filters, io, measure, morphology

# camera calibration
UM_PER_PX = {"IC5": 0.3247, "IC10": 0.1610, "IC50": 0.03256, "IC100": 0.01618}
MAG_LABEL = {"IC5": "5x", "IC10": "10x", "IC50": "50x", "IC100": "100x"}

# analysis tier per objective
TIER = {"IC5": "global", "IC10": "regional", "IC50": "particle"}

MIN_FEATURE_UM2 = 2.0
GRID_N = 20
HULL_TOP_N = 500

# defect classes
PORE_ASPECT_MAX = 2.0
PORE_CIRC_MIN = 0.5
FIBER_ASPECT_MIN = 3.0
ALIGN_ASPECT_MIN = 2.0

RDF_MAG = "IC10"
RDF_RMAX_UM = 100.0
RDF_NBINS = 50

HI_GRIDS = [10, 20, 40, 80]


# Discovery and segmentation

def find_images(images_dir):
    records = []
    for path in sorted(images_dir.rglob("*.png")):
        match = re.search(r"(Image_?\d+)_(IC\d+)", path.stem)
        if match is None:
            continue
        image_id = match.group(1).replace("_", "")
        records.append((path.parent.name, image_id, match.group(2), path))
    return records


def flatten_illumination(gray):
    sigma = max(gray.shape) / 8
    background = filters.gaussian(gray, sigma=sigma)
    flat = gray / np.maximum(background, 1e-6)
    return flat / flat.max()


def segment_phases(gray, um_per_px):
    flat = flatten_illumination(gray)
    t1, t2 = filters.threshold_multiotsu(flat, classes=3)

    dark = flat < t1
    gray_phase = (flat >= t1) & (flat < t2)

    # unimodal guard
    if dark.mean() > 0.35:
        dark = flat < np.quantile(flat, 0.05)

    min_px = int(MIN_FEATURE_UM2 / um_per_px**2)
    dark = morphology.remove_small_objects(dark, max_size=max(min_px, 4))
    return dark, gray_phase


def local_fraction_map(mask, n=GRID_N):
    h, w = mask.shape
    ys = np.linspace(0, h, n + 1, dtype=int)
    xs = np.linspace(0, w, n + 1, dtype=int)
    return np.array([
        [mask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(n)]
        for i in range(n)
    ])


# Per-feature arrays

def extract_features(props, um_per_px):
    areas_um2 = np.array([p.area for p in props]) * um_per_px**2
    perims = np.array([p.perimeter for p in props]) * um_per_px
    axis_major = np.array([p.axis_major_length for p in props])

    with np.errstate(divide="ignore", invalid="ignore"):
        circ = np.where(perims > 0, 4 * np.pi * areas_um2 / perims**2, 0)

    aspect = np.array([
        p.axis_major_length / p.axis_minor_length
        if p.axis_minor_length > 0 else 1
        for p in props
    ])

    hull_idx = np.unique(np.concatenate([
        np.argsort(areas_um2)[::-1][:HULL_TOP_N],
        np.argsort(axis_major)[::-1][:HULL_TOP_N],
    ])) if len(props) else np.array([], dtype=int)

    feret_max = 0.0
    if len(hull_idx):
        feret_max = (max(props[i].feret_diameter_max for i in hull_idx)
                     * um_per_px)

    return {
        "areas_um2": areas_um2,
        "eq_d": np.array([p.equivalent_diameter_area
                          for p in props]) * um_per_px,
        "major": axis_major * um_per_px,
        "circ": circ,
        "aspect": aspect,
        "orient": np.array([p.orientation for p in props]),
        "centroids": np.array([p.centroid for p in props]) * um_per_px
                     if props else np.empty((0, 2)),
        "feret_max_um": feret_max,
    }


# Metrics

def morphology_stats(feats, mask, um_per_px):
    eq_d, areas = feats["eq_d"], feats["areas_um2"]
    area_mm2 = mask.size * (um_per_px * 1e-3) ** 2

    pct = {f"d{q}_um": np.percentile(eq_d, q) if len(eq_d) else 0.0
           for q in (10, 25, 50, 75, 90)}

    return {
        "dark_area_pct": 100 * mask.mean(),
        "n_features": len(eq_d),
        "density_per_mm2": len(eq_d) / area_mm2,
        **pct,
        "diam_area_weighted_um":
            (eq_d * areas).sum() / areas.sum() if areas.sum() else 0.0,
        "diam_max_um": eq_d.max() if len(eq_d) else 0.0,
        "feret_max_um": feats["feret_max_um"],
        "aspect_median": np.median(feats["aspect"]) if len(eq_d) else 0.0,
    }


def defect_stats(feats, mask, um_per_px):
    areas, aspect, circ = feats["areas_um2"], feats["aspect"], feats["circ"]
    area_mm2 = mask.size * (um_per_px * 1e-3) ** 2
    image_area_um2 = mask.size * um_per_px**2

    pores = (aspect < PORE_ASPECT_MAX) & (circ > PORE_CIRC_MIN)
    fibers = aspect >= FIBER_ASPECT_MIN

    # alignment order parameter
    alignment = 0.0
    sel = aspect >= ALIGN_ASPECT_MIN
    if sel.sum() >= 5:
        alignment = np.abs(np.exp(2j * feats["orient"][sel]).mean())

    return {
        "pore_density_per_mm2": pores.sum() / area_mm2,
        "pore_area_pct": 100 * areas[pores].sum() / image_area_um2,
        "pullout_area_fraction":
            areas[fibers].sum() / areas.sum() if areas.sum() else 0.0,
        "alignment_index": alignment,
    }, pores, fibers


def pair_correlation(pts, height_um, width_um, rmax, nbins):
    inner = pts[(pts[:, 0] > rmax) & (pts[:, 0] < height_um - rmax)
                & (pts[:, 1] > rmax) & (pts[:, 1] < width_um - rmax)]
    if len(inner) < 20 or len(pts) < 50:
        return None, None

    tree = spatial.cKDTree(pts)
    edges = np.linspace(0, rmax, nbins + 1)
    counts = np.zeros(nbins)
    for p in inner:
        idx = tree.query_ball_point(p, rmax)
        d = np.linalg.norm(pts[idx] - p, axis=1)
        counts += np.histogram(d[d > 1e-9], bins=edges)[0]

    density = len(pts) / (height_um * width_um)
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    g = counts / (len(inner) * density * shell)
    r = 0.5 * (edges[1:] + edges[:-1])
    return r, g


def percentile_row(d):
    qs = [10, 25, 50, 75, 90]
    if not len(d):
        return {f"d{q}": 0.0 for q in qs} | {"dmax": 0.0}
    return {f"d{q}": np.percentile(d, q) for q in qs} | {"dmax": d.max()}


# Outputs

def write_rdf(rdf, samples, out_path):
    cols = [s for s in samples if s in rdf]
    means = {s: np.mean(rdf[s], axis=0) for s in cols}

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_um"] + cols)
        for i, r in enumerate(rdf["r"]):
            w.writerow([f"{r:.2f}"] + [f"{means[s][i]:.4f}" for s in cols])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="images", type=pathlib.Path)
    parser.add_argument("--out", default="results", type=pathlib.Path)
    args = parser.parse_args()

    extras = args.out / "extras"
    extras.mkdir(parents=True, exist_ok=True)

    records = find_images(args.images)
    print(f"found {len(records)} images")

    rows, class_rows, rdf = [], [], {}

    for sample, image_id, mag, path in records:
        um = UM_PER_PX[mag]
        gray = color.rgb2gray(io.imread(path))

        mask, gray_phase = segment_phases(gray, um)
        feats = extract_features(measure.regionprops(measure.label(mask)), um)
        defect_row, pores, fibers = defect_stats(feats, mask, um)

        row = {"sample": sample, "image": image_id, "mag": mag,
               "tier": TIER[mag], "um_per_px": um,
               "gray_phase_pct": 100 * gray_phase.mean(),
               **morphology_stats(feats, mask, um), **defect_row}
        rows.append(row)

        if mag == "IC50":
            for cls, sel, size in [("pore", pores, feats["eq_d"]),
                                   ("fiber", fibers, feats["major"])]:
                class_rows.append({"sample": sample, "image": image_id,
                                   "class": cls, "n": int(sel.sum()),
                                   **percentile_row(size[sel])})

        if mag == RDF_MAG:
            h_um, w_um = mask.shape[0] * um, mask.shape[1] * um
            r, g = pair_correlation(feats["centroids"], h_um, w_um,
                                    RDF_RMAX_UM, RDF_NBINS)
            if r is not None:
                rdf.setdefault(sample, []).append(g)
                rdf["r"] = r

        print(f"{sample:26s} {image_id:7s} {MAG_LABEL[mag]:>4s} "
              f"dark={row['dark_area_pct']:5.2f}%  n={row['n_features']:5d}  "
              f"pullout={row['pullout_area_fraction']:.2f}", flush=True)

    samples = sorted({r["sample"] for r in rows})

    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(extras / "sizes_50x.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(class_rows[0]))
        w.writeheader()
        w.writerows(class_rows)

    if "r" in rdf:
        write_rdf(rdf, samples, extras / "rdf_10x.csv")

    print(f"wrote {args.out / 'summary.csv'} and extras")


if __name__ == "__main__":
    main()
