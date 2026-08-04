"""Pore and carbon-fiber segmentation for wk2 micrographs.

Usage: python segment.py [--images DIR] [--out DIR]
"""

import argparse
import csv
import pathlib
import re
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import (color, exposure, filters, io, measure, morphology,
                     segmentation, transform, util)

# camera calibration
UM_PER_PX = {"IC5": 0.3247, "IC10": 0.1610, "IC50": 0.03256, "IC100": 0.01618}
MAG_LABEL = {"IC5": "5x", "IC10": "10x", "IC50": "50x", "IC100": "100x"}

IMAGE_RE = re.compile(r"(Image_?\d+)_(IC\d+)")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# classification thresholds
PORE_ASPECT_MAX = 2.0
PORE_CIRC_MIN = 0.50
FIBER_ASPECT_MIN = 3.0
ALIGN_ASPECT_MIN = 2.0

DEFAULT_MIN_FEATURE_UM2 = 2.0
OVERLAY_DOWNSCALE = 0.35


# Discovery and io

@dataclass(frozen=True)
class ImageRecord:
    sample: str
    image: str
    mag: str
    path: pathlib.Path


def find_images(images_dir: pathlib.Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(images_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = IMAGE_RE.search(path.stem)
        if match is None:
            continue
        image_id = match.group(1).replace("_", "")
        mag = match.group(2)
        if mag not in UM_PER_PX:
            continue
        records.append(ImageRecord(path.parent.name, image_id, mag, path))
    return records


def read_rgb(path: pathlib.Path) -> np.ndarray:
    image = io.imread(path)
    if image.ndim == 2:
        return color.gray2rgb(util.img_as_float(image))
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    return util.img_as_float(image[..., :3])


def image_stem(record: ImageRecord) -> str:
    return f"{record.sample}_{record.image}_{record.mag}"


# Segmentation

def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    factor = max(1, int(np.ceil(max(gray.shape) / 1200)))
    if factor > 1:
        small = transform.downscale_local_mean(gray, (factor, factor))
    else:
        small = gray

    sigma = max(small.shape) / 8.0
    background = filters.gaussian(small, sigma=sigma, preserve_range=True)
    if factor > 1:
        background = transform.resize(
            background,
            gray.shape,
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        )

    flat = gray / np.maximum(background, 1e-6)
    lo, hi = np.percentile(flat, (0.2, 99.8))
    flat = exposure.rescale_intensity(flat, in_range=(lo, hi),
                                      out_range=(0.0, 1.0))
    return np.clip(flat, 0.0, 1.0)


def clean_mask(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    mask = morphology.remove_small_objects(mask, max_size=max(min_area_px, 4))
    mask = morphology.remove_small_holes(mask,
                                         max_size=max(min_area_px // 2, 4))
    return ndi.binary_fill_holes(mask)


def segment_dark_and_gray(
    gray: np.ndarray, min_area_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat = flatten_illumination(gray)

    try:
        t1, t2 = filters.threshold_multiotsu(flat, classes=3)
    except ValueError:
        # flat image fallback
        t1, t2 = np.quantile(flat, (0.08, 0.35))

    dark = flat < t1
    gray_phase = (flat >= t1) & (flat < t2)

    # unimodal guard
    if dark.mean() > 0.35:
        dark = flat < np.quantile(flat, 0.08)

    dark = clean_mask(dark, min_area_px)
    gray_phase = morphology.remove_small_objects(gray_phase,
                                                max_size=max(min_area_px, 4))
    return dark, gray_phase


# Feature classification

def circularity(prop: measure._regionprops.RegionProperties,
                um_per_px: float) -> float:
    perimeter_um = prop.perimeter * um_per_px
    area_um2 = prop.area * um_per_px**2
    if perimeter_um <= 0:
        return 0.0
    return float(4.0 * np.pi * area_um2 / perimeter_um**2)


def aspect_ratio(prop: measure._regionprops.RegionProperties) -> float:
    if prop.axis_minor_length <= 0:
        return float("inf")
    return float(prop.axis_major_length / prop.axis_minor_length)


def classify_features(
    dark_mask: np.ndarray,
    um_per_px: float,
    min_pore_area_um2: float,
    min_fiber_length_um: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    labels = measure.label(dark_mask)
    pore_mask = np.zeros(dark_mask.shape, dtype=bool)
    fiber_mask = np.zeros(dark_mask.shape, dtype=bool)
    features: list[dict[str, float | int | str]] = []

    for prop in measure.regionprops(labels):
        area_um2 = float(prop.area * um_per_px**2)
        eq_d_um = float(prop.equivalent_diameter_area * um_per_px)
        major_um = float(prop.axis_major_length * um_per_px)
        minor_um = float(prop.axis_minor_length * um_per_px)
        aspect = aspect_ratio(prop)
        circ = circularity(prop, um_per_px)
        eccentricity = float(prop.eccentricity)

        is_pore = (
            area_um2 >= min_pore_area_um2
            and aspect <= PORE_ASPECT_MAX
            and circ >= PORE_CIRC_MIN
        )
        is_fiber = (
            aspect >= FIBER_ASPECT_MIN
            and major_um >= min_fiber_length_um
            and not is_pore
        )

        feature_class = "other_dark"
        if is_pore:
            feature_class = "pore"
            pore_mask[labels == prop.label] = True
        elif is_fiber:
            feature_class = "carbon_fiber"
            fiber_mask[labels == prop.label] = True

        y0, x0, y1, x1 = prop.bbox
        cy, cx = prop.centroid
        features.append({
            "label": int(prop.label),
            "class": feature_class,
            "area_um2": area_um2,
            "equivalent_diameter_um": eq_d_um,
            "major_axis_um": major_um,
            "minor_axis_um": minor_um,
            "aspect_ratio": aspect,
            "circularity": circ,
            "eccentricity": eccentricity,
            "orientation_rad": float(prop.orientation),
            "centroid_x_um": float(cx * um_per_px),
            "centroid_y_um": float(cy * um_per_px),
            "bbox_x0_px": int(x0),
            "bbox_y0_px": int(y0),
            "bbox_x1_px": int(x1),
            "bbox_y1_px": int(y1),
        })

    return pore_mask, fiber_mask, features


# Summaries and outputs

def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def alignment_index(features: list[dict[str, float | int | str]]) -> float:
    angles = [
        float(f["orientation_rad"])
        for f in features
        if float(f["aspect_ratio"]) >= ALIGN_ASPECT_MIN
    ]
    if len(angles) < 5:
        return 0.0
    return float(abs(np.exp(2j * np.asarray(angles)).mean()))


def image_summary(
    record: ImageRecord,
    shape: tuple[int, int],
    um_per_px: float,
    dark_mask: np.ndarray,
    gray_phase: np.ndarray,
    pore_mask: np.ndarray,
    fiber_mask: np.ndarray,
    features: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    image_area_um2 = shape[0] * shape[1] * um_per_px**2
    image_area_mm2 = image_area_um2 * 1e-6

    pores = [f for f in features if f["class"] == "pore"]
    fibers = [f for f in features if f["class"] == "carbon_fiber"]
    pore_d = [float(f["equivalent_diameter_um"]) for f in pores]
    fiber_len = [float(f["major_axis_um"]) for f in fibers]

    return {
        "sample": record.sample,
        "image": record.image,
        "mag": record.mag,
        "objective": MAG_LABEL[record.mag],
        "path": str(record.path),
        "height_px": shape[0],
        "width_px": shape[1],
        "um_per_px": um_per_px,
        "image_area_um2": image_area_um2,
        "dark_area_pct": 100.0 * float(dark_mask.mean()),
        "gray_phase_area_pct": 100.0 * float(gray_phase.mean()),
        "pore_count": len(pores),
        "pore_area_pct": 100.0 * float(
            pore_mask.sum() * um_per_px**2 / image_area_um2),
        "pore_density_per_mm2":
            len(pores) / image_area_mm2 if image_area_mm2 else 0.0,
        "pore_d10_um": percentile(pore_d, 10),
        "pore_d50_um": percentile(pore_d, 50),
        "pore_d90_um": percentile(pore_d, 90),
        "pore_dmax_um": max(pore_d) if pore_d else 0.0,
        "carbon_fiber_count": len(fibers),
        "carbon_fiber_area_pct": 100.0 * float(
            fiber_mask.sum() * um_per_px**2 / image_area_um2),
        "carbon_fiber_density_per_mm2":
            len(fibers) / image_area_mm2 if image_area_mm2 else 0.0,
        "carbon_fiber_length_d50_um": percentile(fiber_len, 50),
        "carbon_fiber_length_d90_um": percentile(fiber_len, 90),
        "carbon_fiber_length_max_um": max(fiber_len) if fiber_len else 0.0,
        "carbon_fiber_alignment_index": alignment_index(fibers),
        "other_dark_count":
            sum(1 for f in features if f["class"] == "other_dark"),
    }


def write_mask(mask: np.ndarray, path: pathlib.Path) -> None:
    io.imsave(path, (mask.astype(np.uint8) * 255), check_contrast=False)


def save_overlay(
    rgb: np.ndarray,
    pore_mask: np.ndarray,
    fiber_mask: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    scale = OVERLAY_DOWNSCALE
    small = transform.resize_local_mean(rgb, output_shape=(
        max(1, int(rgb.shape[0] * scale)),
        max(1, int(rgb.shape[1] * scale)),
        3,
    ))
    pore_small = transform.resize_local_mean(
        pore_mask.astype(float), output_shape=small.shape[:2]
    ) > 0.2
    fiber_small = transform.resize_local_mean(
        fiber_mask.astype(float), output_shape=small.shape[:2]
    ) > 0.2

    cyan = np.array([0.0, 0.9, 1.0])
    red = np.array([1.0, 0.0, 0.0])

    overlay = small.copy()
    overlay[pore_small] = 0.35 * overlay[pore_small] + 0.65 * cyan
    overlay[fiber_small] = 0.35 * overlay[fiber_small] + 0.65 * red

    pore_edge = segmentation.find_boundaries(pore_small, mode="outer")
    fiber_edge = segmentation.find_boundaries(fiber_small, mode="outer")
    overlay[pore_edge] = [0.0, 1.0, 1.0]
    overlay[fiber_edge] = [1.0, 0.0, 0.0]

    io.imsave(out_path, util.img_as_ubyte(np.clip(overlay, 0.0, 1.0)),
              check_contrast=False)


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_keys = [
        k for k, v in rows[0].items()
        if isinstance(v, (int, float)) and k not in {"height_px", "width_px"}
    ]
    out: list[dict[str, object]] = []
    for sample in sorted({str(r["sample"]) for r in rows}):
        mags = sorted({str(r["mag"]) for r in rows if r["sample"] == sample})
        for mag in mags:
            group = [r for r in rows
                     if r["sample"] == sample and r["mag"] == mag]
            row: dict[str, object] = {
                "sample": sample,
                "mag": mag,
                "objective": MAG_LABEL.get(mag, mag),
                "n_images": len(group),
            }
            for key in numeric_keys:
                vals = [float(r[key]) for r in group]
                row[f"mean_{key}"] = float(np.mean(vals))
            out.append(row)
    return out


# Pipeline

def process_image(
    record: ImageRecord,
    out_dir: pathlib.Path,
    min_feature_um2: float,
    min_pore_area_um2: float,
    min_fiber_length_um: float,
    overlays: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    um = UM_PER_PX[record.mag]
    min_area_px = int(np.ceil(min_feature_um2 / um**2))

    rgb = read_rgb(record.path)
    gray = color.rgb2gray(rgb)
    dark_mask, gray_phase = segment_dark_and_gray(gray, min_area_px)

    pore_mask, fiber_mask, features = classify_features(
        dark_mask, um, min_pore_area_um2, min_fiber_length_um)

    stem = image_stem(record)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    write_mask(dark_mask, masks_dir / f"{stem}_dark.png")
    write_mask(pore_mask, masks_dir / f"{stem}_pores.png")
    write_mask(fiber_mask, masks_dir / f"{stem}_carbon_fiber.png")
    write_mask(gray_phase, masks_dir / f"{stem}_gray_phase.png")

    if overlays:
        overlay_dir = out_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        save_overlay(rgb, pore_mask, fiber_mask,
                     overlay_dir / f"{stem}_segmentation.png")

    feature_rows: list[dict[str, object]] = []
    for feature in features:
        feature_rows.append({
            "sample": record.sample,
            "image": record.image,
            "mag": record.mag,
            "source": str(record.path),
            **feature,
        })

    row = image_summary(record, gray.shape, um, dark_mask, gray_phase,
                        pore_mask, fiber_mask, features)
    return row, feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="images", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results") / "segmentation")
    parser.add_argument("--min-feature-um2", type=float,
                        default=DEFAULT_MIN_FEATURE_UM2,
                        help="area floor for dark objects")
    parser.add_argument("--min-pore-area-um2", type=float,
                        default=DEFAULT_MIN_FEATURE_UM2,
                        help="area floor for pores")
    parser.add_argument("--min-fiber-length-um", default=10.0, type=float,
                        help="length floor for carbon fiber")
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    records = find_images(args.images)
    if not records:
        raise SystemExit(f"no microscope images found under {args.images}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    print(f"found {len(records)} images")
    for record in records:
        row, features = process_image(
            record,
            args.out,
            args.min_feature_um2,
            args.min_pore_area_um2,
            args.min_fiber_length_um,
            overlays=not args.no_overlays,
        )
        summary_rows.append(row)
        feature_rows.extend(features)
        print(
            f"{record.sample:26s} {record.image:7s} "
            f"{MAG_LABEL[record.mag]:>4s} "
            f"pores={row['pore_count']:5d} ({row['pore_area_pct']:5.2f}%)  "
            f"carbon_fiber={row['carbon_fiber_count']:4d} "
            f"({row['carbon_fiber_area_pct']:5.2f}%)",
            flush=True,
        )

    write_csv(args.out / "image_summary.csv", summary_rows)
    write_csv(args.out / "features.csv", feature_rows)
    write_csv(args.out / "sample_summary.csv", sample_summary(summary_rows))
    print(f"wrote segmentation results to {args.out}")


if __name__ == "__main__":
    main()
