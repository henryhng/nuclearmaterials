"""Grain and SiC dispersoid metrics (run sem.py / edx_extract.py first).

Usage: python grains.py
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import graph, measure, morphology, segmentation

import sem
from sites import (GB_NEAR_PX, MERGE_GRAY, MIN_GRAIN_PX, RES, SIC_SITES,
                   load)

# images with usable channeling contrast
GRAIN_IMAGES = {
    'vc+Sic001.tif': 'VC+SiC',
    'vc+Sic003.tif': 'VC+SiC',
    'wc+Sic+Cf001.tif': 'WC+SiC+Cf',
    'wc+Sic+Cf004.tif': 'WC+SiC+Cf',
    'vc+Sic+Cf002.tif': 'VC+SiC+Cf',
    'vc+Sic+Cf005.tif': 'VC+SiC+Cf',
    'wc+ZrC+Ta+Cf002.tif': 'WC+ZrC+Ta+Cf',
    'wc+ZrC+Ta+Cf003.tif': 'WC+ZrC+Ta+Cf',
}


# Grain segmentation

def segment_grains(field, pore_mask, min_grain_px=MIN_GRAIN_PX):
    sm = ndimage.gaussian_filter(
        ndimage.median_filter(field.astype(float), 3), 1.5)
    grad = ndimage.gaussian_gradient_magnitude(sm, 1.5)
    gradm = grad.copy()
    gradm[pore_mask] = grad.max()

    markers, _ = ndimage.label(
        morphology.h_minima(grad, 0.85 * np.median(grad)))
    ws = segmentation.watershed(gradm, markers, mask=~pore_mask)

    rag = graph.rag_mean_color(np.stack([sm] * 3, -1) / 255, ws)

    def weight(g, s, d, n):
        diff = g.nodes[d]['mean color'][0] - g.nodes[n]['mean color'][0]
        return {'weight': abs(diff)}

    def merge(g, s, d):
        g.nodes[d]['total color'] += g.nodes[s]['total color']
        g.nodes[d]['pixel count'] += g.nodes[s]['pixel count']
        g.nodes[d]['mean color'] = (g.nodes[d]['total color']
                                    / g.nodes[d]['pixel count'])

    labels = graph.merge_hierarchical(ws, rag, thresh=MERGE_GRAY / 255,
                                      rag_copy=False, in_place_merge=True,
                                      merge_func=merge,
                                      weight_func=weight) + 1
    labels[ws == 0] = 0

    small = np.isin(labels, [r.label for r in measure.regionprops(labels)
                             if r.area < min_grain_px])
    labels[small] = 0
    for _ in range(12):
        grow = (labels == 0) & ~pore_mask
        if not grow.any():
            break
        labels = np.where(grow, ndimage.grey_dilation(labels, 3), labels)
    return labels


def grain_metrics(labels, pore_mask, px_um, min_grain_px=MIN_GRAIN_PX):
    bounds = segmentation.find_boundaries(labels, mode='thick') & (labels > 0)
    regions = [r for r in measure.regionprops(labels)
               if r.area >= min_grain_px]
    d_eq = np.array([2 * np.sqrt(r.area / np.pi) * px_um for r in regions])
    areas = np.array([r.area for r in regions], dtype=float)

    solid = (~pore_mask).sum()
    gb_len_um = bounds.sum() / 2 * px_um   # thick boundary counts both sides
    gb_density = gb_len_um / (solid * px_um ** 2)

    near_gb = ndimage.distance_transform_edt(~bounds) <= GB_NEAR_PX
    pore_labels = measure.label(pore_mask)
    inter = sum(near_gb[tuple(np.round(r.centroid).astype(int))]
                for r in measure.regionprops(pore_labels))
    n_pores = pore_labels.max()
    on_gb = 100 * inter / max(n_pores, 1)
    baseline = 100 * near_gb[~pore_mask].mean()

    return bounds, dict(
        n_grains=len(regions),
        d_mean_um=d_eq.mean(), d_median_um=np.median(d_eq),
        d_aw_um=(d_eq * areas).sum() / areas.sum(),
        gb_density_per_um=gb_density,
        pores_on_gb_pct=on_gb,
        gb_area_baseline_pct=baseline,
        gb_enrichment=on_gb / max(baseline, 1e-9),
    ), d_eq


# SiC dispersoids

def sic_mask(si_guid, t_guid):
    counts = load('xraymap', si_guid) * load('time_live', t_guid)
    sm = ndimage.gaussian_filter(counts, 3)
    bg = np.median(sm)
    mask = sm > bg + 4 * sm[sm < np.percentile(sm, 90)].std()
    return morphology.opening(mask, morphology.disk(2))


def sic_particles():
    rows = []
    for s in SIC_SITES:
        mask = sic_mask(s['si'], s['time'])
        px_um = s['width_um'] / mask.shape[1]

        labels = measure.label(mask)
        regions = [r for r in measure.regionprops(labels) if r.area >= 12]
        d_eq = np.array([2 * np.sqrt(r.area / np.pi) * px_um
                         for r in regions])
        cents = np.array([r.centroid for r in regions]) * px_um
        if len(cents) > 1:
            dm = np.linalg.norm(cents[:, None] - cents[None], axis=-1)
            np.fill_diagonal(dm, np.inf)
            nn = dm.min(axis=1).mean()
        else:
            nn = np.nan
        area_um2 = mask.size * px_um ** 2
        rows.append(dict(site=s['tag'], sample=s['sample'], n=len(regions),
                         area_pct=round(100 * mask.mean(), 2),
                         d_mean_um=round(d_eq.mean(), 2) if len(d_eq) else 0,
                         density_mm2=round(len(regions) / area_um2 * 1e6, 0),
                         nn_spacing_um=round(nn, 2)))
    return rows


def main():
    grain_rows, size_rows = [], []
    for name, sample in GRAIN_IMAGES.items():
        path = sem.DATA / name
        meta = sem.zeiss_meta(path)
        px_um = sem.px_size_um(meta)
        field = sem.load_field(path)
        pores = sem.segment_dark(field)

        labels = segment_grains(field, pores)
        bounds, metrics, d_eq = grain_metrics(labels, pores, px_um)
        grain_rows.append({'sample': sample, 'image': name,
                           'mag': meta['mag'],
                           **{k: round(v, 3) for k, v in metrics.items()}})
        size_rows += [{'sample': sample, 'image': name, 'd_eq_um': d}
                      for d in d_eq]

    pd.DataFrame(grain_rows).to_csv(RES / 'grain_stats.csv', index=False)
    pd.DataFrame(size_rows).to_csv(RES / 'grain_sizes.csv', index=False)
    print(pd.DataFrame(grain_rows).to_string(index=False))

    rows = sic_particles()
    pd.DataFrame(rows).to_csv(RES / 'sic_particles.csv', index=False)
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
