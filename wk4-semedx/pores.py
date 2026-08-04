"""Size-unbiased pore-boundary attachment and distribution shape tests.

Usage: python pores.py
"""

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage import measure, morphology

import grains as ga
import sem
from sites import GB_NEAR_PX, N_TRIALS, RES

RNG = np.random.default_rng(0)


# grain-grain boundaries only
def grain_grain_bounds(labels):
    big = labels.max() + 1
    hi = ndimage.maximum_filter(labels, size=3)
    lo = ndimage.minimum_filter(np.where(labels > 0, labels, big), size=3)
    return (hi > lo) & (hi > 0) & (lo < big)


def attachment(band, solid, regions, px_um):
    # footprint overlap vs size-matched random placement
    h, w = band.shape
    rows = []
    for r in regions:
        fp = r.image
        fh, fw = fp.shape
        if fh >= h or fw >= w:
            continue
        r0, c0, _, _ = r.bbox
        observed = bool((band[r0:r0 + fh, c0:c0 + fw] & fp).any())

        # size-matched null
        hits = tries = draws = 0
        while tries < N_TRIALS and draws < 20 * N_TRIALS:
            y = RNG.integers(0, h - fh)
            x = RNG.integers(0, w - fw)
            win = (slice(y, y + fh), slice(x, x + fw))
            draws += 1
            if (solid[win] & fp).sum() < 0.5 * fp.sum():
                continue
            hits += bool((band[win] & fp).any())
            tries += 1
        if tries:
            rows.append({'d_eq_um': 2 * np.sqrt(r.area / np.pi) * px_um,
                         'on_gb': observed, 'p_null': hits / tries})
    return rows


def image_attachment(name, sample):
    path = sem.DATA / name
    meta = sem.zeiss_meta(path)
    px_um = sem.px_size_um(meta)
    field = sem.load_field(path)
    pores = sem.segment_dark(field)

    labels = ga.segment_grains(field, pores)
    band = morphology.binary_dilation(grain_grain_bounds(labels),
                                      morphology.disk(GB_NEAR_PX))
    
    regions = measure.regionprops(measure.label(pores))
    rows = attachment(band, ~pores, regions, px_um)
    return [{'sample': sample, 'image': name, **r} for r in rows]


def lognormal_test(d):
    ld = np.log(d[d > 0])
    ad = stats.anderson(ld, dist='norm')
    return {
        'n': len(ld),
        'median_um': round(float(np.exp(np.median(ld))), 4),
        'sigma_log': round(float(ld.std(ddof=1)), 4),
        'ad_stat': round(float(ad.statistic), 3),
        'ad_crit_5pct': round(float(ad.critical_values[2]), 3),
        'lognormal': bool(ad.statistic < ad.critical_values[2]),
    }


def main():
    rows = []
    for name, sample in ga.GRAIN_IMAGES.items():
        rows += image_attachment(name, sample)
    pores = pd.DataFrame(rows)
    pores.to_csv(RES / 'pore_gb.csv', index=False)

    tests = []
    for sample, g in pores.groupby('sample'):
        obs = g.on_gb.mean()
        null = g.p_null.mean()
        # binomial spread of the null
        se = np.sqrt(null * (1 - null) / len(g))
        tests.append({
            'quantity': 'pore_attachment', 'sample': sample, 'n_pores': len(g),
            'observed_pct': round(100 * obs, 1),
            'null_pct': round(100 * null, 1),
            'enrichment': round(obs / null, 3) if null else np.nan,
            'z': round((obs - null) / se, 2) if se else np.nan,
        })

    grains = pd.read_csv(RES / 'grain_sizes.csv')
    for sample, g in grains.groupby('sample'):
        tests.append({'quantity': 'grain_d_eq', 'sample': sample,
                      **lognormal_test(g.d_eq_um.to_numpy())})

    out = pd.DataFrame(tests)
    out.to_csv(RES / 'dist_tests.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
