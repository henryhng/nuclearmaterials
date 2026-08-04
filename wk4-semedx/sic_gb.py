"""SiC placement relative to grain boundaries on co-registered EDX fields.

Usage: python sic_gb.py
"""

import numpy as np
import pandas as pd
from skimage import measure, morphology

import grains as ga
import sem
from pores import attachment, grain_grain_bounds
from sites import (GB_NEAR_PX, MIN_SIC_PX, RES, SIC_SITES, grain_floor_px,
                   load, norm8)


def analyse(s):
    electron = norm8(load('electron', s['electron']))
    px_um = s['width_um'] / electron.shape[1]

    sic = ga.sic_mask(s['si'], s['time'])
    pores = sem.segment_dark(electron)

    min_px = grain_floor_px(px_um)
    labels = ga.segment_grains(electron, pores | sic, min_px)
    bounds = grain_grain_bounds(labels)
    band = morphology.binary_dilation(bounds, morphology.disk(GB_NEAR_PX))

    regions = [r for r in measure.regionprops(measure.label(sic))
               if r.area >= MIN_SIC_PX]
    rows = attachment(band, ~pores, regions, px_um)

    grains = [r for r in measure.regionprops(labels) if r.area >= min_px]
    d = np.array([2 * np.sqrt(r.area / np.pi) * px_um for r in grains])
    a = np.array([r.area for r in grains], dtype=float)

    df = pd.DataFrame(rows)
    obs, null = df.on_gb.mean(), df.p_null.mean()
    se = np.sqrt(null * (1 - null) / len(df))
    return {
        'site': s['tag'], 'sample': s['sample'], 'field_um': s['width_um'],
        'n_grains': len(grains),
        'grain_d_aw_um': round(float((d * a).sum() / a.sum()), 2),
        'n_sic': len(df),
        'sic_d_mean_um': round(float(df.d_eq_um.mean()), 3),
        'sic_on_gb_pct': round(100 * obs, 1),
        'null_pct': round(100 * null, 1),
        'enrichment': round(obs / null, 2),
        'z': round((obs - null) / se, 2),
    }


def main():
    out = pd.DataFrame([analyse(s) for s in SIC_SITES])
    out.to_csv(RES / 'sic_gb.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
