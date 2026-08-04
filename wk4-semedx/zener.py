"""Zener pinning evaluation: grain size, SiC dispersion, boundary contact.

Usage: python zener.py
"""

import numpy as np
import pandas as pd
from skimage import measure, morphology

import grains as ga
import sem
from pores import attachment, grain_grain_bounds
from sites import (GB_NEAR_PX, MIN_SIC_PX, RES, SIC_SITES, grain_floor_px,
                   load, norm8)


def grain_pass(field, mask, px_um):
    min_px = grain_floor_px(px_um)
    labels = ga.segment_grains(field, mask, min_px)
    bounds = grain_grain_bounds(labels)

    regions = [r for r in measure.regionprops(labels) if r.area >= min_px]
    d = np.array([2 * np.sqrt(r.area / np.pi) * px_um for r in regions])
    a = np.array([r.area for r in regions], dtype=float)
    d_aw = float((d * a).sum() / a.sum())

    gb_density = bounds.sum() / 2 * px_um / ((~mask).sum() * px_um ** 2)
    band = morphology.binary_dilation(bounds, morphology.disk(GB_NEAR_PX))
    return d_aw, len(regions), float(gb_density), band


def att_stats(rows):
    df = pd.DataFrame(rows)
    obs, null = df.on_gb.mean(), df.p_null.mean()
    se = np.sqrt(null * (1 - null) / len(df))
    return dict(n_sic=len(df), on_gb_pct=round(100 * obs, 1),
                null_pct=round(100 * null, 1),
                enrichment=round(obs / null, 2),
                z=round((obs - null) / se, 1))


def sic_stats(mask, px_um):
    regions = [r for r in measure.regionprops(measure.label(mask))
               if r.area >= MIN_SIC_PX]
    d = np.array([2 * np.sqrt(r.area / np.pi) * px_um for r in regions])
    cents = np.array([r.centroid for r in regions]) * px_um
    dm = np.linalg.norm(cents[:, None] - cents[None], axis=-1)
    np.fill_diagonal(dm, np.inf)
    return dict(n=len(regions), f_pct=round(100 * mask.mean(), 2),
                r_um=round(float(d.mean() / 2), 3),
                nn_um=round(float(dm.min(axis=1).mean()), 2)), regions


# Field passes

def sem_fields():
    rows, att_rows = [], []
    for name, sample in ga.GRAIN_IMAGES.items():
        meta = sem.zeiss_meta(sem.DATA / name)
        px_um = sem.px_size_um(meta)
        field = sem.load_field(sem.DATA / name)
        dark = sem.segment_dark(field)

        d_aw, n, gb_density, band = grain_pass(field, dark, px_um)
        row = dict(source='SEM', field=name, sample=sample, mag=meta['mag'],
                   n_grains=n, d_aw_um=round(d_aw, 1),
                   gb_density_per_um=round(gb_density, 4))

        # dark features in VC+SiC are SiC
        if sample == 'VC+SiC':
            regions = measure.regionprops(measure.label(dark))
            att = attachment(band, ~dark, regions, px_um)
            att_rows += att
            row.update(att_stats(att))
        rows.append(row)
    return rows, att_stats(att_rows)


def edx_fields():
    rows = []
    for s in SIC_SITES:
        electron = norm8(load('electron', s['electron']))
        px_um = s['width_um'] / electron.shape[1]
        sic = ga.sic_mask(s['si'], s['time'])
        pores = sem.segment_dark(electron)

        d_aw, n, gb_density, band = grain_pass(electron, pores | sic, px_um)
        stats, regions = sic_stats(sic, px_um)
        row = dict(source='EDX', field=s['tag'], sample=s['sample'],
                   mag=f"{s['width_um']} um",
                   n_grains=n, d_aw_um=round(d_aw, 1),
                   gb_density_per_um=round(gb_density, 4), **stats)

        # placement only where the electron image resolves grains
        if s['tag'] in ('Site 6a', 'Site 6b'):
            row.update(att_stats(attachment(band, ~pores, regions, px_um)))
        rows.append(row)
    return rows


# Zener evaluation

def zener_table(fields):
    df = pd.DataFrame(fields)
    out = []
    for sample, sic_src in [('VC+SiC+Cf', ['Site 6a', 'Site 6b']),
                            ('VC+SiC', ['Site 3'])]:
        g_sem = df[(df['sample'] == sample) & (df.source == 'SEM')]
        g_edx = df[(df['sample'] == sample) & (df.source == 'EDX')]
        sic = g_edx[g_edx.field.isin(sic_src)]
        r, f = sic.r_um.mean(), sic.f_pct.mean() / 100
        d_max = 4 * r / (3 * f)
        d_obs = g_sem.d_aw_um.mean()
        out.append(dict(
            sample=sample, sic_source='+'.join(sic_src),
            sic_r_um=round(r, 3), sic_f_pct=round(100 * f, 2),
            zener_dmax_um=round(d_max, 1),
            d_aw_sem_um=round(d_obs, 1),
            d_aw_edx_um=round(g_edx.d_aw_um.mean(), 1),
            observed_over_limit=round(d_obs / d_max, 2)))
    return pd.DataFrame(out)


def main():
    sem_rows, sem_att = sem_fields()
    fields = sem_rows + edx_fields()
    pd.DataFrame(fields).to_csv(RES / 'zener_fields.csv', index=False)

    zener = zener_table(fields)
    zener.to_csv(RES / 'zener_summary.csv', index=False)

    print(pd.DataFrame(fields).to_string(index=False))
    print()
    print('VC+SiC pooled SiC-boundary contact (SEM):', sem_att)
    print()
    print(zener.to_string(index=False))


if __name__ == '__main__':
    main()
