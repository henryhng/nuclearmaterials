"""SEM image metrics for SEM SEM-EDX 07.30.

Usage: python sem.py
"""

import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import measure, morphology

from sites import RES, SEM_DATA as DATA

SAMPLES = {
    'wc+ZrC+Ta+Cf': 'WC+ZrC+Ta+Cf',
    'vc+Sic+Cf': 'VC+SiC+Cf',
    'vc+Sic': 'VC+SiC',
    'wc+Sic+Cf': 'WC+SiC+Cf',
}
SAMPLE_ORDER = ['VC+SiC', 'VC+SiC+Cf', 'WC+SiC+Cf', 'WC+ZrC+Ta+Cf']

MIN_PORE_PX = 6
FIBER_ASPECT = 3.0
FIBER_LEN_UM = 2.0


# Metadata and loading

def zeiss_meta(path):
    tag = Image.open(path).tag_v2[34118]
    keys = {
        'AP_IMAGE_PIXEL_SIZE': 'pixel_size', 'AP_MAG': 'mag',
        'AP_WD': 'wd', 'AP_ACTUALKV': 'eht', 'DP_DETECTOR_TYPE': 'detector',
        'AP_WIDTH': 'width', 'AP_HEIGHT': 'height', 'AP_TIME': 'time',
    }
    meta = {}
    for m in re.finditer(r'(AP_|DP_)[A-Z0-9_]+\r\n([^\r]+)', tag):
        key = m.group(0).split('\r\n')[0]
        if key in keys:
            meta[keys[key]] = m.group(2).split('=')[-1].strip().lstrip(':')
    return meta


def px_size_um(meta):
    val, unit = meta['pixel_size'].split()
    return float(val) * {'nm': 1e-3, 'µm': 1.0, 'pm': 1e-6}[unit]


def load_field(path):
    # databar strip
    img = np.array(Image.open(path).convert('L'))
    extreme = (img < 8) | (img > 247)
    bar_rows = np.where(extreme.mean(axis=1) > 0.6)[0]
    cut = bar_rows.min() if len(bar_rows) else img.shape[0]
    return img[:cut]


def find_images():
    records = []
    for path in sorted(DATA.glob('*.tif')):
        stem = re.match(r'(.+?)(\d+)$', path.stem)
        key, num = stem.groups()
        records.append((SAMPLES[key], int(num), path))
    return records


# Segmentation

def segment_dark(field):
    blur = ndimage.gaussian_filter(field.astype(float), 1.5)
    med = np.median(blur)
    mad = np.median(np.abs(blur - med))
    spread = max(1.4826 * mad, blur.std() / 3, 2.0)
    mask = blur < med - 4.5 * spread
    return ndimage.binary_closing(mask, morphology.disk(1))


def dark_features(field, px_um):
    mask = segment_dark(field)
    labels = measure.label(mask)
    pores, fibers = [], []
    for r in measure.regionprops(labels):
        if r.area < MIN_PORE_PX:
            continue
        d_eq = 2 * np.sqrt(r.area / np.pi) * px_um
        aspect = r.axis_major_length / max(r.axis_minor_length, 1.0)
        length = r.axis_major_length * px_um
        feat = dict(area_um2=r.area * px_um ** 2, d_eq_um=d_eq,
                    aspect=aspect, length_um=length)
        if aspect >= FIBER_ASPECT and length >= FIBER_LEN_UM:
            fibers.append(feat)
        else:
            pores.append(feat)
    return mask, pores, fibers


def contrast_length(field, mask, px_um):
    # autocorrelation FWHM
    f = ndimage.gaussian_filter(field.astype(float), 2)
    f[mask] = np.nan
    f = f - np.nanmean(f)
    f = np.nan_to_num(f)
    ac = np.fft.irfft2(np.abs(np.fft.rfft2(f)) ** 2, s=f.shape)
    if ac[0, 0] <= 0:
        return np.nan
    ac /= ac[0, 0]
    n = min(f.shape[0], f.shape[1]) // 2
    prof = np.minimum(ac[0, :n], ac[:n, 0])
    below = np.where(prof < 0.5)[0]
    return (below[0] if len(below) else n) * px_um


def main():
    RES.mkdir(parents=True, exist_ok=True)

    meta_rows, stat_rows, size_rows = [], [], []
    for sample, num, path in find_images():
        meta = zeiss_meta(path)
        px_um = px_size_um(meta)
        field = load_field(path)
        mask, pores, fibers = dark_features(field, px_um)
        corr_um = contrast_length(field, mask, px_um)

        area_um2 = field.size * px_um ** 2
        pore_area = sum(p['area_um2'] for p in pores)
        fiber_area = sum(f['area_um2'] for f in fibers)
        meta_rows.append({'sample': sample, 'image': path.name, **meta,
                          'px_um': px_um})
        stat_rows.append({
            'sample': sample, 'image': path.name, 'mag': meta['mag'],
            'field_um2': round(area_um2, 1),
            'porosity_pct': round(100 * pore_area / area_um2, 3),
            'fiber_pct': round(100 * fiber_area / area_um2, 3),
            'n_pores': len(pores),
            'pore_density_mm2': round(len(pores) / area_um2 * 1e6, 0),
            'pore_d_mean_um': round(np.mean([p['d_eq_um'] for p in pores]), 3)
            if pores else 0,
            'pore_d_median_um':
                round(np.median([p['d_eq_um'] for p in pores]), 3)
            if pores else 0,
            'contrast_len_um': round(corr_um, 2),
        })
        for p in pores:
            size_rows.append({'sample': sample, 'image': path.name,
                              'kind': 'pore', **p})
        for f in fibers:
            size_rows.append({'sample': sample, 'image': path.name,
                              'kind': 'fiber', **f})

    stats_df = pd.DataFrame(stat_rows)
    pd.DataFrame(meta_rows).to_csv(RES / 'sem_meta.csv', index=False)
    stats_df.to_csv(RES / 'sem_stats.csv', index=False)
    pd.DataFrame(size_rows).to_csv(RES / 'pore_sizes.csv', index=False)
    print(stats_df.to_string(index=False))


if __name__ == '__main__':
    main()
