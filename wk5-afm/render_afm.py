"""Render AFM 08.03 Nanoscope scans."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/workspaces/Repos/garcia')
SRC = ROOT / 'data' / 'AFM 08.03'
OUT = Path(__file__).parent / 'images'

HEADER_LEN = 40960
MIN_LINES = 8
CMAP = 'afmhot'

LEVELED = {'Height Sensor', 'Height'}


# Header parsing

def parse_header(raw):
    text = raw[:HEADER_LEN].decode('latin-1').replace('\r', '')
    sens = {}
    for m in re.finditer(r'\\@Sens\. (\S+): V ([\d.eE+-]+) ?(\S*)', text):
        sens[m.group(1)] = (float(m.group(2)), m.group(3))

    channels = []
    for block in text.split('\\*Ciao image list')[1:]:
        get = lambda key: re.search(rf'\\{key}: (.+)', block).group(1)
        name = re.search(r'@2:Image Data: S \[.*?\] "(.+?)"', block).group(1)
        scale = re.search(
            r'@2:Z scale: V \[(.*?)\] \(([\d.eE+-]+) (\S+)/LSB\)', block)
        sens_val, sens_unit = 1.0, ''
        if scale.group(1):
            sens_val, sens_unit = sens.get(
                scale.group(1).replace('Sens. ', ''), (1.0, ''))
        unit = sens_unit.split('/')[0] if '/' in sens_unit else scale.group(3)
        size = get('Scan Size').split()
        size_um = float(size[0]) / (1e3 if size[-1] == 'nm' else 1.0)
        channels.append({
            'name': name,
            'offset': int(get('Data offset')),
            'cols': int(get('Samps/line')),
            'rows': int(get('Number of lines')),
            'size_um': size_um,
            'factor': float(scale.group(2)) * sens_val,
            'unit': unit,
        })
    return channels


def read_channel(raw, ch):
    n = ch['rows'] * ch['cols']
    data = np.frombuffer(raw, '<i2', count=n, offset=ch['offset'])
    return data.reshape(ch['rows'], ch['cols']).astype(float) * ch['factor']


# Leveling

def level(z):
    rows, cols = z.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    basis = np.column_stack([x.ravel(), y.ravel(), np.ones(z.size)])
    coef, *_ = np.linalg.lstsq(basis, z.ravel(), rcond=None)
    z = z - (basis @ coef).reshape(z.shape)
    return z - np.median(z, axis=1, keepdims=True)


# Rendering

def slug(name):
    return name.lower().replace('+', '').replace(' ', '')


def render(stem, ch, z, path):
    size = ch['size_um']
    lo, hi = np.percentile(z, [0.5, 99.5])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(z, cmap=CMAP, vmin=lo, vmax=hi, origin='lower',
                   extent=[0, size, 0, size])
    ax.set_xlabel('x (\u00b5m)')
    ax.set_ylabel('y (\u00b5m)')
    ax.set_title(f'{stem}  \u2014  {ch["name"]}', fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(ch['unit'] or 'arb')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def overview(entries, path):
    entries = sorted(entries, key=lambda e: e[0])
    cols = 4
    rows = -(-len(entries) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows),
                             squeeze=False)
    for ax, (stem, size, z) in zip(axes.flat, entries):
        lo, hi = np.percentile(z, [0.5, 99.5])
        ax.imshow(z, cmap=CMAP, vmin=lo, vmax=hi, origin='lower',
                  extent=[0, size, 0, size])
        ax.set_title(f'{stem} ({size:g} \u00b5m)', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(entries):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    heights = []
    count = 0

    files = sorted(p for p in SRC.iterdir()
                   if p.suffix not in ('.pfc', '.bin') and p.is_file())
    for path in files:
        raw = path.read_bytes()
        channels = parse_header(raw)
        first = channels[0]
        live = int((read_channel(raw, first).std(axis=1) > 0).sum())
        if first['rows'] < MIN_LINES or live < first['rows'] // 4:
            print(f'skip {path.name}: aborted scan '
                  f'({live}/{first["rows"]} lines captured)')
            continue
        stem = slug(path.name)
        for ch in channels:
            z = read_channel(raw, ch)
            if ch['name'] in LEVELED:
                z = level(z)
            render(stem, ch, z, OUT / f'{stem}_{slug(ch["name"])}.png')
            count += 1
            if ch['name'] == 'Height Sensor':
                heights.append((stem, ch['size_um'], z))

    overview(heights, OUT / 'overview_height.png')
    print(f'{count} channel images + overview -> {OUT}')


if __name__ == '__main__':
    main()
