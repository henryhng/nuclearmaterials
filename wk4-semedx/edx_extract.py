"""EDX extraction: AZtec docx report images and raw dat decoding.

Usage: python edx_extract.py
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sites import EDX_DATA as DATA, EXTRACTED as OUT

EDS_SHAPE = (768, 1024)
THUMB_PX = 512 * 384


def report_meta(xml):
    # metadata table rows come as label/value cell pairs
    text = re.sub(r'<[^>]+>', '\n', xml)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    meta = {}
    for i, l in enumerate(lines):
        if l.endswith(':') and i + 1 < len(lines):
            meta[l[:-1]] = lines[i + 1]
    return meta


def extract_reports():
    rows = []
    for docx in sorted(DATA.glob('reports/*.docx')):
        m = re.match(r'Project 1_(Site \d)_([\d-]+)_([\d-]+)', docx.stem)
        site, date, time = m.groups()
        sdir = OUT / site.replace(' ', '').lower()
        sdir.mkdir(parents=True, exist_ok=True)

        z = zipfile.ZipFile(docx)
        meta = report_meta(z.read('word/document.xml').decode('utf-8'))
        media = [n for n in z.namelist() if n.startswith('word/media/')]
        kind = 'layered' if meta else 'maps'

        for n in media:
            img = Image.open(z.open(n))
            if img.size == (1, 1) or img.size[0] < 200:
                continue
            name = f"{kind}_{Path(n).stem}{Path(n).suffix}"
            img.save(sdir / name)
            rows.append({'site': site, 'report': kind, 'file': name,
                         'w': img.size[0], 'h': img.size[1], **meta})
    pd.DataFrame(rows).to_csv(OUT / 'report_images.csv', index=False)
    return rows


def decode_raw():
    # u16 files: electron image + 512x384 thumbnail, f32: count-rate maps
    raw = OUT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    index = []
    for f in sorted(DATA.glob('data/*.dat')):
        size = f.stat().st_size
        head = f.read_bytes()[:20]
        if head.startswith(b'OINA.'):
            continue
        if size == 1966080:
            a = np.fromfile(f, dtype='<u2')[:EDS_SHAPE[0] * EDS_SHAPE[1]]
            kind = 'electron'
        elif size == 3932160:
            a = np.fromfile(f, dtype='<f4')[:EDS_SHAPE[0] * EDS_SHAPE[1]]
            kind = 'xraymap'
        elif size == 3145728:
            a = np.fromfile(f, dtype='<f4')
            kind = 'time_' + ('live' if f.stem.endswith('live') else 'real')
        else:
            continue
        a = a.reshape(EDS_SHAPE)
        np.save(raw / f'{kind}_{f.stem[:8]}.npy', a)
        index.append({'kind': kind, 'guid': f.stem, 'file': f.name,
                      'mean': float(a.mean()), 'max': float(a.max())})
    pd.DataFrame(index).to_csv(OUT / 'raw_index.csv', index=False)
    return index


if __name__ == '__main__':
    rows = extract_reports()
    print(f'report images: {len(rows)}')
    index = decode_raw()
    from collections import Counter
    print(Counter(r['kind'] for r in index))
