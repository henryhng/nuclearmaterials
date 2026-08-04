"""EDX map statistics for the AZtec project (run edx_extract.py first).

Usage: python edx.py
"""

import numpy as np
import pandas as pd
from scipy import ndimage

from sites import ACQS, RES, load

# AZtec auto-ID artifacts
ARTIFACTS = {'Ge', 'Sb'}
SMOOTH_PX = 3


def phase_fractions(a, counts):
    # dominant element per pixel, normalized to suppress yield bias
    els = [e for e in a['maps'] if e not in ARTIFACTS]
    stack = np.stack([ndimage.gaussian_filter(counts[e], SMOOTH_PX)
                      for e in els])
    stack /= stack.mean(axis=(1, 2), keepdims=True)
    dom = stack.argmax(axis=0)
    return {els[i]: float((dom == i).mean()) for i in range(len(els))}


def main():
    RES.mkdir(parents=True, exist_ok=True)

    acq_rows, phase_rows = [], []
    for a in ACQS:
        live = load('time_live', a['time'])
        real = load('time_real', a['time'])
        counts = {el: load('xraymap', g) * live
                  for el, g in a['maps'].items()}
        tag = f"{a['site']}{a['run']}"

        totals = {el: float(c.sum()) for el, c in counts.items()}
        real_total = float(real.sum())
        live_total = float(live.sum())

        acq_rows.append({
            'tag': tag, 'sample': a['sample'], 'width_um': a['width_um'],
            'map_time_s': round(real_total, 1),
            'live_s': round(live_total, 1),
            'dead_pct': round(100 * (1 - live_total / real_total), 1),
            **{f'{el}_kcounts': round(v / 1e3, 1)
               for el, v in totals.items()},
        })

        fracs = phase_fractions(a, counts)
        phase_rows.append({'tag': tag, 'sample': a['sample'],
                           **{f'{el}_dom_pct': round(v * 100, 2)
                              for el, v in fracs.items()}})

    pd.DataFrame(acq_rows).to_csv(RES / 'edx_acq.csv', index=False)
    pd.DataFrame(phase_rows).to_csv(RES / 'edx_phases.csv', index=False)
    print(pd.DataFrame(acq_rows).to_string(index=False))
    print()
    print(pd.DataFrame(phase_rows).to_string(index=False))


if __name__ == '__main__':
    main()
