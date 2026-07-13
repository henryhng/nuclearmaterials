#!/usr/bin/env python3
"""Nature-style figures (Inter, panel letters, vector PDF) -> figures/nature.

    QT_QPA_PLATFORM=offscreen python3 nature_figs.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _f in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"):
    font_manager.fontManager.addfont(str(_f))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stored_energy import parse_blocks

RES = Path("../lammps/outputs/results")
OUT = Path("../figures/nature")
PDFDIR = Path("../figures/pdf")

COL = {"Si": "#2a78d6", "C": "#1baf7a", "Fe": "#eda100", "W": "#4a3aa7"}
PT, FIT = "#1f77b4", "#ff7f0e"
SINGLE, DOUBLE = 3.5, 7.2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.4,
    "ytick.major.size": 2.4, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.0,
    "legend.frameon": False, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter", "mathtext.bf": "Inter:bold",
    "mathtext.default": "regular",
})


def letter(ax, s, dx=-0.22):
    ax.text(dx, 1.06, s, transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="bottom", ha="left")


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(OUT / name)


def ed_values(path, species=None):
    rows = list(csv.DictReader(open(path)))
    return np.array([float(r["e_d_eV"]) for r in rows
                     if r.get("e_d_eV")
                     and (species is None or r.get("species") == species)])


def kinetics_csv(tag, traj, reference):
    """WS counts vs step, cached to CSV."""
    cache = RES / f"kinetics_{tag}.csv"
    if cache.exists():
        rows = list(csv.DictReader(open(cache)))
        return (np.array([int(r["step"]) for r in rows]),
                np.array([int(r["vacancies"]) for r in rows]),
                np.array([int(r["interstitials"]) for r in rows]))
    from ovito.io import import_file
    from ovito.modifiers import WignerSeitzAnalysisModifier
    from ovito.pipeline import FileSource
    pipeline = import_file(str(traj))
    ws = WignerSeitzAnalysisModifier(per_type_occupancies=True)
    ws.reference = FileSource()
    ws.reference.load(str(reference))
    pipeline.modifiers.append(ws)
    steps, vac, inter = [], [], []
    for frame in range(pipeline.source.num_frames):
        data = pipeline.compute(frame)
        steps.append(int(data.attributes.get("Timestep", frame)))
        vac.append(int(data.attributes["WignerSeitz.vacancy_count"]))
        inter.append(int(data.attributes["WignerSeitz.interstitial_count"]))
    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "vacancies", "interstitials"])
        w.writerows(zip(steps, vac, inter))
    return np.array(steps), np.array(vac), np.array(inter)


def fig_validation():
    rows = [
        ("eV/FP Fe", 4.87, 1.2, 5.0, 6.0, "Fe", False),
        ("eV/FP SiC", 5.74, 1.4, 5.0, 7.0, "C", False),
        ("eV/FP W", 12.29, 3.1, 12.0, 13.0, "W", False),
        ("$E_d$(Si) sphere", 36.2, 2.0, 33.0, 37.0, "Si", False),
        ("$E_d$(C) soft $\\langle 111\\rangle$", 22.0, 2.0, 20.0, 22.0,
         "C", False),
        ("$E_d$(W) soft", 42.5, 2.5, 42.0, 53.0, "W", False),
        ("$E_d$(W) sphere", 102.2, 9.5, 83.0, 128.0, "W", False),
        ("efficiency Fe", 0.337, 0.084, 0.33, 0.40, "Fe", False),
        ("efficiency SiC", 1.48, 0.37, 1.0, 2.0, "C", False),
        ("efficiency W (pending)", 0.337, 0.084, 0.22, 0.26, "W", True),
        ("W clustered fraction", 0.58, 0.15, 0.50, 0.70, "W", False),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE, 3.1))
    for i, (lab, v, e, lo, hi, ck, hollow) in enumerate(rows):
        y = len(rows) - 1 - i
        c = (lo + hi) / 2
        ax.barh(y, (hi - lo) / c, left=lo / c, height=0.6,
                color="#e8e7e2", zorder=1)
        ax.errorbar(v / c, y, xerr=e / c, fmt="o", ms=3.5, color=COL[ck],
                    mfc="white" if hollow else COL[ck], mew=0.9,
                    capsize=1.5, lw=0.9, zorder=3)
    ax.axvline(1.0, color="0.45", lw=0.6, ls=(0, (4, 3)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6)
    ax.set_xlabel("value / literature-band centre")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    save(fig, "validation.pdf")


def fig_renders():
    imgs = [("Fe", "../figures/renders/traj_Fe_spike_1_frame100_damage_render.png"),
            ("W", "../figures/renders/comparable/W_10keV_spike1_frame40_render.png"),
            ("SiC", "../figures/renders/comparable/SiC_10keV_spike1_frame40_render.png")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))
    for ax, (lab, p), s in zip(axes, imgs, "abc"):
        ax.imshow(mpimg.imread(p))
        ax.set_axis_off()
        ax.text(0.02, 1.02, s, transform=ax.transAxes, fontweight="bold",
                fontsize=9, va="bottom")
        ax.text(0.13, 1.03, lab, transform=ax.transAxes, fontsize=7.5,
                va="bottom", color="0.3")
    fig.subplots_adjust(wspace=0.02)
    save(fig, "renders.pdf")


def fig_kinetics():
    sic = kinetics_csv("SiC", "../lammps/outputs/traj_SiC_spike_1.lammpstrj",
                       "../lammps/structures/perfect_lattice_SiC.data")
    w = kinetics_csv("W", "../lammps/outputs/traj_W_spike_1.lammpstrj",
                     "../lammps/structures/perfect_lattice_W.data")
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 4.2))
    for ax, (steps, vac, inter), lab, ck, s in zip(
            axes, [sic, w], ["SiC", "W"], ["C", "W"], "ab"):
        ax.plot(steps / 1e3, inter, "-s", ms=2.5, color="0.6",
                label="interstitials")
        ax.plot(steps / 1e3, vac, "-o", ms=2, lw=1.2, color=COL[ck],
                label="vacancies")
        ax.set_ylabel("defect count")
        ax.legend(loc="upper right")
        letter(ax, s)
        ax.text(0.98, 0.55, lab, transform=ax.transAxes, ha="right",
                fontsize=7.5, color="0.3")
    axes[1].set_xlabel(r"MD step ($\times 10^3$)")
    fig.tight_layout(h_pad=1.2)
    save(fig, "kinetics.pdf")


def fig_thermo():
    fig, axes = plt.subplots(4, 1, figsize=(SINGLE, 4.6))
    axes = axes.reshape(2, 2, order="F")
    for j, (tag, log) in enumerate(
            (("SiC", "../lammps/outputs/logs/log.run_SiC_cascade_1"),
             ("W", "../lammps/outputs/logs/log.run_W_cascade_1"))):
        blocks, _ = parse_blocks(Path(log))
        step = np.concatenate([b["Step"] for b in blocks]) / 1e3
        temp = np.concatenate([b["Temp"] for b in blocks])
        pe = np.concatenate([b["PotEng"] for b in blocks]) / 1e6
        axes[0, j].plot(step, temp, lw=0.7, color="#d94040")
        axes[0, j].set_ylabel("T (K)")
        axes[0, j].text(0.97, 0.85, tag, transform=axes[0, j].transAxes,
                        ha="right", fontsize=7.5, color="0.3")
        axes[1, j].plot(step, pe, lw=0.7, color=PT)
        axes[1, j].set_ylabel(r"$E_{pot}$ ($\times 10^6$ eV)")
        if j == 1:
            axes[1, j].set_xlabel(r"MD step ($\times 10^3$)")
    for ax, s in zip((axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]), "abcd"):
        ax.text(0.02, 0.8, s, transform=ax.transAxes, fontweight="bold",
                fontsize=9)
    fig.tight_layout(h_pad=1.0, w_pad=2.0)
    save(fig, "thermo.pdf")


def fig_stored():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE, 3.8))
    vals = {"Fe": 121.7, "W": 147.5, "SiC": 898.4}
    for i, (m, v) in enumerate(vals.items()):
        ck = "C" if m == "SiC" else m
        ax1.bar(i, v, width=0.6, color=COL[ck])
        ax1.text(i, v + 18, f"{v:.0f}", ha="center", fontsize=6.5)
    ax1.annotate("6.1×", (2, 500), ha="center", fontsize=9)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(vals)
    ax1.set_ylabel("stored energy at 0 K (eV per cascade)")
    letter(ax1, "a")

    data = {"Fe": (0.337, 0.440, 0.035, None),
            "W": (0.337, 0.382, 0.036, None),
            "SiC": (1.483, 2.115, 0.11, 1.748)}
    xt, xl = [], []
    for i, (m, (std, own, err, ed30)) in enumerate(data.items()):
        x0 = i * 3
        ck = "C" if m == "SiC" else m
        ax2.plot(x0, std, "o", ms=4, color=COL[ck])
        ax2.errorbar(x0 + 1, own, yerr=err, fmt="s", ms=3.5, color=COL[ck],
                     capsize=1.5, lw=0.9)
        xt += [x0, x0 + 1]
        xl += [f"{m}\nstd", f"{m}\nown"]
        if ed30 is not None:
            ax2.plot(x0 + 2, ed30, "D", ms=3.5, color=COL[ck], mfc="white",
                     mew=0.9)
            xt.append(x0 + 2)
            xl.append("SiC\n$E_d$=30")
    ax2.set_xticks(xt)
    ax2.set_xticklabels(xl, fontsize=6)
    ax2.set_ylabel("surviving FP / NRT")
    letter(ax2, "b", dx=-0.14)
    fig.tight_layout(w_pad=2.5)
    save(fig, "stored.pdf")


def fig_attribution():
    fig = plt.figure(figsize=(SINGLE, 3.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.65,
                          wspace=0.75)
    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    segs = [("C vac", 215.4, COL["C"], 1.0), ("C int", 305.0, COL["C"], .55),
            ("Si vac", 99.7, COL["Si"], 1.0), ("Si int", 199.6, COL["Si"],
                                               .55),
            ("anti", 58.1, COL["Fe"], .8), ("far", 20.6, "0.6", .6)]
    base = 0
    for i, (lab, v, c, a) in enumerate(segs):
        ax.bar(i, v, bottom=base, width=0.6, color=c, alpha=a)
        ax.text(i, base + v + 12, f"{v:.0f}", ha="center", fontsize=6)
        base += v
    ax.axhline(898.4, color="0.45", lw=0.6, ls=(0, (4, 3)))
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels([s[0] for s in segs], fontsize=6)
    ax.set_ylabel("cumulative stored energy (eV)")
    letter(ax, "a", dx=-0.18)

    ax2.bar([0, 1], [102 / 50, 35 / 20], width=0.55,
            color=[COL["C"], "0.6"])
    ax2.axhline(1, color="0.45", lw=0.6, ls=(0, (4, 3)))
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["C:Si\npopulation", "Si:C\nthreshold"], fontsize=6)
    ax2.text(0, 2.08, "2.04", ha="center", fontsize=6.5)
    ax2.text(1, 1.79, "1.75", ha="center", fontsize=6.5)
    ax2.set_ylabel("ratio")
    letter(ax2, "b", dx=-0.3)

    ax3.bar([0, 1], [5.0, 6.2], width=0.55, color=[COL["C"], COL["Si"]])
    ax3.text(0, 5.12, "5.0", ha="center", fontsize=6.5)
    ax3.text(1, 6.32, "6.2", ha="center", fontsize=6.5)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["per\nC pair", "per\nSi pair"], fontsize=6)
    ax3.set_ylabel("stored energy per FP (eV)")
    letter(ax3, "c", dx=-0.34)
    fig.tight_layout(w_pad=2.2)
    save(fig, "attribution.pdf")


def fig_ed():
    fig = plt.figure(figsize=(SINGLE, 4.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.75,
                          wspace=0.55)
    specs = [("Fe", RES / "ed_results_Fe.csv", None),
             ("W", RES / "ed_results_W.csv", None),
             ("Si", RES / "ed_results_SiC.csv", "Si"),
             ("C", RES / "ed_results_SiC.csv", "C")]
    for i, ((name, path, sp), s) in enumerate(zip(specs, "abcd")):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        v = ed_values(path, sp)
        ax.hist(v, bins=np.arange(0, 260, 12.5), color=COL[name],
                edgecolor="white", linewidth=0.5)
        ax.axvline(v.mean(), color="k", lw=0.7)
        ax.set_xlim(0, 200 if name != "W" else 260)
        ax.set_xlabel("$E_d$ (eV)", fontsize=6.5)
        if i % 2 == 0:
            ax.set_ylabel("directions")
        ax.text(0.95, 0.85,
                f"{name}\n{v.mean():.0f}±{v.std(ddof=1)/len(v)**0.5:.0f} eV",
                transform=ax.transAxes, ha="right", fontsize=6, color="0.25")
        letter(ax, s, dx=-0.3)
    ax = fig.add_subplot(gs[2, :])
    pairs = [("Fe", ed_values(RES / "ed_results_Fe_bisect_upperbound.csv"),
              ed_values(RES / "ed_results_Fe.csv")),
             ("W", ed_values(RES / "ed_results_W_bisect_upperbound.csv"),
              ed_values(RES / "ed_results_W.csv"))]
    for i, (name, bis, ramp) in enumerate(pairs):
        b, r = bis.mean(), ramp.mean()
        ax.plot([b, r], [i, i], color=COL[name], lw=1.4)
        ax.plot(b, i, "o", ms=4.5, color=COL[name], mfc="white", mew=1)
        ax.plot(r, i, "o", ms=4.5, color=COL[name])
        ax.annotate(f"bisection {b:.0f} → ramp {r:.0f} eV", (r, i),
                    xytext=(8, 6), textcoords="offset points", fontsize=6.5,
                    color="0.25")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fe", "W"])
    ax.set_ylim(-0.5, 1.7)
    ax.set_xlabel("sphere-averaged $E_d$ (eV)")
    letter(ax, "e", dx=-0.055)
    save(fig, "ed.pdf")


def fig_pdfgr():
    rows = list(csv.DictReader(open(PDFDIR / "pdf_min_SiC_cascade1_r25.csv")))
    r = np.array([float(x["r_A"]) for x in rows])
    gd = np.array([float(x["G_damaged"]) for x in rows])
    gp = np.array([float(x["G_pristine"]) for x in rows])
    m = r <= 12
    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 4.8))
    ax = axes[0]
    ax.plot(r[m], gp[m], lw=0.6, color="0.55", label="pristine")
    ax.plot(r[m], gd[m], lw=0.6, color=COL["C"], label="damaged")
    ax.set_ylabel(r"$G(r)$ (Å$^{-2}$)")
    ax.legend(ncol=2)
    letter(ax, "a", dx=-0.17)

    ax = axes[1]
    names = [("dG_1-1", "Si-Si", COL["Si"]),
             ("dG_1-2", "Si-C", "0.45"),
             ("dG_2-2", "C-C", COL["C"])]
    dr = r[1] - r[0]
    off = 800.0
    for i, (k, lab, c) in enumerate(names):
        d = np.array([float(x[k]) for x in rows])
        ax.plot(r[m], d[m] + i * off, lw=0.6, color=c)
        ax.text(0.15, i * off + 230, lab, fontsize=6.5, color=c,
                fontweight="bold")
        ax.text(11.9, i * off + 230,
                f"∫|ΔG| = {np.trapezoid(np.abs(d[m]), r[m]):.1f}",
                fontsize=6, color="0.35", ha="right")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("ΔG(r), element-pair partials (offset)")
    letter(ax, "b", dx=-0.17)

    ax = axes[2]
    prow = list(csv.DictReader(open(PDFDIR /
                                    "pipeline_pdf_min_SiC_cascade1_r25.csv")))
    rp = np.array([float(x["r_A"]) for x in prow])
    dproc = np.array([float(x["G_damaged_proc"]) for x in prow])
    pproc = np.array([float(x["G_pristine_proc"]) for x in prow])
    ax.plot(rp, pproc, lw=0.6, color="0.55", label="pristine")
    ax.plot(rp, dproc, lw=0.6, color="#d94040", label="damaged")
    ax.set_ylabel(r"$G(r)$, XPD-processed")
    ax.set_xlabel(r"$r$ (Å)")
    ax.legend(ncol=2)
    letter(ax, "c", dx=-0.17)
    fig.tight_layout(h_pad=1.0)
    save(fig, "pdfgr.pdf")


def fig_adp():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE, 1.9))
    ax1.bar([0, 1], [0.0950, 0.0947], width=0.55,
            color=[COL["Si"], COL["C"]])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Si", "C"], fontsize=6.5)
    ax1.set_ylabel(r"on-site $\langle u^2\rangle$ (Å$^2$)")
    ax1.set_ylim(0, 0.115)
    letter(ax1, "a", dx=-0.42)
    dose = [4e-4, 0.02, 0.5]
    ratio = [1.00, 1.25, 2.8]
    ax2.semilogx(dose[0], ratio[0], "o", ms=4.5, color=COL["C"])
    ax2.errorbar(dose[1], ratio[1], yerr=1.0, fmt="s", ms=3.5, color="0.5",
                 capsize=1.5, lw=0.9)
    ax2.semilogx(dose[2], ratio[2], "D", ms=3.5, color="0.3")
    ax2.axhline(1, color="0.45", lw=0.6, ls=(0, (4, 3)))
    ax2.set_xlabel("dose (dpa)")
    ax2.set_ylabel(r"C/Si $\langle u^2\rangle$ ratio")
    letter(ax2, "b", dx=-0.36)
    fig.tight_layout(w_pad=2.2)
    save(fig, "adp.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_validation()
    fig_renders()
    fig_kinetics()
    fig_thermo()
    fig_stored()
    fig_attribution()
    fig_ed()
    fig_pdfgr()
    fig_adp()


if __name__ == "__main__":
    main()
