#!/usr/bin/env python3
"""Render cascade damage: defect atoms colored by peak displacement.

    python3 render.py <dump> [--threshold A] [--frame N]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from ovito.io import import_file
from ovito.modifiers import (
    CalculateDisplacementsModifier,
    ColorCodingModifier,
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
    WignerSeitzAnalysisModifier,
)
from ovito.pipeline import FileSource
from ovito.qt_compat import QtCore, QtGui
from ovito.vis import (
    ColorLegendOverlay,
    PythonViewportOverlay,
    TachyonRenderer,
    Viewport,
    ViewportOverlayInterface,
)

import materials


GRADIENTS = {
    "jet": ColorCodingModifier.Jet,
    "hot": ColorCodingModifier.Hot,
    "viridis": ColorCodingModifier.Viridis,
    "rainbow": ColorCodingModifier.Rainbow,
}


class ScaleBarOverlay(ViewportOverlayInterface):
    """Physical-distance scale bar in the bottom-left corner."""

    def __init__(self, length_ang: float, base_point=(0.0, 0.0, 0.0)):
        super().__init__()
        self.length_ang = length_ang
        self.base_point = base_point

    def render(self, canvas, **kwargs):
        frac_h = canvas.project_length(self.base_point, self.length_ang)
        if frac_h is None or frac_h <= 0:
            return
        _, h_px = canvas.logical_size
        bar_px = frac_h * h_px  # fraction of canvas height

        margin = 40
        bar_thickness = 5
        x0 = margin
        y0 = h_px - margin

        with canvas.qt_painter() as painter:
            # Backing panel.
            panel = QtCore.QRect(
                int(x0 - 14),
                int(y0 - bar_thickness - 34),
                int(bar_px + 28),
                60,
            )
            painter.fillRect(panel, QtGui.QColor(0, 0, 0, 140))

            pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(255, 255, 255))

            # Bar and caps.
            painter.drawRect(
                QtCore.QRectF(x0, y0 - bar_thickness, bar_px, bar_thickness))
            painter.drawLine(
                QtCore.QPointF(x0, y0 - bar_thickness - 6),
                QtCore.QPointF(x0, y0 + 4))
            painter.drawLine(
                QtCore.QPointF(x0 + bar_px, y0 - bar_thickness - 6),
                QtCore.QPointF(x0 + bar_px, y0 + 4))

            # Length label.
            font = painter.font()
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(255, 255, 255))
            label_rect = QtCore.QRectF(
                x0 - 14, y0 - bar_thickness - 30, bar_px + 28, 20)
            painter.drawText(
                label_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{self.length_ang:.0f} Å",
            )


class TitleLabelOverlay(ViewportOverlayInterface):
    """Material and PKA-energy label, top-left."""

    def __init__(self, text: str, font_pt: int = 30):
        super().__init__()
        self.text = text
        self.font_pt = font_pt

    def render(self, canvas, **kwargs):
        w_px, _ = canvas.logical_size
        margin = 30

        with canvas.qt_painter() as painter:
            font = painter.font()
            font.setPointSize(self.font_pt)
            font.setBold(True)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            text_rect = metrics.boundingRect(self.text)
            panel = QtCore.QRect(
                int(margin - 12),
                int(margin - 8),
                int(text_rect.width() + 24),
                int(text_rect.height() + 16),
            )
            painter.fillRect(panel, QtGui.QColor(0, 0, 0, 140))
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(
                QtCore.QRectF(margin, margin, w_px, text_rect.height() + 10),
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignTop,
                self.text,
            )


def infer_material(filename: str) -> str:
    """Best-effort material guess from the filename; prefer --material."""
    name = filename.lower()
    if "tib2" in name:
        return "TiB2"
    if "sic" in name:
        return "SiC"
    if "_fe_" in name or name.startswith("fe_"):
        return "Fe"
    if "_w_" in name or name.startswith("w_"):
        return "W"
    if "10kev" in name:
        return "SiC"
    return "Unknown material"


def default_threshold(material: str) -> float:
    """NN distance of the material; atoms past it have left their site."""
    mat = materials.get(material)
    return mat["nn"] if mat else 2.5


def output_stem(material: str, energy: str, dump_path: Path) -> str:
    """Descriptive output stem, e.g. Fe_10keV_relax1."""
    m = re.search(r"(spike|relax)_(\d+)", dump_path.name)
    tag = f"{m.group(1)}{m.group(2)}" if m else dump_path.stem
    return f"{material.split()[0]}_{energy.replace(' ', '')}_{tag}"


def compute_cumulative_max_displacement(
    dump_path: Path, up_to_frame: int) -> np.ndarray:
    """Largest displacement each atom reaches across frames 0..up_to_frame."""
    scan_pipeline = import_file(str(dump_path), sort_particles=True)
    scan_pipeline.modifiers.append(CalculateDisplacementsModifier())

    max_disp = None
    for f in range(up_to_frame + 1):
        data = scan_pipeline.compute(f)
        disp = np.asarray(data.particles["Displacement Magnitude"][...])
        if max_disp is None:
            max_disp = disp.copy()
        else:
            np.maximum(max_disp, disp, out=max_disp)
    return max_disp


def build_pipeline(
    dump_path: Path,
    threshold: float,
    gradient_name: str,
    max_disp_array: np.ndarray,
    color_min: float = None,
    color_max: float = None):
    """Pipeline that keeps defect atoms and colors them by peak severity."""
    pipeline = import_file(str(dump_path), sort_particles=True)
    num_frames = pipeline.source.num_frames

    def inject_max_displacement(frame, data):
        data.particles_.create_property("MaxDisplacement", data=max_disp_array)

    pipeline.modifiers.append(inject_max_displacement)

    # Keep atoms past threshold.
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression=f"MaxDisplacement < {threshold}")
    )
    pipeline.modifiers.append(DeleteSelectedModifier())

    # Color by peak severity.
    gradient_cls = GRADIENTS[gradient_name]
    color_coding = ColorCodingModifier(
        property="MaxDisplacement", gradient=gradient_cls())
    if color_max is not None:
        # Fixed range keeps a few channeled outliers from flattening contrast.
        color_coding.auto_adjust_range = False
        color_coding.start_value = threshold if color_min is None else color_min
        color_coding.end_value = color_max
    else:
        color_coding.auto_adjust_range = True
    pipeline.modifiers.append(color_coding)

    return pipeline, num_frames, color_coding


def count_defects_wigner_seitz(
    dump_path: Path, frame: int, reference: Path = None) -> dict:
    """Vacancy/interstitial/Frenkel-pair counts via Wigner-Seitz analysis.

    Reference is the ideal lattice (pass --reference perfect_lattice_SiC.data); it
    falls back to frame 0 of the trajectory if none is given. per_type
    occupancies are enabled so SiC antisites are distinguishable.
    """
    pipeline = import_file(str(dump_path))
    ws = WignerSeitzAnalysisModifier(per_type_occupancies=True)
    ws.reference = FileSource()
    ws.reference.load(str(reference) if reference else str(dump_path))
    pipeline.modifiers.append(ws)

    data = pipeline.compute(frame)
    vacancies = int(data.attributes["WignerSeitz.vacancy_count"])
    interstitials = int(data.attributes["WignerSeitz.interstitial_count"])
    total = pipeline.compute(0).particles.count
    frenkel_pairs = min(vacancies, interstitials)
    fraction = frenkel_pairs / total if total else float("nan")
    return {
        "vacancies": vacancies,
        "interstitials": interstitials,
        "frenkel_pairs": frenkel_pairs,
        "total": total,
        "fraction": fraction,
    }


def report_defects(dump_path: Path, frame: int, threshold: float,
                   displaced_count: int, max_disp: float, ws: dict):
    """Print Wigner-Seitz defect counts plus the displaced-atom render stats."""
    print(f"--- Defect summary: {dump_path.name}, frame {frame} ---")
    print(f"  Total atoms            : {ws['total']}")
    print(f"  Vacancies (WS)         : {ws['vacancies']}")
    print(f"  Interstitials (WS)     : {ws['interstitials']}")
    print(f"  Surviving Frenkel pairs: {ws['frenkel_pairs']}")
    print(f"  Frenkel-pair fraction  : {ws['fraction']:.6f}")
    print(f"  Displaced atoms shown  : {displaced_count} "
          f"(> {threshold} Ang, cumulative)")
    if displaced_count:
        print(f"  Max displacement ever  : {max_disp:.2f} Ang")


def compute_view_orientation(positions: np.ndarray):
    """Camera orientation (view, up) maximizing point-cloud spread via PCA."""
    if positions is None or len(positions) < 3:
        return None, None

    centered = positions - positions.mean(axis=0)
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
    view_dir = eigvecs[:, 0]
    up_dir = eigvecs[:, -1]

    # Guard near-planar cloud.
    if abs(np.dot(view_dir, up_dir)) > 0.99:
        up_dir = eigvecs[:, 1]

    return view_dir, up_dir


def render_snapshot(
    pipeline,
    data,
    frame: int,
    out_path: Path,
    color_coding,
    scalebar_ang: float,
    label_text: str,
    size=(1600, 1200)):
    """Render one frame with legend, scale bar, and title overlays."""
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Perspective)

    positions = (
        np.asarray(data.particles.positions[...])
        if data.particles.count else None)
    view_dir, up_dir = compute_view_orientation(positions)
    if view_dir is not None:
        vp.camera_dir = tuple(view_dir)
        vp.camera_up = tuple(up_dir)
        print("  Camera auto-oriented from defect point-cloud spread "
              "(view along min-variance axis, up = max-variance axis)")

    vp.zoom_all()

    # Color key.
    legend = ColorLegendOverlay(
        modifier=color_coding,
        title="Max Displacement Ever (Å)",
        orientation=QtCore.Qt.Orientation.Vertical,
        alignment=QtCore.Qt.AlignmentFlag.AlignRight
        | QtCore.Qt.AlignmentFlag.AlignVCenter,
        legend_size=0.35,
        font_size=0.04,
        ticks_enabled=True,
        border_enabled=True,
        background_enabled=True,
        background_color=(1.0, 1.0, 1.0),
    )
    vp.overlays.append(legend)

    # Scale bar, sized against the cell center.
    cell = data.cell
    center = cell[:, 3] + 0.5 * (cell[:, 0] + cell[:, 1] + cell[:, 2])
    scale_overlay = ScaleBarOverlay(
        length_ang=scalebar_ang, base_point=tuple(center))
    vp.overlays.append(PythonViewportOverlay(delegate=scale_overlay))

    # Title label.
    title_overlay = TitleLabelOverlay(text=label_text)
    vp.overlays.append(PythonViewportOverlay(delegate=title_overlay))

    renderer = TachyonRenderer(ambient_occlusion=True, shadows=True)
    vp.render_image(
        filename=str(out_path),
        size=size,
        renderer=renderer,
        background=(1, 1, 1),
        frame=frame,
    )
    pipeline.remove_from_scene()
    print(f"  Snapshot written to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize cascade damage (whole-sim cumulative) with "
                    "legend and scale bar")
    parser.add_argument(
        "dump_file", type=str,
        help="Path to LAMMPS dump/trajectory file")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Displacement threshold in Angstrom (default: material NN "
             "distance, 2.5 if unknown)")
    parser.add_argument(
        "--frame", type=int, default=-1,
        help="Frame to render positions from (default: -1 = last). Damage is "
             "always scanned cumulatively from frame 0 through this frame.")
    parser.add_argument(
        "--scalebar", type=float, default=20.0,
        help="Scale bar length in Angstrom (default: 20)")
    parser.add_argument(
        "--gradient", type=str, default="jet",
        choices=sorted(GRADIENTS.keys()),
        help="Color gradient for displacement severity (default: jet)")
    parser.add_argument(
        "--color-min", type=float, default=None,
        help="Fixed color-scale minimum in Angstrom (default: threshold)")
    parser.add_argument(
        "--color-max", type=float, default=None,
        help="Fixed color-scale maximum in Angstrom; caps channeled outliers "
             "(default: auto-range)")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Ideal-lattice reference for Wigner-Seitz (e.g. "
             "perfect_lattice_SiC.data). Defaults to frame 0 of the trajectory.")
    parser.add_argument(
        "--outdir", type=str, default=".",
        help="Directory to save the rendered snapshot (default: current dir)")
    parser.add_argument(
        "--material", type=str, default=None,
        help="Material label drawn on the image (default: best-effort guess "
             "from filename)")
    parser.add_argument(
        "--energy", type=str, default="10 keV",
        help="Radiation strength label drawn on the image (default: '10 keV')")
    parser.add_argument(
        "--time-ps", type=str, default=None,
        help="Real simulated time label, e.g. '2.26 ps'. Pull from the LAMMPS "
             "log's 'time' column at this frame's step (dt is adaptive, so it "
             "can't be inferred from the dump alone). Omit to leave it off.")
    return parser.parse_args()


def main():
    args = parse_args()

    dump_path = Path(args.dump_file)
    if not dump_path.exists():
        print(f"Error: dump file not found: {dump_path}")
        sys.exit(1)

    probe = import_file(str(dump_path))
    num_frames = probe.source.num_frames
    print(f"  Loaded trajectory: {num_frames} frame(s) in {dump_path.name}")

    frame = args.frame if args.frame >= 0 else (num_frames - 1)
    if frame >= num_frames:
        print(f"Error: requested frame {frame} but only {num_frames} "
              f"frame(s) exist.")
        sys.exit(1)

    material = (
        args.material if args.material
        else infer_material(dump_path.name))
    threshold = (args.threshold if args.threshold is not None
                 else default_threshold(material))
    print(f"  Displacement threshold: {threshold} Ang ({material})")

    print(f"  Scanning frames 0-{frame} for cumulative max displacement...")
    max_disp_array = compute_cumulative_max_displacement(dump_path, frame)

    pipeline, _, color_coding = build_pipeline(
        dump_path, threshold, args.gradient, max_disp_array,
        args.color_min, args.color_max)

    data = pipeline.compute(frame)
    displaced = data.particles.count
    disp = data.particles["MaxDisplacement"][...] if displaced else []
    max_disp = float(np.max(disp)) if len(disp) else 0.0

    reference = Path(args.reference) if args.reference else None
    ws = count_defects_wigner_seitz(dump_path, frame, reference)
    report_defects(dump_path, frame, threshold, displaced, max_disp, ws)

    label_text = f"{material} — {args.energy} PKA"
    if args.time_ps:
        label_text += f" ({args.time_ps})"

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(material, args.energy, dump_path)
    out_path = out_dir / f"{stem}_frame{frame}_render.png"
    render_snapshot(
        pipeline, data, frame, out_path, color_coding, args.scalebar,
        label_text)


if __name__ == "__main__":
    main()
