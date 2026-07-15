#!/usr/bin/env python3
"""Animate cascade damage over time: per-frame cumulative displacement.

    python3 animate.py <dump> [--every N] [--fast] [--fps F]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ovito.io import import_file
from ovito.modifiers import CalculateDisplacementsModifier
from ovito.qt_compat import QtCore
from ovito.vis import (
    ColorLegendOverlay,
    PythonViewportOverlay,
    TachyonRenderer,
    Viewport,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import (
    GRADIENTS,
    ScaleBarOverlay,
    TitleLabelOverlay,
    build_pipeline,
    compute_view_orientation,
    default_threshold,
    infer_material,
    output_stem,
)


def cumulative_snapshots(dump_path: Path, frames):
    """Running max displacement (0..f) captured at each requested frame."""
    targets = set(frames)
    last = max(frames)
    scan = import_file(str(dump_path), sort_particles=True)
    scan.modifiers.append(CalculateDisplacementsModifier())

    running = None
    snaps = {}
    for f in range(last + 1):
        data = scan.compute(f)
        disp = np.asarray(data.particles["Displacement Magnitude"][...])
        running = disp.copy() if running is None else np.maximum(running, disp)
        if f in targets:
            snaps[f] = running.copy()
    return snaps


def parse_log_time(log_path: Path):
    """Step -> time (ps) from a LAMMPS log's thermo table (Step, Time cols)."""
    steps, times = [], []
    cols = None
    for line in Path(log_path).read_text().splitlines():
        parts = line.split()
        if "Step" in parts and "Time" in parts:
            cols = (parts.index("Step"), parts.index("Time"))
            continue
        if cols is None:
            continue
        try:
            steps.append(int(parts[cols[0]]))
            times.append(float(parts[cols[1]]))
        except (ValueError, IndexError):
            cols = None  # table ended
    order = np.argsort(steps)
    s, t = np.array(steps)[order], np.array(times)[order]
    uniq = np.concatenate(([True], np.diff(s) > 0))
    return s[uniq], t[uniq]


def build_time_map(logfile, frame_steps, time_end, pka_step):
    """Frame timestep -> ps since the PKA. Log-interpolated or linear."""
    if logfile:
        s, t = parse_log_time(Path(logfile))
        t0 = float(np.interp(pka_step, s, t))
        return lambda step: float(np.interp(step, s, t)) - t0
    if time_end is not None:
        s0, s1 = frame_steps[0], frame_steps[-1]
        span = s1 - s0
        return lambda step: time_end * (step - s0) / span if span else 0.0
    return None


def derive_color_max(snap_final: np.ndarray, threshold: float) -> float:
    """Robust upper color bound from the final displaced population."""
    displaced = snap_final[snap_final >= threshold]
    if not len(displaced):
        return threshold * 4.0
    return float(np.percentile(displaced, 98))


def build_viewport(dump_path, threshold, gradient, snap_final, frame_final,
                   color_min, color_max, scalebar_ang, cell,
                   title_pt, legend_size, legend_font):
    """Fixed camera + persistent overlays shared by every animation frame."""
    pipeline, _, cc = build_pipeline(
        dump_path, threshold, gradient, snap_final, color_min, color_max)
    data = pipeline.compute(frame_final)

    vp = Viewport(type=Viewport.Type.Perspective)
    positions = (
        np.asarray(data.particles.positions[...])
        if data.particles.count else None)
    view_dir, up_dir = compute_view_orientation(positions)
    if view_dir is not None:
        vp.camera_dir = tuple(view_dir)
        vp.camera_up = tuple(up_dir)

    pipeline.add_to_scene()
    vp.zoom_all()
    cam_pos, fov = vp.camera_pos, vp.fov * 1.25  # zoom out for overlay clearance
    pipeline.remove_from_scene()

    legend = ColorLegendOverlay(
        modifier=cc,
        title="Max Displacement Ever (Å)",
        orientation=QtCore.Qt.Orientation.Vertical,
        alignment=QtCore.Qt.AlignmentFlag.AlignRight
        | QtCore.Qt.AlignmentFlag.AlignVCenter,
        legend_size=legend_size,
        font_size=legend_font,
        ticks_enabled=True,
        border_enabled=True,
        background_enabled=True,
        background_color=(1.0, 1.0, 1.0),
    )
    center = cell[:, 3] + 0.5 * (cell[:, 0] + cell[:, 1] + cell[:, 2])
    scale_overlay = ScaleBarOverlay(
        length_ang=scalebar_ang, base_point=tuple(center))
    title_overlay = TitleLabelOverlay(text="", font_pt=title_pt)

    vp.overlays.append(legend)
    vp.overlays.append(PythonViewportOverlay(delegate=scale_overlay))
    vp.overlays.append(PythonViewportOverlay(delegate=title_overlay))

    return vp, cam_pos, fov, title_overlay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate cumulative cascade damage across frames as a GIF")
    parser.add_argument("dump_file", type=str)
    parser.add_argument("--threshold", type=float, default=None,
                        help="displacement cut in Ang (default: material NN)")
    parser.add_argument("--every", type=int, default=1,
                        help="render every Nth frame (default: 1)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1,
                        help="last frame (default: -1 = final)")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--size", type=int, nargs=2, default=(1400, 1050),
                        metavar=("W", "H"))
    parser.add_argument("--fast", action="store_true",
                        help="disable ambient occlusion and shadows for speed")
    parser.add_argument("--scalebar", type=float, default=20.0)
    parser.add_argument("--gradient", type=str, default="jet",
                        choices=sorted(GRADIENTS.keys()))
    parser.add_argument("--color-min", type=float, default=None)
    parser.add_argument("--color-max", type=float, default=None,
                        help="fixed color max (default: 98th pct of final)")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--material", type=str, default=None)
    parser.add_argument("--energy", type=str, default="10 keV")
    parser.add_argument("--logfile", type=str, default=None,
                        help="LAMMPS log for exact step->time labels")
    parser.add_argument("--time-end", type=float, default=None,
                        help="total ps at last frame (linear label if no log)")
    parser.add_argument("--pka-step", type=int, default=None,
                        help="timestep of PKA launch = t0 (default: first frame)")
    parser.add_argument("--title-pt", type=int, default=16,
                        help="title font size (default: 16)")
    parser.add_argument("--legend-size", type=float, default=0.30)
    parser.add_argument("--legend-font", type=float, default=0.038)
    return parser.parse_args()


def main():
    args = parse_args()

    dump_path = Path(args.dump_file)
    if not dump_path.exists():
        print(f"Error: dump file not found: {dump_path}")
        sys.exit(1)

    probe = import_file(str(dump_path))
    num_frames = probe.source.num_frames
    end = args.end if args.end >= 0 else num_frames - 1
    frames = list(range(args.start, end + 1, args.every))
    if not frames:
        print("Error: no frames selected.")
        sys.exit(1)
    print(f"  {dump_path.name}: {num_frames} frames, rendering {len(frames)} "
          f"(frames {frames[0]}-{frames[-1]}, every {args.every})")

    material = args.material or infer_material(dump_path.name)
    threshold = (args.threshold if args.threshold is not None
                 else default_threshold(material))
    print(f"  Displacement threshold: {threshold} Ang ({material})")

    print("  Scanning cumulative displacement...")
    snaps = cumulative_snapshots(dump_path, frames)

    color_min = threshold if args.color_min is None else args.color_min
    color_max = (args.color_max if args.color_max is not None
                 else derive_color_max(snaps[frames[-1]], threshold))
    print(f"  Color range fixed: {color_min:.1f}-{color_max:.1f} Ang")

    cell = probe.compute(frames[-1]).cell

    frame_steps = [int(probe.compute(f).attributes.get("Timestep", f))
                   for f in frames]
    pka_step = args.pka_step if args.pka_step is not None else frame_steps[0]
    time_of = build_time_map(args.logfile, frame_steps, args.time_end, pka_step)
    if time_of:
        print(f"  Time labels: {time_of(frame_steps[0]):.2f} -> "
              f"{time_of(frame_steps[-1]):.2f} ps since PKA")

    vp, cam_pos, fov, title_overlay = build_viewport(
        dump_path, threshold, args.gradient, snaps[frames[-1]],
        frames[-1], color_min, color_max, args.scalebar, cell,
        args.title_pt, args.legend_size, args.legend_font)

    renderer = TachyonRenderer(
        ambient_occlusion=not args.fast, shadows=not args.fast)

    # Unique per run: concurrent renders must not share frame files.
    scratch = Path(tempfile.mkdtemp(prefix="cascade_anim_"))
    png_paths = []
    for i, f in enumerate(frames):
        pipeline, _, _ = build_pipeline(
            dump_path, threshold, args.gradient, snaps[f],
            color_min, color_max)
        data = pipeline.compute(f)
        step = frame_steps[i]
        if time_of:
            tag = f"t = {time_of(step):.1f} ps"
        else:
            tag = f"step {step}"
        title_overlay.text = f"{material} — {args.energy} PKA  ({tag})"

        pipeline.add_to_scene()
        vp.camera_pos, vp.fov = cam_pos, fov
        out_png = scratch / f"frame_{i:04d}.png"
        vp.render_image(
            filename=str(out_png), size=tuple(args.size), renderer=renderer,
            background=(1, 1, 1), frame=f)
        pipeline.remove_from_scene()
        png_paths.append(out_png)
        print(f"    [{i + 1}/{len(frames)}] frame {f} "
              f"({data.particles.count} displaced)")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = (out_dir / f"{output_stem(material, args.energy, dump_path)}"
                          "_evolution.gif")
    imgs = [Image.open(p).convert("RGB") for p in png_paths]
    imgs[0].save(
        gif_path, save_all=True, append_images=imgs[1:],
        duration=1000.0 / args.fps, loop=0, optimize=True)
    print(f"  Animation written to: {gif_path}")


if __name__ == "__main__":
    main()
