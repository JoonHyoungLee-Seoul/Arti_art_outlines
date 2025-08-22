#!/usr/bin/env python3
"""
Teacher (Tier‑A) generation wrapper for ControlSketch.

This CLI reads repository‑relative configuration (pipeline.yaml, presets.yaml),
iterates over a metadata CSV, and for each row produces a high‑quality
SwiftSketch‑style vector sketch using the ControlSketch optimization method.

Key responsibilities:
1) Resolve repo‑relative paths and preset→K mapping.
2) Batch over meta.normalized.csv and locate input image (or SDXL dict).
3) Invoke ControlSketch's object_sketching.py as a subprocess with correct args.
4) Normalize outputs into cache/outlines/{id}/{preset}/TIERA/*.svg|.png
5) Generate thumbnails, write JSONL logs, support resume/dry‑run, and parallel jobs.

Usage example (from README_Student.md):
  env ART_ROOT=. \
  python art_outlines/scripts/gen_tiera.py \
    --input art_outlines/data/images \
    --meta art_outlines/data/meta.normalized.csv \
    --out art_outlines/cache/outlines \
    --preset 32 \
    --engine controlskt \
    --jobs 4

Notes:
- This script assumes ControlSketch is available under the swiftsketch repo
  cloned alongside this project, or otherwise available on disk. Set
  --controlskt_root if it's in a non‑standard location.
- We only implement the 'controlskt' engine for now; 'swiftsketch' is a TODO.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

try:
    import yaml  # PyYAML for reading pipeline/presets
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr
    )
    raise

try:
    from PIL import Image  # for thumbnails
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: Pillow is required. Install with: pip install pillow", file=sys.stderr
    )
    raise


# -------------------------------------------------------------
# Data structures
# -------------------------------------------------------------


@dataclass
class Paths:
    repo_root: Path
    images_dir: Path
    outlines_dir: Path
    presets_file: Path
    stylespec_file: Path
    eval_dir: Path


# -------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------


def resolve_repo_root() -> Path:
    """Resolve repository root using ART_ROOT if provided, else current dir.

    Using a single base avoids absolute paths in config and makes CI portable.
    """
    return Path(os.environ.get("ART_ROOT", ".")).resolve()


def load_pipeline_and_paths(repo_root: Path, pipeline_file: Path) -> Paths:
    """Load pipeline.yaml and return normalized path container.

    The YAML should contain repo‑relative entries in paths.* which we resolve
    against repo_root. We intentionally avoid absolute paths.
    """
    with open(pipeline_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    paths_cfg = (config or {}).get("paths", {})

    def R(rel: str) -> Path:
        return (repo_root / rel).resolve()

    return Paths(
        repo_root=repo_root,
        images_dir=R(paths_cfg.get("images_dir", "art_outlines/data/images")),
        outlines_dir=R(paths_cfg.get("outlines_dir", "art_outlines/cache/outlines")),
        presets_file=R(
            paths_cfg.get("presets_file", "art_outlines/configs/presets.yaml")
        ),
        stylespec_file=R(
            paths_cfg.get("stylespec_file", "art_outlines/configs/stylespec_v0.1.md")
        ),
        eval_dir=R(paths_cfg.get("eval_dir", "art_outlines/eval")),
    )


def load_presets_k(presets_file: Path, preset_value: int) -> Tuple[str, int]:
    """Map a numeric preset (8/16/32/64) to internal preset key and K value.

    We consult presets.yaml's ui_map (e.g., 32→"detailed") and then read
    presets[that_key].k. Returns (preset_key, k).
    """
    with open(presets_file, "r", encoding="utf-8") as f:
        presets = yaml.safe_load(f) or {}

    ui_map = (presets or {}).get("ui_map", {})
    preset_key = ui_map.get(int(preset_value))
    if not preset_key:
        raise ValueError(f"Preset {preset_value} not found in ui_map of {presets_file}")
    preset_info = (presets or {}).get("presets", {}).get(preset_key)
    if not preset_info or "k" not in preset_info:
        raise ValueError(f"Preset key '{preset_key}' missing 'k' in {presets_file}")
    return preset_key, int(preset_info["k"])  # K (num_strokes)


def find_image_for_id(images_dir: Path, art_id: str) -> Optional[Path]:
    """Search for an image file matching the given id with common extensions.

    We try case‑insensitive matches for: .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff.
    """
    candidates = []
    base = art_id
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
        p = images_dir / f"{base}{ext}"
        if p.exists():
            return p
        # Also support lowercase/uppercase inconsistencies by scanning directory once
    # Fallback: scan directory for any filename starting with id + '.'
    for child in images_dir.iterdir():
        if child.is_file() and child.stem == art_id:
            return child
    return None


def find_sdxl_dict_for_id(sdxl_dir: Path, art_id: str) -> Optional[Path]:
    """Locate SDXL dict file by id with .npz or .npy extension, if provided.
    Returns path or None if not found.
    """
    for ext in (".npz", ".npy"):
        p = sdxl_dir / f"{art_id}{ext}"
        if p.exists():
            return p
    return None


def ensure_dir(path: Path) -> None:
    """Create directory tree if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy a file if it exists; return True if copied."""
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    return False


def write_jsonl(path: Path, record: Dict) -> None:
    """Append a JSON record to a .jsonl file (one JSON per line)."""
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_thumbnail(png_path: Path, thumb_path: Path, size: int = 256) -> None:
    """Generate a thumbnail PNG from the given PNG, preserving aspect ratio."""
    try:
        with Image.open(png_path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            ensure_dir(thumb_path.parent)
            im.save(thumb_path)
    except Exception as exc:
        # Non‑fatal: thumbnail generation failure should not abort the pipeline
        print(f"WARN: thumbnail failed for {png_path}: {exc}", file=sys.stderr)


# -------------------------------------------------------------
# ControlSketch integration
# -------------------------------------------------------------


def validate_svg(svg_path: Path) -> bool:
    """Lightweight SVG validation: ensure file exists, has <svg> root and path commands.

    This is not a strict XML validator, but catches empty/invalid outputs quickly.
    """
    try:
        if not svg_path.exists() or svg_path.stat().st_size == 0:
            return False
        text = svg_path.read_text(encoding="utf-8", errors="ignore")
        text_lower = text.lower()
        if "<svg" not in text_lower:
            return False
        # Expect at least one path-like element or a polyline
        has_path = (
            ("<path" in text_lower)
            or ("<polyline" in text_lower)
            or ("<polygon" in text_lower)
        )
        if not has_path:
            return False
        return True
    except Exception:
        return False


def build_controlskt_cmd(
    controlskt_root: Path,
    target_path: Path,
    output_dir: Path,
    num_strokes: int,
    use_cpu: bool = False,
    fix_scale: bool = True,
    render_size: int = 512,
    output_svg_size: int = 512,
    caption: str = "",
    num_iter: Optional[int] = None,
    save_interval: Optional[int] = None,
) -> List[str]:
    """Construct the subprocess command to invoke ControlSketch.

    We directly execute the object_sketching.py file with the required args.
    ControlSketch will create nested output directories under output_dir.
    """
    script = controlskt_root / "ControlSketch" / "object_sketching.py"
    if not script.exists():
        raise FileNotFoundError(f"ControlSketch script not found at {script}")
    cmd = [
        sys.executable,
        str(script),
        "--target",
        str(target_path),
        "--output_dir",
        str(output_dir),
        "--num_strokes",
        str(num_strokes),
        "--use_cpu",
        "1" if use_cpu else "0",
        "--fix_scale",
        "1" if fix_scale else "0",
        "--render_size",
        str(render_size),
        "--output_svg_size",
        str(output_svg_size),
    ]
    if caption:
        cmd += ["--caption", caption]
    # Optionally pass fewer iterations for quick smoke tests
    if num_iter is not None:
        cmd += ["--num_iter", str(num_iter)]
    if save_interval is not None:
        cmd += ["--save_interval", str(save_interval)]
    return cmd


def run_controlskt(
    cmd: List[str], timeout_sec: Optional[int] = None
) -> Tuple[int, str]:
    """Run the ControlSketch subprocess and return (returncode, combined_output)."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        text=True,
    )
    return proc.returncode, proc.stdout


# -------------------------------------------------------------
# Main worker
# -------------------------------------------------------------


def process_one(
    row: Dict[str, str],
    paths: Paths,
    controlskt_root: Path,
    preset_value: int,
    k_value: int,
    out_root: Path,
    resume: bool,
    dry_run: bool,
    sdxl_dir: Optional[Path],
    logs_dir: Path,
    use_cpu: bool,
    timeout_sec: Optional[int],
    cs_num_iter: Optional[int],
    cs_save_interval: Optional[int],
) -> Tuple[str, bool, Optional[str]]:
    """Process a single CSV row and produce Tier‑A outputs.

    Returns a tuple: (art_id, success, error_message_or_None)
    """
    art_id = (row.get("id") or "").strip()
    title = (row.get("title") or "").strip()
    if not art_id:
        return "", False, "missing_id"

    # Build output directory for the prescribed cache structure
    tier_dir = out_root / art_id / str(preset_value) / "TIERA"
    ensure_dir(tier_dir)

    # Skip if resume and final files already exist
    final_svg = tier_dir / "final.svg"
    final_png = tier_dir / "final.png"
    if resume and final_svg.exists() and final_png.exists():
        write_jsonl(
            logs_dir / "resume.jsonl",
            {
                "id": art_id,
                "preset": preset_value,
                "k": k_value,
                "status": "skipped_existing",
                "time": time.time(),
            },
        )
        return art_id, True, None

    # Locate input target: image preferred, fallback to SDXL dict if provided
    target_path = find_image_for_id(paths.images_dir, art_id)
    used_target_type = "image"
    if target_path is None and sdxl_dir:
        target_path = find_sdxl_dict_for_id(sdxl_dir, art_id)
        used_target_type = "sdxl_dict"

    if target_path is None:
        write_jsonl(
            logs_dir / "failures.jsonl",
            {
                "id": art_id,
                "preset": preset_value,
                "k": k_value,
                "status": "missing_input",
                "time": time.time(),
            },
        )
        return art_id, False, "missing_input"

    if dry_run:
        write_jsonl(
            logs_dir / "dryrun.jsonl",
            {
                "id": art_id,
                "preset": preset_value,
                "k": k_value,
                "status": "dry_run",
                "target": str(target_path),
                "time": time.time(),
            },
        )
        return art_id, True, None

    # Invoke ControlSketch into a temp run directory under our tier_dir
    # We let ControlSketch create nested structure, then copy normalized outputs.
    run_dir = tier_dir / "_controlskt_run"
    ensure_dir(run_dir)
    cmd = build_controlskt_cmd(
        controlskt_root=controlskt_root,
        target_path=target_path,
        output_dir=run_dir,
        num_strokes=k_value,
        use_cpu=use_cpu,
        fix_scale=True,
        render_size=512,
        output_svg_size=512,
        caption="",
        num_iter=cs_num_iter,
        save_interval=cs_save_interval,
    )

    rc, out = run_controlskt(cmd, timeout_sec=timeout_sec)
    if rc != 0:
        # Log failure with captured output
        write_jsonl(
            logs_dir / "failures.jsonl",
            {
                "id": art_id,
                "preset": preset_value,
                "k": k_value,
                "status": "controlskt_error",
                "rc": rc,
                "output": out,
                "time": time.time(),
            },
        )
        return art_id, False, f"controlskt_error(rc={rc})"

    # ControlSketch final artifacts live under run_dir/<test_name>/<wandb_name>/
    # We search for final_svg.svg and final_sketch.png and copy them to tier_dir.
    found_svg: Optional[Path] = None
    found_png: Optional[Path] = None
    for root, _, files in os.walk(run_dir):
        for fn in files:
            if fn == "final_svg.svg":
                found_svg = Path(root) / fn
            elif fn == "final_sketch.png":
                found_png = Path(root) / fn
    if not found_svg or not found_png:
        write_jsonl(
            logs_dir / "failures.jsonl",
            {
                "id": art_id,
                "preset": preset_value,
                "k": k_value,
                "status": "missing_final_outputs",
                "time": time.time(),
            },
        )
        return art_id, False, "missing_final_outputs"

    # Normalize to our canonical names
    copy_if_exists(found_svg, final_svg)
    copy_if_exists(found_png, final_png)

    # Basic validation: final_svg exists and is non‑empty
    ok = final_svg.exists() and final_svg.stat().st_size > 0 and final_png.exists()
    if not ok:
        write_jsonl(
            logs_dir / "failures.jsonl",
            {
                "id": art_id,
                "preset": preset_value,
                "k": k_value,
                "status": "validation_failed",
                "time": time.time(),
            },
        )
        return art_id, False, "validation_failed"

    # Create a small thumbnail
    generate_thumbnail(final_png, tier_dir / "thumb_256.png", size=256)

    # Log success
    write_jsonl(
        logs_dir / "success.jsonl",
        {
            "id": art_id,
            "preset": preset_value,
            "k": k_value,
            "status": "ok",
            "target_type": used_target_type,
            "svg": str(final_svg.relative_to(paths.repo_root)),
            "png": str(final_png.relative_to(paths.repo_root)),
            "time": time.time(),
        },
    )
    return art_id, True, None


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Tier‑A generation wrapper."""
    p = argparse.ArgumentParser(
        description="Generate Tier‑A (Teacher) sketches via ControlSketch"
    )
    p.add_argument(
        "--input",
        type=str,
        required=False,
        default="art_outlines/data/images",
        help="Directory containing input images (id.ext)",
    )
    p.add_argument(
        "--meta",
        type=str,
        required=False,
        default="art_outlines/data/meta.normalized.csv",
        help="CSV with columns: id,title,artist,rights,genre,tags,notes",
    )
    p.add_argument(
        "--out",
        type=str,
        required=False,
        default="art_outlines/cache/outlines",
        help="Output root directory for cache/outlines/{id}/{preset}/TIERA",
    )
    p.add_argument(
        "--preset",
        type=int,
        required=False,
        default=32,
        choices=[8, 16, 32, 64],
        help="Preset selector; maps to k via presets.yaml ui_map",
    )
    p.add_argument(
        "--engine",
        type=str,
        required=False,
        default="controlskt",
        choices=["controlskt"],
        help="Teacher engine: currently only 'controlskt' supported",
    )
    p.add_argument(
        "--controlskt_root",
        type=str,
        required=False,
        default="swiftsketch",
        help="Path to swiftsketch repository root (contains ControlSketch/)",
    )
    p.add_argument("--jobs", type=int, required=False, default=2, help="Parallel jobs")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip items with existing final.svg and final.png",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run ControlSketch; log planned actions",
    )
    p.add_argument(
        "--sdxl_dir",
        type=str,
        required=False,
        default="",
        help="Optional SDXL dict directory for missing images",
    )
    p.add_argument(
        "--use_cpu", action="store_true", help="Force ControlSketch to run on CPU"
    )
    p.add_argument(
        "--timeout",
        type=int,
        required=False,
        default=0,
        help="Per‑item timeout seconds (0=none)",
    )
    p.add_argument(
        "--controlskt_num_iter",
        type=int,
        required=False,
        default=0,
        help="Override ControlSketch --num_iter for quick runs (0=default)",
    )
    p.add_argument(
        "--controlskt_save_interval",
        type=int,
        required=False,
        default=0,
        help="Override ControlSketch --save_interval (0=default)",
    )
    return p.parse_args()


def main() -> int:
    """Program entrypoint for batch Tier‑A generation."""
    args = parse_args()

    # Resolve repository root and core paths via pipeline.yaml
    repo_root = resolve_repo_root()
    pipeline_file = repo_root / "art_outlines/configs/pipeline.yaml"
    paths = load_pipeline_and_paths(repo_root, pipeline_file)

    # Resolve preset→K mapping from presets.yaml
    preset_key, k_value = load_presets_k(paths.presets_file, args.preset)

    # Inputs/outputs (allow overrides via CLI)
    images_dir = (repo_root / args.input).resolve()
    meta_csv = (repo_root / args.meta).resolve()
    out_root = (repo_root / args.out).resolve()
    controlskt_root = (repo_root / args.controlskt_root).resolve()
    sdxl_dir = Path(args.sdxl_dir).resolve() if args.sdxl_dir else None
    ensure_dir(out_root)

    # Logs path under cache/outlines/logs/YYYYMMDD.jsonl (rolled by day)
    day = time.strftime("%Y%m%d")
    logs_dir = out_root / "logs" / day
    ensure_dir(logs_dir)

    # Read metadata rows
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Worker wrapper for executor
    def do_row(row: Dict[str, str]):
        return process_one(
            row=row,
            paths=paths,
            controlskt_root=controlskt_root,
            preset_value=args.preset,
            k_value=k_value,
            out_root=out_root,
            resume=args.resume,
            dry_run=args.dry_run,
            sdxl_dir=sdxl_dir,
            logs_dir=logs_dir,
            use_cpu=args.use_cpu,
            timeout_sec=(args.timeout or None),
            cs_num_iter=(args.controlskt_num_iter or None),
            cs_save_interval=(args.controlskt_save_interval or None),
        )

    # Execute with basic parallelism
    total = len(rows)
    ok_count = 0
    fail_count = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        futures = [ex.submit(do_row, row) for row in rows]
        for fut in as_completed(futures):
            art_id, ok, err = fut.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                # Also mirror to stderr for visibility during runs
                print(f"FAIL id={art_id} preset={args.preset}: {err}", file=sys.stderr)

    # Summary line
    summary = {
        "preset": args.preset,
        "k": k_value,
        "total": total,
        "ok": ok_count,
        "fail": fail_count,
        "time": time.time(),
    }
    write_jsonl(logs_dir / "summary.jsonl", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
