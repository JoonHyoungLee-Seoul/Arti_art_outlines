"""
Generate image captions for images in a folder using BLIP and merge them
into a filtered CSV derived from meta.normalized.csv.

This script replicates and extends the functionality from image2text.ipynb:
 - Loads BLIP (Salesforce/blip-image-captioning-base)
 - Captions all images in `Clipped_images/` whose filenames match the `id`
   column in `meta.normalized.csv` (filename stem equals id)
 - Filters the CSV to those ids only
 - Adds a new column `image_description` with the generated caption
 - Writes the result to `meta.normalized.200.csv`

Defaults assume the repository layout:
 - Input images directory: ./Clipped_images
 - Input CSV: ./meta.normalized.csv
 - Output CSV: ./meta.normalized.200.csv

Usage examples:
  python image2text_batch.py
  python image2text_batch.py --images-dir Clipped_images --csv meta.normalized.csv --output meta.normalized.200.csv

Notes:
 - The BLIP model will be downloaded on first run (cached afterwards).
 - The `id` column is read as string to avoid losing leading zeros.
 - Only rows with matching images are kept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image

try:
    import torch
except ImportError as exc:  # Defensive: give a helpful error
    raise SystemExit(
        "torch is required. Please install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
except ImportError as exc:  # Defensive: give a helpful error
    raise SystemExit(
        "transformers is required. Please install with: pip install transformers pillow"
    ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description="Caption images and update CSV with BLIP")
    parser.add_argument(
        "--images-dir",
        default=str(Path("Clipped_images")),
        help="Directory containing images whose filename stem equals CSV id",
    )
    parser.add_argument(
        "--csv",
        default=str(Path("meta.normalized.csv")),
        help="Path to the source CSV with an 'id' column",
    )
    parser.add_argument(
        "--output",
        default=str(Path("meta.normalized.200.csv")),
        help="Output CSV path to write filtered rows with captions",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum new tokens to generate for captions",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Force device selection (default: auto detect)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of rows/images to process after filtering (for quick tests)",
    )
    return parser.parse_args()


def discover_image_id_set(images_dir: Path) -> Tuple[Set[str], Dict[str, Path]]:
    """Return the set of image ids present in a directory and a map id->path.

    We consider standard image extensions; the id is the filename without extension.
    """
    allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
    id_set: Set[str] = set()
    id_to_path: Dict[str, Path] = {}

    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in allowed_exts:
            continue
        image_id = path.stem
        id_set.add(image_id)
        # Prefer first seen; if duplicates across extensions, keep first
        id_to_path.setdefault(image_id, path)

    return id_set, id_to_path


def load_blip(device_preference: Optional[str] = None) -> Tuple[BlipProcessor, BlipForConditionalGeneration, torch.device]:
    """Load BLIP processor and model once and move model to the selected device."""
    if device_preference is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_preference == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(device_preference)

    print(f"Loading BLIP on device: {device}")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # Prefer safetensors to avoid torch.load vulnerability restrictions
    try:
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_safetensors=True
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to load BLIP model. Please upgrade torch to >=2.6 or ensure safetensors weights are available.\n"
            f"Underlying error: {exc}"
        )
    model.to(device)
    model.eval()
    return processor, model, device


def generate_caption(
    image_path: Path,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Generate a caption for a single image path using BLIP."""
    # Open image in RGB mode to avoid issues with palettes or alpha
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.decode(output_ids[0], skip_special_tokens=True)
    return text


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    source_csv_path = Path(args.csv).resolve()
    output_csv_path = Path(args.output).resolve()

    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not source_csv_path.exists():
        raise SystemExit(f"Source CSV not found: {source_csv_path}")
    if output_csv_path.exists() and not args.overwrite:
        print(f"Output exists; use --overwrite to replace: {output_csv_path}")
        return

    # 1) Discover present image ids
    present_id_set, id_to_path = discover_image_id_set(images_dir)
    if not present_id_set:
        raise SystemExit(f"No images found in: {images_dir}")
    print(f"Found {len(present_id_set)} image files in {images_dir}")

    # 2) Load CSV and filter to present ids
    try:
        df = pd.read_csv(source_csv_path, dtype={"id": str})
    except Exception as exc:
        raise SystemExit(f"Failed to read CSV {source_csv_path}: {exc}")

    if "id" not in df.columns:
        raise SystemExit("CSV must contain an 'id' column")

    # Normalize ids as strings and strip whitespace just in case
    df["id"] = df["id"].astype(str).str.strip()
    filtered = df[df["id"].isin(present_id_set)].copy()
    print(f"Filtered CSV rows: {len(filtered)} (from {len(df)})")

    # Optional small-run limit for smoke tests
    if args.limit is not None and args.limit > 0:
        filtered = filtered.head(args.limit).copy()

    # Prepare the new column that will hold generated captions
    filtered["image_description"] = ""

    # 3) Load model once
    processor, model, device = load_blip(args.device)

    # 4) Generate captions
    total = len(filtered)
    for index, row in filtered.iterrows():
        image_id = row["id"]
        image_path = id_to_path.get(image_id)
        if image_path is None:
            # Should not happen due to the filter, but guard regardless
            continue
        try:
            caption = generate_caption(
                image_path=image_path,
                processor=processor,
                model=model,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[WARN] Failed to caption {image_path.name}: {exc}")
            caption = ""

        filtered.at[index, "image_description"] = caption
        # Lightweight progress feedback
        current_row = filtered.index.get_loc(index) + 1
        if current_row % 10 == 0 or current_row == total:
            print(f"Captioned {current_row}/{total}")

    # 5) Write output CSV
    filtered.to_csv(output_csv_path, index=False)
    print(f"Wrote {len(filtered)} rows to: {output_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)


