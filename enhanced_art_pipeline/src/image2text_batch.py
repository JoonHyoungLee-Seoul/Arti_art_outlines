"""
Generate image captions for images in a folder using BLIP and merge them
into a filtered CSV derived from meta.normalized.csv.

This script replicates and extends the functionality from image2text.ipynb:
 - Loads BLIP (Salesforce/blip-image-captioning-base)
 - Captions all images in `Clipped_images/` whose filenames match the `id`
   column in `meta.normalized.csv` (filename stem equals id)
 - Filters the CSV to those ids only
 - Adds a new column `caption` with the generated caption
 - Adds a new column `object_name` with the primary object/subject word for ControlSketch
 - Writes the result to `meta.normalized.200.csv`

Defaults assume the repository layout:
 - Input images directory: ../enhanced_art_pipeline/data/Clipped_images
 - Input CSV: ../shared_data/meta.normalized.csv
 - Output CSV: ../enhanced_art_pipeline/data/meta.normalized.200.csv

Usage examples:
  cd enhanced_art_pipeline && python src/image2text_batch.py
  python src/image2text_batch.py --images-dir ../enhanced_art_pipeline/data/Clipped_images --csv ../shared_data/meta.normalized.csv --output ../enhanced_art_pipeline/data/meta.normalized.200.csv

Notes:
 - The BLIP model will be downloaded on first run (cached afterwards).
 - The `id` column is read as string to avoid losing leading zeros.
 - Only rows with matching images are kept.
 - object_name extraction uses NLP (NLTK) with part-of-speech tagging for consistent results.
"""

from __future__ import annotations

import argparse
import re
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

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.corpus import stopwords
except ImportError as exc:  # Defensive: give a helpful error
    raise SystemExit(
        "nltk is required for object extraction. Please install with: pip install nltk"
    ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description="Caption images and update CSV with BLIP")
    parser.add_argument(
        "--images-dir",
        default=str(Path("../art_outlines/data/images")),
        help="Directory containing images whose filename stem equals CSV id",
    )
    parser.add_argument(
        "--csv",
        default=str(Path("../shared_data/meta.normalized.csv")),
        help="Path to the source CSV with an 'id' column",
    )
    parser.add_argument(
        "--output",
        default=str(Path("../enhanced_art_pipeline/data/meta.normalized.200.csv")),
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


def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    required_data = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' 
                          else f'taggers/{data_name}' if 'tagger' in data_name 
                          else f'corpora/{data_name}')
        except LookupError:
            print(f"Downloading NLTK data: {data_name}")
            nltk.download(data_name, quiet=True)


def extract_object_name(caption: str) -> str:
    """Extract the primary object/subject noun from a caption using NLP.
    
    Uses natural language processing techniques for robust object extraction:
    1. Part-of-speech tagging to identify nouns
    2. Stopword filtering to focus on meaningful content
    3. Priority scoring based on position and grammatical role
    4. Canonical mapping for consistency
    
    Args:
        caption: The generated image caption
        
    Returns:
        A single word representing the main object/subject
    """
    if not caption or not isinstance(caption, str):
        return "unknown"
    
    # Clean the caption
    caption = caption.strip()
    
    try:
        # Tokenize and get part-of-speech tags
        tokens = word_tokenize(caption.lower())
        pos_tags = pos_tag(tokens)
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Art-specific stopwords to ignore
        art_stopwords = {'painting', 'picture', 'image', 'photo', 'portrait', 'artwork', 'drawing', 'sketch'}
        
        # Find noun candidates with scoring
        noun_candidates = []
        
        for i, (word, pos) in enumerate(pos_tags):
            # Focus on nouns (NN, NNS) and proper nouns (NNP, NNPS)
            if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                # Skip stopwords and art-specific terms
                if word not in stop_words and word not in art_stopwords and len(word) > 2:
                    # Score based on position (earlier = higher score) and type
                    position_score = len(pos_tags) - i  # Earlier words get higher scores
                    pos_score = 3 if pos in ['NNP', 'NNPS'] else 2 if pos == 'NN' else 1  # Proper nouns preferred
                    
                    # Convert plural to singular for common cases
                    singular_word = word
                    if pos in ['NNS', 'NNPS']:  # Only process actual plurals
                        # Handle irregular plurals first
                        irregular_plurals = {
                            'men': 'man', 'women': 'woman', 'children': 'child',
                            'people': 'person', 'feet': 'foot', 'teeth': 'tooth',
                            'geese': 'goose', 'mice': 'mouse', 'oxen': 'ox',
                            'roses': 'rose', 'lilies': 'lily', 'daisies': 'daisy',
                            'leaves': 'leaf', 'wolves': 'wolf', 'knives': 'knife'
                        }
                        
                        if word in irregular_plurals:
                            singular_word = irregular_plurals[word]
                        elif word.endswith('ies') and len(word) > 4:
                            singular_word = word[:-3] + 'y'
                        elif word.endswith('ves') and len(word) > 4:
                            singular_word = word[:-3] + 'f'
                        elif word.endswith('es') and len(word) > 3 and word[-3] in 'sxzh':
                            singular_word = word[:-2]
                        elif word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
                            singular_word = word[:-1]
                    
                    total_score = position_score + pos_score
                    noun_candidates.append((singular_word, total_score, word))
        
        # Sort by score (highest first)
        noun_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply canonical mapping to the best candidate
        if noun_candidates:
            best_word = noun_candidates[0][0]
            
            # Handle collective nouns by looking for "of [subject]" or prioritize the object over the container
            if best_word in ['group', 'crowd', 'collection', 'set', 'pair', 'flock']:
                # Look for "group of X" pattern
                group_pattern = rf'{best_word}\s+of\s+(\w+)'
                match = re.search(group_pattern, caption.lower())
                if match:
                    subject = match.group(1)
                    # Apply plural conversion if needed
                    if subject in ['men', 'women', 'children', 'people', 'sailors']:
                        irregular_plurals = {'men': 'man', 'women': 'woman', 'children': 'child', 'people': 'person', 'sailors': 'sailor'}
                        subject = irregular_plurals.get(subject, subject)
                    elif subject.endswith('s') and len(subject) > 3 and not subject.endswith('ss'):
                        subject = subject[:-1]  # Simple plural removal
                    return apply_canonical_mapping(subject)
            
            # For cases like "antique vases", prefer the object (vase) over the descriptor (antique)
            if best_word in ['antique', 'ancient', 'modern', 'old', 'new', 'beautiful', 'magnificent']:
                # Look for the next noun in candidates
                if len(noun_candidates) > 1:
                    second_best = noun_candidates[1][0]
                    return apply_canonical_mapping(second_best)
            
            return apply_canonical_mapping(best_word)
        
    except Exception as exc:
        print(f"[WARN] NLP extraction failed: {exc}, falling back to simple extraction")
        # Fallback to simple extraction if NLP fails
        return simple_extract_fallback(caption)
    
    return "object"


def apply_canonical_mapping(word: str) -> str:
    """Apply canonical mapping to normalize object names for ControlSketch consistency."""
    # Mapping variations to canonical forms for better ControlSketch performance
    canonical_map = {
        # People
        'gentleman': 'man', 'male': 'man', 'boy': 'man', 'guy': 'man',
        'lady': 'woman', 'female': 'woman', 'girl': 'woman', 'gal': 'woman',
        'baby': 'child', 'infant': 'child', 'toddler': 'child',
        'figure': 'person', 'individual': 'person',
        
        # Animals
        'puppy': 'dog', 'canine': 'dog',
        'kitten': 'cat', 'feline': 'cat',
        'stallion': 'horse', 'mare': 'horse',
        'eagle': 'bird', 'dove': 'bird', 'owl': 'bird', 'crow': 'bird',
        
        # Objects
        'vessel': 'ship', 'boat': 'ship',
        'house': 'building', 'church': 'building', 'castle': 'building', 'structure': 'building',
        'rose': 'flower', 'lily': 'flower', 'bloom': 'flower', 'blossom': 'flower',
        'oak': 'tree', 'pine': 'tree',
        
        # Nature
        'foliage': 'plant', 'vegetation': 'plant',
        'sky': 'background', 'cloud': 'sky',
        'water': 'background', 'sea': 'water', 'ocean': 'water',
    }
    
    return canonical_map.get(word, word)


def simple_extract_fallback(caption: str) -> str:
    """Simple fallback extraction when NLP fails."""
    caption = caption.lower().strip()
    
    # Look for obvious subjects first
    obvious_subjects = ['woman', 'man', 'child', 'person', 'horse', 'dog', 'cat', 'bird', 
                       'flower', 'tree', 'building', 'ship', 'lion', 'angel']
    
    for subject in obvious_subjects:
        if subject in caption:
            return subject
    
    # Extract first meaningful word that's not a common descriptor
    words = caption.split()
    skip_words = {'a', 'an', 'the', 'of', 'in', 'on', 'with', 'and', 'painting', 'picture', 'portrait'}
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word and len(clean_word) > 2 and clean_word not in skip_words:
            return clean_word
    
    return "object"


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

    # Prepare the new columns that will hold generated captions and object names
    filtered["caption"] = ""
    filtered["object_name"] = ""

    # 3) Ensure NLTK data is available
    ensure_nltk_data()

    # 4) Load model once
    processor, model, device = load_blip(args.device)

    # 5) Generate captions and extract object names
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
            # Extract object name from the caption
            object_name = extract_object_name(caption)
        except Exception as exc:
            print(f"[WARN] Failed to caption {image_path.name}: {exc}")
            caption = ""
            object_name = "unknown"

        filtered.at[index, "caption"] = caption
        filtered.at[index, "object_name"] = object_name
        
        # Lightweight progress feedback
        current_row = filtered.index.get_loc(index) + 1
        if current_row % 10 == 0 or current_row == total:
            print(f"Captioned {current_row}/{total} (latest object: '{object_name}' from '{caption[:50]}...')")

    # 5) Write output CSV
    filtered.to_csv(output_csv_path, index=False)
    print(f"Wrote {len(filtered)} rows to: {output_csv_path}")
    print("Added columns: caption, object_name")
    print("object_name can now be used with ControlSketch: python object_sketching.py --target ./data/image.png --object_name \"[object_name]\" --caption \"[caption]\"")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)