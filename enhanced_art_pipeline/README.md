# Enhanced Art Pipeline

This directory contains enhanced tools for processing art datasets with AI-powered image captioning and object name extraction.

## Features

- **AI Image Captioning**: Generate descriptive captions using BLIP model
- **Smart Object Extraction**: NLP-powered extraction of primary subjects for ControlSketch
- **Batch Processing**: Process large datasets efficiently
- **ControlSketch Integration**: Optimized for `python object_sketching.py` workflow

## Directory Structure

```
enhanced_art_pipeline/
├── src/                           # Source code
│   └── image2text_batch.py       # Main batch processing script
├── data/                          # Generated/processed data
│   ├── meta.normalized.csv       # Symlink to shared master data
│   └── meta.normalized.200.csv   # Enhanced data with captions + object_names
├── configs/
│   └── original/                 # Symlink to ../art_outlines/configs/
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate captions and object names**:
   ```bash
   cd enhanced_art_pipeline
   python src/image2text_batch.py
   ```

3. **Use with ControlSketch**:
   ```bash
   python object_sketching.py --target ./data/image.png --object_name "man" --caption "a painting of a man with a beard"
   ```

## Output

The script generates two new columns:
- `caption`: AI-generated descriptive text
- `object_name`: Single word for the primary subject (optimized for ControlSketch)

## Dependencies

See `requirements.txt` for specific versions. The main dependencies are:
- PyTorch + Transformers (for BLIP model)
- NLTK (for object name extraction)
- Pandas (for data processing)