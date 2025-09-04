# ARTI Project Structure

## Overview

This repository contains tools and data for AI-powered art processing, specifically focused on outline generation and ControlSketch integration.

## Directory Structure

```
ARTI/
├── art_outlines/                    # 🎨 Original Art Processing Pipeline
│   ├── cache/                       # Generated outlines and processing cache
│   ├── configs/                     # Pipeline configurations (YAML)
│   ├── data/                        # Original datasets and images
│   ├── eval/                        # Evaluation tools
│   └── scripts/                     # Core processing scripts
│
├── enhanced_art_pipeline/           # 🤖 AI-Enhanced Pipeline
│   ├── src/                         # Enhanced scripts with AI capabilities
│   │   └── image2text_batch.py     # BLIP captioning + NLP object extraction
│   ├── data/                        # AI-generated results
│   │   ├── meta.normalized.csv     # → symlink to shared_data/
│   │   └── meta.normalized.200.csv # Enhanced with captions + object_names
│   ├── configs/                     # → symlink to art_outlines/configs/
│   ├── docs/                        # Enhanced pipeline documentation
│   ├── requirements.txt             # Specific dependencies
│   └── README.md                    # Usage guide
│
├── shared_data/                     # 📊 Master Datasets
│   └── meta.normalized.csv         # Master art metadata (9,977 items)
│
├── DexiNed/                        # 🖼️ Edge Detection Submodule
├── swiftsketch/                    # ✏️ SwiftSketch Submodule
├── image_clipping/                 # 📸 Image Processing Tools
│
└── Arti_art_outlines_backup/       # 💾 Safety Backup
```

## Workflows

### 1. Original Art Processing
```bash
cd art_outlines
# Use original configs and scripts
```

### 2. AI-Enhanced Processing  
```bash
cd enhanced_art_pipeline
python src/image2text_batch.py
# Generates captions + object_names for ControlSketch
```

### 3. ControlSketch Integration
```bash
# Use enhanced data directly
python object_sketching.py --target ./data/image.png \
    --object_name "man" \
    --caption "a painting of a man with a beard"
```

## Key Benefits

- 🎯 **Clear separation** of original vs enhanced functionality
- 💾 **No data duplication** via intelligent symlinking  
- 📈 **Scalable** - easy to add more AI pipelines
- 🔗 **Integrated** - seamless ControlSketch workflow
- 📚 **Well-documented** - each component explained