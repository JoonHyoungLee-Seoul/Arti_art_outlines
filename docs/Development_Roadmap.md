# ARTI Development Roadmap

## Project Goal
Develop a **Student model** that can reproduce SwiftSketch-style artistic outlines for immediate serving, with Tier-A (genuine SwiftSketch/ControlSketch) as quality anchor and automatic promotion pipeline.

## Core Concepts

### Tier System
- **Tier-A (Teacher/High Quality)**
  - High-quality sketches created with **SwiftSketch/ControlSketch**
  - Used primarily for **Student training & anchoring** (representative works can be used for serving)
- **Tier-B (Immediate Serving)**  
  - **Student model** (priority) + **Stylizer fallback**
  - Never expose "raw edges" to users - always **SwiftSketch-style results only**

### StyleSpec Framework
Quantitative style definition including:
- **Stroke count/length distributions**  
- **Curvature profiles**
- **Silhouette dominance ratios**
- **Closed path percentages**
- **Line width profiles**
- **Intersection rates**

### Auto-Promotion Logic
When Tier-A results become available for the same artwork/preset, automatically replace Student cache (promotion).

## Two-Week Development Sprint

### Week 1: Data/Spec Foundation & Student Baseline

#### D1: Project Scaffolding ✅
- [x] Repository structure (`art_outlines/`, `configs/`, `scripts/`, `cache/`)
- [x] `StyleSpec_v0.1.md` initial draft
- [x] `presets.yaml` (8/16/32/64 stroke presets)
- [x] `meta.csv` schema (`id,title,artist,rights,genre,tags`)
- **DoD**: Repository builds successfully, spec documents committed

#### D2: Teacher Sample Generation
- [ ] Extract 150 balanced samples across 3 genres (portrait/architecture/still-life)
- [ ] Generate **SwiftSketch** 32-stroke outputs (fallback to 16-stroke)
- [ ] Collect generation success/failure logs with thumbnails
- **DoD**: Minimum 100+ samples in `cache/outlines/{id}/TIERA/*`

#### D3: Teacher Refinement & Statistics
- [ ] Implement SVG parser (cubic Bezier parameters/length/curvature/closed path ratio)
- [ ] Calculate **style statistics**: length/curvature histograms, silhouette:internal ratios, line intersections
- [ ] Refine `StyleSpec_v0.1.md` based on **Teacher statistics**
- **DoD**: `eval/style_stats.json` generated, StyleSpec values confirmed

#### D4: Student Model Architecture
- [ ] **Input**: RGB(512) + optional (Depth/Saliency toggles)
- [ ] **Output tensor**: `K×(x0,y0,c1x,c1y,c2x,c2y,x3,y3,width,pen)` (normalized 0-1)
- [ ] Dataloader: (image, Teacher-SVG→tensor) pair loading
- **DoD**: Single batch forward pass, output tensor shape validation

#### D5: Loss Functions (Primary) & Mini Training
- [ ] **Coordinate/shape loss**: Matching (Hungarian) + L1/L2
- [ ] **Rendering loss**: pydiffvg raster → Chamfer/LPIPS
- [ ] 2-3 hour mini training for convergence check (overfitting OK)
- **DoD**: Chamfer/SSIM improvement trend graphs vs Teacher saved

#### D6: Loss Functions (Secondary) - Style Distribution
- [ ] Length/curvature/direction histograms → **KL/EMD loss**
- [ ] Silhouette dominance (≥τ), closed path ratio (≥90%) **constraints**
- [ ] Vector postprocessing module (short path removal/smoothing)
- **DoD**: Distribution loss applied, training logs/metrics improvement confirmed

#### D7: Multi-K Support & Primary Benchmark
- [ ] **Multi-head (8/16/32/64)** or variable-K (top-K) implementation
- [ ] Fixed validation set (genre separation), automated metrics scripts
- [ ] **Checkpoint v0.1** saved
- **DoD**: 8/16/32/64 all inference successful, basic metrics table generated

### Week 2: Optimization/Compression & Serving Integration

#### D8: Speed Optimization & ONNX Export
- [ ] Model compression (channel reduction/token reduction), operation fusion
- [ ] **ONNX export** + ORT runtime validation (CPU/DirectML/MPS)
- [ ] INT8 quantization (QAT or PTQ), quality loss assessment
- **DoD**: 512px inference latency reduced, ONNX inference scripts functional

#### D9: Stylizer (Fallback) Implementation
- [ ] RDP simplification (curvature weighted), path merging, closed curve enforcement
- [ ] **Optimization loop** based on StyleSpec (match length/curvature distributions)
- **DoD**: Fallback stylizer minimizes quality degradation when Student fails

#### D10: Serving Pipeline & Cache Management
- [ ] Cache key design `(art_id,preset,tier,model_ver,params_hash)`
- [ ] **Serving priority**: Tier-A > Student > Stylizer
- [ ] "**Cleaner Version**" button → immediate Tier-A replacement if available
- **DoD**: Local server/laptop workflow demo functional

#### D11: Human Evaluation (Mini Test)
- [ ] 10-15 person survey (drawing ease/similarity/satisfaction)
- [ ] Tier-A vs Student vs Stylizer **blind comparison**
- **DoD**: Human scores collected, preset/StyleSpec fine-tuning

#### D12: Hard Case Mining & Retraining
- [ ] Automatic bottom 10%/human dissatisfaction samples **listing**
- [ ] Teacher augmentation (add samples if possible) → Student **retraining**
- **DoD**: Hard case metrics improved, checkpoint v0.2 saved

#### D13: Documentation & Deployment Preparation
- [ ] `README_Student.md` (I/O/loss/metrics/serving API)
- [ ] `StyleSpec_v0.1.md` finalized with example images
- [ ] `scripts/` usage documentation (help/example commands)
- **DoD**: Team members can execute directly (≤3 button clicks)

#### D14: Sprint Review & Go/No-Go Decision
- [ ] Target metrics achievement confirmation
- [ ] Go: Batch cache all data as Tier-B=Student, gradually augment Tier-A
- [ ] No-Go: Create action items for bottleneck/loss weight/architecture modifications
- **DoD**: Decision made, next sprint backlog generated

## Completion Criteria

### Technical Metrics
- **Style Preservation**: Length/curvature distribution KL ≤ 0.1 (vs Teacher)
- **Shape Similarity**: Chamfer/LPIPS ≥90% Teacher level
- **Structural Quality**: Closed path ratio ≥90%, silhouette dominance ≥70%
- **Performance**: 512px ONNX CPU/DirectML/MPS **real-use response** secured
- **Human Evaluation**: Drawing ease/similarity average scores **within Teacher - 0.3**

### StyleSpec v0.1 Initial Values
- **Stroke Count**: Easy≤120 / Standard≤300 / Detailed≤600
- **Silhouette Dominance**: ≥70% of total length
- **Closed Path Ratio**: ≥90%
- **Average Curvature Range**: Within μ±σ (Teacher statistics based)
- **Intersection Rate**: Total intersections/strokes ≤ 3%
- **Line Width Profile**: Silhouette 2.5-3.0px / Internal 1.5-2.0px

## Current Implementation Status

### ✅ Completed
- [x] CUDA pipeline infrastructure
- [x] BiRefNet foreground/background separation
- [x] DexiNed background edge detection
- [x] Outline merging engine
- [x] Complete pipeline automation
- [x] 7 sample dataset processing

### 🔄 In Progress  
- [ ] Student model architecture design
- [ ] Teacher sample generation pipeline
- [ ] StyleSpec quantitative framework
- [ ] Auto-promotion cache system

### 📋 Planned
- [ ] Student training pipeline
- [ ] ONNX optimization and deployment
- [ ] Human evaluation framework
- [ ] Production serving API

---

*This roadmap is actively maintained and updated based on development progress. For implementation details, see `CUDA_Pipeline_Implementation.md`.*