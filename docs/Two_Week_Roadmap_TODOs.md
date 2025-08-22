# Two-Week Roadmap TODOs (AI Contributor Guide)

This document enumerates detailed, atomic TODOs for AI/code agents to execute the Styled Outline pipeline per the Arti guide. Follow items in order, check each acceptance criterion, and prefer repo-relative paths and idempotent scripts.

## General Rules
- Use repo-relative paths; respect `ART_ROOT` when provided.
- Never commit generated artifacts (cache/outlines, eval/*.json|*.csv, images except `.gitkeep`, weights).
- Keep config in `art_outlines/configs/`; do not hardcode absolute paths.
- For heavy jobs, support `--resume`, `--dry-run`, and `--jobs` flags.
- Log to JSONL under `cache/outlines/logs/YYYYMMDD/*.jsonl`.

---

## Week 1 — Data/Spec Fixation & Student Baseline

### D1: Scaffolding (COMPLETED)
- Verify directory tree `art_outlines/{configs,scripts,data/images,cache/outlines,eval}`
- Initialize configs:
  - `configs/presets.yaml` with presets 8/16/32/64 and postprocess rules
  - `configs/pipeline.yaml` with preprocess/teacher/student/fallback/serve
  - `configs/stylespec_v0.1.md` with initial thresholds
- Validate `data/meta.csv` or generate `meta.normalized.csv` with columns:
  - `id,title,artist,rights,genre,tags,notes`
  - Mapping from raw headers; set `rights=CC0-1.0`
- Add `README_Student.md` (runbook and CLI contracts)
- Acceptance:
  - Files exist and pass basic YAML/MD visual sanity
  - `meta.normalized.csv` header correct

### D2: Teacher Sample & Minimal Generation
- Implement ControlSketch wrapper `scripts/gen_tiera.py` (COMPLETED base):
  - Load config, map preset→K, iterate rows, exec ControlSketch, normalize outputs
  - Output: `cache/outlines/{id}/{preset}/TIERA/final.svg|final.png|thumb_256.png`
  - Logs: `success.jsonl|failures.jsonl|summary.jsonl` per day
  - Flags: `--resume`, `--dry-run`, `--jobs`, `--timeout`, `--use_cpu`, `--controlskt_num_iter`, `--controlskt_save_interval`
  - Optional SDXL dict fallback `--sdxl_dir`
- Environment notes:
  - macOS: install PyTorch CPU/MPS, `pydiffvg`, `controlnet-aux`, `CLIP`, set `HF_HOME`
  - Exclude CUDA-only packages (bitsandbytes/flash-attn/triton), remove `decord` if wheels missing
- Acceptance:
  - Dry-run passes on sample (ok=rows, fail=0)
  - Real run produces artifacts for ≥3 ids locally

### D3: Teacher Refinement & Style Statistics
- Implement `scripts/parse_svg_stats.py`:
  - Input: glob `cache/outlines/*/TIERA/*.svg`
  - Compute per-SVG and aggregate: path count, total length, curvature histogram, direction histogram, closed path ratio, intersection rate, silhouette dominance approximation
  - Output: `eval/style_stats.json` with aggregate μ/σ and hist bins
- Update `stylespec_v0.1.md` numbers from stats (μ±σ, thresholds) — programmatic writer optional
- Acceptance:
  - Running script writes JSON with expected keys and numeric arrays

### D4: Student Baseline (Encoder-Decoder Skeleton)
- Create `scripts/train_student.py` skeleton:
  - Dataloader: pairs (image, Teacher SVG tensorized target)
  - Model: lightweight encoder (e.g., MobileViT/CNN) + transformer decoder
  - Output shape: `K×(x0,y0,c1x,c1y,c2x,c2y,x3,y3,width,pen)` normalized [0,1]
  - CLI flags: `--train_csv`, `--tiera_dir`, `--img_size`, `--k_list`, `--out`
- Add quick unit tests or shape assertions to avoid NaNs
- Acceptance:
  - Single batch forward succeeds; tensor shapes match config

### D5: Losses (Phase 1) & Mini Training
- Implement losses:
  - Assignment (Hungarian) + L1/L2 coordinate loss
  - Rendering loss: pydiffvg raster → Chamfer/LPIPS
- Run a mini overfit (few samples) to verify convergence
- Acceptance:
  - Training log shows decreasing loss; checkpoint saved

### D6: Losses (Phase 2) — Style Distributions & Constraints
- Add histogram losses (KL/EMD) for length/curvature/direction
- Constraints: silhouette dominance ≥ τ, closed ratio ≥ 90%, intersection rate ≤ 3%
- Add vector postprocess: smoothing, short path removal
- Acceptance:
  - Loss curves reflect distribution terms; validation improves style metrics

### D7: Multi‑K & First Bench
- Support K ∈ {8,16,32,64} (multi-head or variable K)
- Implement `scripts/eval_metrics.py`:
  - Geom: Chamfer, SSIM/LPIPS vs Teacher
  - Vector: curvature/length KL, closed path ratio, intersection rate, path/node counts
  - Saliency coverage (optional placeholder)
  - Output: `eval/leaderboard.csv`
- Acceptance:
  - Inference succeeds for all K; leaderboard generated

---

## Week 2 — Optimization/Deployment & Serving

### D8: Speed Optimization & ONNX
- Prune channels/tokens, fuse ops; ensure stable outputs
- `scripts/export_onnx.py`:
  - Export ONNX at 512px, validate with ONNX Runtime (CPU/MPS/OpenVINO acceptable)
  - Optional INT8 (QAT/PTQ)
- Acceptance:
  - ONNX inference parity within tolerance; latency reduced

### D9: Fallback Stylizer
- Implement `scripts/stylizer_fallback.py`:
  - RDP simplification (curvature-weighted), path merge, enforce closed paths
  - Optimize to match `stylespec` histograms and constraints
- Acceptance:
  - On Student failure, stylizer keeps style metrics within thresholds

### D10: Serving Pipeline & Cache Keys
- Implement `scripts/serve_demo.py`:
  - Cache key `(art_id,preset,tier,model_ver,params_hash)`
  - Priority: TIERA > STUDENT > FALLBACK, auto-upgrade when Tier‑A appears
  - UI hooks: options toggle (subject-only, line width), reflected in SVG postprocess
- Acceptance:
  - Local demo serves images; replacement/upgrade logic verified

### D11: Human Evaluation (Mini Test)
- Script to sample 10–15 items, export forms, collect 5-point ratings (ease-to-trace, similarity, satisfaction)
- Aggregate and compare Tier‑A vs Student vs Fallback
- Acceptance:
  - CSV of human scores; adjustments fed back to presets/stylespec

### D12: Hard Case Mining & Retraining
- Identify bottom 10% by metrics/human feedback; generate additional Tier‑A if possible
- Retrain Student; save `ckpt v0.2`
- Acceptance:
  - Metric gains on hard cases; checkpoint updated

### D13: Documentation & Release Prep
- Finalize `README_Student.md`, `stylespec_v0.1.md` (final), script CLI help
- Repro steps reduced to ≤3 commands where possible
- Acceptance:
  - Teammate can reproduce end-to-end

### D14: Sprint Review & Go/No-Go
- Validate target metrics:
  - Style KL ≤ 0.1; Chamfer/LPIPS ≥ 90%; closed ≥ 90%; silhouette ≥ 70%
  - ONNX real-time readiness at 512px on CPU/MPS
  - Human scores within Teacher − 0.3
- If Go: cache Student across dataset; Tier‑A incremental augmentation
- Acceptance:
  - Decision recorded; backlog for next sprint created

---

## Appendices

### A. Environment Notes (macOS, no CUDA)
- Python 3.9.x recommended
- PyTorch CPU/MPS builds only; avoid CUDA indices
- Filter out: `bitsandbytes`, `flash-attn`, `triton`, optionally `decord` if wheels unavailable
- Install: `controlnet-aux`, `git+https://github.com/openai/CLIP.git`, `pydiffvg` (requires `cairo`, `pkg-config`)
- Set `HF_HOME` to writable path

### B. QA Checklist (per PR)
- Lints pass; no absolute paths; configs under `art_outlines/configs/`
- Scripts support `--dry-run`, `--resume`, `--jobs`
- No large artifacts tracked; `.gitattributes` for LFS if needed
