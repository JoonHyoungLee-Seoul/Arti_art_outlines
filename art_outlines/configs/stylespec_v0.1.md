# StyleSpec v0.1 (Initial)

Initial thresholds; to be calibrated with Teacher stats (eval/style_stats.json).

## Stroke Count (by preset)
- Easy ≤ 120
- Standard ≤ 300
- Detailed ≤ 600

## Silhouette Dominance
- ≥ 70% of total stroke length

## Closed Path Ratio
- ≥ 90%

## Curvature/Direction Distribution
- Match Teacher μ±σ: KL/EMD ≤ 0.1

## Intersections (Crossing Rate)
- crossings per stroke ≤ 3%

## Line Width Profile
- Silhouette: 2.5–3.0 px
- Interior: 1.5–2.0 px

## Silhouette-First Rule
- Prefer keeping/generating silhouette strokes; limit interior strokes

## Notes
- These values are placeholders derived from guide; refine via scripts/parse_svg_stats.py outputs.
- Apply during training constraints, fallback stylizer optimization, and postprocess validation.
