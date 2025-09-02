#!/usr/bin/env python3
"""
Merge SwiftSketch Figure + DexiNed Background Outlines
Combines SwiftSketch figure outlines with DexiNed background edges into a single clean outline.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

def load_swiftsketch_outline(swift_path: str) -> np.ndarray:
    """
    Load SwiftSketch outline and extract black sketch lines
    Returns: Binary mask where 255 = sketch lines, 0 = background
    """
    img = Image.open(swift_path).convert("RGB")
    img_arr = np.array(img)
    
    # Extract black sketch lines (figure outline)
    # SwiftSketch uses black lines on white background
    black_pixels = np.all(img_arr <= [50, 50, 50], axis=2)  # Allow for slight variations
    
    # Convert to binary mask: 255 for sketch lines, 0 for background
    sketch_mask = (black_pixels * 255).astype(np.uint8)
    
    print(f"[Info] SwiftSketch: {img.size}, sketch pixels: {np.count_nonzero(sketch_mask):,}")
    return sketch_mask

def load_dexined_edges(dexi_path: str) -> np.ndarray:
    """
    Load DexiNed background edges
    Returns: Binary mask where 255 = edge lines, 0 = background
    """
    img = Image.open(dexi_path).convert("L")
    edge_arr = np.array(img)
    
    print(f"[Info] DexiNed: {img.size}, edge pixels: {np.count_nonzero(edge_arr):,}")
    return edge_arr

def create_foreground_mask(swift_sketch: np.ndarray, dilation_size: int = 10) -> np.ndarray:
    """
    Create mask to identify figure area from SwiftSketch outline
    Dilates the sketch lines to create a broader figure region
    """
    # Dilate sketch lines to create figure region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size*2+1, dilation_size*2+1))
    figure_mask = cv2.dilate(swift_sketch, kernel, iterations=1)
    
    # Fill any holes in the figure mask using morphological closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size*4+1, dilation_size*4+1))
    figure_mask = cv2.morphologyEx(figure_mask, cv2.MORPH_CLOSE, kernel_close)
    
    print(f"[Info] Figure mask: {np.count_nonzero(figure_mask):,} pixels")
    return figure_mask

def merge_outlines(swift_sketch: np.ndarray, dexi_edges: np.ndarray, 
                  figure_priority: bool = True, 
                  bg_intensity: int = 180,
                  figure_intensity: int = 255) -> np.ndarray:
    """
    Merge SwiftSketch figure outline with DexiNed background edges
    
    Args:
        swift_sketch: SwiftSketch binary outline (figure)
        dexi_edges: DexiNed binary edges (background)
        figure_priority: If True, figure lines override background edges in overlap areas
        bg_intensity: Intensity for background edges (0-255)
        figure_intensity: Intensity for figure lines (0-255)
    
    Returns: Merged outline image
    """
    target_shape = dexi_edges.shape  # Use DexiNed size as target
    
    # Resize SwiftSketch to match DexiNed dimensions
    swift_resized = cv2.resize(swift_sketch, (target_shape[1], target_shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    print(f"[Info] Resized SwiftSketch from {swift_sketch.shape} to {swift_resized.shape}")
    
    # Create figure mask to suppress background edges near figure
    figure_mask = create_foreground_mask(swift_resized, dilation_size=5)
    
    # Start with background edges at reduced intensity
    merged = np.zeros_like(dexi_edges, dtype=np.uint8)
    
    if figure_priority:
        # Suppress background edges in figure region
        bg_edges_masked = dexi_edges.copy()
        bg_edges_masked[figure_mask > 0] = 0  # Remove background edges near figure
        
        # Add background edges at reduced intensity
        merged[bg_edges_masked > 0] = bg_intensity
        
        # Add figure outline at full intensity (overrides background)
        merged[swift_resized > 0] = figure_intensity
        
        print(f"[Info] Merge mode: Figure priority")
        print(f"[Info] Background edges (after masking): {np.count_nonzero(bg_edges_masked):,}")
        print(f"[Info] Figure lines: {np.count_nonzero(swift_resized):,}")
    else:
        # Simple additive merge
        merged[dexi_edges > 0] = bg_intensity
        merged[swift_resized > 0] = figure_intensity
        
        print(f"[Info] Merge mode: Additive")
    
    total_outline_pixels = np.count_nonzero(merged)
    print(f"[Info] Final merged outline: {total_outline_pixels:,} pixels")
    
    return merged

def process_merge(swift_path: str, dexi_path: str, output_path: str, 
                 figure_priority: bool = True,
                 bg_intensity: int = 180):
    """Process single merge operation"""
    
    # Load inputs
    swift_sketch = load_swiftsketch_outline(swift_path)
    dexi_edges = load_dexined_edges(dexi_path)
    
    # Merge outlines
    merged_outline = merge_outlines(swift_sketch, dexi_edges, 
                                   figure_priority=figure_priority,
                                   bg_intensity=bg_intensity)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(merged_outline).save(output_path)
    print(f"[OK] Saved merged outline: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Merge SwiftSketch Figure + DexiNed Background Outlines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge sample 436332
  python merge_outlines.py \\
    --swift swiftsketch/ControlSketch/output_sketches/436332/436332_32_strokes/final_sketch.png \\
    --dexi image_clipping/bg_outlines_dexined/436332_bg_edges.png \\
    --output merged_outlines/436332_merged.png
  
  # With custom intensities  
  python merge_outlines.py -s swift.png -d dexi.png -o merged.png --bg-intensity 150
        """
    )
    
    # Input files
    parser.add_argument('-s', '--swift', required=True,
                       help='SwiftSketch figure outline PNG file')
    parser.add_argument('-d', '--dexi', required=True, 
                       help='DexiNed background edges PNG file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output merged outline PNG file')
    
    # Merge options
    parser.add_argument('--bg-intensity', type=int, default=180,
                       help='Background edge intensity 0-255 (default: 180)')
    parser.add_argument('--figure-intensity', type=int, default=255,
                       help='Figure outline intensity 0-255 (default: 255)')
    parser.add_argument('--no-figure-priority', action='store_true',
                       help='Disable figure priority mode (simple additive merge)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.swift):
        print(f"[Error] SwiftSketch file not found: {args.swift}")
        sys.exit(1)
        
    if not os.path.exists(args.dexi):
        print(f"[Error] DexiNed file not found: {args.dexi}")
        sys.exit(2)
    
    # Process merge
    figure_priority = not args.no_figure_priority
    process_merge(args.swift, args.dexi, args.output,
                 figure_priority=figure_priority, 
                 bg_intensity=args.bg_intensity)

if __name__ == "__main__":
    main()