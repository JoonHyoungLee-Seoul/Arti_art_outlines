#!/usr/bin/env python3
"""
Fast Batch Merge for SwiftSketch + DexiNed Outlines
Processes multiple samples in a single Python session to eliminate startup overhead.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def load_swiftsketch_outline(swift_path: str) -> np.ndarray:
    """Load SwiftSketch outline and extract black sketch lines"""
    img = Image.open(swift_path).convert("RGB")
    img_arr = np.array(img)
    black_pixels = np.all(img_arr <= [50, 50, 50], axis=2)
    sketch_mask = (black_pixels * 255).astype(np.uint8)
    return sketch_mask

def load_dexined_edges(dexi_path: str) -> np.ndarray:
    """Load DexiNed background edges"""
    img = Image.open(dexi_path).convert("L")
    return np.array(img)

def create_foreground_mask(swift_sketch: np.ndarray, dilation_size: int = 5) -> np.ndarray:
    """Create figure region mask from SwiftSketch outline"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size*2+1, dilation_size*2+1))
    figure_mask = cv2.dilate(swift_sketch, kernel, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size*4+1, dilation_size*4+1))
    figure_mask = cv2.morphologyEx(figure_mask, cv2.MORPH_CLOSE, kernel_close)
    return figure_mask

def merge_outlines(swift_sketch: np.ndarray, dexi_edges: np.ndarray, 
                  figure_priority: bool = True, 
                  bg_intensity: int = 180,
                  figure_intensity: int = 255) -> np.ndarray:
    """Fast merge function"""
    target_shape = dexi_edges.shape
    swift_resized = cv2.resize(swift_sketch, (target_shape[1], target_shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    merged = np.zeros_like(dexi_edges, dtype=np.uint8)
    
    if figure_priority:
        figure_mask = create_foreground_mask(swift_resized, dilation_size=5)
        bg_edges_masked = dexi_edges.copy()
        bg_edges_masked[figure_mask > 0] = 0
        merged[bg_edges_masked > 0] = bg_intensity
        merged[swift_resized > 0] = figure_intensity
    else:
        merged[dexi_edges > 0] = bg_intensity
        merged[swift_resized > 0] = figure_intensity
    
    return merged

def process_sample(sample_id: str, swift_base_dir: str, dexi_dir: str, output_dir: str,
                  bg_intensity: int = 180, figure_priority: bool = True) -> bool:
    """Process single sample"""
    
    # Build file paths
    swift_path = os.path.join(swift_base_dir, sample_id, f"{sample_id}_32_strokes", "final_sketch.png")
    dexi_path = os.path.join(dexi_dir, f"{sample_id}_bg_edges.png") 
    output_path = os.path.join(output_dir, f"{sample_id}_merged.png")
    
    # Check if files exist
    if not os.path.exists(swift_path):
        print(f"[Skip] {sample_id}: SwiftSketch not found: {swift_path}")
        return False
        
    if not os.path.exists(dexi_path):
        print(f"[Skip] {sample_id}: DexiNed edges not found: {dexi_path}")
        return False
    
    try:
        # Load inputs
        start_time = time.time()
        swift_sketch = load_swiftsketch_outline(swift_path)
        dexi_edges = load_dexined_edges(dexi_path)
        load_time = time.time()
        
        # Merge outlines
        merged_outline = merge_outlines(swift_sketch, dexi_edges, 
                                       figure_priority=figure_priority,
                                       bg_intensity=bg_intensity)
        merge_time = time.time()
        
        # Save result
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(merged_outline).save(output_path)
        save_time = time.time()
        
        # Performance stats
        total_time = save_time - start_time
        outline_pixels = np.count_nonzero(merged_outline)
        
        print(f"[OK] {sample_id}: {total_time*1000:.0f}ms "
              f"(load:{(load_time-start_time)*1000:.0f}ms, "
              f"merge:{(merge_time-load_time)*1000:.0f}ms, "
              f"save:{(save_time-merge_time)*1000:.0f}ms) "
              f"→ {outline_pixels:,} outline pixels")
        
        return True
        
    except Exception as e:
        print(f"[Error] {sample_id}: {e}")
        return False

def process_folder_pairs(swift_folder: str, dexi_folder: str, output_dir: str,
                        bg_intensity: int = 180, figure_priority: bool = True) -> int:
    """
    Process folder pairs - match files by name pattern
    """
    if not os.path.exists(swift_folder):
        print(f"[Error] SwiftSketch folder not found: {swift_folder}")
        return 0
        
    if not os.path.exists(dexi_folder):
        print(f"[Error] DexiNed folder not found: {dexi_folder}")
        return 0
    
    # Find matching pairs
    swift_files = {f for f in os.listdir(swift_folder) if f.endswith('.png')}
    dexi_files = {f for f in os.listdir(dexi_folder) if f.endswith('_bg_edges.png')}
    
    # Extract base names for matching
    swift_bases = {os.path.splitext(f)[0] for f in swift_files}
    dexi_bases = {f.replace('_bg_edges.png', '') for f in dexi_files}
    
    # Find matching pairs
    matching_bases = swift_bases & dexi_bases
    
    if not matching_bases:
        print(f"[Warning] No matching files found between folders")
        print(f"  SwiftSketch files: {sorted(swift_files)[:3]}...")
        print(f"  DexiNed files: {sorted(dexi_files)[:3]}...")
        return 0
    
    print(f"[Info] Found {len(matching_bases)} matching pairs: {sorted(matching_bases)}")
    
    # Process each pair
    success_count = 0
    start_total = time.time()
    
    for base_name in sorted(matching_bases):
        swift_path = os.path.join(swift_folder, f"{base_name}.png")
        dexi_path = os.path.join(dexi_folder, f"{base_name}_bg_edges.png")
        output_path = os.path.join(output_dir, f"{base_name}_merged.png")
        
        try:
            start_time = time.time()
            swift_sketch = load_swiftsketch_outline(swift_path)
            dexi_edges = load_dexined_edges(dexi_path)
            load_time = time.time()
            
            merged_outline = merge_outlines(swift_sketch, dexi_edges,
                                           figure_priority=figure_priority,
                                           bg_intensity=bg_intensity)
            merge_time = time.time()
            
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(merged_outline).save(output_path)
            save_time = time.time()
            
            total_time = save_time - start_time
            outline_pixels = np.count_nonzero(merged_outline)
            
            print(f"[OK] {base_name}: {total_time*1000:.0f}ms → {outline_pixels:,} outline pixels")
            success_count += 1
            
        except Exception as e:
            print(f"[Error] {base_name}: {e}")
    
    end_total = time.time()
    total_time = end_total - start_total
    
    print(f"\n📊 Folder processing: {success_count}/{len(matching_bases)} files, {total_time:.2f}s total")
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description="Fast Batch Merge for SwiftSketch + DexiNed Outlines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific samples by ID
  python batch_merge.py --samples 436691 436708 437972
  
  # Process all available samples (auto-detect)
  python batch_merge.py --all
  
  # Process from custom folders (match by filename)
  python batch_merge.py --swift-folder ./my_sketches/ --dexi-folder ./my_edges/ -o ./results/
  
  # Custom settings
  python batch_merge.py --samples 436332 782306 --bg-intensity 150 --additive
        """
    )
    
    # Processing mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--samples', nargs='+', help='Specific sample IDs to process')
    mode_group.add_argument('--all', action='store_true', help='Process all available samples')
    mode_group.add_argument('--swift-folder', help='SwiftSketch folder (match with DexiNed folder by filename)')
    
    # Required for folder mode
    parser.add_argument('--dexi-folder', help='DexiNed edges folder (required with --swift-folder)')
    parser.add_argument('-o', '--output-dir', default='./merged_outlines',
                       help='Output directory for merged outlines')
    
    # Directories (for sample-based processing)
    parser.add_argument('--swift-dir', default='../swiftsketch/ControlSketch/output_sketches',
                       help='SwiftSketch output base directory (for --samples/--all mode)')
    parser.add_argument('--dexi-dir', default='./bg_outlines_dexined',
                       help='DexiNed edges directory (for --samples/--all mode)')
    
    # Merge options
    parser.add_argument('--bg-intensity', type=int, default=180,
                       help='Background edge intensity (default: 180)')
    parser.add_argument('--additive', action='store_true',
                       help='Use additive mode instead of figure priority')
    
    args = parser.parse_args()
    
    # Validate folder mode requirements
    if args.swift_folder:
        if not args.dexi_folder:
            print("[Error] --dexi-folder is required when using --swift-folder")
            sys.exit(1)
        
        # Process folder pairs
        success_count = process_folder_pairs(
            args.swift_folder,
            args.dexi_folder, 
            args.output_dir,
            bg_intensity=args.bg_intensity,
            figure_priority=not args.additive
        )
        
        if success_count == 0:
            sys.exit(1)
        return
    
    # Sample-based processing (existing logic)
    if args.all:
        # Auto-detect available samples
        swift_samples = set()
        if os.path.exists(args.swift_dir):
            swift_samples = {d for d in os.listdir(args.swift_dir) 
                           if os.path.isdir(os.path.join(args.swift_dir, d)) and d.isdigit()}
        
        dexi_samples = set()
        if os.path.exists(args.dexi_dir):
            dexi_files = [f for f in os.listdir(args.dexi_dir) if f.endswith('_bg_edges.png')]
            dexi_samples = {f.replace('_bg_edges.png', '') for f in dexi_files}
        
        # Only process samples that have both SwiftSketch and DexiNed outputs
        samples = sorted(swift_samples & dexi_samples)
        print(f"[Info] Auto-detected {len(samples)} samples: {samples}")
    else:
        samples = args.samples
    
    if not samples:
        print("[Error] No samples to process!")
        sys.exit(1)
    
    # Process all samples in single session
    print(f"[Info] Starting batch merge for {len(samples)} samples...")
    start_total = time.time()
    
    success_count = 0
    for sample_id in samples:
        success = process_sample(
            sample_id, 
            args.swift_dir, 
            args.dexi_dir, 
            args.output_dir,
            bg_intensity=args.bg_intensity,
            figure_priority=not args.additive
        )
        if success:
            success_count += 1
    
    end_total = time.time()
    total_time = end_total - start_total
    
    print(f"\n🎉 Batch merge completed!")
    print(f"📊 Results: {success_count}/{len(samples)} samples processed successfully")
    print(f"⚡ Total time: {total_time:.2f}s ({total_time/len(samples):.2f}s per sample)")
    print(f"🚀 Speed improvement: ~{10*len(samples)/total_time:.1f}x faster than individual calls")

if __name__ == "__main__":
    main()