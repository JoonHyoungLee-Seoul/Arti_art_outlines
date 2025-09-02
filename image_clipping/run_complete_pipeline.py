#!/usr/bin/env python3
"""
Complete DexiNed + SwiftSketch Outline Pipeline
Runs the full pipeline: Background extraction → Edge detection → Merge with SwiftSketch
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_pipeline_for_sample(sample_id: str, 
                           input_image: str,
                           swiftsketch_output: str,
                           output_dir: str = "./pipeline_output") -> str:
    """
    Run complete pipeline for a single sample
    
    Args:
        sample_id: Sample identifier (e.g., '436332')
        input_image: Original input image path
        swiftsketch_output: SwiftSketch final_sketch.png path
        output_dir: Output directory for pipeline results
    
    Returns: Path to final merged outline
    """
    
    print(f"[Pipeline] Processing sample {sample_id}")
    
    # Create output directories
    bg_dir = os.path.join(output_dir, "backgrounds")
    fg_dir = os.path.join(output_dir, "foregrounds")  
    edges_dir = os.path.join(output_dir, "background_edges")
    merged_dir = os.path.join(output_dir, "merged_outlines")
    
    for dir_path in [bg_dir, fg_dir, edges_dir, merged_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Step 1: Extract foreground and background
    print(f"[Step 1] Extracting foreground/background...")
    cmd1 = [
        sys.executable, "run_cutout.py",
        "-i", input_image,
        "--fg-dir", fg_dir,
        "--bg-dir", bg_dir
    ]
    
    result1 = subprocess.run(cmd1, cwd="image_clipping", capture_output=True, text=True)
    if result1.returncode != 0:
        print(f"[Error] Step 1 failed: {result1.stderr}")
        return None
        
    bg_file = os.path.join(bg_dir, f"{sample_id}_bg.png")
    if not os.path.exists(bg_file):
        print(f"[Error] Background file not created: {bg_file}")
        return None
    
    # Step 2: Extract background edges with DexiNed
    print(f"[Step 2] Extracting background edges...")
    cmd2 = [
        sys.executable, "dexined_bg_cuda.py",
        "-i", bg_file,
        "-o", edges_dir
    ]
    
    result2 = subprocess.run(cmd2, cwd="image_clipping", capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"[Error] Step 2 failed: {result2.stderr}")
        return None
        
    edges_file = os.path.join(edges_dir, f"{sample_id}_bg_edges.png")
    if not os.path.exists(edges_file):
        print(f"[Error] Edges file not created: {edges_file}")
        return None
    
    # Step 3: Merge SwiftSketch + DexiNed outlines
    print(f"[Step 3] Merging outlines...")
    merged_file = os.path.join(merged_dir, f"{sample_id}_final_outline.png")
    
    cmd3 = [
        sys.executable, "merge_outlines.py",
        "--swift", swiftsketch_output,
        "--dexi", edges_file,
        "--output", merged_file
    ]
    
    result3 = subprocess.run(cmd3, cwd="image_clipping", capture_output=True, text=True)
    if result3.returncode != 0:
        print(f"[Error] Step 3 failed: {result3.stderr}")
        return None
    
    print(f"[Pipeline] ✅ Complete! Final outline: {merged_file}")
    return merged_file

def main():
    parser = argparse.ArgumentParser(
        description="Complete DexiNed + SwiftSketch Outline Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_complete_pipeline.py \\
    --sample 436332 \\
    --input image_clipping/images/436332.jpg \\
    --swift swiftsketch/ControlSketch/output_sketches/436332/436332_32_strokes/final_sketch.png \\
    --output ./complete_pipeline_output
        """
    )
    
    parser.add_argument('--sample', required=True,
                       help='Sample ID (e.g., 436332)')
    parser.add_argument('--input', required=True,
                       help='Original input image')
    parser.add_argument('--swift', required=True,
                       help='SwiftSketch final_sketch.png path')
    parser.add_argument('--output', default='./pipeline_output',
                       help='Pipeline output directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"[Error] Input image not found: {args.input}")
        sys.exit(1)
        
    if not os.path.exists(args.swift):
        print(f"[Error] SwiftSketch output not found: {args.swift}")
        sys.exit(2)
    
    # Run pipeline
    final_outline = run_pipeline_for_sample(
        args.sample, 
        args.input, 
        args.swift, 
        args.output
    )
    
    if final_outline:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Final merged outline: {final_outline}")
    else:
        print(f"\n❌ Pipeline failed!")
        sys.exit(3)

if __name__ == "__main__":
    main()