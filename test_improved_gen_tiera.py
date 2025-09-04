#!/usr/bin/env python3
"""
Test script for the improved gen_tiera.py with high-quality mode.
This script tests a single image to verify the improvements work correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_improved_gen_tiera():
    """Test the improved gen_tiera.py with high-quality settings."""
    
    # Test with a single image from the dataset
    test_image_id = "435573"  # Flora and Zephyr painting
    
    # Create test command with high-quality mode
    cmd = [
        sys.executable,
        "art_outlines/scripts/gen_tiera.py",
        "--input", "enhanced_art_pipeline/data/Clipped_images",
        "--meta", "enhanced_art_pipeline/data/split_csvs/meta.normalized.200_1.csv",
        "--out", "art_outlines/cache/test_outlines",
        "--preset", "32",
        "--object_name_column", "object_name",
        "--caption_column", "caption",
        "--jobs", "1",
        "--high_quality_mode",  # Enable high-quality mode
        "--keep_intermediates",  # Keep intermediate files for inspection
        "--verbose",  # Enable verbose output with progress bars
        "--controlskt_save_interval", "100",  # Use high-quality save interval
    ]
    
    print("Testing improved gen_tiera.py with high-quality mode...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        print()
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            print()
        
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            
            # Check if output files were created
            output_dir = Path("art_outlines/cache/test_outlines")
            if output_dir.exists():
                print(f"✅ Output directory created: {output_dir}")
                
                # List contents
                for item in output_dir.rglob("*"):
                    if item.is_file():
                        print(f"  📄 {item.relative_to(output_dir)}")
            else:
                print("❌ Output directory not found")
        else:
            print("❌ Test failed")
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 30 minutes")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_improved_gen_tiera()