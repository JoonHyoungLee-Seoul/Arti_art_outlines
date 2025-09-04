#!/usr/bin/env python3
"""
Performance Benchmark: Individual vs Batch Merge Processing
Demonstrates the speed improvement of batch processing over individual Python calls.
"""

import os
import time
import subprocess
import tempfile

def benchmark_individual_calls(samples: list, swift_dir: str, dexi_dir: str) -> float:
    """Benchmark individual Python process calls"""
    print("🐌 Testing individual calls (old method)...")
    
    start_time = time.time()
    success_count = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for sample in samples:
            swift_path = os.path.join(swift_dir, sample, f"{sample}_32_strokes", "final_sketch.png")
            dexi_path = os.path.join(dexi_dir, f"{sample}_bg_edges.png")
            output_path = os.path.join(temp_dir, f"{sample}_test.png")
            
            if os.path.exists(swift_path) and os.path.exists(dexi_path):
                cmd = [
                    "python", "merge_outlines.py",
                    "--swift", swift_path,
                    "--dexi", dexi_path, 
                    "--output", output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                      cwd="/home/elicer/ARTI/image_clipping")
                if result.returncode == 0:
                    success_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   ⏱️ Individual calls: {total_time:.2f}s for {success_count} samples")
    print(f"   📊 Average: {total_time/len(samples):.2f}s per sample")
    
    return total_time

def benchmark_batch_processing(samples: list, swift_dir: str, dexi_dir: str) -> float:
    """Benchmark batch processing"""
    print("🚀 Testing batch processing (new method)...")
    
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python", "batch_merge.py",
            "--samples"] + samples + [
            "--swift-dir", swift_dir,
            "--dexi-dir", dexi_dir,
            "--output-dir", temp_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True,
                              cwd="/home/elicer/ARTI/image_clipping")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   ⏱️ Batch processing: {total_time:.2f}s for {len(samples)} samples")
    print(f"   📊 Average: {total_time/len(samples):.2f}s per sample")
    
    return total_time

def main():
    print("🧪 Performance Benchmark: Individual vs Batch Merge")
    print("=" * 60)
    
    # Test samples
    samples = ["436332", "436691", "436708", "437972", "782306"]
    swift_dir = "../swiftsketch/ControlSketch/output_sketches"
    dexi_dir = "./bg_outlines_dexined"
    
    print(f"📋 Testing {len(samples)} samples: {samples}")
    print()
    
    # Benchmark individual calls
    individual_time = benchmark_individual_calls(samples, swift_dir, dexi_dir)
    print()
    
    # Benchmark batch processing
    batch_time = benchmark_batch_processing(samples, swift_dir, dexi_dir)
    print()
    
    # Calculate improvement
    if batch_time > 0:
        speedup = individual_time / batch_time
        time_saved = individual_time - batch_time
        
        print("📈 Performance Summary:")
        print(f"   🐌 Individual calls: {individual_time:.2f}s")
        print(f"   🚀 Batch processing: {batch_time:.2f}s")
        print(f"   ⚡ Speed improvement: {speedup:.1f}x faster")
        print(f"   💾 Time saved: {time_saved:.2f}s ({100*time_saved/individual_time:.1f}%)")
        
        if speedup > 10:
            print("   🎉 Significant performance improvement achieved!")
        elif speedup > 5:
            print("   ✅ Good performance improvement")
        else:
            print("   ⚠️ Marginal improvement - check for bottlenecks")

if __name__ == "__main__":
    main()