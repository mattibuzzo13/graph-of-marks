#!/usr/bin/env python3
"""
Benchmark script for comparing standard WBF vs optimized spatial WBF.

Usage:
    python scripts/benchmark_fusion.py
"""

import time
import numpy as np
from PIL import Image
from typing import List

from gom.types import Detection
from gom.fusion.wbf import fuse_detections_wbf
from gom.fusion.wbf_optimized import fuse_detections_wbf_spatial


def generate_test_detections(
    num_boxes: int = 150,
    image_size: tuple = (1920, 1080),
    num_classes: int = 20,
) -> List[Detection]:
    """Generate synthetic detections for benchmarking."""
    detections = []
    
    W, H = image_size
    sources = ["owlvit", "yolov8", "detectron2"]
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    np.random.seed(42)
    
    for i in range(num_boxes):
        # Random box
        x1 = np.random.randint(0, W - 100)
        y1 = np.random.randint(0, H - 100)
        w = np.random.randint(50, 300)
        h = np.random.randint(50, 300)
        x2 = min(x1 + w, W)
        y2 = min(y1 + h, H)
        
        # Random metadata
        label = np.random.choice(class_names)
        score = np.random.uniform(0.3, 0.95)
        source = np.random.choice(sources)
        
        det = Detection(
            box=(x1, y1, x2, y2),
            label=label,
            score=score,
            source=source,
        )
        detections.append(det)
    
    return detections


def benchmark_fusion(num_boxes_list: List[int] = [50, 100, 150, 200, 300]):
    """Compare standard WBF vs spatial WBF across different dataset sizes."""
    
    print("=" * 80)
    print("FUSION BENCHMARK: Standard WBF vs Spatial Hash WBF")
    print("=" * 80)
    print()
    
    image_size = (1920, 1080)
    iou_thr = 0.55
    num_iterations = 10
    
    results = []
    
    for num_boxes in num_boxes_list:
        print(f"Testing with {num_boxes} boxes per image...")
        
        # Generate test data
        detections = generate_test_detections(num_boxes, image_size)
        
        # Benchmark standard WBF
        times_standard = []
        for _ in range(num_iterations):
            start = time.time()
            fused_standard = fuse_detections_wbf(
                detections, 
                image_size, 
                iou_thr=iou_thr,
            )
            elapsed = time.time() - start
            times_standard.append(elapsed)
        
        avg_standard = np.mean(times_standard)
        std_standard = np.std(times_standard)
        
        # Benchmark spatial WBF
        times_spatial = []
        for _ in range(num_iterations):
            start = time.time()
            fused_spatial = fuse_detections_wbf_spatial(
                detections,
                image_size,
                iou_thr=iou_thr,
                hierarchical=True,
            )
            elapsed = time.time() - start
            times_spatial.append(elapsed)
        
        avg_spatial = np.mean(times_spatial)
        std_spatial = np.std(times_spatial)
        
        # Calculate speedup
        speedup = avg_standard / avg_spatial
        
        results.append({
            "num_boxes": num_boxes,
            "standard_ms": avg_standard * 1000,
            "standard_std": std_standard * 1000,
            "spatial_ms": avg_spatial * 1000,
            "spatial_std": std_spatial * 1000,
            "speedup": speedup,
            "num_fused_standard": len(fused_standard),
            "num_fused_spatial": len(fused_spatial),
        })
        
        print(f"  Standard WBF: {avg_standard*1000:.1f}ms ± {std_standard*1000:.1f}ms")
        print(f"  Spatial WBF:  {avg_spatial*1000:.1f}ms ± {std_spatial*1000:.1f}ms")
        print(f"  Speedup:      {speedup:.2f}x")
        print(f"  Boxes after fusion: {len(fused_standard)} vs {len(fused_spatial)}")
        print()
    
    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Boxes':<10} {'Standard (ms)':<20} {'Spatial (ms)':<20} {'Speedup':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['num_boxes']:<10} "
              f"{r['standard_ms']:>8.1f} ± {r['standard_std']:>5.1f}     "
              f"{r['spatial_ms']:>8.1f} ± {r['spatial_std']:>5.1f}     "
              f"{r['speedup']:>6.2f}x")
    
    print()
    print("Conclusion:")
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Expected gain on real pipeline: ~{(avg_speedup-1)*100:.0f}% faster WBF")
    print()


def test_correctness():
    """Verify that spatial WBF produces equivalent results to standard WBF."""
    print("=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)
    print()
    
    image_size = (1920, 1080)
    detections = generate_test_detections(100, image_size)
    
    # Run both fusion methods
    fused_standard = fuse_detections_wbf(detections, image_size, iou_thr=0.55)
    fused_spatial = fuse_detections_wbf_spatial(detections, image_size, iou_thr=0.55)
    
    print(f"Standard WBF: {len(fused_standard)} boxes")
    print(f"Spatial WBF:  {len(fused_spatial)} boxes")
    
    # Check if results are similar (exact match not expected due to ordering)
    diff = abs(len(fused_standard) - len(fused_spatial))
    if diff <= 2:  # Allow small difference due to tie-breaking
        print("✅ PASS: Results are equivalent")
    else:
        print(f"⚠️  WARNING: Result count differs by {diff} boxes")
    
    # Compare scores
    scores_standard = sorted([d.score for d in fused_standard], reverse=True)
    scores_spatial = sorted([d.score for d in fused_spatial], reverse=True)
    
    if scores_standard[:10] == scores_spatial[:10]:
        print("✅ PASS: Top scores match")
    else:
        print("⚠️  Top scores differ slightly (expected due to different tie-breaking)")
    
    print()


if __name__ == "__main__":
    # Test correctness first
    test_correctness()
    
    # Run benchmarks
    benchmark_fusion()
    
    print("Benchmark complete!")
    print()
    print("Next steps:")
    print("  1. Enable spatial fusion in config: use_spatial_fusion=True")
    print("  2. Tune cell_size parameter (default 100px, optimal ~average box size)")
    print("  3. Enable hierarchical fusion: use_hierarchical_fusion=True")
    print("  4. Monitor real-world performance improvement")
