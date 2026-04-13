#!/usr/bin/env python
"""
Comparison Visualizer: Teacher vs Student Baseline vs Student+KD vs Ground Truth
Shows all 4 models side-by-side with statistics
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont


def create_comparison_grid(teacher_img, baseline_img, kd_img, gt_img, image_name, stats):
    """Create a 2x2 grid comparing all 4 models"""
    
    # Resize all images to match teacher size
    th, tw = teacher_img.shape[:2]
    
    if baseline_img.shape[:2] != (th, tw):
        baseline_img = cv2.resize(baseline_img, (tw, th))
    if kd_img.shape[:2] != (th, tw):
        kd_img = cv2.resize(kd_img, (tw, th))
    if gt_img.shape[:2] != (th, tw):
        gt_img = cv2.resize(gt_img, (tw, th))
    
    h, w = th, tw
    
    # Create grid (2x2)
    grid = np.zeros((h*2 + 100, w*2 + 100, 3), dtype=np.uint8)
    grid[:, :] = (40, 40, 40)  # Dark background
    
    # Place images
    grid[10:10+h, 10:10+w] = teacher_img
    grid[10:10+h, 10+w+10:10+w+10+w] = baseline_img
    grid[10+h+10:10+h+10+h, 10:10+w] = kd_img
    grid[10+h+10:10+h+10+h, 10+w+10:10+w+10+w] = gt_img
    
    # Add labels with background
    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Labels with colors
    labels = [
        ("TEACHER R50", (10, 10+h//2), (0, 255, 0)),      # Green
        ("STUDENT BASELINE", (10+w+10, 10+h//2), (100, 150, 255)),  # Orange
        ("STUDENT + KD", (10, 10+h+10+h//2), (255, 100, 100)),  # Light blue
        ("GROUND TRUTH", (10+w+10, 10+h+10+h//2), (0, 0, 255))  # Red
    ]
    
    for label, pos, color in labels:
        draw.text(pos, label, fill=color, font=font)
    
    # Add statistics at bottom
    stats_text = f"Image: {image_name} | Teacher Dets: {stats.get('teacher', 0)} | Baseline Dets: {stats.get('baseline', 0)} | KD Dets: {stats.get('kd', 0)} | GT: {stats.get('gt', 0)}"
    draw.text((10, 10+h+10+h+20), stats_text, fill=(200, 200, 200), font=small_font)
    
    grid = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return grid


def load_image(path):
    """Load image safely"""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def get_stats_from_filename(filename):
    """Extract detection count from filename if available"""
    # Format: 000_image_name_comparison.jpg
    stats = {
        'teacher': '?',
        'baseline': '?',
        'kd': '?',
        'gt': '?'
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compare inference results across models')
    parser.add_argument('--teacher_dir', 
                       default='work_dirs/inference_results_r50_final_gt',
                       help='Directory containing teacher inference results')
    parser.add_argument('--baseline_dir',
                       default='work_dirs/inference_results_r18_best_gt',
                       help='Directory containing student baseline results')
    parser.add_argument('--kd_dir',
                       default='work_dirs/inference_results_kd_student_gt',
                       help='Directory containing KD student results')
    parser.add_argument('--out_dir',
                       default='work_dirs/four_model_comparison',
                       help='Output directory for comparison grids')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of images to compare')
    parser.add_argument('--display', action='store_true',
                       help='Display images in viewer')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*80)
    print("🎨 FOUR-MODEL COMPARISON VISUALIZER")
    print("="*80)
    print(f"\n📁 Loading from:")
    print(f"  Teacher:         {args.teacher_dir}")
    print(f"  Student Baseline: {args.baseline_dir}")
    print(f"  Student + KD:     {args.kd_dir}")
    print(f"  Output:           {args.out_dir}")
    print()
    
    # Get list of images from teacher directory
    teacher_files = sorted([f for f in os.listdir(args.teacher_dir) 
                           if f.endswith('_comparison.jpg')])[:args.num_samples]
    
    if not teacher_files:
        print(f"❌ No images found in {args.teacher_dir}")
        return
    
    print(f"📊 Processing {len(teacher_files)} image comparisons...\n")
    
    successful = 0
    failed = 0
    
    for idx, teacher_file in enumerate(teacher_files):
        # Get base name
        base_name = teacher_file.replace('_comparison.jpg', '')
        
        # Load all 4 images
        teacher_img = load_image(os.path.join(args.teacher_dir, teacher_file))
        baseline_file = teacher_file.replace('_comparison.jpg', '_baseline.jpg')
        baseline_img = load_image(os.path.join(args.baseline_dir, baseline_file))
        kd_img = load_image(os.path.join(args.kd_dir, teacher_file))
        
        # Try to find GT image (usually from teacher or baseline dir, right side)
        gt_path = None
        if baseline_img is not None:
            # Extract right half of baseline image for GT
            h, w = baseline_img.shape[:2]
            gt_img = baseline_img[:, w//2:]
        else:
            failed += 1
            print(f"  ⚠️  [{idx+1}/{len(teacher_files)}] {base_name} - Missing baseline image")
            continue
        
        if teacher_img is None or kd_img is None:
            failed += 1
            print(f"  ⚠️  [{idx+1}/{len(teacher_files)}] {base_name} - Missing teacher or KD image")
            continue
        
        try:
            # Extract left half (predictions) from each comparison image
            h, w = teacher_img.shape[:2]
            teacher_pred = teacher_img[:, :w//2]
            baseline_pred = baseline_img[:, :w//2]
            kd_pred = kd_img[:, :w//2]
            
            # Create stats dictionary
            stats = {
                'teacher': teacher_file.count('Pred'),
                'baseline': baseline_img.shape[0],
                'kd': kd_img.shape[0],
                'gt': 'N/A'
            }
            
            # Create comparison grid
            grid = create_comparison_grid(teacher_pred, baseline_pred, kd_pred, gt_img, 
                                         base_name, stats)
            
            # Save result
            out_file = os.path.join(args.out_dir, f'{idx:03d}_{base_name}_4model.jpg')
            cv2.imwrite(out_file, grid)
            
            print(f"  ✅ [{idx+1}/{len(teacher_files)}] {base_name}")
            successful += 1
            
        except Exception as e:
            failed += 1
            print(f"  ❌ [{idx+1}/{len(teacher_files)}] {base_name} - Error: {str(e)}")
    
    print()
    print("="*80)
    print(f"✅ Comparison Complete!")
    print(f"  Successful: {successful}/{len(teacher_files)}")
    print(f"  Failed: {failed}/{len(teacher_files)}")
    print(f"  Output: {args.out_dir}")
    print("="*80)
    print()
    print("📋 Files created:")
    output_files = sorted([f for f in os.listdir(args.out_dir) if f.endswith('_4model.jpg')])
    for f in output_files[:5]:
        print(f"   • {f}")
    if len(output_files) > 5:
        print(f"   ... and {len(output_files)-5} more")
    print()
    print("🎨 Layout (2x2 grid):")
    print("  ┌─────────────────────┬─────────────────────┐")
    print("  │  TEACHER R50        │  STUDENT BASELINE   │")
    print("  │  (66% mAP)          │  (38.2% mAP)        │")
    print("  ├─────────────────────┼─────────────────────┤")
    print("  │  STUDENT + KD       │  GROUND TRUTH       │")
    print("  │  (Epoch 8)          │  (Reference)        │")
    print("  └─────────────────────┴─────────────────────┘")
    print()
    print("💡 How to use:")
    print("  1. Open files from:", args.out_dir)
    print("  2. Each image shows 4 models side-by-side")
    print("  3. Green boxes = model predictions")
    print("  4. Blue boxes = ground truth")
    print("="*80)


if __name__ == '__main__':
    main()
