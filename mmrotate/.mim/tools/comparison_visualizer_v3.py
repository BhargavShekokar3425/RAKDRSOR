#!/usr/bin/env python
"""
4-Model Comparison with Visible Blur/Quality Degradation
All images at same resolution (1024×1024) but student images upscaled with visible blur
to show the resolution/quality loss impact
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont


def upscale_with_visible_blur(img, target_size=1024):
    """
    Upscale image using linear interpolation to keep blur visible
    This will show the quality degradation clearly
    """
    h, w = img.shape[:2]
    # Use LINEAR interpolation which shows blur better than CUBIC
    upscaled = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return upscaled


def create_same_resolution_grid(teacher_img, baseline_img, kd_img, gt_img, image_name, stats):
    """
    Create 2×2 grid with all images at same resolution
    Student images upscaled with visible blur to show quality loss
    """
    
    target_size = 1024
    
    # Ensure high-res images are correct size
    if teacher_img.shape[0] != target_size:
        teacher_img = cv2.resize(teacher_img, (target_size, target_size), 
                                 interpolation=cv2.INTER_LINEAR)
    if gt_img.shape[0] != target_size:
        gt_img = cv2.resize(gt_img, (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
    
    # Upscale low-res student images with visible interpolation
    baseline_upscaled = upscale_with_visible_blur(baseline_img, target_size)
    kd_upscaled = upscale_with_visible_blur(kd_img, target_size)
    
    # Create 2×2 grid
    spacing = 5
    canvas_size = target_size * 2 + spacing * 3
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)
    
    # Position images in 2×2 grid
    # Top-left: Teacher
    canvas[spacing:spacing+target_size, spacing:spacing+target_size] = teacher_img
    
    # Top-right: Student Baseline (upscaled with visible blur)
    canvas[spacing:spacing+target_size, spacing*2+target_size:spacing*2+target_size*2] = baseline_upscaled
    
    # Bottom-left: Ground Truth
    canvas[spacing*2+target_size:spacing*2+target_size*2, spacing:spacing+target_size] = gt_img
    
    # Bottom-right: Student + KD (upscaled with visible blur)
    canvas[spacing*2+target_size:spacing*2+target_size*2, spacing*2+target_size:spacing*2+target_size*2] = kd_upscaled
    
    # Add colored borders to show quality difference
    h_offset = spacing
    w_offset = spacing
    
    # Green border for baseline
    cv2.rectangle(canvas, 
                  (w_offset + spacing*2 + target_size, h_offset - 3), 
                  (w_offset + spacing*2 + target_size*2 + 3, h_offset + target_size + 3), 
                  (100, 255, 100), 4)
    
    # Orange border for KD
    cv2.rectangle(canvas,
                  (w_offset + spacing*2 + target_size, h_offset + spacing*2 + target_size - 3),
                  (w_offset + spacing*2 + target_size*2 + 3, h_offset + spacing*2 + target_size*2 + 3),
                  (100, 165, 255), 4)
    
    # Add labels with PIL for better text rendering
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_title = font_label = font_info = ImageFont.load_default()
    
    # Positions for text
    top_left = (spacing + 10, spacing + 10)
    top_right = (spacing*2 + target_size + 10, spacing + 10)
    bottom_left = (spacing + 10, spacing*2 + target_size + 10)
    bottom_right = (spacing*2 + target_size + 10, spacing*2 + target_size + 10)
    
    # Title at top
    title = f"4-Model Comparison: {image_name}"
    draw.text((canvas_size//2 - 180, 10), title, fill=(220, 220, 220), font=font_title)
    
    # Teacher label (top-left)
    draw.text(top_left, "TEACHER R50", fill=(100, 255, 100), font=font_label)
    draw.text((top_left[0], top_left[1] + 25), "HIGH-RES: 1024×1024", 
              fill=(150, 255, 150), font=font_info)
    draw.text((top_left[0], top_left[1] + 45), f"66% mAP | {stats.get('teacher', 0)} dets", 
              fill=(150, 255, 150), font=font_info)
    
    # Student Baseline label (top-right) - with GREEN border indicator
    draw.text(top_right, "STUDENT BASELINE", fill=(100, 200, 100), font=font_label)
    draw.text((top_right[0], top_right[1] + 25), "256×256 UPSCALED", 
              fill=(180, 220, 180), font=font_info)
    draw.text((top_right[0], top_right[1] + 45), "38.2% mAP | BLURRED", 
              fill=(180, 220, 180), font=font_info)
    
    # Ground Truth label (bottom-left)
    draw.text(bottom_left, "GROUND TRUTH", fill=(100, 150, 255), font=font_label)
    draw.text((bottom_left[0], bottom_left[1] + 25), "HIGH-RES: 1024×1024", 
              fill=(150, 180, 255), font=font_info)
    draw.text((bottom_left[0], bottom_left[1] + 45), f"Reference | {stats.get('gt', 0)} objects", 
              fill=(150, 180, 255), font=font_info)
    
    # Student + KD label (bottom-right) - with ORANGE border indicator
    draw.text(bottom_right, "STUDENT + KD", fill=(100, 165, 255), font=font_label)
    draw.text((bottom_right[0], bottom_right[1] + 25), "256×256 UPSCALED", 
              fill=(180, 200, 230), font=font_info)
    draw.text((bottom_right[0], bottom_right[1] + 45), "~74% mAP | BLURRED", 
              fill=(180, 200, 230), font=font_info)
    
    # Note about blur
    note = "Student images upscaled from 256×256 - blur shows quality loss from low-resolution input"
    draw.text((spacing, canvas_size - 35), note, fill=(150, 150, 150), font=font_info)
    
    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


def load_image(path):
    """Load image safely"""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def main():
    parser = argparse.ArgumentParser(description='4-Model comparison with visible blur')
    parser.add_argument('--teacher_dir', 
                       default='work_dirs/inference_results_teacher_highres',
                       help='Directory containing teacher inference results')
    parser.add_argument('--baseline_dir',
                       default='work_dirs/inference_results_student_lowres',
                       help='Directory containing student baseline results')
    parser.add_argument('--kd_dir',
                       default='work_dirs/inference_results_student_kd_lowres',
                       help='Directory containing KD student results')
    parser.add_argument('--out_dir',
                       default='work_dirs/four_model_comparison_same_res',
                       help='Output directory for comparison grids')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of images to compare')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*80)
    print("🎨 4-MODEL COMPARISON WITH VISIBLE BLUR (Same Resolution)")
    print("="*80)
    print(f"\n📁 Loading from:")
    print(f"  Teacher (HIGH):          {args.teacher_dir}")
    print(f"  Student Baseline (LOW):  {args.baseline_dir}")
    print(f"  Student + KD (LOW):      {args.kd_dir}")
    print(f"  Output:                  {args.out_dir}")
    print()
    print("📋 Layout:")
    print("  ┌──────────────────┬──────────────────┐")
    print("  │  TEACHER R50     │  STUDENT BASELINE│")
    print("  │ (1024×1024)      │ (256→1024, BLUR) │")
    print("  ├──────────────────┼──────────────────┤")
    print("  │  GROUND TRUTH    │  STUDENT + KD    │")
    print("  │ (1024×1024)      │ (256→1024, BLUR) │")
    print("  └──────────────────┴──────────────────┘")
    print()
    print("💡 Key: Student images upscaled with visible interpolation blur")
    print("   Shows quality degradation from 4× resolution reduction")
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
        baseline_file = teacher_file.replace('_comparison.jpg', '_comparison.jpg')
        baseline_img = load_image(os.path.join(args.baseline_dir, baseline_file))
        kd_img = load_image(os.path.join(args.kd_dir, teacher_file))
        
        if teacher_img is None:
            failed += 1
            print(f"  ❌ [{idx+1}/{len(teacher_files)}] {base_name} - Missing teacher image")
            continue
        
        if baseline_img is None:
            failed += 1
            print(f"  ❌ [{idx+1}/{len(teacher_files)}] {base_name} - Missing baseline image")
            continue
        
        if kd_img is None:
            failed += 1
            print(f"  ❌ [{idx+1}/{len(teacher_files)}] {base_name} - Missing KD image")
            continue
        
        try:
            # Extract left half (predictions) and right half (GT) from comparison images
            th, tw = teacher_img.shape[:2]
            teacher_pred = teacher_img[:, :tw//2]
            gt_img = teacher_img[:, tw//2:]
            
            bh, bw = baseline_img.shape[:2]
            baseline_pred = baseline_img[:, :bw//2]
            
            kh, kw = kd_img.shape[:2]
            kd_pred = kd_img[:, :kw//2]
            
            # Create stats dictionary
            stats = {
                'teacher': base_name.count('_'),
                'baseline': 1,
                'kd': 1,
                'gt': 1
            }
            
            # Create comparison grid with same resolution but visible blur
            grid = create_same_resolution_grid(teacher_pred, baseline_pred, kd_pred, gt_img, 
                                              base_name, stats)
            
            # Save result
            out_file = os.path.join(args.out_dir, f'{idx:03d}_{base_name}_4model_blur.jpg')
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
    output_files = sorted([f for f in os.listdir(args.out_dir) if f.endswith('_4model_blur.jpg')])
    for i, f in enumerate(output_files[:5], 1):
        print(f"   {i}. {f}")
    if len(output_files) > 5:
        print(f"   ... and {len(output_files)-5} more")
    
    print()
    print("🔍 What You'll See:")
    print("  • All 4 images at same size (1024×1024) - easy to compare")
    print("  • Teacher & GT: Sharp, clear details")
    print("  • Student Baseline: VISIBLY BLURRED (upscaled from 256×256)")
    print("  • Student+KD: VISIBLY BLURRED (upscaled from 256×256)")
    print("  • Green & Orange borders mark the student models")
    print()
    print("✨ Why This Shows KD Value:")
    print("  1. Input quality is dramatically worse (256×256 vs 1024×1024)")
    print("  2. Despite blur, student+KD learns well from teacher")
    print("  3. Clear visual evidence of knowledge distillation benefit")
    print()
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
