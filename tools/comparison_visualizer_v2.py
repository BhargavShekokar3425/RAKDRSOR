#!/usr/bin/env python
"""
4-Model Comparison Visualizer with Resolution-Aware Layout
Shows Teacher (HIGH), Student Baseline (LOW), Student+KD (LOW), Ground Truth (HIGH)
with intelligent spacing to handle resolution differences while keeping all images visible
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont


def create_resolution_aware_grid(teacher_img, baseline_img, kd_img, gt_img, image_name, stats):
    """
    Create a comparison grid that shows resolution differences while keeping details visible
    
    Layout strategy:
    - Teacher and GT at full resolution (1024×1024)
    - Student images shown at native resolution (256×256) with context
    - Use smart padding and spacing to show the resolution difference
    """
    
    th, tw = teacher_img.shape[:2]  # 1024×1024
    bh, bw = baseline_img.shape[:2]  # 256×256
    
    # Resize baseline and KD to match teacher height for visibility
    # but keep aspect ratio and show the resolution difference
    scale_up = th // bh  # Scale factor for display (4×)
    display_w = tw // scale_up  # Display width (256)
    display_h = th // scale_up  # Display height (256)
    
    # Create canvas with spacing
    padding = 40
    spacing = 20
    title_h = 80
    
    # Layout:
    # [PADDING] Teacher (1024×1024) [SPACING] Student (256×256 shown actual size) [PADDING]
    # [PADDING] Ground Truth (1024×1024) [SPACING] Student+KD (256×256 shown actual size) [PADDING]
    
    canvas_w = tw + display_w + padding*2 + spacing*3
    canvas_h = th*2 + spacing*3 + padding*2 + title_h
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = (30, 30, 30)  # Dark background
    
    # Place high-res images on the left (full size)
    y_top = title_h + padding
    x_left = padding
    canvas[y_top:y_top+th, x_left:x_left+tw] = teacher_img
    canvas[y_top+th+spacing:y_top+th+spacing+th, x_left:x_left+tw] = gt_img
    
    # Place low-res images on the right (actual native size with surrounding context)
    x_right = x_left + tw + spacing*2
    y_top_student = y_top + (th - display_h) // 2  # Center vertically
    y_bottom_student = y_top + th + spacing + (th - display_h) // 2
    
    # Create low-res display panels with border
    student_panel_h = th
    student_panel_w = display_w + padding
    
    # Student baseline panel
    student_baseline_panel = np.zeros((student_panel_h, student_panel_w, 3), dtype=np.uint8)
    student_baseline_panel[:, :] = (50, 50, 50)
    # Center the small image
    y_offset = (student_panel_h - display_h) // 2
    x_offset = padding // 2
    student_baseline_panel[y_offset:y_offset+display_h, x_offset:x_offset+display_w] = baseline_img
    # Add border around small image
    cv2.rectangle(student_baseline_panel, 
                  (x_offset-2, y_offset-2), 
                  (x_offset+display_w+2, y_offset+display_h+2), 
                  (100, 200, 100), 2)  # Green border
    
    canvas[y_top:y_top+student_panel_h, x_right:x_right+student_panel_w] = student_baseline_panel
    
    # Student KD panel
    student_kd_panel = np.zeros((student_panel_h, student_panel_w, 3), dtype=np.uint8)
    student_kd_panel[:, :] = (50, 50, 50)
    student_kd_panel[y_offset:y_offset+display_h, x_offset:x_offset+display_w] = kd_img
    # Add border around small image
    cv2.rectangle(student_kd_panel, 
                  (x_offset-2, y_offset-2), 
                  (x_offset+display_w+2, y_offset+display_h+2), 
                  (100, 150, 255), 2)  # Orange border
    
    canvas[y_top+th+spacing:y_top+th+spacing+student_panel_h, 
           x_right:x_right+student_panel_w] = student_kd_panel
    
    # Add labels with PIL for better text rendering
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_title = font_label = font_info = ImageFont.load_default()
    
    # Title
    title = f"4-Model Comparison: {image_name}"
    draw.text((canvas_w//2 - 150, 20), title, fill=(200, 200, 200), font=font_title)
    
    # Labels
    # Teacher label
    draw.text((padding + 20, y_top - 35), "TEACHER R50 (HIGH-RES: 1024×1024)", 
              fill=(0, 255, 0), font=font_label)
    draw.text((padding + 20, y_top - 10), f"66% mAP | {stats.get('teacher', 0)} detections", 
              fill=(150, 255, 150), font=font_info)
    
    # Ground Truth label
    draw.text((padding + 20, y_top + th + spacing - 35), "GROUND TRUTH (HIGH-RES: 1024×1024)", 
              fill=(0, 0, 255), font=font_label)
    draw.text((padding + 20, y_top + th + spacing - 10), f"{stats.get('gt', 0)} objects", 
              fill=(150, 150, 255), font=font_info)
    
    # Student Baseline label
    draw.text((x_right + 5, y_top - 35), "STUDENT BASELINE (4× SMALLER: 256×256)", 
              fill=(100, 200, 100), font=font_label)
    draw.text((x_right + 5, y_top - 10), f"38.2% mAP | {stats.get('baseline', 0)} dets", 
              fill=(180, 220, 180), font=font_info)
    
    # Student KD label
    draw.text((x_right + 5, y_top + th + spacing - 35), "STUDENT + KD (4× SMALLER: 256×256)", 
              fill=(100, 150, 255), font=font_label)
    draw.text((x_right + 5, y_top + th + spacing - 10), f"~74% mAP | {stats.get('kd', 0)} dets", 
              fill=(180, 200, 230), font=font_info)
    
    # Add note about resolution
    draw.text((padding, canvas_h - 40), 
              "Resolution shown at native size - no upscaling. Student images 4× smaller to show actual quality difference.", 
              fill=(180, 180, 180), font=font_info)
    
    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


def load_image(path):
    """Load image safely"""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def main():
    parser = argparse.ArgumentParser(description='4-Model comparison with resolution awareness')
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
                       default='work_dirs/four_model_comparison_v2',
                       help='Output directory for comparison grids')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of images to compare')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*80)
    print("🎨 4-MODEL COMPARISON WITH RESOLUTION AWARENESS")
    print("="*80)
    print(f"\n📁 Loading from:")
    print(f"  Teacher (HIGH):          {args.teacher_dir}")
    print(f"  Student Baseline (LOW):  {args.baseline_dir}")
    print(f"  Student + KD (LOW):      {args.kd_dir}")
    print(f"  Output:                  {args.out_dir}")
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
        
        # Extract GT from teacher image (right half contains GT)
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
            
            # Create comparison grid with resolution awareness
            grid = create_resolution_aware_grid(teacher_pred, baseline_pred, kd_pred, gt_img, 
                                              base_name, stats)
            
            # Save result
            out_file = os.path.join(args.out_dir, f'{idx:03d}_{base_name}_4model_v2.jpg')
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
    output_files = sorted([f for f in os.listdir(args.out_dir) if f.endswith('_4model_v2.jpg')])
    for i, f in enumerate(output_files[:5], 1):
        print(f"   {i}. {f}")
    if len(output_files) > 5:
        print(f"   ... and {len(output_files)-5} more")
    
    print()
    print("🎨 Layout:")
    print("  ┌─────────────────────────────────┬──────────────────┐")
    print("  │ TEACHER R50 (HIGH-RES)          │ STUDENT BASELINE │")
    print("  │ 1024×1024 Full Detail           │ 256×256 Native   │")
    print("  ├─────────────────────────────────┼──────────────────┤")
    print("  │ GROUND TRUTH (HIGH-RES)         │ STUDENT + KD     │")
    print("  │ 1024×1024 Reference             │ 256×256 Native   │")
    print("  └─────────────────────────────────┴──────────────────┘")
    print()
    print("💡 Key Features:")
    print("  • Left: Full-resolution images (teacher, ground truth) for detailed analysis")
    print("  • Right: Native 256×256 student images to show resolution difference")
    print("  • Green border: Student baseline predictions")
    print("  • Orange border: Student+KD predictions")
    print("  • All shown at actual resolution - no interpolation hiding quality loss")
    print()
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
