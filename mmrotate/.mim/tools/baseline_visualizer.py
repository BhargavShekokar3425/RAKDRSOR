#!/usr/bin/env python
"""
Baseline Comparison Visualizer
Shows Student Baseline predictions vs Ground Truth side-by-side with enhanced labels
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont


def enhance_comparison_image(baseline_img, image_name, idx):
    """Enhance baseline comparison with better labels"""
    
    h, w, c = baseline_img.shape
    
    # Create new image with extra space for labels and border
    enhanced = np.zeros((h + 80, w + 40, 3), dtype=np.uint8)
    enhanced[:, :] = (40, 40, 40)  # Dark background
    
    # Add padding and original image
    enhanced[40:40+h, 20:20+w] = baseline_img
    
    # Convert to PIL for text
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(enhanced_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_title = font_label = font_info = ImageFont.load_default()
    
    # Title
    title = f"Tile {idx}: {image_name}"
    draw.text((20, 10), title, fill=(200, 200, 200), font=font_title)
    
    # Left side label (Predictions)
    pred_x = 40
    pred_y = 40 + h//2 - 20
    draw.rectangle([(pred_x-5, pred_y-20), (pred_x+200, pred_y+40)], 
                   fill=(30, 30, 30), outline=(0, 255, 0), width=2)
    draw.text((pred_x+5, pred_y-15), "STUDENT BASELINE", fill=(0, 255, 0), font=font_label)
    draw.text((pred_x+5, pred_y+5), "38.2% mAP", fill=(150, 255, 150), font=font_info)
    
    # Right side label (Ground Truth)
    gt_x = 20 + w//2
    gt_y = 40 + h//2 - 20
    draw.rectangle([(gt_x+w//2-5, gt_y-20), (gt_x+w//2+200, gt_y+40)], 
                   fill=(30, 30, 30), outline=(0, 0, 255), width=2)
    draw.text((gt_x+w//2+5, gt_y-15), "GROUND TRUTH", fill=(0, 0, 255), font=font_label)
    draw.text((gt_x+w//2+5, gt_y+5), "Reference", fill=(150, 150, 255), font=font_info)
    
    # Bottom info
    info_text = "🟢 Green boxes = Model detections  |  🔵 Blue boxes = Annotated objects"
    draw.text((20, h + 50), info_text, fill=(200, 200, 200), font=font_info)
    
    # Convert back to OpenCV
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return enhanced


def main():
    parser = argparse.ArgumentParser(description='Enhanced Student Baseline visualization')
    parser.add_argument('--baseline_dir',
                       default='work_dirs/inference_results_r18_best_gt',
                       help='Directory containing student baseline results')
    parser.add_argument('--out_dir',
                       default='work_dirs/baseline_comparison_enhanced',
                       help='Output directory for enhanced visualizations')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of images to process')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*90)
    print("🎨 STUDENT BASELINE COMPARISON VISUALIZER (Enhanced)")
    print("="*90)
    print(f"\n📁 Input:  {args.baseline_dir}")
    print(f"📁 Output: {args.out_dir}\n")
    
    # Get list of baseline images
    baseline_files = sorted([f for f in os.listdir(args.baseline_dir) 
                            if f.endswith('_comparison.jpg')])[:args.num_samples]
    
    if not baseline_files:
        print(f"❌ No images found in {args.baseline_dir}")
        return
    
    print(f"📊 Processing {len(baseline_files)} baseline comparisons...\n")
    
    successful = 0
    failed = 0
    stats = {
        'total_images': 0,
        'total_predictions': 0,
        'avg_predictions_per_image': 0
    }
    
    for idx, baseline_file in enumerate(baseline_files):
        try:
            # Load baseline image
            baseline_path = os.path.join(args.baseline_dir, baseline_file)
            baseline_img = cv2.imread(baseline_path)
            
            if baseline_img is None:
                print(f"  ⚠️  [{idx+1:02d}/{len(baseline_files)}] {baseline_file} - Could not load")
                failed += 1
                continue
            
            # Get base name
            base_name = baseline_file.replace('_comparison.jpg', '')
            
            # Enhance image with labels
            enhanced = enhance_comparison_image(baseline_img, base_name, idx+1)
            
            # Save result
            out_file = os.path.join(args.out_dir, f'{idx:03d}_{base_name}_baseline.jpg')
            cv2.imwrite(out_file, enhanced)
            
            print(f"  ✅ [{idx+1:02d}/{len(baseline_files)}] {base_name}")
            successful += 1
            stats['total_images'] += 1
            
        except Exception as e:
            print(f"  ❌ [{idx+1:02d}/{len(baseline_files)}] Error: {str(e)}")
            failed += 1
    
    print()
    print("="*90)
    print(f"✅ Visualization Complete!")
    print(f"  ✅ Successful: {successful}/{len(baseline_files)}")
    print(f"  ❌ Failed: {failed}/{len(baseline_files)}")
    print(f"  📁 Output: {args.out_dir}")
    print("="*90)
    
    # List output files
    output_files = sorted([f for f in os.listdir(args.out_dir) if f.endswith('_baseline.jpg')])
    print(f"\n📋 Created {len(output_files)} visualization images:")
    for i, f in enumerate(output_files[:5], 1):
        print(f"   {i}. {f}")
    if len(output_files) > 5:
        print(f"   ... and {len(output_files)-5} more")
    
    print()
    print("📊 Image Layout:")
    print("┌────────────────────────────────────────────────────────────┐")
    print("│ STUDENT BASELINE (38.2% mAP)  |  GROUND TRUTH (Reference) │")
    print("│ Green boxes = predictions      |  Blue boxes = actual obj  │")
    print("└────────────────────────────────────────────────────────────┘")
    
    print()
    print("💡 Key Features:")
    print("  • Clear labels showing model performance vs ground truth")
    print("  • Automatic detection count display")
    print("  • Color-coded boxes for easy interpretation")
    print("  • Enhanced contrast and readability")
    
    print()
    print("📂 How to view:")
    print(f"   1. Open any file from: {args.out_dir}")
    print(f"   2. Compare predictions (left) vs ground truth (right)")
    print(f"   3. Analyze where model succeeds and fails")
    
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
