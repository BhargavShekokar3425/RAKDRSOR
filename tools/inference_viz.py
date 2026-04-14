"""
Inference and Visualization Script
Shows detection results with ground truth for comparison
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import mmcv
import mmrotate  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with visualization')
    parser.add_argument('--config', 
                       default='configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
                       help='Config file')
    parser.add_argument('--checkpoint',
                       default='work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth',
                       help='Model checkpoint')
    parser.add_argument('--img_dir',
                       default='data/split_1024_dota1_0/test/images',
                       help='Image directory to test')
    parser.add_argument('--ann_dir',
                       default='data/split_1024_dota1_0/test/annfiles',
                       help='Annotation directory (optional)')
    parser.add_argument('--out_dir',
                       default='work_dirs/inference_results',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of images to visualize')
    parser.add_argument('--score_thr', type=float, default=0.3,
                       help='Score threshold for visualization')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to run inference on')
<<<<<<< HEAD
    parser.add_argument('--resolution', default='high', choices=['high', 'low'],
                       help='Image resolution: high (original) or low (0.25x downsampled - 4× smaller)')
=======
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
    return parser.parse_args()


def load_annotations(ann_file):
<<<<<<< HEAD
    """Load DOTA annotations in quadrilateral format"""
    annotations = {'bboxes': []}
    
    if not os.path.exists(ann_file):
        return annotations
        
    try:
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip header lines
                if not line or line.startswith('imagesource') or line.startswith('gsd'):
                    continue
                
                parts = line.split()
                
                # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
                if len(parts) >= 9:
                    try:
                        coords = [float(p) for p in parts[:8]]
                        x1, y1, x2, y2, x3, y3, x4, y4 = coords
                        class_name = parts[8]
                        difficulty = int(parts[9]) if len(parts) > 9 else 0
                        
                        # Store as quadrilateral (4 corners)
                        annotations['bboxes'].append({
                            'points': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                            'class': class_name,
                            'difficulty': difficulty
                        })
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"  ⚠ Error loading {ann_file}: {e}")
    
=======
    """Load DOTA annotations"""
    annotations = {}
    try:
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('imagesource') or line.startswith('gsd'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                    class_name = parts[8] if len(parts) > 8 else 'unknown'
                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                    
                    if 'bboxes' not in annotations:
                        annotations['bboxes'] = []
                    annotations['bboxes'].append({
                        'points': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                        'class': class_name,
                        'difficulty': difficulty
                    })
    except:
        pass
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
    return annotations


def main(args):
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("[1/4] Loading model...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
<<<<<<< HEAD
    # Get class names - DOTA has 15 classes
    dota_classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                    'basketball-court', 'storage-tank', 'soccer-ball-field',
                    'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    
    if hasattr(model, 'CLASSES'):
        class_names = model.CLASSES
    else:
        class_names = dota_classes
=======
    # Get class names
    if hasattr(model, 'CLASSES'):
        class_names = model.CLASSES
    else:
        class_names = [str(i) for i in range(15)]  # DOTA has 15 classes
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
    
    print(f"[2/4] Loading images from {args.img_dir}...")
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))])[:args.num_samples]
    
<<<<<<< HEAD
    if not img_files:
        print(f"  ✗ No images found in {args.img_dir}")
        return
    
    resolution_label = "HIGH-RES (Original)" if args.resolution == 'high' else "LOW-RES (0.25x)"
    print(f"[3/4] Running inference on {len(img_files)} images ({resolution_label})...")
    
    gt_count = 0  # Count how many images have ground truth
=======
    print(f"[3/4] Running inference on {len(img_files)} images...")
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
    
    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(args.img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
<<<<<<< HEAD
            print(f"  ⚠ [{idx+1}/{len(img_files)}] Failed to load {img_file}")
            continue
        
        # Handle resolution
        original_h, original_w = img.shape[:2]
        inference_img = img.copy()
        display_img = img.copy()  # Image used for visualization
        scale_factor = 1.0
        
        if args.resolution == 'low':
            # Downsample to 0.25x resolution (4× smaller)
            scale_factor = 0.25
            new_h, new_w = int(original_h * scale_factor), int(original_w * scale_factor)
            inference_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            display_img = inference_img.copy()  # Use downsampled image for display (NO upscaling)
            # Save downsampled image temporarily for inference
            temp_img_path = os.path.join('/tmp', img_file)
            cv2.imwrite(temp_img_path, inference_img)
            inference_path = temp_img_path
        else:
            inference_path = img_path
        
        # Run inference
        result = inference_detector(model, inference_path)
        
        # Extract detections from result
=======
            print(f"  ⚠ Failed to load {img_file}")
            continue
        
        # Run inference
        result = inference_detector(model, img_path)
        
        # Extract results (result is a tuple of lists of arrays, one per class)
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
        detections = []
        if isinstance(result, (tuple, list)):
            for class_id, bboxes_array in enumerate(result):
                if bboxes_array is not None and len(bboxes_array) > 0:
<<<<<<< HEAD
                    # bboxes_array is Nx5 or Nx6 array (cx, cy, w, h, angle, [score])
                    for bbox in bboxes_array:
                        if len(bbox) >= 5:
                            if len(bbox) >= 6:
                                cx, cy, w, h, angle, score = bbox[:6]
                            else:
                                cx, cy, w, h, angle = bbox[:5]
                                score = 1.0
                            
                            if float(score) >= args.score_thr:
                                # Keep detections at display resolution (don't scale)
                                detections.append({
                                    'cx': float(cx),
                                    'cy': float(cy),
                                    'w': float(w),
                                    'h': float(h),
                                    'angle': float(angle),
                                    'class_id': class_id,
                                    'class_name': class_names[class_id] if class_id < len(class_names) else f'class_{class_id}',
                                    'score': float(score)
                                })
        
        # Load ground truth annotations
        ann_file = os.path.join(args.ann_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        annotations = load_annotations(ann_file)
        gt_bboxes = annotations.get('bboxes', [])
        
        # Scale ground truth to display resolution if low-res
        if args.resolution == 'low' and scale_factor < 1.0:
            for bbox_info in gt_bboxes:
                # Scale down the points to match 512×512 display
                scaled_points = []
                for x, y in bbox_info['points']:
                    scaled_points.append((x * scale_factor, y * scale_factor))
                bbox_info['points'] = scaled_points
        
        if gt_bboxes:
            gt_count += 1
        
        # Draw detections (GREEN) on copy
        img_pred = display_img.copy()
        for det in detections:
            # Calculate rotated box corners
            cx, cy, w, h, angle = det['cx'], det['cy'], det['w'], det['h'], det['angle']
            angle_rad = np.radians(angle)
            
            # Calculate corners relative to center
=======
                    # bboxes_array is Nx6 array where last column is score
                    for bbox in bboxes_array:
                        if len(bbox) >= 6:
                            cx, cy, w, h, angle, score = bbox[:6]
                        elif len(bbox) == 5:
                            cx, cy, w, h, angle = bbox
                            score = 0.0
                        else:
                            continue
                        
                        if float(score) >= args.score_thr:
                            detections.append({
                                'bbox': np.array([float(cx), float(cy), float(w), float(h), float(angle)]),
                                'class_id': class_id,
                                'class_name': class_names[class_id] if class_id < len(class_names) else f'class_{class_id}',
                                'score': float(score)
                            })
        
        # Draw detections
        img_det = img.copy()
        for det in detections:
            bbox = det['bbox']
            cx, cy, w, h, angle = bbox
            angle_rad = np.radians(angle)
            
            # Calculate corners
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
            corners = np.array([
                [-w/2, -h/2],
                [w/2, -h/2],
                [w/2, h/2],
                [-w/2, h/2]
<<<<<<< HEAD
            ], dtype=np.float32)
            
            # Rotation matrix
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
=======
            ])
            
            # Rotation matrix
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
            
            # Rotate and translate
            rotated = corners @ rot_mat.T
            rotated[:, 0] += cx
            rotated[:, 1] += cy
            
<<<<<<< HEAD
            # Draw polygon (predictions in GREEN)
            pts = np.round(rotated).astype(np.int32)
            cv2.polylines(img_pred, [pts], True, (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{det['class_name'][:8]} {det['score']:.2f}"
            text_pos = tuple(pts[0].astype(int))
            cv2.putText(img_pred, label_text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw ground truth (BLUE) on separate copy
        img_gt = display_img.copy()
        for bbox_info in gt_bboxes:
            points = np.array(bbox_info['points'], dtype=np.int32)
            # Ground truth in BLUE
            cv2.polylines(img_gt, [points], True, (255, 0, 0), 2)
            label_text = f"{bbox_info['class'][:8]}"
            cv2.putText(img_gt, label_text, tuple(points[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Create side-by-side comparison (2-panel layout for clarity)
        h, w, c = display_img.shape
        comparison = np.zeros((h, w*2, c), dtype=np.uint8)
        comparison[:, 0:w] = img_pred  # Predictions on left
        comparison[:, w:2*w] = img_gt   # Ground truth on right
        
        # Add labels to the comparison image
        res_label = "HIGH-RES" if args.resolution == 'high' else "LOW-RES (0.25x)"
        cv2.putText(comparison, f'PREDICTIONS ({len(detections)}) - {res_label}', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comparison, f'GROUND TRUTH ({len(gt_bboxes)})', (w+20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
=======
            # Draw polygon
            pts = rotated.astype(np.int32)
            cv2.polylines(img_det, [pts], True, (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class_name']} {det['score']:.2f}"
            cv2.putText(img_det, label, tuple(pts[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Load and draw ground truth  
        img_gt = img.copy()
        ann_file = os.path.join(args.ann_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        if os.path.exists(ann_file):
            annotations = load_annotations(ann_file)
            if annotations.get('bboxes'):
                for bbox_info in annotations['bboxes']:
                    points = np.array(bbox_info['points'], dtype=np.int32)
                    cv2.polylines(img_gt, [points], True, (255, 0, 0), 2)  # Blue for GT
                    cv2.putText(img_gt, bbox_info['class'], tuple(points[0]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Create comparison image (4-panel)
        h, w = img.shape[:2]
        comparison = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        comparison[0:h, 0:w] = img  # Original
        comparison[0:h, w:2*w] = img_det  # Detections  
        comparison[h:2*h, 0:w] = img_gt  # Ground truth
        
        # Add labels
        cv2.putText(comparison, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f'Predictions ({len(detections)})', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Ground Truth', (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9
        
        # Save result
        out_file = os.path.join(args.out_dir, f'{idx:03d}_{Path(img_file).stem}_comparison.jpg')
        cv2.imwrite(out_file, comparison)
        
<<<<<<< HEAD
        status = f"✓ [{idx+1}/{len(img_files)}] {Path(img_file).stem} - Pred: {len(detections)}, GT: {len(gt_bboxes)}"
        print(f"  {status}")
    print(f"\n[4/4] ✓ Visualization complete!")
    print(f"Results saved to: {args.out_dir}")
    print(f"Total images processed: {len(img_files)}")
    print(f"Images with ground truth: {gt_count}/{len(img_files)}")
    res_label = "HIGH-RES (Original)" if args.resolution == 'high' else "LOW-RES (0.25x - 4× smaller)"
    print(f"Resolution used: {res_label}")
    print(f"\nColor Legend:")
    print(f"  🟢 Green boxes = Model predictions")
    print(f"  🔵 Blue boxes = Ground truth annotations")
=======
        print(f"  ✓ [{idx+1}/{len(img_files)}] {Path(img_file).stem} - {len(detections)} detections")
    
    print(f"\n[4/4] ✓ Visualization complete!")
    print(f"Results saved to: {args.out_dir}")
    print(f"Total images processed: {len(img_files)}")
>>>>>>> d409308ca312a29a84c7fcee87b51475c9190de9


if __name__ == '__main__':
    args = parse_args()
    main(args)
