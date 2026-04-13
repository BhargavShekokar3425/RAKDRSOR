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
    return parser.parse_args()


def load_annotations(ann_file):
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
    return annotations


def main(args):
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("[1/4] Loading model...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # Get class names
    if hasattr(model, 'CLASSES'):
        class_names = model.CLASSES
    else:
        class_names = [str(i) for i in range(15)]  # DOTA has 15 classes
    
    print(f"[2/4] Loading images from {args.img_dir}...")
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))])[:args.num_samples]
    
    print(f"[3/4] Running inference on {len(img_files)} images...")
    
    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(args.img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ⚠ Failed to load {img_file}")
            continue
        
        # Run inference
        result = inference_detector(model, img_path)
        
        # Extract results (result is a tuple of lists of arrays, one per class)
        detections = []
        if isinstance(result, (tuple, list)):
            for class_id, bboxes_array in enumerate(result):
                if bboxes_array is not None and len(bboxes_array) > 0:
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
            corners = np.array([
                [-w/2, -h/2],
                [w/2, -h/2],
                [w/2, h/2],
                [-w/2, h/2]
            ])
            
            # Rotation matrix
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Rotate and translate
            rotated = corners @ rot_mat.T
            rotated[:, 0] += cx
            rotated[:, 1] += cy
            
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
        
        # Save result
        out_file = os.path.join(args.out_dir, f'{idx:03d}_{Path(img_file).stem}_comparison.jpg')
        cv2.imwrite(out_file, comparison)
        
        print(f"  ✓ [{idx+1}/{len(img_files)}] {Path(img_file).stem} - {len(detections)} detections")
    
    print(f"\n[4/4] ✓ Visualization complete!")
    print(f"Results saved to: {args.out_dir}")
    print(f"Total images processed: {len(img_files)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
