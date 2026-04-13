#!/bin/bash
# Teacher Model Inference (HIGH-RESOLUTION)
# Uses original 1024x1024 images

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════"
echo "🎓 TEACHER MODEL INFERENCE (HIGH-RESOLUTION)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Configuration:"
echo "  Model: Rotated RetinaNet R50 (Teacher)"
echo "  mAP: 66%"
echo "  Resolution: HIGH (1024x1024 - Original)"
echo ""

python tools/inference_viz.py \
  --config configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  --checkpoint work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth \
  --img_dir data/split_1024_dota1_0/trainval/images \
  --ann_dir data/split_1024_dota1_0/trainval/annfiles \
  --out_dir work_dirs/inference_results_teacher_highres \
  --num_samples 20 \
  --score_thr 0.3 \
  --device cuda:0 \
  --resolution high

echo ""
echo "✅ Teacher inference complete!"
echo "📁 Results: work_dirs/inference_results_teacher_highres/"
