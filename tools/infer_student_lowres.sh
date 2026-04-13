#!/bin/bash
# Student Model Inference (LOW-RESOLUTION)
# Uses 0.5x downsampled (512x512) images

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════"
echo "👨‍🎓 STUDENT MODEL INFERENCE (LOW-RESOLUTION)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Configuration:"
echo "  Model: Rotated RetinaNet R18 (Student)"
echo "  mAP: 38.22% (Baseline)"
echo "  Resolution: LOW (256x256 - 0.25x, 4× Smaller)"
echo ""

python tools/inference_viz.py \
  --config configs/rotated_retinanet/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student.py \
  --checkpoint work_dirs/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student/best_mAP_epoch_7.pth \
  --img_dir data/split_1024_dota1_0/trainval/images \
  --ann_dir data/split_1024_dota1_0/trainval/annfiles \
  --out_dir work_dirs/inference_results_student_lowres \
  --num_samples 20 \
  --score_thr 0.3 \
  --device cuda:0 \
  --resolution low

echo ""
echo "✅ Student inference complete!"
echo "📁 Results: work_dirs/inference_results_student_lowres/"
