#!/bin/bash
# Student + KD Model Inference (LOW-RESOLUTION)
# Uses 0.5x downsampled (512x512) images

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════"
echo "👨‍🎓 STUDENT + KD MODEL INFERENCE (LOW-RESOLUTION)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Configuration:"
echo "  Model: Rotated RetinaNet R18 (Student + Knowledge Distillation)"
echo "  mAP: ~74% (Expected after training)"
echo "  Resolution: LOW (256x256 - 0.25x, 4× Smaller)"
echo ""

python tools/inference_viz.py \
  --config configs/rotated_retinanet/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student.py \
  --checkpoint work_dirs/kd_final/latest.pth \
  --img_dir data/split_1024_dota1_0/trainval/images \
  --ann_dir data/split_1024_dota1_0/trainval/annfiles \
  --out_dir work_dirs/inference_results_student_kd_lowres \
  --num_samples 20 \
  --score_thr 0.3 \
  --device cuda:0 \
  --resolution low

echo ""
echo "✅ Student + KD inference complete!"
echo "📁 Results: work_dirs/inference_results_student_kd_lowres/"
