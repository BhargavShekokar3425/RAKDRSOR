#!/bin/bash
# Phase 2B-i KD Training Script
# Teacher: Rotated RetinaNet R50 (66% mAP, FROZEN)
# Student: Rotated RetinaNet R18 (baseline being trained now, will get KD boost)
# Run this AFTER the student baseline training completes

set -e

cd /home/ankita/Bhargav/RARSOP/mmrotate

echo "================================================================================"
echo "PHASE 2B-i STEP 2: KNOWLEDGE DISTILLATION TRAINING"
echo "================================================================================"
echo "Teacher: Rotated RetinaNet R50 (32M params, 66% mAP)"
echo "Student: Rotated RetinaNet R18 (11M params, will be 68-72% after baseline)"
echo "================================================================================"
echo ""

# Get latest best checkpoint from student training
STUDENT_CHECKPOINT=$(ls -t work_dirs/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student/best_mAP_epoch*.pth 2>/dev/null | head -1)

if [ -z "$STUDENT_CHECKPOINT" ]; then
    echo "⚠ Student baseline checkpoint not found!"
    echo "Expected location: work_dirs/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student/best_mAP_epoch*.pth"
    echo "Make sure Step 1 training completes first."
    exit 1
fi

echo "✓ Found student checkpoint: $STUDENT_CHECKPOINT"
echo ""

TEACHER_CHECKPOINT="work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/epoch_12.pth"
echo "✓ Using teacher checkpoint: $TEACHER_CHECKPOINT"
echo ""

echo "Starting KD training..."
echo "================================================================================"
echo ""

python tools/kd_train_2b_simple.py \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --student-checkpoint "$STUDENT_CHECKPOINT" \
  --lambda-feature 1.0 \
  --lambda-cosine 0.5 \
  --lambda-attention 0.5 \
  --epochs 36 \
  --lr 0.004 \
  --batch-size 2 \
  --workers 2

echo ""
echo "================================================================================"
echo "KD Training Complete!"
echo "================================================================================"
echo "Output saved to: work_dirs/kd_2b_phase_r50_to_r18/"
echo "Best checkpoint: work_dirs/kd_2b_phase_r50_to_r18/best_student_kd.pth"
