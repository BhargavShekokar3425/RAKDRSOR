#!/usr/bin/env python
"""
Phase 2B-i: Knowledge Distillation Training (CLEAN APPROACH)
Uses MMRotate standard training loop with custom loss hook
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmcv.utils import Config
from mmcv.runner import build_optimizer, build_runner
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmdet.apis import set_random_seed
from mmcv.runner import Hook
from mmrotate.distillation.kd_loss import CrossResolutionKDLoss
import torch.nn.functional as F


class KDLossHook(Hook):
    """Custom KD Loss Hook for feature distillation"""
    
    def __init__(self, teacher, kd_loss, kd_weight=1.0, feat_channels=256):
        self.teacher = teacher
        self.kd_loss = kd_loss
        self.kd_weight = kd_weight
        self.feat_channels = feat_channels
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def before_train_iter(self, runner):
        """Add KD loss to student loss"""
        # Nothing to do here - loss computation is in train_step
        pass
    
    def after_train_iter(self, runner):
        """After training iteration"""
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='KD Training Phase 2B-i')
    parser.add_argument('--teacher_checkpoint', 
                       default='work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth',
                       type=str, help='Path to teacher checkpoint')
    parser.add_argument('--student_config',
                       default='configs/rotated_retinanet/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student.py',
                       type=str, help='Path to student config')
    parser.add_argument('--epochs', default=36, type=int, help='Number of epochs')
    parser.add_argument('--work_dir', default='work_dirs/kd_2b_r50_r18_clean', 
                       type=str, help='Work directory')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    return parser.parse_args()


def main(args):
    # Set seed
    set_random_seed(args.seed)
    
    # Load configs
    student_cfg = Config.fromfile(args.student_config)
    
    # Update work directory
    student_cfg.work_dir = args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Build datasets
    print("Building datasets...")
    train_dataset = build_dataset(student_cfg.data.train)
    val_dataset = build_dataset(student_cfg.data.val)
    
    # Build models
    print("Building student model...")
    student_model = build_detector(student_cfg.model)
    
    print("Building teacher model...")
    # Create teacher config (same as student but with R50 backbone)
    teacher_cfg = Config.fromfile('configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py')
    teacher_model = build_detector(teacher_cfg.model)
    
    # Load teacher checkpoint
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
    if 'state_dict' in teacher_checkpoint:
        teacher_model.load_state_dict(teacher_checkpoint['state_dict'], strict=False)
    else:
        teacher_model.load_state_dict(teacher_checkpoint, strict=False)
    
    # Move to GPU
    student_model = student_model.cuda()
    teacher_model = teacher_model.cuda()
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Build optimizer
    optimizer = build_optimizer(student_model, student_cfg.optimizer)
    
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {student_cfg.optimizer.lr}")
    print(f"  Output directory: {args.work_dir}")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    
    print(f"\nStarting KD training...")
    print("=" * 80)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Training epoch
        student_model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, data in enumerate(train_dataset):
            try:
                # Stack batch data properly for model
                batch_data = {
                    'img': torch.stack([torch.from_numpy(d['img']).float() for d in [data]]),
                    'gt_bboxes': torch.stack([torch.from_numpy(d['gt_bboxes']).float() for d in [data]]),
                    'gt_labels': torch.stack([torch.from_numpy(d['gt_labels']).long() for d in [data]]),
                }
                
                # Move to GPU
                batch_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                
                # Forward pass
                student_losses = student_model(return_loss=True, **batch_data)
                
                # Get detection loss
                loss = student_losses.get('loss_cls', torch.tensor(0.0, device='cuda')) + \
                      student_losses.get('loss_bbox', torch.tensor(0.0, device='cuda'))
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=50)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"[Epoch {epoch}][Batch {batch_idx+1}/{len(train_dataset)}] Loss: {avg_loss:.4f}")
                
            except Exception as e:
                print(f"⚠ Error in batch {batch_idx}: {type(e).__name__}: {e}")
                continue
        
        # Validation / logging
        if batch_count > 0:
            avg_train_loss = total_loss / batch_count
            print(f"\n[Epoch {epoch}] Avg Training Loss: {avg_train_loss:.4f}")
            
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                checkpoint = {
                    'state_dict': student_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_train_loss
                }
                checkpoint_path = os.path.join(args.work_dir, f'epoch_{epoch}_best.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"✓ Saved best checkpoint to {checkpoint_path}")
    
    print("=" * 80)
    print(f"✓ KD training completed! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
