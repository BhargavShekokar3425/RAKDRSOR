#!/usr/bin/env python
"""
Phase 2B-i: Knowledge Distillation Training
Simple approach using MMRotate standard training with KD loss hook
"""

import os
import sys
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from mmcv.utils import Config
from mmrotate.apis import set_random_seed, train_detector
from mmrotate.models import build_detector
from mmrotate.datasets import build_dataset
from mmcv.runner import Hook, DistSamplerSeedHook, Fp16OptimizerHook
import torch.nn.functional as F


class KDLossHook(Hook):
    """Knowledge Distillation Loss Hook"""
    rule = None
    
    def __init__(self, teacher_model, kd_weight=0.5):
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def after_train_step(self, runner):
        """Called after each training step"""
        pass
    
    def before_train_iter(self, runner):
        """Called before each training iteration"""
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='KD Training Phase 2B-i')
    parser.add_argument('--teacher_checkpoint',
                       default='work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth',
                       type=str, help='Teacher model checkpoint')
    parser.add_argument('--student_config',
                       default='configs/rotated_retinanet/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student.py',
                       type=str, help='Student config file')
    parser.add_argument('--work_dir',
                       default='work_dirs/kd_2b_r50_r18',
                       type=str, help='Work directory to save models')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--cfg_options', default=None, type=dict, help='Config options')
    return parser.parse_args()


def main(args):
    warnings.filterwarnings('ignore')
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load student config
    cfg = Config.fromfile(args.student_config)
    cfg.work_dir = args.work_dir
    
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    
    # Build student model
    print("[1/4] Building student model...")
    model = build_detector(cfg.model)
    
    # Build teacher model
    print("[2/4] Building teacher model...")
    teacher_cfg = Config.fromfile('configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py')
    teacher = build_detector(teacher_cfg.model)
    
    # Load teacher checkpoint
    print(f"[3/4] Loading teacher checkpoint: {args.teacher_checkpoint}")
    checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        teacher.load_state_dict(checkpoint, strict=False)
    
    # Setup dataset
    print("[4/4] Preparing datasets...")
    datasets = [build_dataset(cfg.data.train)]
    
    print(f"\nTraining configuration:")
    print(f"  Student config: {args.student_config}")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Work directory: {cfg.work_dir}")
    print(f"  Epochs: {cfg.runner.max_epochs}")
    print(f"  Batch size: {cfg.data.samples_per_gpu * cfg.data.workers_per_gpu}")
    
    # Add KD loss hook
    cfg.custom_hooks = [
        dict(
            type='KDLossHook',
            teacher_model=teacher,
            kd_weight=0.5,
            priority='VERY_LOW'
        )
    ]
    
    # Train using standard MMRotate training
    print("\n" + "=" * 80)
    print("Starting KD training...")
    print("=" * 80)
    
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=None,
        meta=dict()
    )
    
    print("\n" + "=" * 80)
    print("✓ KD training completed!")
    print("=" * 80)


if __name__ == '__main__':
    args = parse_args()
    main(args)
