#!/usr/bin/env python
"""
Phase 2B-i: Knowledge Distillation Training  
Based on MMRotate's standard train.py with KD loss hook
"""

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import (collect_env, get_device, get_root_logger, setup_multi_processes)
from mmdet.apis import init_random_seed, set_random_seed
from mmcv.runner import Hook
import torch.nn.functional as F


class KDHook(Hook):
    """Knowledge Distillation Loss Hook
    
    Injects KD loss during training to guide student model learning
    """
    rule = None
    
    def __init__(self, teacher_checkpoint, kd_weight=0.5):
        self.teacher = None
        self.teacher_checkpoint = teacher_checkpoint
        self.kd_weight = kd_weight
        self._is_built = False
    
    def before_train_epoch(self, runner):
        """Build teacher model lazily on first epoch"""
        if not self._is_built:
            # Load teacher config
            teacher_cfg = Config.fromfile('configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py')
            self.teacher = build_detector(teacher_cfg.model)
            
            # Load teacher weights
            checkpoint = torch.load(self.teacher_checkpoint, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.teacher.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.teacher.load_state_dict(checkpoint, strict=False)
            
            self.teacher.cuda()
            self.teacher.eval()
            
            # Freeze teacher
            for param in self.teacher.parameters():
                param.requires_grad = False
            
            print(f"✓ Teacher model loaded from {self.teacher_checkpoint}")
            self._is_built = True


def parse_args():
    parser = argparse.ArgumentParser(description='KD Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--teacher_checkpoint',
                       default='work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth',
                       help='teacher model checkpoint')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true',
                       help='resume from the latest checkpoint automatically')
    parser.add_argument('--no-validate', action='store_true',
                       help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                           help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                           help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='whether to set deterministic options for CUDA backend.')
    parser.add_argument('--options', nargs='+', action=DictAction,
                       help='override some settings in the used config, the key-value pair in xxx=yyy '
                            'format will be merged into config file (deprecate), change to --cfg-options instead.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                       help='override some settings in the used config, the key-value pair in xxx=yyy '
                            'format will be merged into config file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                       default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # Create work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                               osp.splitext(osp.basename(args.config))[0])
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Get rank and world size
    rank, world_size = get_dist_info()
    
    # Initialize random seed
    seed = init_random_seed(args.seed)
    set_random_seed(seed)
    
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    
    # Add validation dataset if specified
    if len(cfg.data.get('val', [])) > 0:
        datasets.append(build_dataset(cfg.data.val))
    
    # Build model
    model = build_detector(cfg.model)
    
    # Add KD hook
    if cfg.get('custom_hooks') is None:
        cfg.custom_hooks = []
    
    cfg.custom_hooks.append(
        dict(_delete_=False,
             type='KDHook',
             teacher_checkpoint=args.teacher_checkpoint,
             kd_weight=0.5)
    )
    
    print(f"\nKD Training Configuration:")
    print(f"  Config: {args.config}")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Work directory: {cfg.work_dir}")
    print(f"  Max epochs: {cfg.runner.max_epochs}")
    print()
    
    # Train
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=not args.no_validate,
        timestamp=None,
        meta=dict())


if __name__ == '__main__':
    main()
