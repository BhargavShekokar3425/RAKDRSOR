"""
Knowledge Distillation Loss Module for Cross-Resolution Object Detection
Author: RARSOP Project
Purpose: Feature-level + detection-level distillation with spatial alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialProjectionAdapter(nn.Module):
    """
    Spatial Projection Adapter for aligning teacher and student features
    Handles: Channel matching + Spatial upsampling/downsampling + LayerNorm
    
    Input:
        - Teacher features: [B, C_t, H_t, W_t] (high-res, e.g., 1024)
        - Student features: [B, C_s, H_s, W_s] (low-res, e.g., 512)
    
    Output:
        - Aligned student features: [B, C_t, H_t, W_t]
    """
    
    def __init__(self, in_channels, out_channels, align_mode='bilinear'):
        super(SpatialProjectionAdapter, self).__init__()
        self.align_mode = align_mode
        
        # Channel adaptation
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)
    
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: [B, C_s, H_s, W_s]
            teacher_feat: [B, C_t, H_t, W_t]
        
        Returns:
            aligned_feat: [B, C_t, H_t, W_t]
        """
        # Channel matching
        adapted = self.channel_adapt(student_feat)
        
        # Spatial alignment (upsample student to teacher resolution)
        if adapted.shape[2:] != teacher_feat.shape[2:]:
            adapted = F.interpolate(
                adapted,
                size=teacher_feat.shape[2:],
                mode=self.align_mode,
                align_corners=False if self.align_mode != 'nearest' else None
            )
        
        # Layer normalization
        B, C, H, W = adapted.shape
        adapted_norm = adapted.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        adapted_norm = self.layer_norm(adapted_norm)
        adapted_norm = adapted_norm.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return adapted_norm


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss
    Computes L2 distance between aligned student and teacher features
    
    Loss = MSE(adapted_student_feat, teacher_feat)
    """
    
    def __init__(self, reduction='mean'):
        super(FeatureDistillationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: [B, C, H, W] - adapted student features
            teacher_feat: [B, C, H, W] - teacher features
        
        Returns:
            loss: scalar
        """
        # Detach teacher to avoid backward prop
        teacher_feat = teacher_feat.detach()
        
        # L2 loss
        loss = F.mse_loss(student_feat, teacher_feat, reduction=self.reduction)
        return loss


class CosineSimLoss(nn.Module):
    """
    Cosine Similarity Loss for preserving feature similarity structure
    
    Encourages: student features to maintain similar relative distances
    Formula: 1 - cosine_similarity
    """
    
    def __init__(self, reduction='mean'):
        super(CosineSimLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: [B, C, H, W]
            teacher_feat: [B, C, H, W]
        
        Returns:
            loss: scalar
        """
        teacher_feat = teacher_feat.detach()
        
        # Flatten spatial dimensions
        B, C, H, W = student_feat.shape
        student_flat = student_feat.view(B, C, -1)  # [B, C, H*W]
        teacher_flat = teacher_feat.view(B, C, -1)  # [B, C, H*W]
        
        # Normalize
        student_norm = F.normalize(student_flat, p=2, dim=1)  # [B, C, H*W]
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)  # [B, C, H*W]
        
        # Cosine similarity matrix [B, H*W, H*W]
        student_sim = torch.bmm(student_norm.transpose(1, 2), student_norm)
        teacher_sim = torch.bmm(teacher_norm.transpose(1, 2), teacher_norm)
        
        # MSE between similarity matrices
        loss = F.mse_loss(student_sim, teacher_sim, reduction=self.reduction)
        return loss


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer Loss
    Focuses student on important spatial regions learned by teacher
    
    Uses activation maps to identify important regions
    """
    
    def __init__(self, p=2, reduction='mean'):
        super(AttentionTransferLoss, self).__init__()
        self.p = p
        self.reduction = reduction
    
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: [B, C, H, W]
            teacher_feat: [B, C, H, W]
        
        Returns:
            loss: scalar
        """
        teacher_feat = teacher_feat.detach()
        
        # Generate attention maps (sum across channel dimension)
        student_att = torch.sum(student_feat ** 2, dim=1)  # [B, H, W]
        teacher_att = torch.sum(teacher_feat ** 2, dim=1)  # [B, H, W]
        
        # Normalize
        student_att = F.normalize(student_att.view(student_att.size(0), -1), p=2, dim=1)
        teacher_att = F.normalize(teacher_att.view(teacher_att.size(0), -1), p=2, dim=1)
        
        # Compute loss
        loss = F.kl_div(
            F.log_softmax(student_att / self.p, dim=1),
            F.softmax(teacher_att / self.p, dim=1),
            reduction=self.reduction
        )
        return loss


class CrossResolutionKDLoss(nn.Module):
    """
    Complete Cross-Resolution Knowledge Distillation Loss
    
    Combines:
    - Detection loss (standard training)
    - Feature distillation loss (L2)
    - Cosine similarity loss (structure preservation)
    - Attention transfer loss (spatial importance)
    
    Total Loss = L_det + λ1*L_feature + λ2*L_cosine + λ3*L_attention
    
    Args:
        lambda_feature: Weight for feature distillation (default: 1.0)
        lambda_cosine: Weight for cosine similarity (default: 0.5)
        lambda_attention: Weight for attention transfer (default: 0.5)
    """
    
    def __init__(self, lambda_feature=1.0, lambda_cosine=0.5, lambda_attention=0.5):
        super(CrossResolutionKDLoss, self).__init__()
        
        self.lambda_feature = lambda_feature
        self.lambda_cosine = lambda_cosine
        self.lambda_attention = lambda_attention
        
        # Loss functions
        self.feature_loss = FeatureDistillationLoss()
        self.cosine_loss = CosineSimLoss()
        self.attention_loss = AttentionTransferLoss()
    
    def forward(self, student_feats, teacher_feats, spatial_adapters=None):
        """
        Args:
            student_feats: List of student intermediate features [f1, f2, f3, ...]
            teacher_feats: List of teacher intermediate features (same length)
            spatial_adapters: List of SpatialProjectionAdapter modules
        
        Returns:
            loss_dict: Dictionary with individual loss components
        """
        
        total_kd_loss = 0
        loss_dict = {}
        
        # Process each feature level
        for level_idx, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            
            # Spatial alignment if adapter provided
            if spatial_adapters is not None:
                s_feat_aligned = spatial_adapters[level_idx](s_feat, t_feat)
            else:
                # Fallback: manual alignment
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat_aligned = F.interpolate(
                        s_feat,
                        size=t_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    s_feat_aligned = s_feat
            
            # Feature distillation loss
            loss_feat = self.feature_loss(s_feat_aligned, t_feat)
            total_kd_loss += self.lambda_feature * loss_feat
            loss_dict[f'loss_kd_feat_lvl{level_idx}'] = loss_feat
            
            # Cosine similarity loss
            loss_cosine = self.cosine_loss(s_feat_aligned, t_feat)
            total_kd_loss += self.lambda_cosine * loss_cosine
            loss_dict[f'loss_kd_cosine_lvl{level_idx}'] = loss_cosine
            
            # Attention transfer loss
            loss_att = self.attention_loss(s_feat_aligned, t_feat)
            total_kd_loss += self.lambda_attention * loss_att
            loss_dict[f'loss_kd_attention_lvl{level_idx}'] = loss_att
        
        loss_dict['loss_kd_total'] = total_kd_loss
        return total_kd_loss, loss_dict


def build_spatial_adapters(feature_channels, target_channels):
    """
    Factory function to build spatial adapters for each feature level
    
    Args:
        feature_channels: List of input channel numbers [64, 128, 256, 512] (student)
        target_channels: List of output channel numbers (teacher) [256, 256, 256, 256]
    
    Returns:
        adapters: nn.ModuleList of SpatialProjectionAdapter
    """
    adapters = nn.ModuleList()
    
    for in_ch, out_ch in zip(feature_channels, target_channels):
        adapter = SpatialProjectionAdapter(in_ch, out_ch)
        adapters.append(adapter)
    
    return adapters
