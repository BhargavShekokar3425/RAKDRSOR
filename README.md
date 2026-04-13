# RAKD: Resolution-Agnostic Knowledge Distillation for Remote Sensing Imagery

<div align="center">

**Resolution-Agnostic Knowledge Distillation for Remote Sensing Imagery**

Bhargav Shekokar¹* · Namya Dhingra¹ · Patel Darsh¹

¹ IIT Jodhpur

[📄 Paper](./report.pdf) | [💻 Code](https://github.com/BhargavShekokar3425/RAKDR80R) | [📊 Results](./RESULTS.md)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch 1.13+](https://img.shields.io/badge/PyTorch-1.13%2B-red)](https://pytorch.org/)
[![MMRotate 1.x](https://img.shields.io/badge/Built%20on-MMRotate%201.x-green)](https://github.com/open-mmlab/mmrotate)
[![DOTA v1.0](https://img.shields.io/badge/Dataset-DOTA%20v1.0-orange)](https://captain-whu.github.io/DiRS/)

</div>

---

## Abstract

Object detection in remote sensing imagery demands both high accuracy and computational efficiency. We propose **RAKD (Resolution-Agnostic Knowledge Distillation)**, a framework enabling lightweight rotated object detectors to achieve state-of-the-art performance despite operating at lower input resolutions than high-capacity teachers.

**Core Innovation:** Spatial Projection Adapters (SPAs) explicitly bridge the cross-resolution feature alignment gap, enabling feature-level knowledge transfer where direct comparison was previously infeasible. Combined with a composite distillation loss (feature + cosine + attention), a lightweight ResNet-18 student (11M params) achieves **74.83% mAP at 92 FPS** on DOTA v1.0—a **+4.53% improvement** over independent baseline while maintaining real-time deployment viability.

**Implementation:** Built as extensions to MMRotate [[18]], a professional-grade PyTorch rotated object detection framework, ensuring reproducibility and modularity.

---

## 1. Introduction

Remote sensing image analysis underpins applications spanning precision agriculture, disaster response, and maritime surveillance. Rotated object detection—parameterizing predictions as oriented bounding boxes (x, y, w, h, θ)—has emerged as the critical task, with state-of-the-art methods achieving impressive accuracy on benchmarks like DOTA.

### The Deployment Challenge

**High-accuracy = High-resolution = High compute cost:**
- Teacher models train at 1024×1024 (saturating accuracy) but require 25-35 FPS inference GPUs
- Deployment on UAVs, edge devices requires ≤768×512 resolution and <100 FPS latency
- Direct multi-resolution training contradicts standard practice; teachers and students must use identical input scales

**Existing solutions fall short:**
- **Knowledge distillation (KD)** methods [[7-12]] achieve cross-model compression *at identical resolution*
- **Single-resolution training** independently sacrifices accuracy for deployment efficiency
- **No framework** explicitly handles cross-resolution KD for oriented detection

### Our Contribution

We introduce **RAKD**, addressing two key technical challenges:

1. **Cross-resolution feature alignment:** Teacher features (1024×1024) and student features (768×768) cannot be directly compared due to spatial/channel mismatches
2. **Architecture-agnostic distillation:** KD must work across different backbones without constraining detector design

**Solution:** Lightweight spatial adapters align features + composite loss enforces multi-faceted knowledge transfer.

### Built on MMRotate

RAKD extends **MMRotate** [[18]], a professional PyTorch-based rotated object detection framework:

**MMRotate provides:**
- ✅ Multiple detector architectures (Oriented R-CNN, S2ANet, RetinaNet)
- ✅ DOTA dataset support and preprocessing
- ✅ Modular training infrastructure
- ✅ Industry-standard evaluation metrics

**We added:**
- ⭐ Spatial Projection Adapters (`mmrotate/distillation/`)
- ⭐ Cross-resolution KD trainer (`tools/kd_train.py`)
- ⭐ Modified student configuration with 6 critical hyperparameter fixes

---

## 2. Related Work

### Rotated Object Detection

Oriented detection evolved from RRPN [[14]] through feature-alignment methods (S2ANet, Oriented R-CNN) to one-stage detectors (R3Det, Rot. FCOS). MMRotate [[18]] unified these approaches into a professional framework, yet lightweight rotated detectors for real-time UAV deployment remain underexplored.

### Knowledge Distillation

Hinton et al. [[7]] established the KD paradigm. Extensions for object detection [[8-12]] include:
- **Feature-level** distillation (FitNets [[8]])
- **Attention transfer** [[9]] for spatial focus alignment
- **Task-aligned** distillation (Decoupled [[11]], FGD [[12]])

**Critical limitation:** All prior work assumes teacher-student share identical input resolutions. This assumption breaks for multi-resolution deployment.

### Cross-Resolution Learning

Super-resolution-guided detection [[15]] upsamples to teacher resolution (overhead). Resolution-adaptive networks [[13]] learn at multiple scales but lack explicit KD. **RAKD is the first cross-resolution KD framework for rotated detection.**

---

## 3. Problem Statement

**Notation:** Let $\mathcal{D} = \{(I_i, \mathbf{y}_i)\}_{i=1}^{N}$ be training set with images and rotated annotations $\mathbf{y}_i = \{(x_j, y_j, w_j, h_j, \theta_j, c_j)\}$ (box center, size, le90-convention rotation angle, class).

**Teacher network** $T$ (params $\phi_T$): processes images at $r_T = 1024$ → feature pyramids $\{F_T^i\}_{i=1}^{L}$  
**Student network** $S$ (params $\phi_S$): processes images at $r_S = 768$ → pyramids $\{F_S^i\}_{i=1}^{L}$

### Fundamental Cross-Resolution Mismatches

**1. Spatial mismatch:** $H_T^i > H_S^i$, $W_T^i > W_S^i$ (due to lower input resolution)  
**2. Channel mismatch:** Student configs often use $C_S \neq C_T$ (128 vs. 256 FPN channels)

### Objective

Jointly optimize student and adapter parameters to maximize detection accuracy while maintaining real-time inference:

$$\min_{\phi_S, \phi_A} \mathcal{L}_{\text{det}}(\phi_S) + \lambda_f \mathcal{L}_{\text{feature}} + \lambda_c \mathcal{L}_{\text{cosine}} + \lambda_a \mathcal{L}_{\text{attention}}$$

where $\lambda_f = 1.0, \lambda_c = 0.5, \lambda_a = 0.5$ (optimal from ablation, Table 2).

---

## 4. Proposed Method

### 4.1 Spatial Projection Adapters (SPAs)

Each FPN level uses a lightweight adapter:

$$F_S^{i,\text{adapted}} = \text{LN}\left(\text{Upsample}_{\text{bilinear}}\left(\text{BN}\left(\text{Conv}_{1\times1}(F_S^i)\right), (H_T^i, W_T^i)\right)\right)$$

**Components:**
- 1×1 convolution: Channel adaptation (already 256D, preserves spatial structure)
- Batch normalization: Feature stabilization
- Bilinear interpolation: Spatial alignment to teacher dimensions
- Layer normalization: Stabilizes multi-scale representation learning

**Parameter cost:** 0.26M total (discarded at inference)

### 4.2 Composite Distillation Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda_f \mathcal{L}_{\text{feature}} + \lambda_c \mathcal{L}_{\text{cosine}} + \lambda_a \mathcal{L}_{\text{attention}}$$

**Detection Loss:** Standard focal loss + SmoothL1 (from MMRotate's RetinaNet head)

**Feature Distillation (Pixel-wise matching):**
$$\mathcal{L}_{\text{feature}} = \frac{1}{L} \sum_{i=1}^{L} \left\| F_T^i - F_S^{i,\text{adapted}} \right\|_2^2$$

**Cosine Similarity (Relational structure):**
$$\mathcal{L}_{\text{cosine}} = \frac{1}{L} \sum_{i=1}^{L} \left(1 - \frac{\langle F_T^i, F_S^{i,\text{adapted}} \rangle}{\|F_T^i\|_2 \|F_S^{i,\text{adapted}}\|_2}\right)$$

**Attention Transfer (Spatial focus):**
$$\mathcal{L}_{\text{attention}} = \frac{1}{L} \sum_{i=1}^{L} KL\left(\sigma(A_T^i), \sigma(A_S^i)\right), \quad A^i = \frac{1}{C}\sum_c|F_c^i|$$

### 4.3 Training Protocol

**Phase 1A - Teacher (ResNet-50):**
- Resolution: 1024×1024
- Epochs: 12
- Learning rate: 0.01 (step decay)
- Result: 66% mAP (frozen for Phase 2)

**Phase 1B - Student (ResNet-18):**
- Resolution: 768×768
- Epochs: 36 (3× longer for lightweight model)
- Learning rate: 0.002 (stability fix)
- **Critical 6 hyperparameter fixes:** FPN channels 256 (not 128), L1Loss (not SmoothL1), warmup 1000 iters (not 500), batch size 2 (not 4)
- Result: 70.3% mAP baseline

**Phase 2 - Knowledge Distillation:**
- Teacher: frozen (inference only)
- Student: trainable (backbone + head)
- Adapters: trainable
- Epochs: 36 (same as Phase 1B for fair comparison)
- Loss weights: (1.0, 0.5, 0.5)
- Result: 74.83% mAP (+4.53% improvement)

---

## 5. Experiments and Results

### 5.1 Setup

**Dataset:** DOTA v1.0 (2,806 images, 15 categories, 12,679 train tiles)  
**Protocol:** Single-scale evaluation, mAP@IoU=0.5  
**Hardware:** NVIDIA A100 (40GB GPU)  
**Framework:** PyTorch 1.13 + MMRotate 1.0  

### 5.2 MMRotate Integration (How RAKD Works with MMRotate)

#### What MMRotate Provides

MMRotate is a professional PyTorch rotated object detection framework [[18]]:

```
mmrotate/ (OpenMMLab official)
├── models/               # Detector architectures
│   ├── roi_heads/       # Detection heads (OBB, etc.)
│   └── backbones/       # ResNet, mobilenet, etc.
├── datasets/            # Data loading + DOTA preprocessing
├── core/                # OBB evaluation metrics, post-processing
└── apis/                # Training utilities, inference
```

**Key capabilities we leverage:**
- ✅ Oriented object detection architectures (Oriented R-CNN, RetinaNet, S2ANet)
- ✅ DOTA dataset full pipeline (image tiling at 1024×1024 with 200px overlap)
- ✅ Training infrastructure (SGD with learning rate schedules, checkpointing, evaluation hooks)
- ✅ Multi-GPU distributed training support
- ✅ Evaluation metrics: mAP for rotated boxes (OBB convention)

#### RAKD Custom Extensions to MMRotate

| Component | Location | Purpose | Implementation |
|-----------|----------|---------|-----------------|
| SPA Module | `mmrotate/distillation/kd_loss.py` | Cross-resolution alignment | Pytorch modules |
| KD Losses | `mmrotate/distillation/kd_loss.py` | Feature + cosine + attention | Custom torch.nn losses |
| KD Trainer | `tools/kd_train.py` | Phase 2 training loop | Extends MMRotate training API |
| Student Config | `configs/rotated_retinanet/*student*.py` | 6 hyperparameter fixes | YAML/Python config format |

#### Training Integration with MMRotate

**Standard MMRotate (single model):**
```python
# mmrotate/apis/train.py
trainer = build_trainer(cfg, device)
trainer.train()  # Single model, detection loss only
```

**RAKD KD Training (teacher + student):**
```python
# tools/kd_train.py (our custom addition)
teacher = init_detector(teacher_cfg, teacher_ckpt)  # Frozen, eval mode
student = init_detector(student_cfg, student_ckpt)  # Trainable
adapters = build_spas_for_each_fpn_level()   # 4 lightweight modules

# Custom loss combining detection + KD terms
for epoch in range(epochs):
    for batch in dataloader:
        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_features = teacher.extract_features(batch)
        
        # Student forward (trainable)
        student_features = student.extract_features(batch)
        
        # Adapt student features to teacher spatial dimensions
        adapted_features = [spa(sf) for spa, sf in zip(adapters, student_features)]
        
        # Compute composite loss
        loss = detect_loss(student_outputs, labels) \
             + λ_f * feature_loss(teacher_features, adapted_features) \
             + λ_c * cosine_loss(teacher_features, adapted_features) \
             + λ_a * attention_loss(teacher_features, adapted_features)
        
        loss.backward()
        optimizer.step()
        # Log via MMRotate's hooks
```

#### Modified Student Config (6 Critical Fixes)

Standard MMRotate RetinaNet config required 6 changes for lightweight student:

```python
# configs/rotated_retinanet/rotated_retinanet_obb_r18_fpn_3x_dota_le90_student.py

model = dict(
    backbone=dict(type='ResNet', depth=18),  # Lightweight (vs. R50)
    
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,        # FIX #1: was 128 → causes 0% mAP
        num_outs=5),
    
    head=dict(
        # ... RetinaNet head ...
        loss_bbox=dict(
            type='L1Loss',       # FIX #2: was SmoothL1 → divergence on small objects
            loss_weight=1.0),
)

# FIX #3: Reduced learning rate for stability
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.002, momentum=0.9))  # was 0.01

# FIX #4: Explicit warmup for convergence
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),  # was 500
    dict(type='CosineAnnealingLR', begin=0, end=36, by_epoch=True),
]

# FIX #5: Longer training for small models
train_cfg = dict(max_epochs=36)  # was 12

# FIX #6: Memory-efficient batch size
data = dict(samples_per_gpu=2)  # was 4
```

**Without all 6 fixes:** Student scores ~0% mAP (predicts only background class).  
**With all 6 fixes:** Achieves 70.3% mAP baseline (foundation for Phase 2 KD).

### 5.3 Benchmark Comparison

| Method | Backbone | Params | Input | mAP (%) | FPS |
|--------|----------|--------|-------|---------|-----|
| Oriented R-CNN | R101-FPN | 60M | 1024 | 76.28 | 16 |
| S2ANet | R50-FPN | 38M | 1024 | 74.12 | 28 |
| Rot. FCOS | R50-FPN | 33M | 1024 | 72.31 | 30 |
| Rot. RetinaNet (baseline) | R18-FPN | 11M | 768 | 70.30 | 94 |
| **RAKD (Ours)** | **R18-FPN** | **11M** | **768** | **74.83** | **92** |
| TinyDet | MBv2 | 8M | 640 | 68.14 | 106 |

**Key result:** RAKD achieves **74.83% mAP**—matching single-scale accuracy of 3× heavier models (S2ANet 74.12%) while maintaining real-time speed.

### 5.4 Per-Class Analysis

Largest gains in geometrically complex categories:

| Category | Baseline | RAKD | Δ |
|----------|----------|------|-----|
| Bridge (BR) | 31.6% | 38.8% | +7.2% ⭐ |
| Harbor (HB) | 62.3% | 69.2% | +6.9% |
| Ship (SH) | 82.2% | 87.6% | +5.4% |
| Swimming Pool (SP) | 80.0% | 85.7% | +5.7% |
| Helicopter (HC) | 68.6% | 73.7% | +5.1% |
| Small Vehicle (SV) | 62.2% | 66.0% | +3.8% |
| **Overall** | **70.30** | **74.83** | **+4.53%** |

Pattern: Largest KD benefits in complex geometries (bridges, harbors) where teacher's high-resolution features capture rich spatial structure.

### 5.5 Ablation Study

| Configuration | mAP (%) | Δ | Analysis |
|---|---|---|---|
| Baseline (P1B) | 70.30 | - | Phase 1B checkpoint |
| + SPA architecture only | 71.02 | +0.72 | Layers necessary but insufficient |
| + Channel adapt (1×1) | 72.37 | +1.35 | Feature dimension alignment critical |
| + Bilinear interpolation | 73.21 | +2.19 | Spatial alignment essential |
| + Layer normalization | 74.53 | +3.51 | Stabilization key to convergence |
| Full + L_feature only | 72.41 | +2.11 | Feature matching primary signal |
| Full + L_cosine | 73.61 | +3.31 | Cosine adds complementary structure |
| **Full + L_attention** | **74.83** | **+4.53%** | ⭐ All components optimal |

**Insight:** Each component is complementary; multi-objective formulation avoids diminishing returns.

### 5.6 Speed-Accuracy Trade-off

RAKD occupies Pareto-optimal region:

```
mAP (%)      RAKD (11M, 92 FPS)
   76  ──────────────────────── ⭐
        ╱ Oriented R-CNN (60M)
   74 ─╱ RAKD  S2ANet (38M)
        ╱
   72 ─ Rot. FCOS (33M)
      
   70 ─ Rot. RetinaNet baseline (11M, 94 FPS)
      
   68 ─ TinyDet (8M, 106 FPS)
      └────────────────────────────
        20   50   100   120 FPS → Inference Speed
```

---

## 6. Discussion

### Why RAKD Works

1. **SPAs solve alignment:** Bilinear interpolation bridges resolution gap
2. **Multi-component loss prevents overfitting:** Feature + cosine + attention = balanced supervision
3. **Fair comparison:** Two-phase protocol measures pure KD contribution

### Limitations & Future Work

**Current:** Requires sequential training, evaluated on CNNs only, SPAs discarded at inference.  
**Future:** End-to-end joint training, transformer backbones, efficient knowledge consolidation.

---

## 7. Conclusion

RAKD is the first cross-resolution KD framework for rotated object detection. Through spatial adapters and composite loss, a ResNet-18 student achieves **74.83% mAP at 92 FPS**—a **+4.53% improvement** while maintaining real-time viability. Implementation via MMRotate ensures reproducibility and extensibility.

---

## Installation & Usage

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for setup and [docs/USAGE.md](docs/USAGE.md) for training.

---

## Citation

```bibtex
@article{shekokar2026rakd,
  title={Resolution-Agnostic Knowledge Distillation for Remote Sensing Imagery},
  author={Shekokar, Bhargav and Dhingra, Namya and Darsh, Patel},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

**Also cite MMRotate:**
```bibtex
@article{zhou2022mmrotate,
  title={MMRotate: A PyTorch Toolbox for Rotated Object Detection},
  author={Zhou, Yinxin and others},
  journal={arXiv preprint arXiv:2207.14316},
  year={2022}
}
```

---

## Acknowledgments

OpenMMLab (MMRotate), DOTA dataset maintainers (Wuhan University), institutional GPU resources.

**License:** Apache 2.0 (compatible with MMRotate)  
**Code:** https://github.com/BhargavShekokar3425/RAKDR80R  
**Paper:** [report.pdf](./report.pdf)

