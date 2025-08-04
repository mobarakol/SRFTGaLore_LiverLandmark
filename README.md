# SRFTGaLore_LiverLandmark

## Abstract:
Accurate detection and delineation of anatomical structures in medical imaging are critical for computer-assisted interventions, particularly in laparoscopic liver surgery, where 2D video streams limit depth perception and complicate landmark localization.

While recent works have leveraged monocular depth cues for enhanced landmark detection, challenges remain in fusing RGB and depth features and in efficiently adapting large-scale vision models to surgical domains.

In this paper, we propose a **depth-guided liver landmark segmentation framework** that integrates semantic and geometric cues via pretrained vision foundation model encoders. Specifically:
- We use the **encoder of Segment Anything Model V2 (SAM2)** to extract RGB-based semantic features.
- We use the **encoder of Depth Anything V2 (DA2)** to extract depth-aware geometric features.

To efficiently adapt SAM2 to surgical data, we introduce **SRFT-GaLore**, a novel low-rank gradient projection method that replaces the computationally expensive SVD in GaLore with a **Subsampled Randomized Fourier Transform (SRFT)**. This reduces projection complexity, enabling efficient fine-tuning of high-dimensional attention layers without sacrificing representational power.

Additionally, we design a **cross-attention fusion module** to effectively integrate depth and RGB modalities for accurate landmark segmentation.


## Dataset:
This project uses the **L3D dataset** introduced in:

> *Pei, Jialun, et al. "Depth-Driven Geometric Prompt Learning for Laparoscopic Liver Landmark Detection." MICCAI 2024.*

The dataset is publicly available via the [D2GPLand GitHub repository](https://github.com/PJLallen/D2GPLand).  
All rights belong to the original authors.

##  Key Components and Novelty

The core contributions of this work are implemented in the following files:

| File | Class / Function | Description |
|------|------------------|-------------|
| `models/model.py` | `Model` | Main segmentation framework that integrates SAM2, DA2, SRFT-GaLore, and cross-attention fusion.  |
| `models/dataloader.py` | `dataset` | Dataloader + data augmentation. |
| `utils/galore.py` | `SRFTGaLoreProjector` | Implements the SRFT-based low-rank projection replacing SVD in GaLore. |

## Environment Setup:
We recommend creating a fresh Python 3.10 environment and installing dependencies via pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To install dependencies manually:
```bash
pip install torch torchvision accelerate einops opencv-python \
             scikit-learn scikit-image matplotlib pandas \
             transformers peft surface-distance tensorly
```


## Training command:

```bash
python train.py
```

## Inference command:
```bash
python inference.py
```
