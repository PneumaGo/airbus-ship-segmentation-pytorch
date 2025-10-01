# Airbus Ship Detection with PyTorch (U-Net34 / ResUNet34)

This repository contains a PyTorch implementation of **U-Net34 (ResNet-34 encoder + U-Net decoder)** for the [Kaggle Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection).  

The model segments ships in satellite images using an encoder-decoder architecture with skip connections. The training uses **focal loss + dice loss**, and predictions are exported in Kaggle-compatible **RLE (Run-Length Encoding)** format.

---


## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Validation & Metrics](#validation--metrics)
- [Visualization](#visualization)
- [Submission](#submission)
- [Known Issues & Fixes](#known-issues--fixes)
- [Improvements & Next Steps](#improvements--next-steps)
- [Acknowledgements](#acknowledgements)
- [Demonstration/Inference](#Demonstration-Inference)

---


## Overview
This project implements a **ResNet34-based U-Net** for semantic segmentation of ships from aerial imagery.  

**Main steps implemented in the script:**
1. Load Airbus ship dataset and parse segmentation masks from CSV (RLE format).  
2. Preprocess data: resize images/masks to 256×256, normalize, and build PyTorch datasets/dataloaders.  
3. Build U-Net34 architecture:
   - ResNet-34 as encoder (pretrained weights loaded).
   - Decoder with transposed convolutions and skip connections.  
4. Define custom **loss functions**:
   - `FocalLoss` (to handle class imbalance).
   - `Dice Loss` (for segmentation overlap quality).
   - `MixedLoss` = `alpha * focal + (-log(dice))`.  
5. Train the model with Adam optimizer for 60 epochs.  
6. Evaluate using **Dice** and **IoU** metrics.  
7. Visualize input, prediction, and ground truth side by side.  
8. Generate **Kaggle submission CSV** using RLE encoding.  

---


## Key Features
- **U-Net34 (ResUNet)** implementation from scratch.  
- **Forward hooks** to capture intermediate ResNet feature maps.  
- **Mixed Loss (Focal + Dice)** for robust training on imbalanced masks.  
- **Custom RLE Encoder** to produce Kaggle submission files.  
- **Training pipeline with validation loop** and metrics.  
- **Visualization utilities** to inspect predictions vs. ground truth.  

---


## Repository Structure
├── airbus_ship_detection_pytorch_resunet34.py # Main training & inference script
├── README.md # Project documentation
├── submission.csv # Example Kaggle submission (generated)
└── /input/ # Dataset (to be placed here)
├── airbus-ship-detection/
│ ├── train_v2/
│ ├── test_v2/
│ └── train_ship_segmentations_v2.csv
└── fine-tuning-resnet34-on-ship-detection/
└── models/Resnet34_lable_256_1.h5 # Optional pretrained weights

---


## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/airbus-ship-detection.git
cd airbus-ship-detection
pip install -r requirements.txt
```

### Dependencies
The project relies on the following Python packages:
- **torchvision** — deep learning framework and vision models
- **pandas, numpy** — data handling and numerical computations
- **Pillow** — image processing
- **matplotlib** — visualization
- **scikit-learn** — dataset splitting and utilities
- **tqdm** — progress bars

---


## Data Preparation
- The dataset comes from **Kaggle Airbus Ship Detection Challenge.**
- Masks are provided as **Run-Length Encoded (RLE)** strings in train_ship_segmentations_v2.csv.
- The script decodes RLE into binary masks of shape (768, 768).
- Images and masks are resized to **256×256** for training.

### Transforms used:
- **Images: resize → tensor → normalize (for test).
- **Masks: resize → tensor → binarize.

---


## Model Architecture
- **Encoder:** ResNet-34 truncated backbone (nn.Sequential of first 8 layers).
- **Decoder:** Custom U-Net decoder blocks with transposed conv + skip connections.
- **Hooks:** Forward hooks (SaveFeatures) extract encoder feature maps.
- **Final layer:** ConvTranspose → 1-channel binary mask output.

### Architecture Flow
```text
Input (256x256) → ResNet-34 encoder → SaveFeatures
      ↓
Upsample blocks (x4 with skip connections)
      ↓
ConvTranspose2d → 1 channel mask
```

---


## Training Details
- **Batch size:** 64 (adjust if OOM).
- **Image size:** 256×256.
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-7).
- **Loss function:** MixedLoss(alpha=10, gamma=2).
- **Epochs:** 60.
- **Device:** GPU (cuda if available).
Pretrained weights (Resnet34_lable_256_1.h5) can be loaded for faster convergence.

---


## Validation & Metrics
During validation, the following are reported:
- **Mixed Loss**
- **Dice Coefficient**
- **IoU (Intersection over Union)**

Implementation ensures metrics are averaged across validation batches.

---


## Visualization
The script includes a function Show_images() to display:
- **Input Image**
- **Predicted Mask**
- **Ground Truth Mask**

Helps in visually checking segmentation performance.

---


## Submission
- Predictions are thresholded at 0.5.
- Masks resized back to (768,768) and encoded via RLE.
- Results saved in submission.csv:

ImageId | EncodedPixels

00021ddc3.jpg | 1 10 20 5 ...
...

---


## Known Issues & Fixes

### Hook Removal Bug:
In SaveFeatures.remove() the code is missing parentheses:

    def remove(self): self.hook.remove()

Otherwise, hooks may not be detached properly.

### - **Batch Size / Memory:**
    If out-of-memory occurs, reduce bs from 64 → 8 or 16.

### - **Pretrained Weights Loading:**
    State dict keys are renamed to match model layers (rn. prefix added).

---


## Improvements & Next Steps
- Add data augmentations (flips, rotations, color jitter).
- Implement early stopping & LR scheduling.
- Use mixed precision training for speed (via torch.cuda.amp).
- Try different backbones (ResNet50, EfficientNet) using libraries like segmentation_models.pytorch.
- Add checkpoint saving/loading for best models.
- Explore multi-scale training for better ship detection.

---


## Acknowledgements
- Airbus Ship Detection Challenge
- PyTorch & Torchvision teams
- ResNet-34 backbone pretrained weights

---

## Demonstration Inference
```
https://huggingface.co/spaces/PneumaGo/Satellite_Ship_Detection
```

---


## License
MIT License. Free to use and modify with attribution.
