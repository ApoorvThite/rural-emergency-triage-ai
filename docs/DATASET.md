# Dataset Documentation

## Training Datasets

### 1. RSNA Intracranial Hemorrhage Detection
- **Source**: Kaggle Competition
- **Size**: ~25,000 CT head scans
- **Labels**: 5 hemorrhage subtypes
- **Format**: DICOM
- **URL**: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

### 2. SIIM-ACR Pneumothorax Segmentation
- **Source**: Kaggle Competition
- **Size**: ~12,000 chest X-rays
- **Labels**: Pneumothorax segmentation masks
- **Format**: DICOM + PNG masks
- **URL**: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

### 3. MIMIC-CXR
- **Source**: PhysioNet
- **Size**: 377,110 chest X-rays
- **Labels**: 14 pathologies + free-text reports
- **Format**: DICOM + structured reports
- **URL**: https://physionet.org/content/mimic-cxr/

## Data Preprocessing

1. **DICOM to PNG conversion**
2. **Windowing** (brain: 40/80, lung: -600/1500)
3. **Resizing** to 512x512 or 224x224
4. **Normalization** (ImageNet stats)
5. **Augmentation** (rotation, flip, brightness)

## Data Split

- Training: 70%
- Validation: 15%
- Test: 15%

Stratified by condition prevalence.

