# Dataset Download Guide

## Quick Start

### Step 1: Setup Kaggle API
```bash
# Install Kaggle CLI
pip install kaggle

# Get API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Scroll to "API" section
# 3. Click "Create New API Token"
# 4. Move downloaded kaggle.json to ~/.kaggle/

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Verify setup
./scripts/data_download/setup_kaggle.sh
```

### Step 2: Download Datasets

**Option A: Interactive Menu**
```bash
./scripts/download_datasets.sh
```

**Option B: Individual Downloads**
```bash
# RSNA Hemorrhage (~100GB)
./scripts/data_download/download_rsna_hemorrhage.sh

# SIIM Pneumothorax (~12GB)
./scripts/data_download/download_siim_pneumothorax.sh

# MIMIC-CXR (~500GB, requires PhysioNet credentials)
./scripts/data_download/download_mimic_cxr.sh
```

### Step 3: Verify Downloads
```bash
python scripts/verify_datasets.py
```

## Dataset Details

### 1. RSNA Intracranial Hemorrhage Detection

- **Size**: ~100GB
- **Images**: 25,000+ CT head scans
- **Source**: Kaggle
- **License**: Competition data use agreement
- **URL**: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

**Labels**: 5 hemorrhage subtypes
- Epidural
- Subdural  
- Subarachnoid
- Intraventricular
- Intraparenchymal

### 2. SIIM-ACR Pneumothorax Segmentation

- **Size**: ~12GB
- **Images**: 12,000+ chest X-rays
- **Source**: Kaggle
- **License**: Competition data use agreement
- **URL**: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

**Labels**: Binary + segmentation masks

### 3. MIMIC-CXR

- **Size**: ~500GB
- **Images**: 377,000+ chest X-rays
- **Source**: PhysioNet
- **License**: Credentialed access required
- **URL**: https://physionet.org/content/mimic-cxr/

**Requirements**:
1. PhysioNet account
2. CITI training completion
3. Data Use Agreement signed

## Troubleshooting

### Kaggle API Issues

**Problem**: `401 Unauthorized`
```bash
# Solution: Check credentials
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Problem**: Competition rules not accepted
```bash
# Solution: Visit competition page and accept rules
# Example: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/rules
```

### Download Interrupted
```bash
# Resume download by re-running the script
# Kaggle CLI will resume where it left off
./scripts/data_download/download_rsna_hemorrhage.sh
```

### Disk Space Issues
```bash
# Check available space
df -h

# Download to external drive
OUTPUT_DIR="/path/to/external/drive/data" \
./scripts/data_download/download_rsna_hemorrhage.sh
```

## Alternative: Download Smaller Subsets

For development/testing, you can use smaller datasets:
```bash
# Download public datasets (no credentials needed)
./scripts/data_download/download_public_datasets.sh
```

## Data Organization

After download, your structure should look like:
```
data/
├── raw/
│   ├── rsna_hemorrhage/
│   │   ├── stage_2_train_images/
│   │   │   └── *.dcm
│   │   └── stage_2_train.csv
│   ├── siim_pneumothorax/
│   │   ├── dicom-images-train/
│   │   │   └── *.dcm
│   │   └── train-rle.csv
│   └── mimic_cxr/
│       ├── files/
│       └── *.csv
└── processed/
    └── (generated during training)
```

## Next Steps

After downloading:

1. **Verify datasets**: `python scripts/verify_datasets.py`
2. **Explore data**: See `notebooks/01_data_exploration.ipynb`
3. **Start training**: `python src/models/medgemma/train.py --config configs/hemorrhage_detection.yaml`
