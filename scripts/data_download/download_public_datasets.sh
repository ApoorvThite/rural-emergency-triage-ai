#!/bin/bash

# Download smaller, publicly available medical imaging datasets
# These don't require credentials and are good for quick prototyping

set -e

echo "=========================================="
echo "Public Medical Imaging Datasets"
echo "=========================================="
echo ""

OUTPUT_DIR="./data/raw/public_datasets"
mkdir -p "$OUTPUT_DIR"

echo "This script downloads several smaller, publicly available datasets:"
echo "1. NIH Chest X-ray14 (subset)"
echo "2. MURA (Musculoskeletal Radiographs)"
echo "3. Sample CT scans"
echo ""

# 1. NIH Chest X-ray14 - Download via Kaggle
echo "----------------------------------------"
echo "1. NIH Chest X-ray14 Dataset"
echo "----------------------------------------"
echo ""

if command -v kaggle &> /dev/null; then
    read -p "Download NIH Chest X-ray14? (15GB) (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$OUTPUT_DIR/nih_chest_xray"
        cd "$OUTPUT_DIR/nih_chest_xray"
        
        echo "Downloading from Kaggle..."
        kaggle datasets download -d nih-chest-xrays/data
        
        echo "Extracting..."
        unzip -q data.zip
        rm data.zip
        
        cd ../../../..
        echo "✓ NIH Chest X-ray14 downloaded!"
    fi
else
    echo "Kaggle CLI not available. Skipping NIH Chest X-ray14."
fi

echo ""

# 2. MURA Dataset
echo "----------------------------------------"
echo "2. MURA Dataset (Bone X-rays)"
echo "----------------------------------------"
echo ""
echo "MURA requires registration but is freely available."
echo "Visit: https://stanfordmlgroup.github.io/competitions/mura/"
echo ""
echo "Manual download instructions:"
echo "1. Register at the link above"
echo "2. Download MURA-v1.1.zip (~40GB)"
echo "3. Extract to: $OUTPUT_DIR/mura/"
echo ""

# 3. Sample CT Scans from Cancer Imaging Archive
echo "----------------------------------------"
echo "3. Sample CT Scans"
echo "----------------------------------------"
echo ""

read -p "Download sample CT scans from TCIA? (~2GB) (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$OUTPUT_DIR/sample_ct"
    
    echo "Downloading sample CT collection..."
    echo "Using The Cancer Imaging Archive (TCIA) public data..."
    
    # Note: This is a placeholder. TCIA requires their data portal.
    echo ""
    echo "To download CT scans from TCIA:"
    echo "1. Visit: https://www.cancerimagingarchive.net/"
    echo "2. Browse Collections (no registration required for public data)"
    echo "3. Recommended collections:"
    echo "   - CT Colonography Trial"
    echo "   - Head-Neck-PET-CT"
    echo "4. Download using NBIA Data Retriever"
    echo ""
fi

echo ""
echo "=========================================="
echo "Additional Free Resources"
echo "=========================================="
echo ""
echo "Here are more free medical imaging datasets:"
echo ""
echo "Chest X-rays:"
echo "  - CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/"
echo "  - PadChest: http://bimcv.cipf.es/bimcv-projects/padchest/"
echo ""
echo "CT Scans:"
echo "  - COVID-CT: https://github.com/UCSD-AI4H/COVID-CT"
echo "  - LIDC-IDRI: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
echo ""
echo "Pathology:"
echo "  - PatchCamelyon: https://patchcamelyon.grand-challenge.org/"
echo "  - BreakHis: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"
echo ""
echo "All-in-one:"
echo "  - Grand Challenge: https://grand-challenge.org/challenges/"
echo "  - Kaggle Medical: https://www.kaggle.com/datasets?tags=13303-Health"
echo ""

echo "Download complete for available datasets!"
