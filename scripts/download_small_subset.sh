#!/bin/bash

# Download SMALL subset for local development (2-5GB total)

set -e

echo "=========================================="
echo "Small Dataset Download (Local Development)"
echo "=========================================="
echo ""

OUTPUT_DIR="./data/raw/small_subset"
mkdir -p "$OUTPUT_DIR"

echo "This downloads SMALL datasets suitable for laptops:"
echo "  - ~2,000 images total"
echo "  - ~2-5GB storage needed"
echo "  - Enough for prototyping and demo"
echo ""

# 1. Download CQ500 (500 CT scans for hemorrhage)
echo "Downloading CQ500 dataset (500 CT scans, ~2GB)..."
mkdir -p "$OUTPUT_DIR/cq500"

# This is a publicly available dataset
echo "Visit: http://headctstudy.qure.ai/dataset"
echo "Download CQ500 dataset and extract to: $OUTPUT_DIR/cq500"
echo ""
echo "Or use this Kaggle subset:"
kaggle datasets download -d felipekitamura/head-ct-hemorrhage
unzip -q head-ct-hemorrhage.zip -d "$OUTPUT_DIR/cq500/"
rm head-ct-hemorrhage.zip

echo "✓ CQ500 downloaded!"
echo ""

# 2. Download chest X-ray subset
echo "Downloading chest X-ray subset (1,000 images, ~500MB)..."
mkdir -p "$OUTPUT_DIR/chest_xray"

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -q chest-xray-pneumonia.zip -d "$OUTPUT_DIR/chest_xray/"
rm chest-xray-pneumonia.zip

# Keep only a subset
echo "Keeping only 1,000 images for development..."
find "$OUTPUT_DIR/chest_xray" -name "*.jpeg" | tail -n +1001 | xargs rm -f 2>/dev/null || true

echo "✓ Chest X-ray subset downloaded!"
echo ""

# Summary
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
du -sh "$OUTPUT_DIR"
echo ""
echo "You now have:"
echo "  - 500 CT scans for hemorrhage detection"
echo "  - 1,000 chest X-rays for pneumothorax"
echo "  - Total: ~2-5GB"
echo ""
echo "Perfect for laptop development!"
