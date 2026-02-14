#!/bin/bash

# Download SIIM-ACR Pneumothorax Segmentation dataset

set -e  # Exit on error

echo "=========================================="
echo "SIIM Pneumothorax Dataset Download"
echo "=========================================="
echo ""

# Configuration
COMPETITION="siim-acr-pneumothorax-segmentation"
OUTPUT_DIR="./data/raw/siim_pneumothorax"
TEMP_DIR="./data/temp/siim_pneumothorax"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Check Kaggle setup
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found. Run setup_kaggle.sh first."
    exit 1
fi

# Accept competition rules
echo "Note: You must accept the competition rules first!"
echo "Visit: https://www.kaggle.com/competitions/$COMPETITION/rules"
echo "Click 'I Understand and Accept'"
echo ""
read -p "Have you accepted the rules? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please accept the competition rules first, then re-run this script."
    exit 1
fi

# Download competition files
echo "Downloading competition files..."
echo "This will take a while (dataset is ~12GB)..."
echo ""

cd "$TEMP_DIR"

# Download all files
kaggle competitions download -c "$COMPETITION"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download dataset."
    echo "Make sure you've accepted the competition rules."
    exit 1
fi

echo ""
echo "Download complete! Extracting files..."
echo ""

# Extract files
for file in *.zip; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        unzip -q "$file"
        
        # Check for nested zips
        for nested in *.zip; do
            if [ -f "$nested" ] && [ "$nested" != "$file" ]; then
                echo "Extracting nested $nested..."
                unzip -q "$nested"
                rm "$nested"
            fi
        done
        
        rm "$file"
    fi
done

echo ""
echo "Organizing files..."

# Move back to project root
cd ../../..

# Move extracted files to output directory
mv "$TEMP_DIR"/* "$OUTPUT_DIR/" 2>/dev/null || true
rmdir "$TEMP_DIR" 2>/dev/null || true

# Create subdirectories
mkdir -p "$OUTPUT_DIR/dicom-images-train"
mkdir -p "$OUTPUT_DIR/dicom-images-test"

echo ""
echo "✓ Dataset downloaded and organized!"
echo ""
echo "Dataset location: $OUTPUT_DIR"
echo ""

# Show dataset info
echo "Dataset contents:"
ls -lh "$OUTPUT_DIR"
echo ""

# Count files
if [ -d "$OUTPUT_DIR/dicom-images-train" ]; then
    TRAIN_COUNT=$(find "$OUTPUT_DIR/dicom-images-train" -name "*.dcm" 2>/dev/null | wc -l)
    echo "Training images: $TRAIN_COUNT DICOM files"
fi

if [ -f "$OUTPUT_DIR/train-rle.csv" ]; then
    LABEL_COUNT=$(wc -l < "$OUTPUT_DIR/train-rle.csv")
    echo "Training labels: $LABEL_COUNT rows"
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
