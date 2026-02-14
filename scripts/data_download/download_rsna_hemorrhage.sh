#!/bin/bash

# Download RSNA Intracranial Hemorrhage Detection dataset

set -e  # Exit on error

echo "=========================================="
echo "RSNA Hemorrhage Dataset Download"
echo "=========================================="
echo ""

# Configuration
COMPETITION="rsna-intracranial-hemorrhage-detection"
OUTPUT_DIR="./data/raw/rsna_hemorrhage"
TEMP_DIR="./data/temp/rsna_hemorrhage"

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
echo "This will take a while (dataset is ~100GB)..."
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
        unzip -q "$file" -d "../../raw/rsna_hemorrhage/"
        rm "$file"
    fi
done

echo ""
echo "Organizing files..."

# Move back to project root
cd ../../..

# Create subdirectories
mkdir -p "$OUTPUT_DIR/stage_2_train_images"
mkdir -p "$OUTPUT_DIR/stage_2_test_images"

# Move DICOM files if needed
if [ -d "$TEMP_DIR" ]; then
    mv "$TEMP_DIR"/* "$OUTPUT_DIR/" 2>/dev/null || true
    rmdir "$TEMP_DIR" 2>/dev/null || true
fi

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
TRAIN_COUNT=$(find "$OUTPUT_DIR/stage_2_train_images" -name "*.dcm" 2>/dev/null | wc -l)
echo "Training images: $TRAIN_COUNT DICOM files"

if [ -f "$OUTPUT_DIR/stage_2_train.csv" ]; then
    LABEL_COUNT=$(wc -l < "$OUTPUT_DIR/stage_2_train.csv")
    echo "Training labels: $LABEL_COUNT rows"
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
