#!/bin/bash

# Download MIMIC-CXR dataset from PhysioNet

set -e  # Exit on error

echo "=========================================="
echo "MIMIC-CXR Dataset Download"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="./data/raw/mimic_cxr"
MIMIC_VERSION="2.0.0"

# Create directory
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

echo "⚠ IMPORTANT: MIMIC-CXR requires credentialed access"
echo ""
echo "To download MIMIC-CXR, you need to:"
echo "1. Create a PhysioNet account at https://physionet.org/register/"
echo "2. Complete required training (CITI 'Data or Specimens Only Research')"
echo "3. Get credentialed (takes 1-2 weeks)"
echo "4. Sign the Data Use Agreement for MIMIC-CXR"
echo ""
echo "Visit: https://physionet.org/content/mimic-cxr/$MIMIC_VERSION/"
echo ""

read -p "Have you completed all steps above? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please complete the credentialing process first."
    echo "This typically takes 1-2 weeks for PhysioNet to review."
    exit 1
fi

echo ""
echo "You have two options to download MIMIC-CXR:"
echo ""
echo "Option 1: Download via PhysioNet website"
echo "  - Go to https://physionet.org/content/mimic-cxr/$MIMIC_VERSION/"
echo "  - Click 'Files' tab"
echo "  - Download files manually or use wget (instructions on site)"
echo ""
echo "Option 2: Use wget with credentials (recommended)"
echo ""

read -p "Use wget to download? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Enter your PhysioNet username: " USERNAME
    read -sp "Enter your PhysioNet password: " PASSWORD
    echo ""
    echo ""
    
    echo "Downloading MIMIC-CXR files..."
    echo "This will take several hours (dataset is ~500GB)..."
    echo ""
    
    # Base URL
    BASE_URL="https://physionet.org/files/mimic-cxr/$MIMIC_VERSION"
    
    # Download files
    cd "$OUTPUT_DIR"
    
    # Download metadata files first (small)
    echo "Downloading metadata files..."
    wget --user="$USERNAME" --password="$PASSWORD" -r -N -c -np \
         "$BASE_URL/mimic-cxr-$MIMIC_VERSION-metadata.csv.gz"
    wget --user="$USERNAME" --password="$PASSWORD" -r -N -c -np \
         "$BASE_URL/mimic-cxr-$MIMIC_VERSION-split.csv.gz"
    wget --user="$USERNAME" --password="$PASSWORD" -r -N -c -np \
         "$BASE_URL/mimic-cxr-$MIMIC_VERSION-chexpert.csv.gz"
    wget --user="$USERNAME" --password="$PASSWORD" -r -N -c -np \
         "$BASE_URL/mimic-cxr-$MIMIC_VERSION-negbio.csv.gz"
    
    echo ""
    echo "Metadata downloaded. Extracting..."
    gunzip -f *.csv.gz
    
    echo ""
    echo "NOTE: Full image download (~500GB) starting..."
    echo "Consider downloading only a subset for development:"
    echo ""
    echo "  # Download just one patient folder for testing:"
    echo "  wget --user=$USERNAME --password=*** -r -N -c -np \\"
    echo "       $BASE_URL/files/p10/"
    echo ""
    
    read -p "Download full dataset (500GB)? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wget --user="$USERNAME" --password="$PASSWORD" -r -N -c -np \
             --reject "index.html*" "$BASE_URL/files/"
    else
        echo "Downloading sample (first 100 patients only)..."
        for i in {10..19}; do
            wget --user="$USERNAME" --password="$PASSWORD" -r -N -c -np \
                 --reject "index.html*" "$BASE_URL/files/p$i/" || true
        done
    fi
    
    cd ../../../
    
    echo ""
    echo "✓ Download complete!"
    
else
    echo ""
    echo "Manual download instructions:"
    echo ""
    echo "1. Go to: https://physionet.org/content/mimic-cxr/$MIMIC_VERSION/"
    echo "2. Log in with your credentials"
    echo "3. Click 'Files' tab"
    echo "4. Download files to: $OUTPUT_DIR"
    echo ""
    echo "Recommended files:"
    echo "  - mimic-cxr-$MIMIC_VERSION-metadata.csv.gz"
    echo "  - mimic-cxr-$MIMIC_VERSION-split.csv.gz"
    echo "  - mimic-cxr-$MIMIC_VERSION-chexpert.csv.gz"
    echo "  - files/ directory (DICOM images)"
    echo ""
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Extract CSV files if downloaded as .gz"
echo "2. Verify dataset integrity"
echo "3. Run preprocessing scripts"
