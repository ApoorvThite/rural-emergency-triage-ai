#!/bin/bash

# Master script to download all datasets

set -e

echo "=========================================="
echo "Rural Emergency Triage AI"
echo "Dataset Download Manager"
echo "=========================================="
echo ""

# Check if we're in the project root
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

echo "This script will help you download the required datasets."
echo ""
echo "Available datasets:"
echo "  1. RSNA Intracranial Hemorrhage (~100GB) [Kaggle]"
echo "  2. SIIM-ACR Pneumothorax (~12GB) [Kaggle]"
echo "  3. MIMIC-CXR (~500GB) [PhysioNet - requires credentials]"
echo "  4. Public datasets (smaller, no credentials needed)"
echo "  5. Download all Kaggle datasets (1+2)"
echo "  6. Exit"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Setting up Kaggle..."
        ./scripts/data_download/setup_kaggle.sh
        echo ""
        echo "Downloading RSNA Hemorrhage dataset..."
        ./scripts/data_download/download_rsna_hemorrhage.sh
        ;;
    2)
        echo ""
        echo "Setting up Kaggle..."
        ./scripts/data_download/setup_kaggle.sh
        echo ""
        echo "Downloading SIIM Pneumothorax dataset..."
        ./scripts/data_download/download_siim_pneumothorax.sh
        ;;
    3)
        echo ""
        echo "Downloading MIMIC-CXR dataset..."
        ./scripts/data_download/download_mimic_cxr.sh
        ;;
    4)
        echo ""
        echo "Downloading public datasets..."
        ./scripts/data_download/download_public_datasets.sh
        ;;
    5)
        echo ""
        echo "Setting up Kaggle..."
        ./scripts/data_download/setup_kaggle.sh
        echo ""
        echo "Downloading RSNA Hemorrhage dataset..."
        ./scripts/data_download/download_rsna_hemorrhage.sh
        echo ""
        echo "Downloading SIIM Pneumothorax dataset..."
        ./scripts/data_download/download_siim_pneumothorax.sh
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Download Script Completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify downloaded data in ./data/raw/"
echo "2. Run preprocessing scripts"
echo "3. Start training models"
echo ""
echo "For help, see: docs/DATASET.md"
