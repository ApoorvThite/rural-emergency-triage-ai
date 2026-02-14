#!/bin/bash

# Setup Kaggle API credentials

echo "=========================================="
echo "Kaggle API Setup"
echo "=========================================="
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle API..."
    pip install kaggle
fi

# Check for API credentials
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

if [ -f "$KAGGLE_JSON" ]; then
    echo "✓ Kaggle credentials found at $KAGGLE_JSON"
else
    echo "⚠ Kaggle credentials not found!"
    echo ""
    echo "To setup Kaggle API:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Download kaggle.json"
    echo "5. Move it to ~/.kaggle/kaggle.json"
    echo ""
    echo "Run these commands:"
    echo "  mkdir -p ~/.kaggle"
    echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Set correct permissions
chmod 600 "$KAGGLE_JSON"

echo "✓ Kaggle API is ready!"
echo ""

# Test connection
echo "Testing Kaggle API connection..."
kaggle competitions list --page 1 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Successfully connected to Kaggle!"
else
    echo "✗ Failed to connect to Kaggle. Check your credentials."
    exit 1
fi

echo ""
echo "Setup complete!"
