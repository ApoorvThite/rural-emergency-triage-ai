"""
Simple MedGemma training script - no wandb dependency
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

print("✓ Starting training script...")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

def main(config_path: str):
    """Main training function"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Config loaded from {config_path}")
    print(f"✓ Model: {config['model']['name']}")
    print(f"✓ Dataset: {config['data']['dataset_name']}")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    print("\n⚠ Note: This is a simplified training script for testing.")
    print("⚠ For actual training, you'll need the complete implementation.")
    print("\nNext steps:")
    print("1. Download dataset")
    print("2. Implement data loading")
    print("3. Load MedGemma model")
    print("4. Run training loop")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file',
    )
    args = parser.parse_args()
    
    main(args.config)
