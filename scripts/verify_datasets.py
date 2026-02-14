#!/usr/bin/env python3
"""
Verify downloaded datasets
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pydicom


def verify_rsna_hemorrhage(data_dir: str) -> bool:
    """Verify RSNA hemorrhage dataset"""
    print("\n=== Verifying RSNA Hemorrhage Dataset ===")
    
    data_path = Path(data_dir) / "rsna_hemorrhage"
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return False
    
    # Check for CSV file
    csv_file = data_path / "stage_2_train.csv"
    if not csv_file.exists():
        print(f"❌ Labels file not found: {csv_file}")
        return False
    
    # Load and check CSV
    df = pd.read_csv(csv_file)
    print(f"✓ Labels file found: {len(df)} rows")
    
    # Check for DICOM files
    dicom_dir = data_path / "stage_2_train_images"
    if not dicom_dir.exists():
        print(f"❌ DICOM directory not found: {dicom_dir}")
        return False
    
    dicom_files = list(dicom_dir.glob("*.dcm"))
    print(f"✓ Found {len(dicom_files)} DICOM files")
    
    # Verify a sample DICOM
    if dicom_files:
        try:
            sample_dcm = pydicom.dcmread(dicom_files[0])
            print(f"✓ Sample DICOM readable: {sample_dcm.Rows}x{sample_dcm.Columns}")
        except Exception as e:
            print(f"❌ Error reading DICOM: {e}")
            return False
    
    print("✅ RSNA Hemorrhage dataset verified!")
    return True


def verify_siim_pneumothorax(data_dir: str) -> bool:
    """Verify SIIM pneumothorax dataset"""
    print("\n=== Verifying SIIM Pneumothorax Dataset ===")
    
    data_path = Path(data_dir) / "siim_pneumothorax"
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return False
    
    # Check for CSV file
    csv_file = data_path / "train-rle.csv"
    if not csv_file.exists():
        print(f"❌ Labels file not found: {csv_file}")
        return False
    
    # Load and check CSV
    df = pd.read_csv(csv_file)
    print(f"✓ Labels file found: {len(df)} rows")
    print(f"✓ Pneumothorax cases: {df[' EncodedPixels'].notna().sum()}")
    
    # Check for DICOM files
    dicom_dir = data_path / "dicom-images-train"
    if not dicom_dir.exists():
        print(f"❌ DICOM directory not found: {dicom_dir}")
        return False
    
    dicom_files = list(dicom_dir.rglob("*.dcm"))
    print(f"✓ Found {len(dicom_files)} DICOM files")
    
    # Verify a sample DICOM
    if dicom_files:
        try:
            sample_dcm = pydicom.dcmread(dicom_files[0])
            print(f"✓ Sample DICOM readable: {sample_dcm.Rows}x{sample_dcm.Columns}")
        except Exception as e:
            print(f"❌ Error reading DICOM: {e}")
            return False
    
    print("✅ SIIM Pneumothorax dataset verified!")
    return True


def verify_mimic_cxr(data_dir: str) -> bool:
    """Verify MIMIC-CXR dataset"""
    print("\n=== Verifying MIMIC-CXR Dataset ===")
    
    data_path = Path(data_dir) / "mimic_cxr"
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return False
    
    # Check for metadata files
    required_files = [
        "mimic-cxr-2.0.0-metadata.csv",
        "mimic-cxr-2.0.0-split.csv",
        "mimic-cxr-2.0.0-chexpert.csv",
    ]
    
    for file_name in required_files:
        file_path = data_path / file_name
        if not file_path.exists():
            print(f"⚠ Metadata file not found: {file_name}")
        else:
            df = pd.read_csv(file_path)
            print(f"✓ {file_name}: {len(df)} rows")
    
    # Check for DICOM files
    files_dir = data_path / "files"
    if files_dir.exists():
        dicom_files = list(files_dir.rglob("*.dcm"))
        print(f"✓ Found {len(dicom_files)} DICOM files")
        
        if dicom_files:
            print("✅ MIMIC-CXR dataset verified!")
            return True
    else:
        print("⚠ DICOM files directory not found")
        print("  Note: MIMIC-CXR images may need separate download")
    
    return True


def main():
    """Main verification function"""
    print("=" * 50)
    print("Dataset Verification Tool")
    print("=" * 50)
    
    data_dir = "./data/raw"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)
    
    results = {}
    
    # Verify each dataset
    results['RSNA Hemorrhage'] = verify_rsna_hemorrhage(data_dir)
    results['SIIM Pneumothorax'] = verify_siim_pneumothorax(data_dir)
    results['MIMIC-CXR'] = verify_mimic_cxr(data_dir)
    
    # Summary
    print("\n" + "=" * 50)
    print("Verification Summary")
    print("=" * 50)
    
    for dataset, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {dataset}")
    
    print("\n")
    
    if all(results.values()):
        print("🎉 All datasets verified successfully!")
        sys.exit(0)
    else:
        print("⚠ Some datasets need attention")
        sys.exit(1)


if __name__ == "__main__":
    main()
