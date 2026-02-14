"""
Data loading utilities for medical imaging datasets
"""

import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MedicalImageDataset(Dataset):
    """Dataset for medical imaging tasks"""
    
    def __init__(
        self,
        data_dir: str,
        metadata_df: pd.DataFrame,
        image_size: int = 512,
        transform: Optional[A.Compose] = None,
        is_training: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing image files
            metadata_df: DataFrame with image paths and labels
            image_size: Target image size
            transform: Albumentations transforms
            is_training: Whether this is training data
        """
        self.data_dir = data_dir
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata_df.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, row['image_path'])
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get labels
        if 'label' in row:
            label = torch.tensor(row['label'], dtype=torch.long)
        else:
            # Multi-label case
            label_cols = [col for col in row.index if col.startswith('label_')]
            label = torch.tensor(row[label_cols].values, dtype=torch.float32)
        
        return {
            'image': image,
            'label': label,
            'image_id': row.get('image_id', idx),
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess medical image"""
        
        if image_path.endswith('.dcm'):
            # Load DICOM
            dcm = pydicom.dcmread(image_path)
            image = dcm.pixel_array.astype(np.float32)
            
            # Apply windowing based on image type
            if hasattr(dcm, 'SeriesDescription'):
                if 'head' in dcm.SeriesDescription.lower():
                    # Brain window: center=40, width=80
                    image = self._apply_windowing(image, center=40, width=80)
                elif 'chest' in dcm.SeriesDescription.lower():
                    # Lung window: center=-600, width=1500
                    image = self._apply_windowing(image, center=-600, width=1500)
            else:
                # Default: normalize to 0-255
                image = self._normalize_image(image)
                
        else:
            # Load PNG/JPEG
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to 3-channel (RGB) for compatibility with vision models
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        return image
    
    def _apply_windowing(
        self, 
        image: np.ndarray, 
        center: float, 
        width: float
    ) -> np.ndarray:
        """Apply windowing to CT images"""
        img_min = center - width // 2
        img_max = center + width // 2
        image = np.clip(image, img_min, img_max)
        image = ((image - img_min) / (img_max - img_min) * 255.0)
        return image.astype(np.uint8)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        image = image - image.min()
        if image.max() > 0:
            image = image / image.max() * 255.0
        return image.astype(np.uint8)


def get_transforms(image_size: int = 512, is_training: bool = True) -> A.Compose:
    """Get augmentation transforms"""
    
    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def create_dataloaders(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test dataloaders"""
    
    data_config = config['data']
    training_config = config['training']
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        data_dir=data_config['data_dir'],
        metadata_df=train_df,
        image_size=data_config['image_size'],
        transform=get_transforms(data_config['image_size'], is_training=True),
        is_training=True,
    )
    
    val_dataset = MedicalImageDataset(
        data_dir=data_config['data_dir'],
        metadata_df=val_df,
        image_size=data_config['image_size'],
        transform=get_transforms(data_config['image_size'], is_training=False),
        is_training=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['per_device_train_batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True,
    )
    
    test_loader = None
    if test_df is not None:
        test_dataset = MedicalImageDataset(
            data_dir=data_config['data_dir'],
            metadata_df=test_df,
            image_size=data_config['image_size'],
            transform=get_transforms(data_config['image_size'], is_training=False),
            is_training=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config['per_device_eval_batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader


def prepare_rsna_hemorrhage_data(data_dir: str) -> pd.DataFrame:
    """Prepare RSNA hemorrhage dataset metadata"""
    
    # Load labels
    labels_path = os.path.join(data_dir, 'stage_2_train.csv')
    df = pd.read_csv(labels_path)
    
    # Parse image IDs and labels
    df[['image_id', 'label_type']] = df['ID'].str.rsplit('_', n=1, expand=True)
    df['label'] = df['Label']
    
    # Pivot to get one row per image
    df_pivot = df.pivot_table(
        index='image_id',
        columns='label_type',
        values='label',
        fill_value=0
    ).reset_index()
    
    # Add image paths
    df_pivot['image_path'] = df_pivot['image_id'].apply(
        lambda x: f'stage_2_train_images/{x}.dcm'
    )
    
    return df_pivot


def prepare_siim_pneumothorax_data(data_dir: str) -> pd.DataFrame:
    """Prepare SIIM pneumothorax dataset metadata"""
    
    # Load labels
    labels_path = os.path.join(data_dir, 'train-rle.csv')
    df = pd.read_csv(labels_path)
    
    # Binary label: 1 if RLE exists (pneumothorax present), 0 otherwise
    df['label'] = (~df[' EncodedPixels'].isna()).astype(int)
    
    # Add image paths
    df['image_path'] = df['ImageId'].apply(
        lambda x: f'dicom-images-train/{x}.dcm'
    )
    
    # Keep only necessary columns
    df = df[['ImageId', 'image_path', 'label']].rename(columns={'ImageId': 'image_id'})
    
    return df
