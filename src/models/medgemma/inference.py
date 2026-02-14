"""
MedGemma inference script
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union
import numpy as np
from PIL import Image
import yaml
from pathlib import Path

from .train import MedGemmaClassifier


class MedGemmaInference:
    """Inference wrapper for MedGemma classifier"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'cuda',
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training config
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        num_classes = len(self.config['task']['classes'])
        self.model = MedGemmaClassifier(
            model_name=self.config['model']['name'],
            num_classes=num_classes,
            load_in_4bit=self.config['model']['load_in_4bit'],
            device_map=self.config['model']['device_map'],
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.class_names = self.config['task']['classes']
        self.threshold = self.config['task']['threshold']
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_probabilities: bool = True,
    ) -> Dict:
        """
        Make prediction on single image
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # TODO: Apply proper preprocessing (resize, normalize, etc.)
        # This should match the preprocessing used during training
        
        # Convert to tensor and add batch dimension
        # image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        # logits = self.model(image_tensor)
        # probabilities = torch.softmax(logits, dim=1)[0]
        
        # For now, return dummy prediction
        # TODO: Replace with actual inference
        probabilities = torch.rand(len(self.class_names))
        probabilities = probabilities / probabilities.sum()
        
        predicted_class_idx = probabilities.argmax().item()
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_name: prob.item()
                for class_name, prob in zip(self.class_names, probabilities)
            }
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
    ) -> List[Dict]:
        """Make predictions on batch of images"""
        
        return [self.predict(img) for img in images]


if __name__ == '__main__':
    # Example usage
    inference = MedGemmaInference(
        model_path='./data/models/hemorrhage_detection/best_model.pt',
        config_path='./configs/hemorrhage_detection.yaml',
    )
    
    result = inference.predict('./data/raw/sample_ct.dcm')
    print(result)
