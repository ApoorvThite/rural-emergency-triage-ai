"""
MedGemma inference script using conversational format.

Loads a fine-tuned MedGemma model (LoRA adapter) and runs inference
using the same conversational prompt format as training.
"""

import torch
import json
from typing import Dict, List, Union, Optional
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


class MedGemmaInference:
    """Inference wrapper for fine-tuned MedGemma model."""
    
    def __init__(
        self,
        lora_adapter_path: str,
        base_model_id: Optional[str] = None,
    ):
        """
        Args:
            lora_adapter_path: Path to fine-tuned LoRA adapter directory
            base_model_id: Base MedGemma model ID (auto-detected from task_config.json if not provided)
        """
        adapter_path = Path(lora_adapter_path)
        
        # Load task config
        config_file = adapter_path / 'task_config.json'
        if not config_file.exists():
            raise FileNotFoundError(
                f"task_config.json not found in {lora_adapter_path}. "
                "Make sure you're pointing to a directory with a trained model."
            )
        
        with open(config_file, 'r') as f:
            self.task_config = json.load(f)
        
        # Get model ID
        if base_model_id is None:
            base_model_id = self.task_config['model_id']
        
        self.class_labels = self.task_config['class_labels']
        self.prompt = self.task_config['prompt']
        self.task_name = self.task_config['task']
        
        print(f"Loading MedGemma model: {base_model_id}")
        print(f"Task: {self.task_name}")
        print(f"Classes: {len(self.class_labels)}")
        
        # Determine dtype
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.compute_dtype = torch.bfloat16
        else:
            self.compute_dtype = torch.float16
        
        # Load base model
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=self.compute_dtype,
            device_map='auto',
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(str(adapter_path))
        
        print("✓ Model loaded successfully!")
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        max_new_tokens: int = 100,
    ) -> Dict:
        """
        Make prediction on single image using conversational format.
        
        Args:
            image: Image path, numpy array, or PIL Image
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dictionary with prediction, raw response, and parsed class
        """
        # Load image
        if isinstance(image, str):
            if image.endswith('.dcm'):
                import pydicom
                dcm = pydicom.dcmread(image)
                arr = dcm.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                pil_image = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
            else:
                pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        # Build conversational input
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': self.prompt},
                ],
            },
        ]
        
        # Format with chat template
        text_input = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # Process inputs
        inputs = self.processor(
            text=text_input,
            images=[pil_image],
            return_tensors='pt',
        ).to(self.model.device, dtype=self.compute_dtype)
        
        input_len = inputs['input_ids'].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # Decode response
        response = self.processor.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()
        
        # Parse predicted class
        predicted_class_idx = -1
        predicted_class = None
        
        for idx, class_label in enumerate(self.class_labels):
            # Check if the letter prefix is in the response (e.g., "A:", "B:")
            letter = class_label.split(':')[0].strip()
            if letter in response:
                predicted_class_idx = idx
                predicted_class = class_label
                break
        
        # Fallback: default to first class if parsing fails
        if predicted_class_idx == -1:
            predicted_class_idx = 0
            predicted_class = self.class_labels[0]
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'raw_response': response,
            'task': self.task_name,
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        max_new_tokens: int = 100,
    ) -> List[Dict]:
        """Make predictions on batch of images.
        
        Note: Currently processes images sequentially.
        For true batch processing, you'd need to modify the conversational
        format handling to support batches.
        """
        return [self.predict(img, max_new_tokens=max_new_tokens) for img in images]


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MedGemma inference')
    parser.add_argument(
        '--adapter',
        type=str,
        required=True,
        help='Path to LoRA adapter directory',
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file (DICOM, PNG, JPG)',
    )
    args = parser.parse_args()
    
    # Load model
    inference = MedGemmaInference(lora_adapter_path=args.adapter)
    
    # Run prediction
    result = inference.predict(args.image)
    
    print("\n" + "="*60)
    print("MedGemma Prediction")
    print("="*60)
    print(f"Task: {result['task']}")
    print(f"Predicted: {result['predicted_class']}")
    print(f"Raw response: {result['raw_response']}")
    print("="*60)
