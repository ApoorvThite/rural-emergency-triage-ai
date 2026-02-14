"""
MedGemma fine-tuning script using official Google approach:
- AutoModelForImageTextToText (not AutoModelForCausalLM)
- SFTTrainer from HuggingFace TRL
- Conversational format (image + text prompt → answer)
- QLoRA (4-bit quantization + LoRA adapters)

Based on: https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
"""

import os
import yaml
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import Dataset, DatasetDict
from PIL import Image
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.data_loader import (
    prepare_rsna_hemorrhage_data,
    prepare_siim_pneumothorax_data,
)
from src.utils.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
)


def prepare_conversational_dataset(
    image_paths: List[str],
    labels: List[int],
    class_labels: List[str],
    prompt: str,
    train_split: float = 0.9,
) -> DatasetDict:
    """Convert image classification data to MedGemma conversational format.
    
    Args:
        image_paths: List of paths to images
        labels: List of integer labels
        class_labels: List of class names (e.g., ["A: No hemorrhage", "B: Epidural", ...])
        prompt: Task prompt to use
        train_split: Fraction for training set
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    def load_and_format(img_path: str, label: int) -> Dict[str, Any]:
        """Load image and format as conversation."""
        try:
            # Load image
            if img_path.endswith('.dcm'):
                import pydicom
                dcm = pydicom.dcmread(img_path)
                arr = dcm.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                image = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
            else:
                image = Image.open(img_path).convert('RGB')
            
            # Format as conversation
            return {
                'image': image,
                'label': label,
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': prompt},
                        ],
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': class_labels[label]},
                        ],
                    },
                ],
            }
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    # Load and format all examples
    examples = []
    for img_path, label in zip(image_paths, labels):
        result = load_and_format(img_path, label)
        if result is not None:
            examples.append(result)
    
    # Split into train/val
    split_idx = int(len(examples) * train_split)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]
    
    return DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
    })


def create_data_collator(processor: AutoProcessor) -> callable:
    """Create custom data collator for multimodal inputs.
    
    Args:
        processor: MedGemma processor
    
    Returns:
        Collator function
    """
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process examples with text + images."""
        texts = []
        images = []
        
        for example in examples:
            images.append([example['image'].convert('RGB')])
            texts.append(
                processor.apply_chat_template(
                    example['messages'],
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
        
        # Tokenize and process
        batch = processor(text=texts, images=images, return_tensors='pt', padding=True)
        
        # Create labels: mask padding and image tokens
        labels = batch['input_ids'].clone()
        
        # Mask image tokens
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map.get('boi_token', '<image>')
        )
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if isinstance(image_token_id, int):
            labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100  # Additional image placeholder
        
        batch['labels'] = labels
        return batch
    
    return collate_fn


def load_medgemma_model(
    model_id: str,
    load_in_4bit: bool = True,
) -> tuple:
    """Load MedGemma model with QLoRA.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'google/medgemma-4b-it')
        load_in_4bit: Whether to use 4-bit quantization
    
    Returns:
        (model, processor)
    """
    # Determine dtype based on GPU capability
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        compute_dtype = torch.bfloat16
        print("Using bfloat16 (A100/H100 detected)")
    else:
        compute_dtype = torch.float16
        print("Using float16 (T4/V100 detected)")
    
    # QLoRA config
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=compute_dtype,
        )
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        attn_implementation='eager',
        torch_dtype=compute_dtype,
        device_map='auto',
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'right'
    
    return model, processor


def main(config_path: str):
    """Main training function using SFTTrainer."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded from: {config_path}")
    print(f"Task: {config['task']['name']}")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    print("\nPreparing dataset...")
    dataset_name = config['data']['dataset_name']
    
    if dataset_name == 'rsna_hemorrhage':
        df = prepare_rsna_hemorrhage_data(config['data']['data_dir'])
    elif dataset_name == 'siim_pneumothorax':
        df = prepare_siim_pneumothorax_data(config['data']['data_dir'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Extract image paths and labels
    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()
    
    # Get class labels and prompt from config
    class_labels = config['task']['class_labels']
    prompt = config['task']['prompt']
    
    print(f"Total samples: {len(image_paths)}")
    print(f"Classes: {len(class_labels)}")
    
    # Convert to conversational format
    data = prepare_conversational_dataset(
        image_paths=image_paths,
        labels=labels,
        class_labels=class_labels,
        prompt=prompt,
        train_split=config['data'].get('train_split', 0.9),
    )
    
    print(f"Train: {len(data['train'])}, Val: {len(data['validation'])}")
    
    # Load model
    print("\nLoading MedGemma model...")
    model, processor = load_medgemma_model(
        model_id=config['model']['name'],
        load_in_4bit=config['model'].get('load_in_4bit', True),
    )
    
    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        r=config['lora']['r'],
        bias=config['lora']['bias'],
        target_modules=config['lora']['target_modules'],
        task_type='CAUSAL_LM',
        modules_to_save=['lm_head', 'embed_tokens'],
    )
    
    # Training config
    training_config = config['training']
    
    # Adjust batch size based on GPU
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        batch_size = training_config.get('batch_size', 4)
        grad_accum = training_config.get('gradient_accumulation_steps', 4)
    else:
        batch_size = 1
        grad_accum = training_config.get('gradient_accumulation_steps', 16)
    
    # Determine dtype
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        optim='adamw_torch_fused',
        logging_steps=training_config.get('logging_steps', 10),
        save_strategy='epoch',
        eval_strategy='steps',
        eval_steps=training_config.get('eval_steps', 50),
        learning_rate=training_config['learning_rate'],
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type='linear',
        push_to_hub=False,
        report_to='tensorboard',
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataset_kwargs={'skip_prepare_dataset': True},
        remove_unused_columns=False,
        label_names=['labels'],
    )
    
    # Create trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['validation'],
        peft_config=peft_config,
        processing_class=processor,
        data_collator=create_data_collator(processor),
    )
    
    print(f"Effective batch size: {batch_size * grad_accum}")
    print(f"Training steps: ~{len(data['train']) // (batch_size * grad_accum)}")
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # Save metrics
    metrics = trainer.state.log_history
    with open(output_dir / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save task config
    task_config = {
        'model_id': config['model']['name'],
        'task': config['task']['name'],
        'class_labels': class_labels,
        'prompt': prompt,
        'lora_dir': str(output_dir),
    }
    with open(output_dir / 'task_config.json', 'w') as f:
        json.dump(task_config, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Metrics: {output_dir}/training_metrics.json")
    print(f"  Config: {output_dir}/task_config.json")


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
