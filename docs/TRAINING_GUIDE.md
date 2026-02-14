# Training Guide

## Prerequisites

1. **Download Datasets**
   - RSNA Hemorrhage: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection
   - SIIM Pneumothorax: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

2. **Place data in correct directories:**
```
   data/raw/rsna_hemorrhage/
   data/raw/siim_pneumothorax/
```

## Training

### Hemorrhage Detection
```bash
# Using script
./scripts/train_hemorrhage.sh

# Or directly
python src/models/medgemma/train.py --config configs/hemorrhage_detection.yaml
```

### Pneumothorax Detection
```bash
# Using script
./scripts/train_pneumothorax.sh

# Or directly
python src/models/medgemma/train.py --config configs/pneumothorax_detection.yaml
```

## Monitoring

Training progress is logged to console. To enable Weights & Biases logging:

1. Update config file:
```yaml
   wandb:
     enabled: true
     entity: "your-username"
```

2. Login to wandb:
```bash
   wandb login
```

## Inference
```python
from src.models.medgemma.inference import MedGemmaInference

# Load model
model = MedGemmaInference(
    model_path='./data/models/hemorrhage_detection/best_model.pt',
    config_path='./configs/hemorrhage_detection.yaml',
)

# Make prediction
result = model.predict('path/to/ct_scan.dcm')
print(result)
```

## Expected Training Time

- **Hemorrhage Detection**: ~6-8 hours on 1x A100 GPU
- **Pneumothorax Detection**: ~4-6 hours on 1x A100 GPU

## Model Outputs

Trained models are saved to:
- `data/models/{task_name}/best_model.pt` - Best validation checkpoint
- `data/models/{task_name}/final_model.pt` - Final model after all epochs