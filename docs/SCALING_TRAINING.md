# 📈 Scaling Up MedGemma Training

This guide shows you how to scale from the quick demo (20 samples) to production-quality training.

---

## 🎯 Quick Reference: Training Configurations

### Configuration 1: Quick Test (30 min on T4)
```python
# In Colab Cell 10
MAX_TRAIN = 500
MAX_VAL = 50

# In Colab Cell 14
num_train_epochs = 1
```
**Use for**: Testing the pipeline, debugging, quick iterations

---

### Configuration 2: Medium Training (2-4 hours on T4)
```python
# In Colab Cell 10
MAX_TRAIN = 2000
MAX_VAL = 200

# In Colab Cell 14
num_train_epochs = 2
```
**Use for**: Initial model development, baseline performance

---

### Configuration 3: Full Training (12-24 hours on T4, 3-6 hours on A100)
```python
# In Colab Cell 10
MAX_TRAIN = 50000
MAX_VAL = 5000

# In Colab Cell 14
num_train_epochs = 3
```
**Use for**: Production models, Kaggle submission, best performance

---

## 📥 Downloading Full RSNA Dataset

The full RSNA Intracranial Hemorrhage Detection dataset contains ~100K CT scans (~100GB).

### Step 1: Accept Competition Rules
1. Go to https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection
2. Click **"I Understand and Accept"** under Rules
3. Wait for confirmation (usually instant)

### Step 2: Update Colab Cell 8

Replace the dataset download cell with:

```python
import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = "/content/data"
os.makedirs(DATA_DIR, exist_ok=True)

# Download full RSNA dataset
print("Downloading RSNA Intracranial Hemorrhage Detection dataset...")
print("This will take 30-60 minutes (~100GB)")

!kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p {DATA_DIR}/rsna
!cd {DATA_DIR}/rsna && unzip -q stage_2_train.csv.zip
!cd {DATA_DIR}/rsna && unzip -q stage_2_train_images.zip

print("\n✓ Dataset downloaded!")
!du -sh {DATA_DIR}/rsna
```

### Step 3: Update Data Preparation (Cell 10)

Replace the `find_images_and_labels` function with proper RSNA parsing:

```python
def load_rsna_data(data_dir: str) -> list[dict]:
    """Load RSNA hemorrhage dataset with proper labels."""
    csv_path = f"{data_dir}/rsna/stage_2_train.csv"
    df = pd.read_csv(csv_path)
    
    # Parse labels from CSV
    # Format: ID_subtype where subtype is one of:
    # any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural
    
    # Group by image ID
    image_labels = {}
    for _, row in df.iterrows():
        img_id = row['ID'].rsplit('_', 1)[0]
        subtype = row['ID'].rsplit('_', 1)[1]
        label_value = int(row['Label'])
        
        if img_id not in image_labels:
            image_labels[img_id] = {}
        image_labels[img_id][subtype] = label_value
    
    # Convert to examples with single dominant label
    examples = []
    for img_id, labels in image_labels.items():
        img_path = f"{data_dir}/rsna/stage_2_train_images/{img_id}.dcm"
        if not os.path.exists(img_path):
            continue
        
        # Determine dominant hemorrhage type
        if labels.get('epidural', 0) == 1:
            label_idx = 1
        elif labels.get('subdural', 0) == 1:
            label_idx = 2
        elif labels.get('subarachnoid', 0) == 1:
            label_idx = 3
        elif labels.get('intraventricular', 0) == 1:
            label_idx = 4
        elif labels.get('intraparenchymal', 0) == 1:
            label_idx = 5
        else:
            label_idx = 0  # No hemorrhage
        
        examples.append({
            "image_path": img_path,
            "label": label_idx,
        })
    
    return examples

# Use the RSNA loader
raw_examples = load_rsna_data(DATA_DIR)
print(f"\nTotal RSNA examples: {len(raw_examples)}")
```

---

## 🔄 Training Different Tasks

### Task 1: Hemorrhage Detection (CT Scans)

**Already configured!** This is the default in the notebook.

```python
TASK = "hemorrhage"
CLASS_LABELS = [
    "A: No hemorrhage",
    "B: Epidural hemorrhage",
    "C: Subdural hemorrhage",
    "D: Subarachnoid hemorrhage",
    "E: Intraventricular hemorrhage",
    "F: Intraparenchymal hemorrhage",
]
PROMPT = (
    "You are an emergency radiology AI assistant. "
    "Analyze this CT head scan and identify the most likely finding.\n"
    + "\n".join(CLASS_LABELS)
)
```

---

### Task 2: Pneumothorax Detection (Chest X-rays)

**Change Cell 10** to:

```python
TASK = "pneumothorax"
CLASS_LABELS = [
    "A: No pneumothorax",
    "B: Pneumothorax present",
]
PROMPT = (
    "You are an emergency radiology AI assistant. "
    "Analyze this chest X-ray and determine if pneumothorax is present.\n"
    + "\n".join(CLASS_LABELS)
)
DATA_DIR = "/content/data/siim_pneumothorax"
```

**Download SIIM dataset** (Cell 8):

```python
print("Downloading SIIM-ACR Pneumothorax dataset...")
!kaggle competitions download -c siim-acr-pneumothorax-segmentation -p {DATA_DIR}/siim_pneumothorax
!cd {DATA_DIR}/siim_pneumothorax && unzip -q "*.zip"
```

---

### Task 3: Fracture Detection (Custom)

**Create your own task**:

```python
TASK = "fracture"
CLASS_LABELS = [
    "A: No fracture",
    "B: Skull fracture",
    "C: Rib fracture",
    "D: Long bone fracture",
    "E: Vertebral fracture",
]
PROMPT = (
    "You are an emergency radiology AI assistant. "
    "Analyze this X-ray or CT scan and identify any fractures.\n"
    + "\n".join(CLASS_LABELS)
)
DATA_DIR = "/content/data/fracture_dataset"
```

---

## 💾 Using Trained Models Locally

After training in Colab, download your model from Google Drive and use it locally.

### Step 1: Download from Google Drive

1. Go to https://drive.google.com/drive/my-drive
2. Navigate to `rural_triage_ai/models/hemorrhage/`
3. Right-click → **Download**
4. Extract to your local project: `./data/models/hemorrhage/`

### Step 2: Run Local Inference

```python
from src.models.medgemma.inference import MedGemmaInference

# Load your fine-tuned model
model = MedGemmaInference(
    lora_adapter_path="./data/models/hemorrhage"
)

# Run inference on a CT scan
result = model.predict("path/to/ct_scan.dcm")

print(f"Predicted: {result['predicted_class']}")
print(f"Raw response: {result['raw_response']}")
```

**Command-line usage**:

```bash
python -m src.models.medgemma.inference \
  --adapter ./data/models/hemorrhage \
  --image ./data/sample_ct.dcm
```

### Step 3: Batch Processing

```python
import glob
from src.models.medgemma.inference import MedGemmaInference

model = MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")

# Process all DICOM files in a directory
dicom_files = glob.glob("./data/test_cases/*.dcm")
results = model.predict_batch(dicom_files)

for img_path, result in zip(dicom_files, results):
    print(f"{img_path}: {result['predicted_class']}")
```

---

## 📊 Expected Performance by Training Scale

### Quick Test (500 samples, 1 epoch)
- **Training time**: 30 min (T4), 10 min (A100)
- **Accuracy**: 40-60% (not production-ready)
- **Use case**: Pipeline testing only

### Medium Training (2000 samples, 2 epochs)
- **Training time**: 3 hours (T4), 1 hour (A100)
- **Accuracy**: 60-75% (baseline performance)
- **Use case**: Initial development, proof of concept

### Full Training (50K samples, 3 epochs)
- **Training time**: 18 hours (T4), 4 hours (A100)
- **Accuracy**: 80-90% (production-ready)
- **Use case**: Kaggle submission, deployment

### Full Training + Data Augmentation (100K samples, 5 epochs)
- **Training time**: 48 hours (T4), 12 hours (A100)
- **Accuracy**: 85-95% (state-of-the-art)
- **Use case**: Competition winning, clinical deployment

---

## ⚡ Optimization Tips

### 1. Use Colab Pro for Serious Training

| Feature | Free Colab | Colab Pro ($10/mo) |
|---------|------------|-------------------|
| GPU | T4 (16GB) | A100 (40GB) |
| Speed | 1x | 4-8x faster |
| Session | ~12 hours | ~24 hours |
| Priority | Low | High |

**ROI**: For full dataset training, Colab Pro saves 12+ hours → worth $10!

### 2. Enable Checkpointing

Add to Cell 14:

```python
training_args = SFTConfig(
    # ... existing args ...
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep last 3 checkpoints
)
```

If Colab disconnects, you can resume from the last checkpoint.

### 3. Use Mixed Precision Training

Already enabled! The notebook automatically uses:
- **bfloat16** on A100 (faster, more stable)
- **float16** on T4 (memory efficient)

### 4. Increase Batch Size on A100

If using A100, you can increase throughput:

```python
# In Cell 14, replace the batch size logic:
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    batch_size = 8  # Increased from 4
    grad_accum = 2  # Decreased from 4
else:
    batch_size = 1
    grad_accum = 16
```

### 5. Monitor Training with TensorBoard

Add a new cell after Cell 14:

```python
%load_ext tensorboard
%tensorboard --logdir {OUTPUT_DIR}
```

This shows real-time training/validation loss curves.

---

## 🔍 Troubleshooting Large-Scale Training

### "Out of memory" on full dataset

**Solution 1**: Reduce batch size
```python
batch_size = 1
grad_accum = 32  # Increase to maintain effective batch size
```

**Solution 2**: Use gradient checkpointing (already enabled)

**Solution 3**: Reduce image resolution in data loader:
```python
# In load_and_format function
image = image.resize((256, 256))  # Instead of 512x512
```

### Training is too slow

**Solution 1**: Use Colab Pro with A100

**Solution 2**: Reduce dataset size for initial experiments

**Solution 3**: Use fewer epochs:
```python
num_train_epochs = 1  # Instead of 3
```

### Colab keeps disconnecting

**Solution 1**: Keep browser tab active (use the JavaScript trick from COLAB_QUICKSTART.md)

**Solution 2**: Enable checkpointing (see Optimization Tips #2)

**Solution 3**: Use Colab Pro for longer sessions (24 hours vs 12 hours)

---

## 📈 Progressive Training Strategy

**Week 1: Prototype** (500 samples)
- Verify pipeline works end-to-end
- Test different prompts and class labels
- Debug any data loading issues

**Week 2: Baseline** (2000 samples, 2 epochs)
- Establish baseline performance
- Tune hyperparameters (learning rate, LoRA rank)
- Validate on held-out test set

**Week 3: Scale Up** (20K samples, 3 epochs)
- Train on larger subset
- Monitor for overfitting
- Compare with baseline

**Week 4: Full Training** (100K samples, 3-5 epochs)
- Train production model
- Extensive evaluation
- Prepare for deployment/submission

---

## 🎓 Next Steps After Training

1. **Evaluate thoroughly**:
   - Test on held-out test set
   - Calculate confidence intervals
   - Analyze failure cases

2. **Integrate into API**:
   - Use `inference.py` in FastAPI backend
   - Add batch processing endpoint
   - Implement caching for speed

3. **Deploy to tablet**:
   - Export model to ONNX for faster inference
   - Build React Native UI
   - Test offline functionality

4. **Submit to Kaggle**:
   - Generate predictions on test set
   - Format submission CSV
   - Upload to competition

---

## 📚 Additional Resources

- **RSNA Competition**: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection
- **SIIM Competition**: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
- **MedGemma Docs**: https://github.com/google-health/medgemma
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **SFT Training**: https://huggingface.co/docs/trl/sft_trainer

---

**Ready to scale up? Update the configuration in Cell 10 and Cell 14, then re-run the notebook!** 🚀
