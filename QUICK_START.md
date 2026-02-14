# ⚡ Quick Start - MedGemma Training

**Get from zero to trained model in 3 steps.**

---

## 🚀 Step 1: Train in Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ApoorvThite/rural-emergency-triage-ai/blob/master/notebooks/Colab_Training_Complete.ipynb)

### Quick Test (30 min)
```python
# Cell 10 - Keep defaults
MAX_TRAIN = 2000
MAX_VAL = 200

# Cell 14 - Keep defaults
num_train_epochs = 1
```

### Full Training (12-18 hours on T4, 3-4 hours on A100)
```python
# Cell 10 - Scale up dataset
MAX_TRAIN = 50000
MAX_VAL = 5000

# Cell 14 - More epochs
num_train_epochs = 3
```

**Expected Results**:
- Quick test: 60-75% accuracy
- Full training: 80-90% accuracy

---

## 🔄 Step 2: Train Different Tasks

### Hemorrhage Detection (Default)
```python
TASK = "hemorrhage"  # CT head scans, 6 classes
```

### Pneumothorax Detection
```python
TASK = "pneumothorax"  # Chest X-rays, binary
```

**Don't forget**: Update Cell 8 to download the appropriate dataset!

---

## 💾 Step 3: Use Model Locally

### Download from Google Drive
1. Go to `rural_triage_ai/models/hemorrhage/` in Google Drive
2. Download folder
3. Extract to `./data/models/hemorrhage/`

### Run Inference
```python
from src.models.medgemma.inference import MedGemmaInference

model = MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")
result = model.predict("ct_scan.dcm")

print(result['predicted_class'])
# Output: "B: Epidural hemorrhage"
```

### Command Line
```bash
python -m src.models.medgemma.inference \
  --adapter ./data/models/hemorrhage \
  --image ct_scan.dcm
```

---

## 📚 Full Documentation

- **Colab Guide**: [docs/COLAB_QUICKSTART.md](docs/COLAB_QUICKSTART.md)
- **Scaling Training**: [docs/SCALING_TRAINING.md](docs/SCALING_TRAINING.md)
- **Local Inference**: [docs/LOCAL_INFERENCE.md](docs/LOCAL_INFERENCE.md)

---

## 🎯 Training Configurations Cheat Sheet

| Config | Samples | Epochs | Time (T4) | Time (A100) | Accuracy |
|--------|---------|--------|-----------|-------------|----------|
| **Quick Test** | 500 | 1 | 30 min | 10 min | 40-60% |
| **Baseline** | 2,000 | 2 | 3 hours | 1 hour | 60-75% |
| **Production** | 50,000 | 3 | 18 hours | 4 hours | 80-90% |
| **Competition** | 100,000 | 5 | 48 hours | 12 hours | 85-95% |

---

## ⚙️ Prerequisites

### For Colab Training
- ✅ Google Account
- ✅ HuggingFace account with MedGemma access
- ✅ Kaggle API key

### For Local Inference
- ✅ Python 3.10+
- ✅ GPU recommended (but CPU works)
- ✅ Trained model from Colab

---

## 🐛 Common Issues

### "HF_TOKEN not found"
**Fix**: Add to Colab Secrets (🔑 icon in sidebar)

### "Out of memory"
**Fix**: Reduce batch size in Cell 14:
```python
batch_size = 1
grad_accum = 32
```

### "Notebook not found" in Colab
**Fix**: Links are updated! Use the badge above.

---

**Ready? Click the Colab badge and start training!** 🚀
