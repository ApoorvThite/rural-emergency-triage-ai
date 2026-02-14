# 🚀 Google Colab Quick-Start Guide

This guide will help you train MedGemma models for emergency radiology triage using **free Google Colab GPUs**.

---

## 📋 Prerequisites

Before you start, you need:

1. **Google Account** (for Colab access)
2. **HuggingFace Account** with MedGemma access
3. **Kaggle Account** for dataset downloads

### Step 1: Get MedGemma Access

1. Go to [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
2. Click **"Agree and access repository"**
3. Wait for approval (usually instant, sometimes takes 1-2 days)
4. Once approved, go to [HuggingFace Settings → Tokens](https://huggingface.co/settings/tokens)
5. Create a new token with **READ** access
6. **Save this token** — you'll need it in Colab

### Step 2: Get Kaggle API Key

1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Scroll to **API** section
3. Click **"Create New API Token"**
4. Download `kaggle.json` file
5. **Keep this file** — you'll upload it to Colab

---

## 🎯 Training Your First Model

### Open the Notebook

1. Go to the notebook:
   - **GitHub**: [Colab_Training_Complete.ipynb](../notebooks/Colab_Training_Complete.ipynb)
   - **Direct Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/rural-emergency-triage-ai/blob/main/notebooks/Colab_Training_Complete.ipynb)

2. **Select GPU Runtime**:
   - Click **Runtime** → **Change runtime type**
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (free) or **A100** (Colab Pro)
   - Click **Save**

### Run the Notebook

**Just run all cells in order!** The notebook is fully self-contained.

#### Cell-by-Cell Overview:

1. **Environment Setup** (2-3 min)
   - Checks GPU availability
   - Installs all dependencies

2. **Authentication** (1 min)
   - **HuggingFace Token**: 
     - Click 🔑 **Secrets** in left sidebar
     - Add secret: `HF_TOKEN` = your HuggingFace token
   - **Kaggle API**:
     - Upload your `kaggle.json` when prompted

3. **Download Dataset** (5-10 min)
   - Downloads ~500MB head CT dataset
   - Or configure for full RSNA dataset (100GB+)

4. **Prepare Data** (2-5 min)
   - Converts images to conversational format
   - Creates train/validation splits

5. **Load MedGemma** (3-5 min)
   - Downloads MedGemma 4B model
   - Applies QLoRA (4-bit quantization)

6. **Train!** (1-8 hours depending on GPU)
   - **T4 GPU** (free): ~4-8 hours for 2000 images
   - **A100 GPU** (Colab Pro): ~1-3 hours for 2000 images
   - You can **close your laptop** — training runs in the cloud!

7. **Evaluate** (5-10 min)
   - Runs inference on validation set
   - Shows classification metrics & confusion matrix

8. **Save Model** (1-2 min)
   - Saves LoRA adapter to Google Drive
   - You can download it later!

---

## ⚙️ Configuration Options

### Change the Task

In **Cell 10** (data preparation), change the `TASK` variable:

```python
# For hemorrhage detection (CT scans)
TASK = "hemorrhage"

# For pneumothorax detection (chest X-rays)
TASK = "pneumothorax"
```

### Use Full Dataset

To train on the complete RSNA dataset (~100K images):

1. In **Cell 8**, uncomment the RSNA download section:
   ```python
   # Uncomment these lines:
   !kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p {DATA_DIR}/rsna -f stage_2_train.csv
   !kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p {DATA_DIR}/rsna -f stage_2_train_images.zip
   ```

2. In **Cell 10**, increase dataset size:
   ```python
   MAX_TRAIN = 50000  # Instead of 2000
   MAX_VAL = 5000     # Instead of 200
   ```

3. **Note**: This will take much longer (~12-24 hours on T4)

### Adjust Training Duration

In **Cell 10**, modify:

```python
# Quick test (30 min on T4)
MAX_TRAIN = 500
MAX_VAL = 50

# Medium training (2-4 hours on T4)
MAX_TRAIN = 2000
MAX_VAL = 200

# Full training (12-24 hours on T4)
MAX_TRAIN = 50000
MAX_VAL = 5000
```

---

## 📊 Monitoring Training

### TensorBoard (Real-time Metrics)

While training is running, open a new cell and run:

```python
%load_ext tensorboard
%tensorboard --logdir medgemma-4b-it-hemorrhage
```

This shows:
- Training loss over time
- Validation loss
- Learning rate schedule

### Check Progress

The notebook prints progress every 10 steps:
```
🚀 Starting training...
   Task: hemorrhage
   Model: google/medgemma-4b-it
   Train samples: 1800
   Val samples: 200

[562/562 2:56:02, Epoch 0/1]
Step    Training Loss    Validation Loss
50      4.520300        0.032325
100     0.103300        0.026194
...
```

---

## 💾 Accessing Your Trained Model

After training completes, your model is saved to Google Drive:

```
/content/drive/MyDrive/rural_triage_ai/models/hemorrhage/
├── adapter_config.json
├── adapter_model.safetensors
├── task_config.json
├── training_metrics.json
└── confusion_matrix.png
```

### Download the Model

1. Go to [Google Drive](https://drive.google.com/drive/my-drive)
2. Navigate to `rural_triage_ai/models/hemorrhage/`
3. Right-click folder → **Download**
4. Extract the ZIP file

### Use the Model Locally

```python
from src.models.medgemma.inference import MedGemmaInference

# Load your fine-tuned model
model = MedGemmaInference(
    lora_adapter_path="./models/hemorrhage"
)

# Run inference
result = model.predict("path/to/ct_scan.dcm")
print(result['predicted_class'])
```

---

## 🐛 Troubleshooting

### "GPU not available"

**Solution**: Make sure you selected GPU runtime:
- Runtime → Change runtime type → GPU → T4

### "HF_TOKEN not found"

**Solution**: Add your HuggingFace token to Colab Secrets:
1. Click 🔑 **Secrets** (left sidebar)
2. Add new secret: `HF_TOKEN`
3. Paste your token
4. Toggle **notebook access** ON

### "You must accept the competition rules"

**Solution**: For RSNA dataset, you need to:
1. Go to [RSNA Competition](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
2. Click **"I Understand and Accept"**
3. Then re-run the download cell

### "Out of memory" error

**Solutions**:
1. Reduce batch size in Cell 14:
   ```python
   batch_size = 1
   grad_accum = 16
   ```

2. Use fewer training samples in Cell 10:
   ```python
   MAX_TRAIN = 1000
   MAX_VAL = 100
   ```

3. Upgrade to Colab Pro for A100 GPU (40GB memory)

### Training is too slow

**Solutions**:
1. **Use smaller dataset** for testing (500-1000 images)
2. **Upgrade to Colab Pro** for A100 GPU (4-8x faster)
3. **Train overnight** — Colab sessions last up to 12 hours

### Colab disconnected

**Don't worry!** Your model is saved to Google Drive every epoch.

To resume:
1. Re-run all cells up to training
2. The trainer will automatically resume from the last checkpoint

---

## 💡 Tips & Best Practices

### 1. Start Small, Then Scale

```python
# Day 1: Quick test (30 min)
MAX_TRAIN = 500

# Day 2: Medium run (2-4 hours)
MAX_TRAIN = 2000

# Day 3: Full training (overnight)
MAX_TRAIN = 50000
```

### 2. Use Colab Pro for Serious Training

| Feature | Free Colab | Colab Pro ($10/mo) |
|---------|------------|-------------------|
| GPU | T4 (16GB) | A100 (40GB) |
| Training Speed | 1x | 4-8x faster |
| Session Length | ~12 hours | ~24 hours |
| Queue Priority | Low | High |

**Worth it for full dataset training!**

### 3. Save Intermediate Results

Add this cell after training to save checkpoints:

```python
# Save every 500 steps
training_args.save_steps = 500
training_args.save_total_limit = 3  # Keep last 3 checkpoints
```

### 4. Monitor GPU Usage

```python
!nvidia-smi
```

Shows:
- GPU memory usage
- GPU utilization %
- Temperature

### 5. Keep Your Session Alive

Colab disconnects after ~90 min of inactivity. To prevent:
1. Open browser console (F12)
2. Paste this JavaScript:
   ```javascript
   function KeepClicking(){
     console.log("Clicking");
     document.querySelector("colab-connect-button").click();
   }
   setInterval(KeepClicking, 60000);
   ```

---

## 📈 Expected Results

After training on **2000 images for 1 epoch**:

### Hemorrhage Detection (6 classes)
- **Accuracy**: 60-75%
- **Training time**: 2-4 hours (T4), 1 hour (A100)

### Pneumothorax Detection (binary)
- **Accuracy**: 75-85%
- **Training time**: 1-2 hours (T4), 30 min (A100)

**For better results**: Train on full dataset (50K+ images) for 3 epochs.

---

## 🎓 Next Steps

After training your first model:

1. **Evaluate on test set** — Use the evaluation cells
2. **Try different tasks** — Pneumothorax, fracture detection
3. **Integrate into API** — See `src/api/` for FastAPI backend
4. **Deploy to tablet** — See `docs/DEPLOYMENT.md`
5. **Submit to Kaggle** — MedGemma Impact Challenge!

---

## 📚 Additional Resources

- **Official MedGemma Docs**: https://github.com/google-health/medgemma
- **HuggingFace TRL**: https://huggingface.co/docs/trl
- **PEFT/LoRA**: https://huggingface.co/docs/peft
- **Project README**: [../README.md](../README.md)
- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

## ❓ Need Help?

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/rural-emergency-triage-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/rural-emergency-triage-ai/discussions)
- **Email**: your.email@example.com

---

**Happy Training! 🚀**
