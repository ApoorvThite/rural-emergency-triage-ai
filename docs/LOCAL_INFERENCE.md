# 🔮 Local Inference Guide

This guide shows you how to use your Colab-trained MedGemma models locally for inference.

---

## 📥 Step 1: Download Trained Model from Google Drive

After training completes in Colab, your model is saved to Google Drive.

### Option A: Download via Web Interface

1. Go to [Google Drive](https://drive.google.com/drive/my-drive)
2. Navigate to `rural_triage_ai/models/hemorrhage/` (or your task name)
3. You should see these files:
   ```
   hemorrhage/
   ├── adapter_config.json
   ├── adapter_model.safetensors
   ├── preprocessor_config.json
   ├── special_tokens_map.json
   ├── tokenizer_config.json
   ├── tokenizer.json
   ├── task_config.json
   ├── training_metrics.json
   └── confusion_matrix.png
   ```
4. Right-click the `hemorrhage` folder → **Download**
5. Extract the ZIP file to your project:
   ```bash
   cd rural-emergency-triage-ai
   mkdir -p data/models
   unzip ~/Downloads/hemorrhage.zip -d data/models/
   ```

### Option B: Download via Command Line (with rclone)

```bash
# Install rclone
brew install rclone  # macOS
# or: sudo apt install rclone  # Linux

# Configure Google Drive
rclone config

# Sync the model folder
rclone copy "gdrive:rural_triage_ai/models/hemorrhage" \
  ./data/models/hemorrhage -P
```

---

## 🚀 Step 2: Run Inference

### Single Image Prediction

```python
from src.models.medgemma.inference import MedGemmaInference

# Load your fine-tuned model
model = MedGemmaInference(
    lora_adapter_path="./data/models/hemorrhage"
)

# Run inference on a CT scan
result = model.predict("./data/test_cases/ct_scan_001.dcm")

print(f"Task: {result['task']}")
print(f"Predicted: {result['predicted_class']}")
print(f"Raw response: {result['raw_response']}")
```

**Output**:
```
Task: hemorrhage_detection
Predicted: B: Epidural hemorrhage
Raw response: B: Epidural hemorrhage
```

### Command-Line Interface

```bash
python -m src.models.medgemma.inference \
  --adapter ./data/models/hemorrhage \
  --image ./data/test_cases/ct_scan_001.dcm
```

**Output**:
```
Loading MedGemma model: google/medgemma-4b-it
Task: hemorrhage_detection
Classes: 6
✓ Model loaded successfully!

============================================================
MedGemma Prediction
============================================================
Task: hemorrhage_detection
Predicted: B: Epidural hemorrhage
Raw response: B: Epidural hemorrhage
============================================================
```

---

## 📊 Step 3: Batch Processing

### Process Multiple Images

```python
import glob
from src.models.medgemma.inference import MedGemmaInference

# Load model once
model = MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")

# Get all DICOM files
dicom_files = glob.glob("./data/test_cases/*.dcm")

# Batch inference
results = model.predict_batch(dicom_files)

# Display results
for img_path, result in zip(dicom_files, results):
    print(f"{img_path}: {result['predicted_class']}")
```

### Save Results to CSV

```python
import pandas as pd
from pathlib import Path

# Run batch inference
results = model.predict_batch(dicom_files)

# Create DataFrame
df = pd.DataFrame([
    {
        "image_path": Path(img).name,
        "predicted_class": res["predicted_class"],
        "predicted_idx": res["predicted_class_idx"],
        "raw_response": res["raw_response"],
    }
    for img, res in zip(dicom_files, results)
])

# Save to CSV
df.to_csv("./results/predictions.csv", index=False)
print(f"Saved {len(df)} predictions to results/predictions.csv")
```

---

## 🔄 Step 4: Using Different Models

### Switch Between Tasks

```python
# Hemorrhage detection
hemorrhage_model = MedGemmaInference(
    lora_adapter_path="./data/models/hemorrhage"
)

# Pneumothorax detection
pneumothorax_model = MedGemmaInference(
    lora_adapter_path="./data/models/pneumothorax"
)

# Use appropriate model for each image type
ct_result = hemorrhage_model.predict("ct_scan.dcm")
xray_result = pneumothorax_model.predict("chest_xray.png")
```

### Load from Custom Path

```python
# If you saved model elsewhere
model = MedGemmaInference(
    lora_adapter_path="/path/to/custom/model"
)
```

---

## 🎨 Step 5: Visualize Predictions

### Display Image with Prediction

```python
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import numpy as np

def visualize_prediction(image_path: str, model: MedGemmaInference):
    """Display image with prediction overlay."""
    
    # Load image
    if image_path.endswith('.dcm'):
        dcm = pydicom.dcmread(image_path)
        arr = dcm.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.open(image_path)
    
    # Get prediction
    result = model.predict(image_path)
    
    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(
        f"Prediction: {result['predicted_class']}\n"
        f"Raw: {result['raw_response']}",
        fontsize=14,
        pad=20
    )
    plt.tight_layout()
    plt.savefig(f"./results/{Path(image_path).stem}_prediction.png", dpi=150)
    plt.show()

# Use it
model = MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")
visualize_prediction("./data/test_cases/ct_001.dcm", model)
```

### Create Prediction Report

```python
from datetime import datetime

def generate_report(image_paths: list, model: MedGemmaInference, output_path: str):
    """Generate HTML report with all predictions."""
    
    results = model.predict_batch(image_paths)
    
    html = f"""
    <html>
    <head>
        <title>MedGemma Predictions Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .prediction {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
            .positive {{ background-color: #ffe6e6; }}
            .negative {{ background-color: #e6ffe6; }}
        </style>
    </head>
    <body>
        <h1>MedGemma Predictions Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Model: {model.task_name}</p>
        <p>Total images: {len(image_paths)}</p>
        <hr>
    """
    
    for img_path, result in zip(image_paths, results):
        is_positive = result['predicted_class_idx'] > 0
        css_class = "positive" if is_positive else "negative"
        
        html += f"""
        <div class="prediction {css_class}">
            <h3>{Path(img_path).name}</h3>
            <p><strong>Prediction:</strong> {result['predicted_class']}</p>
            <p><strong>Raw response:</strong> {result['raw_response']}</p>
        </div>
        """
    
    html += "</body></html>"
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Report saved to {output_path}")

# Generate report
model = MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")
dicom_files = glob.glob("./data/test_cases/*.dcm")
generate_report(dicom_files, model, "./results/predictions_report.html")
```

---

## ⚡ Performance Optimization

### 1. Batch Processing for Speed

```python
# SLOW: Process one at a time
for img in images:
    result = model.predict(img)

# FASTER: Use batch processing
results = model.predict_batch(images)
```

### 2. Reduce Max Tokens for Faster Inference

```python
# Default: max_new_tokens=100
result = model.predict(image_path, max_new_tokens=20)
# Classification only needs ~10 tokens for "B: Epidural hemorrhage"
```

### 3. Use GPU if Available

The inference script automatically uses GPU if available:
```python
# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 4. Load Model Once, Reuse Many Times

```python
# SLOW: Load model for each prediction
for img in images:
    model = MedGemmaInference(...)  # DON'T DO THIS
    result = model.predict(img)

# FAST: Load once, predict many
model = MedGemmaInference(...)
for img in images:
    result = model.predict(img)
```

---

## 🔍 Troubleshooting

### "task_config.json not found"

**Cause**: You're pointing to the wrong directory or the model wasn't saved correctly.

**Solution**: Check that your adapter path contains `task_config.json`:
```bash
ls -la ./data/models/hemorrhage/
# Should show: task_config.json, adapter_model.safetensors, etc.
```

### "CUDA out of memory"

**Cause**: GPU doesn't have enough memory for the model.

**Solution 1**: The model automatically uses 4-bit quantization, but if still OOM:
```python
# Reduce image resolution before inference
from PIL import Image
img = Image.open(image_path).resize((256, 256))
```

**Solution 2**: Use CPU (slower but works):
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
```

### Predictions are always the same class

**Cause**: Model wasn't trained properly or overfitted to one class.

**Solution**: 
1. Check training metrics in `training_metrics.json`
2. Retrain with more diverse data
3. Increase training epochs
4. Check class balance in training data

### Slow inference on CPU

**Expected**: CPU inference is 10-50x slower than GPU.

**Solutions**:
1. Use a machine with GPU
2. Export to ONNX for faster CPU inference (advanced)
3. Reduce `max_new_tokens` to minimum needed
4. Process images in smaller batches

---

## 📈 Benchmarking Inference Speed

```python
import time
from src.models.medgemma.inference import MedGemmaInference

model = MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")

# Warm-up (first inference is slower due to model loading)
_ = model.predict("./data/test_cases/ct_001.dcm")

# Benchmark
test_images = glob.glob("./data/test_cases/*.dcm")[:10]
start = time.time()
results = model.predict_batch(test_images)
elapsed = time.time() - start

print(f"Processed {len(test_images)} images in {elapsed:.2f}s")
print(f"Average: {elapsed/len(test_images):.2f}s per image")
```

**Expected performance**:
- **GPU (A100)**: 0.5-1s per image
- **GPU (T4)**: 1-2s per image
- **CPU**: 10-30s per image

---

## 🎯 Integration Examples

### FastAPI Endpoint

```python
from fastapi import FastAPI, File, UploadFile
from src.models.medgemma.inference import MedGemmaInference
import tempfile

app = FastAPI()

# Load model at startup
model = MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict hemorrhage type from uploaded CT scan."""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Run inference
    result = model.predict(tmp_path)
    
    # Clean up
    os.unlink(tmp_path)
    
    return {
        "task": result["task"],
        "predicted_class": result["predicted_class"],
        "predicted_idx": result["predicted_class_idx"],
        "raw_response": result["raw_response"],
    }
```

### Streamlit App

```python
import streamlit as st
from src.models.medgemma.inference import MedGemmaInference
from PIL import Image

st.title("🏥 MedGemma Hemorrhage Detection")

# Load model (cached)
@st.cache_resource
def load_model():
    return MedGemmaInference(lora_adapter_path="./data/models/hemorrhage")

model = load_model()

# File upload
uploaded_file = st.file_uploader("Upload CT scan (DICOM or PNG)", type=["dcm", "png", "jpg"])

if uploaded_file:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Display image
    st.image(tmp_path, caption="Uploaded CT scan", use_column_width=True)
    
    # Predict
    with st.spinner("Analyzing..."):
        result = model.predict(tmp_path)
    
    # Display result
    st.success(f"**Prediction:** {result['predicted_class']}")
    st.info(f"**Raw response:** {result['raw_response']}")
    
    os.unlink(tmp_path)
```

---

## 📚 Next Steps

1. **Integrate into your application**:
   - Add to FastAPI backend (`src/api/`)
   - Build UI with React Native
   - Deploy to tablet for offline use

2. **Optimize for production**:
   - Export to ONNX for faster inference
   - Add caching for repeated predictions
   - Implement batch processing API

3. **Monitor performance**:
   - Log predictions and confidence
   - Track inference latency
   - Collect feedback for model improvement

---

**You're now ready to use your trained MedGemma models locally!** 🎉
