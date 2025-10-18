# Multi-Approach Swin Transformer Training Guide

## 🎯 Overview

This training pipeline handles **5 different CWT preprocessing approaches** and trains **3 model types** for each:

### 5 Approaches (Input Representations)
1. **single_lead**: Single lead (Lead II) - (3, 224, 224) via replication
2. **3_channel**: 3 representative leads (I, II, V3) - (3, 224, 224) native
3. **12_channel**: All 12 leads - (12, 224, 224) preserves all info ⭐
4. **concatenated**: All leads concatenated horizontally - (3, 224, 224)
5. **weighted**: Clinically-weighted average - (3, 224, 224)

### 3 Model Types Per Approach
- **Scalogram Model**: Trained on scalograms (power representation)
- **Phasogram Model**: Trained on phasograms (phase representation)
- **Fusion Model**: Dual-stream combining both scalograms & phasograms

**Total**: 5 approaches × 3 models = **15 models**

---

## 🚀 Quick Start

### Option 1: Train Single Approach (Fastest)
```python
# Train only 12-channel approach (recommended for best performance)
python complete_training_script.py
```

This trains 3 models:
- `swin_scalogram_12_channel.pth`
- `swin_phasogram_12_channel.pth`
- `dual_swin_fusion_12_channel.pth`

### Option 2: Train All Approaches (Comprehensive)
Uncomment the "OPTION 2" section in the script to train all 15 models.

### Option 3: Custom Comparison
Uncomment "OPTION 3" and modify `comparison_approaches` list.

---

## 📊 Model Architecture Details

### Input Shape Handling

| Approach | Input Shape | Model Modification |
|----------|-------------|-------------------|
| single_lead | (224, 224) | Replicate to (3, 224, 224) |
| 3_channel | (224, 224, 3) | Transpose to (3, 224, 224) |
| **12_channel** | **(12, 224, 224)** | **Modify first conv: 3→12 channels** |
| concatenated | (224, 224) | Replicate to (3, 224, 224) |
| weighted | (224, 224) | Replicate to (3, 224, 224) |

### Key Model Modifications

**For 12-channel approach:**
```python
# First conv layer modified from (3, 96, 4, 4) to (12, 96, 4, 4)
# Pretrained weights averaged across new channels
new_weight = pretrained_weight.repeat(1, 4, 1, 1) / 4.0
```

**Dual-Stream Fusion:**
```python
# Two Swin backbones (one for scalogram, one for phasogram)
# Features concatenated: 768 + 768 = 1536 dims
# Fusion head reduces to num_classes
```

---

## 🔧 Configuration

### Key Hyperparameters
```python
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4      # Single models
LEARNING_RATE = 5e-5      # Fusion models (lower)
WEIGHT_DECAY = 0.05
PATIENCE = 10
```

### Training Strategy
- **Optimizer**: AdamW with weight decay
- **Loss**: BCEWithLogitsLoss with class weights (handles imbalance)
- **LR Schedule**: 
  - 3 epochs warmup (0.1× → 1.0×)
  - Cosine annealing (1.0× → 0.1×)
- **Early Stopping**: Patience of 10 epochs based on validation F1
- **Gradient Clipping**: Max norm = 1.0

---

## 📈 Expected Performance

Based on CWT literature and ECG classification benchmarks:

| Approach | Expected F1 Macro | Notes |
|----------|------------------|-------|
| single_lead | 0.75 - 0.80 | Baseline, fast |
| 3_channel | 0.78 - 0.83 | Better than single |
| **12_channel** | **0.82 - 0.88** | **Best - all info preserved** ⭐ |
| concatenated | 0.77 - 0.82 | Spatial arrangement |
| weighted | 0.76 - 0.81 | Clinical prior |

**Fusion models typically add +2-4% F1 over single representation**

---

## 💾 Model Loading

### Load Trained Model
```python
# For single-representation models
model = SwinSmall(num_classes=5, num_input_channels=12, pretrained=False)
model.load_state_dict(torch.load('swin_scalogram_12_channel.pth'))
model.eval()

# For fusion models
model = DualStreamSwinFusion(num_classes=5, num_input_channels=12, pretrained=False)
model.load_state_dict(torch.load('dual_swin_fusion_12_channel.pth'))
model.eval()
```

### Inference
```python
# Single input
with torch.no_grad():
    output = model(scalogram_tensor)
    probs = torch.sigmoid(output)

# Dual input (fusion)
with torch.no_grad():
    output = model(scalogram_tensor, phasogram_tensor)
    probs = torch.sigmoid(output)
```

---

## 🎨 Customization

### Train Different Approach
```python
results = train_approach(
    approach='3_channel',  # Change this
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    mlb=mlb,
    cwt_base_path='/path/to/cwt/representations',
    device='cuda'
)
```

### Adjust Hyperparameters
Modify the global constants at the top of the script:
```python
BATCH_SIZE = 32           # Increase if GPU memory allows
NUM_EPOCHS = 50           # More epochs for better convergence
LEARNING_RATE = 5e-5      # Lower LR for fine-tuning
PATIENCE = 15             # More patience for stable training
```

### Use Different Swin Variant
```python
# In SwinSmall class, change:
from torchvision.models import swin_b, Swin_B_Weights

self.backbone = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
# Swin-Base: ~88M parameters, better accuracy
```

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 8

# Use gradient accumulation
accumulation_steps = 2  # Effective batch size = 8 × 2 = 16

# Enable mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Slow Training
```python
# Increase num_workers
NUM_WORKERS = 8

# Use pin_memory
DataLoader(..., pin_memory=True)

# Use compiled model (PyTorch 2.0+)
model = torch.compile(model)
```

### Poor Performance
1. **Check data loading**: Verify CWT files exist and are correctly normalized
2. **Increase epochs**: Try NUM_EPOCHS = 50
3. **Lower learning rate**: Try LEARNING_RATE = 5e-5
4. **Use 12-channel approach**: Best information preservation

---

## 📊 Results Analysis

### View Training Progress
```python
# Training logs show per-epoch metrics
# Monitor validation F1 for early stopping

Epoch 15/30 | Loss: 0.2145 | Val F1: 0.8234 | AUC: 0.9012
  ✓ New best F1: 0.8234
```

### Compare Models
```python
# Script automatically prints comparison table
# Shows F1, AUC, Hamming accuracy for all models

Approach             | Model           | F1 Macro   | AUC Macro  
------------------------------------------------------------------
12_channel          | scalogram       | 0.8456     | 0.9234
12_channel          | phasogram       | 0.8312     | 0.9156
12_channel          | fusion          | 0.8678     | 0.9345  ⭐
```

### Best Practices
1. **Always start with 12_channel** - best performance
2. **Compare fusion vs single** - fusion usually better
3. **Check per-class F1** - some diseases harder to detect
4. **Use optimal thresholds** - found automatically per class

---

## 🔬 Research Extensions

### Ensemble Methods
```python
# Average predictions from multiple models
pred_ensemble = (pred_scalo + pred_phaso + pred_fusion) / 3
```

### Transfer Learning
```python
# Fine-tune on related dataset
model.load_state_dict(torch.load('pretrained_model.pth'))
# Train with lower LR (1e-5)
```

### Attention Visualization
```python
# Extract attention maps from Swin Transformer
# Analyze which time-frequency regions are important
```

---

## 📝 Citation

If you use this code, consider citing:
- Original Swin Transformer paper (Liu et al., 2021)
- PTB-XL dataset (Wagner et al., 2020)
- Your approach/modifications

---

## 🤝 Support

For issues:
1. Check dataset paths are correct
2. Verify CWT files generated successfully
3. Ensure GPU has sufficient memory (≥8GB recommended)
4. Check PyTorch/CUDA compatibility

**Recommended Setup:**
- PyTorch ≥2.0
- torchvision ≥0.15
- CUDA ≥11.7
- GPU: ≥8GB VRAM (16GB for batch_size=32)

```
 mamba create -n ecg-cwt --file requirements.txt -c conda-forge -y
 eval "$(mamba shell hook --shell zsh)"
 mamba activate ecg-cwt  
 pip install torchvision
```# ECG-Classification
