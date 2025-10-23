# ============================================================================
# STEP 3: Train CNN Models on CWT Representations
# ============================================================================
# Run this after 2_generate_cwt.py
# Uses memory-efficient data loading with PyTorch DataLoader
# Tests multiple model architectures: Scalogram, Phasogram, Fusion

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

print("="*80)
print("STEP 3: TRAIN CNN MODELS ON CWT REPRESENTATIONS")
print("="*80)
print(f"Device: {DEVICE}")

# ============================================================================
# DATASET CLASS (Memory-Efficient)
# ============================================================================

class CWTDataset(Dataset):
    """
    Memory-efficient dataset that loads CWT data on-the-fly
    Uses memory mapping to avoid loading entire dataset into RAM
    """
    
    def __init__(self, scalo_path, phaso_path, labels, mode='scalogram'):
        """
        Args:
            scalo_path: Path to scalogram .npy file
            phaso_path: Path to phasogram .npy file
            labels: (N, num_classes) numpy array
            mode: 'scalogram', 'phasogram', 'both', or 'fusion'
        """
        self.scalograms = np.load(scalo_path, mmap_mode='r')
        self.phasograms = np.load(phaso_path, mmap_mode='r')
        self.labels = torch.FloatTensor(labels)
        self.mode = mode
        
        print(f"  Dataset loaded: {len(self.labels)} samples, mode={mode}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load data on-the-fly from memory-mapped files
        scalo = torch.FloatTensor(self.scalograms[idx])
        phaso = torch.FloatTensor(self.phasograms[idx])
        label = self.labels[idx]
        
        if self.mode == 'scalogram':
            return scalo, label
        elif self.mode == 'phasogram':
            return phaso, label
        elif self.mode == 'both':
            return (scalo, phaso), label
        elif self.mode == 'fusion':
            # Concatenate along channel dimension: (12, H, W) + (12, H, W) = (24, H, W)
            fused = torch.cat([scalo, phaso], dim=0)
            return fused, label
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ResidualBlock2D(nn.Module):
    """Residual block for 2D CNN"""
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class CWT2DCNN(nn.Module):
    """
    2D CNN for CWT representations
    Treats 12 ECG leads as input channels
    """
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Pooling (combine avg and max)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  CWT2DCNN: {n_params/1e6:.1f}M parameters")
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_ch, out_ch, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        return ResidualBlock2D(in_ch, out_ch, stride, downsample)
    
    def forward(self, x):
        # x: (B, channels, H, W)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        
        return self.fc(x)


class DualStreamCNN(nn.Module):
    """
    Dual-stream CNN for scalogram + phasogram fusion
    Two parallel branches that share no weights
    """
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        # Two independent branches
        self.scalogram_branch = CWT2DCNN(num_classes, num_channels)
        self.phasogram_branch = CWT2DCNN(num_classes, num_channels)
        
        # Remove final FC layers from branches
        self.scalogram_branch.fc = nn.Identity()
        self.phasogram_branch.fc = nn.Identity()
        
        # Fusion head
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),  # *2 for concat pooling, *2 for two branches
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  DualStreamCNN: {n_params/1e6:.1f}M parameters")
    
    def forward(self, scalogram, phasogram):
        feat_scalo = self.scalogram_branch(scalogram)
        feat_phaso = self.phasogram_branch(phasogram)
        
        combined = torch.cat([feat_scalo, feat_phaso], dim=1)
        return self.fusion_fc(combined)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, is_dual=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if is_dual:
            (x1, x2), y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x1, x2)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
        
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * y.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion, device, is_dual=False):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        if is_dual:
            (x1, x2), y = batch
            x1, x2 = x1.to(device), x2.to(device)
            out = model(x1, x2)
        else:
            x, y = batch
            x = x.to(device)
            out = model(x)
        
        loss = criterion(out, y.to(device))
        running_loss += loss.item() * y.size(0)
        
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(y.numpy())
    
    return running_loss / len(dataloader.dataset), np.vstack(all_preds), np.vstack(all_labels)


def compute_metrics(y_true, y_pred, y_scores):
    """Compute evaluation metrics"""
    try:
        macro_auc = roc_auc_score(y_true, y_scores, average='macro')
    except:
        macro_auc = 0.0
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f_beta = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
    
    return {
        'macro_auc': macro_auc,
        'f1_macro': f1_macro,
        'f_beta_macro': f_beta
    }


def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold per class using F1 score"""
    thresholds = []
    for i in range(y_true.shape[1]):
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_scores[:, i] > thresh).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        thresholds.append(best_thresh)
    return np.array(thresholds)

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_model(config, metadata, device):
    """Train a single model configuration"""
    
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"{'='*80}")
    
    # Load labels
    y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
    y_val = np.load(os.path.join(PROCESSED_PATH, 'y_val.npy'))
    y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
    # Create datasets
    mode = config['mode']
    is_dual = (config['model'] == 'DualStream')
    
    print(f"\nCreating datasets (mode={mode})...")
    train_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'train_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'train_phasograms.npy'),
        y_train, mode=mode
    )
    val_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'val_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'val_phasograms.npy'),
        y_val, mode=mode
    )
    test_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'test_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'test_phasograms.npy'),
        y_test, mode=mode
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model...")
    num_classes = metadata['num_classes']
    
    if config['model'] == 'DualStream':
        model = DualStreamCNN(num_classes=num_classes, num_channels=12)
    elif config['model'] == 'CWT2DCNN':
        # Adjust channels for fusion mode (24 channels = 12 scalo + 12 phaso)
        num_ch = 24 if mode == 'fusion' else 12
        model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    best_val_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, is_dual)
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device, is_dual)
        
        # Compute metrics (using 0.5 threshold)
        val_pred_binary = (val_preds > 0.5).astype(int)
        val_metrics = compute_metrics(val_labels, val_pred_binary, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['macro_auc'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_metrics['macro_auc']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
        
        # Save best model
        if val_metrics['macro_auc'] > best_val_auc:
            best_val_auc = val_metrics['macro_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': config
            }, os.path.join(PROCESSED_PATH, f"best_{config['name']}.pth"))
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        
        scheduler.step(val_metrics['macro_auc'])
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stopping early")
            break
    
    # Test with best model
    print(f"\nTesting {config['name']}...")
    checkpoint = torch.load(os.path.join(PROCESSED_PATH, f"best_{config['name']}.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device, is_dual)
    
    # Find optimal thresholds on validation set
    print("Finding optimal thresholds on validation set...")
    _, val_preds_final, val_labels_final = validate(model, val_loader, criterion, device, is_dual)
    optimal_thresholds = find_optimal_threshold(val_labels_final, val_preds_final)
    
    # Apply optimal thresholds to test set
    test_pred_optimal = np.zeros_like(test_preds)
    for i in range(test_preds.shape[1]):
        test_pred_optimal[:, i] = (test_preds[:, i] > optimal_thresholds[i]).astype(int)
    
    test_metrics = compute_metrics(test_labels, test_pred_optimal, test_preds)
    
    print(f"\nTest Results - {config['name']}:")
    print(f"  AUC:    {test_metrics['macro_auc']:.4f}")
    print(f"  F1:     {test_metrics['f1_macro']:.4f}")
    print(f"  F-beta: {test_metrics['f_beta_macro']:.4f}")
    
    # Save results
    results = {
        'config': config,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'optimal_thresholds': optimal_thresholds.tolist(),
        'history': history
    }
    
    with open(os.path.join(PROCESSED_PATH, f"results_{config['name']}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    # Load metadata
    print("\n[1/2] Loading metadata...")
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Dataset info:")
    print(f"  Classes: {metadata['num_classes']} - {metadata['classes']}")
    print(f"  Train: {metadata['train_size']} samples")
    print(f"  Val:   {metadata['val_size']} samples")
    print(f"  Test:  {metadata['test_size']} samples")
    
    # Define model configurations to train
    configs = [
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN'},
        {'mode': 'phasogram', 'model': 'CWT2DCNN', 'name': 'Phasogram-2DCNN'},
        {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN'},
    ]
    
    # Train all models
    print("\n[2/2] Training models...")
    all_results = {}
    
    for config in configs:
        results = train_model(config, metadata, DEVICE)
        all_results[config['name']] = results['test_metrics']
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"{'Model':<30} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
    print("-" * 80)
    
    for name, metrics in all_results.items():
        print(f"{name:<30} | {metrics['macro_auc']:.4f}   | "
              f"{metrics['f1_macro']:.4f}   | {metrics['f_beta_macro']:.4f}")
    
    # Save final results
    with open(os.path.join(PROCESSED_PATH, 'final_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("STEP 3 COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {PROCESSED_PATH}")
    print("\nPipeline finished successfully!")


if __name__ == '__main__':
    main()