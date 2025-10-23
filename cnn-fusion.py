# ============================================================================
# MEMORY-EFFICIENT PIPELINE (FULL-SIZED MODELS): Raw ECG → Standardized → CWT → CNN Models
# ============================================================================

import os
import pickle
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score, hamming_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wfdb
import pywt
from scipy.ndimage import zoom
from config.constants import DATA_PATH, PROCESSED_PATH

# ============================================================================
# PART 1: MEMORY-EFFICIENT DATA LOADING
# ============================================================================

def load_ptbxl_dataset(data_path, processed_path, sampling_rate=100):
    """Load PTB-XL dataset metadata only"""
    Y = pd.read_csv(data_path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    return Y

class ECGDataset(Dataset):
    """Memory-efficient ECG dataset that loads signals on-the-fly"""
    
    def __init__(self, df, data_path, sampling_rate=100, transform=None):
        self.df = df
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.filenames = df.filename_lr.values if sampling_rate == 100 else df.filename_hr.values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load single ECG signal
        filename = self.filenames[idx]
        if self.sampling_rate == 100:
            signal, meta = wfdb.rdsamp(self.data_path + filename)
        else:
            signal, meta = wfdb.rdsamp(self.data_path + filename)
        
        signal = signal.astype(np.float32)
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal

def aggregate_diagnostic_labels(df, scp_statements_path):
    """Aggregate SCP codes into superclasses"""
    aggregation_df = pd.read_csv(scp_statements_path, index_col=0)
    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                c = diag_agg_df.loc[key].diagnostic_class
                if str(c) != 'nan':
                    tmp.append(c)
        return list(set(tmp))
    
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    df['diagnostic_len'] = df.diagnostic_superclass.apply(lambda x: len(x))
    
    return df

def prepare_labels(df, min_samples=0):
    """Convert to multi-hot encoding"""
    mlb = MultiLabelBinarizer()
    
    counts = pd.Series(np.concatenate(df.diagnostic_superclass.values)).value_counts()
    counts = counts[counts > min_samples]
    
    df.diagnostic_superclass = df.diagnostic_superclass.apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )
    df['diagnostic_len'] = df.diagnostic_superclass.apply(lambda x: len(x))
    
    df_filtered = df[df.diagnostic_len > 0]
    
    mlb.fit(df_filtered.diagnostic_superclass.values)
    y = mlb.transform(df_filtered.diagnostic_superclass.values)
    
    print(f"Classes: {mlb.classes_}")
    print(f"Number of samples: {len(df_filtered)}")
    
    return df_filtered, y, mlb

# ============================================================================
# PART 2: BATCHED STANDARDIZATION
# ============================================================================

class Standardizer:
    """Z-score standardization fitted on batches"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit_on_dataset(self, ecg_dataset, num_samples=1000):
        """Fit scaler on a subset of the dataset"""
        print("Computing standardization statistics...")
        
        # Sample random indices for fitting
        indices = np.random.choice(len(ecg_dataset), min(num_samples, len(ecg_dataset)), replace=False)
        
        all_data = []
        for idx in tqdm(indices, desc="Fitting standardizer"):
            ecg = ecg_dataset[idx]
            all_data.append(ecg.flatten())
        
        all_data = np.concatenate(all_data)
        self.mean_ = np.mean(all_data)
        self.std_ = np.std(all_data)
        self.fitted = True
        
        print(f"Standardization stats - Mean: {self.mean_:.4f}, Std: {self.std_:.4f}")
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler not fitted yet")
        return (data - self.mean_) / self.std_

# ============================================================================
# PART 3: ON-THE-FLY CWT GENERATION (FULL SIZE)
# ============================================================================

class CWTGenerator:
    """Generate scalograms and phasograms from standardized ECG signals"""
    
    def __init__(self, sampling_rate=100, image_size=224, wavelet='cmor1.5-1.0'):
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.wavelet = wavelet
        
        # Generate scales for target frequency range (ORIGINAL SIZE)
        freq_min, freq_max = 0.5, 40.0
        n_scales = 128  # ORIGINAL NUMBER OF SCALES
        
        cf = pywt.central_frequency(wavelet)
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_scales)
        self.scales = (cf * sampling_rate) / freqs
        
        print(f"CWT Generator initialized:")
        print(f"  Wavelet: {wavelet}")
        print(f"  Scales: {len(self.scales)} (freq range: {freq_min}-{freq_max} Hz)")
        print(f"  Output size: {image_size}×{image_size}")
    
    def compute_cwt_single_lead(self, signal_1d):
        """Compute CWT for a single lead"""
        try:
            coefficients, _ = pywt.cwt(
                signal_1d,
                self.scales,
                self.wavelet,
                sampling_period=1.0 / self.sampling_rate
            )
            return coefficients
        except Exception as e:
            print(f"CWT error: {e}")
            return None
    
    def generate_scalogram(self, coefficients):
        """Generate scalogram from CWT coefficients"""
        # Power spectrum
        scalogram = np.abs(coefficients) ** 2
        
        # Log scaling
        scalogram = np.log10(scalogram + 1e-10)
        
        # Robust normalization (percentile-based)
        p5, p95 = np.percentile(scalogram, [5, 95])
        scalogram = np.clip(scalogram, p5, p95)
        
        # Normalize to [0, 1]
        min_val, max_val = scalogram.min(), scalogram.max()
        if max_val - min_val > 1e-10:
            scalogram = (scalogram - min_val) / (max_val - min_val)
        else:
            scalogram = np.zeros_like(scalogram)
        
        return scalogram.astype(np.float32)
    
    def generate_phasogram(self, coefficients):
        """Generate phasogram from CWT coefficients"""
        # Extract phase
        phase = np.angle(coefficients)
        
        # Normalize phase from [-π, π] to [0, 1]
        phasogram = (phase + np.pi) / (2 * np.pi)
        
        return phasogram.astype(np.float32)
    
    def resize_to_image(self, cwt_matrix):
        """Resize CWT matrix to target image size"""
        zoom_factors = (
            self.image_size / cwt_matrix.shape[0],
            self.image_size / cwt_matrix.shape[1]
        )
        return zoom(cwt_matrix, zoom_factors, order=1)
    
    def process_single_ecg(self, ecg_signal):
        """
        Process single ECG to generate scalogram and phasogram (12 leads)
        
        Args:
            ecg_signal: (time, 12) array
            
        Returns:
            scalogram: (12, H, W) array
            phasogram: (12, H, W) array
        """
        # Ensure shape is (12, time)
        if ecg_signal.shape[0] != 12:
            ecg_signal = ecg_signal.T
        
        scalograms = []
        phasograms = []
        
        for lead_idx in range(12):
            # Compute CWT for this lead
            coeffs = self.compute_cwt_single_lead(ecg_signal[lead_idx])
            
            if coeffs is None:
                # If CWT fails, use zeros
                scalograms.append(np.zeros((self.image_size, self.image_size)))
                phasograms.append(np.zeros((self.image_size, self.image_size)))
                continue
            
            # Generate scalogram and phasogram
            scalo = self.generate_scalogram(coeffs)
            phaso = self.generate_phasogram(coeffs)
            
            # Resize to target size
            scalo_resized = self.resize_to_image(scalo)
            phaso_resized = self.resize_to_image(phaso)
            
            scalograms.append(scalo_resized)
            phasograms.append(phaso_resized)
        
        # Stack all leads: (12, H, W)
        scalogram_12ch = np.stack(scalograms, axis=0)
        phasogram_12ch = np.stack(phasograms, axis=0)
        
        return scalogram_12ch, phasogram_12ch

class CWTOnTheFlyDataset(Dataset):
    """Generate CWT representations on-the-fly during training"""
    
    def __init__(self, ecg_dataset, cwt_generator, labels, standardizer=None, mode='scalogram'):
        self.ecg_dataset = ecg_dataset
        self.cwt_generator = cwt_generator
        self.labels = torch.FloatTensor(labels)
        self.standardizer = standardizer
        self.mode = mode
    
    def __len__(self):
        return len(self.ecg_dataset)
    
    def __getitem__(self, idx):
        # Load and standardize ECG
        ecg = self.ecg_dataset[idx]
        
        if self.standardizer:
            ecg = self.standardizer.transform(ecg)
        
        # Generate CWT
        scalogram, phasogram = self.cwt_generator.process_single_ecg(ecg)
        
        if self.mode == 'scalogram':
            return torch.FloatTensor(scalogram), self.labels[idx]
        elif self.mode == 'phasogram':
            return torch.FloatTensor(phasogram), self.labels[idx]
        elif self.mode == 'both':
            return (torch.FloatTensor(scalogram), torch.FloatTensor(phasogram)), self.labels[idx]
        elif self.mode == 'fusion':
            # Concatenate along channel dimension
            fused = np.concatenate([scalogram, phasogram], axis=0)
            return torch.FloatTensor(fused), self.labels[idx]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# ============================================================================
# PART 4: ORIGINAL MODEL ARCHITECTURES (FULL SIZE)
# ============================================================================

class CWT2DCNN(nn.Module):
    """
    2D CNN for CWT treating 12 leads as channels
    ORIGINAL FULL-SIZED MODEL
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
        
        # Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"CWT2DCNN: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")
    
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
        # x: (B, 12, H, W)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        
        return self.fc(x)

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

class DualStreamCNN(nn.Module):
    """Dual-stream CNN for scalogram + phasogram fusion"""
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        self.scalogram_branch = CWT2DCNN(num_classes, num_channels)
        self.phasogram_branch = CWT2DCNN(num_classes, num_channels)
        
        # Remove final FC layers
        self.scalogram_branch.fc = nn.Identity()
        self.phasogram_branch.fc = nn.Identity()
        
        # Fusion head
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),  # *2 for concat pooling, *2 for two branches
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"DualStreamCNN: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")
    
    def forward(self, scalogram, phasogram):
        feat_scalo = self.scalogram_branch(scalogram)
        feat_phaso = self.phasogram_branch(phasogram)
        
        combined = torch.cat([feat_scalo, feat_phaso], dim=1)
        return self.fusion_fc(combined)

# ============================================================================
# PART 5: TRAINING FUNCTIONS WITH GRADIENT ACCUMULATION
# ============================================================================

def train_epoch_memory_efficient(model, dataloader, criterion, optimizer, device, accumulation_steps=4):
    """Train for one epoch with gradient accumulation"""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        if isinstance(batch[0], tuple):
            (x1, x2), y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
        
        loss = criterion(outputs, y) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps * y.size(0)
    
    # Handle any remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def validate_memory_efficient(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        if isinstance(batch[0], tuple):
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
    macro_auc = roc_auc_score(y_true, y_scores, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f_beta = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
    
    return {
        'macro_auc': macro_auc,
        'f1_macro': f1_macro,
        'f_beta_macro': f_beta
    }

# ============================================================================
# PART 6: MAIN PIPELINE WITH ON-THE-FLY PROCESSING
# ============================================================================

def main():
    # Configuration (ORIGINAL SIZES)
    from config.constants import DATA_PATH, PROCESSED_PATH
    DATA_PATH = DATA_PATH
    PROCESSED_PATH = PROCESSED_PATH
    SAMPLING_RATE = 100
    IMAGE_SIZE = 224  # ORIGINAL IMAGE SIZE
    BATCH_SIZE = 8    # Reduced batch size with gradient accumulation
    ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("MEMORY-EFFICIENT PIPELINE WITH FULL-SIZED MODELS")
    print("="*80)
    
    # Step 1: Load metadata only
    print("\n[1/7] Loading PTB-XL dataset metadata...")
    Y = load_ptbxl_dataset(DATA_PATH, PROCESSED_PATH, SAMPLING_RATE)
    
    # Step 2: Process labels
    print("\n[2/7] Processing labels...")
    Y = aggregate_diagnostic_labels(Y, DATA_PATH + 'scp_statements.csv')
    Y_filtered, y, mlb = prepare_labels(Y, min_samples=0)
    
    # Step 3: Split data
    print("\n[3/7] Splitting data...")
    train_df = Y_filtered[Y_filtered.strat_fold <= 8]
    val_df = Y_filtered[Y_filtered.strat_fold == 9]
    test_df = Y_filtered[Y_filtered.strat_fold == 10]
    
    y_train = y[Y_filtered.strat_fold <= 8]
    y_val = y[Y_filtered.strat_fold == 9]
    y_test = y[Y_filtered.strat_fold == 10]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Step 4: Create ECG datasets (load on-the-fly)
    print("\n[4/7] Creating memory-efficient ECG datasets...")
    train_ecg_dataset = ECGDataset(train_df, DATA_PATH, SAMPLING_RATE)
    val_ecg_dataset = ECGDataset(val_df, DATA_PATH, SAMPLING_RATE)
    test_ecg_dataset = ECGDataset(test_df, DATA_PATH, SAMPLING_RATE)
    
    # Step 5: Compute standardization stats
    print("\n[5/7] Computing standardization statistics...")
    standardizer = Standardizer()
    standardizer.fit_on_dataset(train_ecg_dataset, num_samples=1000)
    
    # Step 6: Initialize CWT generator (ORIGINAL SIZE)
    print("\n[6/7] Initializing CWT generator...")
    cwt_gen = CWTGenerator(sampling_rate=SAMPLING_RATE, image_size=IMAGE_SIZE)
    
    # Step 7: Train different model configurations
    print("\n[7/7] Training models...")
    
    configs = [
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN'},
        {'mode': 'phasogram', 'model': 'CWT2DCNN', 'name': 'Phasogram-2DCNN'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'Fusion-DualStream'},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Training: {config['name']}")
        print(f"{'='*80}")
        
        # Create CWT datasets (generate on-the-fly)
        train_dataset = CWTOnTheFlyDataset(train_ecg_dataset, cwt_gen, y_train, standardizer, mode=config['mode'])
        val_dataset = CWTOnTheFlyDataset(val_ecg_dataset, cwt_gen, y_val, standardizer, mode=config['mode'])
        test_dataset = CWTOnTheFlyDataset(test_ecg_dataset, cwt_gen, y_test, standardizer, mode=config['mode'])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Create model
        if config['model'] == 'DualStream':
            model = DualStreamCNN(num_classes=len(mlb.classes_), num_channels=12)
        else:
            model = CWT2DCNN(num_classes=len(mlb.classes_), num_channels=12)
        
        model = model.to(DEVICE)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Training loop
        best_val_auc = 0.0
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            
            # Train with gradient accumulation
            train_loss = train_epoch_memory_efficient(
                model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS
            )
            
            # Validate
            val_loss, val_preds, val_labels = validate_memory_efficient(model, val_loader, criterion, DEVICE)
            
            # Compute metrics
            val_pred_binary = (val_preds > 0.5).astype(int)
            val_metrics = compute_metrics(val_labels, val_pred_binary, val_preds)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_metrics['macro_auc']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
            
            # Save best model
            if val_metrics['macro_auc'] > best_val_auc:
                best_val_auc = val_metrics['macro_auc']
                torch.save(model.state_dict(), f"best_{config['name']}.pth")
                print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
            
            scheduler.step(val_loss)
        
        # Test
        print(f"\nTesting {config['name']}...")
        model.load_state_dict(torch.load(f"best_{config['name']}.pth"))
        test_loss, test_preds, test_labels = validate_memory_efficient(model, test_loader, criterion, DEVICE)
        
        test_pred_binary = (test_preds > 0.5).astype(int)
        test_metrics = compute_metrics(test_labels, test_pred_binary, test_preds)
        
        results[config['name']] = {
            'auc': test_metrics['macro_auc'],
            'f1': test_metrics['f1_macro'],
            'f_beta': test_metrics['f_beta_macro'],
            'y_true': test_labels,
            'y_pred': test_pred_binary,
            'y_scores': test_preds
        }
        
        print(f"\nTest Results - {config['name']}:")
        print(f"  AUC: {test_metrics['macro_auc']:.4f}")
        print(f"  F1: {test_metrics['f1_macro']:.4f}")
        print(f"  F-beta: {test_metrics['f_beta_macro']:.4f}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"{'Model':<30} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
    print("-" * 80)
    
    # Prepare simplified results for JSON
    results_summary = {}
    
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['auc']:.4f}   | {metrics['f1']:.4f}   | {metrics['f_beta']:.4f}")
        
        # Store only numeric results for JSON
        results_summary[name] = {
            'auc': float(metrics['auc']),
            'f1': float(metrics['f1']),
            'f_beta': float(metrics['f_beta'])
        }
    
    # Save numeric results
    import json
    with open('memory_efficient_fullsize_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create comparative visualization
    print("\n" + "="*80)
    print("GENERATING COMPARATIVE VISUALIZATIONS")
    print("="*80)
    
    # Plot metric comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    model_names = list(results_summary.keys())
    
    # AUC comparison
    aucs = [results_summary[name]['auc'] for name in model_names]
    axes[0].bar(range(len(model_names)), aucs, color='steelblue')
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('Macro AUC Comparison')
    axes[0].set_ylim([0.7, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    # F1 comparison
    f1s = [results_summary[name]['f1'] for name in model_names]
    axes[1].bar(range(len(model_names)), f1s, color='coral')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Macro F1 Comparison')
    axes[1].set_ylim([0.7, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    
    # F-beta comparison
    f_betas = [results_summary[name]['f_beta'] for name in model_names]
    axes[2].bar(range(len(model_names)), f_betas, color='mediumseagreen')
    axes[2].set_xticks(range(len(model_names)))
    axes[2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[2].set_ylabel('F-beta Score')
    axes[2].set_title('Macro F-beta Comparison')
    axes[2].set_ylim([0.7, 1.0])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_efficient_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Memory-efficient pipeline complete!")
    print("✓ Results saved to memory_efficient_fullsize_results.json")
    print("✓ Metrics comparison saved as memory_efficient_metrics_comparison.png")

if __name__ == '__main__':
    main()