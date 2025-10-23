# ============================================================================
# COMPLETE PIPELINE: Raw ECG → Standardized → CWT → CNN Models
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
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wfdb
import pywt
from scipy.ndimage import zoom
from config.constants import (PROCESSED_PATH, DATA_PATH)

# ============================================================================
# PART 1: DATA LOADING (Same as XResNet1D)
# ============================================================================

def load_ptbxl_dataset(data_path, processed_path, sampling_rate=100):
    """Load PTB-XL dataset"""
    Y = pd.read_csv(data_path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = load_raw_signals(Y, sampling_rate, data_path, processed_path)
    return X, Y


def load_raw_signals(df, sampling_rate, data_path, processed_path):
    """Load raw ECG signals"""
    os.makedirs(processed_path, exist_ok=True)
    cache_file = os.path.join(processed_path, f'raw{sampling_rate}.npy')
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
    else:
        print(f"Loading and caching raw signals at {sampling_rate}Hz")
        if sampling_rate == 100:
            data = [wfdb.rdsamp(data_path + f) for f in tqdm(df.filename_lr)]
        else:
            data = [wfdb.rdsamp(data_path + f) for f in tqdm(df.filename_hr)]
        
        data = np.array([signal for signal, meta in data])
        np.save(cache_file, data)
    
    return data


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


def prepare_labels(X, Y, min_samples=0):
    """Convert to multi-hot encoding"""
    mlb = MultiLabelBinarizer()
    
    counts = pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()
    counts = counts[counts > min_samples]
    
    Y.diagnostic_superclass = Y.diagnostic_superclass.apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )
    Y['diagnostic_len'] = Y.diagnostic_superclass.apply(lambda x: len(x))
    
    X = X[Y.diagnostic_len > 0]
    Y = Y[Y.diagnostic_len > 0]
    
    mlb.fit(Y.diagnostic_superclass.values)
    y = mlb.transform(Y.diagnostic_superclass.values)
    
    print(f"Classes: {mlb.classes_}")
    print(f"Number of samples: {len(X)}")
    
    return X, Y, y, mlb

# ============================================================================
# PART 2: STANDARDIZATION (Z-Score Normalization)
# ============================================================================

def preprocess_signals(X_train, X_val, X_test):
    """Standardize signals using Z-score normalization"""
    # Fit StandardScaler on training data
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    
    # Apply to all sets
    X_train_scaled = apply_standardizer(X_train, ss)
    X_val_scaled = apply_standardizer(X_val, ss)
    X_test_scaled = apply_standardizer(X_test, ss)
    
    print(f"Standardization stats - Mean: {ss.mean_[0]:.4f}, Std: {ss.scale_[0]:.4f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, ss


def apply_standardizer(X, ss):
    """Apply standardization to signals"""
    X_tmp = []
    for x in tqdm(X, desc="Standardizing"):
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    return np.array(X_tmp)

# ============================================================================
# PART 3: CWT GENERATION FROM STANDARDIZED SIGNALS
# ============================================================================

class CWTGenerator:
    """Generate scalograms and phasograms from standardized ECG signals"""
    
    def __init__(self, sampling_rate=100, image_size=224, wavelet='cmor1.5-1.0'):
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.wavelet = wavelet
        
        # Generate scales for target frequency range
        freq_min, freq_max = 0.5, 40.0
        n_scales = 128
        
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
    
    def process_12_lead_ecg(self, ecg_12_lead):
        """
        Process 12-lead ECG to generate scalogram and phasogram
        
        Args:
            ecg_12_lead: (time, 12) or (12, time) array
            
        Returns:
            scalogram: (12, H, W) array
            phasogram: (12, H, W) array
        """
        # Ensure shape is (12, time)
        if ecg_12_lead.shape[0] != 12:
            ecg_12_lead = ecg_12_lead.T
        
        scalograms = []
        phasograms = []
        
        for lead_idx in range(12):
            # Compute CWT for this lead
            coeffs = self.compute_cwt_single_lead(ecg_12_lead[lead_idx])
            
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
    
    def process_dataset(self, X, cache_dir=None, cache_name='cwt'):
        """
        Process entire dataset
        
        Args:
            X: (N, time, 12) or (N, 12, time) array of ECG signals
            cache_dir: Directory to cache results
            cache_name: Name for cache files
            
        Returns:
            scalograms: (N, 12, H, W) array
            phasograms: (N, 12, H, W) array
        """
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            scalo_cache = os.path.join(cache_dir, f'{cache_name}_scalograms.npy')
            phaso_cache = os.path.join(cache_dir, f'{cache_name}_phasograms.npy')
            
            if os.path.exists(scalo_cache) and os.path.exists(phaso_cache):
                print(f"Loading cached CWT data from {cache_dir}")
                scalograms = np.load(scalo_cache)
                phasograms = np.load(phaso_cache)
                return scalograms, phasograms
        
        print(f"Generating CWT representations for {len(X)} samples...")
        scalograms = []
        phasograms = []
        
        for i, ecg in enumerate(tqdm(X)):
            scalo, phaso = self.process_12_lead_ecg(ecg)
            scalograms.append(scalo)
            phasograms.append(phaso)
        
        scalograms = np.array(scalograms)
        phasograms = np.array(phasograms)
        
        # Cache results
        if cache_dir is not None:
            print(f"Caching CWT data to {cache_dir}")
            np.save(scalo_cache, scalograms)
            np.save(phaso_cache, phasograms)
        
        print(f"Generated scalograms: {scalograms.shape}")
        print(f"Generated phasograms: {phasograms.shape}")
        
        return scalograms, phasograms

# ============================================================================
# PART 4: PYTORCH DATASETS
# ============================================================================

class CWTDataset(Dataset):
    """Dataset for CWT representations"""
    
    def __init__(self, scalograms, phasograms, labels, mode='scalogram'):
        """
        Args:
            scalograms: (N, 12, H, W) array
            phasograms: (N, 12, H, W) array
            labels: (N, num_classes) array
            mode: 'scalogram', 'phasogram', 'both', or 'fusion'
        """
        self.scalograms = torch.FloatTensor(scalograms)
        self.phasograms = torch.FloatTensor(phasograms)
        self.labels = torch.FloatTensor(labels)
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.mode == 'scalogram':
            return self.scalograms[idx], self.labels[idx]
        elif self.mode == 'phasogram':
            return self.phasograms[idx], self.labels[idx]
        elif self.mode == 'both':
            return (self.scalograms[idx], self.phasograms[idx]), self.labels[idx]
        elif self.mode == 'fusion':
            # Concatenate along channel dimension: (12, H, W) + (12, H, W) = (24, H, W)
            fused = torch.cat([self.scalograms[idx], self.phasograms[idx]], dim=0)
            return fused, self.labels[idx]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# ============================================================================
# PART 5: MODEL ARCHITECTURES
# ============================================================================

class CWT1DCNN(nn.Module):
    """
    1D CNN for CWT coefficients (12 channels)
    Treats CWT as (batch, 12_channels, time) sequence
    """
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        # Process 12 channels along time axis
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        self.conv2 = self._make_block(64, 128, stride=2)
        self.conv3 = self._make_block(128, 256, stride=2)
        self.conv4 = self._make_block(256, 512, stride=2)
        
        # Pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"CWT1DCNN: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")
    
    def _make_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Input: (B, 12, H, W) - treat as (B, 12, H*W) for 1D processing
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)  # Flatten spatial dimensions
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        
        return self.fc(x)


class CWT2DCNN(nn.Module):
    """
    2D CNN for CWT treating 12 leads as channels
    More appropriate than transformers for time-frequency data
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
# PART 6: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if isinstance(batch[0], tuple):
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
    
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
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
# PART 7: MAIN TRAINING PIPELINE
# ============================================================================

def main():
    # Configuration
    DATA_PATH = DATA_PATH  # UPDATE THIS
    PROCESSED_PATH = PROCESSED_PATH
    SAMPLING_RATE = 100
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("STANDARDIZED ECG → CWT → CNN PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    print("\n[1/7] Loading PTB-XL dataset...")
    X, Y = load_ptbxl_dataset(DATA_PATH, PROCESSED_PATH, SAMPLING_RATE)
    
    # Step 2: Process labels
    print("\n[2/7] Processing labels...")
    Y = aggregate_diagnostic_labels(Y, DATA_PATH + 'scp_statements.csv')
    X, Y, y, mlb = prepare_labels(X, Y, min_samples=0)
    
    # Step 3: Split data
    print("\n[3/7] Splitting data...")
    X_train = X[Y.strat_fold <= 8]
    y_train = y[Y.strat_fold <= 8]
    X_val = X[Y.strat_fold == 9]
    y_val = y[Y.strat_fold == 9]
    X_test = X[Y.strat_fold == 10]
    y_test = y[Y.strat_fold == 10]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 4: Standardize
    print("\n[4/7] Standardizing signals (Z-score)...")
    X_train, X_val, X_test, scaler = preprocess_signals(X_train, X_val, X_test)
    
    # Step 5: Generate CWT representations
    print("\n[5/7] Generating CWT representations from standardized signals...")
    cwt_gen = CWTGenerator(sampling_rate=SAMPLING_RATE, image_size=IMAGE_SIZE)
    
    scalo_train, phaso_train = cwt_gen.process_dataset(
        X_train, cache_dir=PROCESSED_PATH, cache_name='train'
    )
    scalo_val, phaso_val = cwt_gen.process_dataset(
        X_val, cache_dir=PROCESSED_PATH, cache_name='val'
    )
    scalo_test, phaso_test = cwt_gen.process_dataset(
        X_test, cache_dir=PROCESSED_PATH, cache_name='test'
    )
    
    # Step 6: Create datasets
    print("\n[6/7] Creating PyTorch datasets...")
    
    # Try different model configurations
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
        
        # Create datasets
        if config['mode'] == 'both':
            train_dataset = CWTDataset(scalo_train, phaso_train, y_train, mode='both')
            val_dataset = CWTDataset(scalo_val, phaso_val, y_val, mode='both')
            test_dataset = CWTDataset(scalo_test, phaso_test, y_test, mode='both')
        else:
            train_dataset = CWTDataset(scalo_train, phaso_train, y_train, mode=config['mode'])
            val_dataset = CWTDataset(scalo_val, phaso_val, y_val, mode=config['mode'])
            test_dataset = CWTDataset(scalo_test, phaso_test, y_test, mode=config['mode'])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
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
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            
            # Validate
            val_loss, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
            
            # Compute metrics (using 0.5 threshold for now)
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
        test_loss, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)
        
        test_pred_binary = (test_preds > 0.5).astype(int)
        test_metrics = compute_metrics(test_labels, test_pred_binary, test_preds)
        
        results[config['name']] = {
            'auc': test_metrics['macro_auc'],
            'f1': test_metrics['f1_macro'],
            'f_beta': test_metrics['f_beta_macro']
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
    
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['auc']:.4f}   | {metrics['f1']:.4f}   | {metrics['f_beta']:.4f}")
    
    # Save results
    import json
    with open('standardized_cwt_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Pipeline complete! Results saved to standardized_cwt_results.json")


if __name__ == '__main__':
    main()