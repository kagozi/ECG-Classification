import torch
import numpy as np
from tqdm import tqdm
from .metrics import compute_metrics
import pandas as pd
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


# ============================================================================
# TRAINING FUNCTIONS & UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation score stops improving"""
    
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop



def train_one_epoch(model, dataloader, criterion, optimizer, device, threshold=0.5, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    
    # Use scaler if provided, otherwise None for normal training
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal forward/backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        
        # Get predictions
        scores = torch.sigmoid(outputs)
        preds = (scores > threshold).float()
        
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_scores.append(scores.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Aggregate results
    epoch_loss = running_loss / len(dataloader.dataset)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_scores = np.vstack(all_scores)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_scores, threshold)
    metrics['loss'] = epoch_loss
    
    return metrics

def validate(model, dataloader, criterion, device, threshold=0.5):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            
            # Get predictions
            scores = torch.sigmoid(outputs)
            preds = (scores > threshold).float()
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Aggregate results
    epoch_loss = running_loss / len(dataloader.dataset)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_scores = np.vstack(all_scores)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_scores, threshold)
    metrics['loss'] = epoch_loss
    
    return metrics, all_labels, all_preds, all_scores

# ============================================================================
# COMPLETE TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs=25, patience=5, threshold=0.5, 
                save_path='best_model.pth', class_names=None):
    """
    Complete training loop with validation and early stopping
    
    Returns:
        history: Dictionary with training history
        best_model_state: State dict of best model
    """
    
    # Initialize tracking
    history = {
        'train_loss': [], 'train_f1_macro': [], 'train_auc_macro': [],
        'val_loss': [], 'val_f1_macro': [], 'val_auc_macro': [],
        'lr': []
    }
    
    best_f1 = 0.0
    best_model_state = None
    early_stopping = EarlyStopping(patience=patience, mode='max')
    scaler = GradScaler()
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, threshold, scaler=scaler)
        
        # Validate
        val_metrics, _, _, _ = validate(model, val_loader, criterion, device, threshold)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['train_auc_macro'].append(train_metrics['auc_macro'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_auc_macro'].append(val_metrics['auc_macro'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Train F1: {train_metrics['f1_macro']:.4f} | "
              f"Train AUC: {train_metrics['auc_macro']:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f} | "
              f"Val F1:   {val_metrics['f1_macro']:.4f} | "
              f"Val AUC:   {val_metrics['auc_macro']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'metrics': val_metrics
            }, save_path)
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['f1_macro']):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETED - Best Val F1: {best_f1:.4f}")
    print("="*80)
    
    return history, best_model_state

# ============================================================================
# EVALUATION & TESTING
# ============================================================================

def evaluate_model(model, test_loader, device, threshold=0.5, class_names=None):
    """Comprehensive model evaluation on test set"""
    
    print("\n" + "="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    # Get predictions
    criterion = nn.BCEWithLogitsLoss()
    test_metrics, y_true, y_pred, y_scores = validate(model, test_loader, criterion, device, threshold)
    
    # Print overall metrics
    print("\n" + "="*40)
    print("OVERALL METRICS")
    print("="*40)
    print(f"F1 (Macro):     {test_metrics['f1_macro']:.4f}")
    print(f"F1 (Micro):     {test_metrics['f1_micro']:.4f}")
    print(f"F1 (Weighted):  {test_metrics['f1_weighted']:.4f}")
    print(f"F1 (Samples):   {test_metrics['f1_samples']:.4f}")
    print(f"AUC (Macro):    {test_metrics['auc_macro']:.4f}")
    print(f"AUC (Micro):    {test_metrics['auc_micro']:.4f}")
    print(f"Accuracy:       {test_metrics['exact_match_accuracy']:.4f}")
    print(f"Precision:      {test_metrics['precision_macro']:.4f}")
    print(f"Recall:         {test_metrics['recall_macro']:.4f}")
    print(f"Hamming Loss:   {test_metrics['hamming_loss']:.4f}")
    print(f"Hamming Accuracy: {test_metrics['hamming_accuracy']:.4f}")
    print(f"Label Accuracy: {test_metrics['label_accuracy']:.4f}")

    
    
    # Per-class metrics
    if class_names is not None:
        print("\n" + "="*40)
        print("PER-CLASS METRICS")
        print("="*40)
        
        from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
        
        per_class_metrics = []
        for i, class_name in enumerate(class_names):
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            try:
                auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            except:
                auc = 0.0
            prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            support = y_true[:, i].sum()
            
            per_class_metrics.append({
                'Class': class_name,
                'F1': f1,
                'AUC': auc,
                'Precision': prec,
                'Recall': rec,
                'Support': int(support)
            })
        
        df_metrics = pd.DataFrame(per_class_metrics)
        print(df_metrics.to_string(index=False))
    
    return test_metrics, y_true, y_pred, y_scores




# ============================================================================
# TEST TIME AUGMENTATION (TTA)
# ============================================================================

def predict_with_tta(model, dataloader, device, n_augments=5):
    """Predictions with test-time augmentation"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for _ in range(n_augments):
            batch_scores = []
            batch_labels = []
            
            for images, labels in tqdm(dataloader, desc=f'TTA pass'):
                images = images.to(device)
                outputs = model(images)
                scores = torch.sigmoid(outputs)
                
                batch_scores.append(scores.cpu().numpy())
                if len(batch_labels) == 0:
                    batch_labels.append(labels.cpu().numpy())
            
            all_scores.append(np.vstack(batch_scores))
            if len(all_labels) == 0:
                all_labels = np.vstack(batch_labels)
    
    # Average predictions across augmentations
    avg_scores = np.mean(all_scores, axis=0)
    
    return all_labels, avg_scores
