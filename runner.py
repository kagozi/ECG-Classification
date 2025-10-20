"""
ECG Classification with Scalograms and Phasograms - Complete Training Script
============================================================================

This script trains state-of-the-art models on ECG scalograms and/or phasograms 
for multi-label classification with support for fusion models.
"""
import numpy as np
import torch
import pandas as pd
from utils.data_loader import create_dataloaders
from models.model_create import create_model
from utils.loss_functions import FocalLoss
from utils.training import train_model, predict_with_tta, evaluate_model
from utils.metrics import compute_metrics
from utils.plotting import plot_training_history, plot_confusion_matrices, plot_roc_curves, plot_precision_recall_curves, plot_confusion_matrix_all_classes
from config.constants import RESULTS_PATH
import os

# ============================================================================
# COMPLETE PIPELINE FUNCTION
# ============================================================================

def run_complete_pipeline(train_df, y_train, val_df, y_val, test_df, y_test,
                         model_name='resnet50', mode='composite_scalogram',
                         batch_size=16, num_epochs=25, lr=5e-5, patience=5, 
                         threshold=0.5, device='cuda', class_names=None, 
                         use_tta=False, swin_model_name='swin_base_patch4_window7_224',
                         fusion_type='early'):
    """
    Run complete training and evaluation pipeline
    
    Args:
        train_df, val_df, test_df: DataFrames with scalogram/phasogram paths
        y_train, y_val, y_test: Binary label arrays
        model_name: Model architecture ('resnet50', 'efficientnet_b3', 'densenet121', 
                    'vit', 'swin', 'swin_fusion')
        mode: Data mode ('composite_scalogram', 'composite_phasogram', 'composite_both',
              'lead2_scalogram', 'lead2_phasogram', 'lead2_both')
        batch_size: Batch size
        num_epochs: Maximum epochs
        lr: Learning rate
        patience: Early stopping patience
        threshold: Classification threshold
        device: Device to use
        class_names: List of class names
        use_tta: Use test-time augmentation
        swin_model_name: Specific Swin variant (for 'swin' and 'swin_fusion')
        fusion_type: 'early' or 'late' (for 'swin_fusion')
    
    Returns:
        results: Dictionary with all results
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    print("="*80)
    print("ECG CLASSIFICATION PIPELINE")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Data Mode: {mode}")
    if model_name in ['swin', 'swin_fusion']:
        print(f"Swin Variant: {swin_model_name}")
    if model_name == 'swin_fusion':
        print(f"Fusion Type: {fusion_type}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Classes: {len(class_names) if class_names else y_train.shape[1]}")
    print("="*80 + "\n")
    
    # 1. Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, y_train, val_df, y_val, test_df, y_test,
        mode=mode, batch_size=batch_size, num_workers=4, 
        image_size=224, augment=True
    )
    
    # 2. Create model
    print(f"\nCreating {model_name} model...")
    num_classes = y_train.shape[1]
    
    # Adjust dropout for different models
    dropout = 0.3 if model_name in ['vit', 'swin', 'swin_fusion'] else 0.5
    
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=True,
        swin_model_name=swin_model_name,
        fusion_type=fusion_type
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")
    
    # 3. Setup training
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 4. Train
    save_path = f'best_{model_name}_{mode}.pth'
    history, best_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=num_epochs, patience=patience, threshold=threshold,
        save_path=save_path, class_names=class_names
    )
    
    # 5. Load best model and evaluate
    print("\nLoading best model...")
    model.load_state_dict(best_state)
    
    if use_tta:
        print("\nEvaluating with Test-Time Augmentation...")
        y_true, y_scores = predict_with_tta(model, test_loader, device, n_augments=5)
        y_pred = (y_scores > threshold).astype(float)
        test_metrics = compute_metrics(y_true, y_pred, y_scores, threshold)
    else:
        test_metrics, y_true, y_pred, y_scores = evaluate_model(
            model, test_loader, device, threshold=threshold, class_names=class_names
        )
    
    # 6. Visualizations
    print("\nGenerating visualizations...")
    plot_training_history(history, save_path=os.path.join(RESULTS_PATH, f'{model_name}_{mode}_history.png'))
    
    if class_names:
        plot_training_history(history, save_path=os.path.join(RESULTS_PATH, f'{model_name}_{mode}_history.png'))
        plot_confusion_matrices(y_true, y_pred, class_names, save_path=os.path.join(RESULTS_PATH, f'{model_name}_{mode}_confusion.png'))
        plot_roc_curves(y_true, y_scores, class_names, save_path=os.path.join(RESULTS_PATH, f'{model_name}_{mode}_roc.png'))
        plot_precision_recall_curves(y_true, y_scores, class_names, save_path=os.path.join(RESULTS_PATH, f'{model_name}_{mode}_pr.png'))
        plot_confusion_matrix_all_classes(y_true, y_pred, class_names, save_path=os.path.join(RESULTS_PATH, f'{model_name}_{mode}_confusion_all_classes.png'))

    
    # 7. Return results
    results = {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores
        },
        'model_path': save_path,
        'config': {
            'model_name': model_name,
            'mode': mode,
            'swin_model_name': swin_model_name if model_name in ['swin', 'swin_fusion'] else None,
            'fusion_type': fusion_type if model_name == 'swin_fusion' else None
        }
    }
    
    return results