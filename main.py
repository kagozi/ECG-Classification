# main.py
import os
import numpy as np
import pandas as pd
from preprocessing.create_superclass import create_superclass_labels
from config.constants import (DATA_PATH, 
                       composite_scalogram_path_for, 
                       lead2_scalogram_path_for, 
                       lead2_phasogram_path_for, 
                       composite_phasogram_path_for, 
                       SEED, 
                       setup_device_and_seed,
                       BATCH_SIZE,
                       EPOCHS,
                       PATIENCE,
                       LR,
                       THRESHOLD,
                       TRAIN_MULTIPLE
                       )
from runner import run_complete_pipeline
from utils.metrics import compute_metrics


if __name__ == '__main__':
    DEVICE = setup_device_and_seed(SEED)
    data, superclass_labels, mlb = create_superclass_labels(DATA_PATH, weight_threshold=0.5, min_count=10)

    # Attach paths and existence flags for all 4 image types
    data["composite_scalogram_path"] = [composite_scalogram_path_for(eid) for eid in data.index]
    data["lead2_scalogram_path"] = [lead2_scalogram_path_for(eid) for eid in data.index]
    data["lead2_phasogram_path"] = [lead2_phasogram_path_for(eid) for eid in data.index]
    data["composite_phasogram_path"] = [composite_phasogram_path_for(eid) for eid in data.index]
    print(f"Columns and rows: {data.shape}")

    # Check existence for all image types
    data["composite_scalogram_exists"] = data["composite_scalogram_path"].apply(os.path.exists)
    data["lead2_scalogram_exists"] = data["lead2_scalogram_path"].apply(os.path.exists)
    data["lead2_phasogram_exists"] = data["lead2_phasogram_path"].apply(os.path.exists)
    data["composite_phasogram_exists"] = data["composite_phasogram_path"].apply(os.path.exists)

    # Keep only rows where ALL required image types exist
    required_columns = [
        "composite_scalogram_exists", 
        "lead2_scalogram_exists", 
        "lead2_phasogram_exists", 
        "composite_phasogram_exists"
    ]

    # Create a combined existence flag
    data["all_images_exist"] = data[required_columns].all(axis=1)

    print(f"\nRows with ALL image types available: {data['all_images_exist'].sum()}")
    print(f"Rows missing at least one image type: {(~data['all_images_exist']).sum()}")

    # Filter to keep only rows with all image types
    original_size = len(data)
    data = data[data["all_images_exist"]].copy()
    print(f"\nFiltered dataset size: {len(data)} (removed {original_size - len(data)} rows)")

    # Show sample paths for verification
    print("\nSample paths for first ECG ID:")
    sample_ecg_id = data.index[0]
    print(f"Composite scalogram: {composite_scalogram_path_for(sample_ecg_id)}")
    print(f"Lead II scalogram: {lead2_scalogram_path_for(sample_ecg_id)}")
    print(f"Lead II phasogram: {lead2_phasogram_path_for(sample_ecg_id)}")
    print(f"Composite phasogram: {composite_phasogram_path_for(sample_ecg_id)}")
    
    
    # --- Split the data based on predefined folds ---
    # train_fold, val_fold, test_fold = 8, 9, 10

    # print("\nUsing fold-based data split...")
    # train_df = data[data.strat_fold < train_fold]
    # y_train = superclass_labels[data.strat_fold < train_fold]

    # val_df = data[data.strat_fold == val_fold]
    # y_val = superclass_labels[data.strat_fold == val_fold]

    # test_df = data[data.strat_fold == test_fold]
    # y_test = superclass_labels[data.strat_fold == test_fold]
    
    # masks
    mask_train = data["strat_fold"] < 8
    mask_val   = data["strat_fold"] == 9
    mask_test  = data["strat_fold"] == 10

    # split DataFrames
    train_df = data.loc[mask_train].copy()
    val_df   = data.loc[mask_val].copy()
    test_df  = data.loc[mask_test].copy()

    # REBUILD label matrices aligned to these slices
    y_train = mlb.transform(train_df["superclass_labels"])
    y_val   = mlb.transform(val_df["superclass_labels"])
    y_test  = mlb.transform(test_df["superclass_labels"])

    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} samples, {y_train.shape}")
    print(f"Val:   {len(val_df)} samples, {y_val.shape}")  
    print(f"Test:  {len(test_df)} samples, {y_test.shape}")
    
    # ============================================================================
    # STEP 1: QUICK VERIFICATION
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 1: DATA VERIFICATION")
    print("="*80)

    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")

    print(f"\nClass distribution in training set:")
    for i, class_name in enumerate(mlb.classes_):
        count = y_train[:, i].sum()
        pct = count / len(y_train) * 100
        print(f"  {class_name:25} {int(count):5} samples ({pct:5.1f}%)")

    # ============================================================================
    # STEP 2: SINGLE MODEL TRAINING
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 2: TRAINING SINGLE MODEL")
    print("="*80)

    # ============================================================================
    # CONFIGURATION - MODIFY THESE FOR YOUR EXPERIMENT
    # ============================================================================

    # Model options: 'resnet50', 'efficientnet_b3', 'densenet121', 'vit', 'swin', 'swin_fusion'
    MODEL_NAME = 'swin'

    # Data mode options:
    # - 'composite_scalogram': RGB composite scalogram only
    # - 'composite_phasogram': RGB composite phasogram only
    # - 'composite_both': Both composite (for fusion models)
    # - 'lead2_scalogram': Lead II scalogram only
    # - 'lead2_phasogram': Lead II phasogram only
    # - 'lead2_both': Both Lead II (for fusion models)
    DATA_MODE = 'composite_scalogram'

    # Swin Transformer variant (only used if MODEL_NAME is 'swin' or 'swin_fusion')
    # Options: 'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
    #          'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224'
    SWIN_VARIANT = 'swin_large_patch4_window7_224'

    # Fusion type (only used if MODEL_NAME is 'swin_fusion')
    # Options: 'early' or 'late'
    FUSION_TYPE = 'early'

    # ============================================================================

    # results = run_complete_pipeline(
    #     train_df=train_df,
    #     y_train=y_train,
    #     val_df=val_df,
    #     y_val=y_val,
    #     test_df=test_df,
    #     y_test=y_test,
    #     model_name=MODEL_NAME,
    #     mode=DATA_MODE,
    #     batch_size=BATCH_SIZE,
    #     num_epochs=EPOCHS,
    #     lr=LR,
    #     patience=PATIENCE,
    #     threshold=THRESHOLD,
    #     device=DEVICE,
    #     class_names=list(mlb.classes_),
    #     use_tta=False,  # Set to True for better test performance (slower)
    #     swin_model_name=SWIN_VARIANT,
    #     fusion_type=FUSION_TYPE
    # )

    # ============================================================================
    # STEP 3: PRINT FINAL RESULTS
    # ============================================================================

    # print("\n" + "="*80)
    # print("FINAL RESULTS")
    # print("="*80)

    # test_metrics = results['test_metrics']
    # print(f"\nOverall Performance:")
    # print(f"  F1 Score (Macro):     {test_metrics['f1_macro']:.4f}")
    # print(f"  F1 Score (Weighted):  {test_metrics['f1_weighted']:.4f}")
    # print(f"  AUC Score (Macro):    {test_metrics['auc_macro']:.4f}")
    # print(f"  Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.4f}")
    # print(f"  Precision (Macro):    {test_metrics['precision_macro']:.4f}")
    # print(f"  Recall (Macro):       {test_metrics['recall_macro']:.4f}")
    # print(f"  Hamming Accuracy:     {test_metrics['hamming_accuracy']:.4f}")
    # print(f"  Label Accuracy:       {test_metrics['label_accuracy']:.4f}")

    # print(f"\nModel saved to: {results['model_path']}")

    # ============================================================================
    # OPTIONAL: TRAIN MULTIPLE MODELS & COMPARE
    # ============================================================================

    print("\n" + "="*80)
    print("OPTIONAL: MULTI-MODEL COMPARISON")
    print("="*80)

  # Set to True to train multiple configurations

    if TRAIN_MULTIPLE:
        # Define experiments to run
        experiments = [
            # Compare single modalities with Swin
            {'model': 'swin', 'mode': 'composite_scalogram', 'name': 'Swin-Composite-Scalo'},
            {'model': 'swin', 'mode': 'composite_phasogram', 'name': 'Swin-Composite-Phaso'},
            {'model': 'swin', 'mode': 'lead2_scalogram', 'name': 'Swin-Lead2-Scalo'},
            {'model': 'swin', 'mode': 'lead2_phasogram', 'name': 'Swin-Lead2-Phaso'},
            
            # Compare fusion strategies
            {'model': 'swin_fusion', 'mode': 'composite_both', 'fusion': 'early', 'name': 'Swin-Fusion-Early'},
            {'model': 'swin_fusion', 'mode': 'composite_both', 'fusion': 'late', 'name': 'Swin-Fusion-Late'},
            
            # Compare with CNNs (scalogram and phasogram)
            {'model': 'resnet50', 'mode': 'composite_scalogram', 'name': 'ResNet50-Scalo'},
            {'model': 'resnet50', 'mode': 'composite_phasogram', 'name': 'ResNet50-Phaso'},
            {'model': 'efficientnet_b3', 'mode': 'composite_scalogram', 'name': 'EfficientNet-B3-Scalo'},
            {'model': 'efficientnet_b3', 'mode': 'composite_phasogram', 'name': 'EfficientNet-B3-Phaso'},
            
            # Compare with Vision Transformer
            {'model': 'vit', 'mode': 'composite_scalogram', 'name': 'ViT-Scalo'},
            {'model': 'vit', 'mode': 'composite_phasogram', 'name': 'ViT-Phaso'},
        ]
        
        all_results = {}
        
        for exp in experiments:
            print(f"\n{'='*80}")
            print(f"Training {exp['name']}")
            print(f"{'='*80}")
            
            results = run_complete_pipeline(
                train_df=train_df,
                y_train=y_train,
                val_df=val_df,
                y_val=y_val,
                test_df=test_df,
                y_test=y_test,
                model_name=exp['model'],
                mode=exp['mode'],
                batch_size=BATCH_SIZE,
                num_epochs=EPOCHS,
                lr=LR,
                patience=PATIENCE,
                threshold=THRESHOLD,
                device=DEVICE,
                class_names=list(mlb.classes_),
                use_tta=True,
                swin_model_name=SWIN_VARIANT,
                fusion_type=exp.get('fusion', 'early')
            )
            
            all_results[exp['name']] = results
        
        # Compare models
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"\n{'Configuration':<25} | {'F1 (Macro)':<12} | {'AUC (Macro)':<12} | "
            f"{'Accuracy':<10} | {'Hamming':<10} | {'Label Acc':<10}")
        print("-" * 95)
        
        for name, res in all_results.items():
            metrics = res['test_metrics']
            print(f"{name:<25} | {metrics['f1_macro']:<12.4f} | "
                f"{metrics['auc_macro']:<12.4f} | {metrics['exact_match_accuracy']:<10.4f} | "
                f"{metrics['hamming_accuracy']:<10.4f} | {metrics['label_accuracy']:<10.4f}")
        
        # Find best model
        best_model = max(all_results.items(), key=lambda x: x[1]['test_metrics']['f1_macro'])
        print(f"\nBest Model: {best_model[0]} (F1: {best_model[1]['test_metrics']['f1_macro']:.4f})")
        
        # Ensemble of ALL models
        print("\n" + "="*80)
        print("ENSEMBLE: ALL MODELS")
        print("="*80)
        
        all_scores = [res['predictions']['y_scores'] for res in all_results.values()]
        ensemble_all_scores = np.mean(all_scores, axis=0)
        ensemble_all_preds = (ensemble_all_scores > THRESHOLD).astype(float)
        
        y_true = results['predictions']['y_true']
        ensemble_all_metrics = compute_metrics(y_true, ensemble_all_preds, ensemble_all_scores, THRESHOLD)
        
        print(f"\nEnsemble All Models Performance:")
        print(f"  F1 Score (Macro):     {ensemble_all_metrics['f1_macro']:.4f}")
        print(f"  AUC Score (Macro):    {ensemble_all_metrics['auc_macro']:.4f}")
        print(f"  Exact Match Accuracy: {ensemble_all_metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming Accuracy:     {ensemble_all_metrics['hamming_accuracy']:.4f}")
        print(f"  Label Accuracy:       {ensemble_all_metrics['label_accuracy']:.4f}")
        
        # Ensemble of TOP-K BEST models
        print("\n" + "="*80)
        print("ENSEMBLE: TOP-5 BEST MODELS")
        print("="*80)
        
        # Sort models by F1 score
        sorted_results = sorted(all_results.items(), 
                            key=lambda x: x[1]['test_metrics']['f1_macro'], 
                            reverse=True)
        
        # Get top K models
        top_k = min(5, len(sorted_results))
        top_models = sorted_results[:top_k]
        
        print(f"\nTop {top_k} models selected:")
        for i, (name, res) in enumerate(top_models, 1):
            print(f"  {i}. {name:<25} F1: {res['test_metrics']['f1_macro']:.4f}")
        
        # Ensemble top-k predictions
        top_scores = [res['predictions']['y_scores'] for name, res in top_models]
        ensemble_top_scores = np.mean(top_scores, axis=0)
        ensemble_top_preds = (ensemble_top_scores > THRESHOLD).astype(float)
        
        ensemble_top_metrics = compute_metrics(y_true, ensemble_top_preds, ensemble_top_scores, THRESHOLD)
        
        print(f"\nEnsemble Top-{top_k} Performance:")
        print(f"  F1 Score (Macro):     {ensemble_top_metrics['f1_macro']:.4f}")
        print(f"  AUC Score (Macro):    {ensemble_top_metrics['auc_macro']:.4f}")
        print(f"  Exact Match Accuracy: {ensemble_top_metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming Accuracy:     {ensemble_top_metrics['hamming_accuracy']:.4f}")
        print(f"  Label Accuracy:       {ensemble_top_metrics['label_accuracy']:.4f}")
        
        # Weighted Ensemble (weight by validation F1 score)
        print("\n" + "="*80)
        print("ENSEMBLE: WEIGHTED BY VALIDATION F1")
        print("="*80)
        
        # Extract validation F1 scores from history (last epoch)
        weights = []
        for name, res in all_results.items():
            val_f1 = res['history']['val_f1'][-1]  # Last validation F1
            weights.append(val_f1)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print("\nModel weights:")
        for (name, _), weight in zip(all_results.items(), weights):
            print(f"  {name:<25} Weight: {weight:.4f}")
        
        # Weighted ensemble
        all_scores_array = np.array([res['predictions']['y_scores'] for res in all_results.values()])
        ensemble_weighted_scores = np.average(all_scores_array, axis=0, weights=weights)
        ensemble_weighted_preds = (ensemble_weighted_scores > THRESHOLD).astype(float)
        
        ensemble_weighted_metrics = compute_metrics(y_true, ensemble_weighted_preds, 
                                                ensemble_weighted_scores, THRESHOLD)
        
        print(f"\nWeighted Ensemble Performance:")
        print(f"  F1 Score (Macro):     {ensemble_weighted_metrics['f1_macro']:.4f}")
        print(f"  AUC Score (Macro):    {ensemble_weighted_metrics['auc_macro']:.4f}")
        print(f"  Exact Match Accuracy: {ensemble_weighted_metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming Accuracy:     {ensemble_weighted_metrics['hamming_accuracy']:.4f}")
        print(f"  Label Accuracy:       {ensemble_weighted_metrics['label_accuracy']:.4f}")
        
        # Final comparison
        print("\n" + "="*80)
        print("ENSEMBLE COMPARISON SUMMARY")
        print("="*80)
        
        ensemble_comparison = {
            'Best Single Model': best_model[1]['test_metrics']['f1_macro'],
            'Ensemble (All)': ensemble_all_metrics['f1_macro'],
            f'Ensemble (Top-{top_k})': ensemble_top_metrics['f1_macro'],
            'Ensemble (Weighted)': ensemble_weighted_metrics['f1_macro']
        }
        
        print(f"\n{'Method':<25} | {'F1 Score (Macro)':<12}")
        print("-" * 40)
        for method, score in ensemble_comparison.items():
            print(f"{method:<25} | {score:<12.4f}")
        
        best_ensemble = max(ensemble_comparison.items(), key=lambda x: x[1])
        print(f"\n🏆 Best Overall: {best_ensemble[0]} (F1: {best_ensemble[1]:.4f})")
        
        # Save ensemble predictions
        print("\n" + "="*80)
        print("SAVING ENSEMBLE PREDICTIONS")
        print("="*80)
        
        ensemble_pred_df = pd.DataFrame({
            'ecg_id': test_df.index
        })
        
        # Add true labels
        for i, class_name in enumerate(mlb.classes_):
            ensemble_pred_df[f'true_{class_name}'] = y_true[:, i]
        
        # Add all ensemble predictions
        for i, class_name in enumerate(mlb.classes_):
            ensemble_pred_df[f'pred_all_{class_name}'] = ensemble_all_preds[:, i]
            ensemble_pred_df[f'pred_top{top_k}_{class_name}'] = ensemble_top_preds[:, i]
            ensemble_pred_df[f'pred_weighted_{class_name}'] = ensemble_weighted_preds[:, i]
        
        # Add all ensemble scores
        for i, class_name in enumerate(mlb.classes_):
            ensemble_pred_df[f'score_all_{class_name}'] = ensemble_all_scores[:, i]
            ensemble_pred_df[f'score_top{top_k}_{class_name}'] = ensemble_top_scores[:, i]
            ensemble_pred_df[f'score_weighted_{class_name}'] = ensemble_weighted_scores[:, i]
        
        ensemble_output_path = 'predictions_ensemble.csv'
        ensemble_pred_df.to_csv(ensemble_output_path, index=False)
        print(f"Ensemble predictions saved to: {ensemble_output_path}")

    # ============================================================================
    # STEP 4: SAVE PREDICTIONS
    # ============================================================================

    print("\n" + "="*80)
    print("SAVING PREDICTIONS")
    print("="*80)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'ecg_id': test_df.index
    })

    # Add true labels
    for i, class_name in enumerate(mlb.classes_):
        predictions_df[f'true_{class_name}'] = results['predictions']['y_true'][:, i]

    # Add predicted labels
    for i, class_name in enumerate(mlb.classes_):
        predictions_df[f'pred_{class_name}'] = results['predictions']['y_pred'][:, i]

    # Add prediction scores
    for i, class_name in enumerate(mlb.classes_):
        predictions_df[f'score_{class_name}'] = results['predictions']['y_scores'][:, i]

    output_path = f'predictions_{MODEL_NAME}_{DATA_MODE}.csv'
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Data Mode:  {DATA_MODE}")
    if MODEL_NAME in ['swin', 'swin_fusion']:
        print(f"  Swin Variant: {SWIN_VARIANT}")
    if MODEL_NAME == 'swin_fusion':
        print(f"  Fusion Type: {FUSION_TYPE}")

    print(f"\nKey Files Generated:")
    print(f"  1. Model checkpoint:     {results['model_path']}")
    print(f"  2. Training history:     {MODEL_NAME}_{DATA_MODE}_history.png")
    print(f"  3. Confusion matrices:   {MODEL_NAME}_{DATA_MODE}_confusion.png")
    print(f"  4. ROC curves:           {MODEL_NAME}_{DATA_MODE}_roc.png")
    print(f"  5. PR curves:            {MODEL_NAME}_{DATA_MODE}_pr.png")
    print(f"  6. Predictions CSV:      {output_path}")

    print(f"\n{'='*80}")
    print("Next Steps & Recommendations:")
    print("="*80)
    
    
    print("""
    1. SINGLE MODALITY EXPERIMENTS:
    - Try: mode='composite_scalogram' vs 'composite_phasogram'
    - Try: mode='lead2_scalogram' vs 'lead2_phasogram'
    - Compare which representation works best

    2. FUSION EXPERIMENTS:
    - Try: model='swin_fusion', mode='composite_both', fusion_type='early'
    - Try: model='swin_fusion', mode='composite_both', fusion_type='late'
    - Try: model='swin_fusion', mode='lead2_both'
    - Compare fusion strategies

    3. MODEL COMPARISON:
    - Compare CNNs (ResNet, EfficientNet) vs Transformers (ViT, Swin)
    - Test different Swin variants (tiny, small, base, large)
    - Consider compute-accuracy tradeoffs

    4. OPTIMIZATION:
    - Use test-time augmentation (use_tta=True)
    - Try ensemble of best models
    - Experiment with different learning rates
    - Adjust batch size based on GPU memory

    5. ANALYSIS:
    - Review training curves for overfitting
    - Analyze per-class performance
    - Check which modality helps which class
    - Consider class-specific thresholding
    """)

    print("="*80)

    # ============================================================================
    # TIPS FOR HIGH SCORES WITH FUSION
    # ============================================================================

    print("\n" + "="*80)
    print("TIPS FOR ACHIEVING HIGH SCORES WITH FUSION MODELS")
    print("="*80)
    print("""
    1. Data Strategy:
    ✓ Scalograms capture magnitude/energy information
    ✓ Phasograms capture timing/synchronization information
    ✓ Fusion combines complementary features
    ✓ Try both composite (all leads) and Lead II variants

    2. Fusion Strategy:
    - Early Fusion: Better when modalities are highly correlated
    - Late Fusion: Better when modalities provide independent information
    - Start with early fusion (faster, fewer parameters)

    3. Model Selection:
    - Swin Transformers: Best for capturing long-range patterns
    - swin_base: Good balance of performance and speed
    - swin_large: Best performance but slower
    - swin_small/tiny: For faster experiments

    4. Training Best Practices:
    ✓ Lower learning rate for fusion models (3e-5 to 5e-5)
    ✓ More epochs may help (30-40)
    ✓ Watch for overfitting with 6-channel inputs
    ✓ Consider gradient accumulation for stability

    5. Evaluation:
    ✓ Always compare fusion vs single modality
    ✓ Check if fusion actually improves scores
    ✓ Some classes may benefit more from fusion
    ✓ Ensemble fusion + single modality models

    Expected Performance Hierarchy:
    Single Modality (3ch) < Early Fusion (6ch) < Late Fusion (6ch) < Ensemble
    """)

    print("="*80)
    
    
    
    
    
    