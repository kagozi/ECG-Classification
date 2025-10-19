# main.py
import os
import numpy as np
from create_superclass import create_superclass_labels
from constants import DATA_PATH, composite_scalogram_path_for, lead2_scalogram_path_for, lead2_phasogram_path_for, composite_phasogram_path_for


if __name__ == '__main__':
    data, superclass_labels, mlb = create_superclass_labels(DATA_PATH, weight_threshold=0.5, min_count=10)
    category_counts = data['superclass_labels'].value_counts()
    print(category_counts)
        
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
    train_fold, val_fold, test_fold = 8, 9, 10

    print("\nUsing fold-based data split...")
    train_df = data[data.strat_fold < train_fold]
    y_train = superclass_labels[data.strat_fold < train_fold]

    val_df = data[data.strat_fold == val_fold]
    y_val = superclass_labels[data.strat_fold == val_fold]

    test_df = data[data.strat_fold == test_fold]
    y_test = superclass_labels[data.strat_fold == test_fold]

    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} samples, {y_train.shape}")
    print(f"Val:   {len(val_df)} samples, {y_val.shape}")  
    print(f"Test:  {len(test_df)} samples, {y_test.shape}")
    
    
    
    
    
    