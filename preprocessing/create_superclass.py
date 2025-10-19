import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Define the path to the dataset
def create_superclass_labels(data_path, weight_threshold=0.5, min_count=10):
    """Create superclass labels and remove combinations with less than min_count samples"""
    
    # Load data
    print("Loading PTB-XL data...")
    data = pd.read_csv(data_path + 'ptbxl_database.csv', index_col='ecg_id')
    data.scp_codes = data.scp_codes.apply(lambda x: ast.literal_eval(x))
    print(f"Original dataset size: {len(data)}")
    
    # Load SCP statements
    scp_statements = pd.read_csv(data_path + 'scp_statements.csv', index_col=0)
    
    # Define superclass mapping
    superclass_mapping = {
        'NORM': 'Normal',
        'MI': 'Myocardial_Infarction',
        'STTC': 'ST_T_Changes',
        'CD': 'Conduction_Disturbance', 
        'HYP': 'Hypertrophy'
    }
    
    def map_to_superclasses(scp_codes):
        superclasses = []
        for code, weight in scp_codes.items():
            if weight >= weight_threshold and code in scp_statements.index:
                superclass = scp_statements.loc[code, 'diagnostic_class']
                if pd.notna(superclass) and superclass in superclass_mapping:
                    superclasses.append(superclass_mapping[superclass])
        return list(set(superclasses))
    
    # Apply superclass mapping
    data['superclass_labels'] = data.scp_codes.apply(map_to_superclasses)
    
    # Remove samples with no valid superclass labels
    data = data[data['superclass_labels'].apply(len) > 0].copy()
    print(f"Samples with valid superclass labels: {len(data)}")
    
    # Count each label combination
    label_counts = data['superclass_labels'].apply(tuple).value_counts()
    print(f"\nUnique label combinations: {len(label_counts)}")
    
    # Identify combinations to keep (those with count >= min_count)
    valid_combinations = label_counts[label_counts >= min_count].index.tolist()
    invalid_combinations = label_counts[label_counts < min_count].index.tolist()
    
    print(f"Valid combinations (count >= {min_count}): {len(valid_combinations)}")
    print(f"Invalid combinations (count < {min_count}): {len(invalid_combinations)}")
    
    # Show what will be removed
    if invalid_combinations:
        print(f"\nRemoving the following combinations (count < {min_count}):")
        for combo in invalid_combinations:
            count = label_counts[combo]
            print(f"  {list(combo)}: {count} samples")
    
    # Filter data to only include valid combinations
    original_size = len(data)
    data = data[data['superclass_labels'].apply(tuple).isin(valid_combinations)].copy()
    removed_count = original_size - len(data)
    
    print(f"\nFiltered out {removed_count} samples with rare label combinations")
    print(f"Remaining samples: {len(data)}")
    
    # Create binary labels
    mlb = MultiLabelBinarizer()
    superclass_binary = mlb.fit_transform(data['superclass_labels'])
    
    print(f"\nFinal superclasses ({len(mlb.classes_)}): {list(mlb.classes_)}")
    
    # Print class distribution
    class_counts = superclass_binary.sum(axis=0)
    print("\nClass distribution:")
    for i, class_name in enumerate(mlb.classes_):
        print(f"{class_name}: {class_counts[i]} samples ({class_counts[i]/len(data)*100:.1f}%)")
    
    # Show final label combination counts
    final_counts = data['superclass_labels'].apply(tuple).value_counts()
    print(f"\nFinal label combinations ({len(final_counts)}):")
    for combo, count in final_counts.items():
        print(f"  {list(combo)}: {count} samples")
    
    return data, superclass_binary, mlb