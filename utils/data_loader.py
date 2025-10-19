import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from .data_transforms import get_transforms

# ============================================================================
# FLEXIBLE DATASET CLASS FOR PRECOMPUTED SCALOGRAMS AND PHASOGRAMS
# ============================================================================

class PrecomputedDataset(Dataset):
    """
    Flexible dataset for loading pre-saved scalogram and/or phasogram files.
    
    Supports multiple modes:
    - 'composite_scalogram': RGB composite scalogram only
    - 'composite_phasogram': RGB composite phasogram only
    - 'composite_both': Both composite scalogram and phasogram (stacked)
    - 'lead2_scalogram': Lead II scalogram only
    - 'lead2_phasogram': Lead II phasogram only
    - 'lead2_both': Both Lead II scalogram and phasogram (stacked)
    """
    
    def __init__(self, df, labels, mode='composite_scalogram', transform=None):
        """
        Args:
            df: DataFrame with path columns:
                - 'composite_scalogram_path'
                - 'composite_phasogram_path'
                - 'lead2_scalogram_path'
                - 'lead2_phasogram_path'
            labels: Binary label array (N, num_classes)
            mode: One of ['composite_scalogram', 'composite_phasogram', 'composite_both',
                         'lead2_scalogram', 'lead2_phasogram', 'lead2_both']
            transform: torchvision transforms (applied to each image independently)
        """
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
        # Validate mode
        valid_modes = [
            'composite_scalogram', 'composite_phasogram', 'composite_both',
            'lead2_scalogram', 'lead2_phasogram', 'lead2_both'
        ]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got '{mode}'")
        
        # Set up paths based on mode
        if mode == 'composite_scalogram':
            self.scalogram_paths = df['composite_scalogram_path'].values
            self.phasogram_paths = None
            self.num_channels = 3
            
        elif mode == 'composite_phasogram':
            self.scalogram_paths = None
            self.phasogram_paths = df['composite_phasogram_path'].values
            self.num_channels = 3
            
        elif mode == 'composite_both':
            self.scalogram_paths = df['composite_scalogram_path'].values
            self.phasogram_paths = df['composite_phasogram_path'].values
            self.num_channels = 6
            
        elif mode == 'lead2_scalogram':
            self.scalogram_paths = df['lead2_scalogram_path'].values
            self.phasogram_paths = None
            self.num_channels = 3
            
        elif mode == 'lead2_phasogram':
            self.scalogram_paths = None
            self.phasogram_paths = df['lead2_phasogram_path'].values
            self.num_channels = 3
            
        elif mode == 'lead2_both':
            self.scalogram_paths = df['lead2_scalogram_path'].values
            self.phasogram_paths = df['lead2_phasogram_path'].values
            self.num_channels = 6
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        images = []
        
        # Load scalogram if needed
        if self.scalogram_paths is not None:
            scalogram = np.load(self.scalogram_paths[idx])
            images.append(scalogram)
        
        # Load phasogram if needed
        if self.phasogram_paths is not None:
            phasogram = np.load(self.phasogram_paths[idx])
            images.append(phasogram)
        
        # Process based on mode
        if len(images) == 1:
            # Single image mode (scalogram OR phasogram only)
            image = images[0]
            
            if self.transform:
                # Convert to PIL for transforms
                image_pil = Image.fromarray(image)
                image = self.transform(image_pil)
            else:
                # Convert to tensor manually
                image = torch.from_numpy(image).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC -> CHW
        
        else:
            # Both scalogram and phasogram - concatenate along channel dimension
            if self.transform:
                # Apply transform to each image separately
                transformed = []
                for img in images:
                    img_pil = Image.fromarray(img)
                    img_tensor = self.transform(img_pil)
                    transformed.append(img_tensor)
                
                # Concatenate along channel dimension
                image = torch.cat(transformed, dim=0)  # (6, H, W)
            
            else:
                # Convert both to tensors and concatenate
                tensors = []
                for img in images:
                    tensor = torch.from_numpy(img).float() / 255.0
                    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
                    tensors.append(tensor)
                
                image = torch.cat(tensors, dim=0)  # (6, H, W)
        
        label = torch.from_numpy(self.labels[idx]).float()
        
        return image, label
    
    def get_num_channels(self):
        """Return the number of input channels for this dataset configuration"""
        return self.num_channels


def create_dataloaders(train_df, y_train, val_df, y_val, test_df, y_test,
                    mode='composite_scalogram', batch_size=16, num_workers=4, 
                    image_size=224, augment=True):
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_df, val_df, test_df: DataFrames with image paths
        y_train, y_val, y_test: Label arrays
        mode: Dataset mode - one of:
            - 'composite_scalogram': RGB composite scalogram only
            - 'composite_phasogram': RGB composite phasogram only
            - 'composite_both': Both composite (6 channels)
            - 'lead2_scalogram': Lead II scalogram only
            - 'lead2_phasogram': Lead II phasogram only
            - 'lead2_both': Both Lead II (6 channels)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Image size for transforms
        augment: Whether to apply augmentation to training data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, val_transform = get_transforms(image_size=image_size, augment=augment)
    
    # Create datasets with specified mode
    train_dataset = PrecomputedDataset(
        train_df, y_train, mode=mode, transform=train_transform
    )
    val_dataset = PrecomputedDataset(
        val_df, y_val, mode=mode, transform=val_transform
    )
    test_dataset = PrecomputedDataset(
        test_df, y_test, mode=mode, transform=val_transform
    )
    
    # Get number of channels for model configuration
    num_channels = train_dataset.get_num_channels()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Dataloaders created with mode='{mode}' ({num_channels} channels):")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader