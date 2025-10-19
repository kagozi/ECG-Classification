import torch
import torch.nn as nn
import timm

# ============================================================================
# MODEL 1: ResNet50 with Custom Head
# ============================================================================

class ResNet50ECG(nn.Module):
    """ResNet50 with custom classification head for multi-label ECG"""
    
    def __init__(self, num_classes=5, dropout=0.5, pretrained=True):
        super(ResNet50ECG, self).__init__()
        
        # Load pretrained ResNet50
        from torchvision import models
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove original FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============================================================================
# MODEL 2: EfficientNet-B3 with Custom Head
# ============================================================================

class EfficientNetB3ECG(nn.Module):
    """EfficientNet-B3 for ECG classification"""
    
    def __init__(self, num_classes=5, dropout=0.5, pretrained=True):
        super(EfficientNetB3ECG, self).__init__()
        
        # Load pretrained EfficientNet-B3
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# MODEL 3: DenseNet121 with Custom Head
# ============================================================================

class DenseNet121ECG(nn.Module):
    """DenseNet121 for ECG classification"""
    
    def __init__(self, num_classes=5, dropout=0.5, pretrained=True):
        super(DenseNet121ECG, self).__init__()
        
        # Load pretrained DenseNet121
        from torchvision import models
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get number of features
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# MODEL 4: Vision Transformer (ViT) with Custom Head
# ============================================================================

class ViTECG(nn.Module):
    """Vision Transformer for ECG classification"""
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True):
        super(ViTECG, self).__init__()
        
        # Load pretrained ViT-B/16
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None)
        
        # Get number of features
        num_features = self.backbone.heads.head.in_features
        
        # Replace head
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# MODEL 5: Swin Transformer - Single Modality (Scalogram OR Phasogram)
# ============================================================================

class SwinTransformerECG(nn.Module):
    """
    Swin Transformer for ECG scalogram or phasogram classification.
    Works with either 3-channel (single modality) inputs.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, 
                 model_name='swin_base_patch4_window7_224'):
        super(SwinTransformerECG, self).__init__()
        
        # Load pretrained Swin Transformer from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            in_chans=3      # Standard 3-channel input
        )
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============================================================================
# MODEL 6: Swin Transformer - Fusion (Scalogram AND Phasogram)
# ============================================================================

class SwinTransformerFusionECG(nn.Module):
    """
    Swin Transformer with fusion for combined scalogram and phasogram analysis.
    Handles 6-channel inputs (3 from scalogram + 3 from phasogram).
    
    Two fusion strategies:
    1. 'early': Concatenate channels at input (6-channel input to single backbone)
    2. 'late': Two separate backbones with feature fusion
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224', fusion_type='early'):
        super(SwinTransformerFusionECG, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'early':
            # Single backbone with 6 input channels
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=6  # 6 channels for concatenated scalogram + phasogram
            )
            
            num_features = self.backbone.num_features
            
            # Single classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(dropout / 2),
                nn.Linear(512, num_classes)
            )
        
        elif fusion_type == 'late':
            # Two separate backbones (one for scalogram, one for phasogram)
            self.backbone_scalogram = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=3
            )
            
            self.backbone_phasogram = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=3
            )
            
            num_features = self.backbone_scalogram.num_features
            
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(num_features * 2, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),
                nn.Dropout(dropout)
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(dropout / 2),
                nn.Linear(512, num_classes)
            )
        
        else:
            raise ValueError(f"fusion_type must be 'early' or 'late', got {fusion_type}")
    
    def forward(self, x):
        if self.fusion_type == 'early':
            # x shape: (batch, 6, H, W)
            features = self.backbone(x)
            output = self.classifier(features)
        
        else:  # late fusion
            # Split input into scalogram and phasogram
            # x shape: (batch, 6, H, W)
            scalogram = x[:, :3, :, :]  # First 3 channels
            phasogram = x[:, 3:, :, :]  # Last 3 channels
            
            # Extract features separately
            features_scalo = self.backbone_scalogram(scalogram)
            features_phaso = self.backbone_phasogram(phasogram)
            
            # Concatenate features
            combined_features = torch.cat([features_scalo, features_phaso], dim=1)
            
            # Fusion and classification
            fused = self.fusion(combined_features)
            output = self.classifier(fused)
        
        return output


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Single modality (scalogram or phasogram only)
# Use with mode='composite_scalogram' or 'composite_phasogram' or 'lead2_scalogram' or 'lead2_phasogram'
model = SwinTransformerECG(
    num_classes=5,
    dropout=0.3,
    pretrained=True,
    model_name='swin_base_patch4_window7_224'  # or 'swin_small_patch4_window7_224', 'swin_large_patch4_window7_224'
)

# Example 2: Early fusion (6-channel input)
# Use with mode='composite_both' or 'lead2_both'
model = SwinTransformerFusionECG(
    num_classes=5,
    dropout=0.3,
    pretrained=True,
    model_name='swin_base_patch4_window7_224',
    fusion_type='early'
)

# Example 3: Late fusion (two separate backbones)
# Use with mode='composite_both' or 'lead2_both'
model = SwinTransformerFusionECG(
    num_classes=5,
    dropout=0.3,
    pretrained=True,
    model_name='swin_base_patch4_window7_224',
    fusion_type='late'
)

# Available Swin Transformer variants in timm:
# - 'swin_tiny_patch4_window7_224'    : Smallest, fastest
# - 'swin_small_patch4_window7_224'   : Small model
# - 'swin_base_patch4_window7_224'    : Balanced (recommended)
# - 'swin_large_patch4_window7_224'   : Large model, best performance
# - 'swin_base_patch4_window12_384'   : For 384x384 images
"""