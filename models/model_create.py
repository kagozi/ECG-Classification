from .models import ResNet50ECG, EfficientNetB3ECG, DenseNet121ECG, ViTECG, SwinTransformerECG, SwinTransformerFusionECG
# ============================================================================
# MODEL FACTORY
# ============================================================================
def create_model(model_name='resnet50', num_classes=5, dropout=0.5, pretrained=True, 
                 swin_model_name='swin_base_patch4_window7_224', fusion_type='early'):
    """
    Factory function to create models
    
    Args:
        model_name: 'resnet50', 'efficientnet_b3', 'densenet121', 'vit', 'swin', or 'swin_fusion'
        num_classes: Number of output classes
        dropout: Dropout rate
        pretrained: Use pretrained weights
        swin_model_name: Specific Swin variant (only for 'swin' and 'swin_fusion')
        fusion_type: 'early' or 'late' (only for 'swin_fusion')
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet50':
        model = ResNet50ECG(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    
    elif model_name == 'efficientnet_b3':
        model = EfficientNetB3ECG(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    
    elif model_name == 'densenet121':
        model = DenseNet121ECG(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    
    elif model_name == 'vit':
        model = ViTECG(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    
    elif model_name == 'swin':
        model = SwinTransformerECG(
            num_classes=num_classes, 
            dropout=dropout, 
            pretrained=pretrained,
            model_name=swin_model_name
        )
    
    elif model_name == 'swin_fusion':
        model = SwinTransformerFusionECG(
            num_classes=num_classes, 
            dropout=dropout, 
            pretrained=pretrained,
            model_name=swin_model_name,
            fusion_type=fusion_type
        )
    
    else:
        available_models = ['resnet50', 'efficientnet_b3', 'densenet121', 'vit', 'swin', 'swin_fusion']
        raise ValueError(f"Model {model_name} not supported. Choose from {available_models}")
    
    return model


