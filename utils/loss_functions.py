import torch
import torch.nn as nn
import torch.nn.functional as F
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class FocalLossWithClassWeights(nn.Module):
    def __init__(self, class_alpha=[0.10, 0.20, 0.30, 0.20, 0.20], gamma=2.0, smoothing=0.1, device='cuda'):
        super(FocalLossWithClassWeights, self).__init__()
        self.class_alpha = torch.tensor(class_alpha).to(device)
        self.gamma = gamma
        self.smoothing = smoothing
        self.device = device

    def forward(self, inputs, targets):
        # One-hot encode and apply label smoothing
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        targets = targets * (1 - self.smoothing) + self.smoothing / inputs.size(1)

        # Compute log-softmax and cross-entropy manually
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(targets * log_probs).sum(dim=1)

        # Compute focal loss components
        pt = torch.exp(-ce_loss)
        alpha_weights = self.class_alpha[targets.argmax(dim=1)]
        focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification"""
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, inputs, targets):
        # Calculating Probabilities
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            inputs_sigmoid = inputs_sigmoid + self.clip
            inputs_sigmoid = torch.clamp(inputs_sigmoid, max=1.0)
        
        # Basic CE calculation
        targets = targets.type_as(inputs)
        loss_pos = targets * torch.log(inputs_sigmoid)
        loss_neg = (1 - targets) * torch.log(1 - inputs_sigmoid)
        
        # Asymmetric Focusing
        loss_pos = loss_pos * (1 - inputs_sigmoid) ** self.gamma_pos
        loss_neg = loss_neg * inputs_sigmoid ** self.gamma_neg
        
        loss = -loss_pos - loss_neg
        return loss.mean()