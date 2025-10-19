from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, 
                                  precision_score, recall_score, hamming_loss)
# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true, y_pred, y_scores, threshold=0.5):
    """Compute comprehensive metrics for multi-label classification"""    
    metrics = {}
    
    # F1 Scores
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # AUC Scores
    try:
        metrics['auc_micro'] = roc_auc_score(y_true, y_scores, average='micro')
        metrics['auc_macro'] = roc_auc_score(y_true, y_scores, average='macro')
        metrics['auc_weighted'] = roc_auc_score(y_true, y_scores, average='weighted')
    except:
        metrics['auc_micro'] = 0.0
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
    
    # Other metrics
    metrics['exact_match_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['hamming_accuracy'] = 1 - hamming_loss(y_true, y_pred)    # Label-wise
    metrics['label_accuracy'] = (y_true == y_pred).mean()
    
    return metrics