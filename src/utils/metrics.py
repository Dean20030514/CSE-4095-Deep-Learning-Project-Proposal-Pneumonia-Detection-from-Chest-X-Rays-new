from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Dict[int, str],
                   y_probs: Optional[np.ndarray] = None) -> Tuple[Dict, np.ndarray]:
    """
    计算分类任务的完整评估指标
    
    Args:
        y_true: 真实标签 (N,)
        y_pred: 预测标签 (N,)
        labels: 类别索引到名称的映射 {idx: name}
        y_probs: 预测概率 (N, C), 可选,用于计算 AUC 等指标
    
    Returns:
        metrics: 包含各种指标的字典
        cm: 混淆矩阵
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(labels.keys()))
    
    # Basic metrics
    macro_f1 = report.get('macro avg', {}).get('f1-score', np.nan)
    
    out = {
        'macro_f1': float(macro_f1),
        'per_class': {labels[i]: report.get(str(i), {}) for i in labels.keys()},
        'overall': {
            'accuracy': float(report.get('accuracy', np.nan)),
            'macro_precision': float(report.get('macro avg', {}).get('precision', np.nan)),
            'macro_recall': float(report.get('macro avg', {}).get('recall', np.nan)),
        }
    }
    
    # Add additional metrics
    try:
        out['overall']['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        out['overall']['mcc'] = None
    
    try:
        out['overall']['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    except ValueError:
        out['overall']['cohen_kappa'] = None
    
    # Compute sensitivity and specificity for binary/multi-class
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        out['overall']['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        out['overall']['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # AUC metrics (if probabilities provided)
    if y_probs is not None:
        try:
            num_classes = y_probs.shape[1]
            if num_classes == 2:
                # Binary: use positive class probability
                out['roc_auc'] = float(roc_auc_score(y_true, y_probs[:, 1]))
                out['pr_auc'] = float(average_precision_score(y_true, y_probs[:, 1]))
            else:
                # Multi-class: macro average
                out['roc_auc'] = float(roc_auc_score(y_true, y_probs, 
                                                     multi_class='ovr', average='macro'))
                out['pr_auc'] = float(average_precision_score(y_true, y_probs, average='macro'))
        except (ValueError, IndexError):
            out['roc_auc'] = None
            out['pr_auc'] = None
    
    return out, cm


def compute_sensitivity_specificity(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从混淆矩阵计算每个类的 sensitivity (recall) 和 specificity
    
    Args:
        cm: 混淆矩阵 (C, C)
    
    Returns:
        sensitivity: 每个类的灵敏度 (C,)
        specificity: 每个类的特异性 (C,)
    """
    num_classes = cm.shape[0]
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return sensitivity, specificity
