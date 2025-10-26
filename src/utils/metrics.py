from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Dict[int, str]) -> Tuple[Dict, np.ndarray]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(labels.keys()))
    # derive macro_f1 key for convenience
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
    return out, cm
