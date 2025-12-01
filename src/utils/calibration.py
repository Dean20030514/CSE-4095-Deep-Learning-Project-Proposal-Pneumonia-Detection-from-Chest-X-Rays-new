"""
Calibration utilities for model confidence assessment.
Includes temperature scaling and reliability diagram generation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    Learns a single scalar parameter to scale logits before softmax.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        """Apply temperature scaling to logits"""
        return logits / self.temperature
    
    def fit(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        lr: float = 0.01, 
        max_iter: int = 50
    ) -> float:
        """
        Fit temperature parameter using validation set.
        
        Args:
            logits: model logits (N, C)
            labels: ground truth labels (N,)
            lr: learning rate for optimization
            max_iter: maximum iterations for L-BFGS optimizer
        
        Returns:
            Optimal temperature value
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        logits = logits.clone()
        labels = labels.clone()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return self.temperature.item()


def compute_calibration_metrics(y_true, y_probs, n_bins=10):
    """
    Compute calibration metrics including ECE and reliability diagram data.
    
    Args:
        y_true: true labels (N,)
        y_probs: predicted probabilities (N, C)
        n_bins: number of bins for calibration
    
    Returns:
        metrics: dict with ECE, MCE, and bin statistics
    """
    # Get predicted class and confidence
    y_pred = y_probs.argmax(axis=1)
    confidences = y_probs.max(axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute per-bin statistics
    bin_stats = []
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()
            
            bin_stats.append({
                'bin_id': i,
                'bin_lower': float(bin_edges[i]),
                'bin_upper': float(bin_edges[i + 1]),
                'accuracy': float(bin_acc),
                'confidence': float(bin_conf),
                'count': int(bin_count),
                'gap': float(abs(bin_acc - bin_conf))
            })
            
            # ECE: weighted average of calibration errors
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)
            
            # MCE: maximum calibration error
            mce = max(mce, abs(bin_acc - bin_conf))
        else:
            bin_stats.append({
                'bin_id': i,
                'bin_lower': float(bin_edges[i]),
                'bin_upper': float(bin_edges[i + 1]),
                'accuracy': 0.0,
                'confidence': 0.0,
                'count': 0,
                'gap': 0.0
            })
    
    # Compute Brier score
    # For multi-class: average over all classes
    brier_scores = []
    for c in range(y_probs.shape[1]):
        y_true_binary = (y_true == c).astype(float)
        brier_scores.append(brier_score_loss(y_true_binary, y_probs[:, c]))
    
    avg_brier = np.mean(brier_scores)
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': float(avg_brier),
        'bins': bin_stats
    }


def plot_reliability_diagram(calibration_metrics, save_path=None):
    """
    Plot reliability diagram from calibration metrics.
    
    Args:
        calibration_metrics: output from compute_calibration_metrics
        save_path: path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    bins = calibration_metrics['bins']
    confidences = [b['confidence'] for b in bins if b['count'] > 0]
    accuracies = [b['accuracy'] for b in bins if b['count'] > 0]
    counts = [b['count'] for b in bins if b['count'] > 0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax.bar(confidences, accuracies, width=0.08, alpha=0.7, 
           edgecolor='black', label='Model')
    
    # Add sample counts as text
    for conf, acc, count in zip(confidences, accuracies, counts):
        ax.text(conf, acc + 0.02, str(count), ha='center', fontsize=9)
    
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f"Reliability Diagram\nECE: {calibration_metrics['ece']:.4f}, "
                 f"Brier: {calibration_metrics['brier_score']:.4f}", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Reliability diagram saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Calibration utilities module")
    print("Use compute_calibration_metrics() to assess model calibration")
    print("Use TemperatureScaling to calibrate model outputs")
