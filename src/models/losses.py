"""
损失函数模块

提供用于肺炎检测的自定义损失函数实现。
"""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    Particularly useful for medical imaging where class imbalance is common.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        gamma: Focusing parameter (default: 1.5). Higher values increase focus on hard examples.
        weight: Per-class weights for handling imbalance (optional)
        reduction: Reduction method ('mean', 'sum', or 'none')
    
    Example:
        >>> loss_fn = FocalLoss(gamma=2.0, weight=torch.tensor([1.0, 2.0]))
        >>> loss = loss_fn(logits, targets)
    """
    def __init__(self, gamma: float = 1.5, weight=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 Focal Loss。
        
        Args:
            logits: Model outputs (N, C) before softmax
            targets: Ground truth labels (N,)
        
        Returns:
            Focal loss value
        """
        # 计算 log probabilities（数值稳定）
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # 获取目标类别的 log probability
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算 probability pt
        pt = torch.exp(log_pt)
        
        # Focal Loss 公式: -(1-pt)^gamma * log(pt)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = -focal_weight * log_pt
        
        # 应用类别权重（如果提供）
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]
        
        # 应用 reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy Loss with Label Smoothing.
    
    Reduces overconfidence by smoothing hard labels.
    
    Args:
        smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = uniform distribution)
        reduction: Reduction method ('mean', 'sum', or 'none')
    
    Example:
        >>> loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        >>> loss = loss_fn(logits, targets)
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算带标签平滑的交叉熵损失。
        
        Args:
            logits: Model outputs (N, C) before softmax
            targets: Ground truth labels (N,)
        
        Returns:
            Label smoothing cross entropy loss value
        """
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # 创建平滑的标签分布
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算损失
        loss = (-smooth_labels * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_name: str, weight: torch.Tensor = None, **kwargs) -> nn.Module:
    """
    根据名称获取损失函数。
    
    Args:
        loss_name: 损失函数名称 ('focal', 'cross_entropy', 'weighted_ce', 'label_smoothing')
        weight: 类别权重
        **kwargs: 其他参数 (如 gamma, smoothing 等)
    
    Returns:
        损失函数模块
    
    Example:
        >>> loss_fn = get_loss_function('focal', weight=weights, gamma=2.0)
    """
    loss_name = loss_name.lower()
    
    if loss_name in ['focal', 'focal_loss']:
        gamma = kwargs.get('gamma', 1.5)
        return FocalLoss(gamma=gamma, weight=weight)
    
    elif loss_name in ['label_smoothing', 'smooth_ce']:
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    elif loss_name in ['cross_entropy', 'ce']:
        return nn.CrossEntropyLoss()
    
    elif loss_name in ['weighted_ce', 'weighted_cross_entropy']:
        return nn.CrossEntropyLoss(weight=weight)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Supported: focal, cross_entropy, weighted_ce, label_smoothing")

