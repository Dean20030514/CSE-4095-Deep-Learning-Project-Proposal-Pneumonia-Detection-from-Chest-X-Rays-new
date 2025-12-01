"""
预测不确定性估计模块

使用 Monte Carlo Dropout 和其他技术估计模型预测的不确定性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def enable_dropout(model: nn.Module):
    """启用模型中的所有 Dropout 层"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_samples: int = 30,
    return_samples: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 Monte Carlo Dropout 估计预测不确定性。
    
    通过多次前向传播（保持 Dropout 激活）来估计预测的不确定性。
    
    Args:
        model: PyTorch 模型（必须包含 Dropout 层）
        input_tensor: 输入张量 (N, C, H, W)
        n_samples: MC 采样次数
        return_samples: 是否返回所有采样结果
    
    Returns:
        mean_probs: 平均预测概率 (N, num_classes)
        uncertainty: 预测不确定性（标准差）(N, num_classes)
        [可选] samples: 所有采样结果 (n_samples, N, num_classes)
    
    Example:
        >>> mean_probs, uncertainty = mc_dropout_predict(model, images, n_samples=30)
        >>> high_uncertainty_mask = uncertainty.max(dim=1).values > 0.1
    """
    # 启用 Dropout
    model.eval()
    enable_dropout(model)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            predictions.append(probs)
    
    # Stack predictions
    predictions = torch.stack(predictions)  # (n_samples, N, num_classes)
    
    # 计算均值和标准差
    mean_probs = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)
    
    # 注意：不再重复调用 model.eval()，因为在函数开始时已经调用过了
    # enable_dropout() 仅启用 Dropout 层，而模型本身保持 eval 模式
    
    if return_samples:
        return mean_probs, uncertainty, predictions
    return mean_probs, uncertainty


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    计算预测分布的熵（不确定性度量）。
    
    Args:
        probs: 概率分布 (N, num_classes)
    
    Returns:
        entropy: 熵值 (N,)
    """
    # 避免 log(0)
    eps = 1e-10
    probs_clamped = probs.clamp(min=eps)
    entropy = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=1)
    return entropy


def compute_mutual_information(mc_predictions: torch.Tensor) -> torch.Tensor:
    """
    计算互信息（认知不确定性的度量）。
    
    Mutual Information = Total Entropy - Expected Entropy
    
    Args:
        mc_predictions: MC 采样预测 (n_samples, N, num_classes)
    
    Returns:
        mutual_info: 互信息 (N,)
    """
    # 平均预测的熵
    mean_probs = mc_predictions.mean(dim=0)
    total_entropy = compute_entropy(mean_probs)
    
    # 期望熵
    entropies = torch.stack([compute_entropy(p) for p in mc_predictions])
    expected_entropy = entropies.mean(dim=0)
    
    # 互信息
    mutual_info = total_entropy - expected_entropy
    return mutual_info


class UncertaintyEstimator:
    """
    不确定性估计器类。
    
    提供多种不确定性估计方法：
    - Monte Carlo Dropout
    - 温度缩放
    - 熵计算
    
    Example:
        >>> estimator = UncertaintyEstimator(model)
        >>> results = estimator.estimate(images)
        >>> print(f"Epistemic uncertainty: {results['epistemic']}")
    """
    
    def __init__(
        self, 
        model: nn.Module,
        n_mc_samples: int = 30,
        temperature: float = 1.0
    ):
        """
        初始化不确定性估计器。
        
        Args:
            model: PyTorch 模型
            n_mc_samples: MC Dropout 采样次数
            temperature: 温度参数（用于校准）
        """
        self.model = model
        self.n_mc_samples = n_mc_samples
        self.temperature = temperature
    
    def estimate(
        self, 
        input_tensor: torch.Tensor,
        use_mc_dropout: bool = True
    ) -> dict:
        """
        估计预测不确定性。
        
        Args:
            input_tensor: 输入张量
            use_mc_dropout: 是否使用 MC Dropout
        
        Returns:
            包含各种不确定性度量的字典：
            - predictions: 预测类别
            - probabilities: 预测概率
            - entropy: 预测熵（总不确定性）
            - aleatoric: 偶然不确定性（数据固有）
            - epistemic: 认知不确定性（模型不确定）
            - confidence: 预测置信度
        """
        results = {}
        
        if use_mc_dropout:
            mean_probs, std_probs, mc_samples = mc_dropout_predict(
                self.model, input_tensor, 
                n_samples=self.n_mc_samples,
                return_samples=True
            )
            
            # 认知不确定性（通过互信息）
            epistemic = compute_mutual_information(mc_samples)
            
            # 偶然不确定性（平均熵）
            entropies = torch.stack([compute_entropy(p) for p in mc_samples])
            aleatoric = entropies.mean(dim=0)
            
            results['epistemic'] = epistemic
            results['aleatoric'] = aleatoric
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensor)
                mean_probs = F.softmax(logits / self.temperature, dim=1)
            std_probs = torch.zeros_like(mean_probs)
        
        # 总不确定性（熵）
        total_entropy = compute_entropy(mean_probs)
        
        # 预测和置信度
        predictions = mean_probs.argmax(dim=1)
        confidence = mean_probs.max(dim=1).values
        
        results.update({
            'predictions': predictions,
            'probabilities': mean_probs,
            'std': std_probs,
            'entropy': total_entropy,
            'confidence': confidence,
        })
        
        return results
    
    def is_uncertain(
        self, 
        input_tensor: torch.Tensor,
        entropy_threshold: float = 0.5,
        confidence_threshold: float = 0.7
    ) -> torch.Tensor:
        """
        判断预测是否具有高不确定性。
        
        Args:
            input_tensor: 输入张量
            entropy_threshold: 熵阈值
            confidence_threshold: 置信度阈值
        
        Returns:
            布尔张量，True 表示高不确定性
        """
        results = self.estimate(input_tensor, use_mc_dropout=True)
        
        # 熵高或置信度低的样本被标记为不确定
        high_entropy = results['entropy'] > entropy_threshold
        low_confidence = results['confidence'] < confidence_threshold
        
        return high_entropy | low_confidence


def calibrate_temperature(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    max_iter: int = 50
) -> float:
    """
    使用温度缩放校准模型预测。
    
    Args:
        model: PyTorch 模型
        val_loader: 验证数据加载器
        device: 设备
        max_iter: 优化最大迭代次数
    
    Returns:
        最优温度值
    """
    from torch.optim import LBFGS
    
    # 收集所有 logits 和标签
    all_logits = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # 优化温度
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = all_logits / temperature
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    optimal_temp = temperature.item()
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    return optimal_temp


if __name__ == '__main__':
    print("Uncertainty Estimation module")
    print("Usage: from src.utils.uncertainty import mc_dropout_predict, UncertaintyEstimator")

