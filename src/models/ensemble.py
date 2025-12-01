"""
模型集成模块

支持多模型集成预测以提高准确性和鲁棒性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from pathlib import Path


class ModelEnsemble(nn.Module):
    """
    多模型集成预测器。
    
    支持多种集成策略：
    - 平均概率 (average)
    - 加权平均 (weighted)
    - 投票 (voting)
    
    Example:
        >>> models = [model1, model2, model3]
        >>> ensemble = ModelEnsemble(models, strategy='weighted', weights=[0.5, 0.3, 0.2])
        >>> probs = ensemble(images)
    """
    
    def __init__(
        self, 
        models: List[nn.Module], 
        strategy: str = 'average',
        weights: Optional[List[float]] = None,
        temperature: float = 1.0
    ):
        """
        初始化模型集成。
        
        Args:
            models: 模型列表
            strategy: 集成策略 ('average', 'weighted', 'voting')
            weights: 模型权重（仅用于 'weighted' 策略）
            temperature: 温度参数（用于软化概率分布）
        """
        super().__init__()
        
        if not models:
            raise ValueError("At least one model is required")
        
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.temperature = temperature
        
        # 设置权重
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # 设置为评估模式
        for model in self.models:
            model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        集成预测。
        
        Args:
            x: 输入张量 (N, C, H, W)
        
        Returns:
            集成后的概率分布 (N, num_classes)
        """
        if self.strategy == 'voting':
            return self._hard_voting(x)
        elif self.strategy == 'weighted':
            return self._weighted_average(x)
        else:  # average
            return self._simple_average(x)
    
    def _simple_average(self, x: torch.Tensor) -> torch.Tensor:
        """简单平均集成"""
        probs_list = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits / self.temperature, dim=1)
                probs_list.append(probs)
        
        # 平均概率
        return torch.stack(probs_list).mean(dim=0)
    
    def _weighted_average(self, x: torch.Tensor) -> torch.Tensor:
        """加权平均集成"""
        weighted_probs = None
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits / self.temperature, dim=1)
                if weighted_probs is None:
                    weighted_probs = weight * probs
                else:
                    weighted_probs += weight * probs
        
        return weighted_probs
    
    def _hard_voting(self, x: torch.Tensor) -> torch.Tensor:
        """硬投票集成"""
        # 防御性检查：确保至少有一个模型
        if not self.models:
            raise ValueError("No models in ensemble for hard voting")
        
        votes = []
        num_classes = None
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                if num_classes is None:
                    num_classes = logits.size(1)
                preds = logits.argmax(dim=1)
                votes.append(preds)
        
        # 统计投票
        votes = torch.stack(votes, dim=0)  # (num_models, batch_size)
        
        # 转换为 one-hot 并求和
        one_hot_votes = F.one_hot(votes, num_classes=num_classes).float()  # (num_models, batch_size, num_classes)
        vote_counts = one_hot_votes.sum(dim=0)  # (batch_size, num_classes)
        
        # 归一化为概率
        return vote_counts / len(self.models)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        获取预测类别。
        
        Args:
            x: 输入张量
            threshold: 分类阈值（用于二分类）
        
        Returns:
            预测类别 (N,)
        """
        probs = self.forward(x)
        
        if probs.size(1) == 2:
            # 二分类使用阈值
            return (probs[:, 1] >= threshold).long()
        else:
            # 多分类使用 argmax
            return probs.argmax(dim=1)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple:
        """
        获取预测及其不确定性。
        
        Args:
            x: 输入张量
        
        Returns:
            (predictions, probabilities, uncertainties)
        """
        probs_list = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits / self.temperature, dim=1)
                probs_list.append(probs)
        
        probs_stack = torch.stack(probs_list)  # (num_models, batch_size, num_classes)
        
        # 平均概率
        mean_probs = probs_stack.mean(dim=0)
        
        # 不确定性（标准差）
        uncertainty = probs_stack.std(dim=0)
        
        # 预测
        predictions = mean_probs.argmax(dim=1)
        
        return predictions, mean_probs, uncertainty


def load_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    model_builder,
    num_classes: int,
    strategy: str = 'average',
    weights: Optional[List[float]] = None,
    device: str = 'cuda'
) -> ModelEnsemble:
    """
    从多个 checkpoint 加载模型集成。
    
    Args:
        checkpoint_paths: checkpoint 文件路径列表
        model_builder: 模型构建函数，接受 (model_name, num_classes) 参数
        num_classes: 类别数
        strategy: 集成策略
        weights: 模型权重
        device: 设备
    
    Returns:
        ModelEnsemble 实例
    
    Example:
        >>> from src.models.factory import build_model
        >>> paths = ['model1.pt', 'model2.pt', 'model3.pt']
        >>> ensemble = load_ensemble_from_checkpoints(paths, build_model, num_classes=2)
    """
    models = []
    
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        model_name = cfg.get('model', 'resnet18')
        
        model, _ = model_builder(model_name, num_classes)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        model.eval()
        
        models.append(model)
        print(f"Loaded: {ckpt_path} ({model_name})")
    
    ensemble = ModelEnsemble(models, strategy=strategy, weights=weights)
    return ensemble


if __name__ == '__main__':
    print("Model Ensemble module")
    print("Usage: from src.models.ensemble import ModelEnsemble, load_ensemble_from_checkpoints")

