"""
单元测试：损失函数模块

测试 FocalLoss 和其他损失函数
"""
import pytest
import torch
import torch.nn as nn

from src.models.losses import FocalLoss, LabelSmoothingCrossEntropy, get_loss_function


class TestFocalLossModule:
    """测试独立的 FocalLoss 模块"""
    
    def test_focal_loss_initialization(self):
        """测试 FocalLoss 初始化"""
        loss_fn = FocalLoss(gamma=2.0)
        assert loss_fn.gamma == 2.0
        assert loss_fn.weight is None
        assert loss_fn.reduction == 'mean'
    
    def test_focal_loss_forward(self):
        """测试 FocalLoss 前向传播"""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_focal_loss_with_weights(self):
        """测试带权重的 FocalLoss"""
        weights = torch.tensor([1.0, 2.0])
        loss_fn = FocalLoss(gamma=2.0, weight=weights)
        
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
    
    def test_focal_loss_gradient(self):
        """测试 FocalLoss 梯度"""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestLabelSmoothingLoss:
    """测试标签平滑损失"""
    
    def test_label_smoothing_initialization(self):
        """测试初始化"""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        assert loss_fn.smoothing == 0.1
        assert loss_fn.reduction == 'mean'
    
    def test_label_smoothing_forward(self):
        """测试前向传播"""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_label_smoothing_vs_ce(self):
        """测试与标准 CE 的比较"""
        # smoothing=0 应该接近标准 CE
        ls_loss = LabelSmoothingCrossEntropy(smoothing=0.0)
        ce_loss = nn.CrossEntropyLoss()
        
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        
        ls = ls_loss(logits, targets)
        ce = ce_loss(logits, targets)
        
        assert torch.abs(ls - ce) < 0.1
    
    def test_label_smoothing_reduces_confidence(self):
        """测试标签平滑减少过度自信"""
        # 高置信度预测
        logits = torch.tensor([[10.0, -10.0]])
        targets = torch.tensor([0])
        
        ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        ce_loss = nn.CrossEntropyLoss()
        
        ls = ls_loss(logits, targets)
        ce = ce_loss(logits, targets)
        
        # 标签平滑损失应该更大（惩罚过度自信）
        assert ls > ce


class TestGetLossFunction:
    """测试 get_loss_function 工厂函数"""
    
    def test_get_focal_loss(self):
        """测试获取 Focal Loss"""
        loss_fn = get_loss_function('focal', gamma=2.0)
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.gamma == 2.0
    
    def test_get_cross_entropy(self):
        """测试获取 Cross Entropy"""
        loss_fn = get_loss_function('cross_entropy')
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
    
    def test_get_weighted_ce(self):
        """测试获取 Weighted Cross Entropy"""
        weights = torch.tensor([1.0, 2.0])
        loss_fn = get_loss_function('weighted_ce', weight=weights)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
    
    def test_get_label_smoothing(self):
        """测试获取标签平滑损失"""
        loss_fn = get_loss_function('label_smoothing', smoothing=0.1)
        assert isinstance(loss_fn, LabelSmoothingCrossEntropy)
    
    def test_invalid_loss_name(self):
        """测试无效损失函数名称"""
        with pytest.raises(ValueError, match="Unknown loss function"):
            get_loss_function('invalid_loss')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

