"""
单元测试：训练相关功能
"""
import pytest
import torch
import torch.nn as nn
from src.train import FocalLoss, set_seed


class TestFocalLoss:
    """测试Focal Loss实现"""
    
    def test_focal_loss_initialization(self):
        """测试Focal Loss初始化"""
        loss_fn = FocalLoss(gamma=2.0)
        assert loss_fn.gamma == 2.0
    
    def test_focal_loss_forward(self):
        """测试Focal Loss前向传播"""
        loss_fn = FocalLoss(gamma=2.0)
        
        # 创建虚拟数据
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        # 计算损失
        loss = loss_fn(logits, targets)
        
        # 验证损失值
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # 损失应该非负
    
    def test_focal_loss_with_weights(self):
        """测试带类别权重的Focal Loss"""
        weights = torch.tensor([1.0, 2.0])
        loss_fn = FocalLoss(gamma=2.0, weight=weights)
        
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
    
    def test_focal_loss_gradient(self):
        """测试Focal Loss梯度计算"""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        # 验证梯度存在
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestTrainingUtilities:
    """测试训练工具函数"""
    
    def test_set_seed_reproducibility(self):
        """测试随机种子设置的可复现性"""
        set_seed(42)
        tensor1 = torch.randn(10)
        
        set_seed(42)
        tensor2 = torch.randn(10)
        
        # 验证相同种子产生相同结果
        assert torch.allclose(tensor1, tensor2)
    
    def test_set_seed_different_values(self):
        """测试不同种子产生不同结果"""
        set_seed(42)
        tensor1 = torch.randn(10)
        
        set_seed(123)
        tensor2 = torch.randn(10)
        
        # 验证不同种子产生不同结果
        assert not torch.allclose(tensor1, tensor2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

