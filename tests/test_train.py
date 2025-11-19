"""
单元测试：训练相关功能

测试损失函数、种子设置、checkpoint保存等训练工具
"""
import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from src.train import FocalLoss, set_seed, save_checkpoint


class TestFocalLoss:
    """测试Focal Loss实现"""
    
    def test_focal_loss_initialization(self):
        """测试Focal Loss初始化"""
        loss_fn = FocalLoss(gamma=2.0)
        assert loss_fn.gamma == 2.0
        assert loss_fn.weight is None
        assert loss_fn.reduction == 'mean'
    
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
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_focal_loss_with_weights(self):
        """测试带类别权重的Focal Loss"""
        weights = torch.tensor([1.0, 2.0])
        loss_fn = FocalLoss(gamma=2.0, weight=weights)
        
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
        
        # 验证weight被正确存储
        assert torch.allclose(loss_fn.weight, weights)
    
    def test_focal_loss_gradient(self):
        """测试Focal Loss梯度计算"""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        # 验证梯度存在且有效
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()
    
    def test_focal_loss_vs_ce(self):
        """测试Focal Loss与CrossEntropy的关系"""
        # gamma=0时，Focal Loss应该接近CrossEntropy
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        
        focal_loss = FocalLoss(gamma=0.0)
        ce_loss = nn.CrossEntropyLoss()
        
        fl = focal_loss(logits, targets)
        ce = ce_loss(logits, targets)
        
        # 应该非常接近
        assert torch.abs(fl - ce) < 0.1
    
    def test_focal_loss_different_gammas(self):
        """测试不同gamma值的效果"""
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        losses = []
        for gamma in [0.0, 1.0, 2.0, 3.0]:
            loss_fn = FocalLoss(gamma=gamma)
            loss = loss_fn(logits, targets)
            losses.append(loss.item())
        
        # 验证所有损失都是合理的
        for loss in losses:
            assert loss >= 0
            assert loss < 100  # 合理的上界
    
    def test_focal_loss_reduction_modes(self):
        """测试不同的reduction模式"""
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        # mean
        loss_mean = FocalLoss(gamma=2.0, reduction='mean')(logits, targets)
        assert loss_mean.dim() == 0
        
        # sum
        loss_sum = FocalLoss(gamma=2.0, reduction='sum')(logits, targets)
        assert loss_sum.dim() == 0
        
        # none
        loss_none = FocalLoss(gamma=2.0, reduction='none')(logits, targets)
        assert loss_none.dim() == 1
        assert loss_none.shape[0] == 4


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
    
    def test_set_seed_affects_numpy(self):
        """测试种子设置影响numpy"""
        import numpy as np
        
        set_seed(42)
        array1 = np.random.rand(10)
        
        set_seed(42)
        array2 = np.random.rand(10)
        
        # numpy也应该被设置
        assert np.allclose(array1, array2)
    
    def test_set_seed_affects_python_random(self):
        """测试种子设置影响Python random"""
        import random
        
        set_seed(42)
        values1 = [random.random() for _ in range(10)]
        
        set_seed(42)
        values2 = [random.random() for _ in range(10)]
        
        # Python random也应该被设置
        assert values1 == values2


class TestCheckpointSaving:
    """测试checkpoint保存功能"""
    
    def test_save_checkpoint_basic(self, tmp_path):
        """测试基本checkpoint保存"""
        checkpoint_path = tmp_path / 'test_checkpoint.pt'
        
        state = {
            'model': {'weight': torch.randn(10, 10)},
            'epoch': 5,
            'config': {'lr': 0.001}
        }
        
        save_checkpoint(state, checkpoint_path)
        
        # 验证文件存在
        assert checkpoint_path.exists()
        
        # 验证可以加载
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert 'model' in loaded
        assert 'epoch' in loaded
        assert loaded['epoch'] == 5
    
    def test_save_checkpoint_creates_dirs(self, tmp_path):
        """测试checkpoint保存自动创建目录"""
        checkpoint_path = tmp_path / 'sub' / 'dir' / 'checkpoint.pt'
        
        state = {'data': 'test'}
        save_checkpoint(state, checkpoint_path)
        
        # 验证父目录被创建
        assert checkpoint_path.parent.exists()
        assert checkpoint_path.exists()
    
    def test_save_checkpoint_overwrite(self, tmp_path):
        """测试checkpoint覆盖"""
        checkpoint_path = tmp_path / 'checkpoint.pt'
        
        # 保存第一个checkpoint
        state1 = {'epoch': 1}
        save_checkpoint(state1, checkpoint_path)
        
        # 覆盖
        state2 = {'epoch': 2}
        save_checkpoint(state2, checkpoint_path)
        
        # 验证是新的数据
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded['epoch'] == 2
    
    def test_save_checkpoint_complex_state(self, tmp_path):
        """测试保存复杂的checkpoint状态"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = tmp_path / 'complex_checkpoint.pt'
        
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': 10,
            'best_score': 0.95,
            'config': {
                'model': 'resnet18',
                'lr': 0.001
            },
            'classes': {'NORMAL': 0, 'PNEUMONIA': 1},
            'metrics': {
                'val_acc': 0.95,
                'pneumonia_recall': 0.96
            }
        }
        
        save_checkpoint(state, checkpoint_path)
        
        # 验证所有字段都被保存
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert 'model' in loaded
        assert 'optimizer' in loaded
        assert 'epoch' in loaded
        assert 'best_score' in loaded
        assert 'config' in loaded
        assert 'classes' in loaded
        assert 'metrics' in loaded


class TestTrainingConfiguration:
    """测试训练配置相关"""
    
    def test_focal_loss_config(self, sample_config):
        """测试从配置创建Focal Loss"""
        focal_config = sample_config.get('focal', {})
        gamma = focal_config.get('gamma', 1.5)
        
        loss_fn = FocalLoss(gamma=gamma)
        assert loss_fn.gamma == gamma
    
    def test_seed_from_config(self, sample_config):
        """测试从配置读取随机种子"""
        seed = sample_config.get('seed', 42)
        
        set_seed(seed)
        tensor1 = torch.randn(5)
        
        set_seed(seed)
        tensor2 = torch.randn(5)
        
        assert torch.allclose(tensor1, tensor2)


class TestLossEdgeCases:
    """测试损失函数边界情况"""
    
    def test_focal_loss_single_sample(self):
        """测试单样本情况"""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(1, 2)
        targets = torch.tensor([0])
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
    
    def test_focal_loss_large_batch(self):
        """测试大batch情况"""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(1000, 2)
        targets = torch.randint(0, 2, (1000,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
    
    def test_focal_loss_multiclass(self):
        """测试多分类情况"""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(10, 5)  # 5个类别
        targets = torch.randint(0, 5, (10,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
    
    def test_focal_loss_extreme_confidence(self):
        """测试极端置信度情况"""
        loss_fn = FocalLoss(gamma=2.0)
        
        # 非常自信的正确预测
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([0, 1])
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
        assert loss.item() < 1.0  # 应该很小


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
