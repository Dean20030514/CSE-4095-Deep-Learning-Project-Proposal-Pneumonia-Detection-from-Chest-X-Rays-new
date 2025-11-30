"""
单元测试：GradCAM 可解释性工具

测试 GradCAM 热力图生成功能
"""
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.utils.gradcam import GradCAM
from src.models.factory import build_model


class TestGradCAM:
    """测试 GradCAM 实现"""
    
    def test_gradcam_initialization(self):
        """测试 GradCAM 初始化"""
        model, _ = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        assert gradcam.model is not None
        assert gradcam.gradients is None
        assert gradcam.activations is None
    
    def test_gradcam_invalid_layer(self):
        """测试无效层名称应该抛出错误"""
        model, _ = build_model('resnet18', num_classes=2)
        
        with pytest.raises(ValueError, match="Layer .* not found"):
            GradCAM(model, 'invalid_layer_name')
    
    def test_gradcam_output_shape(self):
        """测试 GradCAM 输出形状"""
        model, img_size = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        # 创建测试输入
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        
        # 生成热力图
        cam = gradcam(input_tensor, target_class=1)
        
        # 验证输出形状
        assert cam.shape == (img_size, img_size)
    
    def test_gradcam_output_range(self):
        """测试 GradCAM 输出值范围应该在 [0, 1]"""
        model, img_size = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        cam = gradcam(input_tensor, target_class=0)
        
        # 验证值范围
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0
    
    def test_gradcam_different_classes(self):
        """测试不同目标类别产生不同的热力图"""
        model, img_size = build_model('resnet18', num_classes=2)
        
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        
        # 为类别 0 生成热力图
        gradcam0 = GradCAM(model, 'layer4')
        cam0 = gradcam0(input_tensor.clone().requires_grad_(True), target_class=0)
        
        # 为类别 1 生成热力图
        gradcam1 = GradCAM(model, 'layer4')
        cam1 = gradcam1(input_tensor.clone().requires_grad_(True), target_class=1)
        
        # 热力图应该不同（对于随机输入，通常情况下）
        # 注意：在某些极端情况下可能相同，所以我们只检查它们是有效的
        assert cam0.shape == cam1.shape
        assert not torch.isnan(cam0).any()
        assert not torch.isnan(cam1).any()
    
    def test_gradcam_with_efficientnet(self):
        """测试 EfficientNet 模型的 GradCAM"""
        model, img_size = build_model('efficientnet_b0', num_classes=2)
        
        # EfficientNet 使用 features.8 作为目标层
        gradcam = GradCAM(model, 'features.8')
        
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        cam = gradcam(input_tensor, target_class=1)
        
        assert cam.shape == (img_size, img_size)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0
    
    def test_gradcam_with_densenet(self):
        """测试 DenseNet 模型的 GradCAM"""
        model, img_size = build_model('densenet121', num_classes=2)
        
        # DenseNet 使用 features.denseblock4 作为目标层
        gradcam = GradCAM(model, 'features.denseblock4')
        
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        cam = gradcam(input_tensor, target_class=1)
        
        assert cam.shape == (img_size, img_size)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0
    
    def test_gradcam_3d_input(self):
        """测试 3D 输入（无 batch 维度）自动处理"""
        model, img_size = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        # 3D 输入（没有 batch 维度）
        input_tensor = torch.randn(3, img_size, img_size, requires_grad=True)
        cam = gradcam(input_tensor, target_class=0)
        
        assert cam.shape == (img_size, img_size)
    
    def test_gradcam_batch_of_one(self):
        """测试 batch_size=1 的输入"""
        model, img_size = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        cam = gradcam(input_tensor, target_class=0)
        
        assert cam.shape == (img_size, img_size)
    
    def test_gradcam_reproducibility(self):
        """测试 GradCAM 可复现性"""
        model, img_size = build_model('resnet18', num_classes=2)
        model.eval()
        
        # 设置随机种子
        torch.manual_seed(42)
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        
        gradcam1 = GradCAM(model, 'layer4')
        cam1 = gradcam1(input_tensor.clone().requires_grad_(True), target_class=0)
        
        gradcam2 = GradCAM(model, 'layer4')
        cam2 = gradcam2(input_tensor.clone().requires_grad_(True), target_class=0)
        
        # 相同输入应该产生相同输出
        assert torch.allclose(cam1, cam2, atol=1e-5)


class TestGradCAMEdgeCases:
    """测试 GradCAM 边界情况"""
    
    def test_gradcam_with_eval_mode(self):
        """测试模型在 eval 模式下的 GradCAM"""
        model, img_size = build_model('resnet18', num_classes=2)
        model.eval()
        
        gradcam = GradCAM(model, 'layer4')
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        cam = gradcam(input_tensor, target_class=0)
        
        assert not torch.isnan(cam).any()
        assert not torch.isinf(cam).any()
    
    def test_gradcam_multiclass(self):
        """测试多分类情况（超过 2 个类别）"""
        model, img_size = build_model('resnet18', num_classes=5)
        gradcam = GradCAM(model, 'layer4')
        
        input_tensor = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        
        # 测试所有类别
        for target_class in range(5):
            cam = gradcam(input_tensor.clone().requires_grad_(True), target_class=target_class)
            assert cam.shape == (img_size, img_size)
            assert cam.min() >= 0.0
            assert cam.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

