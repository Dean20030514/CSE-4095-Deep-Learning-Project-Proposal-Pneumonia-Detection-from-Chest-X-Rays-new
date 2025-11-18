"""
单元测试：工具函数
"""
import pytest
import torch
import numpy as np
from src.utils.calibration import (
    compute_calibration_metrics,
    TemperatureScaling
)
from src.utils.gradcam import GradCAM


class TestCalibration:
    """测试模型校准功能"""
    
    def test_compute_calibration_metrics(self):
        """测试校准指标计算"""
        # 创建完美校准的数据
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])
        
        metrics = compute_calibration_metrics(y_true, y_probs, n_bins=5)
        
        # 验证返回的指标
        assert 'ece' in metrics
        assert 'mce' in metrics
        assert 'brier_score' in metrics
        assert 'bins' in metrics
        
        # 验证范围
        assert 0 <= metrics['ece'] <= 1
        assert 0 <= metrics['mce'] <= 1
        assert 0 <= metrics['brier_score'] <= 1
    
    def test_temperature_scaling(self):
        """测试温度缩放"""
        temp_scaler = TemperatureScaling()
        
        # 创建虚拟logits
        logits = torch.randn(10, 2)
        
        # 应用温度缩放
        scaled_logits = temp_scaler(logits)
        
        # 验证形状保持不变
        assert scaled_logits.shape == logits.shape
    
    def test_temperature_scaling_fit(self):
        """测试温度缩放拟合"""
        temp_scaler = TemperatureScaling()
        
        # 创建虚拟数据
        logits = torch.randn(20, 2)
        labels = torch.randint(0, 2, (20,))
        
        # 拟合温度参数
        temperature = temp_scaler.fit(logits, labels, max_iter=10)
        
        # 验证温度值合理
        assert isinstance(temperature, float)
        assert temperature > 0


class TestGradCAM:
    """测试GradCAM功能"""
    
    def test_gradcam_initialization(self):
        """测试GradCAM初始化"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        
        # 应该能成功初始化
        gradcam = GradCAM(model, 'layer4')
        assert gradcam is not None
    
    def test_gradcam_invalid_layer(self):
        """测试GradCAM使用无效层名"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        
        # 应该抛出异常
        with pytest.raises(ValueError):
            GradCAM(model, 'invalid_layer_name')
    
    def test_gradcam_forward(self):
        """测试GradCAM前向传播"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        # 创建虚拟输入
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        # 前向传播
        model.eval()
        logits = model(dummy_input)
        
        # 生成CAM（注意：参数是idx，不是target_class）
        cam = gradcam(logits, idx=1)
        
        # 验证输出
        assert isinstance(cam, torch.Tensor)
        assert cam.dim() == 2  # 2D heatmap
        assert (cam >= 0).all() and (cam <= 1).all()  # 归一化到[0, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

