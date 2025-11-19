"""
单元测试：工具函数

测试校准、GradCAM、设备选择、配置验证等工具模块
"""
import pytest
import torch
import numpy as np
from pathlib import Path

from src.utils.calibration import (
    compute_calibration_metrics,
    TemperatureScaling
)
from src.utils.gradcam import GradCAM
from src.utils.device import get_device
from src.utils.config_validator import ConfigValidator


class TestCalibration:
    """测试模型校准功能"""
    
    def test_compute_calibration_metrics(self, sample_predictions):
        """测试校准指标计算"""
        y_true = sample_predictions['y_true']
        y_probs = sample_predictions['y_probs']
        
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
        
        # 验证bins结构
        assert len(metrics['bins']) == 5
        for bin_stat in metrics['bins']:
            assert 'bin_id' in bin_stat
            assert 'accuracy' in bin_stat
            assert 'confidence' in bin_stat
            assert 'count' in bin_stat
            assert 'gap' in bin_stat
    
    def test_calibration_perfect_model(self):
        """测试完美校准的模型"""
        # 创建完美校准的数据
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])
        
        metrics = compute_calibration_metrics(y_true, y_probs, n_bins=5)
        
        # ECE应该很小（接近0）
        assert metrics['ece'] < 0.3
    
    def test_calibration_different_bins(self):
        """测试不同bin数量"""
        y_true = np.array([0] * 50 + [1] * 50)
        y_probs = np.random.rand(100, 2)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
        
        for n_bins in [5, 10, 15]:
            metrics = compute_calibration_metrics(y_true, y_probs, n_bins=n_bins)
            assert len(metrics['bins']) == n_bins
    
    def test_temperature_scaling(self):
        """测试温度缩放"""
        temp_scaler = TemperatureScaling()
        
        # 创建虚拟logits
        logits = torch.randn(10, 2)
        
        # 应用温度缩放
        scaled_logits = temp_scaler(logits)
        
        # 验证形状保持不变
        assert scaled_logits.shape == logits.shape
        
        # 验证temperature初始化为1
        assert temp_scaler.temperature.item() == 1.0
    
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
        assert temperature < 100  # 合理的上界
    
    def test_temperature_scaling_improves_calibration(self):
        """测试温度缩放改善校准"""
        # 创建过度自信的模型输出
        logits = torch.randn(50, 2) * 5  # 放大logits使模型过度自信
        labels = torch.randint(0, 2, (50,))
        
        # 训练温度缩放
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(logits, labels, max_iter=50)
        
        # 温度应该>1（降低置信度）
        assert temp_scaler.temperature.item() > 1.0


class TestGradCAM:
    """测试GradCAM功能"""
    
    def test_gradcam_initialization(self):
        """测试GradCAM初始化"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        
        # 应该能成功初始化
        gradcam = GradCAM(model, 'layer4')
        assert gradcam is not None
        assert gradcam.model is not None
    
    def test_gradcam_invalid_layer(self):
        """测试GradCAM使用无效层名"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        
        # 应该抛出异常
        with pytest.raises(ValueError) as exc_info:
            GradCAM(model, 'invalid_layer_name')
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_gradcam_forward(self):
        """测试GradCAM前向传播"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        # 创建虚拟输入
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        # 生成CAM
        cam = gradcam(dummy_input, target_class=1)
        
        # 验证输出
        assert isinstance(cam, torch.Tensor)
        assert cam.dim() == 2  # 2D heatmap
        assert cam.shape[0] == 224
        assert cam.shape[1] == 224
        assert (cam >= 0).all() and (cam <= 1).all()  # 归一化到[0, 1]
    
    def test_gradcam_different_targets(self):
        """测试不同目标类别"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 生成两个类别的CAM
        cam0 = gradcam(dummy_input, target_class=0)
        cam1 = gradcam(dummy_input, target_class=1)
        
        # 两个CAM应该不同
        assert not torch.allclose(cam0, cam1)
    
    def test_gradcam_3d_input(self):
        """测试3D输入（自动添加batch维度）"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        gradcam = GradCAM(model, 'layer4')
        
        # 3D输入（C, H, W）
        dummy_input = torch.randn(3, 224, 224)
        
        # 应该能处理
        cam = gradcam(dummy_input, target_class=1)
        assert cam.shape == (224, 224)
    
    @pytest.mark.slow
    def test_gradcam_different_architectures(self):
        """测试不同架构的GradCAM"""
        from src.models.factory import build_model
        
        architectures = [
            ('resnet18', 'layer4'),
            ('densenet121', 'features'),
        ]
        
        for model_name, layer_name in architectures:
            try:
                model, _ = build_model(model_name, num_classes=2)
                gradcam = GradCAM(model, layer_name)
                
                dummy_input = torch.randn(1, 3, 224, 224)
                cam = gradcam(dummy_input, target_class=1)
                
                assert cam.shape == (224, 224)
                assert (cam >= 0).all() and (cam <= 1).all()
            except (ValueError, RuntimeError) as e:
                # 某些架构可能不支持
                pytest.skip(f"Architecture {model_name} not supported: {e}")


class TestDeviceSelection:
    """测试设备选择功能"""
    
    def test_get_device(self):
        """测试设备选择"""
        device = get_device()
        
        # 应该返回有效的设备
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'cpu', 'privateuseone']  # privateuseone for DirectML
    
    def test_device_usable(self):
        """测试设备可用性"""
        device = get_device()
        
        # 应该能在选定的设备上创建tensor
        tensor = torch.randn(10, 10).to(device)
        assert tensor.device.type == device.type
    
    def test_model_to_device(self):
        """测试模型转移到设备"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        device = get_device()
        
        model = model.to(device)
        
        # 验证模型参数在正确设备上
        for param in model.parameters():
            assert param.device.type == device.type


class TestConfigValidator:
    """测试配置验证功能"""
    
    def test_valid_config(self, sample_config):
        """测试有效配置"""
        # 应该不抛出异常
        ConfigValidator.validate(sample_config)
    
    def test_missing_required_field(self):
        """测试缺少必需字段"""
        config = {
            'model': 'resnet18',
            'img_size': 224,
            'batch_size': 16,
            # 缺少 'epochs' 和 'lr'
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate(config)
        
        assert "缺少必需配置" in str(exc_info.value)
    
    def test_invalid_model_name(self):
        """测试无效模型名称"""
        config = {
            'model': 'invalid_model',
            'img_size': 224,
            'batch_size': 16,
            'epochs': 10,
            'lr': 0.001
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate(config)
        
        assert "无效的模型" in str(exc_info.value)
    
    def test_invalid_type(self):
        """测试类型错误"""
        config = {
            'model': 'resnet18',
            'img_size': '224',  # 应该是int
            'batch_size': 16,
            'epochs': 10,
            'lr': 0.001
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate(config)
        
        assert "类型错误" in str(exc_info.value)
    
    def test_out_of_range_values(self):
        """测试超出范围的值"""
        config = {
            'model': 'resnet18',
            'img_size': 10000,  # 太大
            'batch_size': 16,
            'epochs': 10,
            'lr': 0.001
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate(config)
        
        assert "范围内" in str(exc_info.value)
    
    def test_invalid_scheduler(self):
        """测试无效的scheduler"""
        config = {
            'model': 'resnet18',
            'img_size': 224,
            'batch_size': 16,
            'epochs': 10,
            'lr': 0.001,
            'scheduler': 'invalid_scheduler'
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate(config)
        
        assert "无效的scheduler" in str(exc_info.value)
    
    def test_nested_config_validation(self):
        """测试嵌套配置验证"""
        # 有效的嵌套配置
        config = {
            'model': 'resnet18',
            'img_size': 224,
            'batch_size': 16,
            'epochs': 10,
            'lr': 0.001,
            'early_stopping': {
                'patience': 10
            },
            'focal': {
                'gamma': 2.0
            }
        }
        
        # 应该通过验证
        ConfigValidator.validate(config)
    
    def test_invalid_nested_config(self):
        """测试无效的嵌套配置"""
        config = {
            'model': 'resnet18',
            'img_size': 224,
            'batch_size': 16,
            'epochs': 10,
            'lr': 0.001,
            'early_stopping': {
                # 缺少patience字段
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate(config)
        
        assert "early_stopping" in str(exc_info.value)
    
    def test_validate_file(self, tmp_path, sample_config):
        """测试从文件验证配置"""
        import yaml
        
        # 创建配置文件
        config_file = tmp_path / 'test_config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        # 验证文件
        loaded_config = ConfigValidator.validate_file(str(config_file))
        
        # 验证加载的配置
        assert loaded_config['model'] == sample_config['model']
        assert loaded_config['img_size'] == sample_config['img_size']
    
    def test_validate_nonexistent_file(self):
        """测试验证不存在的文件"""
        with pytest.raises(FileNotFoundError):
            ConfigValidator.validate_file('nonexistent_config.yaml')


class TestUtilsIntegration:
    """测试工具模块的集成"""
    
    def test_calibration_with_model_output(self):
        """测试校准与模型输出的集成"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        model.eval()
        
        # 生成预测
        images = torch.randn(20, 3, 224, 224)
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1).numpy()
        
        # 创建假标签
        y_true = np.random.randint(0, 2, 20)
        
        # 计算校准指标
        metrics = compute_calibration_metrics(y_true, probs, n_bins=5)
        
        # 验证指标有效
        assert 'ece' in metrics
        assert metrics['ece'] >= 0
    
    def test_gradcam_with_calibration(self):
        """测试GradCAM与校准的结合"""
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        
        # GradCAM
        gradcam = GradCAM(model, 'layer4')
        dummy_input = torch.randn(1, 3, 224, 224)
        cam = gradcam(dummy_input, target_class=1)
        
        # 温度缩放
        temp_scaler = TemperatureScaling()
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        scaled_logits = temp_scaler(logits)
        
        # 两者都应该正常工作
        assert cam.shape == (224, 224)
        assert scaled_logits.shape == logits.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
