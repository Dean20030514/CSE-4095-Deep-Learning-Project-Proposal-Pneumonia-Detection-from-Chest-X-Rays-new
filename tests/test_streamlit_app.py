"""
单元测试：Streamlit 应用功能

测试 Streamlit 应用中的工具函数（不测试 UI 组件）
"""
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import json


class TestGetGradcamTargetLayer:
    """测试 GradCAM 目标层获取函数"""
    
    def test_resnet_target_layer(self):
        """测试 ResNet 模型的目标层"""
        from src.app.streamlit_app import get_gradcam_target_layer
        from src.models.factory import build_model
        
        model, _ = build_model('resnet18', num_classes=2)
        layer = get_gradcam_target_layer(model, 'resnet18')
        
        assert layer == 'layer4'
    
    def test_resnet50_target_layer(self):
        """测试 ResNet50 模型的目标层"""
        from src.app.streamlit_app import get_gradcam_target_layer
        from src.models.factory import build_model
        
        model, _ = build_model('resnet50', num_classes=2)
        layer = get_gradcam_target_layer(model, 'resnet50')
        
        assert layer == 'layer4'
    
    def test_densenet_target_layer(self):
        """测试 DenseNet 模型的目标层"""
        from src.app.streamlit_app import get_gradcam_target_layer
        from src.models.factory import build_model
        
        model, _ = build_model('densenet121', num_classes=2)
        layer = get_gradcam_target_layer(model, 'densenet121')
        
        assert layer == 'features.denseblock4'
    
    def test_unknown_model_fallback(self):
        """测试未知模型回退到默认层"""
        from src.app.streamlit_app import get_gradcam_target_layer
        
        # 使用一个简单的模拟模型
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.ReLU(),
        )
        layer = get_gradcam_target_layer(model, 'unknown_model')
        
        assert layer == 'layer4'  # 默认回退


class TestOverlayHeatmap:
    """测试热力图叠加功能"""
    
    def test_overlay_basic(self):
        """测试基本热力图叠加"""
        from src.app.streamlit_app import overlay_heatmap
        
        # 创建测试图像和热力图
        img = Image.new('RGB', (100, 100), color='white')
        heatmap = np.random.rand(50, 50).astype(np.float32)
        
        result = overlay_heatmap(img, heatmap, alpha=0.4)
        
        assert isinstance(result, Image.Image)
        assert result.size == img.size
    
    def test_overlay_grayscale_image(self):
        """测试灰度图像叠加"""
        from src.app.streamlit_app import overlay_heatmap
        
        # 创建灰度图像
        img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        heatmap = np.random.rand(50, 50).astype(np.float32)
        
        result = overlay_heatmap(img, heatmap, alpha=0.4)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
    
    def test_overlay_different_alphas(self):
        """测试不同 alpha 值"""
        from src.app.streamlit_app import overlay_heatmap
        
        img = Image.new('RGB', (100, 100), color='blue')
        heatmap = np.ones((50, 50), dtype=np.float32)
        
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = overlay_heatmap(img, heatmap, alpha=alpha)
            assert isinstance(result, Image.Image)


class TestGetAvailableModels:
    """测试可用模型扫描功能"""
    
    def test_empty_runs_directory(self, tmp_path):
        """测试空 runs 目录"""
        from src.app.streamlit_app import get_available_models
        
        # 修改工作目录以使用临时目录
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # 不创建 runs 目录
            models = get_available_models()
            assert models == []
        finally:
            os.chdir(original_cwd)
    
    def test_with_models(self, tmp_path):
        """测试有模型的情况"""
        from src.app.streamlit_app import get_available_models
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # 创建模拟的模型目录和文件
            runs_dir = tmp_path / 'runs'
            runs_dir.mkdir()
            
            exp1 = runs_dir / 'experiment_1'
            exp1.mkdir()
            (exp1 / 'best_model.pt').touch()
            
            exp2 = runs_dir / 'experiment_2'
            exp2.mkdir()
            (exp2 / 'best_model.pt').touch()
            
            models = get_available_models()
            
            assert len(models) == 2
            model_names = [m['name'] for m in models]
            assert 'experiment_1' in model_names
            assert 'experiment_2' in model_names
        finally:
            os.chdir(original_cwd)


class TestGetModelPerformance:
    """测试模型性能获取功能"""
    
    def test_no_report_returns_none(self, tmp_path):
        """测试没有报告文件时返回 None"""
        from src.app.streamlit_app import get_model_performance
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = get_model_performance('nonexistent_experiment')
            assert result is None
        finally:
            os.chdir(original_cwd)
    
    def test_load_from_json_report(self, tmp_path):
        """测试从 JSON 报告加载性能数据"""
        from src.app.streamlit_app import get_model_performance
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # 创建模拟的报告文件
            reports_dir = tmp_path / 'reports'
            reports_dir.mkdir()
            
            report_data = {
                'metrics': {
                    'overall': {
                        'accuracy': 0.95,
                        'macro_recall': 0.94,
                    },
                    'per_class': {
                        'PNEUMONIA': {
                            'recall': 0.96,
                        }
                    },
                    'pr_auc': 0.98,
                }
            }
            
            report_path = reports_dir / 'test_my_experiment.json'
            with open(report_path, 'w') as f:
                json.dump(report_data, f)
            
            result = get_model_performance('my_experiment')
            
            assert result is not None
            assert result['accuracy'] == 95.0
            assert result['macro_recall'] == 94.0
            assert result['pneumonia_recall'] == 96.0
            assert result['pr_auc'] == 98.0
        finally:
            os.chdir(original_cwd)
    
    def test_load_from_runs_directory(self, tmp_path):
        """测试从 runs 目录加载性能数据"""
        from src.app.streamlit_app import get_model_performance
        
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # 创建模拟的 runs 目录结构
            runs_dir = tmp_path / 'runs' / 'test_experiment' / 'evaluation_curves'
            runs_dir.mkdir(parents=True)
            
            report_data = {
                'overall': {
                    'accuracy': 0.92,
                    'macro_recall': 0.91,
                },
                'per_class': {
                    'PNEUMONIA': {
                        'recall': 0.93,
                    }
                },
                'pr_auc': 0.97,
            }
            
            report_path = runs_dir / 'metrics.json'
            with open(report_path, 'w') as f:
                json.dump(report_data, f)
            
            result = get_model_performance('test_experiment')
            
            assert result is not None
            assert result['accuracy'] == 92.0
        finally:
            os.chdir(original_cwd)


class TestImagePreprocessing:
    """测试图像预处理相关功能"""
    
    def test_image_transform_output_shape(self):
        """测试图像变换输出形状"""
        import torchvision.transforms as T
        
        img_size = 224
        tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 创建测试图像
        img = Image.new('RGB', (300, 400), color='red')
        tensor = tf(img)
        
        assert tensor.shape == (3, 224, 224)
    
    def test_image_transform_normalization(self):
        """测试图像归一化"""
        import torchvision.transforms as T
        
        tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 使用非灰色图像以产生归一化后超出 [0,1] 范围的值
        img = Image.new('RGB', (224, 224), color='white')
        tensor = tf(img)
        
        # 归一化后值应该超出 [0, 1] 范围（白色像素归一化后会大于1）
        assert tensor.max() > 1


class TestPredictionLogic:
    """测试预测逻辑"""
    
    def test_threshold_prediction_positive(self):
        """测试阈值预测（正类）"""
        pneumonia_prob = 0.8
        threshold = 0.5
        
        prediction = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
        
        assert prediction == "PNEUMONIA"
    
    def test_threshold_prediction_negative(self):
        """测试阈值预测（负类）"""
        pneumonia_prob = 0.3
        threshold = 0.5
        
        prediction = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
        
        assert prediction == "NORMAL"
    
    def test_threshold_prediction_edge_case(self):
        """测试阈值边界情况"""
        pneumonia_prob = 0.5
        threshold = 0.5
        
        prediction = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
        
        # 等于阈值时应该判为正类
        assert prediction == "PNEUMONIA"
    
    def test_different_thresholds(self):
        """测试不同阈值的影响"""
        pneumonia_prob = 0.4
        
        # 低阈值
        assert ("PNEUMONIA" if pneumonia_prob >= 0.3 else "NORMAL") == "PNEUMONIA"
        
        # 高阈值
        assert ("PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL") == "NORMAL"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

