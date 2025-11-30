"""
单元测试：模型导出功能

测试 ONNX 和 TorchScript 导出功能
"""
import pytest
import torch
import tempfile
from pathlib import Path

from src.models.factory import build_model
from src.utils.export import export_to_onnx, export_to_torchscript


class TestONNXExport:
    """测试 ONNX 导出功能"""
    
    def test_export_resnet18_onnx(self, tmp_path):
        """测试 ResNet18 ONNX 导出"""
        model, img_size = build_model('resnet18', num_classes=2)
        model.eval()
        
        output_path = tmp_path / 'model.onnx'
        result = export_to_onnx(model, img_size, str(output_path))
        
        assert result.exists()
        assert result.suffix == '.onnx'
    
    def test_export_efficientnet_onnx(self, tmp_path):
        """测试 EfficientNet ONNX 导出"""
        model, img_size = build_model('efficientnet_b0', num_classes=2)
        model.eval()
        
        output_path = tmp_path / 'model.onnx'
        result = export_to_onnx(model, img_size, str(output_path))
        
        assert result.exists()
    
    def test_export_creates_parent_dirs(self, tmp_path):
        """测试导出自动创建父目录"""
        model, img_size = build_model('resnet18', num_classes=2)
        
        output_path = tmp_path / 'sub' / 'dir' / 'model.onnx'
        result = export_to_onnx(model, img_size, str(output_path))
        
        assert result.exists()
        assert result.parent.exists()
    
    def test_export_dynamic_batch(self, tmp_path):
        """测试动态 batch 导出"""
        model, img_size = build_model('resnet18', num_classes=2)
        
        output_path = tmp_path / 'model.onnx'
        result = export_to_onnx(model, img_size, str(output_path), dynamic_batch=True)
        
        assert result.exists()


class TestTorchScriptExport:
    """测试 TorchScript 导出功能"""
    
    def test_export_resnet18_torchscript(self, tmp_path):
        """测试 ResNet18 TorchScript 导出"""
        model, img_size = build_model('resnet18', num_classes=2)
        model.eval()
        
        output_path = tmp_path / 'model.pt'
        result = export_to_torchscript(model, img_size, str(output_path))
        
        assert result.exists()
    
    def test_torchscript_trace_method(self, tmp_path):
        """测试 trace 方法导出"""
        model, img_size = build_model('resnet18', num_classes=2)
        
        output_path = tmp_path / 'model.pt'
        result = export_to_torchscript(model, img_size, str(output_path), method='trace')
        
        assert result.exists()
        
        # 验证可以加载
        loaded = torch.jit.load(str(result))
        assert loaded is not None
    
    def test_torchscript_inference(self, tmp_path):
        """测试导出的 TorchScript 模型推理"""
        model, img_size = build_model('resnet18', num_classes=2)
        model.eval()
        
        output_path = tmp_path / 'model.pt'
        export_to_torchscript(model, img_size, str(output_path))
        
        # 加载并推理
        loaded = torch.jit.load(str(output_path))
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        output = loaded(dummy_input)
        
        assert output.shape == (1, 2)
    
    def test_invalid_method(self, tmp_path):
        """测试无效的导出方法"""
        model, img_size = build_model('resnet18', num_classes=2)
        
        output_path = tmp_path / 'model.pt'
        
        with pytest.raises(ValueError, match="Unknown method"):
            export_to_torchscript(model, img_size, str(output_path), method='invalid')


class TestExportConsistency:
    """测试导出模型的一致性"""
    
    def test_onnx_output_consistency(self, tmp_path):
        """测试 ONNX 导出模型输出与原始模型一致"""
        pytest.importorskip('onnxruntime')
        import onnxruntime as ort
        import numpy as np
        
        model, img_size = build_model('resnet18', num_classes=2)
        model.eval()
        
        # 导出 ONNX
        output_path = tmp_path / 'model.onnx'
        export_to_onnx(model, img_size, str(output_path))
        
        # 创建测试输入
        torch.manual_seed(42)
        test_input = torch.randn(1, 3, img_size, img_size)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = model(test_input)
            pytorch_probs = torch.softmax(pytorch_output, dim=1)
        
        # ONNX 推理
        session = ort.InferenceSession(str(output_path))
        onnx_outputs = session.run(None, {'image': test_input.numpy()})
        onnx_logits = onnx_outputs[0]
        onnx_probs = onnx_outputs[1]
        
        # 比较输出
        assert np.allclose(pytorch_output.numpy(), onnx_logits, atol=1e-5)
        assert np.allclose(pytorch_probs.numpy(), onnx_probs, atol=1e-5)
    
    def test_torchscript_output_consistency(self, tmp_path):
        """测试 TorchScript 导出模型输出与原始模型一致"""
        model, img_size = build_model('resnet18', num_classes=2)
        model.eval()
        
        # 导出 TorchScript
        output_path = tmp_path / 'model.pt'
        export_to_torchscript(model, img_size, str(output_path))
        
        # 创建测试输入
        torch.manual_seed(42)
        test_input = torch.randn(1, 3, img_size, img_size)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = model(test_input)
        
        # TorchScript 推理
        loaded = torch.jit.load(str(output_path))
        with torch.no_grad():
            ts_output = loaded(test_input)
        
        # 比较输出
        assert torch.allclose(pytorch_output, ts_output, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

