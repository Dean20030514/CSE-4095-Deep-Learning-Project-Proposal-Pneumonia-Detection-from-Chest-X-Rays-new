"""
单元测试：模型构建和前向传播
"""
import pytest
import torch
from src.models.factory import build_model


class TestModelFactory:
    """测试模型工厂函数"""
    
    @pytest.mark.parametrize("model_name,expected_img_size", [
        ("resnet18", 224),
        ("resnet50", 224),
        ("efficientnet_b0", 224),
        ("efficientnet_b2", 260),
        ("densenet121", 224),
    ])
    def test_build_model_architectures(self, model_name, expected_img_size):
        """测试不同架构的模型构建"""
        num_classes = 2
        model, img_size = build_model(model_name, num_classes)
        
        # 验证返回值
        assert model is not None
        assert isinstance(img_size, int)
        assert img_size == expected_img_size
        
        # 验证模型是nn.Module
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward_pass(self):
        """测试模型前向传播"""
        model, img_size = build_model("resnet18", num_classes=2)
        model.eval()
        
        # 创建虚拟输入
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, img_size, img_size)
        
        # 前向传播
        with torch.no_grad():
            output = model(dummy_input)
        
        # 验证输出形状
        assert output.shape == (batch_size, 2)
    
    def test_model_output_range(self):
        """测试模型输出的logits范围合理"""
        model, img_size = build_model("resnet18", num_classes=2)
        model.eval()
        
        dummy_input = torch.randn(2, 3, img_size, img_size)
        
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.softmax(logits, dim=1)
        
        # 验证概率和为1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
        
        # 验证概率在[0, 1]范围内
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_invalid_model_name(self):
        """测试无效模型名称抛出异常"""
        with pytest.raises(ValueError):
            build_model("invalid_model_name", num_classes=2)


class TestModelTraining:
    """测试模型训练相关功能"""
    
    def test_model_gradient_flow(self):
        """测试模型梯度正常流动"""
        model, _ = build_model("resnet18", num_classes=2)
        model.train()
        
        # 创建虚拟数据
        inputs = torch.randn(2, 3, 224, 224, requires_grad=True)
        targets = torch.tensor([0, 1])
        
        # 前向传播
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度存在
        assert inputs.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

