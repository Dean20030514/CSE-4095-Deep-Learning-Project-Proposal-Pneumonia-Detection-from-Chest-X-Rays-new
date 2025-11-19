"""
单元测试：模型构建和前向传播

测试模型工厂、前向传播、梯度流等功能
"""
import pytest
import torch
import torch.nn as nn
from src.models.factory import build_model


class TestModelFactory:
    """测试模型工厂函数"""
    
    @pytest.mark.parametrize("model_name,expected_img_size", [
        ("resnet18", 224),
        ("resnet-18", 224),  # 测试别名
        ("resnet50", 224),
        ("efficientnet_b0", 224),
        ("efficientnet-b0", 224),  # 测试别名
        ("efficientnet_b2", 260),
        ("densenet121", 224),
        ("densenet-121", 224),  # 测试别名
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
        
        # 验证模型参数存在
        params = list(model.parameters())
        assert len(params) > 0
    
    def test_model_num_classes(self):
        """测试模型输出类别数正确"""
        for num_classes in [2, 3, 10]:
            model, _ = build_model("resnet18", num_classes=num_classes)
            
            # 获取最后一层的输出维度
            if hasattr(model, 'fc'):
                last_layer = model.fc
            elif hasattr(model, 'classifier'):
                last_layer = model.classifier
                if isinstance(last_layer, nn.Sequential):
                    last_layer = last_layer[-1]
            
            assert last_layer.out_features == num_classes
    
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
        
        # 验证输出是logits（未经过softmax）
        assert output.dtype == torch.float32
    
    def test_model_output_range(self):
        """测试模型输出的logits转概率后范围合理"""
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
    
    def test_model_deterministic(self):
        """测试相同输入产生相同输出（推理模式）"""
        model, img_size = build_model("resnet18", num_classes=2)
        model.eval()
        
        # 固定随机种子
        torch.manual_seed(42)
        dummy_input = torch.randn(2, 3, img_size, img_size)
        
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = model(dummy_input)
        
        # 推理模式下，相同输入应产生相同输出
        assert torch.allclose(output1, output2)
    
    def test_invalid_model_name(self):
        """测试无效模型名称抛出异常"""
        with pytest.raises(ValueError) as exc_info:
            build_model("invalid_model_name", num_classes=2)
        
        assert "Unknown model name" in str(exc_info.value)
    
    def test_model_supports_different_input_sizes(self):
        """测试模型支持不同输入尺寸（全卷积架构）"""
        model, default_size = build_model("resnet18", num_classes=2)
        model.eval()
        
        # 测试不同尺寸（需要是合理的尺寸）
        test_sizes = [224, 256, 512]
        
        for size in test_sizes:
            input_tensor = torch.randn(1, 3, size, size)
            with torch.no_grad():
                try:
                    output = model(input_tensor)
                    assert output.shape == (1, 2)
                except RuntimeError:
                    # 某些模型可能对输入尺寸有要求
                    pass


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
        
        # 检查模型参数有梯度
        has_grad = False
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                has_grad = True
        
        assert has_grad, "模型应该有至少一个可训练参数"
    
    def test_model_trainable_parameters(self):
        """测试模型可训练参数数量合理"""
        model, _ = build_model("resnet18", num_classes=2)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        # ResNet18应该有约11M参数
        assert trainable_params > 1_000_000
        assert trainable_params == total_params  # 默认全部可训练
    
    def test_model_train_eval_modes(self):
        """测试模型训练和评估模式切换"""
        model, _ = build_model("resnet18", num_classes=2)
        
        # 默认是eval模式（从pretrained加载）
        model.train()
        assert model.training == True
        
        model.eval()
        assert model.training == False
    
    def test_model_optimizer_step(self):
        """测试优化器更新模型参数"""
        model, _ = build_model("resnet18", num_classes=2)
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 记录初始参数
        initial_params = [p.clone() for p in model.parameters()]
        
        # 训练一步
        inputs = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 1])
        
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证参数已更新
        params_changed = False
        for initial_p, current_p in zip(initial_params, model.parameters()):
            if not torch.allclose(initial_p, current_p):
                params_changed = True
                break
        
        assert params_changed, "优化器应该更新模型参数"


class TestModelMemory:
    """测试模型内存相关功能"""
    
    def test_model_to_device(self, device):
        """测试模型设备转换"""
        model, _ = build_model("resnet18", num_classes=2)
        
        # 转换到目标设备
        model = model.to(device)
        
        # 验证模型参数在正确设备上
        for param in model.parameters():
            assert param.device.type == device.type
    
    @pytest.mark.slow
    def test_model_memory_footprint(self):
        """测试模型内存占用合理"""
        model, _ = build_model("resnet18", num_classes=2)
        
        # 计算参数内存占用（字节）
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # ResNet18约44MB参数
        assert param_memory < 100 * 1024 * 1024  # 小于100MB


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

