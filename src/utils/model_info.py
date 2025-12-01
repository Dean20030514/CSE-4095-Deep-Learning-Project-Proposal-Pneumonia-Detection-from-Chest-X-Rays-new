"""
模型信息和复杂度分析模块

提供模型参数量、FLOPs 计算和内存占用估算功能。
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import sys


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    计算模型参数量。
    
    Args:
        model: PyTorch 模型
        trainable_only: 是否只计算可训练参数
    
    Returns:
        参数数量
    
    Example:
        >>> model = nn.Linear(100, 10)
        >>> print(count_parameters(model))  # 1010 (100*10 + 10)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    计算模型大小（参数量和估计内存占用）。
    
    Args:
        model: PyTorch 模型
    
    Returns:
        包含模型大小信息的字典：
        - total_params: 总参数量
        - trainable_params: 可训练参数量
        - frozen_params: 冻结参数量
        - size_mb: 模型大小（MB，假设 float32）
        - size_mb_fp16: 模型大小（MB，假设 float16）
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    
    # 参数大小估算（bytes）
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'size_mb': total_size / (1024 ** 2),
        'size_mb_fp16': (total_params * 2) / (1024 ** 2),  # float16 = 2 bytes
    }


def estimate_flops_simple(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    简单估算模型 FLOPs（不需要额外依赖）。
    
    注意：这是一个简化估算，只计算卷积和全连接层的 FLOPs。
    对于精确的 FLOPs 计算，建议使用 fvcore 或 ptflops。
    
    Args:
        model: PyTorch 模型
        input_size: 输入大小 (batch, channels, height, width)
        device: 设备
    
    Returns:
        包含 FLOPs 信息的字典
    """
    total_flops = 0
    layer_flops = {}
    
    def hook_fn(module, input, output, name):
        nonlocal total_flops
        flops = 0
        
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs = 2 * Cin * Cout * K * K * Hout * Wout
            batch_size = input[0].size(0)
            output_channels, input_channels = module.weight.size()[:2]
            kernel_size = module.kernel_size
            output_height, output_width = output.size()[2:]
            
            flops = (2 * input_channels * output_channels * 
                    kernel_size[0] * kernel_size[1] * 
                    output_height * output_width * batch_size)
            
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = 2 * in_features * out_features * batch_size
            batch_size = input[0].size(0)
            flops = 2 * module.in_features * module.out_features * batch_size
            
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # BatchNorm FLOPs ≈ 2 * num_features * spatial_size
            flops = 2 * module.num_features * input[0].numel() / input[0].size(0)
        
        if flops > 0:
            layer_flops[name] = flops
            total_flops += flops
    
    # 注册 hooks
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(
            lambda m, i, o, n=name: hook_fn(m, i, o, n)
        )
        hooks.append(hook)
    
    # 前向传播
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    model = model.to(device)
    
    with torch.no_grad():
        model(dummy_input)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    return {
        'total_flops': total_flops,
        'total_gflops': total_flops / 1e9,
        'total_mflops': total_flops / 1e6,
        'layer_flops': layer_flops,
    }


def try_fvcore_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: str = 'cpu'
) -> Optional[Dict[str, float]]:
    """
    使用 fvcore 计算精确的 FLOPs（如果可用）。
    
    Args:
        model: PyTorch 模型
        input_size: 输入大小
        device: 设备
    
    Returns:
        FLOPs 信息字典，如果 fvcore 不可用则返回 None
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        
        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(input_size).to(device)
        
        flops = FlopCountAnalysis(model, dummy_input)
        params = parameter_count(model)
        
        return {
            'total_flops': flops.total(),
            'total_gflops': flops.total() / 1e9,
            'total_mflops': flops.total() / 1e6,
            'params': sum(params.values()),
            'params_m': sum(params.values()) / 1e6,
            'flops_by_operator': dict(flops.by_operator()),
        }
    except ImportError:
        return None


def get_model_complexity(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu'
) -> Dict[str, any]:
    """
    获取完整的模型复杂度信息。
    
    Args:
        model: PyTorch 模型
        input_size: 输入大小
        device: 设备
    
    Returns:
        包含参数量和 FLOPs 的综合信息字典
    """
    # 参数信息
    size_info = get_model_size(model)
    
    # 尝试使用 fvcore 计算精确 FLOPs
    flops_info = try_fvcore_flops(model, input_size, device)
    
    if flops_info is None:
        # 使用简单估算
        flops_info = estimate_flops_simple(model, input_size, device)
        flops_info['method'] = 'simple_estimate'
    else:
        flops_info['method'] = 'fvcore'
    
    return {
        **size_info,
        **flops_info,
    }


def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu'
) -> None:
    """
    打印模型摘要信息。
    
    Args:
        model: PyTorch 模型
        input_size: 输入大小
        device: 设备
    """
    info = get_model_complexity(model, input_size, device)
    
    print("\n" + "=" * 60)
    print("Model Complexity Summary")
    print("=" * 60)
    
    print(f"\nParameter Counts:")
    print(f"  Total Parameters:     {info['total_params']:,}")
    print(f"  Trainable Parameters: {info['trainable_params']:,}")
    print(f"  Frozen Parameters:    {info['frozen_params']:,}")
    print(f"  Params (M):           {info['total_params'] / 1e6:.2f} M")
    
    print(f"\nModel Size:")
    print(f"  FP32 Size:            {info['size_mb']:.2f} MB")
    print(f"  FP16 Size:            {info['size_mb_fp16']:.2f} MB")
    
    print(f"\nCompute (FLOPs):")
    print(f"  Total FLOPs:          {info['total_flops']:,}")
    print(f"  GFLOPs:               {info['total_gflops']:.2f}")
    print(f"  Calculation Method:   {info.get('method', 'unknown')}")
    
    print(f"\nInput Size: {input_size}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # 测试示例
    from src.models.factory import build_model
    
    print("Testing model complexity analysis...")
    
    for model_name in ['resnet18', 'resnet50', 'densenet121']:
        print(f"\n--- {model_name} ---")
        model, img_size = build_model(model_name, num_classes=2)
        print_model_summary(model, input_size=(1, 3, img_size, img_size))

