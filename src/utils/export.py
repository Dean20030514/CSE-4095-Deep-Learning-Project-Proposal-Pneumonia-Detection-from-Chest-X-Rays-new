"""
模型导出模块

提供将 PyTorch 模型导出为 ONNX 等格式的功能。
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, List


def export_to_onnx(
    model: nn.Module,
    img_size: int,
    output_path: str,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None
) -> Path:
    """
    将 PyTorch 模型导出为 ONNX 格式。
    
    Args:
        model: PyTorch 模型
        img_size: 输入图像大小
        output_path: 输出文件路径
        opset_version: ONNX opset 版本
        dynamic_batch: 是否支持动态 batch size
        input_names: 输入名称列表
        output_names: 输出名称列表
    
    Returns:
        导出文件的路径
    
    Example:
        >>> model, _ = build_model('resnet18', 2)
        >>> export_to_onnx(model, 224, 'model.onnx')
    """
    model.eval()
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # 设置动态轴
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'probabilities': {0: 'batch_size'}
        }
    
    # 默认名称
    if input_names is None:
        input_names = ['image']
    if output_names is None:
        output_names = ['logits', 'probabilities']
    
    # 包装模型以输出 logits 和 probabilities
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            return logits, probs
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # 导出
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"[OK] Model exported to ONNX: {output_path}")
    return output_path


def export_to_torchscript(
    model: nn.Module,
    img_size: int,
    output_path: str,
    method: str = 'trace'
) -> Path:
    """
    将 PyTorch 模型导出为 TorchScript 格式。
    
    Args:
        model: PyTorch 模型
        img_size: 输入图像大小
        output_path: 输出文件路径
        method: 导出方法 ('trace' 或 'script')
    
    Returns:
        导出文件的路径
    
    Example:
        >>> model, _ = build_model('resnet18', 2)
        >>> export_to_torchscript(model, 224, 'model.pt')
    """
    model.eval()
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == 'trace':
        dummy_input = torch.randn(1, 3, img_size, img_size)
        scripted_model = torch.jit.trace(model, dummy_input)
    elif method == 'script':
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
    
    scripted_model.save(str(output_path))
    
    print(f"[OK] Model exported to TorchScript: {output_path}")
    return output_path


def verify_onnx_model(onnx_path: str, img_size: int) -> bool:
    """
    验证导出的 ONNX 模型。
    
    Args:
        onnx_path: ONNX 模型路径
        img_size: 输入图像大小
    
    Returns:
        是否验证通过
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[WARNING] onnx or onnxruntime not installed. Skipping verification.")
        return True
    
    # 检查模型结构
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # 运行推理测试
    session = ort.InferenceSession(onnx_path)
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    
    outputs = session.run(None, {'image': dummy_input})
    
    # 检查输出形状
    assert len(outputs) == 2, "Expected 2 outputs (logits, probabilities)"
    assert outputs[0].shape[0] == 1, "Batch size mismatch"
    assert outputs[1].shape[0] == 1, "Batch size mismatch"
    
    # 检查概率和为 1
    prob_sum = outputs[1].sum(axis=1)
    assert np.allclose(prob_sum, 1.0, atol=1e-5), "Probabilities should sum to 1"
    
    print(f"[OK] ONNX model verification passed: {onnx_path}")
    return True


def export_model_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    formats: List[str] = None
) -> dict:
    """
    从 checkpoint 导出模型到多种格式。
    
    Args:
        checkpoint_path: checkpoint 文件路径
        output_dir: 输出目录
        formats: 导出格式列表 ['onnx', 'torchscript']
    
    Returns:
        导出文件路径字典
    
    Example:
        >>> paths = export_model_from_checkpoint('best_model.pt', 'exports/')
    """
    from src.models.factory import build_model
    
    if formats is None:
        formats = ['onnx', 'torchscript']
    
    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model_name = cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    num_classes = len(ckpt['classes'])
    
    # 构建模型
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_paths = {}
    
    # 导出各种格式
    if 'onnx' in formats:
        onnx_path = output_dir / f'{model_name}_pneumonia.onnx'
        export_to_onnx(model, img_size, str(onnx_path))
        exported_paths['onnx'] = str(onnx_path)
    
    if 'torchscript' in formats:
        ts_path = output_dir / f'{model_name}_pneumonia_scripted.pt'
        export_to_torchscript(model, img_size, str(ts_path))
        exported_paths['torchscript'] = str(ts_path)
    
    return exported_paths


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export trained model')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', default='exports', help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['onnx', 'torchscript'],
                       choices=['onnx', 'torchscript'], help='Export formats')
    args = parser.parse_args()
    
    paths = export_model_from_checkpoint(args.checkpoint, args.output_dir, args.formats)
    print(f"\nExported files:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

