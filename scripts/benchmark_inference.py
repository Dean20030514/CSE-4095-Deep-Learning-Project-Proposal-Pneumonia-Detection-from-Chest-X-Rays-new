#!/usr/bin/env python3
"""
推理性能基准测试

测量模型在不同配置下的推理速度和吞吐量。
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.factory import build_model
from src.utils.device import get_device


def benchmark_model(
    model: nn.Module,
    img_size: int,
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    num_warmup: int = 10,
    num_iterations: int = 100,
    use_amp: bool = False,
    device: str = 'cuda'
) -> Dict:
    """
    基准测试模型推理性能。
    
    Args:
        model: PyTorch 模型
        img_size: 输入图像大小
        batch_sizes: 要测试的批次大小列表
        num_warmup: 预热迭代次数
        num_iterations: 测试迭代次数
        use_amp: 是否使用混合精度
        device: 设备
    
    Returns:
        性能结果字典
    """
    model = model.to(device)
    model.eval()
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Inference Benchmark")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"{'='*60}")
    print(f"\n{'Batch Size':<12} {'Latency (ms)':<15} {'Throughput (img/s)':<20} {'Memory (MB)':<15}")
    print("-" * 62)
    
    for batch_size in batch_sizes:
        # 创建输入
        input_tensor = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        # 预热
        for _ in range(num_warmup):
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 测试
        times = []
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        # 计算统计信息
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time
        
        # 获取内存使用
        if device == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_mb = 0
        
        results[batch_size] = {
            'latency_ms': avg_time * 1000,
            'latency_std_ms': std_time * 1000,
            'throughput_fps': throughput,
            'memory_mb': memory_mb
        }
        
        print(f"{batch_size:<12} {avg_time*1000:.2f} ± {std_time*1000:.2f} ms    "
              f"{throughput:.1f}                  {memory_mb:.1f}")
    
    return results


def benchmark_checkpoint(
    checkpoint_path: str,
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    num_iterations: int = 100,
    use_amp: bool = False
) -> Dict:
    """
    从 checkpoint 加载模型并进行基准测试。
    
    Args:
        checkpoint_path: checkpoint 文件路径
        batch_sizes: 批次大小列表
        num_iterations: 迭代次数
        use_amp: 是否使用 AMP
    
    Returns:
        性能结果
    """
    # 加载 checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model_name = cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    num_classes = len(ckpt['classes'])
    
    print(f"Model: {model_name}")
    print(f"Image size: {img_size}")
    print(f"Classes: {num_classes}")
    
    # 构建模型
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    
    device = get_device()
    
    return benchmark_model(
        model, img_size, batch_sizes, 
        num_iterations=num_iterations,
        use_amp=use_amp,
        device=str(device)
    )


def compare_models(
    model_configs: List[Dict],
    img_size: int = 224,
    batch_size: int = 16,
    num_iterations: int = 100
) -> None:
    """
    比较多个模型的性能。
    
    Args:
        model_configs: 模型配置列表 [{'name': 'resnet18', 'num_classes': 2}, ...]
        img_size: 图像大小
        batch_size: 批次大小
        num_iterations: 迭代次数
    """
    device = get_device()
    
    print(f"\n{'='*70}")
    print(f"Model Comparison (img_size={img_size}, batch_size={batch_size})")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'Params (M)':<12} {'Latency (ms)':<15} {'Throughput':<15}")
    print("-" * 62)
    
    for config in model_configs:
        model_name = config['name']
        num_classes = config.get('num_classes', 2)
        
        model, _ = build_model(model_name, num_classes)
        model = model.to(device)
        model.eval()
        
        # 计算参数量
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # 创建输入
        input_tensor = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        if str(device) == 'cuda':
            torch.cuda.synchronize()
        
        # 测试
        times = []
        for _ in range(num_iterations):
            if str(device) == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            
            if str(device) == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        
        print(f"{model_name:<20} {num_params:.2f}         {avg_time*1000:.2f}          {throughput:.1f} img/s")
        
        del model
        if str(device) == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Benchmark model inference performance')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b2', 'densenet121'],
                       help='Model architecture')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32],
                       help='Batch sizes to test')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--compare', action='store_true', help='Compare all supported models')
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较所有支持的模型
        model_configs = [
            {'name': 'resnet18', 'num_classes': 2},
            {'name': 'resnet50', 'num_classes': 2},
            {'name': 'efficientnet_b0', 'num_classes': 2},
            {'name': 'efficientnet_b2', 'num_classes': 2},
            {'name': 'densenet121', 'num_classes': 2},
        ]
        compare_models(model_configs, img_size=args.img_size, 
                      batch_size=args.batch_sizes[0] if args.batch_sizes else 16,
                      num_iterations=args.num_iterations)
    
    elif args.checkpoint:
        # 测试指定的 checkpoint
        benchmark_checkpoint(
            args.checkpoint,
            batch_sizes=args.batch_sizes,
            num_iterations=args.num_iterations,
            use_amp=args.amp
        )
    
    else:
        # 测试指定的模型架构
        device = get_device()
        model, suggested_size = build_model(args.model, num_classes=2)
        img_size = args.img_size or suggested_size
        
        benchmark_model(
            model, img_size,
            batch_sizes=args.batch_sizes,
            num_iterations=args.num_iterations,
            use_amp=args.amp,
            device=str(device)
        )


if __name__ == '__main__':
    main()

