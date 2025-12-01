"""
最优学习率查找脚本

使用 Learning Rate Range Test 方法找到最优学习率。

使用方法:
    python scripts/find_optimal_lr.py --config src/configs/model_efficientnet_b2.yaml
    python scripts/find_optimal_lr.py --model resnet18 --data_root data
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description="Find optimal learning rate")
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--model', type=str, default='efficientnet_b2', help='Model name')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--start_lr', type=float, default=1e-7, help='Starting learning rate')
    parser.add_argument('--end_lr', type=float, default=1.0, help='Ending learning rate')
    parser.add_argument('--num_iters', type=int, default=100, help='Number of iterations')
    parser.add_argument('--output', type=str, default='reports/lr_finder', help='Output directory')
    args = parser.parse_args()
    
    from src.models.factory import build_model
    from src.data.datamodule import build_dataloaders
    from src.utils.device import get_device
    from src.utils.lr_finder import LRFinder
    
    device = get_device()
    print(f"\n[LR FINDER] Starting learning rate range test")
    print(f"  Device: {device}")
    
    # 加载配置
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg.get('model', args.model)
        img_size = cfg.get('img_size', args.img_size)
        batch_size = cfg.get('batch_size', args.batch_size)
        data_root = cfg.get('data_root', args.data_root)
    else:
        model_name = args.model
        img_size = args.img_size
        batch_size = args.batch_size
        data_root = args.data_root
    
    print(f"  Model: {model_name}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR range: [{args.start_lr}, {args.end_lr}]")
    print(f"  Iterations: {args.num_iters}")
    
    # 创建模型
    print(f"\n[MODEL] Building {model_name}...")
    model = build_model(model_name, pretrained=True, num_classes=2).to(device)
    
    # 加载数据
    print(f"\n[DATA] Loading data from {data_root}...")
    loaders, class_weights = build_dataloaders(
        data_root,
        img_size=img_size,
        batch_size=batch_size,
        use_weighted_sampler=False,
        num_workers=0
    )
    train_loader = loaders['train']
    
    # 创建损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.start_lr)
    
    # 运行 LR Range Test
    print(f"\n[LR TEST] Running LR range test...")
    lr_finder = LRFinder(model, optimizer, criterion, device=str(device))
    
    lr_finder.range_test(
        train_loader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iters,
        step_mode='exp'
    )
    
    # 找到建议的学习率
    suggested_lr = lr_finder.suggest_lr()
    
    print(f"\n{'='*60}")
    print("Learning Rate Range Test Results")
    print(f"{'='*60}")
    print(f"\n  Suggested Learning Rate: {suggested_lr:.6f}")
    print(f"\n  Common suggestions based on this result:")
    print(f"    - Conservative (slower): {suggested_lr/10:.6f}")
    print(f"    - Standard: {suggested_lr:.6f}")
    print(f"    - Aggressive (faster): {suggested_lr*2:.6f}")
    
    if args.config:
        print(f"\n  To use in your config ({args.config}):")
        print(f"    lr: {suggested_lr:.6f}")
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图表
    print(f"\n[PLOT] Saving LR finder plot...")
    lr_finder.plot(output_path=str(output_dir / 'lr_finder_plot.png'))
    
    # 保存数据
    import json
    results = {
        'model': model_name,
        'suggested_lr': suggested_lr,
        'start_lr': args.start_lr,
        'end_lr': args.end_lr,
        'num_iters': args.num_iters,
        'lrs': lr_finder.lrs,
        'losses': lr_finder.losses
    }
    
    with open(output_dir / 'lr_finder_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[SAVE] Results saved to {output_dir}")
    print(f"{'='*60}")
    
    # 重置模型和优化器状态
    lr_finder.reset()
    
    return suggested_lr


if __name__ == '__main__':
    main()

