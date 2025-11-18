"""
训练配置文件生成器（优化版）

特点：
- 使用基础模板 + 差异配置，消除重复代码
- 数据驱动的配置生成
- 自动验证生成的配置
- 清晰的分类和注释
"""

import yaml
from pathlib import Path
from typing import Dict, Any


# 配置输出目录
CONFIG_DIR = Path("src/configs")
CONFIG_DIR.mkdir(exist_ok=True, parents=True)


def get_base_config() -> Dict[str, Any]:
    """
    获取基础配置模板
    
    优化针对：RTX 5070 (8GB) + Ryzen 9 9955HX (32线程) + 32GB RAM
    - batch_size: 增大以充分利用8GB显存
    - num_workers: 增大以充分利用32线程CPU
    """
    return {
        'pretrained': True,
        'data_root': 'data',
        'img_size': 384,
        'batch_size': 24,      # 优化: 12→24 (8GB显存)
        'num_workers': 12,     # 优化: 6→12 (32线程CPU)
        'epochs': 100,
        'lr': 0.0005,
        'weight_decay': 0.0001,
        'early_stopping': {'patience': 20},
        'loss': 'focal',
        'focal_gamma': 1.5,
        'use_weighted_sampler': True,
        'augmentation': {
            'horizontal_flip': 0.5,
            'rotation_degrees': 10,
            'brightness': 0.1,
            'contrast': 0.1
        },
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 2,
        'amp': True,
        'save_best_only': False,
        'seed': 42
    }


def merge_config(base: Dict, overrides: Dict) -> Dict:
    """合并配置，overrides覆盖base"""
    result = base.copy()
    result.update(overrides)
    return result


def create_config(filename: str, config_dict: Dict, description: str = "") -> bool:
    """
    创建单个配置文件
    
    Args:
        filename: 配置文件名
        config_dict: 配置字典
        description: 配置描述
    
    Returns:
        是否成功创建（False表示已存在）
    """
    filepath = CONFIG_DIR / filename
    
    if filepath.exists():
        print(f"  [SKIP] {filename} (已存在)")
        return False
    
    with open(filepath, 'w', encoding='utf-8') as f:
        if description:
            f.write(f"# {description}\n")
            f.write("# Auto-generated configuration file\n\n")
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"  [OK] {filename}")
    return True


def main():
    """主函数：生成所有配置文件"""
    
    print("\n" + "="*70)
    print("  配置文件生成器 v2.0（优化版）")
    print("="*70 + "\n")
    
    base_config = get_base_config()
    created_count = 0
    
    # ============================================================================
    # 1. 模型架构对比（5个）
    # ============================================================================
    print("[1/6] 生成模型架构对比配置...")
    
    models = [
        ('efficientnet_b0', 28, 'model_efficientnet_b0', "EfficientNet-B0 @ 384px - Lightweight model"),
        ('resnet18', 32, 'model_resnet18', "ResNet18 @ 384px - Fast baseline"),
        ('resnet50', 20, 'model_resnet50', "ResNet50 @ 384px - Medium capacity"),
        ('densenet121', 24, 'model_densenet121', "DenseNet121 @ 384px - Dense connections"),
        ('efficientnet_b2', 24, 'model_efficientnet_b2', "⭐ EfficientNet-B2 @ 384px - BEST MODEL (98.26%)"),
    ]
    
    for model, batch_size, filename, desc in models:
        config = merge_config(base_config, {
            'model': model,
            'batch_size': batch_size,
            'output_dir': f'runs/{filename}'
        })
        if create_config(f"{filename}.yaml", config, desc):
            created_count += 1
    
    # ============================================================================
    # 2. 学习率实验（3个）
    # ============================================================================
    print("\n[2/6] 生成学习率实验配置...")
    
    learning_rates = [
        (0.0001, 'lr_0.0001', "Learning Rate = 0.0001 - Conservative learning"),
        (0.0005, 'lr_0.0005', "⭐ Learning Rate = 0.0005 - OPTIMAL LR"),
        (0.001, 'lr_0.001', "Learning Rate = 0.001 - Aggressive learning"),
    ]
    
    for lr, filename, desc in learning_rates:
        config = merge_config(base_config, {
            'model': 'efficientnet_b2',
            'lr': lr,
            'output_dir': f'runs/{filename}'
        })
        if create_config(f"{filename}.yaml", config, desc):
            created_count += 1
    
    # ============================================================================
    # 3. 数据增强实验（3个）
    # ============================================================================
    print("\n[3/6] 生成数据增强实验配置...")
    
    augmentations = [
        ('light', {
            'horizontal_flip': 0.3,
            'rotation_degrees': 5,
            'brightness': 0.05,
            'contrast': 0.05
        }, "Light Augmentation - Minimal transformation"),
        
        ('medium', {
            'horizontal_flip': 0.5,
            'rotation_degrees': 10,
            'brightness': 0.1,
            'contrast': 0.1
        }, "Medium Augmentation - Balanced transformation"),
        
        ('aggressive', {
            'horizontal_flip': 0.5,
            'rotation_degrees': 15,
            'brightness': 0.2,
            'contrast': 0.2,
            'scale': [0.9, 1.1],
            'shear': 5
        }, "Aggressive Augmentation - Heavy transformation"),
    ]
    
    for aug_name, aug_params, desc in augmentations:
        config = merge_config(base_config, {
            'model': 'efficientnet_b2',
            'augmentation': aug_params,
            'output_dir': f'runs/aug_{aug_name}'
        })
        if create_config(f"aug_{aug_name}.yaml", config, desc):
            created_count += 1
    
    # ============================================================================
    # 4. 基线模型（3个）
    # ============================================================================
    print("\n[4/6] 生成基线模型配置...")
    
    baselines = [
        ('resnet18', 'baseline_resnet18', "Baseline ResNet18 - Simple baseline"),
        ('efficientnet_b0', 'baseline_efficientnet', "Baseline EfficientNet - Efficient baseline"),
        ('resnet18', 'full_resnet18', "Full ResNet18 - Complete training"),
    ]
    
    for model, filename, desc in baselines:
        config = merge_config(base_config, {
            'model': model,
            'img_size': 224,
            'batch_size': 48,  # 优化: 32→48 (224px更小，显存充足)
            'lr': 0.001 if 'baseline' in filename else 0.0005,
            'loss': 'cross_entropy' if 'baseline' in filename else 'focal',
            'use_weighted_sampler': False if 'baseline' in filename else True,
            'output_dir': f'runs/{filename}'
        })
        # 移除baseline配置中不需要的focal_gamma
        if 'baseline' in filename:
            del config['focal_gamma']
        
        if create_config(f"{filename}.yaml", config, desc):
            created_count += 1
    
    # ============================================================================
    # 5. 特殊配置（2个）
    # ============================================================================
    print("\n[5/6] 生成特殊用途配置...")
    
    # 最终高分辨率模型
    final_config = merge_config(base_config, {
        'model': 'efficientnet_b2',
        'img_size': 512,
        'batch_size': 16,  # 优化: 8→16 (8GB显存足够)
        'warmup_epochs': 3,
        'output_dir': 'runs/final_efficientnet_b2_512'
    })
    if create_config("final_model.yaml", final_config, 
                    "⭐ Final Model: EfficientNet-B2 @ 512px - Production ready"):
        created_count += 1
    
    # 医疗筛查优化
    screening_config = merge_config(base_config, {
        'model': 'resnet18',
        'img_size': 384,
        'batch_size': 32,  # 优化: 16→32 (ResNet18@384显存充足)
        'lr': 0.0002,
        'focal_gamma': 2.0,
        'output_dir': 'runs/medical_screening_optimized'
    })
    if create_config("medical_screening_optimized.yaml", screening_config,
                    "Medical Screening - Maximize pneumonia recall"):
        created_count += 1
    
    # ============================================================================
    # 6. 工具配置（1个）
    # ============================================================================
    print("\n[6/6] 生成工具配置...")
    
    # 快速测试配置
    quick_test_config = merge_config(base_config, {
        'model': 'resnet18',
        'img_size': 224,
        'batch_size': 64,  # 优化: 32→64 (ResNet18@224很小)
        'epochs': 3,
        'lr': 0.001,
        'warmup_epochs': 1,
        'num_workers': 16,  # 快速测试用最大workers
        'output_dir': 'runs/quick_test'
    })
    if create_config("quick_test_resnet18.yaml", quick_test_config,
                    "🔧 Quick Test - Fast prototyping (3 epochs)"):
        created_count += 1
    
    # ============================================================================
    # 汇总报告
    # ============================================================================
    print("\n" + "="*70)
    print(f"  [SUCCESS] 配置生成完成！")
    print("="*70)
    
    total_configs = len(list(CONFIG_DIR.glob('*.yaml')))
    print(f"\n[STATISTICS] 统计信息：")
    print(f"  - 本次新建：{created_count}个")
    print(f"  - 配置总数：{total_configs}个")
    print(f"  - 保存位置：{CONFIG_DIR.absolute()}")
    
    print(f"\n[CATEGORIES] 配置分类：")
    print(f"  - 模型架构对比：5个")
    print(f"  - 学习率实验：3个")
    print(f"  - 数据增强实验：3个")
    print(f"  - 基线模型：3个")
    print(f"  - 特殊配置：2个 (final_model, medical_screening)")
    print(f"  - 工具配置：1个 (quick_test)")
    print(f"  " + "-"*60)
    print(f"  总计：17个配置文件")
    
    print(f"\n[USAGE] 使用方法：")
    print(f"  # 快速测试")
    print(f"  python src/train.py --config src/configs/quick_test_resnet18.yaml")
    print(f"")
    print(f"  # 最佳模型")
    print(f"  python src/train.py --config src/configs/model_efficientnet_b2.yaml")
    print(f"")
    print(f"  # 批量训练")
    print(f"  .\\scripts\\automated_full_training.ps1")
    
    print(f"\n[DOCS] 文档：")
    print(f"  - TRAINING_GUIDE.md - 训练指南")
    print(f"  - src/configs/README.md - 配置说明")
    
    # 可选：验证生成的配置
    print(f"\n[VALIDATION] 验证配置（可选）：")
    print(f"  python src/utils/config_validator.py src/configs/final_model.yaml")
    
    print()


if __name__ == '__main__':
    main()
