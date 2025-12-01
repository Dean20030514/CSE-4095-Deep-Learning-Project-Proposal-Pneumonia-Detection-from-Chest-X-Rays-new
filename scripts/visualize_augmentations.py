"""
数据增强可视化工具

用于可视化和比较不同数据增强配置对图像的影响。
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_image(image_path: str) -> Image.Image:
    """加载图像"""
    return Image.open(image_path).convert('RGB')


def visualize_augmentations(
    image_path: str,
    config_path: str = None,
    augment_level: str = 'medium',
    num_samples: int = 9,
    output_path: str = None,
    use_albumentations: bool = True
):
    """
    可视化数据增强效果。
    
    Args:
        image_path: 输入图像路径
        config_path: 配置文件路径（可选）
        augment_level: 增强级别 ('light', 'medium', 'heavy', 'aggressive')
        num_samples: 生成的增强样本数量
        output_path: 输出图像路径（可选）
        use_albumentations: 是否使用 albumentations
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return
    
    # 加载配置
    aug_config = {}
    if config_path:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            aug_config = cfg.get('augmentation', {})
            augment_level = cfg.get('augment_level', augment_level)
            use_albumentations = cfg.get('use_albumentations', use_albumentations)
    
    # 加载原始图像
    original = load_image(image_path)
    img_size = 224  # 默认尺寸
    
    # 获取变换
    from src.data.datamodule import _albumentations_transforms, _torchvision_transforms
    
    if use_albumentations:
        try:
            transform = _albumentations_transforms(
                img_size, 
                is_train=True, 
                augment_level=augment_level,
                aug_config=aug_config
            )
            transform_type = "Albumentations"
        except ImportError:
            print("Albumentations not available, falling back to torchvision")
            transform = _torchvision_transforms(img_size, is_train=True)
            transform_type = "TorchVision"
    else:
        transform = _torchvision_transforms(img_size, is_train=True)
        transform_type = "TorchVision"
    
    # 生成增强样本
    print(f"Generating {num_samples} augmented samples...")
    print(f"Transform library: {transform_type}")
    print(f"Augmentation level: {augment_level}")
    
    augmented_images = []
    for i in range(num_samples):
        if use_albumentations and transform_type == "Albumentations":
            # Albumentations 需要 numpy 数组
            img_np = np.array(original)
            augmented = transform(image=img_np)['image']
            # 转回 PIL 格式用于显示（反归一化）
            augmented_display = augmented.permute(1, 2, 0).numpy()
            augmented_display = augmented_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            augmented_display = (augmented_display * 255).clip(0, 255).astype(np.uint8)
        else:
            # TorchVision
            augmented = transform(original)
            augmented_display = augmented.permute(1, 2, 0).numpy()
            augmented_display = augmented_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            augmented_display = (augmented_display * 255).clip(0, 255).astype(np.uint8)
        
        augmented_images.append(augmented_display)
    
    # 创建可视化
    cols = int(np.ceil(np.sqrt(num_samples + 1)))
    rows = int(np.ceil((num_samples + 1) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # 显示原始图像
    axes[0].imshow(original.resize((img_size, img_size)))
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 显示增强后的图像
    for i, aug_img in enumerate(augmented_images):
        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(f"Augmented #{i + 1}", fontsize=10)
        axes[i + 1].axis('off')
    
    # 隐藏空白子图
    for j in range(num_samples + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(
        f"Data Augmentation Visualization\n"
        f"Library: {transform_type} | Level: {augment_level}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_augmentation_levels(
    image_path: str,
    output_path: str = None
):
    """
    比较不同增强级别的效果。
    
    Args:
        image_path: 输入图像路径
        output_path: 输出路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required")
        return
    
    from src.data.datamodule import _albumentations_transforms
    
    original = load_image(image_path)
    img_size = 224
    
    levels = ['light', 'medium', 'heavy', 'aggressive']
    
    fig, axes = plt.subplots(2, len(levels) + 1, figsize=(4 * (len(levels) + 1), 8))
    
    # 第一行：原始 + 各级别的一个样本
    axes[0, 0].imshow(original.resize((img_size, img_size)))
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    for i, level in enumerate(levels):
        try:
            transform = _albumentations_transforms(img_size, is_train=True, augment_level=level)
            img_np = np.array(original)
            augmented = transform(image=img_np)['image']
            
            augmented_display = augmented.permute(1, 2, 0).numpy()
            augmented_display = augmented_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            augmented_display = (augmented_display * 255).clip(0, 255).astype(np.uint8)
            
            axes[0, i + 1].imshow(augmented_display)
            axes[0, i + 1].set_title(f"{level.capitalize()}", fontsize=12, fontweight='bold')
            axes[0, i + 1].axis('off')
        except Exception as e:
            axes[0, i + 1].text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center')
            axes[0, i + 1].axis('off')
    
    # 第二行：每个级别的另一个样本
    axes[1, 0].axis('off')
    
    for i, level in enumerate(levels):
        try:
            transform = _albumentations_transforms(img_size, is_train=True, augment_level=level)
            img_np = np.array(original)
            augmented = transform(image=img_np)['image']
            
            augmented_display = augmented.permute(1, 2, 0).numpy()
            augmented_display = augmented_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            augmented_display = (augmented_display * 255).clip(0, 255).astype(np.uint8)
            
            axes[1, i + 1].imshow(augmented_display)
            axes[1, i + 1].set_title(f"Sample 2", fontsize=10)
            axes[1, i + 1].axis('off')
        except Exception as e:
            axes[1, i + 1].axis('off')
    
    plt.suptitle("Augmentation Level Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize data augmentations")
    parser.add_argument('image', type=str, help='Input image path')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--level', type=str, default='medium',
                       choices=['light', 'medium', 'heavy', 'aggressive'],
                       help='Augmentation level')
    parser.add_argument('--samples', type=int, default=9, help='Number of samples')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare all augmentation levels')
    parser.add_argument('--no-albumentations', action='store_true',
                       help='Use torchvision instead of albumentations')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    if args.compare:
        compare_augmentation_levels(args.image, args.output)
    else:
        visualize_augmentations(
            args.image,
            config_path=args.config,
            augment_level=args.level,
            num_samples=args.samples,
            output_path=args.output,
            use_albumentations=not args.no_albumentations
        )


if __name__ == '__main__':
    main()

