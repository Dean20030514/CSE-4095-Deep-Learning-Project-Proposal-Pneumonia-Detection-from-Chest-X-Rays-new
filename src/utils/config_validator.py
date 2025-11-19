"""
配置文件验证模块

提供配置文件的完整性和有效性检查，防止运行时错误。
"""
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path


class ConfigValidator:
    """配置文件验证器"""
    
    # 必需字段及其类型
    REQUIRED_FIELDS = {
        'model': str,
        'img_size': int,
        'batch_size': int,
        'epochs': int,
        'lr': float,
    }
    
    # 可选字段及其类型
    OPTIONAL_FIELDS = {
        'pretrained': bool,
        'data_root': str,
        'num_workers': int,
        'weight_decay': float,
        'loss': str,
        'focal_gamma': float,
        'use_weighted_sampler': bool,
        'sampler': str,  # 新增：sampler 字段
        'augment_level': str,
        'augmentation': dict,  # 新增：嵌套的增强配置
        'optimizer': str,
        'scheduler': str,
        'warmup_epochs': int,
        'step_size': int,  # 新增：StepLR 配置
        'gamma': float,  # 新增：scheduler gamma
        'amp': bool,
        'output_dir': str,
        'save_best_only': bool,
        'save_last_interval': int,  # 新增：last checkpoint 保存间隔
        'seed': int,
        'val_interval': int,
        'use_albumentations': bool,
        'early_stopping': dict,  # 新增：early stopping 配置
        'focal': dict,  # 新增：focal loss 配置
        'allow_nondeterministic': bool,  # 新增：性能优化选项
        'allow_tf32': bool,  # 新增：TF32 选项
        'memory_efficient': bool,  # 新增：显存优化选项
    }
    
    # 有效值的枚举
    VALID_MODELS = [
        'resnet18', 'resnet50', 
        'efficientnet_b0', 'efficientnet_b2',
        'densenet121'
    ]
    
    VALID_LOSSES = ['cross_entropy', 'focal', 'weighted_ce']
    VALID_OPTIMIZERS = ['adam', 'adamw', 'sgd']
    VALID_SCHEDULERS = ['step', 'cosine', 'exponential', 'none']
    VALID_AUGMENT_LEVELS = ['light', 'medium', 'heavy', 'aggressive']
    VALID_SAMPLERS = ['weighted_random', 'random', 'none']
    
    @classmethod
    def validate(cls, config: Dict[str, Any]) -> None:
        """
        验证配置文件的完整性和有效性
        
        Args:
            config: 配置字典
        
        Raises:
            ValueError: 配置无效时抛出详细错误信息
        """
        errors: List[str] = []
        
        # 1. 检查必需字段
        for field, field_type in cls.REQUIRED_FIELDS.items():
            if field not in config:
                errors.append(f"❌ 缺少必需配置: {field}")
            elif not isinstance(config.get(field), field_type):
                actual_type = type(config[field]).__name__
                expected_type = field_type.__name__
                errors.append(
                    f"❌ 配置 '{field}' 类型错误: "
                    f"期望 {expected_type}, 实际 {actual_type}"
                )
        
        # 2. 检查可选字段类型
        for field, field_type in cls.OPTIONAL_FIELDS.items():
            if field in config and not isinstance(config[field], field_type):
                actual_type = type(config[field]).__name__
                expected_type = field_type.__name__
                errors.append(
                    f"⚠️ 可选配置 '{field}' 类型错误: "
                    f"期望 {expected_type}, 实际 {actual_type}"
                )
        
        # 如果基本类型检查失败，提前返回
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(errors))
        
        # 3. 验证模型名称
        model = config['model'].lower()
        if model not in cls.VALID_MODELS:
            errors.append(
                f"❌ 无效的模型: '{model}'. "
                f"支持的模型: {', '.join(cls.VALID_MODELS)}"
            )
        
        # 4. 验证超参数范围
        img_size = config['img_size']
        if img_size < 32 or img_size > 1024:
            errors.append(f"❌ img_size ({img_size}) 必须在 [32, 1024] 范围内")
        
        batch_size = config['batch_size']
        if batch_size < 1 or batch_size > 1024:
            errors.append(f"❌ batch_size ({batch_size}) 必须在 [1, 1024] 范围内")
        
        epochs = config['epochs']
        if epochs < 1 or epochs > 1000:
            errors.append(f"❌ epochs ({epochs}) 必须在 [1, 1000] 范围内")
        
        lr = config['lr']
        if lr <= 0 or lr > 1:
            errors.append(f"❌ lr ({lr}) 必须在 (0, 1] 范围内")
        
        # 5. 验证可选字段的有效值
        if 'loss' in config:
            loss = config['loss'].lower()
            if loss not in cls.VALID_LOSSES:
                errors.append(
                    f"❌ 无效的loss: '{loss}'. "
                    f"支持: {', '.join(cls.VALID_LOSSES)}"
                )
        
        if 'optimizer' in config:
            optimizer = config['optimizer'].lower()
            if optimizer not in cls.VALID_OPTIMIZERS:
                errors.append(
                    f"❌ 无效的optimizer: '{optimizer}'. "
                    f"支持: {', '.join(cls.VALID_OPTIMIZERS)}"
                )
        
        if 'scheduler' in config:
            scheduler = config['scheduler'].lower()
            if scheduler not in cls.VALID_SCHEDULERS:
                errors.append(
                    f"❌ 无效的scheduler: '{scheduler}'. "
                    f"支持: {', '.join(cls.VALID_SCHEDULERS)}"
                )
        
        if 'augment_level' in config:
            augment_level = config['augment_level'].lower()
            if augment_level not in cls.VALID_AUGMENT_LEVELS:
                errors.append(
                    f"❌ 无效的augment_level: '{augment_level}'. "
                    f"支持: {', '.join(cls.VALID_AUGMENT_LEVELS)}"
                )
        
        if 'sampler' in config:
            sampler = config['sampler'].lower()
            if sampler not in cls.VALID_SAMPLERS:
                errors.append(
                    f"❌ 无效的sampler: '{sampler}'. "
                    f"支持: {', '.join(cls.VALID_SAMPLERS)}"
                )
        
        # 6. 验证嵌套结构
        if 'early_stopping' in config:
            es_cfg = config['early_stopping']
            if not isinstance(es_cfg, dict):
                errors.append("❌ 'early_stopping' 必须是字典类型")
            elif 'patience' not in es_cfg:
                errors.append("❌ 'early_stopping' 缺少必需字段 'patience'")
            elif not isinstance(es_cfg['patience'], int) or es_cfg['patience'] < 0:
                errors.append("❌ 'early_stopping.patience' 必须是非负整数")
        
        if 'focal' in config:
            focal_cfg = config['focal']
            if not isinstance(focal_cfg, dict):
                errors.append("❌ 'focal' 必须是字典类型")
            elif 'gamma' in focal_cfg:
                gamma = focal_cfg['gamma']
                if not isinstance(gamma, (int, float)) or gamma < 0:
                    errors.append("❌ 'focal.gamma' 必须是非负数值")
        
        if 'augmentation' in config:
            aug_cfg = config['augmentation']
            if not isinstance(aug_cfg, dict):
                errors.append("❌ 'augmentation' 必须是字典类型")
        
        # 7. 验证逻辑一致性
        if 'focal_gamma' in config and config.get('loss', '').lower() != 'focal':
            errors.append(
                "⚠️ 警告: 配置了focal_gamma但loss不是'focal'"
            )
        
        if 'focal' in config and config.get('loss', '').lower() != 'focal':
            errors.append(
                "⚠️ 警告: 配置了focal但loss不是'focal'"
            )
        
        if 'weight_decay' in config:
            weight_decay = config['weight_decay']
            if weight_decay < 0 or weight_decay > 1:
                errors.append(
                    f"❌ weight_decay ({weight_decay}) 必须在 [0, 1] 范围内"
                )
        
        if 'num_workers' in config:
            num_workers = config['num_workers']
            if num_workers < 0 or num_workers > 32:
                errors.append(
                    f"⚠️ num_workers ({num_workers}) 超出推荐范围 [0, 32]"
                )
        
        # 如果有错误，抛出异常
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(errors))
        
        print("[OK] 配置验证通过")
    
    @classmethod
    def validate_file(cls, config_path: str) -> Dict[str, Any]:
        """
        从文件加载并验证配置
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            验证后的配置字典
        
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置无效
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        
        if config is None:
            raise ValueError("配置文件为空")
        
        cls.validate(config)
        
        return config


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """
    便捷函数：加载并验证配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        验证后的配置字典
    
    Example:
        >>> config = load_and_validate_config('configs/model.yaml')
        ✅ 配置验证通过
    """
    return ConfigValidator.validate_file(config_path)


if __name__ == '__main__':
    # 测试示例
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        try:
            config = load_and_validate_config(config_path)
            print(f"\n配置文件有效: {config_path}")
            print("\n主要配置:")
            print(f"  Model: {config['model']}")
            print(f"  Image Size: {config['img_size']}")
            print(f"  Batch Size: {config['batch_size']}")
            print(f"  Epochs: {config['epochs']}")
            print(f"  Learning Rate: {config['lr']}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n错误: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("用法: python src/utils/config_validator.py <config_file>")
        print("示例: python src/utils/config_validator.py src/configs/final_model.yaml")

