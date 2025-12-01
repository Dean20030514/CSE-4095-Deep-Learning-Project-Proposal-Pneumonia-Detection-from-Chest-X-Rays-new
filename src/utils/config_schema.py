"""
配置文件 Pydantic Schema 验证模块

使用 Pydantic 进行更强大的配置验证，支持类型检查、默认值和自定义验证器。
"""
from typing import Dict, Any, Optional, Literal, Union
from pathlib import Path

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore


if PYDANTIC_AVAILABLE:
    
    class FocalLossConfig(BaseModel):
        """Focal Loss 配置"""
        gamma: float = Field(default=1.5, ge=0.0, le=5.0, description="Focal loss gamma parameter")
    
    
    class EarlyStoppingConfig(BaseModel):
        """Early Stopping 配置"""
        patience: int = Field(default=10, ge=1, description="Epochs to wait before stopping")
        min_delta: float = Field(default=0.001, ge=0, description="Minimum improvement threshold")
    
    
    class AugmentationConfig(BaseModel):
        """数据增强配置"""
        horizontal_flip: float = Field(default=0.5, ge=0.0, le=1.0)
        rotation_limit: int = Field(default=15, ge=0, le=180)
        brightness_limit: float = Field(default=0.2, ge=0.0, le=1.0)
        contrast_limit: float = Field(default=0.2, ge=0.0, le=1.0)
        blur_limit: int = Field(default=3, ge=0, le=11)
        noise_var_limit: float = Field(default=0.02, ge=0.0, le=0.5)
    
    
    class TrainingConfig(BaseModel):
        """完整训练配置 Schema"""
        
        # 必需字段
        model: Literal['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b2', 'densenet121'] = Field(
            description="Model architecture name"
        )
        img_size: int = Field(ge=32, le=1024, description="Input image size")
        batch_size: int = Field(ge=1, le=1024, description="Training batch size")
        epochs: int = Field(ge=1, le=1000, description="Number of training epochs")
        lr: float = Field(gt=0, le=1, description="Learning rate")
        
        # 可选字段
        pretrained: bool = Field(default=True, description="Use pretrained weights")
        data_root: str = Field(default="data", description="Data root directory")
        num_workers: int = Field(default=4, ge=0, le=32, description="DataLoader workers")
        weight_decay: float = Field(default=0.01, ge=0, le=1, description="Weight decay")
        
        loss: Literal['cross_entropy', 'focal', 'weighted_ce'] = Field(
            default='focal', description="Loss function"
        )
        focal: Optional[FocalLossConfig] = Field(default=None, description="Focal loss config")
        focal_gamma: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="Legacy focal gamma")
        
        optimizer: Literal['adam', 'adamw', 'sgd'] = Field(default='adamw', description="Optimizer")
        scheduler: Literal['step', 'cosine', 'exponential', 'none'] = Field(
            default='cosine', description="LR scheduler"
        )
        warmup_epochs: int = Field(default=0, ge=0, description="Warmup epochs")
        
        sampler: Literal['weighted_random', 'random', 'none'] = Field(
            default='weighted_random', description="Data sampler"
        )
        use_weighted_sampler: bool = Field(default=True, description="Use weighted sampler")
        
        augment_level: Literal['light', 'medium', 'heavy', 'aggressive'] = Field(
            default='medium', description="Augmentation level"
        )
        augmentation: Optional[AugmentationConfig] = Field(
            default=None, description="Custom augmentation config"
        )
        use_albumentations: bool = Field(default=True, description="Use albumentations library")
        
        amp: bool = Field(default=True, description="Use automatic mixed precision")
        bfloat16: bool = Field(default=False, description="Use bfloat16 for AMP")
        
        seed: int = Field(default=42, ge=0, description="Random seed")
        output_dir: str = Field(default="runs", description="Output directory")
        save_best_only: bool = Field(default=True, description="Only save best model")
        
        early_stopping: Optional[EarlyStoppingConfig] = Field(
            default=None, description="Early stopping config"
        )
        
        use_tensorboard: bool = Field(default=False, description="Enable TensorBoard logging")
        gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation")
        
        @field_validator('img_size')
        @classmethod
        def validate_img_size(cls, v: int) -> int:
            """验证图像尺寸是否合理"""
            if v % 32 != 0:
                raise ValueError(f'img_size ({v}) should be divisible by 32 for optimal performance')
            return v
        
        @field_validator('batch_size')
        @classmethod
        def validate_batch_size(cls, v: int) -> int:
            """验证 batch_size 是否为 2 的幂"""
            if v > 0 and (v & (v - 1)) != 0 and v > 16:
                import warnings
                warnings.warn(
                    f'batch_size ({v}) is not a power of 2, '
                    f'which may reduce GPU efficiency',
                    UserWarning
                )
            return v
        
        @model_validator(mode='after')
        def validate_focal_config(self) -> 'TrainingConfig':
            """验证 focal loss 配置一致性"""
            if self.loss != 'focal':
                if self.focal is not None or self.focal_gamma is not None:
                    import warnings
                    warnings.warn(
                        f"focal/focal_gamma configured but loss is '{self.loss}', "
                        f"focal settings will be ignored",
                        UserWarning
                    )
            return self
        
        def get_focal_gamma(self) -> float:
            """获取 focal gamma 值，支持两种配置格式"""
            if self.focal is not None:
                return self.focal.gamma
            if self.focal_gamma is not None:
                return self.focal_gamma
            return 1.5  # 默认值


def validate_config_with_pydantic(config: Dict[str, Any]) -> 'TrainingConfig':
    """
    使用 Pydantic 验证配置。
    
    Args:
        config: 配置字典
    
    Returns:
        验证后的 TrainingConfig 对象
    
    Raises:
        ImportError: Pydantic 未安装
        ValidationError: 配置验证失败
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is not installed. "
            "Install it with: pip install pydantic>=2.0.0"
        )
    
    return TrainingConfig(**config)


def load_and_validate_config_pydantic(config_path: str) -> 'TrainingConfig':
    """
    从文件加载并使用 Pydantic 验证配置。
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        验证后的 TrainingConfig 对象
    """
    import yaml
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config file is empty")
    
    return validate_config_with_pydantic(config)


if __name__ == '__main__':
    # 测试示例
    import sys
    
    if PYDANTIC_AVAILABLE:
        print("Pydantic is available, testing schema validation...")
        
        # 测试有效配置
        valid_config = {
            'model': 'resnet18',
            'img_size': 224,
            'batch_size': 32,
            'epochs': 50,
            'lr': 0.001,
        }
        
        try:
            config = validate_config_with_pydantic(valid_config)
            print(f"✅ Valid config: {config.model} @ {config.img_size}px")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
        
        # 测试无效配置
        invalid_config = {
            'model': 'invalid_model',
            'img_size': 10,  # Too small
            'batch_size': 2000,  # Too large
            'epochs': 50,
            'lr': 0.001,
        }
        
        try:
            config = validate_config_with_pydantic(invalid_config)
            print(f"Unexpected success for invalid config")
        except Exception as e:
            print(f"✅ Correctly caught invalid config: {type(e).__name__}")
    else:
        print("⚠️ Pydantic not installed. Install with: pip install pydantic>=2.0.0")

