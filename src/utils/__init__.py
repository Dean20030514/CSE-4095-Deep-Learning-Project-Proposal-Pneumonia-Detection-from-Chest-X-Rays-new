"""
工具模块

提供各种辅助工具和功能。
"""
from .device import get_device
from .metrics import compute_metrics, compute_sensitivity_specificity
from .gradcam import GradCAM
from .config_validator import ConfigValidator, load_and_validate_config
from .export import export_to_onnx, export_to_torchscript, export_model_from_checkpoint, export_quantized_model
from .lr_finder import LRFinder, find_lr
from .uncertainty import mc_dropout_predict, UncertaintyEstimator, compute_entropy
from .calibration import TemperatureScaling, compute_calibration_metrics
from .dataset_hash import compute_dataset_hash, verify_dataset_hash, save_dataset_hash
from .model_info import get_model_size, get_model_complexity, count_parameters

# 可选导入 (Pydantic 可能不可用)
try:
    from .config_schema import (
        TrainingConfig, 
        validate_config_with_pydantic, 
        load_and_validate_config_pydantic,
        PYDANTIC_AVAILABLE
    )
except ImportError:
    PYDANTIC_AVAILABLE = False

__all__ = [
    # 设备管理
    'get_device',
    # 评估指标
    'compute_metrics',
    'compute_sensitivity_specificity',
    # 可解释性
    'GradCAM',
    # 配置验证
    'ConfigValidator',
    'load_and_validate_config',
    # 模型导出
    'export_to_onnx',
    'export_to_torchscript',
    'export_model_from_checkpoint',
    'export_quantized_model',
    # 学习率查找
    'LRFinder',
    'find_lr',
    # 不确定性估计
    'mc_dropout_predict',
    'UncertaintyEstimator',
    'compute_entropy',
    # 校准
    'TemperatureScaling',
    'compute_calibration_metrics',
    # 数据集哈希
    'compute_dataset_hash',
    'verify_dataset_hash',
    'save_dataset_hash',
    # 模型信息
    'get_model_size',
    'get_model_complexity',
    'count_parameters',
]

