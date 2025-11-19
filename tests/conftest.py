"""
Pytest配置和共享fixtures

提供测试所需的共享fixtures和配置
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch


# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """返回项目根目录路径"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_config():
    """返回示例训练配置字典"""
    return {
        'model': 'resnet18',
        'img_size': 224,
        'batch_size': 16,
        'epochs': 10,
        'lr': 0.001,
        'seed': 42,
        'loss': 'focal',
        'focal': {
            'gamma': 1.5
        },
        'weight_decay': 1e-4,
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'amp': True,
        'num_workers': 0,
        'use_albumentations': False,
        'scheduler': 'cosine',
        'output_dir': 'runs/test',
        'val_interval': 1,
        'save_last_interval': 5,
        'allow_nondeterministic': False,
        'allow_tf32': True,
        'memory_efficient': False,
        'early_stopping': {
            'patience': 10
        }
    }


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """
    创建临时测试数据集
    
    结构:
    tmp_path/
        train/
            NORMAL/ (5 images)
            PNEUMONIA/ (5 images)
        val/
            NORMAL/ (3 images)
            PNEUMONIA/ (3 images)
        test/
            NORMAL/ (3 images)
            PNEUMONIA/ (3 images)
    """
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            (tmp_path / split / cls).mkdir(parents=True, exist_ok=True)
            
            # 根据split决定图像数量
            num_images = 5 if split == 'train' else 3
            
            # 创建测试图像（224x224 RGB）
            for i in range(num_images):
                # 创建随机图像
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(tmp_path / split / cls / f'img_{i}.jpg')
    
    return tmp_path


@pytest.fixture
def mock_checkpoint(tmp_path, sample_config):
    """创建模拟的模型checkpoint"""
    from src.models.factory import build_model
    
    model, _ = build_model('resnet18', num_classes=2)
    
    checkpoint = {
        'model': model.state_dict(),
        'classes': {'NORMAL': 0, 'PNEUMONIA': 1},
        'config': sample_config,
        'epoch': 10,
        'best_score': 0.95,
        'metrics': {
            'val_acc': 0.95,
            'pneumonia_recall': 0.96,
            'macro_f1': 0.94
        }
    }
    
    ckpt_path = tmp_path / 'test_model.pt'
    torch.save(checkpoint, ckpt_path)
    
    return ckpt_path


@pytest.fixture
def sample_predictions():
    """返回示例预测结果（用于指标测试）"""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])  # 75% accuracy
    y_probs = np.array([
        [0.9, 0.1],  # Correct NORMAL
        [0.8, 0.2],  # Correct NORMAL
        [0.4, 0.6],  # Wrong: predicted PNEUMONIA
        [0.7, 0.3],  # Correct NORMAL
        [0.2, 0.8],  # Correct PNEUMONIA
        [0.3, 0.7],  # Correct PNEUMONIA
        [0.6, 0.4],  # Wrong: predicted NORMAL
        [0.1, 0.9],  # Correct PNEUMONIA
    ])
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'labels': {0: 'NORMAL', 1: 'PNEUMONIA'}
    }


@pytest.fixture
def device():
    """返回测试用设备（优先CPU以保证测试速度）"""
    return torch.device('cpu')


def pytest_configure(config):
    """pytest配置钩子"""
    config.addinivalue_line("markers", "slow: 标记运行较慢的测试")
    config.addinivalue_line("markers", "integration: 标记集成测试")
    config.addinivalue_line("markers", "gpu: 需要GPU的测试")
    config.addinivalue_line("markers", "windows: Windows特定的测试")

