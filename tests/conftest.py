"""
Pytest配置和共享fixtures
"""
import pytest
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """返回项目根目录路径"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_config():
    """返回示例配置字典"""
    return {
        'model': 'resnet18',
        'img_size': 224,
        'batch_size': 16,
        'epochs': 10,
        'lr': 0.001,
        'seed': 42,
        'loss': 'focal',
        'focal_gamma': 1.5,
        'use_weighted_sampler': True,
        'augment_level': 'medium',
        'amp': True
    }


def pytest_configure(config):
    """pytest配置钩子"""
    config.addinivalue_line("markers", "slow: 标记运行较慢的测试")
    config.addinivalue_line("markers", "integration: 标记集成测试")

