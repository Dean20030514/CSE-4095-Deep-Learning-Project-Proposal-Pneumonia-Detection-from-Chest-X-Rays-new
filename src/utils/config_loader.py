"""
配置文件加载器

支持配置继承，减少重复配置。使用 `_base_` 字段指定基础配置文件。

Example:
    # src/configs/_base.yaml
    model: resnet18
    epochs: 100
    lr: 0.001

    # src/configs/model_efficientnet.yaml
    _base_: _base.yaml
    model: efficientnet_b2
    lr: 0.0005
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典，override 中的值会覆盖 base 中的值。

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key == '_base_':
            continue  # 跳过 _base_ 字段

        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = deepcopy(value)

    return result


def load_config(config_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件，支持继承。

    使用 `_base_` 字段指定基础配置文件，支持多级继承。
    子配置中的值会覆盖基础配置中的值。

    Args:
        config_path: 配置文件路径
        base_dir: 基础配置文件搜索目录（默认为 config_path 所在目录）

    Returns:
        合并后的配置字典

    Example:
        >>> cfg = load_config('src/configs/model_efficientnet_b2.yaml')
        >>> print(cfg['model'])  # 'efficientnet_b2'

    Config file format:
        ```yaml
        # 子配置文件
        _base_: _base.yaml  # 相对于当前配置文件目录
        model: efficientnet_b2
        lr: 0.0005
        ```
    """
    config_path = Path(config_path)

    if base_dir is None:
        base_dir = config_path.parent
    else:
        base_dir = Path(base_dir)

    # 加载当前配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # 处理继承
    if '_base_' in config:
        base_path = config['_base_']

        # 支持相对路径和绝对路径
        if not Path(base_path).is_absolute():
            base_path = base_dir / base_path

        if not Path(base_path).exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")

        # 递归加载基础配置
        base_config = load_config(str(base_path), base_dir=base_dir)

        # 合并配置
        config = deep_merge(base_config, config)

    return config


def validate_config_inheritance(config_dir: str) -> Dict[str, Any]:
    """
    验证目录中所有配置文件的继承关系。

    Args:
        config_dir: 配置文件目录

    Returns:
        验证结果字典
    """
    config_dir = Path(config_dir)
    results = {
        'valid': [],
        'invalid': [],
        'errors': []
    }

    for config_file in config_dir.glob('*.yaml'):
        if config_file.name.startswith('_'):
            continue  # 跳过基础配置文件

        try:
            cfg = load_config(str(config_file))
            results['valid'].append(str(config_file.name))
        except Exception as e:
            results['invalid'].append(str(config_file.name))
            results['errors'].append(f"{config_file.name}: {str(e)}")

    return results


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Config loader with inheritance support')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--validate-dir', help='Validate all configs in directory')
    parser.add_argument('--output', '-o', help='Output merged config to file')
    args = parser.parse_args()

    if args.validate_dir:
        results = validate_config_inheritance(args.validate_dir)
        print(f"Valid: {len(results['valid'])}")
        print(f"Invalid: {len(results['invalid'])}")
        for err in results['errors']:
            print(f"  - {err}")
    else:
        cfg = load_config(args.config)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"Merged config saved to: {args.output}")
        else:
            print(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
