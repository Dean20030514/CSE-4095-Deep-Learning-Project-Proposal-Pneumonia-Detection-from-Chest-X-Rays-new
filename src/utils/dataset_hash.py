"""
数据集哈希验证模块

计算和验证数据集哈希，确保训练和评估使用相同的数据。
"""
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
import json


def compute_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """
    计算单个文件的哈希值。
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ('md5', 'sha256')
    
    Returns:
        文件哈希值的十六进制字符串
    """
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        # 分块读取以处理大文件
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_dataset_hash(
    data_dir: Path,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
    algorithm: str = 'md5',
    include_content: bool = False
) -> Dict[str, str]:
    """
    计算数据集目录的哈希值。
    
    Args:
        data_dir: 数据集目录路径
        extensions: 要包含的文件扩展名
        algorithm: 哈希算法
        include_content: 是否包含文件内容哈希（更慢但更准确）
    
    Returns:
        包含哈希信息的字典：
        - structure_hash: 基于文件名和目录结构的哈希
        - content_hash: 基于文件内容的哈希（如果 include_content=True）
        - file_count: 文件总数
        - splits: 每个 split 的文件数
    
    Example:
        >>> hash_info = compute_dataset_hash(Path('data'))
        >>> print(hash_info['structure_hash'][:16])
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # 收集所有图像文件
    all_files = []
    splits_count = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            split_files = []
            for ext in extensions:
                split_files.extend(split_dir.glob(f'**/*{ext}'))
                split_files.extend(split_dir.glob(f'**/*{ext.upper()}'))
            
            splits_count[split] = len(split_files)
            all_files.extend(split_files)
    
    if not all_files:
        raise ValueError(f"No image files found in {data_dir}")
    
    # 按相对路径排序以确保一致性
    all_files = sorted(all_files, key=lambda p: str(p.relative_to(data_dir)))
    
    # 计算结构哈希（基于文件名）
    structure_hasher = hashlib.md5()
    for file_path in all_files:
        relative_path = str(file_path.relative_to(data_dir))
        structure_hasher.update(relative_path.encode())
    
    result = {
        'structure_hash': structure_hasher.hexdigest(),
        'file_count': len(all_files),
        'splits': splits_count,
    }
    
    # 可选：计算内容哈希
    if include_content:
        content_hasher = hashlib.md5()
        for file_path in all_files:
            file_hash = compute_file_hash(file_path, algorithm)
            content_hasher.update(file_hash.encode())
        
        result['content_hash'] = content_hasher.hexdigest()
    
    return result


def save_dataset_hash(
    data_dir: Path,
    output_path: Path,
    include_content: bool = False
) -> Dict[str, str]:
    """
    计算并保存数据集哈希到 JSON 文件。
    
    Args:
        data_dir: 数据集目录
        output_path: 输出 JSON 文件路径
        include_content: 是否包含内容哈希
    
    Returns:
        哈希信息字典
    """
    hash_info = compute_dataset_hash(data_dir, include_content=include_content)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hash_info, f, indent=2)
    
    print(f"Dataset hash saved to: {output_path}")
    print(f"  Structure hash: {hash_info['structure_hash'][:16]}...")
    print(f"  Total files: {hash_info['file_count']}")
    
    return hash_info


def verify_dataset_hash(
    data_dir: Path,
    expected_hash: str,
    hash_type: str = 'structure'
) -> bool:
    """
    验证数据集哈希是否匹配。
    
    Args:
        data_dir: 数据集目录
        expected_hash: 期望的哈希值
        hash_type: 哈希类型 ('structure' 或 'content')
    
    Returns:
        哈希是否匹配
    """
    include_content = (hash_type == 'content')
    current_hash = compute_dataset_hash(data_dir, include_content=include_content)
    
    hash_key = f'{hash_type}_hash'
    if hash_key not in current_hash:
        raise ValueError(f"Hash type '{hash_type}' not available")
    
    actual = current_hash[hash_key]
    matches = actual == expected_hash
    
    if not matches:
        print(f"⚠️ Dataset hash mismatch!")
        print(f"  Expected: {expected_hash[:16]}...")
        print(f"  Actual:   {actual[:16]}...")
    
    return matches


def get_checkpoint_dataset_hash(checkpoint_path: Path) -> Optional[str]:
    """
    从 checkpoint 中提取数据集哈希。
    
    Args:
        checkpoint_path: checkpoint 文件路径
    
    Returns:
        数据集哈希值，如果不存在则返回 None
    """
    import torch
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return ckpt.get('dataset_hash')


def add_dataset_hash_to_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    output_path: Optional[Path] = None
) -> None:
    """
    将数据集哈希添加到现有 checkpoint。
    
    Args:
        checkpoint_path: 原始 checkpoint 路径
        data_dir: 数据集目录
        output_path: 输出路径（默认覆盖原文件）
    """
    import torch
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hash_info = compute_dataset_hash(data_dir)
    
    ckpt['dataset_hash'] = hash_info['structure_hash']
    ckpt['dataset_info'] = hash_info
    
    output = output_path or checkpoint_path
    torch.save(ckpt, output)
    
    print(f"Dataset hash added to checkpoint: {output}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
        
        print(f"Computing dataset hash for: {data_path}")
        hash_info = compute_dataset_hash(data_path)
        
        print(f"\nDataset Hash Information:")
        print(f"  Structure hash: {hash_info['structure_hash']}")
        print(f"  Total files: {hash_info['file_count']}")
        print(f"  Splits: {hash_info['splits']}")
    else:
        print("Usage: python src/utils/dataset_hash.py <data_directory>")
        print("Example: python src/utils/dataset_hash.py data/")

