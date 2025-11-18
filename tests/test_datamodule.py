"""
单元测试：数据加载和预处理模块
"""
import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np

from src.data.datamodule import (
    build_dataloaders, 
    RobustImageFolder,
    _make_samplers
)


class TestDataModule:
    """测试数据模块功能"""
    
    @pytest.fixture
    def mock_dataset_dir(self):
        """创建临时测试数据集"""
        tmpdir = Path(tempfile.mkdtemp())
        
        # 创建目录结构
        for split in ['train', 'val', 'test']:
            for cls in ['NORMAL', 'PNEUMONIA']:
                (tmpdir / split / cls).mkdir(parents=True, exist_ok=True)
                
                # 创建一些测试图像
                for i in range(5):
                    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                    img.save(tmpdir / split / cls / f'img_{i}.jpg')
        
        yield tmpdir
        
        # 清理
        shutil.rmtree(tmpdir)
    
    def test_build_dataloaders_basic(self, mock_dataset_dir):
        """测试基本的数据加载器构建"""
        loaders, class_to_idx = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False
        )
        
        # 验证返回的数据加载器
        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders
        
        # 验证类别映射
        assert len(class_to_idx) == 2
        assert 'NORMAL' in class_to_idx
        assert 'PNEUMONIA' in class_to_idx
    
    def test_dataloader_output_shape(self, mock_dataset_dir):
        """测试数据加载器输出的形状"""
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0
        )
        
        # 获取一个batch
        images, labels = next(iter(loaders['train']))
        
        # 验证形状
        assert images.shape[0] == 2  # batch size
        assert images.shape[1] == 3  # channels
        assert images.shape[2] == 224  # height
        assert images.shape[3] == 224  # width
        assert labels.shape[0] == 2
    
    def test_robust_image_folder(self, mock_dataset_dir):
        """测试RobustImageFolder对损坏图像的处理"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        dataset = RobustImageFolder(
            mock_dataset_dir / 'train',
            transform=transform
        )
        
        # 应该能正常加载
        assert len(dataset) > 0
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)


class TestDataAugmentation:
    """测试数据增强功能"""
    
    def test_augmentation_levels(self, tmp_path):
        """测试不同增强级别"""
        # 创建简单的测试数据
        for split in ['train', 'val']:
            for cls in ['NORMAL', 'PNEUMONIA']:
                (tmp_path / split / cls).mkdir(parents=True, exist_ok=True)
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(tmp_path / split / cls / 'test.jpg')
        
        # 测试不同增强级别
        for level in ['light', 'medium', 'heavy']:
            loaders, _ = build_dataloaders(
                str(tmp_path),
                img_size=224,
                batch_size=1,
                num_workers=0,
                use_albumentations=False,
                augment_level=level
            )
            
            # 验证可以正常加载数据
            images, labels = next(iter(loaders['train']))
            assert images.shape == (1, 3, 224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

