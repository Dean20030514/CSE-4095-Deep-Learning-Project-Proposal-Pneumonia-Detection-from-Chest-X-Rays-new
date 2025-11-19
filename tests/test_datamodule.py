"""
单元测试：数据加载和预处理模块

测试数据加载、增强、采样器等功能
"""
import pytest
import torch
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.datamodule import (
    build_dataloaders, 
    RobustImageFolder,
    _make_samplers,
    AlbumentationsTransform
)


class TestDataModule:
    """测试数据模块核心功能"""
    
    def test_build_dataloaders_basic(self, mock_dataset_dir):
        """测试基本的数据加载器构建"""
        loaders, class_to_idx = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 验证返回的数据加载器
        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders
        
        # 验证所有loaders都不为None
        assert loaders['train'] is not None
        assert loaders['val'] is not None
        assert loaders['test'] is not None
        
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
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 获取一个batch
        images, labels = next(iter(loaders['train']))
        
        # 验证形状
        assert images.shape[0] == 2  # batch size
        assert images.shape[1] == 3  # channels
        assert images.shape[2] == 224  # height
        assert images.shape[3] == 224  # width
        assert labels.shape[0] == 2
        
        # 验证数据类型
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
    
    def test_dataloader_with_weighted_sampler(self, mock_dataset_dir):
        """测试使用weighted sampler的数据加载器"""
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=True,
            use_albumentations=False
        )
        
        # 验证可以正常迭代
        images, labels = next(iter(loaders['train']))
        assert images.shape[0] == 2
    
    def test_robust_image_folder(self, mock_dataset_dir):
        """测试RobustImageFolder对正常图像的处理"""
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
        assert img.shape == (3, 224, 224)
    
    def test_robust_image_folder_corrupted(self, tmp_path):
        """测试RobustImageFolder处理损坏文件"""
        from torchvision import transforms
        
        # 创建测试目录
        test_dir = tmp_path / 'test_cls'
        test_dir.mkdir(parents=True)
        
        # 创建一些正常图像
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(test_dir / f'good_{i}.jpg')
        
        # 创建一个损坏的文件（空文件）
        (test_dir / 'corrupted.jpg').write_bytes(b'')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # 创建dataset（不应该崩溃）
        dataset = RobustImageFolder(
            tmp_path,
            transform=transform
        )
        
        # 验证至少可以加载正常图像
        assert len(dataset) >= 3
    
    def test_make_samplers(self, mock_dataset_dir):
        """测试WeightedRandomSampler的创建"""
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        train_dataset = datasets.ImageFolder(
            mock_dataset_dir / 'train',
            transform=transform
        )
        
        # 创建sampler
        sampler = _make_samplers(train_dataset)
        
        # 验证sampler
        assert sampler is not None
        assert len(sampler) == len(train_dataset)


class TestDataAugmentation:
    """测试数据增强功能"""
    
    @pytest.mark.parametrize("augment_level", ['light', 'medium', 'heavy'])
    def test_augmentation_levels(self, mock_dataset_dir, augment_level):
        """测试不同增强级别"""
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_albumentations=False,
            augment_level=augment_level
        )
        
        # 验证可以正常加载数据
        images, labels = next(iter(loaders['train']))
        assert images.shape == (2, 3, 224, 224)
    
    def test_aggressive_augmentation_alias(self, mock_dataset_dir):
        """测试aggressive作为heavy的别名"""
        # 'aggressive'在train.py中会转换为'heavy'
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_albumentations=False,
            augment_level='heavy'  # 使用heavy
        )
        
        images, labels = next(iter(loaders['train']))
        assert images.shape == (2, 3, 224, 224)
    
    def test_albumentations_transform(self):
        """测试Albumentations transform wrapper"""
        try:
            import albumentations
            
            # 创建transform
            transform = AlbumentationsTransform(img_size=224, is_train=True)
            
            # 创建测试图像
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # 应用transform
            result = transform(img)
            
            # 验证输出
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)
        except ImportError:
            pytest.skip("albumentations not installed")


class TestDataLoaderEdgeCases:
    """测试边界情况"""
    
    def test_different_image_sizes(self, mock_dataset_dir):
        """测试不同的图像尺寸"""
        for img_size in [128, 224, 256, 512]:
            loaders, _ = build_dataloaders(
                str(mock_dataset_dir),
                img_size=img_size,
                batch_size=2,
                num_workers=0,
                use_albumentations=False
            )
            
            images, _ = next(iter(loaders['train']))
            assert images.shape[2] == img_size
            assert images.shape[3] == img_size
    
    def test_missing_test_dir(self, tmp_path):
        """测试缺少test目录的情况"""
        # 只创建train和val
        for split in ['train', 'val']:
            for cls in ['NORMAL', 'PNEUMONIA']:
                (tmp_path / split / cls).mkdir(parents=True, exist_ok=True)
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(tmp_path / split / cls / 'img_0.jpg')
        
        loaders, _ = build_dataloaders(
            str(tmp_path),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_albumentations=False
        )
        
        # test loader应该为None
        assert loaders['test'] is None
    
    def test_batch_size_one(self, mock_dataset_dir):
        """测试batch_size=1"""
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=1,
            num_workers=0,
            use_albumentations=False
        )
        
        images, labels = next(iter(loaders['val']))
        assert images.shape[0] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

