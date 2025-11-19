from typing import Dict, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# Try to import albumentations for better augmentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available, using basic torchvision transforms")


class RobustImageFolder(datasets.ImageFolder):
    """
    ImageFolder with error handling for corrupted images.
    Skips corrupted files instead of crashing.
    Implements retry logic with maximum attempts to avoid infinite loops.
    """
    def __init__(self, *args, max_retry_attempts=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retry_attempts = max_retry_attempts
    
    def __getitem__(self, index):
        """
        Load image with robust error handling.
        
        If an image fails to load, tries the next image up to max_retry_attempts times.
        This prevents infinite loops on datasets with many corrupted images.
        """
        for attempt in range(self.max_retry_attempts):
            try:
                return super().__getitem__(index)
            except (OSError, ValueError, IOError, RuntimeError) as e:
                if attempt == 0:
                    # 只在第一次失败时打印警告，避免日志泛滥
                    print(f"Warning: Failed to load image at index {index}: {e}")
                
                if attempt == self.max_retry_attempts - 1:
                    # 达到最大重试次数，抛出异常
                    raise RuntimeError(
                        f"Failed to load {self.max_retry_attempts} consecutive images. "
                        f"Dataset may contain too many corrupted files. "
                        f"Please check data integrity."
                    ) from e
                
                # 尝试下一个索引
                index = (index + 1) % len(self)
        
        # 不应该到达这里，但作为安全保障
        raise RuntimeError("Unexpected error in RobustImageFolder.__getitem__")


class AlbumentationsTransform:
    """
    Picklable wrapper for albumentations transforms.
    This class can be serialized by multiprocessing on Windows.
    """
    def __init__(self, img_size: int, is_train: bool = True):
        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __call__(self, img):
        """Convert PIL to numpy and apply albumentations transform."""
        if isinstance(img, Image.Image):
            img = np.array(img)
        return self.transform(image=img)['image']


def _albumentations_transforms(img_size: int, is_train: bool = True):
    """
    Create albumentations-based transforms (stronger augmentation).
    Returns a picklable transform object for Windows multiprocessing compatibility.
    """
    return AlbumentationsTransform(img_size, is_train)


def _default_transforms(img_size: int, augment_level: str = 'medium'):
    """
    Fallback torchvision transforms.
    
    Args:
        img_size: target image size
        augment_level: 'light', 'medium', or 'heavy'
    """
    if augment_level == 'light':
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif augment_level == 'heavy':
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:  # medium (default)
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def _make_samplers(train_dataset) -> Optional[WeightedRandomSampler]:
    # Compute class weights for WeightedRandomSampler
    targets = np.array(train_dataset.targets)
    classes, counts = np.unique(targets, return_counts=True)
    class_count = dict(zip(classes, counts))
    num_classes = len(classes)
    total = len(train_dataset)
    weights_per_class = {c: total / (num_classes * class_count[c]) for c in classes}
    sample_weights = np.array([weights_per_class[int(t)] for t in targets], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
    return sampler


def build_dataloaders(data_root: str, img_size: int, batch_size: int, num_workers: int = 4,
                      use_weighted_sampler: bool = True, use_albumentations: bool = True,
                      augment_level: str = 'medium') -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """
    Build train/val/test dataloaders.
    
    Args:
        data_root: root directory containing train/val/test folders
        img_size: target image size
        batch_size: batch size
        num_workers: number of dataloader workers
        use_weighted_sampler: whether to use weighted random sampler for training
        use_albumentations: whether to use albumentations (if available)
        augment_level: 'light', 'medium', or 'heavy' (only for torchvision)
    
    Returns:
        loaders: dict of dataloaders
        class_to_idx: class name to index mapping
    """
    root = Path(data_root)
    train_dir = root / 'train'
    val_dir = root / 'val'
    test_dir = root / 'test'
    
    # Check if directories exist
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Choose transform strategy
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        print("Using albumentations for data augmentation")
        train_tf = _albumentations_transforms(img_size, is_train=True)
        val_tf = _albumentations_transforms(img_size, is_train=False)
    else:
        print(f"Using torchvision transforms (augment_level={augment_level})")
        train_tf, val_tf = _default_transforms(img_size, augment_level)

    # Create datasets with error handling
    train_ds = RobustImageFolder(train_dir, transform=train_tf)
    val_ds = RobustImageFolder(val_dir, transform=val_tf)
    class_to_idx = train_ds.class_to_idx
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val: {len(val_ds)} images")
    print(f"  Classes: {class_to_idx}")

    sampler = _make_samplers(train_ds) if use_weighted_sampler else None

    # 性能优化的 DataLoader 配置
    # pin_memory只在GPU模式下有用，CPU模式下禁用以避免警告
    import torch
    use_pin_memory = torch.cuda.is_available()
    
    # 基础配置
    dataloader_kwargs = {'pin_memory': use_pin_memory}
    
    # 只在使用多进程时添加 persistent_workers 和 prefetch_factor
    if num_workers > 0:
        dataloader_kwargs.update({
            'persistent_workers': True,  # 保持 worker 进程活跃
            'prefetch_factor': 2,  # 预加载 2 个 batch
        })
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,  # Drop incomplete batches for stable training
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **dataloader_kwargs
    )

    # Check for test set
    if test_dir.exists():
        test_ds = RobustImageFolder(test_dir, transform=val_tf)
        test_loader = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            **dataloader_kwargs
        )
        print(f"  Test: {len(test_ds)} images")
    else:
        test_loader = None

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return loaders, class_to_idx
