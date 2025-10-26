from typing import Dict, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np


def _default_transforms(img_size: int):
    # Fallback torchvision transforms (no albumentations dependency)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
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
                      use_weighted_sampler: bool = True) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    root = Path(data_root)
    train_dir = root / 'train'
    val_dir = root / 'val'
    test_dir = root / 'test'

    train_tf, val_tf = _default_transforms(img_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    class_to_idx = train_ds.class_to_idx

    sampler = _make_samplers(train_ds) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders = {"train": train_loader, "val": val_loader}

    if test_dir.exists():
        test_ds = datasets.ImageFolder(test_dir, transform=val_tf)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        loaders["test"] = test_loader

    return loaders, class_to_idx
