"""
训练工具模块

提供训练过程中常用的工具函数和类。
"""
import random
from typing import Dict, Any, Optional, Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may reduce performance)

    Example:
        >>> set_seed(42)
        >>> # Training will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
    is_best: bool = False,
    best_path: Optional[Path] = None
):
    """
    Save model checkpoint to disk.

    Args:
        state: Checkpoint state dict
        path: Path to save checkpoint
        is_best: If True, also save as best checkpoint
        best_path: Path for best checkpoint (default: same dir as path)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

    if is_best:
        if best_path is None:
            best_path = path.parent / 'best_model.pt'
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load checkpoint and restore state.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        scaler: Optional AMP scaler to restore state
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Checkpoint dict with metadata
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    model.load_state_dict(ckpt['model'], strict=strict)

    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])

    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])

    return ckpt


class EarlyStopping:
    """
    Early stopping handler.

    Stops training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement to wait
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' or 'max' - whether to minimize or maximize metric
        verbose: If True, print messages

    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(100):
        ...     val_acc = train_epoch()
        ...     if early_stop(val_acc, epoch):
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0

        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False

        self.counter += 1

        if self.verbose and self.counter > 0:
            print(f"EarlyStopping: {self.counter}/{self.patience} "
                  f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")

        return self.counter >= self.patience

    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.

    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch in dataloader:
        ...     loss = compute_loss()
        ...     loss_meter.update(loss.item(), batch_size)
        >>> print(f"Average loss: {loss_meter.avg:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.

        Args:
            val: Value to add
            n: Number of items (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze all model parameters.

    Args:
        model: PyTorch model
        freeze: If True, freeze parameters; if False, unfreeze
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def freeze_bn(model: nn.Module):
    """
    Freeze BatchNorm layers (set to eval mode).

    Useful for fine-tuning with small batch sizes.

    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
