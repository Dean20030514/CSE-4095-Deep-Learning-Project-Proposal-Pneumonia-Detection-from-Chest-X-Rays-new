import argparse
import csv
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

from src.models.factory import build_model
from src.utils.device import get_device
from src.data.datamodule import build_dataloaders


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    Particularly useful for medical imaging where class imbalance is common.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        gamma: Focusing parameter (default: 1.5). Higher values increase focus on hard examples.
        weight: Per-class weights for handling imbalance (optional)
    
    Example:
        >>> loss_fn = FocalLoss(gamma=2.0, weight=torch.tensor([1.0, 2.0]))
        >>> loss = loss_fn(logits, targets)
    """
    def __init__(self, gamma: float = 1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        logp = -self.ce(logits, targets)
        p = torch.exp(logp)
        loss = -((1 - p) ** self.gamma) * logp
        return loss.mean()


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility.
    
    Ensures deterministic behavior across random, numpy, and PyTorch operations.
    This is critical for reproducible experiments and debugging.
    
    Args:
        seed: Integer seed value (default: 42)
    
    Note:
        Sets cudnn.deterministic=True which may reduce performance slightly
        but ensures reproducible results on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: Dict, path: Path):
    """Save model checkpoint to disk.
    
    Creates parent directories if they don't exist and saves the checkpoint
    dictionary containing model weights, optimizer state, config, etc.
    
    Args:
        state: Dictionary containing checkpoint data (model, optimizer, config, classes, etc.)
        path: Target file path for saving the checkpoint
    
    Example:
        >>> state = {'model': model.state_dict(), 'epoch': 10, 'config': cfg}
        >>> save_checkpoint(state, Path('runs/experiment/best.pt'))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='runs')
    parser.add_argument('--save_best_by', default='macro_recall')
    # 可选的命令行覆盖参数
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch_size from config')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--augment_level', type=str, default=None, choices=['light', 'medium', 'heavy', 'aggressive'], help='Override augmentation level from config')
    parser.add_argument('--model', type=str, default=None, help='Override model architecture from config (resnet18, resnet50, efficientnet_b0, efficientnet_b2, densenet121)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 设置随机种子
    seed = int(cfg.get('seed', 42))
    set_seed(seed)

    # 命令行参数优先于配置文件
    model_name = args.model if args.model is not None else cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    # 命令行参数优先于配置文件
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get('batch_size', 16))
    epochs = int(args.epochs if args.epochs is not None else cfg.get('epochs', 10))
    lr = float(args.lr if args.lr is not None else cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    loss_name = cfg.get('loss', 'weighted_ce')
    use_sampler = cfg.get('sampler', 'weighted_random') == 'weighted_random'
    use_amp = cfg.get('amp', False)
    num_workers = int(cfg.get('num_workers', 4))
    use_albumentations = cfg.get('use_albumentations', True)
    # 命令行参数优先于配置文件
    augment_level = args.augment_level if args.augment_level is not None else cfg.get('augment_level', 'medium')
    # Handle 'aggressive' as alias for 'heavy'
    if augment_level == 'aggressive':
        augment_level = 'heavy'

    loaders, class_to_idx = build_dataloaders(
        args.data_root, img_size, batch_size, 
        num_workers=num_workers, 
        use_weighted_sampler=use_sampler,
        use_albumentations=use_albumentations,
        augment_level=augment_level
    )
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 找到 PNEUMONIA 类的索引(用于计算 recall)
    pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))

    model, _ = build_model(model_name, num_classes)

    device = get_device()
    model = model.to(device)

    # class weights from train loader counts
    counts = torch.zeros(num_classes)
    for _, targets in loaders['train']:
        for t in targets:
            counts[t] += 1
    weights = (counts.sum() / (num_classes * counts.clamp(min=1))).to(device)

    if loss_name.startswith('focal'):
        loss_fn = FocalLoss(gamma=float(cfg.get('focal', {}).get('gamma', 1.5)), weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda', enabled=use_amp)

    best_score = -1.0
    patience_cfg = cfg.get('early_stopping', {}) or {}
    patience = int(patience_cfg.get('patience', 0))
    no_improve = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / 'best.pt'
    last_ckpt = save_dir / 'last.pt'
    
    # 初始化 CSV 日志
    log_cfg = cfg.get('log', {})
    csv_path = save_dir / log_cfg.get('csv', 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'epoch', 'train_loss', 'val_acc', 'val_loss',
        'pneumonia_recall', 'pneumonia_precision', 'pneumonia_f1',
        'normal_recall', 'macro_recall', 'macro_f1', 'lr'
    ])
    csv_writer.writeheader()
    csv_file.flush()
    
    print(f"Training config: {model_name} @ {img_size}px, {epochs} epochs, lr={lr}, loss={loss_name}")
    print(f"Augmentation: {'albumentations' if use_albumentations else f'torchvision ({augment_level})'}")
    print(f"Using device: {device}, AMP: {use_amp}, Seed: {seed}")
    print(f"Pneumonia class index: {pneumonia_idx}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_samples = 0
        
        for images, targets in tqdm(loaders['train'], desc=f"Train {epoch}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
        
        train_loss = running_loss / max(1, train_samples)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Validation loop: compute detailed metrics including pneumonia recall
        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, targets in tqdm(loaders['val'], desc=f"Val {epoch}/{epochs}"):
                images = images.to(device)
                targets = targets.to(device)
                
                with autocast('cuda', enabled=use_amp):
                    logits = model(images)
                    loss = loss_fn(logits, targets)
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
        
        val_loss = val_loss / max(1, val_samples)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 计算整体准确率
        val_acc = (all_preds == all_targets).mean()
        
        # 计算 per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        # 提取 PNEUMONIA 类指标
        pneumonia_recall = recalls[pneumonia_idx] if pneumonia_idx < len(recalls) else 0.0
        pneumonia_precision = precisions[pneumonia_idx] if pneumonia_idx < len(precisions) else 0.0
        pneumonia_f1 = f1s[pneumonia_idx] if pneumonia_idx < len(f1s) else 0.0
        
        # 计算 NORMAL 类指标(假设是另一个类)
        normal_idx = 1 - pneumonia_idx if num_classes == 2 else 0
        normal_recall = recalls[normal_idx] if normal_idx < len(recalls) else 0.0
        
        # 宏平均
        macro_recall = recalls.mean()
        macro_f1 = f1s.mean()
        
        # 根据配置选择 best metric
        if args.save_best_by == 'pneumonia_recall':
            score = pneumonia_recall
        elif args.save_best_by == 'macro_f1':
            score = macro_f1
        elif args.save_best_by == 'macro_recall':
            score = macro_recall
        else:
            score = val_acc
        
        # 打印指标
        print(f"Epoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Pneumonia - Recall: {pneumonia_recall:.4f}, Precision: {pneumonia_precision:.4f}, F1: {pneumonia_f1:.4f}")
        print(f"  Normal Recall: {normal_recall:.4f} | Macro Recall: {macro_recall:.4f} | Macro F1: {macro_f1:.4f}")
        print(f"  LR: {current_lr:.6f} | Best {args.save_best_by}: {best_score:.4f}")
        
        # 写入 CSV
        csv_writer.writerow({
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'val_acc': f"{val_acc:.6f}",
            'val_loss': f"{val_loss:.6f}",
            'pneumonia_recall': f"{pneumonia_recall:.6f}",
            'pneumonia_precision': f"{pneumonia_precision:.6f}",
            'pneumonia_f1': f"{pneumonia_f1:.6f}",
            'normal_recall': f"{normal_recall:.6f}",
            'macro_recall': f"{macro_recall:.6f}",
            'macro_f1': f"{macro_f1:.6f}",
            'lr': f"{current_lr:.8f}"
        })
        csv_file.flush()
        
        # 保存 last checkpoint(包含完整配置)
        ckpt_state = {
            'model': model.state_dict(),
            'classes': class_to_idx,
            'config': cfg,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_score': best_score,
            'metrics': {
                'val_acc': float(val_acc),
                'pneumonia_recall': float(pneumonia_recall),
                'macro_f1': float(macro_f1)
            }
        }
        save_checkpoint(ckpt_state, last_ckpt)
        
        # 保存 best checkpoint
        if score > best_score:
            best_score = score
            save_checkpoint(ckpt_state, best_ckpt)
            print(f"  [BEST] Saved best checkpoint: {best_ckpt}")
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement in {no_improve} epochs)")
                break
    
    csv_file.close()
    print(f"\n{'='*60}")
    print(f"Training completed! Best {args.save_best_by} = {best_score:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Metrics log: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
