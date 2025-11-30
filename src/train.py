import argparse
import csv
import logging
import random
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

from src.models.factory import build_model
from src.utils.device import get_device
from src.data.datamodule import build_dataloaders
from src.utils.config_validator import ConfigValidator


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    Particularly useful for medical imaging where class imbalance is common.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        gamma: Focusing parameter (default: 1.5). Higher values increase focus on hard examples.
        weight: Per-class weights for handling imbalance (optional)
        reduction: Reduction method ('mean', 'sum', or 'none')
    
    Example:
        >>> loss_fn = FocalLoss(gamma=2.0, weight=torch.tensor([1.0, 2.0]))
        >>> loss = loss_fn(logits, targets)
    """
    def __init__(self, gamma: float = 1.5, weight=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Model outputs (N, C) before softmax
            targets: Ground truth labels (N,)
        
        Returns:
            Focal loss value
        """
        # 计算 log probabilities（数值稳定）
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # 获取目标类别的 log probability
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算 probability pt
        pt = torch.exp(log_pt)
        
        # Focal Loss 公式: -(1-pt)^gamma * log(pt)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = -focal_weight * log_pt
        
        # 应用类别权重（如果提供）
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]
        
        # 应用 reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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
    parser = argparse.ArgumentParser(description='Training script for pneumonia detection')
    parser.add_argument('--config', required=True, help='Path to training config YAML file')
    parser.add_argument('--data_root', default='data', help='Root directory of dataset')
    parser.add_argument('--save_dir', default='runs', help='Directory to save checkpoints')
    parser.add_argument('--save_best_by', default='macro_recall', 
                       choices=['accuracy', 'macro_recall', 'macro_f1', 'pneumonia_recall'],
                       help='Metric to select best model')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--validate_config', action='store_true', help='Validate config and exit')
    
    # 可选的命令行覆盖参数
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch_size from config')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--augment_level', type=str, default=None, 
                       choices=['light', 'medium', 'heavy', 'aggressive'], 
                       help='Override augmentation level from config')
    parser.add_argument('--model', type=str, default=None, 
                       help='Override model architecture from config')
    args = parser.parse_args()

    # 加载配置
    print(f"\n[CONFIG] Loading configuration from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 验证配置
    try:
        ConfigValidator.validate(cfg)
        if args.validate_config:
            print("[OK] Configuration is valid!")
            return
    except ValueError as e:
        print("[ERROR] Configuration validation failed:")
        print(str(e))
        if args.validate_config:
            return
        print("\n[WARNING] Continuing with potentially invalid configuration...")
        print("  Use --validate_config to only validate without training")
    
    # 设置随机种子
    seed = int(cfg.get('seed', 42))
    set_seed(seed)
    
    # ========================================
    # 性能优化设置
    # ========================================
    # 可选：允许非确定性行为以提升性能（会影响可复现性）
    allow_nondeterministic = cfg.get('allow_nondeterministic', False)
    if allow_nondeterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("\n[WARNING] Non-deterministic mode enabled - results may vary between runs")
    
    # 启用 TF32 精度 (RTX 30/40/50系列支持,加速矩阵运算)
    # 不影响可复现性，但会有轻微精度差异
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.get('allow_tf32', True)
        torch.backends.cudnn.allow_tf32 = cfg.get('allow_tf32', True)
    
    print("\n[Performance Settings]")
    print(f"  - cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    if torch.cuda.is_available():
        print(f"  - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 命令行参数优先于配置文件
    model_name = args.model if args.model is not None else cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    # 命令行参数优先于配置文件
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get('batch_size', 16))
    epochs = int(args.epochs if args.epochs is not None else cfg.get('epochs', 10))
    lr = float(args.lr if args.lr is not None else cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    loss_name = cfg.get('loss', 'weighted_ce')
    
    # 兼容两种配置方式：sampler 和 use_weighted_sampler
    if 'sampler' in cfg:
        use_sampler = (cfg['sampler'] == 'weighted_random')
    elif 'use_weighted_sampler' in cfg:
        use_sampler = cfg['use_weighted_sampler']
    else:
        use_sampler = True  # 默认使用 weighted sampler
    
    use_amp = cfg.get('amp', False)
    num_workers = int(cfg.get('num_workers', 4))
    use_albumentations = cfg.get('use_albumentations', True)
    # 命令行参数优先于配置文件
    augment_level = args.augment_level if args.augment_level is not None else cfg.get('augment_level', 'medium')
    # Handle 'aggressive' as alias for 'heavy'
    if augment_level == 'aggressive':
        augment_level = 'heavy'

    # 获取增强配置（如果存在）
    aug_config = cfg.get('augmentation', None)
    
    loaders, class_to_idx = build_dataloaders(
        args.data_root, img_size, batch_size, 
        num_workers=num_workers, 
        use_weighted_sampler=use_sampler,
        use_albumentations=use_albumentations,
        augment_level=augment_level,
        aug_config=aug_config
    )
    num_classes = len(class_to_idx)
    # idx_to_class = {v: k for k, v in class_to_idx.items()}  # Unused for now
    
    # 找到 PNEUMONIA 类的索引(用于计算 recall)
    pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))

    model, _ = build_model(model_name, num_classes)

    device = get_device()
    model = model.to(device)
    
    # 优化 3: 使用 channels_last 内存格式 (更高效的内存访问模式)
    model = model.to(memory_format=torch.channels_last)
    print("  - Memory format: channels_last")
    
    # 优化 6: 显存优化模式（可选）
    memory_efficient = cfg.get('memory_efficient', False)
    if memory_efficient and torch.cuda.is_available():
        print("\n[Memory Efficient Mode Enabled]")
        
        # 启用梯度检查点（降低显存使用，但增加计算时间）
        # 注意：不是所有模型都支持，需要模型显式实现
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
            print("  ✓ Gradient checkpointing enabled")
        
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        print("  ✓ CUDA cache cleared")
        
        # 启用 PyTorch 的内存高效功能（如果可用）
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("  ✓ Memory-efficient attention enabled")
        
        # 设置更积极的显存释放策略
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # 限制每个进程最多使用 90% 的 GPU 显存
            torch.cuda.set_per_process_memory_fraction(0.9, device=device)
            print("  ✓ GPU memory limit set to 90%")

    # class weights from train dataset targets (高效方式，无需遍历整个数据加载器)
    train_targets = np.array(loaders['train'].dataset.targets)
    counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
    weights = (counts.sum() / (num_classes * counts.clamp(min=1))).to(device)

    if loss_name.startswith('focal'):
        # 兼容两种配置格式: focal.gamma 或 focal_gamma
        gamma = cfg.get('focal', {}).get('gamma', cfg.get('focal_gamma', 1.5))
        loss_fn = FocalLoss(gamma=float(gamma), weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 可配置的学习率调度器
    scheduler_name = cfg.get('scheduler', 'cosine').lower()
    warmup_epochs = int(cfg.get('warmup_epochs', 0))
    
    # 创建主调度器
    if scheduler_name == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    elif scheduler_name == 'step':
        step_size = int(cfg.get('step_size', 10))
        gamma = float(cfg.get('gamma', 0.1))
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'exponential':
        gamma = float(cfg.get('gamma', 0.95))
        main_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == 'none':
        # 不使用调度器（学习率保持恒定）
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        print(f"[WARNING] Unknown scheduler '{scheduler_name}', using cosine")
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    
    # 添加学习率预热
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[warmup_epochs]
        )
        print(f"  - Learning rate scheduler: {scheduler_name} with {warmup_epochs} warmup epochs")
    else:
        scheduler = main_scheduler
        print(f"  - Learning rate scheduler: {scheduler_name}")
    
    # AMP只在GPU模式下启用，CPU模式自动禁用以避免警告
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp_actual = use_amp and torch.cuda.is_available()  # CPU强制禁用AMP
    scaler = GradScaler(device_type, enabled=use_amp_actual)

    best_score = -1.0
    start_epoch = 1
    # 支持两种配置写法:
    # 1. 新格式: early_stopping: { patience: 20 }
    # 2. 旧格式: patience: 20
    patience_cfg = cfg.get('early_stopping', {}) or {}
    patience = int(patience_cfg.get('patience', cfg.get('patience', 0)))
    no_improve = 0
    
    # ========================================
    # Checkpoint恢复功能
    # ========================================
    if args.resume:
        print(f"\n[RESUME] Loading checkpoint from: {args.resume}")
        try:
            resume_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(resume_ckpt['model'])
            optimizer.load_state_dict(resume_ckpt.get('optimizer', optimizer.state_dict()))
            scheduler.load_state_dict(resume_ckpt.get('scheduler', scheduler.state_dict()))
            start_epoch = resume_ckpt.get('epoch', 0) + 1
            best_score = resume_ckpt.get('best_score', -1.0)
            print(f"[RESUME] Resumed from epoch {start_epoch-1}, best_score={best_score:.4f}")
        except (RuntimeError, KeyError, FileNotFoundError) as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            print("[WARNING] Starting training from scratch...")
    
    # 优先使用配置文件中的 output_dir,否则使用命令行参数的 save_dir
    output_dir = cfg.get('output_dir', args.save_dir)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / 'best_model.pt'
    last_ckpt = save_dir / 'last_model.pt'
    
    # 优化 4: 验证频率配置 (减少验证次数以加速训练)
    val_interval = int(cfg.get('val_interval', 1))  # 每N个epoch验证一次,默认每次都验证
    print(f"  - Validation interval: every {val_interval} epoch(s)")
    
    # 优化 5: last checkpoint 保存间隔（避免频繁IO）
    save_last_interval = int(cfg.get('save_last_interval', 5))  # 默认每5个epoch保存一次
    print(f"  - Save last checkpoint: every {save_last_interval} epoch(s)")
    
    # 初始化 CSV 日志和训练日志
    csv_path = save_dir / 'metrics_history.csv'
    train_log_path = save_dir / 'train.log'
    
    # 保存配置文件副本到输出目录
    config_copy_path = save_dir / 'config.yaml'
    shutil.copy(args.config, config_copy_path)
    
    # 设置日志系统
    logger = logging.getLogger('pneumonia_training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除已有 handlers
    
    # 文件 handler
    file_handler = logging.FileHandler(train_log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    # 控制台 handler（处理 Windows 编码问题）
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                record.msg = record.msg.encode('ascii', errors='ignore').decode('ascii')
                super().emit(record)
    
    console_handler = SafeStreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    def log_print(msg):
        """同时输出到控制台和日志文件"""
        logger.info(msg)
    
    # 使用 ExitStack 管理文件句柄，确保异常时也能正确关闭
    with ExitStack() as stack:
        csv_file = stack.enter_context(open(csv_path, 'w', newline='', encoding='utf-8'))
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            'epoch', 'train_loss', 'val_acc', 'val_loss',
            'pneumonia_recall', 'pneumonia_precision', 'pneumonia_f1',
            'normal_recall', 'macro_recall', 'macro_f1', 'lr'
        ])
        csv_writer.writeheader()
        csv_file.flush()
        
        log_print(f"\n{'='*60}")
        log_print(f"Experiment: {save_dir.name}")
        log_print(f"Output directory: {save_dir}")
        log_print(f"{'='*60}")
        log_print(f"Training config: {model_name} @ {img_size}px, {epochs} epochs, lr={lr}, loss={loss_name}")
        log_print(f"Augmentation: {'albumentations' if use_albumentations else f'torchvision ({augment_level})'}")
        amp_status = f"{use_amp_actual}" + (" (CPU: disabled)" if use_amp and not use_amp_actual else "")
        log_print(f"Using device: {device}, AMP: {amp_status}, Seed: {seed}")
        log_print(f"Pneumonia class index: {pneumonia_idx}")
        if args.resume:
            log_print(f"Resume: Starting from epoch {start_epoch}")
        log_print(f"{'='*60}\n")

        # 记录训练开始时间
        training_start_time = time.time()
        
        # 梯度裁剪配置
        max_grad_norm = float(cfg.get('max_grad_norm', 1.0))
        use_grad_clip = cfg.get('gradient_clipping', True)

        # 训练循环
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            running_loss = 0.0
            train_samples = 0
            
            for images, targets in tqdm(loaders['train'], desc=f"Train {epoch}/{epochs}"):
                images = images.to(device, memory_format=torch.channels_last)
                targets = targets.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type, enabled=use_amp_actual):
                    logits = model(images)
                    loss = loss_fn(logits, targets)
                
                scaler.scale(loss).backward()
                
                # 混合精度训练的梯度裁剪
                if use_grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * images.size(0)
                train_samples += images.size(0)
            
            train_loss = running_loss / max(1, train_samples)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # 根据 val_interval 决定是否进行验证
            should_validate = (epoch % val_interval == 0) or (epoch == epochs)
            
            if not should_validate:
                # 跳过验证,只打印训练损失
                log_print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f} | LR: {current_lr:.6f} (validation skipped)")
                continue
            
            # Validation loop: compute detailed metrics including pneumonia recall
            model.eval()
            all_preds = []
            all_targets = []
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for images, targets in tqdm(loaders['val'], desc=f"Val {epoch}/{epochs}"):
                    images = images.to(device, memory_format=torch.channels_last)
                    targets = targets.to(device)
                    
                    with autocast(device_type, enabled=use_amp_actual):
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
            log_print(f"Epoch {epoch}/{epochs}:")
            log_print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            log_print(f"  Pneumonia - Recall: {pneumonia_recall:.4f}, Precision: {pneumonia_precision:.4f}, F1: {pneumonia_f1:.4f}")
            log_print(f"  Normal Recall: {normal_recall:.4f} | Macro Recall: {macro_recall:.4f} | Macro F1: {macro_f1:.4f}")
            log_print(f"  LR: {current_lr:.6f} | Best {args.save_best_by}: {best_score:.4f}")
            
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
            
            # 保存 last checkpoint（按间隔或最后一个epoch）
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
            
            # 按间隔保存或最后一个epoch
            if epoch % save_last_interval == 0 or epoch == epochs:
                save_checkpoint(ckpt_state, last_ckpt)
                log_print(f"  [SAVE] Last checkpoint saved: {last_ckpt}")
            
            # 保存 best checkpoint
            if score > best_score:
                best_score = score
                save_checkpoint(ckpt_state, best_ckpt)
                log_print(f"  [BEST] Saved best checkpoint: {best_ckpt}")
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    log_print(f"\nEarly stopping at epoch {epoch} (no improvement in {no_improve} epochs)")
                    break
        
        # 训练完成,打印最终总结
        training_end_time = time.time()
        
        log_print("\n" + "=" * 60)
        log_print("Training completed!")
        log_print("=" * 60)
        log_print(f"Best {args.save_best_by}: {best_score:.4f}")
        log_print(f"Total epochs trained: {epoch}")
        total_time = training_end_time - training_start_time
        log_print(f"Total training time: {total_time/3600:.2f} hours")
        
        log_print("\nOutput files:")
        log_print(f"  - Best model: {best_ckpt}")
        log_print(f"  - Last model: {last_ckpt}")
        log_print(f"  - Metrics CSV: {csv_path}")
        log_print(f"  - Training log: {train_log_path}")
        log_print(f"  - Config copy: {config_copy_path}")
        
        log_print("\nNext steps:")
        log_print(f"  1. Evaluate: python src/eval.py --ckpt {best_ckpt} --data_root {args.data_root} --split test")
        log_print(f"  2. Analyze: python scripts/error_analysis.py --ckpt {best_ckpt}")
        log_print("  3. Demo: streamlit run src/app/streamlit_app.py")
        log_print("=" * 60)
    
    # 文件已通过 ExitStack 自动关闭
    
    try:
        print("\n[OK] Training completed successfully!")
    except UnicodeEncodeError:
        print("\n[OK] Training completed successfully!")


if __name__ == '__main__':
    main()
