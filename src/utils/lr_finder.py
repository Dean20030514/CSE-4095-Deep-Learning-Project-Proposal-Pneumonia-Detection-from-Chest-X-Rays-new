"""
学习率查找器

自动寻找最佳学习率范围。
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class LRFinder:
    """
    学习率范围测试器。
    
    通过在一个 mini-batch 上逐步增加学习率，记录损失变化，
    帮助选择合适的学习率范围。
    
    Reference: Smith, L.N. "Cyclical Learning Rates for Training Neural Networks" (2017)
    
    Example:
        >>> finder = LRFinder(model, optimizer, loss_fn)
        >>> finder.find(train_loader, start_lr=1e-7, end_lr=10, num_iters=100)
        >>> finder.plot()
        >>> suggested_lr = finder.suggest_lr()
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: Optimizer, 
        loss_fn: nn.Module,
        device: str = 'cuda'
    ):
        """
        初始化学习率查找器。
        
        Args:
            model: PyTorch 模型
            optimizer: 优化器
            loss_fn: 损失函数
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # 保存初始状态
        self._initial_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        self._initial_optimizer_state = optimizer.state_dict()
        
        # 记录结果
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.smoothed_losses: List[float] = []
    
    def find(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iters: int = 100,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 5.0
    ) -> Tuple[List[float], List[float]]:
        """
        执行学习率范围测试。
        
        Args:
            train_loader: 训练数据加载器
            start_lr: 起始学习率
            end_lr: 结束学习率
            num_iters: 迭代次数
            smooth_factor: 损失平滑因子
            diverge_threshold: 发散阈值（当损失超过初始损失的此倍数时停止）
        
        Returns:
            (学习率列表, 损失列表)
        """
        # 计算学习率乘数
        lr_mult = (end_lr / start_lr) ** (1 / num_iters)
        
        # 设置初始学习率
        lr = start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 重置记录
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        
        best_loss = float('inf')
        avg_loss = 0.0
        
        self.model.train()
        data_iter = iter(train_loader)
        
        pbar = tqdm(range(num_iters), desc="Finding LR")
        for iteration in pbar:
            # 获取数据
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # 检查是否发散
            if iteration == 0:
                best_loss = loss.item()
            
            # 平滑损失
            avg_loss = smooth_factor * loss.item() + (1 - smooth_factor) * avg_loss
            smoothed_loss = avg_loss / (1 - smooth_factor ** (iteration + 1))
            
            # 记录
            self.lrs.append(lr)
            self.losses.append(loss.item())
            self.smoothed_losses.append(smoothed_loss)
            
            # 更新进度条
            pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})
            
            # 检查发散
            if smoothed_loss > diverge_threshold * best_loss:
                print(f"\nStopping early: loss diverged at lr={lr:.2e}")
                break
            
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新学习率
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        # 恢复初始状态
        self._restore_state()
        
        return self.lrs, self.losses
    
    def _restore_state(self):
        """恢复模型和优化器的初始状态"""
        self.model.load_state_dict(self._initial_model_state)
        self.optimizer.load_state_dict(self._initial_optimizer_state)
    
    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        建议最佳学习率。
        
        使用损失下降最快的点作为建议学习率。
        
        Args:
            skip_start: 跳过开始的迭代次数
            skip_end: 跳过结束的迭代次数
        
        Returns:
            建议的学习率
        """
        if len(self.smoothed_losses) < skip_start + skip_end + 1:
            raise ValueError("Not enough data points. Run find() first.")
        
        # 计算损失梯度
        losses = np.array(self.smoothed_losses[skip_start:-skip_end])
        lrs = np.array(self.lrs[skip_start:-skip_end])
        
        # 找到损失下降最快的点
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        
        suggested_lr = lrs[min_grad_idx]
        
        # 通常取该点学习率的 1/10 作为安全起点
        return suggested_lr / 10
    
    def plot(
        self, 
        skip_start: int = 10, 
        skip_end: int = 5,
        log_scale: bool = True,
        show_suggestion: bool = True,
        save_path: Optional[str] = None
    ):
        """
        绘制学习率-损失曲线。
        
        Args:
            skip_start: 跳过开始的点
            skip_end: 跳过结束的点
            log_scale: 是否使用对数刻度
            show_suggestion: 是否显示建议学习率
            save_path: 保存路径（可选）
        """
        if not self.lrs:
            raise ValueError("No data to plot. Run find() first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lrs = self.lrs[skip_start:-skip_end] if skip_end else self.lrs[skip_start:]
        losses = self.smoothed_losses[skip_start:-skip_end] if skip_end else self.smoothed_losses[skip_start:]
        
        ax.plot(lrs, losses, linewidth=2)
        
        if show_suggestion:
            try:
                suggested = self.suggest_lr(skip_start, skip_end)
                ax.axvline(x=suggested, color='r', linestyle='--', 
                          label=f'Suggested LR: {suggested:.2e}')
                ax.legend()
            except ValueError:
                pass
        
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Rate Finder', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        plt.close()


def find_lr(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer_cls=torch.optim.Adam,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_iters: int = 100,
    device: str = 'cuda'
) -> float:
    """
    便捷函数：快速找到建议的学习率。
    
    Args:
        model: PyTorch 模型
        train_loader: 训练数据加载器
        loss_fn: 损失函数
        optimizer_cls: 优化器类
        start_lr: 起始学习率
        end_lr: 结束学习率
        num_iters: 迭代次数
        device: 设备
    
    Returns:
        建议的学习率
    
    Example:
        >>> suggested_lr = find_lr(model, train_loader, nn.CrossEntropyLoss())
        >>> print(f"Suggested LR: {suggested_lr}")
    """
    model = model.to(device)
    optimizer = optimizer_cls(model.parameters(), lr=start_lr)
    
    finder = LRFinder(model, optimizer, loss_fn, device)
    finder.find(train_loader, start_lr, end_lr, num_iters)
    
    return finder.suggest_lr()


if __name__ == '__main__':
    print("Learning Rate Finder module")
    print("Usage: from src.utils.lr_finder import LRFinder, find_lr")

