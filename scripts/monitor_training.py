"""
训练实时监控脚本

监控训练进度、GPU使用情况和性能指标。

使用方法:
    python scripts/monitor_training.py --run_dir runs/model_efficientnet_b2
    python scripts/monitor_training.py --run_dir runs/model_efficientnet_b2 --interval 5
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_gpu_info():
    """获取 GPU 使用信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = props.total_memory / (1024**3)
            
            info.append({
                'device': i,
                'name': props.name,
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'memory_total_gb': memory_total,
                'utilization_pct': (memory_reserved / memory_total) * 100
            })
        return info
    except Exception:
        return None


def get_system_info():
    """获取系统资源使用信息"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent
        }
    except ImportError:
        return None


def parse_metrics_csv(csv_path: Path):
    """解析训练指标 CSV 文件"""
    if not csv_path.exists():
        return None
    
    import csv
    metrics = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append({k: float(v) if v.replace('.', '').replace('-', '').isdigit() else v 
                           for k, v in row.items()})
    return metrics


def parse_train_log(log_path: Path, num_lines: int = 20):
    """解析训练日志文件最后几行"""
    if not log_path.exists():
        return []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return lines[-num_lines:]


def clear_screen():
    """清除终端屏幕"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_dashboard(run_dir: Path, show_log: bool = True):
    """打印监控面板"""
    clear_screen()
    
    print("=" * 70)
    print(f"  TRAINING MONITOR - {run_dir.name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # GPU 信息
    gpu_info = get_gpu_info()
    if gpu_info:
        print("\n[GPU Status]")
        for gpu in gpu_info:
            bar_width = 30
            filled = int(gpu['utilization_pct'] / 100 * bar_width)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"  GPU {gpu['device']} ({gpu['name']})")
            print(f"    Memory: [{bar}] {gpu['memory_reserved_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB ({gpu['utilization_pct']:.1f}%)")
    
    # 系统信息
    sys_info = get_system_info()
    if sys_info:
        print("\n[System Status]")
        print(f"  CPU: {sys_info['cpu_percent']:.1f}%")
        print(f"  RAM: {sys_info['memory_used_gb']:.1f}/{sys_info['memory_total_gb']:.1f} GB ({sys_info['memory_percent']:.1f}%)")
    
    # 训练指标
    csv_path = run_dir / 'metrics_history.csv'
    metrics = parse_metrics_csv(csv_path)
    
    if metrics:
        print("\n[Training Progress]")
        latest = metrics[-1]
        total_epochs = len(metrics)
        
        print(f"  Epoch: {total_epochs}")
        
        if 'train_loss' in latest:
            print(f"  Train Loss: {float(latest.get('train_loss', 0)):.4f}")
        if 'val_loss' in latest:
            print(f"  Val Loss: {float(latest.get('val_loss', 0)):.4f}")
        if 'macro_recall' in latest:
            print(f"  Macro Recall: {float(latest.get('macro_recall', 0)):.4f}")
        if 'macro_f1' in latest:
            print(f"  Macro F1: {float(latest.get('macro_f1', 0)):.4f}")
        if 'lr' in latest:
            print(f"  Learning Rate: {float(latest.get('lr', 0)):.6f}")
        
        # 找到最佳指标
        if len(metrics) > 1:
            best_recall = max(float(m.get('macro_recall', 0)) for m in metrics)
            best_f1 = max(float(m.get('macro_f1', 0)) for m in metrics)
            print(f"\n  Best Macro Recall: {best_recall:.4f}")
            print(f"  Best Macro F1: {best_f1:.4f}")
    else:
        print("\n[Training Progress]")
        print("  Waiting for training to start...")
    
    # 训练日志
    if show_log:
        log_path = run_dir / 'train.log'
        log_lines = parse_train_log(log_path, num_lines=10)
        
        if log_lines:
            print("\n[Recent Log]")
            print("-" * 70)
            for line in log_lines:
                # 截断过长的行
                line = line.strip()
                if len(line) > 68:
                    line = line[:65] + '...'
                print(f"  {line}")
            print("-" * 70)
    
    # 文件状态
    print("\n[Files]")
    best_ckpt = run_dir / 'best_model.pt'
    last_ckpt = run_dir / 'last_model.pt'
    print(f"  Best checkpoint: {'✓' if best_ckpt.exists() else '○'}")
    print(f"  Last checkpoint: {'✓' if last_ckpt.exists() else '○'}")
    print(f"  Metrics CSV: {'✓' if csv_path.exists() else '○'}")
    
    print("\n" + "=" * 70)
    print("  Press Ctrl+C to exit")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument('--run_dir', type=str, required=True, help='Training run directory')
    parser.add_argument('--interval', type=int, default=10, help='Refresh interval in seconds')
    parser.add_argument('--no_log', action='store_true', help='Hide log output')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    print(f"Monitoring: {run_dir}")
    print(f"Refresh interval: {args.interval} seconds")
    print("Starting in 2 seconds...")
    time.sleep(2)
    
    try:
        while True:
            print_dashboard(run_dir, show_log=not args.no_log)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == '__main__':
    main()

