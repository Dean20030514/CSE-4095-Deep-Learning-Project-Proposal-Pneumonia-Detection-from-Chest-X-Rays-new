"""
自动超参数优化脚本

功能：
1. 根据训练结果自动调整超参数
2. 迭代训练直到找到最优配置
3. 保存优化历史和最佳模型

使用方法：
    python scripts/auto_optimize_hyperparams.py --config src/configs/baseline.yaml --iterations 10 --target pneumonia_recall

策略：
- 如果性能提升 → 沿当前方向继续调整
- 如果性能下降 → 反向调整或尝试其他参数
- 自动记录所有尝试，避免重复

作者：CSE-4095 Deep Learning Team
"""

import argparse
import json
import yaml
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import shutil


class HyperparameterOptimizer:
    """自动超参数优化器"""
    
    def __init__(self, base_config_path: str, target_metric: str = 'macro_recall',
                 output_dir: str = 'runs/auto_optimization'):
        """
        Args:
            base_config_path: 基础配置文件路径
            target_metric: 优化目标指标 (macro_recall, pneumonia_recall, val_acc等)
            output_dir: 输出目录
        """
        self.base_config_path = Path(base_config_path)
        self.target_metric = target_metric
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载基础配置
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        
        # 优化历史
        self.history = []
        self.best_config = None
        self.best_score = -1.0
        self.iteration = 0
        
        # 参数搜索空间（定义每个参数的可选值和调整策略）
        self.param_space = {
            'lr': {
                'values': [0.0001, 0.0003, 0.0005, 0.001, 0.003],
                'type': 'continuous',
                'scale': 'log'
            },
            'batch_size': {
                'values': [8, 16, 32],
                'type': 'discrete'
            },
            'augment_level': {
                'values': ['light', 'medium', 'aggressive'],
                'type': 'categorical'
            },
            'model': {
                'values': ['resnet18', 'densenet121', 'efficientnet_b0'],
                'type': 'categorical'
            },
            'img_size': {
                'values': [224, 384],
                'type': 'discrete'
            }
        }
        
        # 当前探索的参数
        self.current_param_to_optimize = 'lr'
        
        print(f"[AutoOptimizer] 初始化完成")
        print(f"  - 基础配置: {base_config_path}")
        print(f"  - 目标指标: {target_metric}")
        print(f"  - 输出目录: {output_dir}")
    
    def get_next_config(self) -> Dict:
        """
        根据历史结果生成下一个配置
        
        策略：
        1. 前3次：快速探索不同学习率
        2. 找到较好的lr后：探索augmentation
        3. 再探索model架构
        4. 最后微调batch_size和img_size
        """
        new_config = self.base_config.copy()
        
        if self.iteration == 0:
            # 第一次：使用基础配置
            return new_config
        
        # 根据迭代次数选择优化策略
        if self.iteration <= 3:
            # 前3次：探索学习率
            lr_values = self.param_space['lr']['values']
            new_config['lr'] = lr_values[min(self.iteration - 1, len(lr_values) - 1)]
            print(f"  [探索学习率] lr={new_config['lr']}")
        
        elif self.iteration <= 6:
            # 第4-6次：使用最佳lr，探索augmentation
            if self.best_config:
                new_config['lr'] = self.best_config.get('lr', self.base_config['lr'])
            
            aug_values = self.param_space['augment_level']['values']
            aug_idx = (self.iteration - 4) % len(aug_values)
            new_config['augment_level'] = aug_values[aug_idx]
            print(f"  [探索增强] augment_level={new_config['augment_level']}")
        
        elif self.iteration <= 9:
            # 第7-9次：使用最佳lr和aug，探索模型架构
            if self.best_config:
                new_config['lr'] = self.best_config.get('lr', self.base_config['lr'])
                new_config['augment_level'] = self.best_config.get('augment_level', 'medium')
            
            model_values = self.param_space['model']['values']
            model_idx = (self.iteration - 7) % len(model_values)
            new_config['model'] = model_values[model_idx]
            print(f"  [探索架构] model={new_config['model']}")
        
        else:
            # 之后：基于最佳配置进行微调
            if self.best_config:
                new_config = self.best_config.copy()
                # 随机微调某个参数
                param_to_tune = np.random.choice(['batch_size', 'img_size'])
                values = self.param_space[param_to_tune]['values']
                current_val = new_config.get(param_to_tune, values[0])
                
                # 尝试不同的值
                if current_val in values:
                    idx = values.index(current_val)
                    new_idx = (idx + 1) % len(values)
                    new_config[param_to_tune] = values[new_idx]
                
                print(f"  [微调] {param_to_tune}={new_config[param_to_tune]}")
        
        return new_config
    
    def run_training(self, config: Dict) -> Dict:
        """
        运行一次训练
        
        Returns:
            包含训练结果的字典
        """
        # 保存配置到临时文件
        temp_config_path = self.output_dir / f'config_iter{self.iteration}.yaml'
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # 设置输出目录
        run_dir = self.output_dir / f'iteration_{self.iteration}'
        config['output_dir'] = str(run_dir)
        
        # 更新配置文件
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        print(f"\n{'='*60}")
        print(f"迭代 {self.iteration}: 开始训练")
        print(f"{'='*60}")
        print(f"配置: {temp_config_path}")
        print(f"输出: {run_dir}")
        
        # 构建训练命令
        cmd = [
            'python', 'src/train.py',
            '--config', str(temp_config_path)
        ]
        
        # 运行训练
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode != 0:
                print(f"[ERROR] 训练失败: {result.stderr}")
                return None
            
            print(f"[SUCCESS] 训练完成")
            
        except Exception as e:
            print(f"[ERROR] 运行失败: {e}")
            return None
        
        # 读取训练结果
        return self.extract_results(run_dir)
    
    def extract_results(self, run_dir: Path) -> Dict:
        """
        从训练输出中提取结果
        
        Args:
            run_dir: 训练输出目录
        
        Returns:
            包含指标的字典
        """
        metrics_file = run_dir / 'metrics_history.csv'
        
        if not metrics_file.exists():
            print(f"[WARNING] 找不到metrics文件: {metrics_file}")
            return None
        
        try:
            df = pd.read_csv(metrics_file)
            
            # 获取最佳epoch的指标
            if self.target_metric in df.columns:
                best_idx = df[self.target_metric].idxmax()
                best_row = df.loc[best_idx]
                
                results = {
                    'epoch': int(best_row['epoch']),
                    'val_acc': float(best_row.get('val_acc', 0)),
                    'macro_recall': float(best_row.get('macro_recall', 0)),
                    'pneumonia_recall': float(best_row.get('pneumonia_recall', 0)),
                    'val_loss': float(best_row.get('val_loss', 0)),
                    self.target_metric: float(best_row[self.target_metric])
                }
                
                return results
            else:
                print(f"[WARNING] 找不到目标指标: {self.target_metric}")
                return None
                
        except Exception as e:
            print(f"[ERROR] 读取结果失败: {e}")
            return None
    
    def update_history(self, config: Dict, results: Dict):
        """更新优化历史"""
        if results is None:
            return
        
        score = results.get(self.target_metric, 0)
        
        # 记录历史
        entry = {
            'iteration': self.iteration,
            'config': config.copy(),
            'results': results.copy(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(entry)
        
        # 更新最佳配置
        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()
            
            print(f"\n🎉 新的最佳配置!")
            print(f"  {self.target_metric}: {score:.4f} (↑ {score - self.best_score:.4f})")
            
            # 保存最佳配置
            best_config_path = self.output_dir / 'best_config.yaml'
            with open(best_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.best_config, f)
            
            # 复制最佳模型
            src_model = Path(config['output_dir']) / 'best_model.pt'
            dst_model = self.output_dir / 'best_model.pt'
            if src_model.exists():
                shutil.copy(src_model, dst_model)
                print(f"  模型已保存: {dst_model}")
    
    def save_history(self):
        """保存优化历史"""
        history_file = self.output_dir / 'optimization_history.json'
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        print(f"\n[保存] 优化历史: {history_file}")
        
        # 生成CSV摘要
        summary_data = []
        for entry in self.history:
            row = {
                'iteration': entry['iteration'],
                'score': entry['score'],
            }
            # 添加主要配置参数
            for param in ['lr', 'model', 'augment_level', 'batch_size', 'img_size']:
                row[param] = entry['config'].get(param, 'N/A')
            
            # 添加主要结果
            if entry['results']:
                for metric in ['val_acc', 'macro_recall', 'pneumonia_recall']:
                    row[metric] = entry['results'].get(metric, 0)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / 'optimization_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        print(f"[保存] 优化摘要: {summary_file}")
        
        # 打印Top 3配置
        if len(summary_data) > 0:
            print(f"\n{'='*60}")
            print(f"Top 3 配置:")
            print(f"{'='*60}")
            top3 = summary_df.nlargest(3, 'score')
            print(top3.to_string(index=False))
    
    def optimize(self, max_iterations: int = 10):
        """
        运行优化循环
        
        Args:
            max_iterations: 最大迭代次数
        """
        print(f"\n{'='*60}")
        print(f"开始自动超参数优化")
        print(f"{'='*60}")
        print(f"最大迭代次数: {max_iterations}")
        print(f"目标指标: {self.target_metric}")
        print(f"\n")
        
        for i in range(max_iterations):
            self.iteration = i
            
            # 生成下一个配置
            config = self.get_next_config()
            
            # 运行训练
            results = self.run_training(config)
            
            # 更新历史
            self.update_history(config, results)
            
            # 保存当前进度
            self.save_history()
            
            print(f"\n当前进度: {i+1}/{max_iterations}")
            if self.best_score > 0:
                print(f"最佳 {self.target_metric}: {self.best_score:.4f}")
        
        # 最终报告
        self.print_final_report()
    
    def print_final_report(self):
        """打印最终优化报告"""
        print(f"\n{'='*60}")
        print(f"优化完成!")
        print(f"{'='*60}")
        
        if self.best_config:
            print(f"\n🏆 最佳配置:")
            print(f"  {self.target_metric}: {self.best_score:.4f}")
            print(f"\n配置详情:")
            for key, value in self.best_config.items():
                if key not in ['output_dir']:
                    print(f"  {key}: {value}")
            
            print(f"\n最佳配置已保存:")
            print(f"  配置: {self.output_dir / 'best_config.yaml'}")
            print(f"  模型: {self.output_dir / 'best_model.pt'}")
            print(f"  历史: {self.output_dir / 'optimization_history.json'}")
            print(f"  摘要: {self.output_dir / 'optimization_summary.csv'}")
            
            # 给出使用建议
            print(f"\n📝 使用最佳配置:")
            print(f"  python src/train.py --config {self.output_dir / 'best_config.yaml'}")
        else:
            print("\n⚠️ 未找到有效配置")


def main():
    parser = argparse.ArgumentParser(description='自动超参数优化')
    parser.add_argument('--config', required=True, help='基础配置文件路径')
    parser.add_argument('--iterations', type=int, default=10, help='最大迭代次数')
    parser.add_argument('--target', default='macro_recall', 
                       choices=['macro_recall', 'pneumonia_recall', 'val_acc', 'macro_f1'],
                       help='优化目标指标')
    parser.add_argument('--output_dir', default='runs/auto_optimization', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(
        base_config_path=args.config,
        target_metric=args.target,
        output_dir=args.output_dir
    )
    
    # 运行优化
    optimizer.optimize(max_iterations=args.iterations)


if __name__ == '__main__':
    main()

