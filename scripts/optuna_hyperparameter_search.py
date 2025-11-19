"""
使用Optuna进行智能超参数搜索

Optuna是一个自动化超参数优化框架，使用贝叶斯优化算法
可以更智能地探索参数空间，比随机搜索或网格搜索更高效

安装：pip install optuna

使用方法：
    python scripts/optuna_hyperparameter_search.py --n_trials 20 --target pneumonia_recall

优势：
- 智能采样：基于过去的试验结果选择下一个参数组合
- 剪枝机制：提前终止表现不佳的训练
- 可视化：内置优化过程可视化
- 分布式：支持并行搜索

作者：CSE-4095 Deep Learning Team
"""

import argparse
import yaml
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict
import optuna
from optuna.trial import TrialState
from datetime import datetime
import json


class OptunaTrainer:
    """使用Optuna进行超参数优化的训练器"""
    
    def __init__(self, base_config_path: str, target_metric: str = 'macro_recall',
                 study_name: str = None, storage_path: str = None):
        """
        Args:
            base_config_path: 基础配置文件
            target_metric: 优化目标 (macro_recall, pneumonia_recall等)
            study_name: Optuna study名称
            storage_path: 数据库存储路径（用于持久化和分布式）
        """
        self.base_config_path = Path(base_config_path)
        self.target_metric = target_metric
        
        # 加载基础配置
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        
        # Optuna配置
        self.study_name = study_name or f"pneumonia_{target_metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_path = storage_path
        
        # 输出目录
        self.output_dir = Path('runs/optuna_optimization') / self.study_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[OptunaTrainer] 初始化完成")
        print(f"  - Study: {self.study_name}")
        print(f"  - 目标指标: {target_metric}")
        print(f"  - 输出: {self.output_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna的目标函数
        
        Args:
            trial: Optuna试验对象
        
        Returns:
            目标指标的值（Optuna会最大化这个值）
        """
        # 1. 建议超参数
        config = self.base_config.copy()
        
        # 学习率（对数尺度）
        config['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        # Batch size（离散值）
        config['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        # 模型架构
        config['model'] = trial.suggest_categorical(
            'model', 
            ['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0']
        )
        
        # 数据增强级别
        config['augment_level'] = trial.suggest_categorical(
            'augment_level', 
            ['light', 'medium', 'aggressive']
        )
        
        # 图像尺寸
        config['img_size'] = trial.suggest_categorical('img_size', [224, 384])
        
        # Weight decay
        config['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        
        # 损失函数
        config['loss'] = trial.suggest_categorical('loss', ['weighted_ce', 'focal'])
        
        if config['loss'] == 'focal':
            # Focal loss gamma
            config['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)
        
        # Epochs（可以设置较小值加速搜索）
        config['epochs'] = trial.suggest_int('epochs', 10, 30)
        
        # 2. 设置输出目录
        trial_dir = self.output_dir / f'trial_{trial.number}'
        config['output_dir'] = str(trial_dir)
        
        # 3. 保存配置
        config_path = trial_dir / 'config.yaml'
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # 4. 运行训练
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: 开始训练")
        print(f"{'='*60}")
        print(f"参数:")
        for key, value in config.items():
            if key not in ['output_dir', 'data_root']:
                print(f"  {key}: {value}")
        
        cmd = [
            'python', 'src/train.py',
            '--config', str(config_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode != 0:
                print(f"[ERROR] 训练失败")
                raise optuna.TrialPruned()
            
            print(f"[SUCCESS] 训练完成")
            
        except Exception as e:
            print(f"[ERROR] 运行失败: {e}")
            raise optuna.TrialPruned()
        
        # 5. 提取结果
        score = self.extract_score(trial_dir)
        
        if score is None:
            raise optuna.TrialPruned()
        
        print(f"\nTrial {trial.number} 结果: {self.target_metric} = {score:.4f}")
        
        return score
    
    def extract_score(self, trial_dir: Path) -> float:
        """从训练结果中提取目标指标"""
        metrics_file = trial_dir / 'metrics_history.csv'
        
        if not metrics_file.exists():
            return None
        
        try:
            df = pd.read_csv(metrics_file)
            
            if self.target_metric in df.columns:
                # 返回最佳epoch的目标指标
                best_score = df[self.target_metric].max()
                return float(best_score)
            else:
                return None
                
        except Exception as e:
            print(f"[ERROR] 读取结果失败: {e}")
            return None
    
    def optimize(self, n_trials: int = 20, n_jobs: int = 1, timeout: int = None):
        """
        运行Optuna优化
        
        Args:
            n_trials: 试验次数
            n_jobs: 并行任务数（>1时启用并行搜索）
            timeout: 超时时间（秒）
        """
        # 创建或加载study
        storage = f'sqlite:///{self.output_dir}/optuna.db' if self.storage_path is None else self.storage_path
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction='maximize',  # 最大化目标指标
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)  # 贝叶斯优化
        )
        
        print(f"\n{'='*60}")
        print(f"开始Optuna超参数搜索")
        print(f"{'='*60}")
        print(f"Trial数量: {n_trials}")
        print(f"并行任务: {n_jobs}")
        print(f"目标指标: {self.target_metric} (最大化)")
        print(f"\n")
        
        # 运行优化
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # 输出结果
        self.print_results(study)
        
        # 保存结果
        self.save_results(study)
        
        return study
    
    def print_results(self, study: optuna.Study):
        """打印优化结果"""
        print(f"\n{'='*60}")
        print(f"优化完成!")
        print(f"{'='*60}")
        
        print(f"\n总试验数: {len(study.trials)}")
        print(f"  - 完成: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
        print(f"  - 剪枝: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
        print(f"  - 失败: {len([t for t in study.trials if t.state == TrialState.FAIL])}")
        
        if len(study.best_trials) > 0:
            print(f"\n🏆 最佳结果:")
            print(f"  Trial {study.best_trial.number}")
            print(f"  {self.target_metric}: {study.best_value:.4f}")
            
            print(f"\n最佳参数:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            
            # 保存最佳配置
            best_config = self.base_config.copy()
            best_config.update(study.best_params)
            
            best_config_path = self.output_dir / 'best_config.yaml'
            with open(best_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(best_config, f)
            
            print(f"\n最佳配置已保存: {best_config_path}")
            
            # 参数重要性分析
            if len(study.trials) >= 10:
                print(f"\n📊 参数重要性分析:")
                try:
                    importances = optuna.importance.get_param_importances(study)
                    for param, importance in sorted(importances.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:5]:
                        print(f"  {param}: {importance:.3f}")
                except Exception:
                    pass
    
    def save_results(self, study: optuna.Study):
        """保存优化结果"""
        # 保存所有试验的摘要
        trials_data = []
        for trial in study.trials:
            if trial.state == TrialState.COMPLETE:
                row = {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'state': trial.state.name
                }
                row.update(trial.params)
                trials_data.append(row)
        
        if len(trials_data) > 0:
            df = pd.DataFrame(trials_data)
            summary_path = self.output_dir / 'trials_summary.csv'
            df.to_csv(summary_path, index=False)
            print(f"\n[保存] 试验摘要: {summary_path}")
            
            # 打印Top 5
            print(f"\nTop 5 配置:")
            top5 = df.nlargest(5, 'value')
            print(top5[['trial_number', 'value', 'lr', 'model', 'augment_level']].to_string(index=False))
        
        # 生成可视化（如果安装了matplotlib）
        try:
            import matplotlib.pyplot as plt
            
            # 优化历史
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig(self.output_dir / 'optimization_history.png', dpi=200, bbox_inches='tight')
            
            # 参数重要性
            if len(study.trials) >= 10:
                fig = optuna.visualization.matplotlib.plot_param_importances(study)
                fig.savefig(self.output_dir / 'param_importances.png', dpi=200, bbox_inches='tight')
            
            # Parallel coordinate plot
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            fig.savefig(self.output_dir / 'parallel_coordinate.png', dpi=200, bbox_inches='tight')
            
            print(f"[保存] 可视化图表: {self.output_dir}/*.png")
            
        except ImportError:
            print("[INFO] matplotlib未安装，跳过可视化")
        except Exception as e:
            print(f"[WARNING] 可视化生成失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='Optuna超参数搜索')
    parser.add_argument('--config', default='src/configs/baseline_resnet18.yaml',
                       help='基础配置文件')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='试验次数')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='并行任务数')
    parser.add_argument('--target', default='macro_recall',
                       choices=['macro_recall', 'pneumonia_recall', 'val_acc', 'macro_f1'],
                       help='优化目标指标')
    parser.add_argument('--timeout', type=int, default=None,
                       help='超时时间（秒）')
    parser.add_argument('--study_name', default=None,
                       help='Study名称（用于恢复或继续优化）')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = OptunaTrainer(
        base_config_path=args.config,
        target_metric=args.target,
        study_name=args.study_name
    )
    
    # 运行优化
    trainer.optimize(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout
    )


if __name__ == '__main__':
    main()

