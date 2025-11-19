"""
Comprehensive experiment comparison and analysis script.
Generates comparison tables and visualizations for all experiments.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def load_experiment_results(runs_dir):
    """
    从 runs 目录加载所有实验的 metrics_history.csv
    
    Returns:
        experiments: dict {experiment_name: DataFrame}
    """
    runs_dir = Path(runs_dir)
    experiments = {}
    
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir():
            # 修复：正确的文件名是 metrics_history.csv
            metrics_file = exp_dir / 'metrics_history.csv'
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    experiments[exp_dir.name] = df
                except Exception as e:
                    print(f"Warning: Failed to load {metrics_file}: {e}")
    
    return experiments


def extract_best_metrics(experiments):
    """
    从每个实验中提取最佳指标(基于 macro_recall)
    
    Returns:
        summary_df: DataFrame with best metrics for each experiment
    """
    summary_rows = []
    
    for exp_name, df in experiments.items():
        if df.empty or 'macro_recall' not in df.columns:
            continue
        
        # Find best epoch based on macro_recall
        best_idx = df['macro_recall'].idxmax()
        best_row = df.loc[best_idx]
        
        # 计算训练时间（如果有时间戳数据）
        if 'timestamp' in df.columns:
            try:
                # 从时间戳计算实际训练时间
                train_time_min = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 60.0
            except Exception:
                train_time_min = np.nan
        else:
            # 估算：假设每个 epoch 平均 3-5 分钟（取决于模型和数据集大小）
            # 这只是粗略估计，实际时间会因硬件和配置而异
            estimated_time_per_epoch = 4.0  # 平均每个epoch 4分钟
            train_time_min = best_row.get('epoch', 0) * estimated_time_per_epoch
        
        summary_rows.append({
            'Experiment': exp_name,
            'Best Epoch': int(best_row.get('epoch', -1)),
            'Val Accuracy': best_row.get('val_acc', np.nan),
            'Macro Recall': best_row.get('macro_recall', np.nan),
            'Macro F1': best_row.get('macro_f1', np.nan),
            'Pneumonia Recall': best_row.get('pneumonia_recall', np.nan),
            'Pneumonia Precision': best_row.get('pneumonia_precision', np.nan),
            'Pneumonia F1': best_row.get('pneumonia_f1', np.nan),
            'Normal Recall': best_row.get('normal_recall', np.nan),
            'Normal Precision': best_row.get('normal_precision', np.nan),
            'Val Loss': best_row.get('val_loss', np.nan),
            'Train Time (min)': train_time_min
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by Macro Recall (descending)
    summary_df = summary_df.sort_values('Macro Recall', ascending=False).reset_index(drop=True)
    
    return summary_df


def plot_experiment_comparison(summary_df, save_dir, top_n=10):
    """
    绘制实验对比图表
    
    Args:
        summary_df: 实验总结 DataFrame
        save_dir: 保存目录
        top_n: 显示前 N 个实验
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Select top experiments
    df = summary_df.head(top_n).copy()
    
    # Plot 1: Macro Recall comparison (horizontal bar)
    _fig1, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))
    
    # 使用colormap对recall值着色
    norm = plt.Normalize(vmin=0.9, vmax=1.0)
    cmap = plt.get_cmap('RdYlGn')
    colors = cmap(norm(df['Macro Recall'].values))
    bars = ax.barh(df['Experiment'], df['Macro Recall'], color=colors, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, df['Macro Recall']):
        ax.text(value + 0.002, bar.get_y() + bar.get_height()/2, 
               f'{value:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Macro Recall (Primary KPI)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(df)} Experiments by Macro Recall', fontsize=14, fontweight='bold')
    ax.set_xlim([0.9, 1.0])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'macro_recall_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'macro_recall_comparison.png'}")
    
    # Plot 2: Multi-metric heatmap
    metric_cols = ['Val Accuracy', 'Macro Recall', 'Macro F1', 
                   'Pneumonia Recall', 'Pneumonia Precision', 'Pneumonia F1']
    
    # Select available columns
    available_cols = [col for col in metric_cols if col in df.columns]
    
    if available_cols:
        heatmap_data = df[['Experiment'] + available_cols].set_index('Experiment')
        
        _fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                   vmin=0.9, vmax=1.0, cbar_kws={'label': 'Score'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title(f'Top {len(df)} Experiments: Multi-Metric Heatmap', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_dir / 'metrics_heatmap.png'}")
    
    # Plot 3: Pneumonia Recall vs Precision scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(df['Pneumonia Recall'], df['Pneumonia Precision'],
                        s=200, c=df['Macro F1'], cmap='viridis', 
                        edgecolors='black', linewidth=1.5, alpha=0.7)
    
    # Annotate points
    for idx, row in df.iterrows():
        ax.annotate(row['Experiment'], 
                   xy=(row['Pneumonia Recall'], row['Pneumonia Precision']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Pneumonia Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pneumonia Precision (PPV)', fontsize=12, fontweight='bold')
    ax.set_title('Pneumonia Detection: Recall vs Precision Trade-off', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.9, 1.01])
    ax.set_ylim([0.9, 1.01])
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Macro F1-Score', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'recall_precision_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'recall_precision_scatter.png'}")
    
    # Plot 4: Training efficiency (Macro Recall vs Training Time)
    _fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(df['Train Time (min)'], df['Macro Recall'],
                        s=200, c=df['Val Accuracy'], cmap='plasma', 
                        edgecolors='black', linewidth=1.5, alpha=0.7)
    
    # Annotate points
    for idx, row in df.iterrows():
        ax.annotate(row['Experiment'], 
                   xy=(row['Train Time (min)'], row['Macro Recall']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Approximate Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro Recall', fontsize=12, fontweight='bold')
    ax.set_title('Training Efficiency: Performance vs Time', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Validation Accuracy', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'efficiency_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'efficiency_comparison.png'}")


def plot_training_curves(experiments, save_dir, top_n=5):
    """
    绘制训练曲线对比
    
    Args:
        experiments: dict {experiment_name: DataFrame}
        save_dir: 保存目录
        top_n: 显示前 N 个实验的曲线
    """
    save_dir = Path(save_dir)
    
    # Select top experiments by final macro_recall
    exp_scores = {name: df['macro_recall'].max() 
                  for name, df in experiments.items() 
                  if 'macro_recall' in df.columns}
    
    top_exps = sorted(exp_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Plot Pneumonia Recall curves
    _fig1, ax = plt.subplots(figsize=(12, 7))
    
    for exp_name, _ in top_exps:
        df = experiments[exp_name]
        if 'pneumonia_recall' in df.columns:
            ax.plot(df['epoch'], df['pneumonia_recall'], 
                   marker='o', linewidth=2, label=exp_name, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pneumonia Recall (Primary KPI)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(top_exps)} Models: Pneumonia Recall Learning Curves', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.01])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'pneumonia_recall_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'pneumonia_recall_curves.png'}")
    
    # Plot Loss curves
    _fig2, ax = plt.subplots(figsize=(12, 7))
    
    for exp_name, _ in top_exps:
        df = experiments[exp_name]
        if 'val_loss' in df.columns:
            ax.plot(df['epoch'], df['val_loss'], 
                   marker='s', linewidth=2, label=exp_name, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(top_exps)} Models: Validation Loss Curves', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'validation_loss_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'validation_loss_curves.png'}")


def generate_summary_report(summary_df, save_path):
    """
    生成 Markdown 格式的实验总结报告
    
    Args:
        summary_df: 实验总结 DataFrame
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment Comparison Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(summary_df)}\n\n")
        
        f.write("---\n\n")
        
        # Top 3 models
        f.write("## 🏆 Top 3 Models by Macro Recall\n\n")
        
        for idx, row in summary_df.head(3).iterrows():
            rank = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉"
            f.write(f"### {rank} Rank {idx + 1}: {row['Experiment']}\n\n")
            f.write(f"- **Macro Recall:** {row['Macro Recall']:.4f}\n")
            f.write(f"- **Val Accuracy:** {row['Val Accuracy']:.4f}\n")
            f.write(f"- **Pneumonia Recall:** {row['Pneumonia Recall']:.4f}\n")
            f.write(f"- **Pneumonia Precision:** {row['Pneumonia Precision']:.4f}\n")
            f.write(f"- **Normal Recall:** {row['Normal Recall']:.4f}\n")
            f.write(f"- **Best Epoch:** {int(row['Best Epoch'])}\n")
            f.write(f"- **Training Time:** ~{row['Train Time (min)']:.1f} minutes\n\n")
        
        f.write("---\n\n")
        
        # Full comparison table
        f.write("## 📊 Complete Experiment Comparison\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        
        f.write("---\n\n")
        
        # Key findings
        f.write("## 🔑 Key Findings\n\n")
        
        best_model = summary_df.iloc[0]
        f.write(f"1. **Best Overall Model:** {best_model['Experiment']}\n")
        f.write(f"   - Achieves {best_model['Macro Recall']:.4f} macro recall\n")
        f.write(f"   - Pneumonia sensitivity: {best_model['Pneumonia Recall']:.4f}\n\n")
        
        fastest = summary_df.loc[summary_df['Train Time (min)'].idxmin()]
        f.write(f"2. **Fastest Training:** {fastest['Experiment']}\n")
        f.write(f"   - Training time: ~{fastest['Train Time (min)']:.1f} minutes\n")
        f.write(f"   - Macro recall: {fastest['Macro Recall']:.4f}\n\n")
        
        best_pneumonia = summary_df.loc[summary_df['Pneumonia Recall'].idxmax()]
        f.write(f"3. **Highest Pneumonia Recall:** {best_pneumonia['Experiment']}\n")
        f.write(f"   - Pneumonia recall: {best_pneumonia['Pneumonia Recall']:.4f}\n")
        f.write(f"   - Minimizes false negatives (critical for medical screening)\n\n")
        
        f.write("---\n\n")
        f.write("**Note:** All metrics are based on validation set performance at the best epoch (selected by macro recall).\n")
    
    print(f"✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare all experiments and generate analysis")
    parser.add_argument('--runs_dir', default='runs', help='Directory containing experiment runs')
    parser.add_argument('--output_dir', default='reports/experiment_analysis', help='Output directory')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top experiments to visualize')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    print(f"Loading experiments from: {args.runs_dir}")
    experiments = load_experiment_results(args.runs_dir)
    print(f"✓ Loaded {len(experiments)} experiments")
    
    if not experiments:
        print("ERROR: No experiment results found!")
        return
    
    # Extract best metrics
    print("\nExtracting best metrics from each experiment...")
    summary_df = extract_best_metrics(experiments)
    
    # Save summary table
    summary_csv = output_dir / 'experiment_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary table: {summary_csv}")
    
    # Print top 5
    print("\n" + "="*80)
    print("TOP 5 EXPERIMENTS BY MACRO RECALL")
    print("="*80)
    print(summary_df.head(5).to_string(index=False))
    print("="*80 + "\n")
    
    # Generate visualizations
    print("Generating comparison plots...")
    plot_experiment_comparison(summary_df, output_dir, top_n=args.top_n)
    
    print("\nGenerating training curves...")
    plot_training_curves(experiments, output_dir, top_n=min(5, len(experiments)))
    
    # Generate markdown report
    print("\nGenerating summary report...")
    generate_summary_report(summary_df, output_dir / 'EXPERIMENT_SUMMARY.md')
    
    print(f"\n✅ Experiment analysis complete! Check {output_dir} for all outputs.")


if __name__ == '__main__':
    main()
