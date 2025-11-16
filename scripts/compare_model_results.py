"""
对比不同模型架构的训练结果
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 模型列表
models = ['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b2', 'densenet121']
model_names = {
    'resnet18': 'ResNet18',
    'resnet50': 'ResNet50', 
    'efficientnet_b0': 'EfficientNet-B0',
    'efficientnet_b2': 'EfficientNet-B2',
    'densenet121': 'DenseNet121'
}

# 加载所有模型的训练数据
all_metrics = {}
best_results = []

for model in models:
    metrics_path = Path(f"runs/model_{model}/metrics.csv")
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        all_metrics[model] = df
        
        # 找到最佳 macro_recall
        best_idx = df['macro_recall'].idxmax()
        best_row = df.loc[best_idx]
        
        best_results.append({
            'model': model_names[model],
            'best_epoch': int(best_row['epoch']),
            'total_epochs': len(df),
            'macro_recall': best_row['macro_recall'],
            'val_acc': best_row['val_acc'],
            'pneumonia_recall': best_row['pneumonia_recall'],
            'normal_recall': best_row['normal_recall'],
            'train_loss': best_row['train_loss'],
            'val_loss': best_row['val_loss']
        })

# 创建对比表格
results_df = pd.DataFrame(best_results)
results_df = results_df.sort_values('macro_recall', ascending=False)

print("\n" + "="*80)
print("模型架构对比结果")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# 保存结果到 JSON
output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "model_comparison.json", 'w', encoding='utf-8') as f:
    json.dump(best_results, f, indent=2, ensure_ascii=False)

# 创建可视化对比图 (4个子图)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('模型架构对比分析', fontsize=16, fontweight='bold')

# 1. Macro Recall 对比
ax1 = axes[0, 0]
bars1 = ax1.bar(results_df['model'], results_df['macro_recall'], color='skyblue', edgecolor='navy')
ax1.set_ylabel('Macro Recall', fontsize=12)
ax1.set_title('验证集 Macro Recall 对比', fontsize=13, fontweight='bold')
ax1.set_ylim([0.95, 1.0])
ax1.tick_params(axis='x', rotation=45)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# 2. 训练效率对比 (最佳轮次 vs 总轮数)
ax2 = axes[0, 1]
x = range(len(results_df))
width = 0.35
bars2a = ax2.bar([i - width/2 for i in x], results_df['best_epoch'], 
                 width, label='最佳轮次', color='coral', edgecolor='darkred')
bars2b = ax2.bar([i + width/2 for i in x], results_df['total_epochs'], 
                 width, label='总轮数', color='lightgreen', edgecolor='darkgreen')
ax2.set_ylabel('轮次', fontsize=12)
ax2.set_title('训练效率对比 (早停机制)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['model'], rotation=45)
ax2.legend()
for bars in [bars2a, bars2b]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 3. 类别召回率对比
ax3 = axes[1, 0]
x = range(len(results_df))
width = 0.35
bars3a = ax3.bar([i - width/2 for i in x], results_df['pneumonia_recall'], 
                 width, label='肺炎召回率', color='#FF6B6B', edgecolor='darkred')
bars3b = ax3.bar([i + width/2 for i in x], results_df['normal_recall'], 
                 width, label='正常召回率', color='#4ECDC4', edgecolor='darkblue')
ax3.set_ylabel('召回率', fontsize=12)
ax3.set_title('类别召回率对比', fontsize=13, fontweight='bold')
ax3.set_ylim([0.92, 1.0])
ax3.set_xticks(x)
ax3.set_xticklabels(results_df['model'], rotation=45)
ax3.legend()

# 4. 训练曲线对比 (Macro Recall)
ax4 = axes[1, 1]
colors = plt.cm.tab10(range(len(models)))
for i, (model, color) in enumerate(zip(models, colors)):
    if model in all_metrics:
        df = all_metrics[model]
        ax4.plot(df['epoch'], df['macro_recall'], 
                label=model_names[model], linewidth=2, color=color, marker='o', markersize=4)
ax4.set_xlabel('轮次', fontsize=12)
ax4.set_ylabel('Macro Recall', fontsize=12)
ax4.set_title('训练曲线对比', fontsize=13, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "plots" / "model_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n可视化图表已保存到: {output_dir / 'plots' / 'model_comparison.png'}")
print(f"对比结果已保存到: {output_dir / 'model_comparison.json'}")

# 生成 Markdown 表格
print("\n" + "="*80)
print("Markdown 表格 (可直接复制到文档):")
print("="*80)
print("\n| 排名 | 模型 | Macro Recall | Val Acc | 肺炎召回 | 正常召回 | 最佳轮次 | 总轮数 |")
print("|------|------|--------------|---------|----------|----------|----------|--------|")
for idx, row in results_df.iterrows():
    rank = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{list(results_df.index).index(idx) + 1}"
    print(f"| {rank} | **{row['model']}** | **{row['macro_recall']:.2%}** | {row['val_acc']:.2%} | "
          f"{row['pneumonia_recall']:.2%} | {row['normal_recall']:.2%} | {row['best_epoch']} | {row['total_epochs']} |")
print("\n")
