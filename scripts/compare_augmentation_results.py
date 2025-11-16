"""
Compare augmentation experiment results.
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(run_dir):
    """Load metrics.csv from a run directory."""
    csv_path = Path(run_dir) / 'metrics.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None

def main():
    # Define experiment runs
    runs = {
        'Light': 'runs/aug_light',
        'Medium': 'runs/aug_medium',
        'Aggressive': 'runs/aug_aggressive'
    }
    
    # Load all metrics
    all_data = {}
    best_results = {}
    
    for name, run_dir in runs.items():
        df = load_metrics(run_dir)
        if df is not None:
            all_data[name] = df
            # Get best epoch metrics
            best_idx = df['macro_recall'].idxmax()
            best_results[name] = {
                'epoch': df.loc[best_idx, 'epoch'],
                'val_acc': df.loc[best_idx, 'val_acc'],
                'pneumonia_recall': df.loc[best_idx, 'pneumonia_recall'],
                'pneumonia_precision': df.loc[best_idx, 'pneumonia_precision'],
                'normal_recall': df.loc[best_idx, 'normal_recall'],
                'macro_recall': df.loc[best_idx, 'macro_recall'],
                'macro_f1': df.loc[best_idx, 'macro_f1'],
                'final_epoch': len(df)
            }
    
    # Print comparison table
    print("\n" + "="*80)
    print("AUGMENTATION LEVEL COMPARISON - BEST RESULTS")
    print("="*80)
    print(f"\n{'Metric':<25} {'Light':>15} {'Medium':>15} {'Aggressive':>15}")
    print("-"*80)
    
    metrics = [
        ('Best Epoch', 'epoch'),
        ('Final Epoch', 'final_epoch'),
        ('Validation Accuracy', 'val_acc'),
        ('Pneumonia Recall', 'pneumonia_recall'),
        ('Pneumonia Precision', 'pneumonia_precision'),
        ('Normal Recall', 'normal_recall'),
        ('Macro Recall', 'macro_recall'),
        ('Macro F1', 'macro_f1')
    ]
    
    for metric_name, metric_key in metrics:
        print(f"{metric_name:<25}", end='')
        for aug_level in ['Light', 'Medium', 'Aggressive']:
            if aug_level in best_results:
                val = best_results[aug_level][metric_key]
                if metric_key in ['epoch', 'final_epoch']:
                    print(f"{int(val):>15}", end='')
                else:
                    print(f"{val:>15.4f}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()
    
    print("="*80)
    
    # Determine winner for key metrics
    print("\n🏆 KEY FINDINGS:")
    print("-"*80)
    
    key_metrics = {
        'Pneumonia Recall (sensitivity)': 'pneumonia_recall',
        'Macro Recall (balanced)': 'macro_recall',
        'Validation Accuracy': 'val_acc',
        'Macro F1': 'macro_f1'
    }
    
    for metric_name, metric_key in key_metrics.items():
        best_aug = max(best_results.items(), key=lambda x: x[1][metric_key])
        print(f"  • Best {metric_name}: {best_aug[0]} ({best_aug[1][metric_key]:.4f})")
    
    # Training efficiency
    print("\n📊 TRAINING EFFICIENCY:")
    print("-"*80)
    for name in ['Light', 'Medium', 'Aggressive']:
        if name in best_results:
            print(f"  • {name}: Converged at epoch {int(best_results[name]['epoch'])} "
                  f"(trained {int(best_results[name]['final_epoch'])} total)")
    
    # Create comparison plots
    if all_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Augmentation Level Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Macro Recall over epochs
        ax = axes[0, 0]
        for name, df in all_data.items():
            ax.plot(df['epoch'], df['macro_recall'], marker='o', label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Macro Recall')
        ax.set_title('Macro Recall (Balanced Metric)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Pneumonia Recall over epochs
        ax = axes[0, 1]
        for name, df in all_data.items():
            ax.plot(df['epoch'], df['pneumonia_recall'], marker='s', label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pneumonia Recall')
        ax.set_title('Pneumonia Recall (Sensitivity)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Training Loss over epochs
        ax = axes[1, 0]
        for name, df in all_data.items():
            ax.plot(df['epoch'], df['train_loss'], marker='^', label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 4: Best metrics comparison (bar chart)
        ax = axes[1, 1]
        metrics_to_plot = ['pneumonia_recall', 'macro_recall', 'val_acc']
        x = range(len(metrics_to_plot))
        width = 0.25
        
        for i, name in enumerate(['Light', 'Medium', 'Aggressive']):
            if name in best_results:
                values = [best_results[name][m] for m in metrics_to_plot]
                ax.bar([xi + i*width for xi in x], values, width, label=name)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Best Metrics Comparison')
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(['Pneumonia\nRecall', 'Macro\nRecall', 'Val\nAccuracy'])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0.90, 1.0])
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path('reports/plots/augmentation_comparison.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Comparison plot saved to: {output_path}")
        
        plt.show()
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
