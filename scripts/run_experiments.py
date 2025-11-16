"""
Batch experiment runner for systematic model comparison.
Runs multiple training configurations and generates comparison reports.
"""

import argparse
import subprocess
import yaml
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def run_experiment(config_path, data_root, run_name, output_dir):
    """
    Run a single training experiment.
    
    Returns:
        metrics: dict with final metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {run_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    cmd = [
        'python', '-m', 'src.train',
        '--config', str(config_path),
        '--data_root', str(data_root),
        '--save_dir', str(save_dir)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ Training completed for {run_name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed for {run_name}: {e}")
        return None
    
    # Run evaluation on validation set
    ckpt_path = save_dir / 'best.pt'
    if not ckpt_path.exists():
        print(f"✗ Checkpoint not found: {ckpt_path}")
        return None
    
    report_path = save_dir / 'eval_report.json'
    eval_cmd = [
        'python', '-m', 'src.eval',
        '--ckpt', str(ckpt_path),
        '--data_root', str(data_root),
        '--split', 'val',
        '--report', str(report_path),
        '--threshold_sweep'
    ]
    
    try:
        subprocess.run(eval_cmd, check=True, capture_output=False, text=True)
        print(f"✓ Evaluation completed for {run_name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed for {run_name}: {e}")
        return None
    
    # Load metrics
    if report_path.exists():
        with open(report_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    else:
        return None


def generate_comparison_report(results, output_path):
    """
    Generate comparison report from multiple experiment results.
    """
    if not results:
        print("No results to compare")
        return
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for run_name, metrics in results.items():
        if metrics is None:
            continue
        
        row = {
            'experiment': run_name,
            'accuracy': metrics['metrics']['overall']['accuracy'],
            'macro_f1': metrics['metrics']['macro_f1'],
            'macro_recall': metrics['metrics']['overall']['macro_recall'],
            'roc_auc': metrics['metrics'].get('roc_auc', None),
            'pr_auc': metrics['metrics'].get('pr_auc', None)
        }
        
        # Add per-class metrics
        for class_name, class_metrics in metrics['metrics']['per_class'].items():
            row[f'{class_name}_recall'] = class_metrics.get('recall', 0)
            row[f'{class_name}_precision'] = class_metrics.get('precision', 0)
            row[f'{class_name}_f1'] = class_metrics.get('f1-score', 0)
        
        # Add threshold sweep results if available
        if 'threshold_sweep' in metrics['metrics']:
            ts = metrics['metrics']['threshold_sweep']
            row['max_recall_threshold'] = ts['max_recall_mode']['threshold']
            row['max_recall_value'] = ts['max_recall_mode']['recall']
            row['balanced_threshold'] = ts['balanced_mode']['threshold']
            row['balanced_f1'] = ts['balanced_mode']['f1']
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison report saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    # Find best experiments
    if 'PNEUMONIA_recall' in df.columns:
        best_recall = df.loc[df['PNEUMONIA_recall'].idxmax()]
        print(f"🏆 Best Pneumonia Recall: {best_recall['experiment']} ({best_recall['PNEUMONIA_recall']:.4f})")
    
    if 'macro_f1' in df.columns:
        best_f1 = df.loc[df['macro_f1'].idxmax()]
        print(f"🏆 Best Macro F1: {best_f1['experiment']} ({best_f1['macro_f1']:.4f})")
    
    if 'accuracy' in df.columns:
        best_acc = df.loc[df['accuracy'].idxmax()]
        print(f"🏆 Best Accuracy: {best_acc['experiment']} ({best_acc['accuracy']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument('--configs', nargs='+', required=True, help='List of config files to run')
    parser.add_argument('--data_root', default='data', help='Data root directory')
    parser.add_argument('--output_dir', default='experiments', help='Output directory for all runs')
    parser.add_argument('--skip_existing', action='store_true', help='Skip if checkpoint already exists')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Record start time
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"Batch Experiment Runner")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of experiments: {len(args.configs)}")
    print(f"{'='*60}\n")
    
    # Run all experiments
    results = {}
    
    for config_path in args.configs:
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            continue
        
        # Generate run name from config filename
        run_name = config_path.stem
        
        # Check if already exists
        if args.skip_existing:
            existing_ckpt = output_dir / run_name / 'best.pt'
            if existing_ckpt.exists():
                print(f"⊘ Skipping {run_name} (checkpoint already exists)")
                
                # Try to load existing results
                existing_report = output_dir / run_name / 'eval_report.json'
                if existing_report.exists():
                    with open(existing_report, 'r') as f:
                        results[run_name] = json.load(f)
                
                continue
        
        # Run experiment
        metrics = run_experiment(config_path, args.data_root, run_name, output_dir)
        results[run_name] = metrics
    
    # Generate comparison report
    if results:
        comparison_path = output_dir / f"comparison_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        generate_comparison_report(results, comparison_path)
    
    # Record end time
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
