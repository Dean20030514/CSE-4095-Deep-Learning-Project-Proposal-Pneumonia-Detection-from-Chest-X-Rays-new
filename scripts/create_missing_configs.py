"""
Master script to create missing configuration files for comprehensive experiments.
Generates configs for: high-res training, focal loss variants, and Track-B advanced models.
"""

from pathlib import Path
import yaml

# Base configuration template
BASE_CONFIG = {
    'seed': 42,
    'optimizer': 'adamw',
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'amp': True,
    'num_workers': 0,
    'use_albumentations': True,
    'early_stopping': {
        'metric': 'pneumonia_recall',
        'patience': 5
    },
    'log': {
        'csv': 'metrics.csv'
    }
}

# Configuration variants to generate
CONFIGS = {
    # High-resolution variants (Track-B)
    'high_res_resnet18_512.yaml': {
        **BASE_CONFIG,
        'model': 'resnet18',
        'img_size': 512,
        'batch_size': 8,  # Reduced for memory
        'epochs': 20,
        'lr': 2e-4,
        'loss': 'weighted_ce',
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'log': {'run_name': 'high_res_resnet18_512', 'csv': 'metrics.csv'}
    },
    
    'high_res_efficientnet_b3_512.yaml': {
        **BASE_CONFIG,
        'model': 'efficientnet_b3',
        'img_size': 512,
        'batch_size': 6,
        'epochs': 20,
        'lr': 1e-4,
        'loss': 'weighted_ce',
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'log': {'run_name': 'high_res_efficientnet_b3', 'csv': 'metrics.csv'}
    },
    
    # Focal Loss variants
    'focal_loss_gamma15.yaml': {
        **BASE_CONFIG,
        'model': 'efficientnet_b0',
        'img_size': 384,
        'batch_size': 16,
        'epochs': 25,
        'lr': 3e-4,
        'loss': 'focal',
        'focal': {'gamma': 1.5},
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'log': {'run_name': 'focal_gamma15', 'csv': 'metrics.csv'}
    },
    
    'focal_loss_gamma20.yaml': {
        **BASE_CONFIG,
        'model': 'efficientnet_b0',
        'img_size': 384,
        'batch_size': 16,
        'epochs': 25,
        'lr': 3e-4,
        'loss': 'focal',
        'focal': {'gamma': 2.0},
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'log': {'run_name': 'focal_gamma20', 'csv': 'metrics.csv'}
    },
    
    'focal_loss_gamma25.yaml': {
        **BASE_CONFIG,
        'model': 'efficientnet_b0',
        'img_size': 384,
        'batch_size': 16,
        'epochs': 25,
        'lr': 3e-4,
        'loss': 'focal',
        'focal': {'gamma': 2.5},
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'log': {'run_name': 'focal_gamma25', 'csv': 'metrics.csv'}
    },
    
    # Mixed resolution ensemble components
    'ensemble_resnet18_448.yaml': {
        **BASE_CONFIG,
        'model': 'resnet18',
        'img_size': 448,
        'batch_size': 12,
        'epochs': 25,
        'lr': 2e-4,
        'loss': 'weighted_ce',
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'log': {'run_name': 'ensemble_resnet18_448', 'csv': 'metrics.csv'}
    },
    
    'ensemble_efficientnet_b1_448.yaml': {
        **BASE_CONFIG,
        'model': 'efficientnet_b1',
        'img_size': 448,
        'batch_size': 12,
        'epochs': 25,
        'lr': 2e-4,
        'loss': 'weighted_ce',
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'log': {'run_name': 'ensemble_efficientnet_b1', 'csv': 'metrics.csv'}
    },
    
    # Fast prototyping configs
    'quick_test_resnet18.yaml': {
        **BASE_CONFIG,
        'model': 'resnet18',
        'img_size': 224,
        'batch_size': 32,
        'epochs': 3,
        'lr': 1e-3,
        'loss': 'weighted_ce',
        'sampler': 'weighted_random',
        'augment_level': 'light',
        'early_stopping': {'metric': 'pneumonia_recall', 'patience': 10},
        'log': {'run_name': 'quick_test', 'csv': 'metrics.csv'}
    },
    
    # Medical screening optimized (max recall)
    'medical_screening_optimized.yaml': {
        **BASE_CONFIG,
        'model': 'resnet18',
        'img_size': 384,
        'batch_size': 16,
        'epochs': 30,
        'lr': 2e-4,
        'loss': 'focal',
        'focal': {'gamma': 2.0},  # More focus on hard examples
        'sampler': 'weighted_random',
        'augment_level': 'medium',
        'early_stopping': {
            'metric': 'pneumonia_recall',  # Optimize for sensitivity
            'patience': 7
        },
        'log': {'run_name': 'medical_screening_optimized', 'csv': 'metrics.csv'}
    }
}


def create_configs(output_dir='src/configs'):
    """Generate all configuration files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating configuration files...")
    print("=" * 60)
    
    for filename, config in CONFIGS.items():
        filepath = output_path / filename
        
        # Check if file already exists
        if filepath.exists():
            print(f"⚠️  Skipping {filename} (already exists)")
            continue
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created: {filepath}")
        print(f"  Model: {config['model']} @ {config['img_size']}px")
        print(f"  Loss: {config['loss']}, Epochs: {config['epochs']}, LR: {config['lr']}")
        print()
    
    print("=" * 60)
    print(f"✓ All configurations created in: {output_path}")
    print()
    print("Usage examples:")
    print("  python -m src.train --config src/configs/high_res_resnet18_512.yaml")
    print("  python -m src.train --config src/configs/focal_loss_gamma20.yaml")
    print("  python -m src.train --config src/configs/quick_test_resnet18.yaml --epochs 2")


if __name__ == '__main__':
    create_configs()
