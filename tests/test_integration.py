"""
集成测试：端到端工作流测试

测试完整的训练、评估、推理流程
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile


@pytest.mark.integration
class TestEndToEndWorkflow:
    """测试端到端工作流"""
    
    @pytest.mark.slow
    def test_minimal_training_loop(self, mock_dataset_dir, tmp_path, device):
        """测试最小化的训练循环"""
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        from src.train import FocalLoss, set_seed
        
        # 设置随机种子
        set_seed(42)
        
        # 构建模型和数据
        model, _ = build_model('resnet18', num_classes=2)
        model = model.to(device)
        
        loaders, class_to_idx = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 设置训练组件
        loss_fn = FocalLoss(gamma=1.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练一个epoch
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, targets in loaders['train']:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 只训练几个batch
            if num_batches >= 3:
                break
        
        # 验证训练有效
        assert num_batches > 0
        assert total_loss > 0
        avg_loss = total_loss / num_batches
        assert avg_loss < 100  # 损失应该在合理范围
    
    @pytest.mark.slow
    def test_training_with_validation(self, mock_dataset_dir, device):
        """测试带验证的训练"""
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        from src.train import FocalLoss
        
        model, _ = build_model('resnet18', num_classes=2)
        model = model.to(device)
        
        loaders, class_to_idx = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        loss_fn = FocalLoss(gamma=1.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练
        model.train()
        for i, (images, targets) in enumerate(loaders['train']):
            if i >= 2:
                break
            
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in loaders['val']:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # 验证预测有效
        assert len(all_preds) > 0
        assert len(all_preds) == len(all_targets)
        
        # 计算准确率
        accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
        assert 0 <= accuracy <= 1
    
    def test_checkpoint_save_and_load(self, mock_dataset_dir, tmp_path):
        """测试checkpoint保存和加载"""
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        from src.train import save_checkpoint
        
        # 创建并训练模型
        model, _ = build_model('resnet18', num_classes=2)
        
        loaders, class_to_idx = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 保存checkpoint
        checkpoint_path = tmp_path / 'test_checkpoint.pt'
        state = {
            'model': model.state_dict(),
            'classes': class_to_idx,
            'config': {'model': 'resnet18', 'img_size': 224}
        }
        save_checkpoint(state, checkpoint_path)
        
        # 加载checkpoint
        loaded_state = torch.load(checkpoint_path, weights_only=False)
        
        # 创建新模型并加载权重
        new_model, _ = build_model('resnet18', num_classes=2)
        new_model.load_state_dict(loaded_state['model'])
        
        # 验证模型可以推理
        new_model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = new_model(dummy_input)
        
        assert output.shape == (1, 2)
    
    def test_inference_pipeline(self, mock_checkpoint, mock_dataset_dir):
        """测试完整的推理流程"""
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        
        # 加载checkpoint
        checkpoint = torch.load(mock_checkpoint, weights_only=False)
        
        # 重建模型
        model, _ = build_model('resnet18', num_classes=2)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # 加载测试数据
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 推理
        predictions = []
        with torch.no_grad():
            for images, _ in loaders['test']:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                predictions.extend(preds.numpy())
        
        # 验证推理结果
        assert len(predictions) > 0
        assert all(p in [0, 1] for p in predictions)


@pytest.mark.integration
class TestMetricsWorkflow:
    """测试指标计算工作流"""
    
    def test_full_metrics_pipeline(self, mock_dataset_dir, device):
        """测试完整的指标计算流程"""
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        from src.utils.metrics import compute_metrics
        
        # 构建模型
        model, _ = build_model('resnet18', num_classes=2)
        model = model.to(device)
        model.eval()
        
        # 加载数据
        loaders, class_to_idx = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 收集预测
        y_true, y_pred, y_probs = [], [], []
        
        with torch.no_grad():
            for images, targets in loaders['val']:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                y_true.extend(targets.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs.append(probs.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.concatenate(y_probs, axis=0)
        
        # 计算指标
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        metrics, cm = compute_metrics(y_true, y_pred, idx_to_class, y_probs)
        
        # 验证指标
        assert 'overall' in metrics
        assert 'per_class' in metrics
        assert 'macro_f1' in metrics
        assert 'roc_auc' in metrics
        assert cm.shape == (2, 2)
    
    def test_calibration_workflow(self, mock_dataset_dir, device):
        """测试校准工作流"""
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        from src.utils.calibration import compute_calibration_metrics, TemperatureScaling
        
        model, _ = build_model('resnet18', num_classes=2)
        model = model.to(device)
        model.eval()
        
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 收集验证集预测（用于拟合温度）
        val_logits, val_labels = [], []
        with torch.no_grad():
            for images, targets in loaders['val']:
                images = images.to(device)
                outputs = model(images)
                val_logits.append(outputs.cpu())
                val_labels.append(targets)
        
        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        # 拟合温度缩放
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(val_logits, val_labels, max_iter=10)
        
        # 应用温度缩放并计算校准指标
        scaled_logits = temp_scaler(val_logits)
        probs = torch.softmax(scaled_logits, dim=1).numpy()
        
        calibration_metrics = compute_calibration_metrics(
            val_labels.numpy(),
            probs,
            n_bins=5
        )
        
        # 验证校准指标
        assert 'ece' in calibration_metrics
        assert 'mce' in calibration_metrics
        assert 'brier_score' in calibration_metrics


@pytest.mark.integration
class TestDataPipeline:
    """测试数据处理流程"""
    
    def test_data_augmentation_consistency(self, mock_dataset_dir):
        """测试数据增强的一致性"""
        from src.data.datamodule import build_dataloaders
        
        # 测试不同增强级别
        for augment_level in ['light', 'medium', 'heavy']:
            loaders, class_to_idx = build_dataloaders(
                str(mock_dataset_dir),
                img_size=224,
                batch_size=2,
                num_workers=0,
                use_weighted_sampler=False,
                use_albumentations=False,
                augment_level=augment_level
            )
            
            # 验证数据可以加载
            train_batch = next(iter(loaders['train']))
            val_batch = next(iter(loaders['val']))
            
            assert train_batch[0].shape == (2, 3, 224, 224)
            assert val_batch[0].shape == (2, 3, 224, 224)
    
    def test_sampler_balancing(self, mock_dataset_dir):
        """测试采样器的平衡效果"""
        from src.data.datamodule import build_dataloaders
        import collections
        
        loaders, class_to_idx = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=2,
            num_workers=0,
            use_weighted_sampler=True,
            use_albumentations=False
        )
        
        # 统计采样的类别分布
        label_counts = collections.Counter()
        
        for i, (_, labels) in enumerate(loaders['train']):
            label_counts.update(labels.numpy().tolist())
            
            # 只采样几个batch
            if i >= 10:
                break
        
        # 验证两个类别都被采样到
        assert len(label_counts) == 2
        assert all(count > 0 for count in label_counts.values())


@pytest.mark.integration
class TestGradCAMWorkflow:
    """测试GradCAM工作流"""
    
    def test_gradcam_generation(self, mock_dataset_dir, device):
        """测试GradCAM生成流程"""
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        from src.utils.gradcam import GradCAM
        
        model, _ = build_model('resnet18', num_classes=2)
        model = model.to(device)
        model.eval()
        
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=224,
            batch_size=1,
            num_workers=0,
            use_weighted_sampler=False,
            use_albumentations=False
        )
        
        # 创建GradCAM
        gradcam = GradCAM(model, 'layer4')
        
        # 获取一张图像
        images, _ = next(iter(loaders['val']))
        images = images.to(device)
        
        # 生成CAM
        cam = gradcam(images, target_class=1)
        
        # 验证CAM
        assert cam.shape == (224, 224)
        assert (cam >= 0).all() and (cam <= 1).all()


@pytest.mark.integration  
@pytest.mark.slow
class TestConfigValidation:
    """测试配置验证工作流"""
    
    def test_config_to_training(self, sample_config, mock_dataset_dir, tmp_path):
        """测试从配置到训练的完整流程"""
        from src.utils.config_validator import ConfigValidator
        from src.models.factory import build_model
        from src.data.datamodule import build_dataloaders
        from src.train import FocalLoss, set_seed
        
        # 验证配置
        ConfigValidator.validate(sample_config)
        
        # 使用配置进行训练设置
        set_seed(sample_config['seed'])
        
        model, _ = build_model(
            sample_config['model'],
            num_classes=2
        )
        
        loaders, _ = build_dataloaders(
            str(mock_dataset_dir),
            img_size=sample_config['img_size'],
            batch_size=sample_config['batch_size'],
            num_workers=sample_config['num_workers'],
            use_weighted_sampler=(sample_config['sampler'] == 'weighted_random'),
            use_albumentations=sample_config['use_albumentations'],
            augment_level=sample_config['augment_level']
        )
        
        loss_fn = FocalLoss(
            gamma=sample_config['focal']['gamma']
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=sample_config['lr'],
            weight_decay=sample_config['weight_decay']
        )
        
        # 验证所有组件都正常工作
        model.train()
        images, targets = next(iter(loaders['train']))
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 验证训练步骤成功
        assert loss.item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])

