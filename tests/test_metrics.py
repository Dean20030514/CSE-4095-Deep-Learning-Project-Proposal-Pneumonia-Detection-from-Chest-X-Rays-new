"""
单元测试：评估指标计算

测试分类指标、混淆矩阵、ROC/PR AUC等
"""
import pytest
import numpy as np
from src.utils.metrics import compute_metrics, compute_sensitivity_specificity


class TestMetrics:
    """测试评估指标计算"""
    
    def test_perfect_predictions(self):
        """测试完美预测的指标"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 验证完美预测
        assert metrics['overall']['accuracy'] == 1.0
        assert metrics['macro_f1'] == 1.0
        assert metrics['overall']['macro_recall'] == 1.0
        assert metrics['overall']['macro_precision'] == 1.0
        
        # 验证混淆矩阵
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # TN
        assert cm[1, 1] == 2  # TP
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN
        
        # 验证二分类的sensitivity和specificity
        assert metrics['overall']['sensitivity'] == 1.0
        assert metrics['overall']['specificity'] == 1.0
    
    def test_worst_predictions(self):
        """测试最差预测的指标"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 验证最差预测
        assert metrics['overall']['accuracy'] == 0.0
        
        # 验证混淆矩阵
        assert cm[0, 0] == 0  # TN
        assert cm[1, 1] == 0  # TP
        assert cm[0, 1] == 2  # FP
        assert cm[1, 0] == 2  # FN
        
        # 验证sensitivity和specificity
        assert metrics['overall']['sensitivity'] == 0.0
        assert metrics['overall']['specificity'] == 0.0
    
    def test_realistic_predictions(self, sample_predictions):
        """测试真实场景的预测结果"""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        labels = sample_predictions['labels']
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 验证指标结构
        assert 'overall' in metrics
        assert 'per_class' in metrics
        assert 'macro_f1' in metrics
        
        # 验证指标范围
        assert 0 <= metrics['overall']['accuracy'] <= 1
        assert 0 <= metrics['macro_f1'] <= 1
        
        # 验证per-class指标
        for class_name in labels.values():
            assert class_name in metrics['per_class']
            class_metrics = metrics['per_class'][class_name]
            assert 'precision' in class_metrics
            assert 'recall' in class_metrics
            assert 'f1-score' in class_metrics
    
    def test_metrics_with_probabilities(self, sample_predictions):
        """测试带概率的指标计算"""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        y_probs = sample_predictions['y_probs']
        labels = sample_predictions['labels']
        
        metrics, cm = compute_metrics(y_true, y_pred, labels, y_probs)
        
        # 验证AUC指标存在
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        
        # 验证AUC范围
        if metrics['roc_auc'] is not None:
            assert 0 <= metrics['roc_auc'] <= 1
        if metrics['pr_auc'] is not None:
            assert 0 <= metrics['pr_auc'] <= 1
    
    def test_additional_metrics(self):
        """测试MCC和Cohen's Kappa"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 验证额外指标存在
        assert 'mcc' in metrics['overall']
        assert 'cohen_kappa' in metrics['overall']
        
        # 验证范围（MCC和Kappa都在[-1, 1]）
        if metrics['overall']['mcc'] is not None:
            assert -1 <= metrics['overall']['mcc'] <= 1
        if metrics['overall']['cohen_kappa'] is not None:
            assert -1 <= metrics['overall']['cohen_kappa'] <= 1
    
    def test_sensitivity_specificity(self):
        """测试灵敏度和特异度计算"""
        # 创建已知的混淆矩阵
        # [[TN, FP],
        #  [FN, TP]]
        cm = np.array([[8, 2],
                       [1, 9]])
        
        sensitivity, specificity = compute_sensitivity_specificity(cm)
        
        # Class 0: TP=8, FN=1, TN=9, FP=2
        # Sensitivity_0 = TP/(TP+FN) = 8/(8+1) = 0.888...
        # Specificity_0 = TN/(TN+FP) = 9/(9+2) = 0.818...
        
        assert len(sensitivity) == 2
        assert len(specificity) == 2
        assert 0 <= sensitivity[0] <= 1
        assert 0 <= specificity[0] <= 1
        
        # 验证计算正确性
        assert np.isclose(sensitivity[0], 8/9)
        assert np.isclose(specificity[0], 9/11)
    
    def test_multiclass_metrics(self):
        """测试多分类指标"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        labels = {0: 'A', 1: 'B', 2: 'C'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 验证混淆矩阵形状
        assert cm.shape == (3, 3)
        
        # 验证指标结构
        assert len(metrics['per_class']) == 3
        
        # 验证sensitivity和specificity数组
        sensitivity, specificity = compute_sensitivity_specificity(cm)
        assert len(sensitivity) == 3
        assert len(specificity) == 3


class TestMetricsEdgeCases:
    """测试边界情况"""
    
    def test_single_class_predictions(self):
        """测试只预测一个类的情况"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # 只预测class 0
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        # 应该能正常计算，不抛出异常
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 验证返回了指标
        assert 'overall' in metrics
        assert 'macro_f1' in metrics
        assert 'per_class' in metrics
        
        # PNEUMONIA类的recall应该是0（没有预测到任何正例）
        assert metrics['per_class']['PNEUMONIA']['recall'] == 0.0
    
    def test_balanced_predictions(self):
        """测试平衡的预测"""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 两个类的准确率应该相同（都是50%）
        assert metrics['per_class']['NORMAL']['recall'] == 0.5
        assert metrics['per_class']['PNEUMONIA']['recall'] == 0.5
    
    def test_zero_division_handling(self):
        """测试零除处理"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        # 不应该抛出除零异常
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 验证返回了指标
        assert isinstance(metrics, dict)
        assert isinstance(cm, np.ndarray)
    
    def test_probabilities_edge_cases(self):
        """测试概率边界情况"""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        
        # 极端概率（接近0和1）
        y_probs = np.array([
            [0.999, 0.001],
            [0.001, 0.999]
        ])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels, y_probs)
        
        # 应该能计算AUC
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics


class TestMetricsConsistency:
    """测试指标一致性"""
    
    def test_confusion_matrix_consistency(self):
        """测试混淆矩阵和指标一致性"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 从混淆矩阵计算准确率
        accuracy_from_cm = (cm[0, 0] + cm[1, 1]) / cm.sum()
        
        # 应该与metrics中的准确率一致
        assert np.isclose(accuracy_from_cm, metrics['overall']['accuracy'])
    
    def test_macro_averages(self):
        """测试宏平均计算正确"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels)
        
        # 完美预测时，宏平均应该等于所有per-class指标
        recalls = [metrics['per_class'][name]['recall'] 
                  for name in labels.values()]
        precisions = [metrics['per_class'][name]['precision'] 
                     for name in labels.values()]
        
        assert np.isclose(metrics['overall']['macro_recall'], np.mean(recalls))
        assert np.isclose(metrics['overall']['macro_precision'], np.mean(precisions))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

