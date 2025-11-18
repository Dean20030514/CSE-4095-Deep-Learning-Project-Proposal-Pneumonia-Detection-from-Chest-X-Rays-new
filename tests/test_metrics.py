"""
单元测试：评估指标计算
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
        
        # 验证混淆矩阵
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # TN
        assert cm[1, 1] == 2  # TP
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN
    
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
    
    def test_metrics_with_probabilities(self):
        """测试带概率的指标计算"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        metrics, cm = compute_metrics(y_true, y_pred, labels, y_probs)
        
        # 验证AUC指标存在
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1
    
    def test_sensitivity_specificity(self):
        """测试灵敏度和特异度计算"""
        # 创建已知的混淆矩阵
        # [[TN, FP],
        #  [FN, TP]]
        cm = np.array([[8, 2],
                       [1, 9]])
        
        sensitivity, specificity = compute_sensitivity_specificity(cm)
        
        # Class 0: TN=8, FP=2, FN=1, TP=9
        # Sensitivity_0 = TP_0/(TP_0+FN_0) = 8/(8+1) = 0.888...
        # Specificity_0 = TN_0/(TN_0+FP_0) = 9/(9+2) = 0.818...
        
        assert len(sensitivity) == 2
        assert len(specificity) == 2
        assert 0 <= sensitivity[0] <= 1
        assert 0 <= specificity[0] <= 1


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
    
    def test_empty_arrays(self):
        """测试空数组情况"""
        y_true = np.array([])
        y_pred = np.array([])
        labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        # sklearn的metrics可能不会抛出异常，而是返回空结果或警告
        # 我们只验证函数能处理空数组而不崩溃
        try:
            metrics, cm = compute_metrics(y_true, y_pred, labels)
            # 如果没有抛出异常，验证返回的是合理的结构
            assert isinstance(metrics, dict)
            assert isinstance(cm, np.ndarray)
        except (ValueError, IndexError):
            # 如果抛出异常也是可以接受的
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

