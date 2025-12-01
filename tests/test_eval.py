"""
单元测试：评估相关功能

测试评估脚本中的阈值扫描和指标计算功能
"""
import pytest
import numpy as np
import torch


class TestThresholdSweep:
    """测试阈值扫描功能"""
    
    def test_threshold_sweep_basic(self):
        """测试基本阈值扫描"""
        from src.eval import threshold_sweep
        
        # 创建测试数据
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_probs = np.array([
            [0.1, 0.9],  # True positive (正确预测 PNEUMONIA)
            [0.2, 0.8],  # True positive
            [0.3, 0.7],  # True positive
            [0.9, 0.1],  # True negative (正确预测 NORMAL)
            [0.8, 0.2],  # True negative
            [0.7, 0.3],  # True negative
        ])
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1)
        
        # 验证结果结构
        assert len(results) > 0
        assert all('threshold' in r for r in results)
        assert all('recall' in r for r in results)
        assert all('precision' in r for r in results)
        assert all('f1' in r for r in results)
    
    def test_threshold_sweep_custom_thresholds(self):
        """测试自定义阈值列表"""
        from src.eval import threshold_sweep
        
        y_true = np.array([1, 0, 1, 0])
        y_probs = np.array([
            [0.2, 0.8],
            [0.9, 0.1],
            [0.3, 0.7],
            [0.6, 0.4],
        ])
        
        custom_thresholds = [0.3, 0.5, 0.7]
        results = threshold_sweep(y_true, y_probs, target_class_idx=1, thresholds=custom_thresholds)
        
        assert len(results) == 3
        thresholds_returned = [r['threshold'] for r in results]
        np.testing.assert_array_almost_equal(thresholds_returned, custom_thresholds)
    
    def test_threshold_sweep_perfect_classifier(self):
        """测试完美分类器的阈值扫描"""
        from src.eval import threshold_sweep
        
        y_true = np.array([1, 1, 0, 0])
        y_probs = np.array([
            [0.0, 1.0],  # Perfect confidence for class 1
            [0.0, 1.0],
            [1.0, 0.0],  # Perfect confidence for class 0
            [1.0, 0.0],
        ])
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1, thresholds=[0.5])
        
        assert len(results) == 1
        # 完美分类器应该有 recall=1, precision=1, f1=1
        assert results[0]['recall'] == 1.0
        assert results[0]['precision'] == 1.0
        assert results[0]['f1'] == 1.0
    
    def test_threshold_sweep_all_positive(self):
        """测试全部预测为正类的情况"""
        from src.eval import threshold_sweep
        
        y_true = np.array([1, 1, 0, 0])
        y_probs = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1, thresholds=[0.5])
        
        # recall should be 1 (all positives found)
        # precision should be 0.5 (2 TP out of 4 predicted)
        assert results[0]['recall'] == 1.0
        assert results[0]['precision'] == 0.5
        assert results[0]['tp'] == 2
        assert results[0]['fp'] == 2
    
    def test_threshold_sweep_confusion_matrix_components(self):
        """测试混淆矩阵组件计算"""
        from src.eval import threshold_sweep
        
        y_true = np.array([1, 1, 0, 0, 0])
        y_probs = np.array([
            [0.2, 0.8],  # TP
            [0.6, 0.4],  # FN (prob < 0.5)
            [0.7, 0.3],  # TN
            [0.4, 0.6],  # FP
            [0.9, 0.1],  # TN
        ])
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1, thresholds=[0.5])
        
        assert results[0]['tp'] == 1
        assert results[0]['fn'] == 1
        assert results[0]['fp'] == 1
        assert results[0]['tn'] == 2


class TestEvalUtilities:
    """测试评估工具函数"""
    
    def test_metrics_ranges(self):
        """验证指标值在有效范围内"""
        from src.eval import threshold_sweep
        
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_probs = np.random.rand(100, 2)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # 归一化
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1)
        
        for r in results:
            assert 0 <= r['recall'] <= 1
            assert 0 <= r['precision'] <= 1
            assert 0 <= r['f1'] <= 1
            assert r['tp'] >= 0
            assert r['fp'] >= 0
            assert r['fn'] >= 0
            assert r['tn'] >= 0
    
    def test_f1_calculation(self):
        """验证F1分数计算正确性"""
        from src.eval import threshold_sweep
        
        y_true = np.array([1, 1, 1, 0, 0])
        y_probs = np.array([
            [0.2, 0.8],
            [0.3, 0.7],
            [0.6, 0.4],  # FN
            [0.7, 0.3],  # TN
            [0.4, 0.6],  # FP
        ])
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1, thresholds=[0.5])
        
        # 手动计算期望的 F1
        # TP=2, FP=1, FN=1, TN=1
        # precision = 2/(2+1) = 0.667
        # recall = 2/(2+1) = 0.667
        # f1 = 2 * 0.667 * 0.667 / (0.667 + 0.667) = 0.667
        expected_f1 = 2/3
        
        np.testing.assert_almost_equal(results[0]['f1'], expected_f1, decimal=3)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_positive_class(self):
        """测试没有正类样本的情况"""
        from src.eval import threshold_sweep
        
        y_true = np.array([0, 0, 0, 0])
        y_probs = np.array([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.9, 0.1],
        ])
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1, thresholds=[0.5])
        
        # 没有正类样本时，recall 应该为 0（除以 0 的情况处理）
        assert results[0]['recall'] == 0.0
        assert results[0]['tp'] == 0
        assert results[0]['fn'] == 0
    
    def test_single_sample(self):
        """测试单个样本的情况"""
        from src.eval import threshold_sweep
        
        y_true = np.array([1])
        y_probs = np.array([[0.3, 0.7]])
        
        results = threshold_sweep(y_true, y_probs, target_class_idx=1, thresholds=[0.5])
        
        assert results[0]['tp'] == 1
        assert results[0]['recall'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

