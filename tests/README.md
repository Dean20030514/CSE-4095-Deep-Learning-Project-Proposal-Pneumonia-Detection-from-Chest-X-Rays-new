# Test Suite Documentation

> **31 unit tests** | 100% pass rate | ~75% code coverage

---

## 🧪 Test File Structure

```
tests/
├── conftest.py               # pytest config and shared fixtures
├── test_datamodule.py       # Data loading tests (4 tests)
├── test_models.py           # Model building tests (9 tests)
├── test_metrics.py          # Evaluation metrics tests (5 tests)
├── test_train.py            # Training functionality tests (6 tests)
└── test_utils.py            # Utility function tests (7 tests)
```

---

## 📊 Test Coverage

### Module Coverage

| Module | Test Count | Coverage | Status |
|--------|------------|----------|--------|
| `src/data/datamodule.py` | 4 | ~80% | ✅ |
| `src/models/factory.py` | 9 | ~90% | ✅ |
| `src/utils/metrics.py` | 5 | ~85% | ✅ |
| `src/train.py` | 6 | ~60% | ✅ |
| `src/utils/*` | 7 | ~75% | ✅ |
| **Total** | **31** | **~75%** | ✅ |

---

## 🚀 Running Tests

### Basic Tests

```powershell
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestModelFactory::test_build_model_architectures -v
```

### Coverage Reports

```powershell
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# View report
# Open htmlcov/index.html in browser

# Show missing lines in terminal
pytest tests/ --cov=src --cov-report=term-missing
```

### Using Scripts

```powershell
# Windows (includes code checking)
.\scripts\run_tests.ps1 -Coverage -Lint

# Linux/Mac
bash scripts/run_tests.sh
```

---

## 📝 Test Details

### test_datamodule.py (4 tests)

Tests data loading and preprocessing:

| Test | Coverage |
|------|----------|
| `test_build_dataloaders_basic` | Basic dataloader construction |
| `test_dataloader_output_shape` | Output shape validation |
| `test_robust_image_folder` | Error handling |
| `test_augmentation_levels` | Different augmentation levels |

**Coverage:**
- ✅ Dataloader creation
- ✅ Data augmentation (light/medium/heavy)
- ✅ Corrupt image handling
- ✅ WeightedRandomSampler

---

### test_models.py (9 tests)

Tests model construction and forward pass:

| Test | Coverage |
|------|----------|
| `test_build_model_architectures` | 5 architecture construction (parametrized) |
| `test_model_forward_pass` | Forward propagation |
| `test_model_output_range` | Output range validation |
| `test_invalid_model_name` | Exception handling |
| `test_model_gradient_flow` | Gradient flow |

**Coverage:**
- ✅ ResNet18/50
- ✅ EfficientNet-B0/B2
- ✅ DenseNet121
- ✅ Forward and backward propagation
- ✅ Exception handling

---

### test_metrics.py (5 tests)

Tests evaluation metrics computation:

| Test | Coverage |
|------|----------|
| `test_perfect_predictions` | Perfect predictions |
| `test_worst_predictions` | Worst predictions |
| `test_metrics_with_probabilities` | Metrics with probabilities |
| `test_sensitivity_specificity` | Sensitivity/specificity |
| `test_empty_arrays` | Edge cases |

**Coverage:**
- ✅ Accuracy, precision, recall, F1
- ✅ Confusion matrix
- ✅ ROC-AUC, PR-AUC
- ✅ Sensitivity and specificity
- ✅ Edge case handling

---

### test_train.py (6 tests)

Tests training-related functionality:

| Test | Coverage |
|------|----------|
| `test_focal_loss_*` | FocalLoss (4 tests) |
| `test_set_seed_*` | Random seed (2 tests) |

**Coverage:**
- ✅ FocalLoss initialization
- ✅ FocalLoss forward pass
- ✅ Class weights
- ✅ Gradient computation
- ✅ Reproducible random seed

---

### test_utils.py (7 tests)

Tests utility functions:

| Test | Coverage |
|------|----------|
| `test_compute_calibration_metrics` | Calibration metrics |
| `test_temperature_scaling` | Temperature scaling |
| `test_temperature_scaling_fit` | Temperature fitting |
| `test_gradcam_*` | GradCAM (3 tests) |

**Coverage:**
- ✅ ECE, MCE, Brier score
- ✅ Temperature scaling calibration
- ✅ GradCAM initialization and generation
- ✅ Exception handling

---

## 🎯 Testing Best Practices

### Before Running Tests

```powershell
# 1. Ensure dependencies installed
pip install -r requirements-dev.txt

# 2. Verify pytest available
pytest --version
```

### Adding New Tests

```python
# Add to corresponding test_*.py file
def test_new_feature():
    """Test new feature"""
    # Arrange
    input_data = ...
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result == expected_value
```

### Debugging Tests

```powershell
# Show detailed output
pytest tests/test_models.py -v -s

# Show detailed traceback
pytest tests/ --tb=long

# Run only failed tests
pytest tests/ --lf

# Stop at first failure
pytest tests/ -x
```

---

## 📈 Coverage Goals

### Current Coverage

```
src/data/datamodule.py     ███████████████░░░░░  80%
src/models/factory.py      ████████████████████  90%
src/utils/metrics.py       █████████████████░░░  85%
src/train.py               ████████████░░░░░░░░  60%
src/utils/calibration.py   ███████████████░░░░░  75%
src/utils/gradcam.py       ███████████████░░░░░  75%
────────────────────────────────────────────────
Overall Coverage           ███████████████░░░░░  ~75%
```

### Future Goals

- 🎯 Short-term: Increase to 80%+
- 🎯 Mid-term: Increase to 85%+
- 🎯 Long-term: Increase to 90%+

---

## 🔍 Testing Strategy

### Unit Tests (Current)

- ✅ Test individual functions/classes
- ✅ Fast execution (<5 seconds)
- ✅ Run independently

### Integration Tests (To Add)

- 🔲 End-to-end training pipeline
- 🔲 Data pipeline
- 🔲 Complete evaluation flow

### Performance Tests (To Add)

- 🔲 Training speed benchmarks
- 🔲 Memory usage monitoring
- 🔲 Inference latency tests

---

## 💡 FAQ

### Q: What if tests fail?

```powershell
# 1. View detailed errors
pytest tests/ -v --tb=long

# 2. Run individual test for debugging
pytest tests/test_models.py::test_function_name -v -s

# 3. Check dependency versions
pip list
```

### Q: How to skip slow tests?

```powershell
pytest tests/ -v -m "not slow"
```

### Q: How to run only specific marked tests?

```powershell
pytest tests/ -v -m "unit"
pytest tests/ -v -m "integration"
```

### Q: Coverage too low?

1. Identify uncovered code
2. Add corresponding test cases
3. Re-run coverage check

---

## 🎊 Test Status

**Current Status:** ✅ Excellent

- ✅ All 31 tests passing
- ✅ Covers core functionality
- ✅ Fast execution (<5 seconds)
- ✅ CI/CD ready

**Suitable as:**
- Code quality reference
- Test-driven development template
- PyTorch project example

---

**Test Framework:** pytest 9.0.1  
**Last Updated:** 2025-11-18  
**Maintenance Status:** Actively maintained
