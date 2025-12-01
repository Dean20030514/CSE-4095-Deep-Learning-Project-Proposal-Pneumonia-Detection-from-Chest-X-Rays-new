# 肺炎检测深度学习项目 - 完整操作指南

> **CSE-4095 深度学习课程项目**  
> ⚠️ **仅供教育研究使用** - 不可用于临床诊断

---

## 📋 目录

1. [项目概述](#一项目概述)
2. [环境准备](#二环境准备)
3. [数据准备](#三数据准备)
4. [模型训练](#四模型训练)
5. [模型评估](#五模型评估)
6. [深度分析](#六深度分析)
7. [演示应用](#七演示应用)
8. [预训练模型备份](#八预训练模型备份)
9. [高级功能](#九高级功能)
10. [统一管理脚本](#十统一管理脚本)
11. [配置参数详解](#十一配置参数详解)
12. [项目结构](#十二项目结构)
13. [常见问题](#十三常见问题)

---

## 一、项目概述

### 1.1 项目目标

本项目构建一个基于深度学习的**胸部X光片肺炎检测系统**，采用二分类方法：
- **NORMAL** - 正常
- **PNEUMONIA** - 肺炎

**核心指标**：最大化肺炎召回率（Pneumonia Recall），减少漏诊

### 1.2 最佳模型性能

| 排名 | 实验 | 宏观召回率 | 准确率 | 肺炎召回率 | GPU训练时间 |
|:---:|------|:----------:|:------:|:----------:|:-----------:|
| 🥇 | aug_aggressive | **98.80%** | 98.81% | 98.82% | ~204分钟 |
| 🥈 | model_densenet121 | 98.45% | 98.30% | 98.11% | ~52分钟 |
| 🥉 | aug_light | 98.40% | 97.96% | 97.41% | ~52分钟 |
| 4 | model_efficientnet_b0 | 98.38% | 98.47% | 98.58% | ~108分钟 |
| 5 | lr_0.0001 | 98.00% | 98.47% | **99.06%** ⭐ | ~152分钟 |

### 1.3 关键发现

根据15个完成的实验对比分析：

1. **最佳综合模型**: `aug_aggressive` - 强力数据增强显著提升性能
   - 验证集宏召回率: 98.80%
   - 训练时间较长但效果最佳

2. **最高效率模型**: `model_densenet121` - 仅52分钟训练
   - 参数量最少 (7M)，效率得分最高 (1.893)
   - 适合资源受限场景

3. **最高肺炎敏感性**: `lr_0.0001` - 99.06%肺炎召回率
   - 仅2例假阴性 (213例肺炎中)
   - 最适合医学筛查场景

### 1.4 支持的模型架构

| 模型名称 | 配置写法 | 默认尺寸 | 特点 |
|----------|----------|:--------:|------|
| ResNet-18 | `resnet18` | 224px | 轻量快速 (24分钟) |
| ResNet-50 | `resnet50` | 224px | 更深层 |
| **EfficientNet-B0** ⭐ | `efficientnet_b0` | 224px | **推荐，多实验最佳基座** |
| EfficientNet-B2 | `efficientnet_b2` | 260px | 更高分辨率 |
| **DenseNet-121** ⭐ | `densenet121` | 224px | **高效率，7M参数** |
| MobileNetV3-Small | `mobilenet_v3_small` | 224px | 移动端部署 |
| MobileNetV3-Large | `mobilenet_v3_large` | 224px | 轻量高性能 |

---

## 二、环境准备

### 2.1 系统要求

- **Python**: 3.8+ (推荐 3.13+)
- **PyTorch**: 2.0+ (支持 CUDA 13.0)
- **RAM**: 8GB (推荐 16GB)
- **GPU**: 8GB+ VRAM (可选但强烈推荐)

### 2.2 安装步骤

#### 方式A：Conda（推荐）

```powershell
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate cxr
```

#### 方式B：pip + venv

```powershell
# 创建虚拟环境
python -m venv .venv

# 激活环境 (Windows)
.\.venv\Scripts\Activate.ps1

# 激活环境 (Linux/Mac)
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 开发环境额外依赖
pip install -r requirements-dev.txt
```

### 2.3 验证环境

```powershell
# 验证Python环境和依赖
python scripts/verify_environment.py
```

输出应显示：
- CUDA可用状态
- PyTorch版本
- 所有必要包已安装

---

## 三、数据准备

### 3.1 数据集来源

**来源**: Kaggle Chest X-Ray Images (Pneumonia)  
**大小**: 1.19 GB (完全去重后)

### 3.2 数据集统计

| 数据集 | 正常图像 | 肺炎图像 | 总计 | 占比 |
|--------|:--------:|:--------:|:----:|:----:|
| 训练集 | 1,399 | 3,608 | 5,007 | 85% |
| 验证集 | 164 | 424 | 588 | 10% |
| 测试集 | 83 | 213 | 296 | 5% |
| **总计** | **1,646** | **4,245** | **5,891** | 100% |

### 3.3 数据目录结构

```
data/
├── train/
│   ├── NORMAL/          # 正常图像
│   └── PNEUMONIA/       # 肺炎图像
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3.4 验证数据完整性

```powershell
python scripts/verify_dataset_integrity.py
```

### 3.5 可视化数据增强效果

```powershell
python scripts/visualize_augmentations.py data/train/PNEUMONIA/sample.jpeg
```

---

## 四、模型训练

### 4.1 配置文件一览 (18个)

#### 模型实验配置
| 配置文件 | 用途 |
|----------|------|
| `model_resnet18.yaml` | ResNet-18 基线 |
| `model_resnet50.yaml` | ResNet-50 深层模型 |
| `model_efficientnet_b0.yaml` | EfficientNet-B0 |
| `model_efficientnet_b2.yaml` | EfficientNet-B2 |
| `model_densenet121.yaml` ⭐ | **高效率模型 (52分钟, 98.45%)** |

#### 学习率实验配置
| 配置文件 | 学习率 |
|----------|:------:|
| `lr_0.0001.yaml` ⭐ | **0.0001 (最高肺炎敏感性 99.06%)** |
| `lr_0.0005.yaml` | 0.0005 |
| `lr_0.001.yaml` | 0.001 |

#### 数据增强实验配置
| 配置文件 | 增强级别 |
|----------|----------|
| `aug_light.yaml` | 轻度增强 |
| `aug_medium.yaml` | 中度增强 |
| `aug_aggressive.yaml` ⭐ | **强力增强 (最佳 98.80%)** |

#### 特殊用途配置
| 配置文件 | 用途 |
|----------|------|
| `quick_test_resnet18.yaml` | 快速测试 (3轮, ~10分钟) |
| `demo_quick.yaml` | 演示快速配置 |
| `baseline_resnet18.yaml` | ResNet基线对照 |
| `baseline_efficientnet.yaml` | EfficientNet基线 |
| `full_resnet18.yaml` | 完整训练 |
| `final_model.yaml` | 最终生产模型 (512px) |
| `medical_screening_optimized.yaml` | 医学筛查优化 |

### 4.2 训练命令

#### 快速测试（验证环境，约10分钟）
```powershell
python src/train.py --config src/configs/quick_test_resnet18.yaml
```

#### 训练最佳模型（推荐，约108分钟GPU）
```powershell
# 最佳综合性能
python src/train.py --config src/configs/aug_aggressive.yaml

# 或选择高效率的DenseNet121（52分钟）
python src/train.py --config src/configs/model_densenet121.yaml
```

#### 训练最终生产模型（最高质量）
```powershell
python src/train.py --config src/configs/final_model.yaml
```

### 4.3 命令行参数覆盖

```powershell
python src/train.py --config <配置文件> `
    --epochs 25 `
    --lr 0.0005 `
    --batch_size 16 `
    --augment_level medium `
    --model efficientnet_b2
```

### 4.4 高级训练参数

| 参数 | 说明 |
|------|------|
| `--resume <checkpoint>` | 从检查点恢复训练 |
| `--auto_eval` | 训练后自动评估 |
| `--export_onnx` | 训练后导出ONNX格式 |
| `--export_torchscript` | 训练后导出TorchScript格式 |
| `--validate_config` | 仅验证配置不训练 |
| `--save_best_by <metric>` | 保存最佳模型的指标 |

### 4.5 恢复训练

```powershell
python src/train.py --config <配置文件> --resume runs/xxx/last_model.pt
```

### 4.6 训练后自动导出

```powershell
# 导出为ONNX格式
python src/train.py --config <配置文件> --export_onnx

# 导出为TorchScript格式
python src/train.py --config <配置文件> --export_torchscript
```

---

## 五、模型评估

### 5.1 基本评估

```powershell
# 验证集评估
python src/eval.py --ckpt runs/aug_aggressive/best_model.pt `
    --data_root data --split val

# 测试集评估
python src/eval.py --ckpt runs/aug_aggressive/best_model.pt `
    --data_root data --split test
```

### 5.2 阈值扫描分析

```powershell
python src/eval.py --ckpt <模型路径> --threshold_sweep
```

### 5.3 保存评估报告

```powershell
python src/eval.py --ckpt <模型路径> --split test `
    --report reports/evaluation_report.json
```

### 5.4 评估指标说明

| 指标 | 说明 | 重要性 |
|------|------|:------:|
| **Pneumonia Recall** | 肺炎检出率（敏感性） | ⭐⭐⭐ |
| **Macro Recall** | 各类别召回率平均 | ⭐⭐⭐ |
| Accuracy | 总体准确率 | ⭐⭐ |
| Precision | 精确率 | ⭐⭐ |
| F1 Score | 精确率和召回率调和平均 | ⭐⭐ |
| ROC-AUC | ROC曲线下面积 | ⭐⭐ |
| PR-AUC | PR曲线下面积 | ⭐⭐ |
| MCC | Matthews相关系数 | ⭐ |
| Cohen's Kappa | 一致性系数 | ⭐ |

---

## 六、深度分析

### 6.1 一键完整分析

```powershell
.\scripts\run_full_analysis.ps1 -Split test
```

生成内容：
- 实验对比 + 排名
- 阈值扫描（5种临床模式）
- 校准分析（ECE, Brier分数）
- 错误分析（FP/FN图库 + 失败模式）

### 6.2 单独分析工具

#### 实验对比分析
```powershell
python scripts/analyze_all_experiments.py
```

#### 阈值扫描分析
```powershell
python scripts/threshold_sweep.py --ckpt <模型路径>
```

#### 模型校准分析
```powershell
python scripts/calibration_analysis.py --ckpt <模型路径>
```

#### 错误案例分析
```powershell
python scripts/error_analysis.py --ckpt <模型路径>
```

#### Grad-CAM可视化
```powershell
python scripts/gradcam_evaluation.py --ckpt <模型路径>
```

#### 绘制指标图表
```powershell
python scripts/plot_metrics.py
```

#### 生成项目报告
```powershell
python scripts/generate_project_report.py
```

---

## 七、演示应用

### 7.1 启动Streamlit演示

```powershell
streamlit run src/app/streamlit_app.py
```

访问 http://localhost:8501

### 7.2 演示功能

- ✅ 上传X光片进行预测
- ✅ 显示预测概率和置信度
- ✅ Grad-CAM热力图可视化
- ✅ 调整分类阈值
- ✅ 批量预测支持

---

## 八、预训练模型备份

### 8.1 可用模型

项目在 `model_backups/` 目录下提供了三个经过验证的预训练模型：

| 模型文件 | 用途 | 关键指标 |
|----------|------|----------|
| `best_overall_val98.80_test97.30.pt` | 🏆 **生产部署** | 测试集97.30%准确率 |
| `best_sensitivity_pneumonia99.06.pt` | 🎯 **医学筛查** | 99.06%肺炎召回率 |
| `production_densenet121_98.45.pt` | 💰 **快速部署** | 52分钟训练, 7M参数 |

### 8.2 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| **最高性能** | best_overall_val98.80 | 综合指标最优 |
| **筛查/分诊** | best_sensitivity_pneumonia99.06 | 最大化检出率 |
| **快速部署** | production_densenet121 | 训练快，参数少 |
| **资源受限** | production_densenet121 | 仅7M参数 |

### 8.3 加载模型

```python
import torch
from src.models.factory import build_model

# 加载检查点
ckpt = torch.load('model_backups/best_overall_val98.80_test97.30.pt')

# 构建模型
model_name = ckpt['config']['model']
num_classes = len(ckpt['classes'])
model, _ = build_model(model_name, num_classes)

# 加载权重
model.load_state_dict(ckpt['model'])
model.eval()
```

---

## 九、高级功能

### 9.1 超参数优化

#### Optuna超参数搜索
```powershell
python scripts/optuna_hyperparameter_search.py
```

#### 自动超参数优化
```powershell
python scripts/auto_optimize_hyperparams.py
```

#### 学习率范围测试
```powershell
python scripts/find_optimal_lr.py --config <配置文件>
```

### 9.2 交叉验证

```powershell
python scripts/cross_validation.py --config <配置文件>
```

### 9.3 模型集成评估

支持三种集成策略：
- `average` - 简单平均概率
- `weighted` - 加权平均
- `voting` - 硬投票

```powershell
python scripts/ensemble_evaluation.py --runs_dir runs --top_k 3
```

### 9.4 不确定性估计

使用MC Dropout进行不确定性估计：

```powershell
python scripts/uncertainty_estimation.py --ckpt <模型路径>
```

### 9.5 推理性能基准测试

```powershell
python scripts/benchmark_inference.py --ckpt <模型路径>
```

### 9.6 域转移分析

```powershell
python scripts/domain_shift_analysis.py
```

### 9.7 标签噪声检测

```powershell
python scripts/label_noise_detection.py
```

### 9.8 训练监控

```powershell
python scripts/monitor_training.py --run_dir runs/<实验名>
```

### 9.9 项目仪表板

```powershell
python scripts/project_dashboard.py
```

---

## 十、统一管理脚本

项目使用单一脚本 `project.ps1` 管理所有操作：

### 10.1 快速启动（推荐新手）

```powershell
.\scripts\project.ps1 -Quick
```

执行：环境检查 → 快速训练(3轮) → 评估 → 启动演示

### 10.2 一键完成所有（推荐）

```powershell
.\scripts\project.ps1 -All
```

执行：
1. ✅ 环境验证
2. ✅ 批量训练所有实验
3. ✅ 深度分析
4. ✅ 生成报告
5. ✅ 启动演示

**常用选项：**
```powershell
.\scripts\project.ps1 -All -QuickMode        # 快速模式（仅高优先级）
.\scripts\project.ps1 -All -SkipTraining     # 跳过训练，仅分析
.\scripts\project.ps1 -All -NoDemo           # 不启动演示
.\scripts\project.ps1 -All -ExportModels     # 导出ONNX/TorchScript
```

### 10.3 批量训练

```powershell
.\scripts\project.ps1 -Train                      # 训练所有实验
.\scripts\project.ps1 -Train -HighPriorityOnly    # 仅高优先级
.\scripts\project.ps1 -Train -StartFrom 5         # 从第5个开始
```

### 10.4 模型分析

```powershell
.\scripts\project.ps1 -Analyze                    # 分析最佳模型
.\scripts\project.ps1 -Analyze -Model <路径>      # 分析指定模型
```

### 10.5 启动演示

```powershell
.\scripts\project.ps1 -Demo
```

### 10.6 运行测试

```powershell
.\scripts\project.ps1 -Test                       # 运行测试
.\scripts\project.ps1 -Test -Coverage -Lint       # 含覆盖率和代码检查
```

### 10.7 查看帮助

```powershell
.\scripts\project.ps1 -Help
```

---

## 十一、配置参数详解

### 11.1 基础参数

```yaml
# 模型配置
model: efficientnet_b2    # 模型架构
pretrained: true          # 使用预训练权重
img_size: 384             # 输入图像尺寸

# 训练配置
batch_size: 24            # 批次大小
epochs: 100               # 训练轮数
lr: 0.0005                # 学习率
weight_decay: 0.0001      # 权重衰减
seed: 42                  # 随机种子

# 数据配置
data_root: data           # 数据根目录
num_workers: 12           # 数据加载线程数
```

### 11.2 损失函数配置

| 损失函数 | 配置值 | 使用场景 |
|----------|--------|----------|
| Weighted CE | `weighted_ce` | 基础类别不平衡处理 |
| **Focal Loss** ⭐ | `focal` | **推荐！聚焦难分类样本** |
| Label Smoothing | `label_smoothing` | 减少过度自信 |
| Cross Entropy | `cross_entropy` | 标准分类 |

```yaml
loss: focal               # 损失函数
focal_gamma: 1.5          # Focal Loss聚焦参数
label_smoothing: 0.0      # 标签平滑系数
```

### 11.3 数据增强配置

| 级别 | 包含操作 | 适用场景 |
|------|----------|----------|
| `light` | 水平翻转(0.5) | 数据量大、快速实验 |
| `medium` ⭐ | 翻转+旋转(±10°)+亮度/对比度(0.15) | **推荐默认** |
| `heavy` | medium+平移(0.1)+更强旋转(±15°) | 数据量小 |
| `aggressive` | 等同于heavy | 同上 |

```yaml
augment_level: medium     # 增强级别

# 或自定义增强
augmentation:
  horizontal_flip: 0.5
  rotation_degrees: 10
  brightness: 0.1
  contrast: 0.1
```

### 11.4 调度器配置

| 调度器 | 说明 |
|--------|------|
| `cosine` | 余弦退火（推荐） |
| `step` | 阶梯衰减 |
| `exponential` | 指数衰减 |
| `none` | 不使用调度器 |

```yaml
scheduler: cosine         # 调度器类型
warmup_epochs: 2          # 学习率预热轮数
```

### 11.5 早停配置

```yaml
early_stopping:
  patience: 20            # 无改善容忍轮数
```

### 11.6 性能优化配置

```yaml
# 混合精度
amp: true                 # 启用AMP (float16)
use_bf16: false           # 使用bfloat16 (需Ampere+ GPU)
allow_tf32: true          # TF32加速 (RTX 30/40/50)

# 内存优化
memory_efficient: false   # 内存高效模式
allow_nondeterministic: false  # 非确定性加速

# 保存配置
save_best_only: false     # 仅保存最佳模型
output_dir: runs/exp_name # 输出目录
```

### 11.7 TensorBoard支持

```yaml
tensorboard: true         # 启用TensorBoard日志
```

查看日志：
```powershell
tensorboard --logdir runs/<实验名>/tensorboard
```

### 11.8 采样器配置

```yaml
use_weighted_sampler: true  # 加权随机采样处理类别不平衡
# 或
sampler: weighted_random
```

---

## 十二、项目结构

```
├── data/                    # 数据集目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   └── test/                # 测试集
│
├── src/                     # 核心源代码
│   ├── train.py             # 训练脚本
│   ├── eval.py              # 评估脚本
│   ├── configs/             # YAML配置文件 (18个)
│   ├── models/              # 模型定义
│   │   ├── factory.py       # 模型工厂
│   │   ├── losses.py        # 损失函数
│   │   └── ensemble.py      # 模型集成
│   ├── data/                # 数据加载模块
│   │   └── datamodule.py    # 数据加载器
│   ├── utils/               # 工具函数 (12个)
│   │   ├── calibration.py   # 温度缩放校准
│   │   ├── config_schema.py # Pydantic配置验证
│   │   ├── config_validator.py # 配置验证器
│   │   ├── dataset_hash.py  # 数据集哈希
│   │   ├── device.py        # 设备检测
│   │   ├── export.py        # 模型导出(ONNX/TorchScript)
│   │   ├── gradcam.py       # Grad-CAM实现
│   │   ├── lr_finder.py     # 学习率查找器
│   │   ├── metrics.py       # 评估指标
│   │   ├── model_info.py    # 模型复杂度分析
│   │   └── uncertainty.py   # 不确定性估计
│   └── app/                 # Streamlit应用
│       └── streamlit_app.py
│
├── scripts/                 # 分析脚本 (27个Python + 1个PowerShell)
│   ├── analyze_all_experiments.py
│   ├── auto_optimize_hyperparams.py
│   ├── benchmark_inference.py
│   ├── calibration_analysis.py
│   ├── create_all_training_configs.py
│   ├── create_optimal_dataset.py
│   ├── cross_validation.py
│   ├── demo_presentation.py
│   ├── domain_shift_analysis.py
│   ├── download_sample_data.py
│   ├── ensemble_evaluation.py
│   ├── error_analysis.py
│   ├── find_optimal_lr.py
│   ├── generate_project_report.py
│   ├── gradcam_evaluation.py
│   ├── label_noise_detection.py
│   ├── monitor_training.py
│   ├── optuna_hyperparameter_search.py
│   ├── plot_metrics.py
│   ├── project_dashboard.py
│   ├── threshold_sweep.py
│   ├── uncertainty_estimation.py
│   ├── verify_dataset_integrity.py
│   ├── verify_environment.py
│   ├── visualize_augmentations.py
│   └── project.ps1            # 统一管理脚本（推荐）
│
├── runs/                    # 实验输出 (15个已完成实验)
│   ├── aug_aggressive/      # 🏆 最佳综合性能
│   ├── model_densenet121/   # ⚗️ 最高效率
│   ├── lr_0.0001/           # 🎯 最高肺炎敏感性
│   ├── model_efficientnet_b0/
│   ├── model_efficientnet_b2/
│   ├── model_resnet18/
│   ├── model_resnet50/
│   ├── ... 其他实验
│   └── <experiment_name>/
│       ├── best_model.pt
│       ├── last_model.pt
│       ├── metrics_history.csv
│       └── train.log
│
├── reports/                 # 分析报告
│   ├── AUTO_PROJECT_REPORT.md
│   ├── COMPREHENSIVE_EXPERIMENTAL_ANALYSIS.md
│   ├── COMPREHENSIVE_EXPERIMENTAL_ANALYSIS_EN.md
│   ├── comprehensive/       # 实验对比报告
│   ├── calibration_*/       # 校准分析
│   ├── error_analysis_*/    # 错误分析
│   ├── gradcam_visualizations/
│   ├── plots/               # 图表
│   └── threshold_sweep_*/
│
├── docs/                    # 文档
│   ├── ANALYSIS_GUIDE.md
│   ├── EXECUTIVE_SUMMARY_EN.md
│   ├── FINAL_PROJECT_REPORT.md
│   ├── MODEL_CARD.md
│   ├── PLAYBOOK.md
│   ├── PRESENTATION_SCRIPT.md
│   ├── PRESENTATION_SLIDES_OUTLINE.md
│   ├── QUICK_RESULTS_REFERENCE.md
│   ├── QUICK_RESULTS_REFERENCE_EN.md
│   └── README.md
│
├── tests/                   # 单元测试 (13个)
│   ├── conftest.py
│   ├── test_datamodule.py
│   ├── test_eval.py
│   ├── test_export.py
│   ├── test_gradcam.py
│   ├── test_integration.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_streamlit_app.py
│   ├── test_train.py
│   └── test_utils.py
│
├── requirements.txt         # 生产依赖
├── requirements-dev.txt     # 开发依赖
├── environment.yml          # Conda环境
└── pyproject.toml           # 项目配置
```

---

## 十三、常见问题

### Q1: CUDA内存不足怎么办？

```yaml
# 减小batch_size
batch_size: 8

# 启用内存高效模式
memory_efficient: true

# 减小图像尺寸
img_size: 224
```

### Q2: 训练速度太慢？

```yaml
# 启用混合精度
amp: true

# 增加数据加载线程
num_workers: 12

# 启用TF32 (RTX 30/40/50系列)
allow_tf32: true
```

### Q3: 如何恢复中断的训练？

```powershell
python src/train.py --config <配置文件> --resume runs/xxx/last_model.pt
```

### Q4: 模型过拟合怎么办？

```yaml
# 增加数据增强
augment_level: heavy

# 增加权重衰减
weight_decay: 0.001

# 使用Label Smoothing
label_smoothing: 0.1

# 减少训练轮数
epochs: 50
```

### Q5: 如何部署模型？

```powershell
# 导出为ONNX
python src/train.py --config <配置文件> --export_onnx

# 导出为TorchScript
python src/train.py --config <配置文件> --export_torchscript
```

### Q6: 如何使用TensorBoard？

```powershell
# 启用TensorBoard日志
# 在配置文件中添加: tensorboard: true

# 查看日志
tensorboard --logdir runs/
```

---

## ⚠️ 医学免责声明

> **本项目仅用于教育和研究目的，不可用于临床诊断或治疗决策。**  
> **请始终咨询专业医疗人员。**

### 伦理考虑

1. 假阴性（漏诊肺炎）比假阳性更危险
2. 筛查场景应使用低阈值以提高召回率
3. 部署前需要在本地数据上进行验证
4. 模型可能无法泛化到所有人群

---

## 📝 快速参考卡片

### 最常用命令

```powershell
# 1. 一键完成所有
.\scripts\project.ps1 -All

# 2. 快速启动（10分钟验证）
.\scripts\project.ps1 -Quick

# 3. 仅训练
.\scripts\project.ps1 -Train

# 4. 仅分析
.\scripts\project.ps1 -Analyze

# 5. 启动演示
.\scripts\project.ps1 -Demo
```

### 最佳配置推荐

```yaml
# 最佳综合性能配置 (aug_aggressive)
model: efficientnet_b0
img_size: 384
batch_size: 24
lr: 0.0005
loss: focal
focal_gamma: 1.5
augment_level: aggressive  # 强力数据增强
scheduler: cosine
warmup_epochs: 2
amp: true
```

```yaml
# 高效训练配置 (model_densenet121)
model: densenet121
img_size: 384
batch_size: 24
lr: 0.0005
loss: focal
augment_level: medium
scheduler: cosine
amp: true
# 训练时间仅52分钟，参数量最少(7M)
```

```yaml
# 最高敏感性配置 (lr_0.0001)
model: efficientnet_b0
img_size: 384
batch_size: 24
lr: 0.0001  # 更低的学习率
loss: focal
augment_level: medium
scheduler: cosine
amp: true
# 肺炎召回率达99.06%
```

---

**最后更新**: 2025年11月  
**项目状态**: ✅ 生产就绪  
**完成实验**: 15个  
**最佳验证召回率**: 98.80% (aug_aggressive)  
**最高肺炎敏感性**: 99.06% (lr_0.0001)
