# ============================================================================
# 肺炎检测项目 - 统一管理脚本
# ============================================================================
#
# 用法：
#   .\scripts\project.ps1 -Quick          # 快速启动（环境检查+快速训练+演示）
#   .\scripts\project.ps1 -Train          # 批量训练所有实验
#   .\scripts\project.ps1 -Analyze        # 分析最佳模型
#   .\scripts\project.ps1 -All            # 一键完成所有（核心流程）
#   .\scripts\project.ps1 -Complete       # 完整流程（包含所有脚本）⭐ NEW
#   .\scripts\project.ps1 -Advanced       # 高级分析脚本 ⭐ NEW
#   .\scripts\project.ps1 -Benchmark      # 性能基准测试 ⭐ NEW
#   .\scripts\project.ps1 -Visualize      # 可视化工具 ⭐ NEW
#   .\scripts\project.ps1 -Demo           # 仅启动演示
#   .\scripts\project.ps1 -Test           # 运行测试
#
# 可组合参数：
#   .\scripts\project.ps1 -Train -HighPriorityOnly   # 仅训练高优先级
#   .\scripts\project.ps1 -All -QuickMode            # 快速完整流程
#   .\scripts\project.ps1 -Analyze -Model <路径>     # 分析指定模型
#   .\scripts\project.ps1 -Complete -SkipTraining    # 完整流程跳过训练
#
# ============================================================================

[CmdletBinding(DefaultParameterSetName='Help')]
param(
    # 主要操作模式（互斥）
    [Parameter(ParameterSetName='Quick')]
    [switch]$Quick,                    # 快速启动

    [Parameter(ParameterSetName='Train')]
    [switch]$Train,                    # 批量训练

    [Parameter(ParameterSetName='Analyze')]
    [switch]$Analyze,                  # 分析模型

    [Parameter(ParameterSetName='All')]
    [switch]$All,                      # 一键完成所有（核心）

    [Parameter(ParameterSetName='Complete')]
    [switch]$Complete,                 # 完整流程（所有脚本）

    [Parameter(ParameterSetName='Advanced')]
    [switch]$Advanced,                 # 高级分析脚本

    [Parameter(ParameterSetName='Benchmark')]
    [switch]$Benchmark,                # 性能基准测试

    [Parameter(ParameterSetName='Visualize')]
    [switch]$Visualize,                # 可视化工具

    [Parameter(ParameterSetName='Demo')]
    [switch]$Demo,                     # 仅启动演示

    [Parameter(ParameterSetName='Test')]
    [switch]$Test,                     # 运行测试

    [Parameter(ParameterSetName='Help')]
    [switch]$Help,                     # 显示帮助

    # 通用选项
    [switch]$HighPriorityOnly,         # 仅高优先级实验
    [switch]$QuickMode,                # 快速模式
    [switch]$SkipValidation,           # 跳过环境验证
    [switch]$SkipTraining,             # 跳过训练
    [switch]$NoDemo,                   # 不启动演示
    [switch]$ExportModels,             # 导出模型
    [switch]$ContinueOnError,          # 遇错继续
    [string]$Model = "",               # 指定模型路径
    [string]$Config = "",              # 指定配置文件
    [int]$StartFrom = 1,               # 从第N个实验开始
    [switch]$Coverage,                 # 测试覆盖率
    [switch]$Lint                      # 代码检查
)

# ============================================================================
# 配置
# ============================================================================

$ErrorActionPreference = if ($ContinueOnError) { "Continue" } else { "Stop" }

$Colors = @{
    Title = "Cyan"; Success = "Green"; Warning = "Yellow"
    Error = "Red"; Info = "Gray"; Highlight = "Magenta"
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $ProjectRoot
Set-Location $ProjectRoot

$RunsDir = Join-Path $ProjectRoot "runs"
$ReportsDir = Join-Path $ProjectRoot "reports"
$LogsDir = Join-Path $ProjectRoot "logs"
$DataDir = Join-Path $ProjectRoot "data"

# 创建必要目录
@($LogsDir, $ReportsDir) | ForEach-Object {
    if (!(Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
}

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogsDir "project_$Timestamp.log"

# ============================================================================
# 工具函数
# ============================================================================

function Write-C { param([string]$Msg, [string]$Color = "White", [switch]$NoNL)
    if ($NoNL) { Write-Host $Msg -ForegroundColor $Color -NoNewline }
    else { Write-Host $Msg -ForegroundColor $Color }
}

function Write-Log { param([string]$Msg, [string]$Color = "White")
    $ts = Get-Date -Format "HH:mm:ss"
    Write-C "[$ts] $Msg" $Color
    Add-Content -Path $LogFile -Value "[$ts] $Msg"
}

function Write-Banner { param([string]$Text)
    $line = "=" * 60
    Write-Host ""; Write-C $line $Colors.Title
    Write-C "  $Text" $Colors.Title; Write-C $line $Colors.Title; Write-Host ""
}

function Write-Step { param([int]$N, [int]$T, [string]$Desc)
    Write-Host ""; Write-C "[$N/$T] $Desc" $Colors.Warning
    Write-C ("-" * 40) $Colors.Info
}

function Test-Success { param([string]$Name)
    if ($LASTEXITCODE -ne 0) {
        Write-Log "X $Name 失败 (退出码: $LASTEXITCODE)" $Colors.Error
        return $false
    }
    Write-Log "OK $Name 完成" $Colors.Success
    return $true
}

function Find-BestModel {
    $best = ""; $bestRecall = 0.0
    Get-ChildItem -Path $RunsDir -Recurse -Filter "best_model.pt" -ErrorAction SilentlyContinue | ForEach-Object {
        $csv = Join-Path $_.DirectoryName "metrics_history.csv"
        if (Test-Path $csv) {
            try {
                $last = Get-Content $csv | Select-Object -Last 1
                if ($last -notmatch "epoch") {
                    $recall = [double]($last -split ",")[3]
                    if ($recall -gt $bestRecall) { $bestRecall = $recall; $best = $_.FullName }
                }
            } catch {}
        }
    }
    if ([string]::IsNullOrEmpty($best)) {
        @("model_efficientnet_b2", "aug_aggressive", "baseline_resnet18") | ForEach-Object {
            $p = Join-Path $RunsDir "$_/best_model.pt"
            if ((Test-Path $p) -and [string]::IsNullOrEmpty($best)) { $best = $p }
        }
    }
    return $best
}

function Find-SampleImage {
    # 找一个示例图像用于可视化
    $trainDir = Join-Path $DataDir "train/PNEUMONIA"
    if (Test-Path $trainDir) {
        $img = Get-ChildItem -Path $trainDir -Filter "*.jpeg" | Select-Object -First 1
        if ($img) { return $img.FullName }
    }
    return $null
}

function Show-Help {
    Write-Banner "肺炎检测项目 - 帮助"
    Write-C "基础用法:" $Colors.Highlight
    Write-C "  .\scripts\project.ps1 -Quick          快速启动（~10分钟）" $Colors.Info
    Write-C "  .\scripts\project.ps1 -Train          批量训练所有实验" $Colors.Info
    Write-C "  .\scripts\project.ps1 -Analyze        分析最佳模型" $Colors.Info
    Write-C "  .\scripts\project.ps1 -All            一键完成所有（核心流程）" $Colors.Info
    Write-C "  .\scripts\project.ps1 -Demo           启动演示应用" $Colors.Info
    Write-C "  .\scripts\project.ps1 -Test           运行测试套件" $Colors.Info
    Write-Host ""
    Write-C "高级用法（NEW）:" $Colors.Highlight
    Write-C "  .\scripts\project.ps1 -Complete       完整流程（运行所有脚本）" $Colors.Info
    Write-C "  .\scripts\project.ps1 -Advanced       高级分析（不确定性/噪声检测等）" $Colors.Info
    Write-C "  .\scripts\project.ps1 -Benchmark      性能基准测试" $Colors.Info
    Write-C "  .\scripts\project.ps1 -Visualize      可视化工具" $Colors.Info
    Write-Host ""
    Write-C "常用组合:" $Colors.Highlight
    Write-C "  -Train -HighPriorityOnly              仅训练高优先级实验" $Colors.Info
    Write-C "  -All -QuickMode                       快速完整流程" $Colors.Info
    Write-C "  -Analyze -Model <路径>                分析指定模型" $Colors.Info
    Write-C "  -Complete -SkipTraining               完整流程跳过训练" $Colors.Info
    Write-C "  -Test -Coverage -Lint                 测试+覆盖率+代码检查" $Colors.Info
    Write-Host ""
    Write-C "脚本覆盖统计:" $Colors.Highlight
    Write-C "  -All:      11个脚本（核心工作流）" $Colors.Info
    Write-C "  -Complete: 26个脚本（全部）" $Colors.Info
    Write-Host ""
}

# ============================================================================
# 核心功能：环境验证
# ============================================================================

function Invoke-Validation {
    Write-Log "检查Python环境..." $Colors.Info
    python --version
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

    Write-Log "验证项目环境..." $Colors.Info
    python scripts/verify_environment.py
    Test-Success "环境验证" | Out-Null

    Write-Log "验证数据集..." $Colors.Info
    python scripts/verify_dataset_integrity.py
    Test-Success "数据集验证" | Out-Null
}

# ============================================================================
# 核心功能：训练
# ============================================================================

function Invoke-Training {
    param([bool]$HighPriority = $false, [int]$Start = 1)

    $experiments = @(
        @{N=1; Name="baseline_resnet18"; Cfg="baseline_resnet18.yaml"; P="High"},
        @{N=2; Name="baseline_efficientnet"; Cfg="baseline_efficientnet.yaml"; P="High"},
        @{N=3; Name="model_efficientnet_b2"; Cfg="model_efficientnet_b2.yaml"; P="High"},
        @{N=4; Name="lr_0.0005"; Cfg="lr_0.0005.yaml"; P="High"},
        @{N=5; Name="final_model"; Cfg="final_model.yaml"; P="High"},
        @{N=6; Name="model_efficientnet_b0"; Cfg="model_efficientnet_b0.yaml"; P="Medium"},
        @{N=7; Name="model_resnet18"; Cfg="model_resnet18.yaml"; P="Medium"},
        @{N=8; Name="model_resnet50"; Cfg="model_resnet50.yaml"; P="Medium"},
        @{N=9; Name="aug_aggressive"; Cfg="aug_aggressive.yaml"; P="Medium"},
        @{N=10; Name="aug_medium"; Cfg="aug_medium.yaml"; P="Medium"},
        @{N=11; Name="model_densenet121"; Cfg="model_densenet121.yaml"; P="Low"},
        @{N=12; Name="lr_0.0001"; Cfg="lr_0.0001.yaml"; P="Low"},
        @{N=13; Name="lr_0.001"; Cfg="lr_0.001.yaml"; P="Low"},
        @{N=14; Name="aug_light"; Cfg="aug_light.yaml"; P="Low"},
        @{N=15; Name="full_resnet18"; Cfg="full_resnet18.yaml"; P="Low"}
    )

    if ($HighPriority) {
        $experiments = $experiments | Where-Object { $_.P -eq "High" }
        Write-Log "模式: 仅高优先级 ($($experiments.Count)个)" $Colors.Warning
    }
    $experiments = $experiments | Where-Object { $_.N -ge $Start }

    $total = $experiments.Count
    $success = 0

    foreach ($exp in $experiments) {
        Write-Log "[$($exp.N)/$total] 训练: $($exp.Name)" $Colors.Warning
        $cfg = "src/configs/$($exp.Cfg)"

        python src/train.py --config $cfg

        if ($LASTEXITCODE -eq 0) {
            $success++
            Write-Log "OK $($exp.Name) 完成" $Colors.Success
        } else {
            Write-Log "X $($exp.Name) 失败" $Colors.Error
            if (-not $ContinueOnError) { throw "训练失败" }
        }
    }

    Write-Log "训练完成: $success/$total 成功" $(if ($success -eq $total) {$Colors.Success} else {$Colors.Warning})
}

# ============================================================================
# 核心功能：分析
# ============================================================================

function Invoke-Analysis {
    param([string]$ModelPath = "")

    if ([string]::IsNullOrEmpty($ModelPath)) { $ModelPath = Find-BestModel }

    if ([string]::IsNullOrEmpty($ModelPath) -or -not (Test-Path $ModelPath)) {
        Write-Log "X 未找到模型！请先训练或指定 -Model 参数" $Colors.Error
        return
    }

    Write-Log "分析模型: $ModelPath" $Colors.Highlight
    $outDir = Join-Path $ReportsDir "analysis_$Timestamp"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    # 1. 实验对比
    Write-Log "执行实验对比分析..." $Colors.Info
    python scripts/analyze_all_experiments.py --runs_dir $RunsDir --output_dir "$outDir/comparison"

    # 2. 阈值扫描
    Write-Log "执行阈值扫描..." $Colors.Info
    python scripts/threshold_sweep.py --ckpt $ModelPath --split val --output_dir "$outDir/threshold"

    # 3. 校准分析
    Write-Log "执行校准分析..." $Colors.Info
    python scripts/calibration_analysis.py --ckpt $ModelPath --split val --fit_temperature --output_dir "$outDir/calibration"

    # 4. 错误分析
    Write-Log "执行错误分析..." $Colors.Info
    python scripts/error_analysis.py --ckpt $ModelPath --split val --output_dir "$outDir/errors" --max_samples 20

    # 5. Grad-CAM
    Write-Log "生成Grad-CAM可视化..." $Colors.Info
    python scripts/gradcam_evaluation.py --ckpt $ModelPath --split val --output_dir "$outDir/gradcam" --num_samples 15

    # 6. 测试集评估
    Write-Log "执行测试集评估..." $Colors.Info
    python src/eval.py --ckpt $ModelPath --data_root data --split test --threshold_sweep --report "$outDir/test_report.json"

    Write-Log "OK 分析完成！结果: $outDir" $Colors.Success
}

# ============================================================================
# 高级分析脚本（NEW）
# ============================================================================

function Invoke-AdvancedAnalysis {
    param([string]$ModelPath = "", [string]$ConfigPath = "")

    if ([string]::IsNullOrEmpty($ModelPath)) { $ModelPath = Find-BestModel }
    if ([string]::IsNullOrEmpty($ConfigPath)) { $ConfigPath = "src/configs/model_efficientnet_b2.yaml" }

    if ([string]::IsNullOrEmpty($ModelPath) -or -not (Test-Path $ModelPath)) {
        Write-Log "X 未找到模型！请先训练或指定 -Model 参数" $Colors.Error
        return
    }

    Write-Log "高级分析模型: $ModelPath" $Colors.Highlight
    $outDir = Join-Path $ReportsDir "advanced_$Timestamp"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $total = 7
    $current = 0

    # 1. 不确定性估计 (MC Dropout)
    $current++; Write-Log "[$current/$total] MC Dropout 不确定性分析..." $Colors.Info
    python scripts/uncertainty_estimation.py --ckpt $ModelPath --split val --output_dir "$outDir/uncertainty" --n_samples 10
    Test-Success "不确定性估计" | Out-Null

    # 2. 域偏移分析
    $current++; Write-Log "[$current/$total] 域偏移分析..." $Colors.Info
    python scripts/domain_shift_analysis.py --ckpt $ModelPath --output_dir "$outDir/domain_shift"
    Test-Success "域偏移分析" | Out-Null

    # 3. 标签噪声检测
    $current++; Write-Log "[$current/$total] 标签噪声检测..." $Colors.Info
    python scripts/label_noise_detection.py --ckpt $ModelPath --output_dir "$outDir/noise_detection"
    Test-Success "标签噪声检测" | Out-Null

    # 4. 模型集成评估
    $current++; Write-Log "[$current/$total] 模型集成评估..." $Colors.Info
    python scripts/ensemble_evaluation.py --runs_dir $RunsDir --top_k 3 --output_dir "$outDir/ensemble"
    Test-Success "集成评估" | Out-Null

    # 5. 学习率范围测试
    $current++; Write-Log "[$current/$total] 学习率范围测试..." $Colors.Info
    python scripts/find_optimal_lr.py --config $ConfigPath --output_dir "$outDir/lr_finder"
    Test-Success "学习率测试" | Out-Null

    # 6. 交叉验证
    $current++; Write-Log "[$current/$total] K折交叉验证..." $Colors.Info
    python scripts/cross_validation.py --config $ConfigPath --k_folds 3 --output_dir "$outDir/cross_val"
    Test-Success "交叉验证" | Out-Null

    # 7. 生成项目报告
    $current++; Write-Log "[$current/$total] 生成综合报告..." $Colors.Info
    python scripts/generate_project_report.py --output_dir "$outDir"
    Test-Success "报告生成" | Out-Null

    Write-Log "OK 高级分析完成！结果: $outDir" $Colors.Success
}

# ============================================================================
# 性能基准测试（NEW）
# ============================================================================

function Invoke-Benchmark {
    param([string]$ModelPath = "")

    if ([string]::IsNullOrEmpty($ModelPath)) { $ModelPath = Find-BestModel }

    if ([string]::IsNullOrEmpty($ModelPath) -or -not (Test-Path $ModelPath)) {
        Write-Log "X 未找到模型！请先训练或指定 -Model 参数" $Colors.Error
        return
    }

    Write-Log "基准测试模型: $ModelPath" $Colors.Highlight
    $outDir = Join-Path $ReportsDir "benchmark_$Timestamp"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $total = 3
    $current = 0

    # 1. 推理性能基准
    $current++; Write-Log "[$current/$total] 推理性能基准测试..." $Colors.Info
    python scripts/benchmark_inference.py --ckpt $ModelPath --output_dir "$outDir/inference" --warmup 10 --iterations 100
    Test-Success "推理基准" | Out-Null

    # 2. Optuna 超参数搜索（快速版，5 trials）
    $current++; Write-Log "[$current/$total] Optuna 超参数优化（快速）..." $Colors.Info
    python scripts/optuna_hyperparameter_search.py --n_trials 5 --output_dir "$outDir/optuna"
    Test-Success "超参数搜索" | Out-Null

    # 3. 自动超参数优化
    $current++; Write-Log "[$current/$total] 自动超参数优化..." $Colors.Info
    python scripts/auto_optimize_hyperparams.py --output_dir "$outDir/auto_optimize"
    Test-Success "自动优化" | Out-Null

    Write-Log "OK 基准测试完成！结果: $outDir" $Colors.Success
}

# ============================================================================
# 可视化工具（NEW）
# ============================================================================

function Invoke-Visualization {
    param([string]$ModelPath = "")

    if ([string]::IsNullOrEmpty($ModelPath)) { $ModelPath = Find-BestModel }

    Write-Log "运行可视化工具..." $Colors.Highlight
    $outDir = Join-Path $ReportsDir "visualization_$Timestamp"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $total = 4
    $current = 0

    # 1. 绘制训练指标
    $current++; Write-Log "[$current/$total] 绘制训练指标图表..." $Colors.Info
    python scripts/plot_metrics.py --runs_dir $RunsDir --output_dir "$outDir/metrics"
    Test-Success "指标图表" | Out-Null

    # 2. 数据增强可视化
    $sampleImg = Find-SampleImage
    if ($sampleImg) {
        $current++; Write-Log "[$current/$total] 数据增强可视化..." $Colors.Info
        python scripts/visualize_augmentations.py $sampleImg --output_dir "$outDir/augmentations"
        Test-Success "增强可视化" | Out-Null
    } else {
        $current++; Write-Log "[$current/$total] 跳过增强可视化（无示例图像）" $Colors.Warning
    }

    # 3. 演示展示
    $current++; Write-Log "[$current/$total] 生成演示展示..." $Colors.Info
    python scripts/demo_presentation.py --output_dir "$outDir/presentation"
    Test-Success "演示展示" | Out-Null

    # 4. 项目仪表板
    $current++; Write-Log "[$current/$total] 生成项目仪表板..." $Colors.Info
    python scripts/project_dashboard.py --runs_dir $RunsDir --output_dir "$outDir/dashboard"
    Test-Success "项目仪表板" | Out-Null

    Write-Log "OK 可视化完成！结果: $outDir" $Colors.Success
}

# ============================================================================
# 核心功能：演示
# ============================================================================

function Invoke-Demo {
    Write-Log "启动Streamlit演示..." $Colors.Info
    Write-C "访问地址: http://localhost:8501" $Colors.Highlight
    Write-C "按 Ctrl+C 停止" $Colors.Info
    streamlit run src/app/streamlit_app.py
}

# ============================================================================
# 核心功能：测试
# ============================================================================

function Invoke-Test {
    param([bool]$Cov = $false, [bool]$DoLint = $false)

    Write-Log "运行测试套件..." $Colors.Info

    $pytestArgs = @("tests/", "-v")
    if ($Cov) { $pytestArgs += "--cov=src", "--cov-report=html" }

    pytest @pytestArgs
    Test-Success "测试" | Out-Null

    if ($DoLint) {
        Write-Log "运行代码检查..." $Colors.Info
        ruff check src/ scripts/ --fix
        Test-Success "代码检查" | Out-Null
    }
}

# ============================================================================
# 完整流程（所有脚本）
# ============================================================================

function Invoke-Complete {
    param([string]$ModelPath = "", [string]$ConfigPath = "")

    Write-Log "完整流程：运行所有26个脚本" $Colors.Highlight

    $totalSteps = 8
    $currentStep = 0

    # 1. 环境验证
    if (-not $SkipValidation) {
        $currentStep++; Write-Step $currentStep $totalSteps "环境验证 (2脚本)"
        Invoke-Validation
    }

    # 2. 生成配置文件
    $currentStep++; Write-Step $currentStep $totalSteps "生成配置文件"
    python scripts/create_all_training_configs.py
    Test-Success "配置生成" | Out-Null

    # 3. 批量训练
    if (-not $SkipTraining) {
        $currentStep++; Write-Step $currentStep $totalSteps "批量训练 (15配置)"
        Invoke-Training -HighPriority ($QuickMode -or $HighPriorityOnly) -Start $StartFrom
    }

    # 4. 核心分析
    $currentStep++; Write-Step $currentStep $totalSteps "核心分析 (6脚本)"
    Invoke-Analysis -ModelPath $ModelPath

    # 5. 高级分析
    $currentStep++; Write-Step $currentStep $totalSteps "高级分析 (7脚本)"
    Invoke-AdvancedAnalysis -ModelPath $ModelPath -ConfigPath $ConfigPath

    # 6. 性能基准
    $currentStep++; Write-Step $currentStep $totalSteps "性能基准 (3脚本)"
    Invoke-Benchmark -ModelPath $ModelPath

    # 7. 可视化
    $currentStep++; Write-Step $currentStep $totalSteps "可视化 (4脚本)"
    Invoke-Visualization -ModelPath $ModelPath

    # 8. 导出模型
    if ($ExportModels) {
        $currentStep++; Write-Step $currentStep $totalSteps "导出模型"
        $best = if ($ModelPath) { $ModelPath } else { Find-BestModel }
        if ($best) {
            $expDir = Join-Path (Split-Path $best) "exported"
            New-Item -ItemType Directory -Path $expDir -Force | Out-Null
            python -c "from src.utils.export import export_model_from_checkpoint; export_model_from_checkpoint('$best', '$expDir/model.onnx', formats=['onnx'])"
            python -c "from src.utils.export import export_model_from_checkpoint; export_model_from_checkpoint('$best', '$expDir/model.pt', formats=['torchscript'])"
        }
    }

    Write-Log "OK 完整流程完成！所有26个脚本已执行" $Colors.Success
}

# ============================================================================
# 主程序
# ============================================================================

$StartTime = Get-Date

# 显示帮助
if ($Help -or ($PSCmdlet.ParameterSetName -eq 'Help')) {
    Show-Help
    exit 0
}

Write-Banner "肺炎检测项目管理器"

# -Quick: 快速启动
if ($Quick) {
    Write-Log "模式: 快速启动" $Colors.Highlight

    if (-not $SkipValidation) { Invoke-Validation }

    Write-Log "快速训练 (3轮)..." $Colors.Warning
    python src/train.py --config src/configs/quick_test_resnet18.yaml
    Test-Success "快速训练" | Out-Null

    $qm = "runs/quick_test_resnet18/best_model.pt"
    if (Test-Path $qm) {
        python src/eval.py --ckpt $qm --data_root data --split val
    }

    if (-not $NoDemo) { Invoke-Demo }
}

# -Train: 批量训练
elseif ($Train) {
    Write-Log "模式: 批量训练" $Colors.Highlight
    if (-not $SkipValidation) { Invoke-Validation }
    Invoke-Training -HighPriority $HighPriorityOnly -Start $StartFrom
}

# -Analyze: 分析
elseif ($Analyze) {
    Write-Log "模式: 模型分析" $Colors.Highlight
    Invoke-Analysis -ModelPath $Model
}

# -All: 一键完成所有（核心流程）
elseif ($All) {
    Write-Log "模式: 一键完成所有（核心流程）" $Colors.Highlight

    # 1. 验证
    if (-not $SkipValidation) {
        Write-Step 1 5 "环境验证"
        Invoke-Validation
    }

    # 2. 训练
    if (-not $SkipTraining) {
        Write-Step 2 5 "批量训练"
        Invoke-Training -HighPriority ($QuickMode -or $HighPriorityOnly) -Start $StartFrom
    }

    # 3. 分析
    Write-Step 3 5 "深度分析"
    Invoke-Analysis -ModelPath $Model

    # 4. 报告
    Write-Step 4 5 "生成报告"
    python scripts/generate_project_report.py

    # 5. 导出
    if ($ExportModels) {
        Write-Step 5 5 "导出模型"
        $best = if ($Model) { $Model } else { Find-BestModel }
        if ($best) {
            $expDir = Join-Path (Split-Path $best) "exported"
            New-Item -ItemType Directory -Path $expDir -Force | Out-Null
            python -c "from src.utils.export import export_model_from_checkpoint; export_model_from_checkpoint('$best', '$expDir/model.onnx', formats=['onnx'])"
            python -c "from src.utils.export import export_model_from_checkpoint; export_model_from_checkpoint('$best', '$expDir/model.pt', formats=['torchscript'])"
        }
    }

    # 6. 演示
    if (-not $NoDemo) {
        Write-Step 5 5 "启动演示"
        Invoke-Demo
    }
}

# -Complete: 完整流程（所有脚本）
elseif ($Complete) {
    Write-Log "模式: 完整流程（运行所有脚本）" $Colors.Highlight
    Invoke-Complete -ModelPath $Model -ConfigPath $Config

    if (-not $NoDemo) {
        Write-Log "启动演示..." $Colors.Info
        Invoke-Demo
    }
}

# -Advanced: 高级分析
elseif ($Advanced) {
    Write-Log "模式: 高级分析" $Colors.Highlight
    Invoke-AdvancedAnalysis -ModelPath $Model -ConfigPath $Config
}

# -Benchmark: 性能基准
elseif ($Benchmark) {
    Write-Log "模式: 性能基准测试" $Colors.Highlight
    Invoke-Benchmark -ModelPath $Model
}

# -Visualize: 可视化
elseif ($Visualize) {
    Write-Log "模式: 可视化工具" $Colors.Highlight
    Invoke-Visualization -ModelPath $Model
}

# -Demo: 仅演示
elseif ($Demo) {
    Invoke-Demo
}

# -Test: 测试
elseif ($Test) {
    Write-Log "模式: 运行测试" $Colors.Highlight
    Invoke-Test -Cov $Coverage -DoLint $Lint
}

# 完成总结
$elapsed = (Get-Date) - $StartTime
Write-Banner "完成！"
Write-C "耗时: $($elapsed.ToString('hh\:mm\:ss'))" $Colors.Info
Write-C "日志: $LogFile" $Colors.Info
