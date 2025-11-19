# 自动化完整训练脚本（优化版v2.0）
# 
# 特点：
# - 支持断点续训
# - 改进的错误处理
# - 实时进度显示
# - 自动保存中间结果
# - 训练完成后自动分析

param(
    [switch]$HighPriorityOnly = $false,   # 仅训练高优先级实验
    [switch]$SkipValidation = $false,     # 跳过环境验证
    [int]$StartFrom = 1,                  # 从第N个实验开始
    [switch]$ContinueOnError = $false     # 遇到错误继续执行
)

# 颜色常量
$Colors = @{
    Title = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Gray"
}

function Write-ColorHost {
    param(
        [string]$Message, 
        [string]$Color = "White", 
        [switch]$NoNewline
    )
    if ($NoNewline) {
        Write-Host $Message -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Message -ForegroundColor $Color
    }
}

function Write-Banner {
    param($Text)
    $line = "=" * 70
    Write-ColorHost "`n$line" $Colors.Title
    Write-ColorHost "  $Text" $Colors.Title
    Write-ColorHost "$line`n" $Colors.Title
}

Write-Banner "肺炎检测项目 - 自动化训练脚本 v2.0"

# 设置项目根目录
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $projectRoot
Write-ColorHost "[INFO] Project root: $projectRoot" $Colors.Info

# 创建日志目录
$logsDir = Join-Path $projectRoot "logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

# 批量训练日志
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$batchLogFile = Join-Path $logsDir "batch_training_$timestamp.txt"
$summaryCsvFile = Join-Path $logsDir "batch_summary_$timestamp.csv"
$progressFile = Join-Path $logsDir "training_progress.json"

function Write-Log {
    param($Message, $Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-ColorHost $logMessage $Color
    Add-Content -Path $batchLogFile -Value $logMessage
}

function Save-Progress {
    param($CompletedExperiments)
    $progressData = @{
        timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        completed = $CompletedExperiments
    }
    $progressData | ConvertTo-Json | Set-Content $progressFile
}

function Load-Progress {
    if (Test-Path $progressFile) {
        try {
            $progress = Get-Content $progressFile | ConvertFrom-Json
            return $progress.completed
        } catch {
            return @()
        }
    }
    return @()
}

function Run-Training {
    param(
        [int]$Index,
        [string]$Name,
        [string]$Command,
        [string]$ExpectedRecall
    )
    
    Write-Log "[$Index] 开始训练: $Name" $Colors.Warning
    Write-ColorHost "  命令: $Command" $Colors.Info
    Write-ColorHost "  预期 Macro Recall: $ExpectedRecall" $Colors.Info
    Write-ColorHost "  (训练中...请等待)" $Colors.Info
    
    $startTime = Get-Date
    $result = [PSCustomObject]@{
        Index = $Index
        Name = $Name
        Status = "Running"
        Duration = 0
        StartTime = $startTime
        EndTime = $null
        Error = ""
        ExpectedRecall = $ExpectedRecall
    }
    
    try {
        # 确保PYTHONPATH设置正确
        $env:PYTHONPATH = $projectRoot
        
        # 执行训练命令（实时显示输出）
        # 使用 & 操作符直接调用，而不是 Invoke-Expression，以保持实时输出
        $commandParts = $Command -split ' ', 2
        if ($commandParts.Count -eq 2) {
            & $commandParts[0] $commandParts[1].Split(' ')
        } else {
            Invoke-Expression $Command
        }
        $exitCode = $LASTEXITCODE
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        $result.EndTime = $endTime
        $result.Duration = [math]::Round($duration.TotalMinutes, 2)
        
        if ($exitCode -eq 0) {
            Write-Log "[$Index] ✅ 完成: $Name (耗时: $($duration.ToString('hh\:mm\:ss')))" $Colors.Success
            $result.Status = "Success"
        } else {
            Write-Log "[$Index] ❌ 失败: $Name (退出码: $exitCode)" $Colors.Error
            $result.Status = "Failed"
            $result.Error = "Exit code: $exitCode"
            
            if (-not $ContinueOnError) {
                throw "Training failed with exit code: $exitCode"
            }
        }
    } catch {
        $endTime = Get-Date
        $duration = $endTime - $startTime
        $result.EndTime = $endTime
        $result.Duration = [math]::Round($duration.TotalMinutes, 2)
        $result.Status = "Error"
        $result.Error = $_.Exception.Message
        
        Write-Log "[$Index] ❌ 异常: $Name - $($_.Exception.Message)" $Colors.Error
        
        if (-not $ContinueOnError) {
            throw
        }
    }
    
    return $result
}

# ============================================================================
# 阶段 1: 环境验证
# ============================================================================
if (-not $SkipValidation) {
    Write-Banner "阶段 1: 环境验证"
    
    Write-Log "检查Python环境..." $Colors.Warning
    $pythonVersion = python --version 2>&1
    Write-Log "  Python版本: $pythonVersion" $Colors.Info
    
    Write-Log "检查PyTorch和CUDA..." $Colors.Warning
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    
    Write-Log "验证项目环境..." $Colors.Warning
    python scripts/verify_environment.py
    
    Write-Log "验证数据集完整性..." $Colors.Warning
    python scripts/verify_dataset_integrity.py
    
    Write-Log "运行测试套件..." $Colors.Warning
    pytest tests/ -q --disable-warnings --no-cov
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorHost "`n[ERROR] 测试失败！请先修复测试问题。" $Colors.Error
        exit 1
    }
    
    Write-Log "✅ 环境验证完成！" $Colors.Success
    Write-Host ""
}

# ============================================================================
# 定义所有实验
# ============================================================================
$allExperiments = @(
    # 高优先级实验（5个）
    @{Index=1; Name="baseline_resnet18"; Config="src/configs/baseline_resnet18.yaml"; Expected="96.62%"; Priority="High"},
    @{Index=2; Name="baseline_efficientnet"; Config="src/configs/baseline_efficientnet.yaml"; Expected="97.93%"; Priority="High"},
    @{Index=3; Name="model_efficientnet_b2"; Config="src/configs/model_efficientnet_b2.yaml"; Expected="98.26%"; Priority="High"},
    @{Index=4; Name="lr_0.0005"; Config="src/configs/lr_0.0005.yaml"; Expected="98.26%"; Priority="High"},
    @{Index=5; Name="final_model"; Config="src/configs/final_model.yaml"; Expected=">98%"; Priority="High"},
    
    # 中优先级实验（5个）
    @{Index=6; Name="model_efficientnet_b0"; Config="src/configs/model_efficientnet_b0.yaml"; Expected="97.41%"; Priority="Medium"},
    @{Index=7; Name="model_resnet18"; Config="src/configs/model_resnet18.yaml"; Expected="97.63%"; Priority="Medium"},
    @{Index=8; Name="model_resnet50"; Config="src/configs/model_resnet50.yaml"; Expected="97.53%"; Priority="Medium"},
    @{Index=9; Name="aug_aggressive"; Config="src/configs/aug_aggressive.yaml"; Expected="98.21%"; Priority="Medium"},
    @{Index=10; Name="aug_medium"; Config="src/configs/aug_medium.yaml"; Expected="98.14%"; Priority="Medium"},
    
    # 低优先级实验（5个）
    @{Index=11; Name="model_densenet121"; Config="src/configs/model_densenet121.yaml"; Expected="97.60%"; Priority="Low"},
    @{Index=12; Name="lr_0.0001"; Config="src/configs/lr_0.0001.yaml"; Expected="97.35%"; Priority="Low"},
    @{Index=13; Name="lr_0.001"; Config="src/configs/lr_0.001.yaml"; Expected="97.96%"; Priority="Low"},
    @{Index=14; Name="aug_light"; Config="src/configs/aug_light.yaml"; Expected="98.21%"; Priority="Low"},
    @{Index=15; Name="full_resnet18"; Config="src/configs/full_resnet18.yaml"; Expected="97.55%"; Priority="Low"}
)

# 筛选实验
if ($HighPriorityOnly) {
    $experiments = $allExperiments | Where-Object { $_.Priority -eq "High" }
    Write-ColorHost "`n[MODE] 仅高优先级实验 ($($experiments.Count)个)" $Colors.Warning
} else {
    $experiments = $allExperiments
    Write-ColorHost "`n[MODE] 所有实验 ($($experiments.Count)个)" $Colors.Warning
}

# 从指定索引开始
$experiments = $experiments | Where-Object { $_.Index -ge $StartFrom }
Write-ColorHost "[MODE] 从实验 #$StartFrom 开始`n" $Colors.Warning

# 加载已完成的实验（断点续训）
$completedExperiments = Load-Progress
if ($completedExperiments.Count -gt 0) {
    Write-ColorHost "[RESUME] 检测到已完成的实验: $($completedExperiments -join ', ')" $Colors.Info
    $experiments = $experiments | Where-Object { $_.Name -notin $completedExperiments }
    Write-ColorHost "[RESUME] 剩余 $($experiments.Count)个实验需要训练`n" $Colors.Info
}

# ============================================================================
# 阶段 2: 批量训练
# ============================================================================
Write-Banner "阶段 2: 批量训练"

$results = [System.Collections.ArrayList]::new()
$totalStartTime = Get-Date
$completedNames = @()

Write-ColorHost "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" $Colors.Info

foreach ($exp in $experiments) {
    $current = $experiments.IndexOf($exp) + 1
    $total = $experiments.Count
    
    Write-Host "`n[$current/$total] " -ForegroundColor Cyan -NoNewline
    Write-Host "实验 #$($exp.Index): $($exp.Name)" -ForegroundColor Cyan
    Write-Host ("━" * 70) -ForegroundColor Cyan
    
    # 构建训练命令
    $command = "python src/train.py --config $($exp.Config)"
    
    # 运行训练
    $result = Run-Training -Index $exp.Index -Name $exp.Name -Command $command -ExpectedRecall $exp.Expected
    [void]$results.Add($result)
    
    # 保存进度
    if ($result.Status -eq "Success") {
        $completedNames += $exp.Name
        Save-Progress -CompletedExperiments $completedNames
    }
    
    # 保存中间结果
    $results | Export-Csv -Path $summaryCsvFile -NoTypeInformation -Encoding UTF8
    
    # 显示进度
    $successCount = @($results | Where-Object { $_.Status -eq "Success" }).Count
    $failedCount = @($results | Where-Object { $_.Status -ne "Success" }).Count
    $progress = [math]::Round(($current / $total) * 100, 1)
    
    Write-ColorHost "`n  进度: $progress% | 成功: $successCount | 失败: $failedCount" $Colors.Info
    $elapsed = (Get-Date) - $totalStartTime
    Write-ColorHost "  已用时间: $($elapsed.ToString('hh\:mm\:ss'))" $Colors.Info
    
    # 短暂休息
    if ($current -lt $total) {
        Start-Sleep -Seconds 3
    }
}

$totalEndTime = Get-Date
$totalDuration = $totalEndTime - $totalStartTime

# ============================================================================
# 阶段 3: 结果汇总
# ============================================================================
Write-Banner "阶段 3: 训练汇总"

$successCount = @($results | Where-Object { $_.Status -eq "Success" }).Count
$failedCount = @($results | Where-Object { $_.Status -ne "Success" }).Count

Write-Log "总耗时: $($totalDuration.ToString('hh\:mm\:ss'))" $Colors.Info
Write-Log "总实验数: $($results.Count)" $Colors.Info
Write-Log "成功: $successCount" $Colors.Success
Write-Log "失败: $failedCount" $(if ($failedCount -gt 0) { $Colors.Error } else { $Colors.Success })

# 显示详细结果表
Write-Host "`n实验结果详情：" -ForegroundColor $Colors.Title
$results | Format-Table -Property Index, Name, Status, @{
    Label="Duration(min)"; 
    Expression={$_.Duration}; 
    FormatString="F2"
}, ExpectedRecall -AutoSize

# 失败实验详情
if ($failedCount -gt 0) {
    Write-Host "`n失败的实验：" -ForegroundColor $Colors.Error
    $results | Where-Object { $_.Status -ne "Success" } | ForEach-Object {
        Write-ColorHost "  ❌ [$($_.Index)] $($_.Name): $($_.Error)" $Colors.Error
    }
}

# 保存最终汇总
$results | Export-Csv -Path $summaryCsvFile -NoTypeInformation -Encoding UTF8
Write-Log "`n📊 详细结果已保存: $summaryCsvFile" $Colors.Success

# ============================================================================
# 阶段 4: 结果分析（如果有成功的实验）
# ============================================================================
if ($successCount -gt 0) {
    Write-Banner "阶段 4: 结果分析"
    
    try {
        Write-Log "分析所有实验结果..." $Colors.Warning
        python scripts/analyze_all_experiments.py
        
        # Note: analyze_all_experiments.py 已经生成了所有需要的可视化图表
        # plot_metrics.py 用于单个实验分析，这里不需要调用
        
        Write-Log "✅ 分析完成！" $Colors.Success
    } catch {
        Write-Log "⚠️ 分析过程出现错误: $_" $Colors.Warning
    }
} else {
    Write-ColorHost "`n⚠️ 没有成功完成的实验，跳过分析阶段" $Colors.Warning
}

# ============================================================================
# 最终总结
# ============================================================================
Write-Banner "训练流程完成"

Write-ColorHost "📊 统计信息：" $Colors.Title
Write-ColorHost "  - 总实验数：$($results.Count)" $Colors.Info
Write-ColorHost "  - 成功：$successCount" $Colors.Success
Write-ColorHost "  - 失败：$failedCount" $(if ($failedCount -gt 0) { $Colors.Error } else { $Colors.Success })
Write-ColorHost "  - 总耗时：$($totalDuration.ToString('hh\:mm\:ss'))" $Colors.Info
Write-ColorHost "  - 平均耗时：$([math]::Round($totalDuration.TotalMinutes / $results.Count, 1))分钟/实验" $Colors.Info

Write-ColorHost "`n📁 输出文件：" $Colors.Title
Write-ColorHost "  - 训练日志：$batchLogFile" $Colors.Info
Write-ColorHost "  - 汇总CSV：$summaryCsvFile" $Colors.Info
Write-ColorHost "  - 进度文件：$progressFile" $Colors.Info

if ($failedCount -eq 0) {
    Write-ColorHost "`n🎉 所有实验训练成功！" $Colors.Success
    
    # 清理进度文件
    if (Test-Path $progressFile) {
        Remove-Item $progressFile
    }
    
    # 播放完成提示音
    try {
        [Console]::Beep(800, 200)
        [Console]::Beep(1000, 200)
        [Console]::Beep(1200, 400)
    } catch {
        # 忽略蜂鸣错误
    }
} else {
    Write-ColorHost "`n⚠️ 部分实验失败，请查看日志文件" $Colors.Warning
    Write-ColorHost "  可以使用 -StartFrom 参数从失败处继续" $Colors.Info
}

Write-ColorHost "`n🚀 后续步骤：" $Colors.Title
Write-ColorHost "  1. 查看训练结果：Import-Csv $summaryCsvFile | Format-Table" $Colors.Info
Write-ColorHost "  2. 查看分析报告：code reports/comprehensive/EXPERIMENT_SUMMARY.md" $Colors.Info
Write-ColorHost "  3. 评估最佳模型：python src/eval.py --ckpt runs/model_efficientnet_b2/best_model.pt --split test" $Colors.Info
Write-ColorHost "  4. 启动演示应用：streamlit run src/app/streamlit_app.py" $Colors.Info

Write-Host "`n" + "="*70 + "`n"

<#
.SYNOPSIS
自动化批量训练脚本

.DESCRIPTION
按顺序训练所有配置的实验，支持断点续训和错误处理

.PARAMETER HighPriorityOnly
仅训练高优先级实验（5个），约4-6小时

.PARAMETER SkipValidation
跳过环境和数据验证步骤

.PARAMETER StartFrom
从第N个实验开始训练（用于中断后继续）

.PARAMETER ContinueOnError
遇到错误时继续执行后续实验，而不是停止

.EXAMPLE
.\scripts\automated_full_training.ps1
训练所有实验

.EXAMPLE
.\scripts\automated_full_training.ps1 -HighPriorityOnly
仅训练高优先级实验

.EXAMPLE
.\scripts\automated_full_training.ps1 -StartFrom 6 -ContinueOnError
从第6个实验开始，遇到错误继续执行

.NOTES
- 支持断点续训：如果中断，再次运行会自动跳过已完成的实验
- 实时保存进度到 logs/training_progress.json
- 详细日志保存到 logs/batch_training_*.txt
- 汇总结果保存到 logs/batch_summary_*.csv
#>
