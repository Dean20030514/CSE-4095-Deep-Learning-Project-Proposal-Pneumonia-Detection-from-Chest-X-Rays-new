# 测试运行脚本（Windows PowerShell）

param(
    [switch]$SkipSlow,  # 跳过慢速测试
    [switch]$Coverage,   # 生成覆盖率报告
    [switch]$Lint,       # 运行代码检查
    [string]$Pattern = ""  # 测试文件模式
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "运行测试套件" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 检查pytest是否安装
try {
    $null = Get-Command pytest -ErrorAction Stop
} catch {
    Write-Host "❌ 错误: pytest未安装" -ForegroundColor Red
    Write-Host "请运行: pip install pytest pytest-cov" -ForegroundColor Yellow
    exit 1
}

# 构建pytest命令
$pytestArgs = @("tests/", "-v", "--tb=short")

if ($SkipSlow) {
    $pytestArgs += '-m', '"not slow"'
}

if ($Pattern) {
    $pytestArgs += "-k", $Pattern
}

# 1. 运行单元测试
Write-Host "`n步骤 1/4: 运行单元测试..." -ForegroundColor Yellow
& pytest @pytestArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 测试失败" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. 运行覆盖率测试
if ($Coverage) {
    Write-Host "`n步骤 2/4: 生成覆盖率报告..." -ForegroundColor Yellow
    & pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 覆盖率报告已生成: htmlcov/index.html" -ForegroundColor Green
    }
}

# 3. 运行代码检查
if ($Lint) {
    Write-Host "`n步骤 3/4: 运行代码检查..." -ForegroundColor Yellow
    
    try {
        $null = Get-Command flake8 -ErrorAction Stop
        & flake8 src/ --max-line-length=100 --extend-ignore=E203,W503
    } catch {
        Write-Host "flake8未安装，跳过代码检查" -ForegroundColor Yellow
    }
}

# 4. 运行类型检查
if ($Lint) {
    Write-Host "`n步骤 4/4: 运行类型检查..." -ForegroundColor Yellow
    
    try {
        $null = Get-Command mypy -ErrorAction Stop
        & mypy src/ --ignore-missing-imports
    } catch {
        Write-Host "mypy未安装，跳过类型检查" -ForegroundColor Yellow
    }
}

# 总结
Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "✅ 测试完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

if ($Coverage) {
    Write-Host "覆盖率报告: htmlcov\index.html" -ForegroundColor Cyan
}

# 使用示例
<#
.SYNOPSIS
运行项目测试套件

.DESCRIPTION
运行单元测试、覆盖率分析、代码检查等

.PARAMETER SkipSlow
跳过标记为slow的测试

.PARAMETER Coverage
生成代码覆盖率报告

.PARAMETER Lint
运行代码质量检查（flake8, mypy）

.PARAMETER Pattern
只运行匹配模式的测试

.EXAMPLE
.\scripts\run_tests.ps1
运行基础测试

.EXAMPLE
.\scripts\run_tests.ps1 -Coverage -Lint
运行完整测试（包含覆盖率和代码检查）

.EXAMPLE
.\scripts\run_tests.ps1 -SkipSlow -Pattern "test_models"
跳过慢速测试，只运行test_models相关测试
#>

