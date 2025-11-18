#!/bin/bash
# 测试运行脚本（Linux/Mac）

set -e  # 遇到错误立即退出

echo "=========================================="
echo "运行测试套件"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查pytest是否安装
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}错误: pytest未安装${NC}"
    echo "请运行: pip install pytest pytest-cov"
    exit 1
fi

# 1. 运行单元测试
echo -e "\n${YELLOW}步骤 1/4: 运行单元测试...${NC}"
pytest tests/ -v -m "not slow" --tb=short

# 2. 运行覆盖率测试
echo -e "\n${YELLOW}步骤 2/4: 生成覆盖率报告...${NC}"
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# 3. 运行代码检查
echo -e "\n${YELLOW}步骤 3/4: 运行代码检查...${NC}"
if command -v flake8 &> /dev/null; then
    flake8 src/ --max-line-length=100 --extend-ignore=E203,W503 || true
else
    echo "flake8未安装，跳过代码检查"
fi

# 4. 运行类型检查
echo -e "\n${YELLOW}步骤 4/4: 运行类型检查...${NC}"
if command -v mypy &> /dev/null; then
    mypy src/ --ignore-missing-imports || true
else
    echo "mypy未安装，跳过类型检查"
fi

# 总结
echo -e "\n=========================================="
echo -e "${GREEN}✅ 测试完成！${NC}"
echo "=========================================="
echo "覆盖率报告已生成: htmlcov/index.html"
echo "可以用浏览器打开查看详细报告"

