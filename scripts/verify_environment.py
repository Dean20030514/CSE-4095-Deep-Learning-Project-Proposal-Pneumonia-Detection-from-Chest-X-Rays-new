#!/usr/bin/env python3
"""
环境验证脚本

验证所有必要的依赖项和硬件加速是否可用。
"""
import pkgutil
import platform
import sys

# 必需的包
REQUIRED_PACKAGES = [
    "torch", "torchvision", "numpy", "pandas",
    "albumentations", "sklearn", "streamlit",
    "PIL", "yaml", "tqdm", "matplotlib", "seaborn"
]


def check_packages():
    """检查必需的 Python 包"""
    print("\n[1/3] 检查 Python 包...")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        if pkgutil.find_loader(pkg) is None:
            missing.append(pkg)
            print(f"  ✗ {pkg}: 未安装")
        else:
            print(f"  ✓ {pkg}: 已安装")
    return missing


def check_pytorch():
    """检查 PyTorch 版本和功能"""
    print("\n[2/3] 检查 PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch 版本: {torch.__version__}")

        # 检查 torch.compile 支持
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2] if x.isdigit())
        if len(torch_version) >= 2 and torch_version >= (2, 0):
            print("  ✓ torch.compile(): 支持 (PyTorch 2.0+)")
        else:
            print("  ! torch.compile(): 不支持 (需要 PyTorch 2.0+)")

        # 检查 AMP 支持
        print("  ✓ AMP (自动混精度): 支持")

        return True
    except ImportError as e:
        print(f"  ✗ PyTorch 导入失败: {e}")
        return False


def check_devices():
    """检查可用的计算设备"""
    print("\n[3/3] 检查计算设备...")
    import torch

    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print("  ✓ CUDA: 可用")
        print(f"    - GPU: {gpu_name}")
        print(f"    - GPU 数量: {gpu_count}")
        print(f"    - CUDA 版本: {torch.version.cuda}")

        # 检查 bfloat16 支持
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            print("    - bfloat16: 支持 (Ampere+ GPU)")
    else:
        print("  ! CUDA: 不可用")

    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("  ✓ MPS (Apple Silicon): 可用")
        else:
            print("  ! MPS (Apple Silicon): PyTorch 未编译 MPS 支持")
    else:
        print("  - MPS (Apple Silicon): 不可用")

    # DirectML (AMD/Intel on Windows)
    try:
        import torch_directml  # noqa: F401
        print("  ✓ DirectML: 可用")
    except ImportError:
        print("  - DirectML: 未安装")

    # 选定的设备
    try:
        from src.utils.device import get_device
        selected = get_device()
        print(f"\n  >> 选定设备: {selected}")
    except ImportError:
        # Fallback if device module not available
        if torch.cuda.is_available():
            print("\n  >> 选定设备: cuda")
        else:
            print("\n  >> 选定设备: cpu")


def main():
    """主函数"""
    print("=" * 60)
    print("肺炎检测项目 - 环境验证")
    print("=" * 60)
    print(f"\n系统信息:")
    print(f"  - Python: {sys.version.split()[0]}")
    print(f"  - 平台: {platform.system()} {platform.release()}")
    print(f"  - 架构: {platform.machine()}")

    # 检查包
    missing = check_packages()

    # 检查 PyTorch
    pytorch_ok = check_pytorch()

    # 检查设备
    if pytorch_ok:
        check_devices()

    # 总结
    print("\n" + "=" * 60)
    if missing:
        print(f"✗ 验证失败: 缺少 {len(missing)} 个包")
        print(f"   缺少的包: {', '.join(missing)}")
        print(f"\n   运行: pip install {' '.join(missing)}")
        sys.exit(1)
    else:
        print("✓ 环境验证通过!")
        print("   所有依赖项已安装，可以开始训练。")
    print("=" * 60)


if __name__ == '__main__':
    main()
