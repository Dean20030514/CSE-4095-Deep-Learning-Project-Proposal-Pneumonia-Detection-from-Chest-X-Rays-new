"""
设备选择模块

自动选择最佳可用计算设备，支持 CUDA、MPS (Apple Silicon)、DirectML 和 CPU。
"""
import torch


def get_device(prefer_mps: bool = True, verbose: bool = False):
    """
    Select best available device in order: CUDA -> MPS (Apple Silicon) -> DirectML -> CPU.

    Args:
        prefer_mps: If True, prefer MPS on Apple Silicon over DirectML (default: True)
        verbose: If True, print device selection info

    Returns:
        torch.device: Best available device object usable in `.to(device)` calls.

    Device Priority:
        1. CUDA (NVIDIA GPUs)
        2. MPS (Apple M1/M2/M3 chips) - if prefer_mps=True
        3. DirectML (AMD/Intel GPUs on Windows)
        4. CPU (fallback)

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    # 1. CUDA (NVIDIA GPUs) - highest priority
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    # 2. MPS (Apple Silicon - M1/M2/M3)
    if prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # 验证 MPS 是否真正可用
            if torch.backends.mps.is_built():
                device = torch.device('mps')
                if verbose:
                    print("[Device] Using MPS (Apple Silicon)")
                return device
        except Exception:
            pass

    # 3. DirectML (AMD/Intel GPUs on Windows)
    try:
        import torch_directml as dml  # type: ignore
        device = dml.device()
        if verbose:
            print("[Device] Using DirectML")
        return device
    except (ModuleNotFoundError, ImportError):
        pass

    # 4. CPU fallback
    if verbose:
        print("[Device] Using CPU")
    return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.

    Returns:
        dict: Device availability and details

    Example:
        >>> info = get_device_info()
        >>> print(info['cuda']['available'])  # True/False
    """
    info = {
        'cuda': {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'mps': {
            'available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'built': hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False,
        },
        'directml': {
            'available': False,
        },
        'cpu': {
            'available': True,
        },
        'selected': str(get_device()),
    }

    # Check DirectML
    try:
        import torch_directml  # type: ignore
        info['directml']['available'] = True
    except (ModuleNotFoundError, ImportError):
        pass

    return info
