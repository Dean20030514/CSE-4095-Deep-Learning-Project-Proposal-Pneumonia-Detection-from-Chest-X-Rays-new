import torch


def get_device():
    """
    Select best available device in order: CUDA -> DirectML (AMD/Intel on Windows) -> CPU.
    Returns a device object usable in `.to(device)` calls.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Try DirectML on Windows (AMD/Intel GPUs)
    try:
        import torch_directml as dml  # type: ignore
        return dml.device()
    except (ModuleNotFoundError, ImportError):
        pass
    return torch.device('cpu')
