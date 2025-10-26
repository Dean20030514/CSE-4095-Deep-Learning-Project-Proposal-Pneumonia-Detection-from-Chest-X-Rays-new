import torch, pkgutil, sys
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
required = [
    "torch", "torchvision", "numpy", "pandas", "albumentations", "sklearn", "streamlit"
]
missing = [m for m in required if pkgutil.find_loader(m) is None]
print("Missing packages:", missing)
if missing:
    sys.exit(1)
print("OK")
